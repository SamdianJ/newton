# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit recorded hand/box geometry and normal response without advancing dynamics."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, sample_smooth
from newton._src.solvers.monolithic.smooth_sdf_builder import build_coefficients

from .articulation_oracle import transform
from .run_hand_tasks import build_model


def control_input_identity(recorded, current):
    """Check retained control hashes without treating missing historical inputs as verified."""

    def by_name(hashes):
        # These four inputs have distinct names; recorded paths may be relative.
        result = {Path(path).name: digest for path, digest in hashes.items()}
        if len(result) != len(hashes):
            raise ValueError("Ambiguous control input filenames")
        return result

    expected, actual = by_name(recorded or {}), by_name(current)
    for name in actual.keys() & expected.keys():
        if actual[name] != expected[name]:
            raise ValueError(f"Control input changed: {name}")
    missing = sorted(actual.keys() - expected.keys())
    return {
        "status": "UNVERIFIED" if missing else "VERIFIED",
        "missing_recorded_inputs": missing,
        "recorded_sha256": expected,
        "current_sha256": actual,
        "scope": "Historical inputs for closure_scan; absent hashes leave a current-control hypothesis",
    }


def surface_samples(vertices, faces, *, refinement=0):
    """Independently integrate a triangular surface with P1Q3 points [m, m²]."""
    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces).reshape(-1, 3)]
    for _ in range(refinement):
        a, b, c = np.moveaxis(triangles, 1, 0)
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = np.concatenate(
            [np.stack(points, axis=1) for points in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
        )
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(triangles).all() or np.any(twice_area <= 0):
        raise ValueError("Surface requires positive finite triangle areas")
    bary = np.full((3, 3), 1 / 6) + np.eye(3) / 2
    return (
        np.einsum("qn,fni->fqi", bary, triangles).reshape(-1, 3),
        np.repeat(cross / twice_area[:, None], 3, axis=0),
        np.repeat(twice_area / 6, 3),
    )


def box_geometry(points, half):
    """Return exact axis-aligned box distance [m] and its piecewise gradient."""
    points = np.asarray(points, dtype=float)
    excess = np.abs(points) - np.asarray(half)
    outside = np.maximum(excess, 0)
    length = np.linalg.norm(outside, axis=1)
    distance = length + np.minimum(excess.max(axis=1), 0)
    gradient = np.zeros_like(points)
    mask = length > 0
    gradient[mask] = np.sign(points[mask]) * outside[mask] / length[mask, None]
    inside = np.flatnonzero(~mask)
    axis = np.argmax(excess[inside], axis=1)
    gradient[inside, axis] = np.where(points[inside, axis] < 0, -1.0, 1.0)
    return distance, gradient


def normal_response(distance, gradient, normals, weights, *, stiffness=1e7, width=1e-4):
    """Integrate normal response, applying the production alignment-mask convention.

    Weights are effective source areas [m²], including the two-sided half factor.
    Stiffness is pressure per indentation [N/m³]. Friction/history are excluded.
    """
    depth = -np.asarray(distance)
    t = np.clip((depth + width) / (2 * width), 0, 1)
    hinge = np.where(depth >= width, depth, width * (2 * t**3 - t**4))
    derivative = np.where(depth >= width, 1.0, 3 * t**2 - 2 * t**3)
    load = stiffness * np.asarray(weights) * hinge * derivative
    masked = np.einsum("ij,ij->i", gradient, normals) > 0
    force = np.sum((load * ~masked)[:, None] * gradient, axis=0)
    penetrated = depth > 0
    return {
        "samples": len(depth),
        "penetrating_samples": int(penetrated.sum()),
        "masked_penetrating_samples": int((masked & penetrated).sum()),
        "max_sample_depth_m": max(0.0, float(depth.max(initial=0))),
        "total_effective_area_m2": float(np.sum(weights)),
        "max_sample_area_m2": float(np.max(weights, initial=0)),
        "active_effective_area_m2": float(np.sum(weights[penetrated & ~masked])),
        "masked_penetrating_area_m2": float(np.sum(weights[penetrated & masked])),
        "normal_load_N": float(load[~masked].sum()),
        "unmasked_normal_load_N": float(load.sum()),
        "source_normal_force_N": force.tolist(),
    }


def transformed(points, frame):
    return points @ frame[:3, :3].T + frame[:3, 3]


class RecordedContact:
    """Use recorded published poses, independent surface integration and qualified SDF queries."""

    def __init__(self, case, assets):
        self.case = case
        self.manifest = json.loads((case.parent / "manifest.json").read_text())
        self.summary = json.loads((case / "summary.json").read_text())
        self.recorded_scene = json.loads((case / "scene.json").read_text())
        self.data = np.load(case / "trajectory.npz", allow_pickle=False)
        self.model, self.mapping, self.curve, _, self.identity = build_model(
            self.summary["task"],
            assets,
            device="cpu",
            offset=self.manifest.get("offset", 0),
            surface_source=self.manifest.get("surface_source", "qualified"),
            height=self.manifest.get("height", 0),
        )
        for key in ("urdf_sha256", "mesh_sha256", "contact_manifest_sha256", "collision_mapping", "surface_source"):
            if self.identity[key] != self.recorded_scene[key]:
                raise ValueError(f"Recorded hand input changed: {key}")
        self.control_identity = control_input_identity(
            self.recorded_scene.get("input_sha256"), self.identity["input_sha256"]
        )
        if self.identity["object_body"] is None:
            raise ValueError("This local normal audit currently requires a rigid box object")
        # CPU hand construction intentionally omits texture storage. Recreate
        # recorded qualified fields directly, verifying both nodes and coefficients.
        directory = assets / "geometry-rebuilt"
        rows = json.loads((directory / "geometry-results.json").read_text())
        fields = {}
        for row in rows:
            index = row["sdf_index"]
            path = directory / f"{index:02d}-{Path(row['source']).stem}" / "nodes.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["node_sha256"]:
                raise ValueError("Qualified SDF nodes changed")
            with np.load(path) as nodes:
                coefficients = build_coefficients(
                    nodes["values"],
                    (np.asarray(row["upper"]) - row["lower"]) / (np.asarray(row["dims"]) - 1),
                    "c2-bspline",
                )
            field = SmoothSDF(coefficients, row["lower"], row["upper"], row["dims"], "c2-bspline", device="cpu")
            if field.identity != self.recorded_scene["smooth_sdf"]["fields"][index]:
                raise ValueError("Offline field does not reproduce the recorded SDF identity")
            fields[row["source"]] = field
        self.fields = {entry["shape"]: fields[entry["source"]] for entry in self.identity["collision_mapping"]}
        self.body = self.identity["object_body"]
        shape = np.flatnonzero(self.model.shape_body.numpy() == self.body)
        if len(shape) != 1:
            raise ValueError("Expected one box shape on the object body")
        self.object_shape = int(shape[0])
        recorded_pairs = {tuple(sorted(row["shapes"])) for row in self.recorded_scene["actor_contact_pairs"]}
        for entry in self.identity["collision_mapping"]:
            if tuple(sorted((entry["shape"], self.object_shape))) not in recorded_pairs:
                raise ValueError("Hand/object pair was not enabled in the recorded solver")
        self.half = self.model.shape_scale.numpy()[self.object_shape].astype(float)
        self.local_frames = [transform(p) for p in self.model.shape_transform.numpy()]
        self.margin = self.model.shape_margin.numpy().astype(float)
        box = newton.Mesh.create_box(1.0, compute_inertia=False)
        self.box_vertices, self.box_indices = box.vertices * self.half, box.indices
        self.box = surface_samples(self.box_vertices, self.box_indices)
        self.meshes = {}
        scale = self.model.shape_scale.numpy().astype(float)
        if not np.array_equal(scale, self.recorded_scene["smooth_sdf"]["sources"]["scale"]):
            raise ValueError("Recorded shape scales changed")
        for entry in self.identity["collision_mapping"]:
            shape = entry["shape"]
            mesh = self.model.shape_source[shape]
            vertices = np.asarray(mesh.vertices, dtype=float) * scale[shape]
            self.meshes[shape] = (vertices, *surface_samples(vertices, mesh.indices))

    def sdf_query(self, shape, points):
        output = wp.empty((len(points), 13), dtype=wp.float64, device="cpu")
        wp.launch(
            sample_smooth,
            dim=len(points),
            inputs=[
                self.fields[shape].data,
                wp.array(points, dtype=wp.vec3d, device="cpu"),
                output,
            ],
            device="cpu",
        )
        return output.numpy()

    def frames(self, body_q):
        return [transform(p) for p in body_q]

    def pair(self, entry, frames, *, shift=None, stiffness=1e7, refinement=0):
        shape = entry["shape"]
        hand = frames[entry["body"]] @ self.local_frames[shape]
        if shift is not None:
            hand[:3, 3] += shift
        box = frames[self.body] @ self.local_frames[self.object_shape]
        vertices, samples, normals, area = self.meshes[shape]
        if refinement:
            samples, normals, area = surface_samples(
                vertices, self.model.shape_source[shape].indices, refinement=refinement
            )
        world = transformed(samples, hand)
        points = (world - box[:3, 3]) @ box[:3, :3]
        distance, gradient = box_geometry(points, self.half)
        offset = self.margin[shape] + self.margin[self.object_shape]
        forward = normal_response(
            distance - offset, gradient @ box[:3, :3].T, normals @ hand[:3, :3].T, area * 0.5, stiffness=stiffness
        )
        box_points, box_normals, box_area = self.box
        if refinement:
            box_points, box_normals, box_area = surface_samples(
                self.box_vertices, self.box_indices, refinement=refinement
            )
        local = (transformed(box_points, box) - hand[:3, 3]) @ hand[:3, :3]
        values = self.sdf_query(shape, local)
        reverse = normal_response(
            values[:, 0] - offset,
            values[:, 1:4] @ hand[:3, :3].T,
            box_normals @ box[:3, :3].T,
            box_area * 0.5,
            stiffness=stiffness,
        )
        local_vertices = (transformed(vertices, hand) - box[:3, 3]) @ box[:3, :3]
        vertex_distance, vertex_gradient = box_geometry(local_vertices, self.half)
        worst = int(np.argmin(vertex_distance))
        return {
            "shape": shape,
            "body": entry["body"],
            "link": self.model.body_label[entry["body"]],
            "vertex_depth_m": max(0.0, float(-vertex_distance[worst])),
            "outward_direction": (box[:3, :3] @ vertex_gradient[worst]).tolist(),
            "hand_to_box": forward,
            "box_to_hand": reverse,
            "normal_force_on_hand_N": (
                np.asarray(forward["source_normal_force_N"]) - reverse["source_normal_force_N"]
            ).tolist(),
        }

    def asset_audit(self):
        result, seen = [], set()
        for entry in self.identity["collision_mapping"]:
            shape, source = entry["shape"], entry["source"]
            if source in seen:
                continue
            seen.add(source)
            vertices, points, normals, area = self.meshes[shape]
            values = self.sdf_query(shape, points)
            lengths = np.linalg.norm(values[:, 1:4], axis=1)
            alignment = np.einsum("ij,ij->i", normals, values[:, 1:4]) / np.maximum(lengths, 1e-30)
            result.append(
                {
                    "source": source,
                    "triangles": len(points) // 3,
                    "surface_area_m2": float(area.sum()),
                    "max_p1q3_sdf_distance_m": float(np.max(np.abs(values[:, 0]))),
                    "opposed_normal_area_fraction": float(area[alignment < 0].sum() / area.sum()),
                    "strongly_opposed_normal_area_fraction": float(area[alignment < -0.5].sum() / area.sum()),
                    "minimum_normal_dot": float(alignment.min()),
                    "sdf_gradient_norm_range": [float(lengths.min()), float(lengths.max())],
                    "vertices": len(vertices),
                }
            )
        return result

    def selected_frame(self, requested):
        index = int(np.argmin(np.abs(self.data["time"] - requested)))
        if abs(self.data["time"][index] - requested) > 0.026:
            raise ValueError("Requested time is outside the recorded frame neighborhood")
        frames = self.frames(self.data["body_q"][index])
        pairs = [self.pair(entry, frames) for entry in self.identity["collision_mapping"]]
        worst = max(pairs, key=lambda row: row["vertex_depth_m"])
        entry = next(e for e in self.identity["collision_mapping"] if e["shape"] == worst["shape"])
        direction = np.asarray(worst["outward_direction"])
        scan = []
        for distance in (0, 0.0005, 0.0015, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02):
            shift = direction * distance
            row = self.pair(entry, frames, shift=shift)
            scan.append({"outward_shift_m": distance, "translation_m": shift.tolist(), **row})
        return {
            "time": float(self.data["time"][index]),
            "frame": index,
            "pairs": pairs,
            "worst_vertex_pair": worst,
            "translation_scan": scan,
            "stiffness_scan": [
                {"stiffness_N_per_m3": k, **self.pair(entry, frames, stiffness=k)} for k in (1e7, 5e7, 1e8)
            ],
            "quadrature_scan": [
                {"triangle_refinement": level, **self.pair(entry, frames, refinement=level)} for level in (0, 1, 2)
            ],
        }

    @staticmethod
    def depth(pair):
        return max(
            pair["vertex_depth_m"],
            pair["hand_to_box"]["max_sample_depth_m"],
            pair["box_to_hand"]["max_sample_depth_m"],
        )

    def depth_limited_pairs(self, index):
        """Translate each selected surface independently; this is not a joint trajectory."""
        frames = self.frames(self.data["body_q"][index])
        pairs = [self.pair(entry, frames) for entry in self.identity["collision_mapping"]]
        selected = []
        for finger in ("thumb", "index", "middle"):
            candidates = [row for row in pairs if f"left_{finger}_" in row["link"] and self.depth(row) > 0]
            if candidates:
                selected.append(max(candidates, key=self.depth))
        result = []
        for original in selected:
            entry = next(e for e in self.identity["collision_mapping"] if e["shape"] == original["shape"])
            direction = np.asarray(original["outward_direction"])
            for target in (0.00075, 0.0015):
                low, high = 0.0, 0.0
                for distance in np.linspace(0, 0.08, 81):
                    current = self.pair(entry, frames, shift=direction * distance)
                    if self.depth(current) <= target:
                        high = float(distance)
                        break
                    low = float(distance)
                else:
                    raise ValueError("No depth-limited state along selected outward direction")
                for _ in range(20):
                    middle = (low + high) / 2
                    current = self.pair(entry, frames, shift=direction * middle)
                    if self.depth(current) <= target:
                        high = middle
                    else:
                        low = middle
                result.append(
                    {
                        "target_depth_m": target,
                        "outward_translation_m": (direction * high).tolist(),
                        "responses": [
                            {"stiffness_N_per_m3": k, **self.pair(entry, frames, shift=direction * high, stiffness=k)}
                            for k in (1e7, 5e7, 1e8)
                        ],
                    }
                )
        return result

    def closing_profile(self):
        """Measure recorded closure frames, preserving the actual published configurations."""
        result = []
        for index, (time, phase) in enumerate(zip(self.data["time"], self.data["phase"], strict=True)):
            if phase != "close":
                continue
            frames = self.frames(self.data["body_q"][index])
            pairs = [self.pair(entry, frames) for entry in self.identity["collision_mapping"]]
            active = [
                row for row in pairs if row["hand_to_box"]["normal_load_N"] + row["box_to_hand"]["normal_load_N"] > 1e-4
            ]
            result.append({"time": float(time), "max_depth_m": max(map(self.depth, pairs)), "pairs": active})
        return result

    def closure_scan(self, index):
        q = self.data["joint_q"][index].copy()
        source = self.model.state()
        frames = self.frames(self.data["body_q"][index])
        rows = []
        for close_time in np.linspace(0, 3, 13):
            target, _ = self.curve.sample(float(close_time))
            for name, value in zip(self.curve.names, target, strict=True):
                q[self.mapping[name][0]] = value
            source.joint_q.assign(q)
            newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
            trial = self.frames(source.body_q.numpy())
            trial[self.body] = frames[self.body]
            pairs = [self.pair(entry, trial) for entry in self.identity["collision_mapping"]]
            active = [p for p in pairs if p["hand_to_box"]["normal_load_N"] + p["box_to_hand"]["normal_load_N"] > 1e-4]
            rows.append({"closure_source_time": float(close_time), "pairs": active})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not already exist")
    files = (
        Path(__file__),
        Path(__file__).with_name("run_hand_tasks.py"),
        Path(__file__).with_name("articulation_oracle.py"),
        Path(__file__).parents[2] / "newton/_src/solvers/monolithic/smooth_sdf.py",
        Path(__file__).parents[2] / "newton/_src/solvers/monolithic/smooth_sdf_builder.py",
        Path(__file__).parents[2] / "newton/examples/softbody/monolithic_sharpa_assets.py",
        Path(__file__).parents[2] / "newton/examples/softbody/sharpa_close.py",
    )
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    audit = RecordedContact(args.case.resolve(), args.assets.resolve())
    selected = audit.selected_frame(args.time)
    result = {
        "scope": (
            "Recorded published geometry and frictionless normal response, not dynamic replay or native-force oracle"
        ),
        "precision": "FP64 local calculations over recorded FP32 published body poses",
        "contact_stiffness_N_per_m3": 1e7,
        "normal_smoothing_m": 1e-4,
        "two_dynamic_endpoints_weight": 0.5,
        "command": sys.argv,
        "case": str(args.case.resolve()),
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.case / "trajectory.npz",
                args.case / "scene.json",
                args.case / "summary.json",
                args.case.parent / "manifest.json",
            )
        },
        "source_sha256": sources,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "asset_identity": audit.identity,
        "assets": audit.asset_audit(),
        "selected": selected,
        "closure_scan_input_identity": audit.control_identity,
        "closure_scan": audit.closure_scan(selected["frame"]),
        "recorded_closing_profile": audit.closing_profile(),
        "independent_surface_retreat": audit.depth_limited_pairs(selected["frame"]),
    }
    result["source_unchanged"] = all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in sources.items()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "worst": selected["worst_vertex_pair"]}))


if __name__ == "__main__":
    main()
