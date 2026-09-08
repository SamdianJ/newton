# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare and measure derived Sharpa collision assets on CUDA (offline)."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData, texture_sample_sdf_grad
from newton.examples.softbody.monolithic_sharpa_assets import sdf_build_identity
from newton.examples.softbody.sharpa_close import sha256


@wp.kernel
def _measure(mesh: wp.uint64, points: wp.array[wp.vec3], sdf: TextureSDFData, result: wp.array2d[float]):
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_point_sign_winding_number(mesh, p, 1.0)
    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    phi = wp.length(p - closest) * query.sign
    sampled, normal = texture_sample_sdf_grad(sdf, p)
    result[i, 0] = phi
    result[i, 1] = sampled
    result[i, 2] = wp.length(normal)
    if not query.result:
        result[i, 0] = wp.nan


@wp.kernel
def _distance(mesh: wp.uint64, points: wp.array[wp.vec3], result: wp.array[float]):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[i], 1.0)
    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    result[i] = wp.length(points[i] - closest)
    if not query.result:
        result[i] = wp.nan


def mesh_handle(mesh):
    return wp.Mesh(
        wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device="cuda:0"),
        wp.array(np.asarray(mesh.faces, dtype=np.int32).flatten(), dtype=int, device="cuda:0"),
        support_winding_number=True,
    )


def deviation(source, derived):
    """Measure both directed vertex distances; not a continuous Hausdorff bound."""
    maxima = []
    for a, b in ((source, derived), (derived, source)):
        handle = mesh_handle(b)
        points = wp.array(np.asarray(a.vertices, dtype=np.float32), dtype=wp.vec3, device="cuda:0")
        out = wp.empty(len(points), dtype=float, device="cuda:0")
        wp.launch(_distance, len(points), [handle.id, points, out], device="cuda:0")
        values = out.numpy()
        if not np.isfinite(values).all():
            raise ValueError("Invalid mesh deviation query")
        maxima.append(float(values.max()))
    return maxima


def prepare(asset_dir, output):
    # Trimesh is already an examples dependency. Runtime loads the resulting npz.
    import trimesh

    asset_dir, output = Path(asset_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = asset_dir / "left_sharpa_wave.urdf"
    root = ET.parse(source).getroot()
    refs = []
    for link in root.findall("link"):
        for ordinal, collision in enumerate(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            uri = mesh.get("filename")
            prefix = "package://left_sharpa_wave/"
            if not uri.startswith(prefix):
                raise ValueError("Unexpected collision URI")
            path = (asset_dir / uri[len(prefix) :]).resolve()
            if not path.is_relative_to(asset_dir) or not path.is_file():
                raise ValueError("Missing collision asset")
            refs.append({"link": link.get("name"), "ordinal": ordinal, "source": uri[len(prefix) :]})
    if len(refs) != 26:
        raise ValueError("Expected exactly 26 collision references")
    rows = []
    for relative in sorted({r["source"] for r in refs}):
        path = asset_dir / relative
        original = trimesh.load(path, force="mesh")
        derived = original.copy()
        repair = "identity"
        if not original.is_watertight:
            if path.stem == "DP_HB1_4F":
                derived.merge_vertices(digits_vertex=7)
                repair = "weld vertices at 1e-7 m decimal grid; no face additions"
            else:
                mesh = newton.Mesh(original.vertices, original.faces.flatten(), compute_inertia=False)
                sdf = mesh.build_sdf(
                    device="cuda:0",
                    target_voxel_size=0.0002,
                    margin=0.002,
                    narrow_band_range=(-0.002, 0.002),
                    sign_method="winding",
                    texture_format="float32",
                )
                surface = sdf.extract_isomesh(device="cuda:0")
                if surface is None:
                    raise ValueError(f"Empty repair surface: {relative}")
                derived = trimesh.Trimesh(surface.vertices, surface.indices.reshape(-1, 3), process=True)
                derived.merge_vertices(digits_vertex=6)
                derived.update_faces(derived.nondegenerate_faces(height=1e-12))
                derived.update_faces(derived.unique_faces())
                derived.remove_unreferenced_vertices()
                repair = "Newton winding SDF 0.2 mm / zero isosurface / weld at 1e-6 m decimal grid"
        if not derived.is_watertight or not derived.is_winding_consistent or derived.volume <= 0:
            raise ValueError(f"Unqualified derived collision mesh: {relative}")
        distances = deviation(original, derived)
        if max(distances) > 0.0004:
            raise ValueError(f"Repair vertex deviation exceeds frozen 0.4 mm: {relative}: {distances}")
        dest = output / (path.stem + ".npz")
        np.savez(
            dest,
            vertices=np.asarray(derived.vertices, dtype=np.float32),
            indices=np.asarray(derived.faces, dtype=np.int32),
        )
        row = {
            "source": relative,
            "source_sha256": sha256(path),
            "derived": dest.name,
            "derived_sha256": sha256(dest),
            "source_watertight": bool(original.is_watertight),
            "repair": repair,
            "vertex_count": len(derived.vertices),
            "face_count": len(derived.faces),
            "watertight": True,
            "winding_consistent": True,
            "volume_m3": float(derived.volume),
            "relative_volume_change": float(derived.volume / original.volume - 1),
            "bidirectional_vertex_distance_max_m": distances,
            "sdf": [],
        }
        handle = mesh_handle(derived)
        ids = np.linspace(0, len(derived.faces) - 1, min(512, len(derived.faces)), dtype=int)
        centers, normals = derived.triangles_center[ids], derived.face_normals[ids]
        # Frozen surface, inside/outside shell probes, including fingertip meshes.
        points = np.concatenate([centers + d * normals for d in (-0.001, -0.0005, 0, 0.0005, 0.001)])
        wp_points = wp.array(points, dtype=wp.vec3, device="cuda:0")
        np.savez(output / (path.stem + "_probes.npz"), points=points)
        for resolution in (32, 64, 128):
            mesh = newton.Mesh(derived.vertices, derived.faces.flatten(), compute_inertia=False)
            start = time.perf_counter()
            sdf = mesh.build_sdf(
                device="cuda:0",
                max_resolution=resolution,
                margin=0.003,
                narrow_band_range=(-0.003, 0.003),
                texture_format="float32",
                sign_method="winding",
                cache_dir=output / "sdf-cache",
            )
            wp.synchronize_device("cuda:0")
            build_seconds = time.perf_counter() - start
            result = wp.empty((len(points), 3), dtype=float, device="cuda:0")
            wp.launch(
                _measure, len(points), [handle.id, wp_points, sdf.to_texture_kernel_data(), result], device="cuda:0"
            )
            values = result.numpy()
            error = np.abs(values[:, 1] - values[:, 0])
            separated = np.abs(values[:, 0]) > 0.0004
            sign_errors = int(np.count_nonzero(values[separated, 0] * values[separated, 1] < 0))
            # This is a measured envelope. Coarse meshes may fail; never auto-relax.
            passed = bool(
                np.isfinite(values).all() and error.max() <= 0.0004 and sign_errors == 0 and np.all(values[:, 2] > 0.0)
            )
            row["sdf"].append(
                {
                    "max_resolution": resolution,
                    "build_seconds": build_seconds,
                    "max_error_m": float(error.max()),
                    "p95_error_m": float(np.percentile(error, 95)),
                    "sign_errors_outside_0_4mm": sign_errors,
                    "minimum_raw_gradient_norm": float(values[:, 2].min()),
                    "probe_count": len(points),
                    "passed": passed,
                }
            )
            np.savez(output / f"{path.stem}_sdf{resolution}.npz", values=values)
        rows.append(row)
        print(relative, row["sdf"], flush=True)
        # Preserve completed measurements if a later asset fails.
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "sharpa-contact-assets/v1",
                    "sdf_build_identity": sdf_build_identity(),
                    "cache_sha256": {p.name: sha256(p) for p in sorted((output / "sdf-cache").glob("*.npz"))},
                    "urdf_sha256": sha256(source),
                    "trimesh_version": trimesh.__version__,
                    "warp_version": wp.__version__,
                    "units": "m",
                    "repair_voxel_m": 0.0002,
                    "repair_vertex_distance_limit_m": 0.0004,
                    "sdf_error_limit_m": 0.0004,
                    "sdf_margin_m": 0.003,
                    "sdf_narrow_band_m": [-0.003, 0.003],
                    "texture_format": "float32",
                    "sign_method": "winding",
                    "references": refs,
                    "meshes": rows,
                    "complete": len(rows) == len({r["source"] for r in refs}),
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.asset_dir, args.output)


if __name__ == "__main__":
    main()
