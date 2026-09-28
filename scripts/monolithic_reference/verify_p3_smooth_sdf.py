# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Freeze and qualify both smooth SDF candidates before allowing solver experiments."""

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.solvers.monolithic.contact_geometry import _mesh64
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF
from newton._src.solvers.monolithic.smooth_sdf_builder import _nodes, sample
from scripts.monolithic_reference.build_p3_smooth_sdf import build_coefficients
from scripts.monolithic_reference.verify_p3_close import ARTIFACTS, build
from scripts.monolithic_reference.verify_p3_coupled import save


@wp.kernel
def _old(table: wp.array[TextureSDFData], sdf_index: int, points: wp.array[wp.vec3d], output: wp.array2d[wp.float64]):
    i = wp.tid()
    value, gradient = _mesh64(table[sdf_index], points[i])
    output[i, 0] = value
    for a in range(3):
        output[i, a + 1] = gradient[a]


def sha256(path):
    """Hash large evidence without retaining another full buffer."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(values, reference):
    error = np.abs(values[:, 0] - reference)
    gradient_norm = np.linalg.norm(values[:, 1:4], axis=1)
    separated = np.abs(reference) > 0.0004
    return {
        "max_error_m": float(error.max()),
        "rms_error_m": float(np.sqrt(np.mean(error**2))),
        "finite": bool(np.isfinite(values).all()),
        "minimum_gradient_norm": float(gradient_norm.min()),
        "sign_errors_outside_0_4mm": int(np.count_nonzero(values[separated, 0] * reference[separated] < 0)),
    }


def geometry_gate(candidate, baseline):
    """Do not treat a smaller mean error as permission to worsen the maximum."""
    return bool(
        candidate["finite"]
        and candidate["minimum_gradient_norm"] > 0
        and candidate["sign_errors_outside_0_4mm"] == 0
        and candidate["max_error_m"] <= 0.0004
        and candidate["max_error_m"] <= baseline["max_error_m"] + 1e-12
        and candidate["rms_error_m"] <= baseline["rms_error_m"] + 1e-12
    )


def frozen_queries():
    """Keep every existing residual-boundary and geometry-rejection probe."""
    root = ARTIFACTS / "p3/pr3"
    residual = root / "t5-resnorm-diagnosis-v1/boundary-results.json"
    geometry = root / "t5-geometry-diagnosis-v1/boundary.json"
    entries = []
    for row in json.loads(residual.read_text()):
        entries.append(
            {"name": row["case"], "kind": "residual", "sdf": row["sdf"], "points": row["points"], "original": row}
        )
    for row in json.loads(geometry.read_text()):
        points = [r["key"]["query"] for r in row["boundary"]]
        entries.append(
            {
                "name": row["name"],
                "kind": "geometry",
                "sdf": row["boundary"][0]["key"]["sdf_index"],
                "points": points,
                "original": row,
            }
        )
    if len(entries) != 9:
        raise ValueError("Expected the five residual and four geometry query groups")
    return entries, {str(p): sha256(p) for p in (residual, geometry)}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    asset_dir = ARTIFACTS / "p1/pr6d-assets/hand"
    sources = {str(p): sha256(p) for p in sorted(asset_dir.rglob("*")) if p.is_file()}
    queries, query_sources = frozen_queries()
    save(output / "frozen-queries.json", queries)
    save(output / "source-assets.json", sources)
    started = time.perf_counter()
    model, _, loaded, *_ = build(18)
    setup_seconds = time.perf_counter() - started
    table = model._texture_sdf_data.numpy()
    mapping = model._shape_sdf_index.numpy()
    unique = {}
    for row in loaded["collision_mapping"]:
        unique.setdefault(int(mapping[row["shape"]]), []).append(row)
    grids = []
    for index, shapes in sorted(unique.items()):
        dims = np.array(model._texture_sdf_subgrid_start_slots[index].shape) * table[index]["subgrid_size"] + 1
        grids.append(
            {
                "sdf_index": index,
                "shapes": shapes,
                "dims": dims,
                "c1_bytes": int(np.prod(dims) * 64),
                "c2_bytes": int(np.prod(dims + 2) * 8),
            }
        )
    save(
        output / "configuration.json",
        {
            "command": sys.argv,
            "representations": args.representation,
            "model_setup_seconds": setup_seconds,
            "grids": grids,
            "total_c1_bytes": sum(r["c1_bytes"] for r in grids),
            "total_c2_bytes": sum(r["c2_bytes"] for r in grids),
            "source_assets_sha256": sha256(output / "source-assets.json"),
            "frozen_query_sources": query_sources,
            "node_policy": "Same box endpoints and logical node count; sample the old FP64 evaluator on the uniform full grid. No rebake. Original FP32 inverse spacing is recorded separately.",
            "probe_policy": "Original probe coordinates rounded once to FP32, as in the frozen mesh-distance reference, then promoted to FP64 for every representation.",
            "limits": {"absolute_m": 0.0004, "comparison_slack_m": 1e-12},
            "physics": "unchanged; this entry point performs no time integration",
            "warp": wp.__version__,
        },
    )
    results = []
    for grid in grids:
        index, shapes = grid["sdf_index"], grid["shapes"]
        dims = np.asarray(grid["dims"])
        sdf = table[index]
        lower, upper = sdf["sdf_box_lower"].astype(float), sdf["sdf_box_upper"].astype(float)
        dx = (upper - lower) / (dims - 1)
        stem = Path(shapes[0]["source"]).stem
        points = np.load(asset_dir / f"{stem}_probes.npz")["points"].astype(np.float32).astype(float)
        archived = np.load(asset_dir / f"{stem}_sdf128.npz")["values"]
        ref = archived[:, 0].astype(float)
        row_dir = output / f"{index:02d}-{stem}"
        row_dir.mkdir()
        begin = time.perf_counter()
        nodes_device = wp.empty(int(np.prod(dims)), dtype=wp.float64, device=model.device)
        wp.launch(
            _nodes,
            len(nodes_device),
            [model._texture_sdf_data, index, wp.vec3i(dims), nodes_device],
            device=model.device,
        )
        nodes = nodes_device.numpy().reshape(tuple(dims[::-1]))
        node_seconds = time.perf_counter() - begin
        np.savez_compressed(row_dir / "nodes.npz", values=nodes, lower=lower, upper=upper)
        old_device = wp.empty((len(points), 4), dtype=wp.float64, device=model.device)
        wp.launch(
            _old,
            len(points),
            [model._texture_sdf_data, index, wp.array(points, dtype=wp.vec3d, device=model.device), old_device],
            device=model.device,
        )
        old = old_device.numpy()
        old_metrics = measure(old, ref)
        archived_metrics = {
            "max_error_m": float(np.max(np.abs(archived[:, 1] - archived[:, 0]))),
            "rms_error_m": float(np.sqrt(np.mean((archived[:, 1].astype(float) - ref) ** 2))),
        }
        np.savez_compressed(row_dir / "probes.npz", points=points, mesh_reference=ref, old=old, archived=archived)
        row = {
            "sdf_index": index,
            "source": shapes[0]["source"],
            "dims": dims,
            "lower": lower,
            "upper": upper,
            "original_inv_dx": sdf["inv_sdf_dx"],
            "candidate_inv_dx": 1 / dx,
            "node_sha256": sha256(row_dir / "nodes.npz"),
            "node_seconds": node_seconds,
            "baseline": old_metrics,
            "archived_baseline": archived_metrics,
            "candidates": {},
        }
        for kind in args.representation:
            if kind == "trilinear":
                continue
            begin = time.perf_counter()
            coefficients = build_coefficients(nodes, dx, kind)
            build_seconds = time.perf_counter() - begin
            field = SmoothSDF(coefficients, lower, upper, dims, kind, device=model.device)
            first = time.perf_counter()
            values = sample(field, points)
            first_seconds = time.perf_counter() - first
            np.savez_compressed(row_dir / f"{kind}.npz", values=values)
            measurements = measure(values, ref)
            normalized = values[:, 1:4] / np.linalg.norm(values[:, 1:4], axis=1)[:, None]
            old_normalized = old[:, 1:4] / np.linalg.norm(old[:, 1:4], axis=1)[:, None]
            angle = np.arccos(np.clip(np.sum(normalized * old_normalized, axis=1), -1, 1))
            surface = np.abs(ref) < 1e-7
            observations = {
                "identity": field.identity,
                "build_seconds": build_seconds,
                "first_query_seconds": first_seconds,
                "coefficient_bytes": coefficients.nbytes,
                "metrics": measurements,
                "passed": geometry_gate(measurements, old_metrics),
                "historical_comparison_passed": geometry_gate(measurements, {**old_metrics, **archived_metrics}),
                "max_phi_change_m": float(np.max(np.abs(values[:, 0] - old[:, 0]))),
                "normal_angle_max_rad": float(angle.max()),
                "normal_angle_rms_rad": float(np.sqrt(np.mean(angle**2))),
                "surface_phi_max_m": float(np.max(np.abs(values[surface, 0]))),
                "surface_phi_rms_m": float(np.sqrt(np.mean(values[surface, 0] ** 2))),
            }
            # Save fixed query values even if this candidate has already lost geometry qualification.
            for j, query in enumerate(queries):
                if query["sdf"] == index:
                    np.savez_compressed(
                        row_dir / f"{kind}-frozen-{j}.npz",
                        points=np.asarray(query["points"]),
                        values=sample(field, query["points"]),
                    )
            row["candidates"][kind] = observations
            print(stem, kind, measurements, "PASS", observations["passed"], flush=True)
            del field, coefficients
            gc.collect()
        results.append(row)
        save(output / "geometry-results.json", results)
        del nodes_device, nodes
        gc.collect()
    summary = {
        "geometry_qualified": {
            kind: all(
                r["candidates"][kind]["passed"] and r["candidates"][kind]["historical_comparison_passed"]
                for r in results
            )
            for kind in args.representation
            if kind != "trilinear"
        },
        "asset_count": len(results),
        "frozen_query_group_count": len(queries),
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "assets_unchanged": all(sha256(p) == digest for p, digest in sources.items()),
        "close": "Gated on all geometry and component checks; not run by this qualification entry point",
    }
    save(output / "summary.json", summary)
    save(
        output / "evidence-hashes.json",
        {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("trilinear", "c1-hermite", "c2-bspline"),
        nargs="+",
        default=["c1-hermite", "c2-bspline"],
    )
    run(parser.parse_args())
