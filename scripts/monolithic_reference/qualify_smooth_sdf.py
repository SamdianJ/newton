# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify actual-grid derivatives and the complete frozen geometric probe union."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF
from scripts.monolithic_reference.build_smooth_sdf import build_coefficients, texture_identity
from scripts.monolithic_reference.sharpa_close_scene import ARTIFACTS, build
from scripts.monolithic_reference.smooth_sdf_oracle import evaluate
from scripts.monolithic_reference.verify_coupled import save
from scripts.monolithic_reference.verify_smooth_sdf import _old, geometry_gate, measure, sample, sha256


def mesh_distance(vertices, indices, points):
    """Independent FP64 all-triangle distance and solid-angle sign, for the nine small probe groups."""
    triangles = vertices.astype(float)[indices.reshape(-1, 3)]
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = np.sum(normal**2, axis=1)
    aa, bb, cc = np.sum(ab * ab, axis=1), np.sum(ab * ac, axis=1), np.sum(ac * ac, axis=1)
    result = []
    for point in points:
        ap = point - a
        ad, cd = np.sum(ap * ab, axis=1), np.sum(ap * ac, axis=1)
        determinant = aa * cc - bb * bb
        u, v = (cc * ad - bb * cd) / determinant, (aa * cd - bb * ad) / determinant
        distance2 = np.where((u >= 0) & (v >= 0) & (u + v <= 1), np.sum(ap * normal, axis=1) ** 2 / n2, np.inf)
        for start, end in ((a, b), (b, c), (c, a)):
            edge = end - start
            t = np.clip(np.sum((point - start) * edge, axis=1) / np.sum(edge**2, axis=1), 0, 1)
            distance2 = np.minimum(distance2, np.sum((point - start - t[:, None] * edge) ** 2, axis=1))
        ra, rb, rc = a - point, b - point, c - point
        la, lb, lc = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1), np.linalg.norm(rc, axis=1)
        numerator = np.sum(ra * np.cross(rb, rc), axis=1)
        denominator = (
            la * lb * lc + np.sum(ra * rb, axis=1) * lc + np.sum(rb * rc, axis=1) * la + np.sum(rc * ra, axis=1) * lb
        )
        inside = abs(np.sum(2 * np.arctan2(numerator, denominator))) > 2 * np.pi
        result.append(np.sqrt(distance2.min()) * (-1 if inside else 1))
    return np.asarray(result)


def run(directory, output):
    output.mkdir(parents=True, exist_ok=False)
    model, *_ = build(18)
    rows = json.loads((directory / "geometry-results.json").read_text())
    queries = json.loads((directory / "frozen-queries.json").read_text())
    results = []
    for row in rows:
        index, stem = row["sdf_index"], Path(row["source"]).stem
        source = directory / f"{index:02d}-{stem}"
        nodes = np.load(source / "nodes.npz")
        lo, hi, values = nodes["lower"], nodes["upper"], nodes["values"]
        dx = (hi - lo) / (np.asarray(values.shape[::-1]) - 1)
        grid_points = np.random.default_rng(975 + index).uniform(lo + dx, hi - dx, (12, 3))
        extra = [point for query in queries if query["sdf"] == index for point in query["points"]]
        points = np.vstack([grid_points, extra]) if extra else grid_points
        baseline = np.load(source / "probes.npz")
        reference = baseline["mesh_reference"]
        old = baseline["old"]
        if extra:
            mesh = np.load(ARTIFACTS / f"p1/pr6d-assets/hand/{stem}.npz")
            extra_ref = mesh_distance(mesh["vertices"], mesh["indices"], np.asarray(extra))
            old_device = wp.empty((len(extra), 4), dtype=wp.float64, device=model.device)
            wp.launch(
                _old,
                len(extra),
                [model._texture_sdf_data, index, wp.array(extra, dtype=wp.vec3d, device=model.device), old_device],
                device=model.device,
            )
            extra_old = old_device.numpy()
            reference = np.r_[reference, extra_ref]
            old = np.vstack([old, extra_old])
            np.savez_compressed(
                output / f"{index:02d}-extra-reference.npz",
                points=np.asarray(extra),
                reference=extra_ref,
                old=extra_old,
            )
        baseline_metrics = measure(old, reference)
        for kind in ("c1-hermite", "c2-bspline"):
            coeff = build_coefficients(values, dx, kind)
            field = SmoothSDF(coeff, lo, hi, values.shape[::-1], kind, device=model.device)
            actual = sample(field, points)
            expected = evaluate(values, lo, hi, points, kind)
            cpu = SmoothSDF(coeff, lo, hi, values.shape[::-1], kind, device="cpu")
            cpu_values = sample(cpu, points)
            error = np.max(abs(actual - expected), axis=0)
            device_error = np.max(abs(actual - cpu_values), axis=0)
            # Compare derivatives in grid units to avoid dimensionful H tolerances.
            units = np.r_[1.0, dx, np.outer(dx, dx).ravel()]
            oracle_pass = bool(np.max(error * units) <= 2e-12 and np.max(device_error * units) <= 2e-12)
            center_index = np.floor((grid_points[0] - lo) / dx)
            spreads = []
            for epsilon in (1e-5, 1e-7, 1e-9):
                maxima = np.zeros(13)
                for axes in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)):
                    center = center_index + np.array([0.31, 0.47, 0.63])
                    center[list(axes)] = center_index[list(axes)]
                    offsets = np.zeros((2 ** len(axes), 3))
                    offsets[:, list(axes)] = np.asarray(list(itertools.product((-1, 1), repeat=len(axes)))) * epsilon
                    limits = sample(field, lo + (center + offsets) * dx)
                    maxima = np.maximum(maxima, np.ptp(limits, axis=0) * units)
                spreads.append(maxima)
            required = 13 if kind == "c2-bspline" else 4
            continuity_pass = bool(
                np.max(spreads[-1][:required]) < 2e-10
                and np.all(spreads[-1][:required] <= 0.02 * spreads[0][:required] + 2e-12)
            )
            geometry_values = np.load(source / f"{kind}.npz")["values"]
            if extra:
                geometry_values = np.vstack([geometry_values, actual[12:]])
            metrics = measure(geometry_values, reference)
            row_out = {
                "sdf": index,
                "source": row["source"],
                "representation": kind,
                "oracle_passed": oracle_pass,
                "continuity_passed": continuity_pass,
                "oracle_error": error,
                "cpu_cuda_error": device_error,
                "normalized_continuity_spreads": spreads,
                "baseline": baseline_metrics,
                "metrics": metrics,
                "geometry_passed": geometry_gate(metrics, baseline_metrics),
                "probe_count": len(reference),
                "node_sha256": sha256(source / "nodes.npz"),
            }
            results.append(row_out)
            np.savez_compressed(
                output / f"{index:02d}-{kind}-oracle.npz", points=points, actual=actual, oracle=expected, cpu=cpu_values
            )
            save(output / "results.json", results)
            print(
                index,
                kind,
                "oracle",
                oracle_pass,
                "continuity",
                continuity_pass,
                "geometry",
                row_out["geometry_passed"],
                flush=True,
            )
    summary = {
        kind: all(
            r["oracle_passed"] and r["continuity_passed"] and r["geometry_passed"]
            for r in results
            if r["representation"] == kind
        )
        for kind in ("c1-hermite", "c2-bspline")
    }
    save(output / "summary.json", summary)
    save(
        directory / "candidate-qualification.json",
        {
            "passed": summary,
            "evidence": str(output),
            "results_sha256": sha256(output / "results.json"),
            "source_textures": texture_identity(model),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.geometry, args.output)
