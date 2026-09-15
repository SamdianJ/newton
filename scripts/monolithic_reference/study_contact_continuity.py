# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export and evaluate the two diagnosed Close inputs without advancing a solver."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import (
    TextureSDFData,
    texture_sample_sdf_grad,
)
from newton._src.solvers.monolithic.contact_continuity import export_nodes as _export_patch
from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase
from scripts.monolithic_reference.contact_continuity import patch_oracle, query_patch
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources


@wp.kernel(enable_backward=False)
def _original_query(
    table: wp.array[TextureSDFData],
    index: int,
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    result: wp.array[wp.vec4],
    exact: wp.array[float],
):
    i = wp.tid()
    phi, g = texture_sample_sdf_grad(table[index], points[i])
    result[i] = wp.vec4(phi, g[0], g[1], g[2])
    closest = wp.mesh_query_point_sign_winding_number(mesh, points[i], 1.0)
    if closest.result:
        exact[i] = closest.sign * wp.length(points[i] - wp.mesh_eval_position(mesh, closest.face, closest.u, closest.v))
    else:
        exact[i] = wp.nan


def _angle(a, b):
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def boundary_lower_bound(nodes):
    """Bound C1 values in the outermost valid cells using their local stencils."""
    boundary_cells = []
    for axis in range(3):
        axes = [np.arange(1, n - 2) for n in nodes.shape]
        axes[axis] = np.array([1, nodes.shape[axis] - 3])
        boundary_cells.append(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3))
    cells = np.unique(np.concatenate(boundary_cells), axis=0)
    result = float("inf")
    for start in range(0, len(cells), 1024):
        c = cells[start : start + 1024]
        stencil = np.empty((len(c), 4, 4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    stencil[:, i, j, k] = nodes[c[:, 0] + i - 1, c[:, 1] + j - 1, c[:, 2] + k - 1]
        corner_min = stencil[:, 1:3, 1:3, 1:3].min(axis=(1, 2, 3))
        bound = sum(0.125 * 1.25**a * np.abs(np.diff(stencil, n=2, axis=a + 1)).max(axis=(1, 2, 3)) for a in range(3))
        result = min(result, float(np.min(corner_min - bound)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnosis-dir", "asset-dir", "contact-dir", "trajectory", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    records = json.loads((args.diagnosis_dir / "position-24-production-sdf-query-v2/sdf-query.json").read_text())
    case = GraspCase(
        SimpleNamespace(
            cube="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz",
            device="cuda:0",
            calibration=str(args.calibration),
            asset_dir=str(args.asset_dir),
            contact_dir=str(args.contact_dir),
            trajectory=str(args.trajectory),
            position_offset_mm=(10, 10, -5),
            disable_aabb=False,
            friction_off=False,
            pcg_mode="production",
            experiment="grasp",
        )
    )
    model = case.model
    shape = records[0]["shape"]
    index = int(model._shape_sdf_index.numpy()[shape])
    descriptor = model._texture_sdf_data.numpy()[index]
    scale = model.shape_scale.numpy()[shape]
    if not np.all(scale == 1):
        raise ValueError("This frozen input requires its original unit-scale shape")
    inverse_dx = descriptor["inv_sdf_dx"].astype(float)
    lower = descriptor["sdf_box_lower"].astype(float)
    upper = descriptor["sdf_box_upper"].astype(float)
    origin = np.floor(records[0]["fine_coordinates"]).astype(int) - 5
    if np.any(origin < 0) or np.any(lower + (origin + 11) / inverse_dx > upper):
        raise ValueError("Frozen patch must be entirely within the original texture")
    values = wp.empty((12, 12, 12), dtype=float, device="cuda:0")
    wp.launch(_export_patch, values.shape, [model._texture_sdf_data, index, wp.vec3i(*origin), values], device="cuda:0")
    patch = values.numpy()
    coordinates = np.array([r["fine_coordinates"] for r in records]) - origin
    points = np.array([r["local_query"] for r in records], dtype=np.float32)

    oracle_meshes = {}

    def original(p, sdf_index=index, shape_index=shape):
        if shape_index not in oracle_meshes:
            source = model.shape_source[shape_index]
            oracle_meshes[shape_index] = wp.Mesh(
                wp.array(source.vertices, dtype=wp.vec3, device="cuda:0"),
                wp.array(source.indices, dtype=int, device="cuda:0"),
                support_winding_number=True,
            )
        result = wp.empty(len(p), dtype=wp.vec4, device="cuda:0")
        exact = wp.empty(len(p), dtype=float, device="cuda:0")
        wp.launch(
            _original_query,
            len(p),
            [
                model._texture_sdf_data,
                sdf_index,
                oracle_meshes[shape_index].id,
                wp.array(p, dtype=wp.vec3, device="cuda:0"),
                result,
                exact,
            ],
            device="cuda:0",
        )
        return result.numpy(), exact.numpy()

    baseline, exact = original(points)
    np.testing.assert_allclose(baseline[:, 0], [r["phi"] for r in records], atol=1e-9, rtol=0)
    output, valid = (
        wp.empty(len(points), dtype=wp.vec4, device="cuda:0"),
        wp.empty(len(points), dtype=int, device="cuda:0"),
    )
    wp.launch(
        query_patch,
        len(points),
        [values, wp.array(coordinates, dtype=wp.vec3, device="cuda:0"), wp.vec3(*inverse_dx), output, valid],
        device="cuda:0",
    )
    candidate = output.numpy()
    oracle = np.array([patch_oracle(patch, p, inverse_dx) for p in coordinates])
    np.testing.assert_allclose(candidate, oracle, atol=4e-6, rtol=1e-5)
    if not valid.numpy().all():
        raise ValueError("Invalid candidate at frozen input")

    # A deterministic local cloud measures geometry effects separately from continuity.
    offsets = np.stack(np.meshgrid(*[np.linspace(-2, 2, 9)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    cloud = coordinates[0] + offsets
    local = lower + (cloud + origin) / inverse_dx
    old_cloud, mesh_cloud = original(local)
    cubic_cloud = np.array([patch_oracle(patch, p, inverse_dx) for p in cloud])
    surface = []
    for seed in cloud:
        p = seed.copy()
        for _ in range(12):
            if np.any(p < 1) or np.any(p >= 10):
                break
            v = patch_oracle(patch, p, inverse_dx)
            g = v[1:]
            if np.linalg.norm(g) <= 1e-8:
                break
            if abs(v[0]) < 1e-9:
                surface.append(lower + (p + origin) / inverse_dx)
                break
            p -= inverse_dx * v[0] * g / (g @ g)
    surface_error = original(surface)[1] if surface else np.array([])

    # Preserve actual committed history and both normals, rather than a fabricated angular example.
    history_dir = args.diagnosis_dir / "position-18-production-history-ablation"
    saved = np.load(history_dir / "zero-probe.npz")
    history = {"old_local": saved["history_xi_local"][5293], "old_normal_local": saved["history_normal_local"][5293]}
    for suffix, filename in (("base", "geometry-base.npz"), ("trial", "geometry-alpha-0.0078125.npz")):
        geometry = np.load(history_dir / filename)
        record = int(np.flatnonzero(geometry["keys"] == 5293)[0])
        body = int(model.shape_body.numpy()[geometry["shape"][record]])
        history["normal_" + suffix] = geometry["normal"][record]
        history["rotation_" + suffix] = geometry["body_q"][body, 3:]
    fixture = args.output / "frozen_inputs.npz"
    np.savez_compressed(
        fixture,
        values=patch,
        coordinates=coordinates,
        inverse_dx=inverse_dx,
        origin=origin,
        lower=lower,
        baseline=baseline,
        **history,
    )
    report = {
        "status": "LOCAL_PROTOTYPE_ONLY",
        "shape": shape,
        "key": 3588,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "original_normal_jump_deg": _angle(baseline[0, 1:], baseline[1, 1:]),
        "candidate_normal_change_deg": _angle(candidate[0, 1:], candidate[1, 1:]),
        "original": baseline.tolist(),
        "candidate": candidate.tolist(),
        "mesh_distance": exact.tolist(),
        "candidate_fp64_max_abs_error": float(np.max(np.abs(candidate - oracle))),
        "local_geometry": {
            "sample_count": len(cloud),
            "candidate_minus_original_distance_max_m": float(np.max(np.abs(cubic_cloud[:, 0] - old_cloud[:, 0]))),
            "original_mesh_distance_error_max_m": float(np.max(np.abs(old_cloud[:, 0] - mesh_cloud))),
            "candidate_mesh_distance_error_max_m": float(np.max(np.abs(cubic_cloud[:, 0] - mesh_cloud))),
            "candidate_gradient_norm_min": float(np.linalg.norm(cubic_cloud[:, 1:], axis=1).min()),
            "candidate_surface_projected_count": len(surface),
            "candidate_surface_mesh_offset_max_m": float(np.max(np.abs(surface_error))) if len(surface_error) else None,
        },
        "unvalidated": [
            "whole-asset geometry",
            "texture and domain boundaries",
            "history physical calibration",
            "full residual and tangent",
            "trajectory and performance",
        ],
    }
    # Check the whole field's support before considering a collision-path change.
    boundary_rows = []
    indices = model._shape_sdf_index.numpy()
    descriptors = model._texture_sdf_data.numpy()
    for sdf_index in sorted({int(i) for i in indices if i >= 0}):
        d = descriptors[sdf_index]
        dimensions = np.rint((d["sdf_box_upper"] - d["sdf_box_lower"]) * d["inv_sdf_dx"]).astype(int) + 1
        grid = wp.empty(tuple(int(i) for i in dimensions), dtype=float, device="cuda:0")
        wp.launch(_export_patch, grid.shape, [model._texture_sdf_data, sdf_index, wp.vec3i(0), grid], device="cuda:0")
        nodes = grid.numpy().astype(float)
        second_differences = [float(np.max(np.abs(np.diff(nodes, n=2, axis=a)))) for a in range(3)]
        # Tensor Catmull-Rom has weight one-norm <= 1.25. This bounds its
        # difference from trilinear interpolation of these SAME canonical nodes.
        bound = 0.125 * sum(1.25**a * v for a, v in enumerate(second_differences))
        boundary = min(float(np.min(np.take(nodes, [0, 1, -2, -1], axis=a))) for a in range(3))
        refined = boundary_lower_bound(nodes)
        padding = 16
        padded = wp.empty(tuple(int(i) + 2 * padding for i in dimensions), dtype=float, device="cuda:0")
        wp.launch(
            _export_patch,
            padded.shape,
            [model._texture_sdf_data, sdf_index, wp.vec3i(-padding), padded],
            device="cuda:0",
        )
        extended = padded.numpy()
        padded_bound = boundary_lower_bound(extended)
        # Sample the whole asset's near-surface region, independently of the failed point.
        near = np.argwhere(np.abs(extended) < 0.002)
        near = near[np.all((near >= 2) & (near < np.array(extended.shape) - 3), axis=1)]
        rng = np.random.default_rng(430 + sdf_index)
        cloud_coordinates = near[rng.choice(len(near), min(4096, len(near)), replace=False)] + rng.uniform(
            -0.49, 0.49, (min(4096, len(near)), 3)
        )
        lo, inv = d["sdf_box_lower"], d["inv_sdf_dx"]
        cloud_positions = lo + (cloud_coordinates - padding) / inv
        owner = int(np.flatnonzero(indices == sdf_index)[0])
        old_values, mesh_values = original(cloud_positions, sdf_index, owner)
        new_values = wp.empty(len(cloud_coordinates), dtype=wp.vec4, device="cuda:0")
        new_valid = wp.empty(len(cloud_coordinates), dtype=int, device="cuda:0")
        wp.launch(
            query_patch,
            len(cloud_coordinates),
            [padded, wp.array(cloud_coordinates, dtype=wp.vec3, device="cuda:0"), wp.vec3(*inv), new_values, new_valid],
            device="cuda:0",
        )
        actual = new_values.numpy()
        np.savez_compressed(
            args.output / f"asset-{sdf_index}-cloud.npz",
            coordinates=cloud_coordinates,
            positions=cloud_positions,
            original=old_values,
            mesh=mesh_values,
            candidate=actual,
            valid=new_valid.numpy(),
        )
        boundary_rows.append(
            {
                "sdf_index": sdf_index,
                "shapes": np.flatnonzero(indices == sdf_index).tolist(),
                "dimensions": dimensions.tolist(),
                "dense_bytes": int(grid.size * 4),
                "canonical_trilinear_difference_bound_m": bound,
                "boundary_two_layers_min_phi_m": boundary,
                "valid_boundary_cells_cubic_lower_bound_m": refined,
                "padded_boundary_lower_bound_m": padded_bound,
                "padded_boundary_clear_of_shell": bool(
                    padded_bound > case.fixture["gap"] + case.fixture["particle_radius"]
                ),
                "padded_bytes": int(padded.size * 4),
                "padding_cells": padding,
                "cloud_count": len(cloud_coordinates),
                "cloud_invalid_count": int(np.sum(new_valid.numpy() == 0)),
                "cloud_original_mesh_error_max_m": float(np.max(np.abs(old_values[:, 0] - mesh_values))),
                "cloud_candidate_mesh_error_max_m": float(np.max(np.abs(actual[:, 0] - mesh_values))),
                "cloud_candidate_original_difference_max_m": float(np.max(np.abs(actual[:, 0] - old_values[:, 0]))),
                "contact_shell_m": float(case.fixture["gap"] + case.fixture["particle_radius"]),
                "boundary_clear_of_shell_with_bound": bool(
                    refined > case.fixture["gap"] + case.fixture["particle_radius"]
                ),
            }
        )
    report["whole_asset_domain_screen"] = boundary_rows
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    evidence_index(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
