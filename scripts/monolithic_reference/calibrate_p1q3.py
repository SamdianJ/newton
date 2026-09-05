# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure P1Q3 tessellation and sampling limits using actual collision/contact.

Run ``uv run -m scripts.monolithic_reference.calibrate_p1q3 --output report.json``.
Failure measurements are retained; this tool never changes production tolerances.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
from newton._src.solvers.monolithic.articulation import eval_articulation_passive_candidate
from newton._src.solvers.monolithic.contact import _MonolithicReturnedStateRole, evaluate_final_contacts

_LENGTH = 0.04
_RADIUS = 0.0001
_STIFFNESS = 2e5
_WIDE_CASES = tuple(f"wide_{location}_{radius}" for location in ("edge", "vertex") for radius in (40, 80, 160))
_CASES = ("plane", "shared_edge", "shared_vertex", "sharp_box", "broad_box", *_WIDE_CASES)


def refined_tetrahedron(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Red-refine one tetrahedron with shared midpoint vertices and positive volumes."""
    if level not in (0, 1, 2):
        raise ValueError("The declared refinement levels are 0, 1 and 2")
    points = [np.array(p, dtype=np.float64) for p in ((0, 0, 0), (_LENGTH, 0, 0), (0, _LENGTH, 0), (0, 0, _LENGTH))]
    tets = [(0, 1, 2, 3)]
    for _ in range(level):
        midpoint = {}
        children = []
        for a, b, c, d in tets:
            edges = []
            for i, j in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                key = tuple(sorted((i, j)))
                if key not in midpoint:
                    midpoint[key] = len(points)
                    points.append((points[i] + points[j]) / 2.0)
                edges.append(midpoint[key])
            ab, ac, ad, bc, bd, cd = edges
            children.extend(
                (
                    (a, ab, ac, ad),
                    (ab, b, bc, bd),
                    (ac, bc, c, cd),
                    (ad, bd, cd, d),
                    (ab, cd, ac, ad),
                    (ab, cd, ad, bd),
                    (ab, cd, bd, bc),
                    (ab, cd, bc, ac),
                )
            )
        tets = children
    points = np.asarray(points)
    tets = np.asarray(tets, dtype=np.int32)
    for tet in tets:
        if np.linalg.det(points[tet[1:]] - points[tet[0]]) < 0:
            tet[1], tet[2] = tet[2], tet[1]
    return points, tets


def _build_scene(device, level, case):
    points, tets = refined_tetrahedron(level)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=0.05, inertia=wp.mat33(0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.Z,
        armature=0.0,
        damping=0.0,
        friction=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=0.0,
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.NONE,
    )
    builder.add_articulation([joint])
    cfg = builder.ShapeConfig(margin=0.0)
    if case == "plane":
        center, feature_width = (0.0, 0.0, 0.001), None
        builder.add_shape_plane(
            body=body, width=0.0, length=0.0, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
        )
    elif case.startswith("wide_"):
        _, location, radius_mm = case.split("_")
        sphere_radius = int(radius_mm) / 1000.0
        direction = np.array([0.0, -1.0, -1.0]) if location == "edge" else np.array([-1.0, -1.0, -1.0])
        anchor = np.array([_LENGTH / 2, 0, 0]) if location == "edge" else np.zeros(3)
        center = tuple(anchor + direction / np.linalg.norm(direction) * (sphere_radius + _RADIUS - 0.008))
        feature_width = 2 * sphere_radius
        builder.add_shape_sphere(
            body=body, radius=sphere_radius, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
        )
    elif case in ("shared_edge", "shared_vertex"):
        center = (_LENGTH / 2, -0.004, -0.004) if case == "shared_edge" else (-0.004, -0.004, -0.004)
        feature_width = 0.016
        builder.add_shape_sphere(
            body=body, radius=feature_width / 2, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
        )
    elif case in ("sharp_box", "broad_box"):
        center = (_LENGTH / 3, _LENGTH / 3, -0.0004)
        feature_width = 0.002 if case == "sharp_box" else 0.08
        builder.add_shape_box(
            body=body,
            hx=feature_width / 2,
            hy=feature_width / 2,
            hz=0.0006,
            xform=wp.transform(center, wp.quat_identity()),
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown case: {case}")
    builder.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        vertices=points.tolist(),
        indices=tets.ravel().tolist(),
        density=1000.0,
        k_mu=10000.0 / 2.6,
        k_lambda=10000.0 * 0.3 / (1.3 * 0.4),
        k_damp=0.0,
        particle_radius=_RADIUS,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    # The same geometric opposite-face Dirichlet condition at every resolution.
    fixed = np.isclose(points.sum(axis=1), _LENGTH, rtol=0, atol=1e-9)
    for index in np.flatnonzero(fixed):
        builder.particle_mass[index] = 0.0
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.0002)
    solver = SolverMonolithic(model, collision_pipeline=pipeline, contact_stiffness=_STIFFNESS)
    state = model.state()
    eval_articulation_passive_candidate(model, state, solver._articulation)
    manifest = {
        "case": case,
        "level": level,
        "surface_multiplier": 4**level,
        "length_m": _LENGTH,
        "density_kg_m3": 1000.0,
        "contact_stiffness_n_m3": _STIFFNESS,
        "shape_center_m": center,
        "feature_width_m": feature_width,
        "fixed_mask": fixed.tolist(),
        "fixed_rule": "reference x+y+z=0.04 m",
        "soft_contact_gap_m": 0.0002,
        "shape_margin_m": 0.0,
        "particle_radius_m": float(model.particle_radius.numpy()[0]),
    }
    for name in (
        "particle_q",
        "particle_mass",
        "tet_indices",
        "tri_indices",
        "tet_materials",
        "body_mass",
        "body_inertia",
        "body_com",
        "shape_scale",
        "shape_transform",
        "shape_type",
        "joint_q",
        "joint_qd",
        "gravity",
    ):
        manifest[name] = getattr(model, name).numpy().tolist()
    manifest["solver_parameters"] = {
        name: getattr(solver, name)
        for name in ("newton_max_iterations", "line_search_max_iterations", "linear_max_iterations", "linear_tolerance")
    }
    manifest["draft_internal_solver_parameters"] = asdict(solver._config)
    faces = model.tri_indices.numpy()
    manifest["maximum_boundary_edge_m"] = float(
        max(np.linalg.norm(points[face[i]] - points[face[j]]) for face in faces for i, j in ((0, 1), (1, 2), (2, 0)))
    )
    manifest["maximum_same_face_sample_spacing_m"] = manifest["maximum_boundary_edge_m"] / 2
    return model, state, solver, manifest


def _point_triangle_distance(point, triangle):
    a, b, c = triangle
    normal = np.cross(b - a, c - a)
    denominator = normal @ normal
    projected = point - normal * ((point - a) @ normal) / denominator
    coordinates = np.linalg.lstsq(np.stack((b - a, c - a), axis=1), projected - a, rcond=None)[0]
    if np.min(coordinates) >= 0 and coordinates.sum() <= 1:
        return float(np.linalg.norm(point - projected))
    result = float("inf")
    for v, w in ((a, b), (b, c), (c, a)):
        fraction = np.clip(((point - v) @ (w - v)) / ((w - v) @ (w - v)), 0.0, 1.0)
        result = min(result, float(np.linalg.norm(point - (v + fraction * (w - v)))))
    return result


def _geometric_penetration(model, state, manifest):
    x = state.particle_q.numpy().astype(np.float64)
    center = np.asarray(manifest["shape_transform"][0][:3]) + np.array([0, 0, state.joint_q.numpy()[0]])
    radius = manifest["particle_radius_m"]
    if manifest["case"] == "plane":
        return max(0.0, float(center[2] + radius - x[:, 2].min()))
    if manifest["case"] in ("shared_edge", "shared_vertex") or manifest["case"].startswith("wide_"):
        distance = min(_point_triangle_distance(center, x[face]) for face in model.tri_indices.numpy())
        return max(0.0, manifest["shape_scale"][0][0] + radius - distance)
    # This known base-interior witness certifies a miss; it is not a global box minimum oracle.
    point = x[:3].mean(axis=0)
    q = np.abs(point - center) - np.asarray(manifest["shape_scale"][0])
    distance = np.linalg.norm(np.maximum(q, 0.0)) + min(float(q.max()), 0.0)
    return max(0.0, float(radius - distance))


def _measure(model, state, solver, manifest):
    contacts = solver.contacts
    count = int(contacts.soft_contact_count.numpy()[0])
    diagnostics = solver._contact._diagnostics(2)
    forces = contacts.force.numpy().astype(np.float64)
    residual = solver._final_residual.numpy().astype(np.float64)
    physical_generalized = solver._contact._projection[2].numpy().astype(np.float64)
    penetration = _geometric_penetration(model, state, manifest)
    missed = penetration > 0 and diagnostics["active_sample_count"] == 0
    status = int(solver._contact_status.numpy()[0])
    return {
        "case": manifest["case"],
        "level": manifest["level"],
        "surface_multiplier": manifest["surface_multiplier"],
        "device": str(model.device),
        "device_name": model.device.name,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "manifest": manifest,
        "tet_count": model.tet_count,
        "boundary_face_count": model.tri_count,
        "particle_count": model.particle_count,
        "records": {
            name: getattr(contacts, name).numpy()[:count].tolist()
            for name in (
                "soft_contact_indices",
                "soft_contact_barycentric",
                "soft_contact_shape",
                "soft_contact_body_pos",
                "soft_contact_normal",
            )
        },
        "record_count": count,
        "capacity": solver.collision_pipeline.soft_contact_max,
        "evaluation_status": int(solver._contact_status.numpy()[0]),
        "normal_force_n": float(np.linalg.norm(forces[:count, :3], axis=1).sum()),
        "physical_world_wrench": forces.tolist(),
        "physical_world_force_resultant_n": forces[:count, :3].sum(axis=0).tolist(),
        "residual_contribution": residual.tolist(),
        "generalized_physical_force": physical_generalized.tolist(),
        "witness_penetration_m": penetration,
        "penetration_oracle": "exact triangle-sphere/plane"
        if "box" not in manifest["case"]
        else "known triangle-interior witness lower bound",
        "sampling_classification": "PHYSICS_EVALUATION_FAILURE"
        if status
        else "UNSUPPORTED_SAMPLING_MISS"
        if missed
        else "DETECTED",
        **diagnostics,
    }


def measure_static(device, *, level: int, case: str):
    """Evaluate actual P1Q3 collision and physical contact force without stepping."""
    model, state, solver, manifest = _build_scene(device, level, case)
    solver.collision_pipeline.collide(state, solver.contacts)
    evaluate_final_contacts(
        model,
        state,
        solver.contacts,
        solver.collision_pipeline,
        solver._articulation,
        solver._contact,
        solver._final_residual,
        solver._contact_status,
        step_generation=0,
        returned_state_role=_MonolithicReturnedStateRole.STATE_OUT_CONVERGED,
    )
    if int(solver._contact_status.numpy()[0]) == 0:
        solver._contact._publish_final_forces(
            state,
            solver.contacts,
            step_generation=0,
            returned_state_role=_MonolithicReturnedStateRole.STATE_OUT_CONVERGED,
        )
    row = _measure(model, state, solver, manifest)
    row["mode"] = "static collision/contact measurement; no solver step claimed"
    return row


def measure_motion(device, *, level: int, case: str, steps: int, dt: float):
    """Record identical physical load and geometric fixed-boundary motion trials."""
    model, state, solver, manifest = _build_scene(device, level, case)
    manifest.update({"applied_joint_force_n": 0.1, "dt_s": dt, "steps": steps})
    control = model.control()
    control.joint_f.assign([0.1])
    target = model.state()
    initial = state.particle_q.numpy().astype(np.float64)
    # Reference volume quadrature includes prescribed nodes, independent of solver's zero fixed mass.
    weights = np.zeros(model.particle_count)
    for tet in model.tet_indices.numpy():
        volume = abs(np.linalg.det(initial[tet[1:]] - initial[tet[0]])) / 6.0
        weights[tet] += volume / 4.0
    weights /= weights.sum()
    trace = []
    for step in range(steps):
        solver.step(state, target, control, None, dt)
        stats = solver.last_stats
        state, target = target, state
        cache_valid = solver._contact.final_force_generation is not None
        penetration = _geometric_penetration(model, state, manifest)
        active = solver._contact._diagnostics(2)["active_sample_count"] if cache_valid else None
        trace.append(
            {
                "step": step,
                "time_s": (step + 1) * dt,
                "status": stats.status.name,
                "failure_reason": stats.failure_reason,
                "converged": stats.converged,
                "rolled_back": stats.rolled_back,
                "rigid_displacement_m": float(state.joint_q.numpy()[0]),
                "soft_volume_mean_displacement_m": (weights @ (state.particle_q.numpy() - initial)).tolist(),
                "max_surface_penetration_m": penetration,
                "active_sample_count": active,
                "sampling_miss": penetration > 0 and active == 0,
                "sampled_max_penetration_m": stats.max_penetration,
                "merit_final": stats.merit_final,
                "min_det_f": stats.min_det_f,
                "contact_force_imbalance": stats.contact_force_imbalance,
                "contact_moment_imbalance": stats.contact_moment_imbalance,
                "contact_sign_error": stats.contact_sign_error,
                "particle_q_m": state.particle_q.numpy().tolist(),
                "normal_force_n": float(np.linalg.norm(solver.contacts.force.numpy()[:, :3], axis=1).sum())
                if cache_valid
                else None,
                "force_cache_valid": cache_valid,
                "physical_world_wrench_resultant": solver.contacts.force.numpy().sum(axis=0).tolist()
                if cache_valid
                else None,
                "residual_contribution": solver._final_residual.numpy().tolist() if cache_valid else None,
                "generalized_physical_force": solver._contact._projection[2].numpy().tolist() if cache_valid else None,
            }
        )
    return {
        "case": case,
        "level": level,
        "device": str(model.device),
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "applied_joint_force_n": 0.1,
        "dt_s": dt,
        "steps": steps,
        "trace": trace,
        "all_steps_success": all(row["converged"] and not row["rolled_back"] for row in trace),
    }


def _relative_to_finest(values, reference_floor=0.0):
    denominator = abs(values[-1])
    if denominator < reference_floor:
        return [None] * len(values)
    return [abs(value - values[-1]) / denominator if denominator else (0.0 if value == 0 else None) for value in values]


def calibrate(device, *, steps: int, dt: float):
    static = [measure_static(device, level=level, case=case) for case in _CASES for level in range(3)]
    plane_rows = [row for row in static if row["case"] == "plane"]
    planar = _relative_to_finest([row["normal_force_n"] for row in plane_rows])
    plane_valid = all(
        row["evaluation_status"] == 0 and row["active_sample_count"] > 0 and row["normal_force_n"] > 0
        for row in plane_rows
    )
    motions, comparisons = [], []
    if steps:
        for case in ("shared_edge", "shared_vertex", *_WIDE_CASES):
            rows = [measure_motion(device, level=level, case=case, steps=steps, dt=dt) for level in range(3)]
            motions.extend(rows)
            metrics = {}
            for name in ("rigid_displacement_m", "max_surface_penetration_m", "soft_volume_mean_displacement_m"):
                values = [row["trace"][-1][name] for row in rows]
                if name == "soft_volume_mean_displacement_m":
                    values = [float(np.linalg.norm(value)) for value in values]
                floor = float(np.spacing(np.float32(_LENGTH))) if "displacement" in name else 0.0
                relative = _relative_to_finest(values, floor)
                metrics[name] = {
                    "values": values,
                    "relative_difference_from_finest": relative,
                    "absolute_difference_from_finest": [abs(value - values[-1]) for value in values],
                    "reference_floor_m": floor,
                    "normalization": "UNMEASURABLE" if any(value is None for value in relative) else "RESOLVED",
                }
            missed = any(step["sampling_miss"] for row in rows for step in row["trace"])
            passed = (
                not missed
                and all(row["all_steps_success"] for row in rows)
                and all(
                    value is not None and value <= 0.1
                    for metric in metrics.values()
                    for value in metric["relative_difference_from_finest"]
                )
            )
            comparisons.append(
                {
                    "case": case,
                    "gate": "PASS" if passed else "FAIL",
                    "threshold": 0.1,
                    "metrics": metrics,
                    "sampling_gate": "FAIL" if missed else "PASS",
                    "normalization_floor_basis": "one float32 coordinate ULP at mesh length; not a calibrated solver noise floor",
                }
            )
    return {
        "device": str(device),
        "static": static,
        "motion": motions,
        "plane_force_relative_to_finest": planar,
        "plane_force_gate": "PASS"
        if plane_valid and all(value is not None and value <= 0.05 for value in planar)
        else "FAIL",
        "plane_force_threshold": 0.05,
        "local_motion_comparisons": comparisons,
        "local_motion_gate": "NOT_RUN"
        if not steps
        else "PASS"
        if all(row["gate"] == "PASS" for row in comparisons)
        else "FAIL",
    }


def _finite_json(value, path="", nonfinite=None):
    if nonfinite is None:
        nonfinite = []
    if isinstance(value, float) and not np.isfinite(value):
        nonfinite.append(path)
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item, path + "/" + key, nonfinite) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item, path + "/" + str(index), nonfinite) for index, item in enumerate(value)]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    if args.dynamic_steps < 0 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("dynamic-steps must be nonnegative and dt finite/positive")
    root = Path(__file__).resolve().parents[2]
    payload = {
        "status": "DRAFT",
        "schema_version": 1,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "superdex_sha": "54ae749a042709e897cad12da66822c71bcd1b97",
        "superdex_executed": False,
        "platform": platform.platform(),
        "warp_version": wp.__version__,
        "seed": None,
        "build_mode": wp.config.mode,
        "prd_sha256": "f6632924870b6c29b7d8ea5f121564744453a18582e5f2cf905a7f1ba5a20090",
        "scope": "C4 draft support measurements; no P0/V0.1 exit or SuperDex comparison claimed",
        "sampling": [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        "evidence": [calibrate(device, steps=args.dynamic_steps, dt=args.dt) for device in args.devices],
    }
    nonfinite = []
    payload = _finite_json(payload, nonfinite=nonfinite)
    payload["nonfinite_fields_stored_as_null"] = nonfinite
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
