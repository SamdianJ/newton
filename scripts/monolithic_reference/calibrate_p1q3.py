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
from scripts.monolithic_reference.p1q3_oracle import integrate_contact_over_mesh

_LENGTH = 0.04
_RADIUS = 0.0001
_STIFFNESS = 2e5
_WIDE_CASES = tuple(f"wide_{location}_{radius}" for location in ("edge", "vertex") for radius in (40, 80, 160))
_SUPPORTED_LOCAL_CASES = ("supported_edge", "supported_vertex")
_SUPPORTED_SHARP_CASES = ("resolved_sharp_box",)
_CASES = (
    "plane",
    "shared_edge",
    "shared_vertex",
    "sharp_box",
    "broad_box",
    *_SUPPORTED_SHARP_CASES,
    *_SUPPORTED_LOCAL_CASES,
    *_WIDE_CASES,
)
_SUPPORT_ENVELOPE_ID = "p1q3-resolved-local-v1"
_SUPPORTED_FIXTURE_IDS = {
    "plane_uniform_face_l012_v1",
    "broad_box_full_face_l012_v1",
    "shared_edge_r30_l234_v1",
    "shared_vertex_r30_l234_v1",
    "sample_between_box_w10_l234_v1",
}


def refined_tetrahedron(level: int, *, length: float = _LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Red-refine one tetrahedron with shared midpoint vertices and positive volumes."""
    if level not in range(5):
        raise ValueError("The declared refinement levels are 0 through 4")
    points = [np.array(p, dtype=np.float64) for p in ((0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length))]
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
    length = _LENGTH
    points, tets = refined_tetrahedron(level, length=length)
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
        shape_kind, support_class = "plane", "affine_full_face"
        support_fixture_id = "plane_uniform_face_l012_v1"
        builder.add_shape_plane(
            body=body, width=0.0, length=0.0, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
        )
    elif case.startswith("wide_") or case.startswith("supported_"):
        if case.startswith("wide_"):
            _, location, radius_mm = case.split("_")
            sphere_radius = int(radius_mm) / 1000.0
            penetration = 0.008
            support_class = "curved_local_patch"
            support_fixture_id = None
        else:
            _, location = case.split("_")
            sphere_radius = 0.03
            penetration = 0.008
            support_class = "resolved_curved_shared_feature"
            support_fixture_id = f"shared_{location}_r30_l234_v1"
        direction = np.array([0.0, -1.0, -1.0]) if location == "edge" else np.array([-1.0, -1.0, -1.0])
        anchor = np.array([length / 2, 0, 0]) if location == "edge" else np.zeros(3)
        center = tuple(anchor + direction / np.linalg.norm(direction) * (sphere_radius + _RADIUS - penetration))
        feature_width = 2 * sphere_radius
        shape_kind = "sphere"
        builder.add_shape_sphere(
            body=body,
            radius=sphere_radius,
            xform=wp.transform(center, wp.quat_identity()),
            cfg=cfg,
        )
    elif case in ("shared_edge", "shared_vertex"):
        center = (length / 2, -0.004, -0.004) if case == "shared_edge" else (-0.004, -0.004, -0.004)
        feature_width = 0.016
        shape_kind, support_class = "sphere", "curved_local_patch"
        support_fixture_id = None
        builder.add_shape_sphere(
            body=body, radius=feature_width / 2, xform=wp.transform(center, wp.quat_identity()), cfg=cfg
        )
    elif case in ("sharp_box", "broad_box", "resolved_sharp_box"):
        center = (length / 3, length / 3, -0.0004)
        feature_width = 0.002 if case == "sharp_box" else 0.01 if case == "resolved_sharp_box" else 0.08
        shape_kind = "box"
        support_class = (
            "inter_sample_local_feature"
            if case == "sharp_box"
            else "resolved_inter_sample_feature"
            if case == "resolved_sharp_box"
            else "affine_full_face"
        )
        support_fixture_id = (
            None
            if case == "sharp_box"
            else "sample_between_box_w10_l234_v1"
            if case == "resolved_sharp_box"
            else "broad_box_full_face_l012_v1"
        )
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
    fixed = np.isclose(points.sum(axis=1), length, rtol=0, atol=1e-9)
    reference_volume_weights = np.zeros(len(points), dtype=np.float64)
    for tet in tets:
        volume = abs(np.linalg.det(points[tet[1:]] - points[tet[0]])) / 6.0
        reference_volume_weights[tet] += volume / 4.0
    expected_total_mass = 1000.0 * reference_volume_weights.sum()
    expected_dynamic_mass = 1000.0 * reference_volume_weights[~fixed].sum()
    reduced_consistent_mass = 0.0
    for tet in tets:
        volume = abs(np.linalg.det(points[tet[1:]] - points[tet[0]])) / 6.0
        for local_a, particle_a in enumerate(tet):
            if fixed[particle_a]:
                continue
            for local_b, particle_b in enumerate(tet):
                if not fixed[particle_b]:
                    reduced_consistent_mass += 1000.0 * volume * (0.1 if local_a == local_b else 0.05)
    for index in np.flatnonzero(fixed):
        builder.particle_mass[index] = 0.0
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.0002)
    solver = SolverMonolithic(
        model, collision_pipeline=pipeline, contact_stiffness=_STIFFNESS, newton_max_iterations=20
    )
    state = model.state()
    eval_articulation_passive_candidate(model, state, solver._articulation)
    manifest = {
        "case": case,
        "level": level,
        "support_base_level": 2 if case in (*_SUPPORTED_LOCAL_CASES, *_SUPPORTED_SHARP_CASES) else 0,
        "surface_multiplier": 4 ** (level - 2 if case in (*_SUPPORTED_LOCAL_CASES, *_SUPPORTED_SHARP_CASES) else level),
        "length_m": length,
        "density_kg_m3": 1000.0,
        "contact_stiffness_n_m3": _STIFFNESS,
        "shape_center_m": center,
        "shape_kind": shape_kind,
        "feature_width_m": feature_width,
        "nominal_initial_penetration_m": penetration if case.startswith(("wide_", "supported_")) else None,
        "support_envelope_id": _SUPPORT_ENVELOPE_ID,
        "declared_support_class": support_class,
        "support_fixture_id": support_fixture_id,
        "fixed_mask": fixed.tolist(),
        "fixed_rule": f"reference x+y+z={length} m",
        "soft_contact_gap_m": 0.0002,
        "shape_margin_m": 0.0,
        "particle_radius_m": float(model.particle_radius.numpy()[0]),
        "continuum_mass_kg": float(expected_total_mass),
        "expected_dynamic_lumped_mass_kg": float(expected_dynamic_mass),
        "actual_dynamic_lumped_mass_kg": float(model.particle_mass.numpy().sum()),
        "fixed_removed_lumped_mass_kg": float(expected_total_mass - expected_dynamic_mass),
        "dynamic_lumped_mass_retained_fraction": float(expected_dynamic_mass / expected_total_mass),
        "reduced_consistent_uniform_mode_mass_kg": float(reduced_consistent_mass),
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
    manifest["solver_internal_config"] = asdict(solver._config)
    faces = model.tri_indices.numpy()
    manifest["maximum_boundary_edge_m"] = float(
        max(np.linalg.norm(points[face[i]] - points[face[j]]) for face in faces for i, j in ((0, 1), (1, 2), (2, 0)))
    )
    manifest["maximum_same_face_sample_spacing_m"] = manifest["maximum_boundary_edge_m"] / 2
    manifest["feature_width_to_maximum_boundary_edge"] = (
        feature_width / manifest["maximum_boundary_edge_m"] if feature_width is not None else None
    )
    manifest["maximum_boundary_edge_to_shape_radius"] = (
        manifest["maximum_boundary_edge_m"] / float(model.shape_scale.numpy()[0][0]) if shape_kind == "sphere" else None
    )
    if case == "resolved_sharp_box":
        barycentric = np.asarray(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
        base_faces = faces[np.all(np.abs(points[faces, 2]) <= 1.0e-12, axis=1)]
        samples = np.concatenate([barycentric @ points[face] for face in base_faces])
        manifest["feature_center_to_nearest_p1q3_sample_m"] = float(
            np.min(np.linalg.norm(samples[:, :2] - np.asarray(center[:2]), axis=1))
        )
        manifest["feature_center_coincident_with_p1q3_sample"] = False
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
    if manifest["case"] in ("shared_edge", "shared_vertex") or manifest["case"].startswith(("wide_", "supported_")):
        distance = min(_point_triangle_distance(center, x[face]) for face in model.tri_indices.numpy())
        return max(0.0, manifest["shape_scale"][0][0] + radius - distance)
    # This known base-interior witness certifies a miss; it is not a global box minimum oracle.
    point = x[:3].mean(axis=0)
    q = np.abs(point - center) - np.asarray(manifest["shape_scale"][0])
    distance = np.linalg.norm(np.maximum(q, 0.0)) + min(float(q.max()), 0.0)
    return max(0.0, float(radius - distance))


def _is_exact_supported_fixture(manifest: dict) -> bool:
    """Match only the measured fixtures; this is not an interpolated envelope."""
    fixture_id = manifest["support_fixture_id"]
    if fixture_id not in _SUPPORTED_FIXTURE_IDS:
        return False
    common = (
        manifest["length_m"] == _LENGTH
        and manifest["density_kg_m3"] == 1000.0
        and manifest["contact_stiffness_n_m3"] == _STIFFNESS
        and abs(manifest["particle_radius_m"] - _RADIUS) <= 1.0e-10
        and manifest["soft_contact_gap_m"] == 0.0002
        and manifest["shape_margin_m"] == 0.0
        and manifest["solver_parameters"]["newton_max_iterations"] == 20
    )
    if not common:
        return False
    if fixture_id == "plane_uniform_face_l012_v1":
        return (
            manifest["case"] == "plane"
            and manifest["level"] in (0, 1, 2)
            and manifest["shape_kind"] == "plane"
            and tuple(manifest["shape_center_m"]) == (0.0, 0.0, 0.001)
        )
    if fixture_id == "broad_box_full_face_l012_v1":
        return (
            manifest["case"] == "broad_box"
            and manifest["level"] in (0, 1, 2)
            and manifest["shape_kind"] == "box"
            and manifest["feature_width_m"] == 0.08
            and tuple(manifest["shape_center_m"]) == (_LENGTH / 3, _LENGTH / 3, -0.0004)
        )
    if fixture_id == "sample_between_box_w10_l234_v1":
        return (
            manifest["case"] == "resolved_sharp_box"
            and manifest["level"] in (2, 3, 4)
            and manifest["shape_kind"] == "box"
            and manifest["feature_width_m"] == 0.01
            and tuple(manifest["shape_center_m"]) == (_LENGTH / 3, _LENGTH / 3, -0.0004)
        )
    location = "edge" if fixture_id == "shared_edge_r30_l234_v1" else "vertex"
    direction = np.asarray((0.0, -1.0, -1.0) if location == "edge" else (-1.0, -1.0, -1.0))
    anchor = np.asarray((_LENGTH / 2, 0.0, 0.0) if location == "edge" else (0.0, 0.0, 0.0))
    expected_center = anchor + direction / np.linalg.norm(direction) * (0.03 + _RADIUS - 0.008)
    return (
        manifest["case"] == f"supported_{location}"
        and manifest["level"] in (2, 3, 4)
        and manifest["shape_kind"] == "sphere"
        and manifest["feature_width_m"] == 0.06
        and abs(manifest["shape_scale"][0][0] - 0.03) <= 1.0e-8
        and manifest["nominal_initial_penetration_m"] == 0.008
        and np.allclose(manifest["shape_center_m"], expected_center, rtol=0.0, atol=1.0e-12)
    )


def _measure(model, state, solver, manifest):
    contacts = solver.contacts
    count = int(contacts.soft_contact_count.numpy()[0])
    diagnostics = solver._contact._diagnostics(2)
    forces = contacts.force.numpy().astype(np.float64)
    residual = solver._final_residual.numpy().astype(np.float64)
    physical_generalized = solver._contact._projection[2].numpy().astype(np.float64)
    penetration = _geometric_penetration(model, state, manifest)
    center = np.asarray(manifest["shape_transform"][0][:3], dtype=np.float64)
    center[2] += float(state.joint_q.numpy()[0])
    oracle_faces = model.tri_indices.numpy()
    oracle_domain = "full_tet_boundary"
    full_boundary_oracle = None
    if manifest["case"] == "plane":
        # The PRD uniform-plane rule is an area-weight check on the intended
        # z=0 contact face.  Adjacent-face edge bands are retained separately
        # as a sampling observation instead of contaminating that 5% rule.
        positions = state.particle_q.numpy()
        uniform_face = np.all(np.abs(positions[oracle_faces, 2]) <= 1.0e-12, axis=1)
        gated_faces = oracle_faces[uniform_face]
        full_boundary_oracle = integrate_contact_over_mesh(
            positions,
            oracle_faces,
            shape=manifest["shape_kind"],
            center=center,
            scale=np.asarray(manifest["shape_scale"][0]),
            particle_radius=manifest["particle_radius_m"],
            stiffness=_STIFFNESS,
        )
        oracle_faces = gated_faces
        oracle_domain = "prd_uniform_z0_face"
    oracle = integrate_contact_over_mesh(
        state.particle_q.numpy(),
        oracle_faces,
        shape=manifest["shape_kind"],
        center=center,
        scale=np.asarray(manifest["shape_scale"][0]),
        particle_radius=manifest["particle_radius_m"],
        stiffness=_STIFFNESS,
    )
    oracle_force = oracle.force_magnitude
    relative_force_error = abs(float(np.linalg.norm(forces[:count, :3], axis=1).sum()) - oracle_force) / max(
        oracle_force, 1.0e-12
    )
    supported = _is_exact_supported_fixture(manifest)
    missed = oracle_force > 1.0e-12 and diagnostics["active_sample_count"] == 0
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
        "support_envelope": "SUPPORTED" if supported else "UNSUPPORTED",
        "support_reason": manifest["support_fixture_id"] if supported else manifest["declared_support_class"],
        "same_mesh_oracle": {
            "method": "independent float64 adaptive degree-5 triangle quadrature",
            "integration_domain": oracle_domain,
            "normal_force_n": oracle_force,
            "physical_world_force_resultant_n": oracle.force_resultant.tolist(),
            "physical_world_moment_resultant_nm": oracle.moment_resultant.tolist(),
            "consistent_nodal_physical_forces_n": oracle.consistent_nodal_forces.tolist(),
            "contact_energy_j": oracle.energy,
            "active_area_m2": oracle.active_area,
            "adaptive_sample_maximum_penetration_m": oracle.adaptive_sample_maximum_penetration,
            "force_magnitude_absolute_error_n": oracle.force_magnitude_absolute_error,
            "force_resultant_absolute_error_n": oracle.force_resultant_absolute_error.tolist(),
            "moment_resultant_absolute_error_nm": oracle.moment_resultant_absolute_error.tolist(),
            "energy_absolute_error_j": oracle.energy_absolute_error,
            "active_area_absolute_error_m2": oracle.active_area_absolute_error,
            "leaf_count": oracle.leaf_count,
            "relative_force_error": relative_force_error,
        },
        "full_boundary_oracle_observation": None
        if full_boundary_oracle is None
        else {
            "normal_force_n": full_boundary_oracle.force_magnitude,
            "relative_force_error": abs(
                float(np.linalg.norm(forces[:count, :3], axis=1).sum()) - full_boundary_oracle.force_magnitude
            )
            / max(full_boundary_oracle.force_magnitude, 1.0e-12),
            "meaning": "includes adjacent-face edge bands outside the PRD uniform-face weight gate",
        },
        "same_mesh_quadrature_threshold": 0.05 if manifest["case"] == "plane" else 0.1,
        "same_mesh_quadrature_gate": "PASS"
        if supported and relative_force_error <= (0.05 if manifest["case"] == "plane" else 0.1)
        else "FAIL"
        if supported
        else "NOT_APPLICABLE_UNSUPPORTED",
        "sampling_classification": "PHYSICS_EVALUATION_FAILURE"
        if status
        else "UNSUPPORTED_SAMPLING_MISS"
        if missed and not supported
        else "SUPPORTED_SAMPLING_MISS"
        if missed
        else "UNSUPPORTED_DETECTED"
        if not supported
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
    applied_joint_force = 0.1
    manifest.update(
        {
            "applied_joint_force_n": applied_joint_force,
            "dt_s": dt,
            "steps": steps,
            "pseudo_static_velocity_reset": False,
        }
    )
    control = model.control()
    control.joint_f.assign([applied_joint_force])
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
        "applied_joint_force_n": applied_joint_force,
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


def _case_levels(case: str) -> tuple[int, int, int]:
    return (2, 3, 4) if case in (*_SUPPORTED_LOCAL_CASES, *_SUPPORTED_SHARP_CASES) else (0, 1, 2)


def measure_motion_comparison(device, *, case: str, steps: int, dt: float):
    """Measure one predeclared 1x/4x/16x family and apply the unchanged 10% gate."""
    rows = [measure_motion(device, level=level, case=case, steps=steps, dt=dt) for level in _case_levels(case)]
    metrics = {}
    for name in (
        "rigid_displacement_m",
        "max_surface_penetration_m",
        "peak_surface_penetration_m",
        "soft_volume_mean_displacement_m",
    ):
        values = (
            [max(step["max_surface_penetration_m"] for step in row["trace"]) for row in rows]
            if name == "peak_surface_penetration_m"
            else [row["trace"][-1][name] for row in rows]
        )
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
    return rows, {
        "case": case,
        "role": "HARD_SUPPORTED_GATE" if case in _SUPPORTED_LOCAL_CASES else "OUTSIDE_ENVELOPE_OBSERVATION",
        "gate": "PASS" if passed else "FAIL",
        "threshold": 0.1,
        "metrics": metrics,
        "sampling_gate": "FAIL" if missed else "PASS",
        "normalization_floor_basis": "one float32 coordinate ULP at mesh length; not a calibrated solver noise floor",
    }


def calibrate(device, *, steps: int, dt: float):
    static = [measure_static(device, level=level, case=case) for case in _CASES for level in _case_levels(case)]
    plane_rows = [row for row in static if row["case"] == "plane"]
    planar = _relative_to_finest([row["normal_force_n"] for row in plane_rows])
    plane_valid = all(
        row["evaluation_status"] == 0 and row["active_sample_count"] > 0 and row["normal_force_n"] > 0
        for row in plane_rows
    )
    supported_rows = [row for row in static if row["support_envelope"] == "SUPPORTED"]
    same_mesh_quadrature_gate = (
        "PASS"
        if supported_rows and all(row["same_mesh_quadrature_gate"] == "PASS" for row in supported_rows)
        else "FAIL"
    )
    sampling_detection_gate = (
        "PASS"
        if supported_rows
        and all(row["sampling_classification"] == "DETECTED" for row in supported_rows)
        and all(row["sampling_classification"] != "SUPPORTED_SAMPLING_MISS" for row in static)
        else "FAIL"
    )
    mass_rows = []
    for row in plane_rows:
        manifest = row["manifest"]
        expected = manifest["expected_dynamic_lumped_mass_kg"]
        actual = manifest["actual_dynamic_lumped_mass_kg"]
        mass_rows.append(
            {
                "level": row["level"],
                "continuum_mass_kg": manifest["continuum_mass_kg"],
                "dynamic_lumped_mass_after_dirichlet_kg": actual,
                "expected_dynamic_lumped_mass_after_dirichlet_kg": expected,
                "fixed_removed_lumped_mass_kg": manifest["fixed_removed_lumped_mass_kg"],
                "dynamic_lumped_mass_retained_fraction": manifest["dynamic_lumped_mass_retained_fraction"],
                "reduced_consistent_uniform_mode_mass_kg": manifest["reduced_consistent_uniform_mode_mass_kg"],
                "lumped_to_reduced_consistent_mass_ratio": actual / manifest["reduced_consistent_uniform_mode_mass_kg"],
                "relative_error": abs(actual - expected) / expected,
            }
        )
    dirichlet_mass_audit_gate = (
        "PASS"
        if all(item["relative_error"] <= 1.0e-6 for item in mass_rows)
        and len({round(item["dynamic_lumped_mass_after_dirichlet_kg"], 12) for item in mass_rows}) == len(mass_rows)
        else "FAIL"
    )
    fem_load_probes = []
    for case in _SUPPORTED_LOCAL_CASES:
        case_rows = [row for row in static if row["case"] == case]
        inverse_mass_norms = []
        probe_rows = []
        for row in case_rows:
            nodal = np.asarray(row["same_mesh_oracle"]["consistent_nodal_physical_forces_n"])
            masses = np.asarray(row["manifest"]["particle_mass"])
            dynamic = masses > 0.0
            inverse_mass_norm = float(np.sqrt(np.sum(nodal[dynamic] ** 2 / masses[dynamic, None])))
            inverse_mass_norms.append(inverse_mass_norm)
            probe_rows.append(
                {
                    "level": row["level"],
                    "surface_multiplier": row["surface_multiplier"],
                    "dynamic_consistent_load_resultant_n": nodal[dynamic].sum(axis=0).tolist(),
                    "fixed_consistent_load_resultant_n": nodal[~dynamic].sum(axis=0).tolist(),
                    "lumped_inverse_mass_load_norm_n_sqrt_kg": inverse_mass_norm,
                }
            )
        relative = _relative_to_finest(inverse_mass_norms)
        fem_load_probes.append(
            {
                "case": case,
                "role": "OBSERVATION_ONLY",
                "probe": "independent oracle consistent nodal RHS evaluated without a dynamic contact solve",
                "rows": probe_rows,
                "inverse_mass_load_norm_relative_to_finest": relative,
                "limitation": "separates load discretization and Dirichlet mass; it is not an elastic K^-1 energy norm",
            }
        )
    motions, comparisons = [], []
    if steps:
        for case in (*_SUPPORTED_LOCAL_CASES, "shared_edge", "shared_vertex", *_WIDE_CASES):
            rows, comparison = measure_motion_comparison(device, case=case, steps=steps, dt=dt)
            motions.extend(rows)
            comparisons.append(comparison)
    hard_comparisons = [row for row in comparisons if row["role"] == "HARD_SUPPORTED_GATE"]
    supported_motion_gate = (
        "NOT_RUN"
        if steps != 20 or dt != 0.001
        else "PASS"
        if len(hard_comparisons) == len(_SUPPORTED_LOCAL_CASES)
        and all(row["gate"] == "PASS" for row in hard_comparisons)
        else "FAIL"
    )
    outside_comparisons = [row for row in comparisons if row["role"] == "OUTSIDE_ENVELOPE_OBSERVATION"]
    outside_motion_observation = (
        "NOT_RUN" if not steps else "PASS" if all(row["gate"] == "PASS" for row in outside_comparisons) else "FAIL"
    )
    plane_force_gate = (
        "PASS" if plane_valid and all(value is not None and value <= 0.05 for value in planar) else "FAIL"
    )
    c4_gate = (
        "PASS"
        if plane_force_gate
        == same_mesh_quadrature_gate
        == sampling_detection_gate
        == dirichlet_mass_audit_gate
        == supported_motion_gate
        == "PASS"
        else "FAIL"
    )
    return {
        "device": str(device),
        "c4_gate": c4_gate,
        "support_envelope": {
            "id": _SUPPORT_ENVELOPE_ID,
            "status": "FROZEN" if c4_gate == "PASS" else "CANDIDATE",
            "semantics": "exact measured fixtures only; no interval or phase interpolation",
            "supported_fixture_ids": sorted(_SUPPORTED_FIXTURE_IDS),
            "fixture_descriptions": {
                "plane_uniform_face_l012_v1": "exact uniform z=0 plane face at levels 0/1/2",
                "broad_box_full_face_l012_v1": "exact 80 mm full-face box at center (L/3,L/3,-0.4 mm)",
                "shared_edge_r30_l234_v1": "exact 30 mm sphere edge placement at levels 2/3/4",
                "shared_vertex_r30_l234_v1": "exact 30 mm sphere vertex placement at levels 2/3/4",
                "sample_between_box_w10_l234_v1": "exact-phase 10 mm local box at levels 2/3/4",
            },
            "supported_local_fixture": {
                "tet_length_m": _LENGTH,
                "sphere_radius_m": 0.03,
                "initial_penetration_m": 0.008,
                "edge_center_m": [0.02, -0.0156270598642227, -0.0156270598642227],
                "vertex_center_m": [-0.012759440949090732, -0.012759440949090732, -0.012759440949090732],
                "levels": [2, 3, 4],
                "surface_multipliers": [1, 4, 16],
                "observed_h_over_r_by_level": [
                    0.4714045207910317,
                    0.23570226039551584,
                    0.11785113019775792,
                ],
                "observed_h_over_r_is_not_an_interval_bound": True,
                "steps": 20,
                "dt_s": 0.001,
                "joint_force_n": 0.1,
                "newton_max_iterations": 20,
            },
            "resolved_sample_between_box_fixture": {
                "feature_width_m": 0.01,
                "center_xy_m": [_LENGTH / 3, _LENGTH / 3],
                "levels": [2, 3, 4],
                "surface_multipliers": [1, 4, 16],
                "observed_feature_width_over_h_by_level": [
                    0.7071067811865475,
                    1.414213562373095,
                    2.82842712474619,
                ],
                "phase_scope": "this exact center phase only; no arbitrary-phase guarantee",
            },
            "shape_capability_is_not_physical_support": True,
            "unsupported_probe_classes": ["curved_local_patch", "inter_sample_local_feature"],
        },
        "static": static,
        "motion": motions,
        "same_mesh_quadrature_gate": same_mesh_quadrature_gate,
        "sampling_detection_gate": sampling_detection_gate,
        "dirichlet_mass_audit_gate": dirichlet_mass_audit_gate,
        "dirichlet_mass_audit": mass_rows,
        "consistent_load_and_dirichlet_mass_observation": fem_load_probes,
        "plane_force_relative_to_finest": planar,
        "plane_force_gate": plane_force_gate,
        "plane_force_threshold": 0.05,
        "local_motion_comparisons": comparisons,
        "supported_local_motion_gate": supported_motion_gate,
        "outside_envelope_motion_observation": outside_motion_observation,
        "elastic_spatial_error_scope": (
            "Cross-grid motion changes P1 FEM, eliminated lumped mass, and P1Q3 together; it is retained but cannot "
            "identify contact quadrature error."
        ),
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


def _source_provenance(root: Path) -> dict:
    paths = (
        "scripts/monolithic_reference/calibrate_p1q3.py",
        "scripts/monolithic_reference/p1q3_oracle.py",
        "newton/_src/solvers/monolithic/collision.py",
        "newton/_src/solvers/monolithic/contact.py",
        "newton/_src/solvers/monolithic/tet.py",
        "newton/_src/solvers/monolithic/solver_monolithic.py",
    )
    sources = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    source_set = hashlib.sha256("".join(f"{path}:{sources[path]}\n" for path in sorted(sources)).encode()).hexdigest()
    return {"source_sha256": sources, "source_set_sha256": source_set}


def _device_kind(device) -> str:
    resolved = wp.get_device(device)
    if resolved.is_cpu:
        return "cpu"
    if resolved.is_cuda:
        return "cuda"
    return "unsupported"


def _freeze_audit(
    *,
    requested_devices,
    evidence,
    dynamic_steps: int,
    dt: float,
    git_dirty: bool,
    nonfinite_fields,
) -> tuple[bool, list[str]]:
    """Return the top-level freeze decision and stable rejection reasons."""
    reasons = []
    requested_kinds = [_device_kind(device) for device in requested_devices]
    evidence_kinds = [_device_kind(row["device"]) for row in evidence]
    if sorted(requested_kinds) != ["cpu", "cuda"]:
        reasons.append("requires_exactly_one_cpu_and_one_cuda_request")
    if sorted(evidence_kinds) != ["cpu", "cuda"] or len(evidence) != 2:
        reasons.append("requires_exactly_one_cpu_and_one_cuda_evidence")
    if dynamic_steps != 20 or dt != 0.001:
        reasons.append("requires_frozen_20_step_dt_1e-3_trial")
    if git_dirty:
        reasons.append("requires_clean_git_tree")
    if nonfinite_fields:
        reasons.append("nonfinite_evidence")
    if len(evidence) != 2 or not all(row["c4_gate"] == "PASS" for row in evidence):
        reasons.append("device_c4_gate_not_pass")
    return not reasons, reasons


def _validate_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing C4 artifact: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--dynamic-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    if args.dynamic_steps < 0 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("dynamic-steps must be nonnegative and dt finite/positive")
    try:
        _validate_new_output(args.output)
    except FileExistsError as error:
        parser.error(str(error))
    if sorted(_device_kind(device) for device in args.devices) != ["cpu", "cuda"]:
        parser.error("--devices must contain exactly one CPU and one CUDA device")
    root = Path(__file__).resolve().parents[2]
    source_provenance = _source_provenance(root)
    evidence = [calibrate(device, steps=args.dynamic_steps, dt=args.dt) for device in args.devices]
    if _source_provenance(root) != source_provenance:
        raise RuntimeError("C4 source files changed while evidence was being generated")
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    payload = {
        "status": "CANDIDATE",
        "schema_version": 3,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": git_dirty,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **source_provenance,
        "superdex_sha": "54ae749a042709e897cad12da66822c71bcd1b97",
        "superdex_executed": False,
        "platform": platform.platform(),
        "warp_version": wp.__version__,
        "seed": None,
        "build_mode": wp.config.mode,
        "prd_sha256": "f6632924870b6c29b7d8ea5f121564744453a18582e5f2cf905a7f1ba5a20090",
        "scope": (
            "C4 p1q3-resolved-local-v1 physical support envelope with resolved shared-edge/shared-vertex fixtures; "
            "legacy small-feature probes remain explicitly outside the envelope; no whole-P0/V0.1 exit or "
            "SuperDex comparison claimed"
        ),
        "sampling": [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        "evidence": evidence,
    }
    nonfinite = []
    payload = _finite_json(payload, nonfinite=nonfinite)
    payload["nonfinite_fields_stored_as_null"] = nonfinite
    frozen, freeze_reasons = _freeze_audit(
        requested_devices=args.devices,
        evidence=payload["evidence"],
        dynamic_steps=args.dynamic_steps,
        dt=args.dt,
        git_dirty=git_dirty,
        nonfinite_fields=nonfinite,
    )
    payload["status"] = "FROZEN" if frozen else "CANDIDATE"
    payload["freeze_audit"] = {
        "gate": "PASS" if frozen else "FAIL",
        "required_device_kinds": ["cpu", "cuda"],
        "rejection_reasons": freeze_reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
