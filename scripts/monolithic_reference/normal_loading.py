# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run and measure the versioned normal-loading trajectory.

Use ``uv run -m scripts.monolithic_reference.normal_loading --device cpu
--output /path/to/evidence``. Calibration, contact-support, and reference
evidence are independent axes; numerical success alone never certifies exit.
"""

import argparse
import enum
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import fields, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton

FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")
LEGACY_FIXTURE = FIXTURE_DIRECTORY / "normal_loading_draft_v1.json"
DEFAULT_FIXTURE = FIXTURE_DIRECTORY / "normal_loading_v2.json"
_STATUS_AXES = ("calibration", "support", "reference")


def load_fixture(path=DEFAULT_FIXTURE):
    """Read a supported version of the normal-loading input."""
    fixture = json.loads(Path(path).read_text())
    _validate_fixture(fixture)
    return fixture


def _axis_status(fixture, axis):
    if fixture["schema_version"] == "normal_loading_draft/v1":
        if axis == "calibration":
            return fixture["calibration_status"]
        if axis == "reference" and fixture["reference_envelope"] is not None:
            return "FROZEN"
        return "UNFROZEN"
    return fixture[f"{axis}_status"]


def _axis_provenance(fixture, axis):
    return fixture.get(f"{axis}_provenance")


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_status_axes(fixture):
    for axis in _STATUS_AXES:
        status = _axis_status(fixture, axis)
        allowed = (
            ("UNFROZEN", "CANDIDATE", "BLOCKED", "FROZEN")
            if axis == "reference"
            else (
                "UNFROZEN",
                "CANDIDATE",
                "FROZEN",
            )
        )
        if status not in allowed:
            raise ValueError(f"Invalid {axis}_status")
        provenance = _axis_provenance(fixture, axis)
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(f"Invalid {axis}_provenance")
        if status == "FROZEN":
            if (
                provenance is None
                or not isinstance(provenance.get("version"), str)
                or not provenance["version"]
                or not _valid_sha256(provenance.get("sha256"))
            ):
                raise ValueError(f"Frozen {axis} evidence requires a nonempty version and SHA-256")
        if status == "BLOCKED" and (
            provenance is None or not isinstance(provenance.get("reason"), str) or not provenance["reason"].strip()
        ):
            raise ValueError(f"Blocked {axis} evidence requires a nonempty reason")
    calibration = _axis_provenance(fixture, "calibration")
    if calibration is not None and calibration.get("candidate_private_config") is not None:
        candidate = calibration["candidate_private_config"]
        if not isinstance(candidate, dict) or any(not isinstance(value, dict) for value in candidate.values()):
            raise ValueError("candidate_private_config must map device classes to config dictionaries")
    if _axis_status(fixture, "reference") == "FROZEN" and fixture["reference_envelope"] is None:
        raise ValueError("Frozen reference evidence requires reference_envelope")


def _validate_fixture(fixture):
    """Check only the concrete single-tet, Z-plane input this runner implements."""
    rigid, soft, contact, drive, limits = (
        fixture[name] for name in ("rigid", "soft", "contact", "drive", "acceptance")
    )
    schema_version = fixture["schema_version"]
    if fixture["status"] != "DRAFT" or schema_version not in ("normal_loading_draft/v1", "normal_loading/v2"):
        raise ValueError("This runner only supports DRAFT normal_loading_draft/v1 or normal_loading/v2 input")
    if schema_version == "normal_loading_draft/v1":
        if fixture["calibration_status"] != "UNFROZEN" or fixture["reference_envelope"] is not None:
            raise ValueError("Legacy normal_loading_draft/v1 must remain unfrozen")
    else:
        _validate_status_axes(fixture)
    if rigid["shape_type"] != "infinite_plane" or rigid["joint_type"] != "PRISMATIC" or contact["quadrature"] != "P1Q3":
        raise ValueError("Only PRISMATIC infinite_plane with P1Q3 is supported")
    if rigid["approach_axis_world"] != [0.0, 0.0, 1.0]:
        raise ValueError("The plane and compressive approach axis must be world +Z")
    if (
        drive["kind"] != "external frozen joint force: kp*(q_target-q)+kd*(qd_target-qd)"
        or drive["trajectory"] != "quintic position and analytic velocity, followed by constant settle target"
    ):
        raise ValueError("Only the declared frozen-force quintic drive is supported")
    for values, positive, nonnegative in (
        (fixture, ("dt_s",), ()),
        (rigid, ("mass_kg",), ()),
        (soft, ("density_kg_m3", "mu_pa", "lambda_pa"), ("particle_radius_m",)),
        (contact, ("stiffness_n_m3",), ("soft_contact_gap_m", "shape_margin_m")),
        (drive, ("kp_n_m", "free_end_s", "loading_end_s", "free_target_m", "loading_target_m"), ("kd_n_s_m",)),
        (
            limits,
            (
                "min_det_f",
                "penetration_acceptance_limit_m",
                "delta_soft_min_m",
                "force_floor_n",
                "force_balance_relative_tolerance",
                "linear_tolerance",
            ),
            (),
        ),
    ):
        for name in positive + nonnegative:
            value = values[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (name in positive and value == 0)
            ):
                raise ValueError(f"Invalid finite range for {name}")
    for values, names in (
        (fixture, ("substeps",)),
        (limits, ("minimum_substeps", "onset_consecutive_substeps", "settled_window_substeps")),
    ):
        if any(type(values[name]) is not int or values[name] <= 0 for name in names):
            raise ValueError("Step counts must be positive integers")
    if type(limits["maximum_consecutive_non_success"]) is not int or limits["maximum_consecutive_non_success"] < 0:
        raise ValueError("Maximum consecutive failures must be a nonnegative integer")
    for name in ("minimum_converged_ratio", "free_space_completion_minimum"):
        if not 0 < limits[name] <= 1:
            raise ValueError(f"Invalid acceptance ratio {name}")
    if not (
        drive["free_end_s"] < drive["loading_end_s"] < fixture["substeps"] * fixture["dt_s"]
        and drive["free_target_m"] < drive["loading_target_m"]
    ):
        raise ValueError("Drive must have ordered free/loading/settle phases and increasing targets")
    for values, name in ((rigid, "inertia_kg_m2"), (rigid, "reference_point_body_m"), (fixture, "gravity_m_s2")):
        vector = np.asarray(values[name], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or (name == "inertia_kg_m2" and np.any(vector <= 0)):
            raise ValueError(f"Invalid three-vector {name}")
    interval = np.asarray(limits["closure_secant_interval_m"], dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not 0 < interval[0] < interval[1]:
        raise ValueError("Secant closure interval must be finite, positive and ordered")
    if (
        soft["tet_indices"] != [[0, 1, 2, 3]]
        or soft["fixed_nodes"] != [False, False, False, True]
        or soft["probe_nodes"] != [3]
        or soft["surface_nodes"] != [0, 1, 2]
        or soft["damping"] != 0
    ):
        raise ValueError("Only one ordered tetrahedron with fixed apex, three base nodes and zero damping is supported")
    positions = np.asarray(soft["rest_positions_m"], dtype=float)
    if positions.shape != (4, 3) or not np.isfinite(positions).all():
        raise ValueError("Expected four finite three-dimensional rest positions")
    if (
        not np.all(positions[:3, 2] == positions[0, 2])
        or positions[3, 2] <= positions[0, 2]
        or np.linalg.det((positions[1:] - positions[0]).T) <= 0
    ):
        raise ValueError("Expected a positive-volume tetrahedron with horizontal base below the apex")
    if (
        not math.isfinite(rigid["initial_plane_z_m"])
        or rigid["initial_plane_z_m"] + soft["particle_radius_m"] + contact["shape_margin_m"] >= positions[0, 2]
    ):
        raise ValueError("The initial plane must lie below the base with positive physical clearance")


def command_at(time_s, fixture):
    """Evaluate the declared quintic command without using it as measured closure."""
    drive = fixture["drive"]
    if time_s < drive["free_end_s"]:
        start, duration, q0, q1, phase = 0.0, drive["free_end_s"], 0.0, drive["free_target_m"], "free_space"
    elif time_s < drive["loading_end_s"]:
        start, duration = drive["free_end_s"], drive["loading_end_s"] - drive["free_end_s"]
        q0, q1, phase = drive["free_target_m"], drive["loading_target_m"], "loading"
    else:
        return drive["loading_target_m"], 0.0, "settle"
    t = min(max((time_s - start) / duration, 0.0), 1.0)
    position = t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
    velocity = 30.0 * t * t * (1.0 - t) ** 2 / duration
    return q0 + (q1 - q0) * position, (q1 - q0) * velocity, phase


def build_scene(fixture, *, device):
    """Build the declared single-finger articulation and anchored tetrahedron."""
    _validate_fixture(fixture)
    rigid, soft, contact = fixture["rigid"], fixture["soft"], fixture["contact"]
    builder = newton.ModelBuilder(gravity=tuple(fixture["gravity_m_s2"]))
    body = builder.add_link(mass=rigid["mass_kg"], inertia=wp.diag(wp.vec3(*rigid["inertia_kg_m2"])))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=tuple(rigid["approach_axis_world"]),
        parent_xform=wp.transform((0.0, 0.0, rigid["initial_plane_z_m"]), wp.quat_identity()),
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
    builder.add_shape_plane(
        body=body, width=0.0, length=0.0, cfg=builder.ShapeConfig(density=0.0, margin=contact["shape_margin_m"])
    )
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=soft["rest_positions_m"],
        indices=np.asarray(soft["tet_indices"]).ravel().tolist(),
        density=soft["density_kg_m3"],
        k_mu=soft["mu_pa"],
        k_lambda=soft["lambda_pa"],
        k_damp=soft["damping"],
        particle_radius=soft["particle_radius_m"],
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    for particle, fixed in enumerate(soft["fixed_nodes"]):
        if fixed:
            builder.particle_mass[particle] = 0.0
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=contact["soft_contact_gap_m"])
    solver = SolverMonolithic(model, collision_pipeline=pipeline, contact_stiffness=contact["stiffness_n_m3"])
    return model, state, model.control(), solver


def _json_value(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _measure(model, state, solver, fixture, step, time_s, command, frozen_force):
    stats = solver.last_stats
    positions = state.particle_q.numpy().copy()
    q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()
    transforms = state.body_q.numpy().copy()
    axis = np.asarray(fixture["rigid"]["approach_axis_world"])
    probe = positions[fixture["soft"]["probe_nodes"]].mean(axis=0)
    surface = positions[fixture["soft"]["surface_nodes"]].mean(axis=0)
    rest = model.particle_q.numpy()
    initial_thickness = np.linalg.norm(
        rest[fixture["soft"]["probe_nodes"]].mean(axis=0) - rest[fixture["soft"]["surface_nodes"]].mean(axis=0)
    )
    reference = np.asarray(
        wp.transform_point(wp.transform(*transforms[0]), wp.vec3(*fixture["rigid"]["reference_point_body_m"]))
    )
    count = int(solver.contacts.soft_contact_count.numpy()[0])
    wrench = solver.contacts.force.numpy()[:count].astype(np.float64).sum(axis=0)
    residual = solver._contact._residual[2].numpy().copy()
    generalized = solver._contact._projection[2].numpy().copy()
    contact_r_x = np.zeros_like(positions)
    contact_r_x[solver._layout.dynamic_particle_ids.numpy()] = residual[1:].reshape(-1, 3)
    public_ratios = [stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x]
    physical = np.concatenate(
        (
            positions.ravel(),
            q,
            qd,
            transforms.ravel(),
            state.particle_qd.numpy().ravel(),
            state.body_qd.numpy().ravel(),
            wrench,
            residual,
            generalized,
            [frozen_force],
        )
    )
    q_error = np.linalg.norm(residual[:1] + generalized[:1]) / max(
        np.linalg.norm(residual[:1]) + np.linalg.norm(generalized[:1]), 1e-30
    )
    x_error = np.linalg.norm(residual[1:] + generalized[1:]) / max(
        np.linalg.norm(residual[1:]) + np.linalg.norm(generalized[1:]), 1e-30
    )
    return _json_value(
        {
            "step": step,
            "time_s": time_s,
            "phase": command[2],
            "command_q_m": command[0],
            "command_qd_m_s": command[1],
            "frozen_joint_force_n": frozen_force,
            "joint_q": q,
            "joint_qd": qd,
            "link_xform": transforms,
            "node_positions_m": positions,
            "soft_com_m": np.average(positions, axis=0, weights=model.particle_mass.numpy()),
            "relative_rigid_probe_m": float(np.dot(reference - probe, axis)),
            "actual_closure_m": None,
            "soft_response_m": float(initial_thickness - np.linalg.norm(probe - surface)),
            "normal_physical_force_n": wrench[:3],
            "normal_compressive_force_n": float(-np.dot(wrench[:3], axis)),
            "physical_world_force_or_wrench": [wrench],
            "generalized_physical_force": generalized,
            "residual_contribution": {"q": residual[:1], "x": contact_r_x},
            "q_sign_projection_error": q_error,
            "x_sign_projection_error": x_error,
            "min_det_f": stats.min_det_f,
            "penetration_m": stats.max_penetration,
            "finite_state": bool(np.isfinite(physical).all()),
            "convergence_status": stats.status.value,
            "converged": stats.converged,
            "committed_unconverged": stats.status == SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS,
            "rolled_back": stats.rolled_back,
            "nonlinear_iterations": stats.nonlinear_iterations,
            "linear_iterations": stats.linear_iterations,
            "nonlinear_convergence_ratios": public_ratios,
            "stats": {field.name: getattr(stats, field.name) for field in fields(stats)},
            "timing_s": {
                "assembly": None,
                "collision": None,
                "linear_solve": None,
                "total": stats.timings["step"] / 1000.0,
            },
            "unavailable_fields": {
                f"timing_s.{name}": "Only total step timing is currently exposed"
                for name in ("assembly", "collision", "linear_solve")
            },
        }
    )


def assess_run(records, fixture, *, execution_error=None):
    """Apply numerical gates separately from three independent evidence axes."""
    limits = fixture["acceptance"]
    onset, streak, maximum_streak, failures = None, 0, 0, 0
    for i, record in enumerate(records):
        streak = streak + 1 if record["normal_compressive_force_n"] > limits["force_floor_n"] else 0
        if onset is None and streak >= limits["onset_consecutive_substeps"]:
            onset = i - streak + 1
        failures = 0 if record["converged"] else failures + 1
        maximum_streak = max(maximum_streak, failures)
    if onset is not None:
        origin = records[onset]["relative_rigid_probe_m"]
        for record in records[onset:]:
            record["actual_closure_m"] = record["relative_rigid_probe_m"] - origin
    free = [r for r in records if r["phase"] == "free_space"]
    settle = [r for r in records if r["phase"] == "settle"][-limits["settled_window_substeps"] :]
    count = len(records)
    converged_ratio = sum(r["converged"] for r in records) / max(count, 1)
    completion = free[-1]["joint_q"][0] / max(free[-1]["command_q_m"], 1e-30) if free else None

    def ratios_ok(r):
        return all(value is not None and value <= 1.0 for value in r["nonlinear_convergence_ratios"])

    def linear_ok(r):
        return r["linear_iterations"] == 0 or all(
            r["stats"][name] is not None and r["stats"][name] <= limits["linear_tolerance"]
            for name in ("rho", "rho_q", "rho_x")
        )

    def contact_ok(r):
        block_sign_ok = all(
            r.get(name) is not None
            and math.isfinite(r[name])
            and 0.0 <= r[name] <= limits["force_balance_relative_tolerance"]
            for name in ("q_sign_projection_error", "x_sign_projection_error")
        )
        return block_sign_ok and all(
            r["stats"][name] is not None
            and math.isfinite(r["stats"][name])
            and 0.0 <= r["stats"][name] <= limits["force_balance_relative_tolerance"]
            for name in (
                "contact_force_imbalance",
                "contact_moment_imbalance",
                "generalized_projection_error",
                "contact_sign_error",
            )
        )

    numerical_gates = {
        "execution_completed": execution_error is None,
        "minimum_steps": count >= limits["minimum_substeps"],
        "finite_state": all(r["finite_state"] for r in records),
        "min_det_f": all(r["min_det_f"] is not None and r["min_det_f"] >= limits["min_det_f"] for r in records),
        "converged_ratio": converged_ratio >= limits["minimum_converged_ratio"],
        "consecutive_non_success": maximum_streak <= limits["maximum_consecutive_non_success"],
        "free_space_no_force": bool(free)
        and all(
            r["active_sample_count"] == 0 and abs(r["normal_compressive_force_n"]) <= limits["force_floor_n"]
            for r in free
        ),
        "free_space_completion": completion is not None and completion >= limits["free_space_completion_minimum"],
        "sustained_contact": onset is not None,
        "settle_penetration": bool(settle)
        and max(r["penetration_m"] for r in settle) <= limits["penetration_acceptance_limit_m"],
        "soft_response": bool(settle) and min(r["soft_response_m"] for r in settle) >= limits["delta_soft_min_m"],
        "converged_nonlinear_gates": all(ratios_ok(r) for r in records if r["converged"]),
        "converged_linear_gates": all(linear_ok(r) for r in records if r["converged"]),
        "contact_balance_and_sign": all(contact_ok(r) for r in records if not r["rolled_back"]),
    }
    evidence_gates = {
        "frozen_calibration": _axis_status(fixture, "calibration") == "FROZEN",
        "frozen_support": _axis_status(fixture, "support") == "FROZEN",
        "frozen_reference": _axis_status(fixture, "reference") == "FROZEN",
    }
    gates = {**numerical_gates, **evidence_gates}
    e2e_numerical_pass = all(numerical_gates.values())
    loading = records[onset:] if onset is not None else []
    force = np.asarray([r["normal_compressive_force_n"] for r in loading])
    closure = np.asarray([r["actual_closure_m"] for r in loading])
    low, high = limits["closure_secant_interval_m"]
    secant = None
    if closure.size and closure.max() >= high:
        interval = [next(i for i, value in enumerate(closure) if value >= bound) for bound in (low, high)]
        secant = float((force[interval[1]] - force[interval[0]]) / (closure[interval[1]] - closure[interval[0]]))
    return {
        "gates": gates,
        "e2e_numerical_pass": e2e_numerical_pass,
        "draft_numerical_pass": e2e_numerical_pass,
        "v01_exit": _v01_exit({"e2e_numerical_pass": e2e_numerical_pass, **evidence_gates}),
        "substeps": count,
        "converged_ratio": converged_ratio,
        "soft_stop_rate": sum(r["committed_unconverged"] for r in records) / max(count, 1),
        "rollback_rate": sum(r["rolled_back"] for r in records) / max(count, 1),
        "maximum_consecutive_non_success": maximum_streak,
        "contact_onset_step": onset,
        "free_space_completion": completion,
        "peak_force_n": float(force.max()) if force.size else None,
        "settled_force_n": float(np.mean([r["normal_compressive_force_n"] for r in settle])) if settle else None,
        "secant_stiffness_n_m": secant,
        "curve_work_j": float(np.trapezoid(force, closure)) if force.size > 1 else None,
    }


def _v01_exit(gates):
    """Require numerical acceptance and Newton-local evidence frozen for V0.1.

    The cross-implementation SuperDex reference remains reported as an
    independent evidence axis, but is deferred until after V0.2 and therefore
    does not block this exit decision.
    """
    required = ("e2e_numerical_pass", "frozen_calibration", "frozen_support")
    return all(gates.get(name) is True for name in required)


def _apply_candidate_internal_config(solver, requested):
    if not isinstance(requested, Mapping):
        raise ValueError("candidate_internal_config must be a mapping")
    config_fields = fields(solver._config)
    expected = {field.name for field in config_fields}
    supplied = set(requested)
    if supplied != expected:
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        raise ValueError(
            f"candidate_internal_config must specify every known field; missing={missing}, unknown={unknown}"
        )
    normalized = {}
    for field in config_fields:
        value = requested[field.name]
        if field.name == "regularization_values":
            if (
                not isinstance(value, (list, tuple))
                or not value
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                    for item in value
                )
            ):
                raise ValueError("regularization_values must be a nonempty finite numeric sequence")
            value = tuple(float(item) for item in value)
            if value[0] != 0.0 or any(left >= right for left, right in pairwise(value)):
                raise ValueError("regularization_values must start at zero and be strictly increasing")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            value = float(value)
            if value < 0.0 or (field.name != "merit_noise" and value == 0.0):
                raise ValueError(f"{field.name} must be positive; merit_noise may be zero")
        normalized[field.name] = value
    solver._config = replace(solver._config, **normalized)


def run_loading(fixture, *, device="cpu", substeps=None, candidate_internal_config=None):
    """Run the fixed input, preserving every unsuccessful substep in the evidence."""
    model, state, control, solver = build_scene(fixture, device=device)
    if candidate_internal_config is not None:
        _apply_candidate_internal_config(solver, candidate_internal_config)
    records = []
    execution_error = None
    dt = fixture["dt_s"]
    for step in range(fixture["substeps"] if substeps is None else substeps):
        time_s = (step + 1) * dt
        command = command_at(time_s, fixture)
        drive = fixture["drive"]
        force = drive["kp_n_m"] * (command[0] - float(state.joint_q.numpy()[0])) + drive["kd_n_s_m"] * (
            command[1] - float(state.joint_qd.numpy()[0])
        )
        control.joint_f.assign(np.asarray([force], dtype=np.float32))
        try:
            solver.step(state, state, control, None, dt)
            record = _measure(model, state, solver, fixture, step, time_s, command, float(np.float32(force)))
            record["active_sample_count"] = solver.last_stats.active_sample_count
            records.append(record)
        except (Exception, KeyboardInterrupt) as error:
            execution_error = {"step": step, "time_s": time_s, "reason": f"{type(error).__name__}: {error}"}
            break
    serialized = json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "status": "DRAFT",
        "record_schema": "normal_loading_measurements/v2",
        "calibration_status": _axis_status(fixture, "calibration"),
        "support_status": _axis_status(fixture, "support"),
        "reference_status": _axis_status(fixture, "reference"),
        "calibration_version": (_axis_provenance(fixture, "calibration") or {}).get("version"),
        "calibration_sha256": (_axis_provenance(fixture, "calibration") or {}).get("sha256"),
        "support_version": (_axis_provenance(fixture, "support") or {}).get("version"),
        "support_sha256": (_axis_provenance(fixture, "support") or {}).get("sha256"),
        "reference_version": (_axis_provenance(fixture, "reference") or {}).get("version"),
        "reference_sha256": (_axis_provenance(fixture, "reference") or {}).get("sha256"),
        "candidate_solver_internal_config": (_axis_provenance(fixture, "calibration") or {}).get(
            "candidate_private_config"
        ),
        "requested_solver_internal_config": _json_value(candidate_internal_config),
        "input_sha256": hashlib.sha256(serialized).hexdigest(),
        "actual_parameters": fixture,
        "execution_error": execution_error,
        "stored_model": {
            "particle_mass_kg": model.particle_mass.numpy().tolist(),
            "particle_rest_positions_m": model.particle_q.numpy().tolist(),
            "tet_indices": model.tet_indices.numpy().tolist(),
            "tet_materials": model.tet_materials.numpy().tolist(),
        },
        "newton_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        ),
        "superdex_sha": "54ae749a042709e897cad12da66822c71bcd1b97",
        "reference_executed": False,
        "device": str(model.device),
        "hardware": model.device.name,
        "platform": platform.platform(),
        "warp_version": wp.__version__,
        "solver_internal_config": _json_value(
            {field.name: getattr(solver._config, field.name) for field in fields(solver._config)}
        ),
        "normal_force_side": "physical force on rigid finger; compressive scalar is minus its projection on approach axis",
        "scope": "C5/C6/C7 measurement; status axes are reported independently",
    }
    return {
        "metadata": metadata,
        "records": records,
        "summary": assess_run(records, fixture, execution_error=execution_error),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate-internal-config",
        type=Path,
        help="Explicitly apply a complete solver-private config JSON object for this calibration run",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate_internal_config.read_text()) if args.candidate_internal_config else None
    result = run_loading(load_fixture(args.fixture), device=args.device, candidate_internal_config=candidate)
    for key in ("metadata", "summary"):
        (args.output / f"{key}.json").write_text(json.dumps(result[key], indent=2, allow_nan=False) + "\n")
    (args.output / "steps.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in result["records"]))
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["draft_numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
