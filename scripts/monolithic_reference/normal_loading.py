# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run and measure a DRAFT normal-loading trajectory; never certify P0/V0.1 exit.

Use ``uv run -m scripts.monolithic_reference.normal_loading --device cpu
--output /path/to/evidence``. Raw records use a separate draft measurement
format, not the frozen cross-implementation step-record schema.
"""

import argparse
import enum
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton

DEFAULT_FIXTURE = Path(__file__).with_name("fixtures") / "normal_loading_draft_v1.json"


def load_fixture(path=DEFAULT_FIXTURE):
    """Read the explicitly unfrozen normal-loading input."""
    fixture = json.loads(Path(path).read_text())
    if fixture["status"] != "DRAFT" or fixture["schema_version"] != "normal_loading_draft/v1":
        raise ValueError("This runner only accepts the normal_loading_draft/v1 DRAFT fixture")
    return fixture


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


def assess_run(records, fixture):
    """Apply the declared provisional gates and expose missing freeze/reference evidence."""
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
        return all(
            r["stats"][name] is not None and r["stats"][name] <= limits["force_balance_relative_tolerance"]
            for name in (
                "contact_force_imbalance",
                "contact_moment_imbalance",
                "generalized_projection_error",
                "contact_sign_error",
            )
        )

    gates = {
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
        "frozen_calibration": fixture["calibration_status"] == "FROZEN",
        "reference_envelope": fixture["reference_envelope"] is not None,
    }
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
        "draft_numerical_pass": all(
            value for key, value in gates.items() if key not in ("frozen_calibration", "reference_envelope")
        ),
        "v01_exit": False,
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


def run_loading(fixture, *, device="cpu", substeps=None):
    """Run the fixed input, preserving every unsuccessful substep in the evidence."""
    model, state, control, solver = build_scene(fixture, device=device)
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
        except RuntimeError as error:
            execution_error = {"step": step, "time_s": time_s, "reason": str(error)}
            break
        record = _measure(model, state, solver, fixture, step, time_s, command, float(np.float32(force)))
        record["active_sample_count"] = solver.last_stats.active_sample_count
        records.append(record)
    serialized = json.dumps(fixture, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "status": "DRAFT",
        "record_schema": "normal_loading_measurements_draft/v1",
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
        "solver_internal_config": {field.name: getattr(solver._config, field.name) for field in fields(solver._config)},
        "normal_force_side": "physical force on rigid finger; compressive scalar is minus its projection on approach axis",
        "scope": "Draft C5/C6/C7 precheck; no frozen comparison manifest or reference envelope",
    }
    return {"metadata": metadata, "records": records, "summary": assess_run(records, fixture)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_loading(load_fixture(args.fixture), device=args.device)
    for key in ("metadata", "summary"):
        (args.output / f"{key}.json").write_text(json.dumps(result[key], indent=2, allow_nan=False) + "\n")
    (args.output / "steps.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in result["records"]))
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["draft_numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
