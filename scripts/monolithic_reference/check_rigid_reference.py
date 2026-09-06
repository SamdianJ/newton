# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure rigid-only BE/Featherstone convergence and isolated R/f/Q signs.

The required tet has no collision shapes and falls freely, completely
decoupled from the single revolute rigid DOF. This is an
internal release check, not a SuperDex or integrator-equality assertion.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
    project_articulation_body_wrenches,
)
from newton.solvers import SolverFeatherstone


def _scene(device):
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    body = builder.add_link(mass=2.0, com=(0.5, 0.0, 0.0), inertia=wp.diag(wp.vec3(0.1)))
    joint = builder.add_joint_revolute(
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
    builder.joint_q[0], builder.joint_qd[0] = 0.125, 2.0
    builder.add_soft_mesh(
        pos=(2.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.02)],
        indices=[0, 1, 2, 3],
        density=1000.0,
        k_mu=1000.0,
        k_lambda=1000.0,
        k_damp=0.0,
        particle_radius=0.001,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def _scalar_be(dt):
    # Moment balance at the pivot: (I_com + m*l*l) * a = tau + l*Fy*cos(q).
    # Gravity Fy=-19.62 N, applied Fy=0.4 N, applied tau_z=0.2 N*m,
    # and direct generalized joint torque=0.7 N*m. No production code is used.
    left, right = -1.0, 1.0
    for _ in range(80):
        q = (left + right) / 2.0
        residual = 0.6 * (q - 0.125 - dt * 2.0) / dt**2 - (0.9 - 9.61 * math.cos(q))
        if residual > 0.0:
            right = q
        else:
            left = q
    q = (left + right) / 2.0
    return q, (q - 0.125) / dt


def _isolated(model, state, device):
    records = []
    state.joint_qd.zero_()
    control = model.control()
    qdd = wp.zeros(1, dtype=float, device=device)
    residual = wp.zeros(1, dtype=float, device=device)
    workspace = MonolithicArticulationWorkspace(model)
    for term in ("gravity", "joint_f", "body_f"):
        model.gravity.assign(np.asarray([[0.0, -9.81 if term == "gravity" else 0.0, 0.0]], dtype=np.float32))
        control.joint_f.fill_(0.7 if term == "joint_f" else 0.0)
        wrench = np.zeros((1, 6), dtype=np.float32)
        if term == "body_f":
            wrench[0, 1], wrench[0, 5] = 0.4, 0.2
        state.body_f.assign(wrench)
        eval_articulation_passive_candidate(model, state, workspace)
        wp.launch(
            project_articulation_body_wrenches,
            1,
            inputs=[
                0,
                1,
                model.articulation_start,
                model.articulation_end,
                model.joint_child,
                workspace.scratch.J,
                state.body_f,
                workspace.generalized_body_force,
            ],
            device=device,
        )
        wp.launch(
            eval_articulation_actor_residual,
            1,
            inputs=[
                0,
                0,
                1,
                workspace.M,
                qdd,
                workspace.C,
                workspace.g,
                control.joint_f,
                workspace.generalized_body_force,
                residual,
            ],
            device=device,
        )
        physical = wrench.copy()
        if term == "gravity":
            physical[0, 1] = -19.62
        analytical_j = np.asarray([-0.5 * math.sin(0.125), 0.5 * math.cos(0.125), 0.0, 0.0, 0.0, 1.0])
        expected_q = 0.7 if term == "joint_f" else float(analytical_j @ physical[0])
        actual_q = float(
            -workspace.g.numpy()[0]
            if term == "gravity"
            else control.joint_f.numpy()[0]
            if term == "joint_f"
            else workspace.generalized_body_force.numpy()[0]
        )
        records.append(
            {
                "term": term,
                "residual_contribution": residual.numpy().tolist(),
                "physical_world_force_or_wrench": None if term == "joint_f" else physical.tolist(),
                "physical_world_force_or_wrench_reason": "Direct generalized joint torque; world wrench not unique"
                if term == "joint_f"
                else None,
                "generalized_physical_force": [float(actual_q)],
                "independent_generalized_force": [expected_q],
                "q_sign_error": float(abs(residual.numpy()[0] + actual_q) / max(abs(actual_q), 1e-12)),
                "q_projection_error": float(abs(actual_q - expected_q) / abs(expected_q)),
                "x_sign_error": None,
                "x_sign_error_reason": "No rigid-soft coupling; these isolated articulation terms have no x contribution",
            }
        )
    return records


def measure(device):
    """Run actual one-step solvers with identical initial state and frozen forces."""
    steps = []
    for dt in (0.02, 0.01, 0.005):
        model, state = _scene(device)
        control = model.control()
        control.joint_f.fill_(0.7)
        state.body_f.assign(np.asarray([[0.0, 0.4, 0.0, 0.0, 0.0, 0.2]], dtype=np.float32))
        initial_q, initial_qd = state.joint_q.numpy(), state.joint_qd.numpy()
        pipeline = MonolithicCollisionPipeline(model)
        monolithic = SolverMonolithic(model, collision_pipeline=pipeline, contact_stiffness=1.0e7)
        featherstone = SolverFeatherstone(model, angular_damping=0.0)
        mono_out, feather_out = model.state(), model.state()
        monolithic.step(state, mono_out, control, None, dt)
        featherstone.step(state, feather_out, control, None, dt)
        q_be, qd_be = _scalar_be(dt)
        stats = monolithic.last_stats
        steps.append(
            {
                "dt_s": dt,
                "joint_q_initial": initial_q.tolist(),
                "joint_qd_initial": initial_qd.tolist(),
                "frozen_joint_f": control.joint_f.numpy().tolist(),
                "frozen_body_f": state.body_f.numpy().tolist(),
                "gravity": model.gravity.numpy().tolist(),
                "joint_q_scalar_be": [q_be],
                "joint_qd_scalar_be": [qd_be],
                "joint_q_monolithic": mono_out.joint_q.numpy().tolist(),
                "joint_qd_monolithic": mono_out.joint_qd.numpy().tolist(),
                "joint_q_featherstone": feather_out.joint_q.numpy().tolist(),
                "joint_qd_featherstone": feather_out.joint_qd.numpy().tolist(),
                "body_q_monolithic": mono_out.body_q.numpy().tolist(),
                "body_q_featherstone": feather_out.body_q.numpy().tolist(),
                "body_qd_monolithic": mono_out.body_qd.numpy().tolist(),
                "body_qd_featherstone": feather_out.body_qd.numpy().tolist(),
                "status": stats.status.name,
                "converged": stats.converged,
                "rolled_back": stats.rolled_back,
                "active_sample_count": stats.active_sample_count,
                "q_dof_count": monolithic._layout.q_dof_count,
                "x_dof_count": 3 * monolithic._layout.dynamic_particle_count,
                "soft_free_fall_error_m": float(
                    np.max(
                        np.abs(
                            mono_out.particle_q.numpy()
                            - state.particle_q.numpy()
                            - np.asarray([0.0, -9.81 * dt**2, 0.0])
                        )
                    )
                ),
                "convergence_ratio": stats.convergence_ratio,
                "convergence_ratio_q": stats.convergence_ratio_q,
                "rho": stats.rho,
            }
        )
    model, state = _scene(device)
    return {
        "device": device,
        "fixture": "revolute_offset_com_decoupled_tet_no_shapes_v1",
        "steps": steps,
        "isolated_forces": _isolated(model, state, device),
    }


def _matches(values, expected, *, atol=1e-6):
    actual, target = np.asarray(values), np.asarray(expected)
    return (
        actual.shape == target.shape and np.isfinite(actual).all() and np.allclose(actual, target, rtol=1e-6, atol=atol)
    )


def assess(result):
    """Check recorded states and forces against the fixed scene's analytical physics."""
    steps = result["steps"]
    terms = result["isolated_forces"]
    if len(steps) != 3 or len(terms) != 3 or {row["term"] for row in terms} != {"gravity", "joint_f", "body_f"}:
        return False
    errors, position_errors, transform_errors = [], [], []
    for row, dt in zip(steps, (0.02, 0.01, 0.005), strict=True):
        if (
            row["dt_s"] != dt
            or not row["converged"]
            or row["rolled_back"]
            or row["status"] != "SUCCESS"
            or row["active_sample_count"] != 0
            or row["q_dof_count"] != 1
            or row["x_dof_count"] != 12
            or not 0.0 <= row["soft_free_fall_error_m"] <= 2e-6
            or not _matches(row["joint_q_initial"], [0.125])
            or not _matches(row["joint_qd_initial"], [2.0])
            or not _matches(row["frozen_joint_f"], [0.7])
            or not _matches(row["frozen_body_f"], [[0.0, 0.4, 0.0, 0.0, 0.0, 0.2]])
            or not _matches(row["gravity"], [[0.0, -9.81, 0.0]])
        ):
            return False
        q_be, v_be = _scalar_be(dt)
        if not _matches(row["joint_q_scalar_be"], [q_be], atol=1e-12) or not _matches(
            row["joint_qd_scalar_be"], [v_be], atol=1e-12
        ):
            return False
        for solver in ("monolithic", "featherstone"):
            q, v = np.asarray(row[f"joint_q_{solver}"]), np.asarray(row[f"joint_qd_{solver}"])
            if q.shape != (1,) or v.shape != (1,) or not np.isfinite(q).all() or not np.isfinite(v).all():
                return False
            angle, speed = float(q[0]), float(v[0])
            pose = [[0.0, 0.0, 0.0, 0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]]
            twist = [[-0.5 * math.sin(angle) * speed, 0.5 * math.cos(angle) * speed, 0.0, 0.0, 0.0, speed]]
            if not _matches(row[f"body_q_{solver}"], pose) or not _matches(row[f"body_qd_{solver}"], twist):
                return False
        q, v = np.asarray(row["joint_q_monolithic"]), np.asarray(row["joint_qd_monolithic"])
        if np.max(np.abs(q - q_be)) > 2e-7 or np.max(np.abs(v - v_be)) > 4e-5:
            return False
        errors.append(float(np.linalg.norm(v - row["joint_qd_featherstone"])))
        position_errors.append(float(np.linalg.norm(q - row["joint_q_featherstone"])))
        transform_errors.append(
            float(np.linalg.norm(np.asarray(row["body_q_monolithic"]) - row["body_q_featherstone"]))
        )
        if not (0.0 <= row["convergence_ratio"] <= 1.0 and 0.0 <= row["convergence_ratio_q"] <= 1.0):
            return False
    if not (errors[1] < 0.4 * errors[0] and errors[2] < 0.4 * errors[1] and errors[-1] < 2e-4):
        return False
    for differences in (position_errors, transform_errors):
        if not (differences[1] < 0.25 * differences[0] and differences[2] < 0.25 * differences[1]):
            return False
    expected_wrenches = {
        "gravity": [[0.0, -19.62, 0.0, 0.0, 0.0, 0.0]],
        "body_f": [[0.0, 0.4, 0.0, 0.0, 0.0, 0.2]],
    }
    jacobian = np.asarray([-0.5 * math.sin(0.125), 0.5 * math.cos(0.125), 0.0, 0.0, 0.0, 1.0])
    for row in terms:
        term = row["term"]
        if term == "joint_f":
            if row["physical_world_force_or_wrench"] is not None:
                return False
            expected = 0.7
        else:
            physical = expected_wrenches[term]
            if not _matches(row["physical_world_force_or_wrench"], physical):
                return False
            expected = float(jacobian @ physical[0])
        # The fixed fixture has nonzero forces; absolute comparison cannot accept 0/0 NaN ratios.
        for key, value in (
            ("residual_contribution", -expected),
            ("generalized_physical_force", expected),
            ("independent_generalized_force", expected),
        ):
            if not _matches(row[key], [value]):
                return False
        residual = float(row["residual_contribution"][0])
        generalized = float(row["generalized_physical_force"][0])
        if abs(residual + generalized) / abs(expected) > 1e-6 or abs(generalized - expected) / abs(expected) > 1e-6:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.device)
    result["passed"] = assess(result)
    result["source_clean"] = not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    result["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
