# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the declared PR-7A force/penetration envelope before grasp integration."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
from newton.examples.softbody.monolithic_soft_ball import load_ball
from newton.examples.softbody.sharpa_close import sha256

CONFIG = {
    "schema": "sharpa-contact-calibration/v1",
    "stiffness_scan_n_m3": [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8],
    "refinements": [1, 2, 3],
    "dt": 0.001,
    "ramp_seconds": 0.5,
    "hold_seconds": 1.0,
    "measurement_seconds": 0.2,
    "gravity": [0, 0, 0],
    "load_definition": "per-jaw load = 2 * ball_mass * 9.81 / (2 * mu); gravity represented by equivalent clamp load",
    "mu": 0.5,
    "tangential_stiffness": 2e6,
    "smoothing_width": 0.0001,
    "gap": 0.002,
    "particle_radius": 0.0002,
    "plate_mass": 0.01,
    "plate_velocity_gain": 10.0,
    "pcg_iteration_p95_budget": 64,
    "gates": {"penetration_m": 0.002, "min_det_f": 0.1, "converged_fraction": 0.99, "force_error": 0.05},
}


def make_case(ball, stiffness, device):
    """Clamp a fully dynamic sphere between fixed and damped force-driven planes."""
    mesh = load_ball(ball)
    b = newton.ModelBuilder(gravity=CONFIG["gravity"])
    base = b.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    plate = b.add_link(mass=CONFIG["plate_mass"], inertia=wp.diag(wp.vec3(1e-5)))
    fixed = b.add_joint_fixed(-1, base)
    drive = b.add_joint_prismatic(
        base,
        plate,
        axis=newton.Axis.Z,
        target_ke=0,
        target_kd=CONFIG["plate_velocity_gain"],
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        armature=0,
        damping=0,
        friction=0,
        limit_ke=0,
        limit_kd=0,
        effort_limit=5.0,
    )
    b.add_articulation([fixed, drive])
    radius = 0.02 + CONFIG["particle_radius"]
    cfg = b.ShapeConfig(margin=0.0)
    b.add_shape_plane(
        body=base, width=0.0, length=0.0, xform=wp.transform((0, 0, -radius), wp.quat_identity()), cfg=cfg
    )
    b.add_shape_plane(
        body=plate, width=0.0, length=0.0, xform=wp.transform((0, 0, radius), wp.quat(1.0, 0.0, 0.0, 0.0)), cfg=cfg
    )
    b.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        mesh=mesh,
        particle_radius=CONFIG["particle_radius"],
        add_surface_mesh_edges=False,
    )
    model = b.finalize(device=device)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=CONFIG["gap"])
    solver = SolverMonolithic(
        model,
        collision_pipeline=pipeline,
        contact_stiffness=stiffness,
        normal_smoothing_width=CONFIG["smoothing_width"],
        friction_coefficient=CONFIG["mu"],
        tangential_stiffness=CONFIG["tangential_stiffness"],
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, device=device),
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
    )
    state, control = model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    load = float(model.particle_mass.numpy().sum()) * 9.81 / CONFIG["mu"]
    return model, solver, state, control, load


def run_case(ball, stiffness, device, output):
    """Run a fixed schedule without adaptive stepping or acceptance changes."""
    _model, solver, state, control, load = make_case(ball, stiffness, device)
    rows = []
    steps = round((CONFIG["ramp_seconds"] + CONFIG["hold_seconds"]) / CONFIG["dt"])
    failure = None
    for step in range(steps):
        t = (step + 1) * CONFIG["dt"]
        u = min(t / CONFIG["ramp_seconds"], 1)
        target_load = load * u * u * (3 - 2 * u)
        control.joint_target_qd.assign([-target_load / CONFIG["plate_velocity_gain"]])
        try:
            solver.step(state, state, control, None, CONFIG["dt"])
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            break
        stats = solver.last_stats
        # Final forces are COM wrenches on rigid bodies, from the same publication as residuals.
        force = solver._contact.final_force_linear.numpy()
        shapes = solver.contacts.soft_contact_shape.numpy()
        count = int(solver.contacts.soft_contact_count.numpy()[0])
        reactions = [float(np.sum(force[:count][shapes[:count] == shape, 2])) for shape in (0, 1)]
        rows.append(
            {
                "time": step * CONFIG["dt"] if stats.rolled_back else t,
                "converged": stats.converged,
                "rollback": stats.rolled_back,
                "status": stats.status.name,
                "reason": stats.failure_reason,
                "penetration": stats.max_penetration,
                "min_det_f": stats.min_det_f,
                "residual_ratio": stats.convergence_ratio,
                "q_residual_ratio": stats.convergence_ratio_q,
                "x_residual_ratio": stats.convergence_ratio_x,
                "reactions": reactions,
                "load": target_load,
                "pd_force": stats.joint_pd_force,
                "linear_iterations": stats.linear_iterations,
            }
        )
        if stats.rolled_back:
            break
    tail = rows[-round(CONFIG["measurement_seconds"] / CONFIG["dt"]) :]
    reaction = np.mean([r["reactions"] for r in tail], axis=0) if tail else np.full(2, np.nan)
    force_error = float(np.max(np.abs(np.abs(reaction) - load)) / load)
    gates = CONFIG["gates"]
    passed = failure is None and len(rows) == steps and not any(r["rollback"] for r in rows)
    passed = passed and np.mean([r["converged"] for r in rows]) >= gates["converged_fraction"]
    passed = passed and max(r["penetration"] for r in rows) <= gates["penetration_m"]
    passed = passed and min(r["min_det_f"] for r in rows) >= gates["min_det_f"] and force_error <= gates["force_error"]
    passed = passed and all(
        max(r["residual_ratio"], r["q_residual_ratio"], r["x_residual_ratio"]) <= 1 for r in rows if r["converged"]
    )
    summary = {
        "passed": bool(passed),
        "ball_sha256": sha256(ball),
        "stiffness": stiffness,
        "load_n": load,
        "force_error": force_error,
        "completed_steps": len(rows),
        "exception": failure,
        "maximum_penetration": max((r["penetration"] for r in rows), default=None),
        "minimum_det_f": min((r["min_det_f"] for r in rows), default=None),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    """Save the protocol before running every predeclared stiffness/grid point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinements", type=int, nargs="+", default=CONFIG["refinements"])
    parser.add_argument("--stiffness", type=float, nargs="+", default=CONFIG["stiffness_scan_n_m3"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    summaries = []
    for stiffness in args.stiffness:
        for refinement in args.refinements:
            ball = Path(__file__).with_name("fixtures") / "soft_ball" / f"ball_r{refinement}.npz"
            summary = run_case(ball, stiffness, args.device, args.output / f"k{stiffness:g}-r{refinement}")
            summaries.append({"refinement": refinement, **summary})
    full = args.stiffness == CONFIG["stiffness_scan_n_m3"] and args.refinements == CONFIG["refinements"]
    passing = [
        k
        for k in CONFIG["stiffness_scan_n_m3"]
        if all(
            any(s["refinement"] == r and s["stiffness"] == k and s["passed"] for s in summaries)
            for r in CONFIG["refinements"]
        )
    ]
    minimum = min(passing) if full and passing else None
    result = {
        "status": "FROZEN" if minimum is not None else "INCOMPLETE_OR_FAILED",
        "protocol": CONFIG,
        "protocol_sha256": sha256(args.output / "protocol.json"),
        "script_sha256": sha256(__file__),
        "device": str(wp.get_device(args.device)),
        "stiffness_floor_n_m3": minimum,
        "initial_grasp_stiffness_n_m3": max(1e7, minimum) if minimum else None,
        "pcg_iteration_p95_budget": 64,
        "results": summaries,
    }
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
