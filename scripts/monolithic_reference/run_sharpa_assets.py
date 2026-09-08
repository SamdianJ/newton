# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a short CUDA hand/free-ball contact diagnostic, not a grasp gate."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_ball
from newton.examples.softbody.sharpa_close import load_fixture, sha256


def run(args):
    fixture_path = Path(__file__).with_name("fixtures") / "sharpa_assets_g5_v1.json"
    fixture = json.loads(fixture_path.read_text())
    dt = fixture["dt"]
    parameters = load_fixture("newton/examples/softbody/sharpa_g1h.json")
    model, manifest = load_hand_ball(
        args.asset_dir, args.contact_dir, args.ball, device=args.device, parameters=parameters, position=args.position
    )
    if not model.device.is_cuda:
        raise ValueError("This mesh contact diagnostic requires CUDA; CPU import has a separate test")
    model.request_contact_attributes("force")
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=fixture["detection_gap_m"])
    configs = [parameters["joints"][n] for n in manifest["joint_names"]]
    solver = SolverMonolithic(
        model,
        collision_pipeline=pipeline,
        contact_stiffness=fixture["normal_stiffness"],
        normal_smoothing_width=fixture["normal_smoothing_width"],
        friction_coefficient=fixture["friction_coefficient"],
        tangential_stiffness=fixture["tangential_stiffness"],
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True,
            limits=True,
            friction=True,
            limit_width=tuple(c["limit_width"] for c in configs),
            friction_velocity_scale=tuple(c["friction_velocity_scale"] for c in configs),
        ),
    )
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.assign(state.joint_q)
    control.joint_target_qd.zero_()
    velocity = np.tile(fixture["initial_ball_velocity_m_s"], (model.particle_count, 1)).astype(np.float32)
    state.particle_qd.assign(velocity)
    rest = state.particle_q.numpy().copy()
    weights = model.particle_mass.numpy().copy()
    centered_rest = rest - np.average(rest, axis=0, weights=weights)
    records = []
    for step in range(args.steps):
        start = time.perf_counter()
        solver.step(state, next_state, control, None, dt)
        wp.synchronize_device(model.device)
        elapsed = time.perf_counter() - start
        stats = solver.last_stats
        if not stats.rolled_back:
            state, next_state = next_state, state
        x = state.particle_q.numpy()
        centered = x - np.average(x, axis=0, weights=weights)
        u, _, vt = np.linalg.svd(centered_rest.T @ (weights[:, None] * centered))
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        deformation = centered - centered_rest @ rotation
        history = stats.contact_history
        records.append(
            {
                "step": step + 1,
                "time": (step + 1) * dt if not stats.rolled_back else step * dt,
                "converged": stats.converged,
                "rollback": stats.rolled_back,
                "status": stats.status.name,
                "reason": stats.failure_reason,
                "step_seconds": elapsed,
                "contacts": stats.active_sample_count,
                "penetration_m": stats.max_penetration,
                "min_det_f": stats.min_det_f,
                "deformation_rms_m": float(np.sqrt(np.average(np.sum(deformation**2, axis=1), weights=weights))),
                "convergence_ratio": stats.convergence_ratio,
                "convergence_ratio_q": stats.convergence_ratio_q,
                "convergence_ratio_x": stats.convergence_ratio_x,
                "linear_iterations": stats.linear_iterations,
                "normal_force_n": history["normal_force_sum"],
                "tangential_force_n": history["tangent_force_sum"],
                "relative_tangential_speed": history["relative_slip_max"],
            }
        )
        if stats.rolled_back:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    manifest.update(
        frozen_fixture_sha256=sha256(fixture_path),
        fixture=fixture,
        formal_fixture_match=args.steps == fixture["steps"] and list(args.position) == fixture["ball_position_m"],
        deformation_metric="mass-weighted RMS after proper rigid Kabsch fit",
        dt=dt,
        steps=args.steps,
        initial_particle_velocity_m_s=fixture["initial_ball_velocity_m_s"],
        q0=model.joint_q.numpy().tolist(),
        control="hold q0",
        normal_stiffness=fixture["normal_stiffness"],
        tangential_stiffness=fixture["tangential_stiffness"],
        friction_coefficient=fixture["friction_coefficient"],
        normal_smoothing_width=fixture["normal_smoothing_width"],
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_density=1000.0,
        scope="G5-assets short contact feasibility; NOT a grasp or G6/G7 acceptance",
        static_pairs=pipeline.soft_contact_pair_count,
        history_capacity=pipeline.soft_contact_max,
        factor_capacity=3 * pipeline.soft_contact_max,
        expected_triplet_capacity=22**2
        + 9 * (model.particle_count + 16 * model.tet_count)
        + 3 * pipeline.soft_contact_max * 31**2,
    )
    passed = len(records) == args.steps and all(r["converged"] and not r["rollback"] for r in records)
    passed = passed and max(r["penetration_m"] for r in records) <= fixture["gates"]["maximum_penetration_m"]
    passed = (
        passed
        and max(r["contacts"] for r in records) >= fixture["gates"]["minimum_peak_active_samples"]
        and min(r["min_det_f"] for r in records) > fixture["gates"]["minimum_det_f"]
    )
    passed = passed and all(
        r["convergence_ratio"] <= 1 and r["convergence_ratio_q"] <= 1 and r["convergence_ratio_x"] <= 1 for r in records
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "trace.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps({"passed": passed, "steps": len(records)}, indent=2) + "\n")
    np.savez(
        args.output / "state.npz", q=state.joint_q.numpy(), x=state.particle_q.numpy(), v=state.particle_qd.numpy()
    )
    if not passed:
        raise RuntimeError("Short contact gate failed; inspect trace, no thresholds were relaxed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--contact-dir", required=True)
    parser.add_argument("--ball", default="scripts/monolithic_reference/fixtures/soft_ball/ball_r1.npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--position", type=float, nargs=3, default=[0.025, -0.030, 0.190])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"error": type(error).__name__, "reason": str(error)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
