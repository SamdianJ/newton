# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay the rotating pinch failure and record its nonlinear iterates."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import iteration
from newton._src.solvers.monolithic.producers import RigidProducer
from newton.examples.softbody.monolithic_recording import save

from .run_dexterous_tasks import DT, REPO, TaskScene


def main():
    """Resume the pinch task with optional diagnostic-only algorithm changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--checkpoint-iterations", type=int, default=400)
    parser.add_argument("--wrist", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rotation-predictor", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=646)
    parser.add_argument("--trace-step", type=int, default=646)
    parser.add_argument("--joint-descent", action="store_true")
    parser.add_argument(
        "--inspect-chunks",
        action="store_true",
        help="Read failing contact chunks with the contact batch CUDA graph disabled",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.iterations <= 0:
        parser.error("steps and iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    scene = TaskScene(
        "pinch-transfer",
        device=args.device,
        newton_iterations=args.checkpoint_iterations,
        wrist=args.wrist,
        alignment_policy="mask" if args.wrist else "guard",
    )
    solver = scene.solver
    solver.load_checkpoint(scene.state, args.checkpoint)
    next_step = int(scene.state._monolithic_coupled.force_generation.numpy()[0]) + 1
    if args.start_step != next_step:
        parser.error(f"checkpoint requires --start-step {next_step}")
    solver.newton_max_iterations = args.iterations
    trace, snapshots = [], {}
    measure = iteration.measure
    predictor = RigidProducer.begin_step
    acceptance = iteration.acceptance
    probes = {}
    batch = solver._contact._batch
    stage = batch._stage
    if args.inspect_chunks:
        batch.graphs.enabled = False

    def observed_stage(*values, **options):
        stage(*values, **options)
        if args.inspect_chunks and physical_step == args.trace_step and batch._flags.numpy()[0]:
            snapshot(solver, "bad_chunk")

    def snapshot(owner, prefix, *, tangent=False):
        c = owner._contact
        batch = c._batch
        n = int(c.pool.candidates.count.numpy()[0])
        data = {
            "candidate_ids": c.pool.candidates.identities.numpy()[:n],
            "flags": batch.flags.numpy(),
            "start_flags": batch.start_flags.numpy(),
            "chunk_flags": batch._flags.numpy(),
            "pool_flags": c.pool.records.flags.numpy(),
        }
        selection = c._spatial
        count = int(selection.count.numpy()[0])
        data.update(spatial_ids=selection.identities.numpy()[:count], spatial_status=selection.status.numpy())
        if tangent:
            data.update(
                r=owner._complete_r.numpy(), B=owner._linear.densify_for_test(generation=owner._generation).raw_matrix
            )
        for name in ("y", "slip", "gradient", "distance", "soft_normal", "rotation"):
            data[name] = getattr(batch.current, name).numpy()[:n]
        for name in ("y", "gradient", "distance", "soft_normal"):
            data[f"start_{name}"] = getattr(batch._start_chunk, name).numpy()[:n]
        for name in ("sources", "targets", "area", "gaps"):
            data[name] = getattr(batch.samples, name).numpy()[:n]
        for name in ("weights", "local_r", "points"):
            data[name] = getattr(batch, name).numpy()[:n]
        snapshots.update({f"{prefix}.{name}": value for name, value in data.items()})
        return data

    def both_descent(*values, **options):
        accepted, energy, residual, improved = acceptance(*values, **options)
        if args.joint_descent:
            accepted = energy and residual
        return accepted, energy, residual, accepted and improved

    def predict_rotation(producer, owner, state_in):
        predictor(producer, owner, state_in)
        if not args.rotation_predictor:
            # Retain the PR-03 no-angular-prediction reproducer after the production fix.
            wp.copy(owner._accepted._monolithic_coupled.rigid_rotation, owner._rigid.quat_n)
            owner._rigid_steps(owner._accepted)[1].fill_(wp.quat_identity(owner._dtype))

    def probe(owner, r, b):
        original = owner._delta.numpy()
        tau = owner._criterion.tau.numpy()
        e0 = owner._evaluate(owner._accepted, DT, build_tangent=False)[0]
        jacobians, energy_gradients = [], []
        for translation_h in (1e-8, 1e-9, 1e-10):
            j = np.zeros_like(b)
            eg = np.zeros(len(r))
            for column in range(len(r)):
                h = translation_h * (100 if column >= len(r) - 3 else 1)
                values = []
                for sign in (-1, 1):
                    d = np.zeros(len(r))
                    d[column] = sign * h
                    owner._delta.assign(d)
                    owner._take_trial(1.0, DT)
                    e, _, _ = owner._evaluate(owner._trial, DT, build_tangent=False)
                    values.append((e, owner._complete_r.numpy().copy()))
                j[:, column] = (values[1][1] - values[0][1]) / (2 * h)
                eg[column] = (values[1][0] - values[0][0]) / (2 * h)
            jacobians.append(j)
            energy_gradients.append(eg)
        direction = np.linalg.solve(b, -r)
        owner._delta.assign(direction)
        profile = []
        for alpha in 2.0 ** -np.arange(13):
            owner._take_trial(float(alpha), DT)
            e, _, _ = owner._evaluate(owner._trial, DT, build_tangent=False)
            rr = owner._complete_r.numpy().copy()
            profile.append(
                {
                    "alpha": alpha,
                    "energy_change": e - e0,
                    "merit": np.linalg.norm(rr / tau) / np.sqrt(len(r)),
                    "residual": rr,
                }
            )
        probes.update(
            J=np.array(jacobians),
            energy_gradients=np.array(energy_gradients),
            B=b,
            r=r,
            direction=direction,
            tau=tau,
            profile=profile,
        )
        owner._delta.assign(original)
        owner._evaluate(owner._accepted, DT)
        owner._assemble(0)

    def observed(owner, *values):
        assessment = measure(owner, *values)
        if physical_step != args.trace_step:
            return assessment
        i = len(trace)
        r = owner._complete_r.numpy()
        c = owner._contact
        records = c.pool.records
        count = int(records.count.numpy()[0])
        ids = records.identities.numpy()[:count]
        trace.append(
            {
                "iteration": i,
                "merit": float(assessment.merit),
                "residual": r,
                "tau": owner._criterion.tau.numpy(),
                "active_contacts": count,
                "contact_ids": ids,
                "contact_energy": c.energy.numpy(),
                "joint_q": owner._accepted.joint_q.numpy(),
                "rigid_com": owner._accepted._monolithic_coupled.rigid_com.numpy(),
                "rigid_rotation": owner._accepted._monolithic_coupled.rigid_rotation.numpy(),
            }
        )
        if i in (0, 1, 2, 3, 4, 5, 10, 20, 40, 79, 80):
            data = snapshot(owner, f"i{i}", tangent=True)
            if i == 0 and args.probe:
                probe(owner, r, data["B"])
        return assessment

    steps = []
    with (
        patch.object(iteration, "measure", observed),
        patch.object(RigidProducer, "begin_step", predict_rotation),
        patch.object(iteration, "acceptance", both_descent),
        patch.object(batch, "_stage", observed_stage),
    ):
        for physical_step in range(args.start_step, args.start_step + args.steps):
            scene.step(physical_step * DT)
            stats = solver.last_stats
            steps.append(
                {
                    "step": physical_step,
                    "status": stats.status.value,
                    "newton": stats.nonlinear_iterations,
                    "converged": stats.converged,
                    "rolled_back": stats.rolled_back,
                }
            )
            if not stats.converged:
                snapshot(solver, "failure")
                break
            if physical_step % 200 == 0:
                print(physical_step, stats.status.value, stats.nonlinear_iterations, flush=True)
    np.savez_compressed(args.output / "iterates.npz", **snapshots)
    save(args.output / "trace.json", trace)
    save(args.output / "probe.json", probes)
    save(args.output / "steps.json", steps)
    save(args.output / "stats.json", solver.last_stats)
    solver.save_checkpoint(scene.state, args.output / "last-published-state.npz")
    save(
        args.output / "manifest.json",
        {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "source_root": str(REPO),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scope": "diagnostic monkeypatches only; production solver unchanged",
            "final_rigid_com": scene.state._monolithic_coupled.rigid_com.numpy(),
            "final_rigid_rotation": scene.state._monolithic_coupled.rigid_rotation.numpy(),
        },
    )
    print(solver.last_stats.status, "iterations", solver.last_stats.nonlinear_iterations, flush=True)
    for row in trace[:: max(1, len(trace) // 10)]:
        print(
            row["iteration"],
            "merit",
            row["merit"],
            "r",
            np.linalg.norm(row["residual"]),
            "contacts",
            row["active_contacts"],
            flush=True,
        )


if __name__ == "__main__":
    main()
