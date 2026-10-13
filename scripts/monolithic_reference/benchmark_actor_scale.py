# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure a loaded window of longer actor chains; not a frozen-scene acceptance gate."""

import argparse
import dataclasses
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import serial, state_arrays
from scripts.monolithic_reference.benchmark_workload import check_gpu_idle
from scripts.monolithic_reference.profiling import capture_count, contact_pair_diagnostics, provenance
from scripts.monolithic_reference.run_actor_coupling import build_model, load_case, make_solver, targets
from scripts.monolithic_reference.run_performance import COUNTERS


def build_chain(rigid_count, route, *, device="cuda:0"):
    """Keep material and control settings fixed while extending the rigid chain."""
    model = build_model(device=device, rigid_count=rigid_count)
    run = make_solver(
        model,
        graph=model.device.is_cuda,
        contact_dataflow=route,
        spatial={"candidate_capacity": 360 * rigid_count, "law_margin": 0.0} if route == "spatial" else None,
        contact_factor_capacity=1080 * rigid_count if route == "spatial" else None,
    )
    return model, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid-count", type=int, choices=(2, 4, 8, 16), required=True)
    parser.add_argument("--route", choices=("legacy", "spatial"), required=True)
    parser.add_argument("--prepare", action="store_true", help="Advance the original loading clock to step 1000")
    parser.add_argument("--checkpoint", type=Path, help="Prepared route-specific directory")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500 or args.prepare == (args.checkpoint is not None):
        parser.error("Choose --prepare or --checkpoint, with --steps in [1, 500]")
    args.output.mkdir(parents=True, exist_ok=False)
    source = provenance()
    check_gpu_idle(wp.get_device("cuda:0"))
    case = load_case()
    model, run = build_chain(args.rigid_count, args.route)
    state, control = run.create_state(dt=case["dt"]), model.control()
    arm = run._articulation_actor
    q, qd = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
    policy = serial(dataclasses.asdict(run._config))
    receipt = {
        "status": "RUNNING",
        "source": source,
        "route": args.route,
        "rigid_count": args.rigid_count,
        "case": case,
        "policy": policy,
        "full_candidates": run._contact.count,
        "actor_count": len(run._actors.actors),
        "scalar_dofs": run._actors.scalar_dof_count,
        "particles": model.particle_count,
        "tets": model.tet_count,
        "query_capacity": 360 * args.rigid_count if args.route == "spatial" else None,
        "factor_capacity": 1080 * args.rigid_count if args.route == "spatial" else None,
        "rows": [],
        "scope": "exploratory loaded-window scaling; not formal full-trajectory qualification",
    }

    def advance(step, *, timed=False):
        position, velocity, _ = targets(case, step)
        q[arm.target_indices[0]], qd[arm.dof_indices[0]] = position, velocity
        control.joint_target_q.assign(q)
        control.joint_target_qd.assign(qd)
        wp.synchronize_device(model.device)
        captures = capture_count(run)
        plan = run._linear._masked_pcg_plan
        start = time.perf_counter()
        run.step(state, state, control, dt=case["dt"])
        wp.synchronize_device(model.device)
        elapsed = time.perf_counter() - start
        stats = run.last_stats
        if not stats.converged or stats.rolled_back:
            raise RuntimeError(f"Step {step}: {stats}")
        for key in ("contact_pool", "contact_spatial"):
            d = stats.variational.get(key, {})
            if d.get("tainted") or d.get("dropped", 0) or d.get("capacity_dropped", 0):
                raise RuntimeError(f"Step {step}: {key} capacity exceeded")
        if timed:
            receipt["rows"].append(
                {
                    "step": step,
                    "seconds": elapsed,
                    "capture_affected": captures != capture_count(run) or plan is not run._linear._masked_pcg_plan,
                    **{key: getattr(stats, key) for key in COUNTERS},
                }
            )

    try:
        with contact_pair_diagnostics(run, enabled=False):
            if args.prepare:
                for step in range(1, 1001):
                    advance(step)
                    if step % 100 == 0:
                        check_gpu_idle(model.device)
                        print(json.dumps({"prepared_step": step}), flush=True)
                run.save_checkpoint(state, args.output / "loaded.npz")
                receipt["checkpoint_sha256"] = hashlib.sha256((args.output / "loaded.npz").read_bytes()).hexdigest()
                receipt["status"] = "PREPARED_1000_CONVERGED"
            else:
                saved = json.loads((args.checkpoint / "receipt.json").read_text())
                checkpoint = args.checkpoint / "loaded.npz"
                assert saved["status"] == "PREPARED_1000_CONVERGED"
                for key in (
                    "route",
                    "rigid_count",
                    "case",
                    "policy",
                    "full_candidates",
                    "query_capacity",
                    "factor_capacity",
                ):
                    if saved[key] != receipt[key]:
                        raise ValueError(f"Checkpoint identity differs: {key}")
                assert saved["source"]["source_hashes"] == source["source_hashes"]
                assert saved["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                receipt["checkpoint_sha256"] = saved["checkpoint_sha256"]
                run.load_checkpoint(state, checkpoint)
                expected = state_arrays(state)
                for step in range(1001, 1021):
                    advance(step)
                run.load_checkpoint(state, checkpoint)
                for key, value in state_arrays(state).items():
                    np.testing.assert_array_equal(value, expected[key], err_msg=key)
                check_gpu_idle(model.device)
                for step in range(1001, 1001 + args.steps):
                    advance(step, timed=True)
                    if step % 50 == 0:
                        check_gpu_idle(model.device)
                times = [r["seconds"] for r in receipt["rows"]]
                receipt.update(
                    status="COMPLETE_WINDOW",
                    solve_seconds=sum(times),
                    mean_ms=float(np.mean(times)) * 1000,
                    p95_ms=float(np.percentile(times, 95)) * 1000,
                )
            arrays = state_arrays(state)
            assert all(np.isfinite(a).all() for a in arrays.values())
            for history, field in (
                ("bound_q", "joint_q"),
                ("bound_qd", "joint_qd"),
                ("bound_x", "particle_q"),
                ("bound_v", "particle_qd"),
            ):
                np.testing.assert_array_equal(arrays["history." + history], arrays[field])
            assert arrays["history.force_valid"][0] == 1
            np.savez_compressed(args.output / "final-state.npz", **arrays)
            receipt["contact_pool"] = serial(run.last_stats.variational.get("contact_pool"))
            receipt["contact_spatial"] = serial(run.last_stats.variational.get("contact_spatial"))
            receipt["totals"] = dict(sum((Counter({k: r[k] for k in COUNTERS}) for r in receipt["rows"]), Counter()))
            receipt["final_state_checks"] = "FINITE_HISTORY_FORCE_PASS"
            check_gpu_idle(model.device)
            assert provenance()["source_hashes"] == source["source_hashes"]
    except Exception as error:
        receipt.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps({k: receipt.get(k) for k in ("status", "rigid_count", "route", "mean_ms", "error")}), flush=True
        )


if __name__ == "__main__":
    main()
