# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure cantilever mesh convergence and synchronized, headless computation cost."""

import argparse
import gc
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_tet_response import ResponseCase


def performance(samples, dt):
    """Separate solver work from diagnostic transfers and NumPy measurements."""
    result = {"steps": len(samples)}
    for key in ("solver_ms", "step_ms", "measurement_ms", "nonlinear_iterations", "linear_iterations"):
        values = [s[key] for s in samples]
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    result["solver_real_time_factor"] = 1000 * dt / result["solver_ms"]["mean"]
    result["measured_step_real_time_factor"] = 1000 * dt / result["step_ms"]["mean"]
    return result


def comparison(rows):
    """Compare only complete, settled repetitions against the finest tested mesh."""
    valid = [row for row in rows if row["verified"]]
    finest = valid[-1] if valid else None
    previous = None
    for row in rows:
        row["relative_difference_to_finest"] = None
        row["relative_change_from_previous"] = None
        if not row["verified"]:
            previous = None
            continue
        tip = row["tip_mean_m"]
        row["relative_difference_to_finest"] = abs(tip - finest["tip_mean_m"]) / abs(finest["tip_mean_m"])
        if previous is not None:
            row["relative_change_from_previous"] = abs(tip - previous["tip_mean_m"]) / abs(tip)
        previous = row
    return {
        "cases": rows,
        "all_verified": bool(rows and all(row["verified"] for row in rows)),
        "reference_refinement": finest["refinement"] if finest else None,
        "reference_scope": "finest verified mesh in this sweep, not continuum truth; fixed dt, no extrapolation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8], choices=range(1, 9))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.refinements != sorted(set(args.refinements)) or args.repeats < 1:
        parser.error("Use strictly increasing refinements and positive repeats")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        parser.error("Set OPENBLAS_NUM_THREADS=1 for comparable reference/measurement overhead")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "device": str(wp.get_device(args.device)),
        "refinements": args.refinements,
        "repeats": args.repeats,
        "warmup_steps_per_mesh": 100,
        "dt_s": 0.02,
        "duration_s": 36,
        "scope": "serial headless wall time; synchronized solver.step; separate diagnostic host work; no rendering or file IO",
        "warmup": "separate solver trajectory for first 2 s; fresh solver/state for each measured full trajectory",
        "environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for refinement in args.refinements:
        print(f"r{refinement}: warmup", flush=True)
        warmup = ResponseCase(args.device, experiment="gravity", variant=0, refinement=refinement)
        for _ in range(100):
            warmup.step()
        del warmup
        gc.collect()
        runs = []
        for repeat in range(args.repeats):
            start = perf_counter()
            case = ResponseCase(args.device, experiment="gravity", variant=0, refinement=refinement)
            wp.synchronize_device(case.model.device)
            setup = perf_counter() - start
            samples, failure = [], None
            directory = args.output / f"r{refinement}" / f"repeat-{repeat + 1}"
            directory.mkdir(parents=True, exist_ok=True)
            try:
                for step in range(round(case.duration / case.dt)):
                    samples.append({"time": (step + 1) * case.dt, **case.step(profile=True)})
                    if (step + 1) % 300 == 0:
                        print(f"r{refinement} repeat {repeat + 1}: {step + 1} steps", flush=True)
            except (RuntimeError, AssertionError) as error:
                failure = str(error)
            summary = case.summary()
            perf = performance(samples, case.dt) if samples else None
            run = {
                "summary": summary,
                "performance": perf,
                "setup_s": setup,
                "failure": failure,
                "phases": {
                    name: performance(part, case.dt)
                    for name, part in (
                        ("ramp", [s for s in samples if s["time"] <= 2]),
                        ("settling", [s for s in samples if 2 < s["time"] < 34]),
                        ("tail", [s for s in samples if s["time"] >= 34]),
                    )
                    if part
                },
            }
            for name, data in (("manifest", case.manifest), ("result", run)):
                (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
            for name, data in (("trace", case.records), ("timing", samples)):
                (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            np.savez(directory / "final-state.npz", q=case.state.particle_q.numpy(), qd=case.state.particle_qd.numpy())
            runs.append(run)
            print(f"r{refinement} repeat {repeat + 1}: {json.dumps(run)}", flush=True)
            counts = {
                "node_count": case.model.particle_count,
                "tet_count": case.model.tet_count,
                "dynamic_dofs": 3 * int(np.sum(~case.fixed)) + 1,
                "linear_static_tip_m": case.linear_static_tip,
            }
            del case
            gc.collect()
        verified = all(r["failure"] is None and r["summary"]["demo_verified"] for r in runs)
        rows.append(
            {
                "refinement": refinement,
                **counts,
                "verified": verified,
                "runs": runs,
                "tip_mean_m": float(np.mean([r["summary"]["tail_mean_tip_m"] for r in runs])) if verified else None,
            }
        )
        result = comparison(rows)
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_verified"]:
        raise SystemExit("Incomplete/unsettled cases: retained diagnostics; see results.json")


if __name__ == "__main__":
    main()
