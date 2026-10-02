# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile explicit short windows of a production Monolithic scene."""

import argparse
import json
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import warp as wp

from newton.examples.softbody.monolithic_recording import state_arrays
from scripts.monolithic_reference.profiling import SolverProfiler, provenance
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays, set_control


def profile(name, *, device, graph, steps, activities=False):
    """Warm, restore and compare uninstrumented and instrumented runs."""
    results = []
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "initial.npz"
        for instrumented in (False, True):
            model, run, state, samples = build_scene(name, device=device, graph=graph)
            control = model.control()
            run.save_checkpoint(state, checkpoint)
            for step in range(1, steps + 1):
                set_control(control, samples, step)
                run.step(state, state, control, dt=0.001)
                if run.last_stats.rolled_back:
                    raise RuntimeError(run.last_stats.failure_reason)
            run.load_checkpoint(state, checkpoint)
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            with SolverProfiler(run, activities=activities) if instrumented else nullcontext() as profiler:
                for step in range(1, steps + 1):
                    set_control(control, samples, step)
                    run.step(state, state, control, dt=0.001)
                    if run.last_stats.rolled_back:
                        raise RuntimeError(run.last_stats.failure_reason)
            wp.synchronize_device(model.device)
            results.append((time.perf_counter() - start, state_arrays(state)))
        baseline, expected = results[0]
        elapsed, actual = results[1]
        errors = compare_arrays(actual, expected)
    return {
        "status": "PASS",
        "provenance": provenance(),
        "scene": name,
        "steps": steps,
        "baseline_seconds": baseline,
        "instrumented_seconds": elapsed,
        "overhead_ratio": elapsed / baseline,
        "state_errors": errors,
        "profile": profiler.report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("tiny-contact", "franka-default", "sharpa-default"), default="tiny-contact")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--activities", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    result = profile(
        args.case, device=args.device, graph=not args.no_graph, steps=args.steps, activities=args.activities
    )
    (args.output / "profile.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
