# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare warmed task windows across checkouts with identical physical inputs.

Run this file with PYTHONPATH=. from the checkout under test. --candidate only
selects the original translation-only, guard-policy scene in the newer runner.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.benchmark_workload import check_gpu_idle
from scripts.monolithic_reference.profiling import capture_count
from scripts.monolithic_reference.run_dexterous_tasks import DT, REPO, TaskScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pinch-transfer", "pinch-place", "cable-pull"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare-step", type=int)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--newton-iterations", type=int, default=400)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    if (
        (args.checkpoint is None) == (args.prepare_step is None)
        or args.steps <= 0
        or args.repeat <= 0
        or args.newton_iterations <= 0
    ):
        parser.error("Choose a checkpoint or positive preparation step, with positive steps/repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    options = {"wrist": False, "alignment_policy": "guard"} if args.candidate else {}
    scene = TaskScene(args.task, newton_iterations=args.newton_iterations, **options)
    run = scene.solver
    run.newton_max_iterations = 80
    check_gpu_idle(scene.model.device)
    receipt = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "newton_import": newton.__file__,
        "device": str(scene.model.device),
        "device_name": scene.model.device.name,
        "warp_version": wp.__version__,
        "physical_digest": run._kinematics.digest.hex(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                REPO / "scripts/monolithic_reference/run_dexterous_tasks.py",
                *sorted((REPO / "newton/_src/solvers/monolithic").glob("*.py")),
            ]
        },
        "samples": [],
    }

    def advance(first, last):
        totals = {"newton": 0, "pcg": 0, "line_search": 0}
        samples = []
        for step in range(first, last + 1):
            wp.synchronize_device(scene.model.device)
            started = time.perf_counter()
            scene.step(step * DT)
            wp.synchronize_device(scene.model.device)
            seconds = time.perf_counter() - started
            stats = run.last_stats
            if not stats.converged or stats.rolled_back:
                raise RuntimeError(f"Step {step}: {stats.status}: {stats.failure_reason}")
            totals["newton"] += stats.nonlinear_iterations
            totals["pcg"] += stats.linear_iterations
            totals["line_search"] += stats.line_search_iterations
            samples.append(
                {
                    "step": step,
                    "seconds": seconds,
                    "newton": stats.nonlinear_iterations,
                    "pcg": stats.linear_iterations,
                    "line_search": stats.line_search_iterations,
                }
            )
        return {
            **totals,
            "step_samples": samples,
            "median_ms": 1000 * float(np.median([r["seconds"] for r in samples])),
            "p95_ms": 1000 * float(np.percentile([r["seconds"] for r in samples], 95)),
        }

    if args.prepare_step is not None:
        if args.prepare_step <= 0:
            parser.error("prepare-step must be positive")
        advance(1, args.prepare_step)
        run.save_checkpoint(scene.state, args.output / "checkpoint.npz")
        receipt["status"] = "PREPARED"
    else:
        receipt["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        for repeat in range(args.repeat + 1):
            check_gpu_idle(scene.model.device)
            run.load_checkpoint(scene.state, args.checkpoint)
            first = int(scene.state._monolithic_coupled.force_generation.numpy()[0]) + 1
            captures = capture_count(run)
            wp.synchronize_device(scene.model.device)
            start = time.perf_counter()
            totals = advance(first, first + args.steps - 1)
            wp.synchronize_device(scene.model.device)
            seconds = time.perf_counter() - start
            if repeat:
                if capture_count(run) != captures:
                    raise RuntimeError("Timed window still captures CUDA graphs")
                receipt["samples"].append({"seconds": seconds, "steps": args.steps, **totals})
        receipt["status"] = "MEASURED"
    receipt["source_unchanged"] = (
        all(
            hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
            for name, digest in receipt["source_sha256"].items()
        )
        and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == receipt["script_sha256"]
    )
    if not receipt["source_unchanged"]:
        raise RuntimeError("Source changed during measurement")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        receipt["status"], [{k: v for k, v in r.items() if k != "step_samples"} for r in receipt["samples"]], flush=True
    )


if __name__ == "__main__":
    main()
