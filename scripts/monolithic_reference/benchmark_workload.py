# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure warmed production windows and objective sequences on frozen inputs."""

import argparse
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import warp as wp

from newton.examples.softbody.monolithic_recording import state_arrays
from scripts.monolithic_reference.profiling import provenance
from scripts.monolithic_reference.record_workload import apply_inputs
from scripts.monolithic_reference.replay_workload import acceptance_trace
from scripts.monolithic_reference.workload_io import digest, load_arrays, open_bundle, write_json
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays


def check_gpu_idle(device):
    """Reject concurrent compute processes instead of accepting polluted samples."""
    if device.is_cuda:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(int(pid) != os.getpid() for pid in output.split()):
            raise RuntimeError("Another process is using the GPU")


def benchmark(root, *, device="cuda:0", warmup=2, repeat=5):
    """Keep setup, restoration, validation and objective probes outside step timing."""
    if warmup < 1 or repeat < 1:
        raise ValueError("Positive warmup and repeat counts are required")
    root = Path(root)
    manifest = open_bundle(root)
    selected = {step for first, last in manifest["windows"] for step in range(first, last + 1)}
    trace = {}
    number = 0
    with (root / "trace.jsonl").open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["step"] != number:
                raise ValueError("Incomplete execution trace")
            if number in selected:
                trace[number] = row
    if number != manifest["steps"]:
        raise ValueError("Incomplete execution trace")
    results = []
    for first, last in manifest["windows"]:
        model, run, state, _ = build_scene(manifest["scene"], device=device, graph=manifest["graph"])
        if run._kinematics.digest.hex() != manifest["physical_digest"]:
            raise ValueError("Scene physical identity changed")
        control = model.control()
        inputs, expected = {}, {}
        for step in range(first, last + 1):
            for prefix, target in (("inputs", inputs), ("state", expected)):
                key = f"{prefix}-{step}.npz"
                target[step] = load_arrays(root / key, manifest["array_schemas"][key])
        objective = run._evaluate
        objective_seconds = []

        def timed_objective(*args, objective=objective, objective_seconds=objective_seconds, **kwargs):
            start = time.perf_counter()
            result = objective(*args, **kwargs)
            # _evaluate already reads E/r back before returning.
            objective_seconds.append(time.perf_counter() - start)
            return result

        samples = []
        for scope in ("step", "objective"):
            for index in range(warmup + repeat):
                check_gpu_idle(model.device)
                start = time.perf_counter()
                run.load_checkpoint(state, root / f"checkpoint-{first}.npz")
                initial_generation = int(state._monolithic_coupled.force_generation.numpy()[0])
                offset = max(run._step_generation - initial_generation, 0)
                restore_seconds = time.perf_counter() - start
                captures = sum(run._graphs.captures.values())
                objective_captures = run._graphs.captures["coupled_objective"]
                seconds, work = [], []
                objective_seconds.clear()
                with patch.object(run, "_evaluate", timed_objective) if scope == "objective" else nullcontext():
                    for step in range(first, last + 1):
                        apply_inputs(inputs[step], model, state, control)
                        wp.synchronize_device(model.device)
                        start = time.perf_counter()
                        run.step(state, state, control, dt=manifest["dt"])
                        wp.synchronize_device(model.device)
                        seconds.append(time.perf_counter() - start)
                        # Restoring an old physical state must not rewind workspace epochs.
                        wanted = dict(expected[step])
                        wanted["history.force_generation"] = wanted["history.force_generation"] + offset
                        compare_arrays(state_arrays(state), wanted, **manifest["comparison"])
                        stats = run.last_stats
                        counts = {
                            name: getattr(stats, name)
                            for name in (
                                "converged",
                                "rolled_back",
                                "nonlinear_iterations",
                                "linear_iterations",
                                "linear_scheduled_slots",
                                "line_search_iterations",
                                "regularization_retries",
                            )
                        }
                        if any(value != trace[step]["stats"][name] for name, value in counts.items()):
                            raise ValueError(f"Execution work changed at step {step}")
                        if acceptance_trace(run._variational["trials"]) != acceptance_trace(trace[step]["trials"]):
                            raise ValueError(f"Acceptance trace changed at step {step}")
                        work.append(counts)
                check_gpu_idle(model.device)
                samples.append(
                    {
                        "scope": scope,
                        "warmup": index < warmup,
                        "seconds": sum(seconds) if scope == "step" else sum(objective_seconds),
                        "step_seconds": seconds,
                        "restore_seconds": restore_seconds,
                        "captures": sum(run._graphs.captures.values()) - captures,
                        "objective_captures": run._graphs.captures["coupled_objective"] - objective_captures,
                        "generation_offset": offset,
                        "work": work,
                    }
                )
        results.append({"window": [first, last], "samples": samples, "graphs": run._graphs.summary()})
    return {
        "status": "PASS",
        "scene": manifest["scene"],
        "source": provenance(),
        "benchmark_sha256": digest(Path(__file__)),
        "bundle_manifest_sha256": digest(root / "manifest.json"),
        "warmup": warmup,
        "repeat": repeat,
        "results": results,
        "measurement": "Production step sums without profiler; objective probes separate. Initial warmup excluded; recurring cache eviction/capture remains real execution cost.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = benchmark(args.bundle, device=args.device, warmup=args.warmup, repeat=args.repeat)
    write_json(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
