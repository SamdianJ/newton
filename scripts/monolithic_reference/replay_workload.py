# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recompute recorded windows or production hotspots in a fresh process."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import warp as wp

from newton.examples.softbody.monolithic_recording import state_arrays
from scripts.monolithic_reference.profiling import SolverProfiler, provenance
from scripts.monolithic_reference.record_workload import apply_inputs
from scripts.monolithic_reference.workload_io import digest, load_arrays, open_bundle, write_json
from scripts.monolithic_reference.workload_objective import replay_objective
from scripts.monolithic_reference.workload_pcg import replay_pcg
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays


def acceptance_trace(trials):
    """Preserve rejected domain evaluations as well as ordinary trial decisions."""
    fields = ("iteration", "retry", "alpha", "accepted", "domain_failure")
    return [tuple(row.get(name) for name in fields) for row in trials]


def replay(root, *, mode="window", device="cuda:0", repeat=5, profile=False):
    """Validate the entire bundle before constructing any GPU objects."""
    root = Path(root)
    manifest = open_bundle(root)
    if (
        manifest["scene"].startswith("actor-coupling")
        and manifest["source"]["source_hashes"] != provenance()["source_hashes"]
    ):
        raise ValueError("Mixed workload source identity differs")
    results = []
    if mode == "hotspot":
        if not manifest["snapshots"]:
            raise ValueError("No captured hotspot calls")
        for name in manifest["snapshots"]:
            if name + "/call.json" not in manifest["files"]:
                raise ValueError("Unauthenticated hotspot metadata")
            path = root / name
            metadata = json.loads((path / "call.json").read_text())
            arrays = load_arrays(path / "input.npz", metadata["input_schema"])
            expected = load_arrays(path / "output.npz", metadata["output_schema"])
            if metadata["boundary"] == "production_pcg_core":
                result = replay_pcg(arrays, metadata, expected, metadata["result"], device=device, repeat=repeat)
            elif metadata["boundary"] in ("objective", "assembly"):
                result = replay_objective(
                    arrays,
                    metadata,
                    expected,
                    scene=manifest["scene"],
                    device=device,
                    graph=manifest["graph"],
                    repeat=repeat,
                )
            else:
                raise ValueError("Unsupported hotspot boundary")
            results.append({"snapshot": name, **result})
    elif mode == "window":
        trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
        if [r["step"] for r in trace] != list(range(1, manifest["steps"] + 1)):
            raise ValueError("Incomplete execution trace")
        for first, last in manifest["windows"]:
            model, run, state, _ = build_scene(manifest["scene"], device=device, graph=manifest["graph"])
            if run._kinematics.digest.hex() != manifest["physical_digest"]:
                raise ValueError("Scene physical identity changed")
            run.load_checkpoint(state, root / f"checkpoint-{first}.npz")
            control = model.control()
            times = []
            profiler = SolverProfiler(run) if profile else None
            with profiler if profile else nullcontext():
                for step in range(first, last + 1):
                    key = f"inputs-{step}.npz"
                    apply_inputs(load_arrays(root / key, manifest["array_schemas"][key]), model, state, control)
                    wp.synchronize_device(model.device)
                    start = time.perf_counter()
                    run.step(state, state, control, dt=manifest["dt"])
                    wp.synchronize_device(model.device)
                    times.append(time.perf_counter() - start)
                    key = f"state-{step}.npz"
                    compare_arrays(
                        state_arrays(state),
                        load_arrays(root / key, manifest["array_schemas"][key]),
                        **manifest["comparison"],
                    )
                    expected = trace[step - 1]["stats"]
                    for name in (
                        "converged",
                        "rolled_back",
                        "nonlinear_iterations",
                        "linear_iterations",
                        "line_search_iterations",
                    ):
                        if getattr(run.last_stats, name) != expected[name]:
                            raise ValueError(f"Execution trace differs at step {step}: {name}")
                    # Recompute acceptance; do not force recorded branch outcomes.
                    actual = acceptance_trace(run._variational["trials"])
                    wanted = acceptance_trace(trace[step - 1]["trials"])
                    if actual != wanted:
                        raise ValueError(f"Acceptance trace differs at step {step}")
            results.append(
                {
                    "window": [first, last],
                    "seconds_including_capture": times,
                    "status": "PASS",
                    "instrumented": profile,
                    "profile": profiler.report() if profiler else None,
                }
            )
    else:
        raise ValueError("Unknown replay mode")
    return {
        "status": "PASS",
        "mode": mode,
        "source": provenance(),
        "recorded_source": manifest["source"],
        "bundle_manifest_sha256": digest(root / "manifest.json"),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("window", "hotspot"), default="window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Instrument window replay; timings are diagnostic")
    parser.add_argument("--verify", action="store_true", help="Compatibility flag; verification is always required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    result = replay(args.bundle, mode=args.mode, device=args.device, repeat=args.repeat, profile=args.profile)
    write_json(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
