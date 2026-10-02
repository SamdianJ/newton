# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recompute recorded windows or production hotspots in a fresh process."""

import argparse
import json
import time
from pathlib import Path

import warp as wp

from newton.examples.softbody.monolithic_recording import state_arrays
from scripts.monolithic_reference.profiling import provenance
from scripts.monolithic_reference.record_workload import apply_inputs
from scripts.monolithic_reference.workload_io import load_arrays, open_bundle, write_json
from scripts.monolithic_reference.workload_pcg import replay_pcg
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays


def replay(root, *, mode="window", device="cuda:0", repeat=5):
    """Validate the entire bundle before constructing any GPU objects."""
    root = Path(root)
    manifest = open_bundle(root)
    results = []
    if mode == "hotspot":
        if not manifest["snapshots"]:
            raise ValueError("No captured hotspot calls")
        for name in manifest["snapshots"]:
            if name + "/call.json" not in manifest["files"]:
                raise ValueError("Unauthenticated hotspot metadata")
            path = root / name
            metadata = json.loads((path / "call.json").read_text())
            if metadata["boundary"] != "production_pcg_core":
                raise ValueError("Unsupported hotspot boundary")
            result = replay_pcg(
                load_arrays(path / "input.npz", metadata["input_schema"]),
                metadata,
                load_arrays(path / "output.npz", metadata["output_schema"]),
                metadata["result"],
                device=device,
                repeat=repeat,
            )
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
                actual = [(r["iteration"], r["retry"], r["alpha"], r["accepted"]) for r in run._variational["trials"]]
                wanted = [(r["iteration"], r["retry"], r["alpha"], r["accepted"]) for r in trace[step - 1]["trials"]]
                if actual != wanted:
                    raise ValueError(f"Acceptance trace differs at step {step}")
            results.append({"window": [first, last], "seconds_including_capture": times, "status": "PASS"})
    else:
        raise ValueError("Unknown replay mode")
    return {
        "status": "PASS",
        "mode": mode,
        "source": provenance(),
        "recorded_source": manifest["source"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("window", "hotspot"), default="window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--verify", action="store_true", help="Compatibility flag; verification is always required")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    result = replay(args.bundle, mode=args.mode, device=args.device, repeat=args.repeat)
    write_json(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
