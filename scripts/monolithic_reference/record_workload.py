# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record real production steps, control inputs, and selected computational calls."""

import argparse
import json
import time
from collections import Counter
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.examples.softbody.monolithic_recording import serial, state_arrays
from scripts.monolithic_reference.profiling import provenance
from scripts.monolithic_reference.workload_io import complete_bundle, save_arrays, write_json
from scripts.monolithic_reference.workload_objective import capture_input, capture_output
from scripts.monolithic_reference.workload_pcg import pcg_input, pcg_output
from scripts.monolithic_reference.workload_scene import build_scene, set_control


def inputs(model, state, control):
    """Enumerate all externally mutable step inputs consumed by this solver."""
    arrays = {
        "control." + name: getattr(control, name).numpy().copy()
        for name in ("joint_f", "joint_target_q", "joint_target_qd")
    }
    arrays.update(
        {
            "body_f": state.body_f.numpy().copy(),
            "particle_f": state.particle_f.numpy().copy(),
            "gravity": model.gravity.numpy().copy(),
        }
    )
    return arrays


def apply_inputs(arrays, model, state, control):
    """Restore input contents without replacing bound allocations."""
    targets = {"control." + name: getattr(control, name) for name in ("joint_f", "joint_target_q", "joint_target_qd")}
    targets.update(body_f=state.body_f, particle_f=state.particle_f, gravity=model.gravity)
    if arrays.keys() != targets.keys():
        raise ValueError("Step input fields differ")
    for name, target in targets.items():
        if (
            arrays[name].shape != target.numpy().shape
            or arrays[name].dtype != np.float32
            or not np.isfinite(arrays[name]).all()
        ):
            raise ValueError(f"Invalid step input: {name}")
    for name, target in targets.items():
        target.assign(arrays[name])


class CallRecorder:
    """Capture selected calls while preserving the original host decisions."""

    def __init__(self, run, root, *, boundaries=("pcg",), call_ordinals=None):
        if set(boundaries) - {"pcg", "objective", "assembly"}:
            raise ValueError("Unsupported hotspot boundary")
        if call_ordinals is not None and any(
            name not in {"pcg", "objective", "assembly"}
            or not isinstance(values, list)
            or any(type(value) is not int or value < 0 for value in values)
            for name, values in call_ordinals.items()
        ):
            raise ValueError("Call ordinals must be nonnegative integer lists for supported boundaries")
        self.run, self.root, self.boundaries = run, root, boundaries
        self.call_ordinals = call_ordinals or {name: [0] for name in boundaries}
        self.step = 0
        self.capture = False
        self.calls = []
        self.snapshots = []
        self.counts = Counter()
        self.patches = ExitStack()

    def _observe(self, owner, method, boundary):
        original = getattr(owner, method)

        def observed(*args, **kwargs):
            ordinal = self.counts[boundary]
            self.counts[boundary] += 1
            row = {"boundary": boundary, "ordinal": ordinal}
            if boundary == "trial":
                row["alpha"] = float(args[0])
            if boundary == "assembly":
                row["iteration"] = int(args[0])
            capture = self.capture and boundary in self.boundaries and ordinal in self.call_ordinals.get(boundary, [0])
            if capture:
                directory = self.root / "snapshots" / f"{self.step:06d}-{boundary}-{ordinal}"
                directory.mkdir(parents=True)
                data, metadata = (
                    pcg_input(owner, *args, **kwargs)
                    if boundary == "pcg"
                    else capture_input(self.run, boundary, args, kwargs)
                )
                metadata["input_schema"] = save_arrays(directory / "input.npz", data)
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                row["failure"] = type(error).__name__
                row["message"] = str(error)
                raise
            finally:
                row["host_seconds_with_recording"] = time.perf_counter() - start
                self.calls.append(row)
            if boundary == "pcg":
                row.update(
                    status=int(result.status),
                    iterations=result.iterations,
                    slots=result.scheduled_slots,
                    graph_rebuilt=result.graph_rebuilt,
                )
            if capture:
                if boundary == "pcg":
                    output, expected_result = pcg_output(owner, args[1], kwargs["delta"])
                else:
                    output, expected_result = capture_output(self.run, boundary, result), None
                metadata["output_schema"] = save_arrays(directory / "output.npz", output)
                metadata["result"] = expected_result
                metadata["step"] = self.step
                write_json(directory / "call.json", metadata)
                self.snapshots.append(str(directory.relative_to(self.root)))
            return result

        self.patches.enter_context(patch.object(owner, method, observed))

    def __enter__(self):
        for method, boundary in (("_evaluate", "objective"), ("_assemble", "assembly"), ("_take_trial", "trial")):
            self._observe(self.run, method, boundary)
        self._observe(self.run._linear, "solve_pcg", "pcg")
        return self

    def __exit__(self, *exc):
        self.patches.close()


def record(
    root,
    *,
    scene="tiny-contact",
    device="cuda:0",
    graph=True,
    steps=10,
    windows=None,
    boundaries=("pcg",),
    call_ordinals=None,
):
    """Advance the original scene once; retain only explicitly selected windows."""
    windows = [(6, 10)] if windows is None else [tuple(w) for w in windows]
    if steps < 1 or not windows or any(a < 1 or b < a or b > steps for a, b in windows):
        raise ValueError("Windows must be nonempty inclusive step ranges within the run")
    if any(a <= previous[1] for previous, (a, _) in pairwise(windows)):
        raise ValueError("Windows must be ordered and disjoint")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    model, run, state, samples = build_scene(scene, device=device, graph=graph)
    if samples is not None and steps >= len(samples):
        raise ValueError("Step range exceeds scene controls")
    control = model.control()
    manifest = {
        "scene": scene,
        "graph": graph,
        "device": str(model.device),
        "dt": 0.001,
        "steps": steps,
        "windows": windows,
        "source": provenance(),
        "physical_digest": run._kinematics.digest.hex(),
        "precision": run.precision.value,
        "pcg_mode": run.pcg_mode.value,
        "array_schemas": {},
        "snapshots": [],
        "comparison": {"rtol": 2e-5, "atol": 2e-6},
    }
    starts = {a for a, _ in windows}
    with (
        CallRecorder(run, root, boundaries=boundaries, call_ordinals=call_ordinals) as recorder,
        (root / "trace.jsonl").open("w") as trace,
    ):
        for step in range(1, steps + 1):
            selected = any(a <= step <= b for a, b in windows)
            set_control(control, samples, step)
            if step in starts:
                run.save_checkpoint(state, root / f"checkpoint-{step}.npz")
            if selected:
                filename = f"inputs-{step}.npz"
                manifest["array_schemas"][filename] = save_arrays(root / filename, inputs(model, state, control))
            recorder.step, recorder.capture = step, step in starts
            recorder.calls, recorder.counts = [], Counter()
            run.step(state, state, control, dt=0.001)
            stats = run.last_stats
            trace.write(
                json.dumps(
                    serial(
                        {"step": step, "calls": recorder.calls, "stats": stats, "trials": run._variational["trials"]}
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            if stats.rolled_back:
                raise RuntimeError(f"Step {step}: {stats.failure_reason}")
            if selected:
                filename = f"state-{step}.npz"
                manifest["array_schemas"][filename] = save_arrays(root / filename, state_arrays(state))
            if step % 250 == 0:
                print(json.dumps({"step": step, "scene": scene}), flush=True)
        manifest["snapshots"] = recorder.snapshots
    complete_bundle(root, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("tiny-contact", "franka-default", "sharpa-default"), default="tiny-contact")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--start", type=int, default=6)
    parser.add_argument(
        "--boundaries", nargs="+", choices=("pcg", "objective", "assembly"), default=["pcg", "objective", "assembly"]
    )
    parser.add_argument("--selection", type=Path, help="JSON containing ordered inclusive windows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text()) if args.selection else {}
    for key, expected in (
        ("schema", "monolithic-workload-selection/1"),
        ("scene", args.case),
        ("steps", args.steps),
        ("comparison", {"rtol": 2e-5, "atol": 2e-6}),
    ):
        if key in selection and selection[key] != expected:
            parser.error(f"Selection {key} differs from the requested workload")
    windows = selection.get("windows", [(args.start, args.steps)])
    record(
        args.output,
        scene=args.case,
        device=args.device,
        graph=not args.no_graph,
        steps=args.steps,
        windows=windows,
        boundaries=args.boundaries,
        call_ordinals=selection.get("call_ordinals"),
    )
    print(json.dumps({"status": "COMPLETE", "output": str(args.output)}))


if __name__ == "__main__":
    main()
