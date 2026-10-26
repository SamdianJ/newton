# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay one authenticated Franka frame without running its trajectory prefix."""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.examples.softbody import monolithic_franka
from newton.examples.softbody.monolithic_franka import FrankaGrasp
from newton.examples.softbody.monolithic_recording import serial, state_arrays
from scripts.monolithic_reference.check_recording import sha
from scripts.monolithic_reference.workload_io import save_arrays, write_json
from scripts.monolithic_reference.workload_pcg import pcg_input


def recorded_frame(recording, step):
    """Authenticate the selected chunk and retain every native state field."""
    recording = recording.resolve()
    index = json.loads((recording / "trajectory-index.json").read_text())
    offset = 0
    for chunk in index["chunks"]:
        if offset <= step < offset + chunk["count"]:
            path = (recording / chunk["path"]).resolve()
            if not path.is_relative_to(recording) or sha(path) != chunk["sha256"]:
                raise ValueError("Frame path/hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                rows = np.flatnonzero(data["frame_steps"] == step)
                if len(rows) != 1:
                    raise ValueError("Missing or duplicate frame")
                frame = {name: data[name][rows[0]].copy() for name in data.files}
            return frame, chunk["sha256"]
        offset += chunk["count"]
    raise ValueError("Frame not found")


def restore(case, frame, *, allow_policy_migration=False):
    """Authenticate old physical inputs before explicitly rebinding only Q1 policy."""
    current = state_arrays(case.state)
    original_digest = frame["history.config_digest"]
    migration = None
    if not np.array_equal(current["history.config_digest"], original_digest) and allow_policy_migration:
        run = case.solver
        if not np.array_equal(run._coupled_config.get("direction_policy"), np.array([1], dtype=np.int32)):
            raise ValueError("Unsupported policy migration")
        policy = run._coupled_config.pop("direction_policy")
        try:
            legacy = run._make_kinematics(
                run.model,
                extra_config={
                    "joint_terms": run._joint_terms.params.numpy(),
                    "joint_switches": np.array([True, True, True], dtype=np.uint8),
                },
            )
        finally:
            run._coupled_config["direction_policy"] = policy
        np.testing.assert_array_equal(
            np.frombuffer(legacy.digest, dtype=np.uint8), original_digest, err_msg="Original physical/policy identity"
        )
        migration = {
            "original_config_digest": original_digest.tobytes().hex(),
            "new_config_digest": current["history.config_digest"].tobytes().hex(),
            "changed_policy": "direction_policy/1",
        }
        frame = dict(frame)
        frame["history.config_digest"] = current["history.config_digest"]
    for name, value in current.items():
        if name not in frame or value.shape != frame[name].shape or value.dtype != frame[name].dtype:
            raise ValueError(f"Frame schema mismatch: {name}")
        if name in ("history.config_digest", "history.layout_digest", "history.actor_layout", "history.schema"):
            np.testing.assert_array_equal(value, frame[name], err_msg=name)
    target = case.solver._states.copy(case.state)
    for name in current:
        owner = target._monolithic_coupled if name.startswith("history.") else target
        getattr(owner, name.removeprefix("history.")).assign(frame[name])
    case.solver._states.validate(target)
    for name, value in state_arrays(target).items():
        np.testing.assert_array_equal(value, frame[name], err_msg=name)
    case.state.assign(target)
    return migration


def main():
    """Save the one-step result, actual PCG inputs and authoritative source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=4229)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-policy-migration", action="store_true")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="production")
    parser.add_argument("--use-cuda-graph", choices=("on", "off"), default="on")
    parser.add_argument("--timing-repetitions", type=int, default=0)
    args = parser.parse_args()
    if args.timing_repetitions < 0:
        parser.error("--timing-repetitions must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    frame, chunk_hash = recorded_frame(args.recording, args.frame)
    factory = monolithic_franka.SolverMonolithic

    class ConfiguredSolver(factory):
        def __init__(self, *a, **kw):
            kw.update(pcg_mode=args.pcg_mode, use_cuda_graph=args.use_cuda_graph == "on")
            super().__init__(*a, **kw)

    with patch.object(monolithic_franka, "SolverMonolithic", ConfiguredSolver):
        case = FrankaGrasp(
            Path("asset/monolithic/data/inputs/franka/sdf-cache"),
        )
    migration = restore(case, frame, allow_policy_migration=args.allow_policy_migration)
    case.steps = args.frame
    if args.timing_repetitions:
        samples = []
        states = []
        for index in range(args.timing_repetitions + 1):
            restore(case, frame, allow_policy_migration=args.allow_policy_migration)
            case.steps, case.failed = args.frame, False
            wp.synchronize_device(case.model.device)
            start = time.perf_counter()
            stats = case.step()
            wp.synchronize_device(case.model.device)
            elapsed = time.perf_counter() - start
            if index:
                samples.append({"seconds": elapsed, "stats": serial(stats)})
                states.append(state_arrays(case.state))
        for index, state in enumerate(states):
            save_arrays(args.output / f"final-{index:02d}.npz", state)
        write_json(
            args.output / "timing.json",
            {
                "frame": args.frame,
                "chunk_sha256": chunk_hash,
                "policy_migration": migration,
                "observer_enabled": False,
                "warmup_excluded": 1,
                "samples": samples,
                "source": {str(p): sha(p) for p in Path("newton/_src/solvers/monolithic").glob("*.py")},
                "tool_sha256": sha(Path(__file__)),
            },
        )
        print("median_seconds", float(np.median([s["seconds"] for s in samples])), flush=True)
        return
    original = case.solver._linear.solve_pcg
    calls = []

    def observed(rhs, y, **kwargs):
        schema, metadata = None, None
        if args.pcg_mode == "production":
            arrays, metadata = pcg_input(case.solver._linear, rhs, y, **kwargs)
            arrays["native_residual"] = case.solver._accepted_r.numpy().copy()
            schema = save_arrays(args.output / f"linear-input-{len(calls):02d}.npz", arrays)
        result = original(rhs, y, **kwargs)
        calls.append(
            {
                "result": serial(result),
                "input": metadata,
                "schema": schema,
                "input_sha256": sha(args.output / f"linear-input-{len(calls):02d}.npz") if schema is not None else None,
            }
        )
        return result

    case.solver._linear.solve_pcg = observed
    start = time.perf_counter()
    stats = case.step()
    report = {
        "frame": args.frame,
        "recording": str(args.recording.resolve()),
        "chunk_sha256": chunk_hash,
        "policy_migration": migration,
        "source": {str(p): sha(p) for p in Path("newton/_src/solvers/monolithic").glob("*.py")},
        "tool_sha256": sha(Path(__file__)),
        "stats": serial(stats),
        "pcg": calls,
        "elapsed_seconds_including_observers": time.perf_counter() - start,
        "long_run": "NOT_RUN",
    }
    write_json(args.output / "result.json", report)
    save_arrays(args.output / "final.npz", state_arrays(case.state))
    save_arrays(
        args.output / "candidate-direction.npz",
        {"delta": case.solver._delta.numpy(), "native_error": case.solver._post_error.numpy()},
    )
    print(stats.status.value, stats.variational["publication_status"], stats.linear_iterations, flush=True)


if __name__ == "__main__":
    main()
