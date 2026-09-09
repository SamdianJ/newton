# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare CUDA mass paths at saved Sharpa poses, not trajectory/history checkpoints."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.sim.articulation import compute_body_spatial_inertia, eval_articulation_mass_matrix
from newton._src.solvers.monolithic import articulation
from scripts.monolithic_reference import run_p2_sharpa_study as sharpa
from scripts.monolithic_reference.p2_measurement import array_fingerprint, evidence_index, snapshot_sources
from scripts.monolithic_reference.profile_release import _rss_bytes


def load_samples(compact, manifest):
    """Require matching assets and exact recorded poses before replaying any candidate."""
    if not compact["complete"] or compact.get("failure"):
        raise ValueError("A complete source trajectory is required")
    names = [name for name in compact["manifest"] if name.endswith("sha256")]
    names += ["joint_names", "mapping"]
    for name in names:
        if manifest.get(name) != compact["manifest"][name]:
            raise ValueError(f"Source asset/configuration mismatch: {name}")
    samples = []
    for time_s in (1.0, 3.5):
        rows = [row for row in compact["common_time"]["samples"] if abs(row["time"] - time_s) < 1e-9]
        if len(rows) != 1 or any(name not in rows[0] for name in ("q", "qd")):
            raise ValueError(f"Missing unique q/qd sample at {time_s} s")
        samples.append(rows[0])
    return samples


def profile_pose(case, workspaces, sample, output, repeats):
    model, state = case.model, case.state
    q, qd = (np.asarray(sample[name], dtype=np.float32) for name in ("q", "qd"))
    if q.shape != state.joint_q.shape or qd.shape != state.joint_qd.shape:
        raise ValueError("Saved q/qd shape does not match the current model")
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError("Saved q/qd must be finite")
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)

    def inertia(workspace):
        wp.launch(
            compute_body_spatial_inertia,
            model.body_count,
            inputs=[model.body_inertia, model.body_mass, state.body_q, workspace.scratch.body_I_s],
            device=model.device,
        )

    def raw(workspace):
        args = [model.articulation_start, model.articulation_end]
        if not workspace.use_optimized_articulation_mass_matrix:
            args += [model.articulation_count, None]
        args += [model.joint_child, model.joint_qd_start, workspace.scratch.body_I_s, workspace.scratch.J, workspace.M]
        owned = workspace.use_optimized_articulation_mass_matrix
        wp.launch(
            articulation._eval_owned_mass_matrix if owned else eval_articulation_mass_matrix,
            workspace.M.shape[1:] if owned else model.articulation_count,
            inputs=args,
            device=model.device,
        )

    def full(workspace):
        if workspace.use_optimized_articulation_mass_matrix:
            inertia(workspace)
            raw(workspace)
        else:
            articulation.eval_mass_matrix(
                model,
                state,
                H=workspace.M,
                J=workspace.scratch.J,
                body_I_s=workspace.scratch.body_I_s,
                joint_S_s=workspace.scratch.joint_S_s,
            )

    def passive(workspace):
        articulation.eval_articulation_passive_candidate(model, state, workspace)

    for workspace in workspaces.values():
        passive(workspace)
        # Coriolis reuses body_I_s; rebuild spatial inertia before raw mass replay.
        inertia(workspace)
    reference, owned = workspaces.values()
    np.testing.assert_array_equal(reference.scratch.J.numpy(), owned.scratch.J.numpy())
    np.testing.assert_array_equal(reference.scratch.body_I_s.numpy(), owned.scratch.body_I_s.numpy())
    inputs = {"q": q, "qd": qd, "J": reference.scratch.J.numpy(), "body_I_s": reference.scratch.body_I_s.numpy()}
    initial_mass = reference.M.numpy().copy()
    start, end = (wp.Event(model.device, enable_timing=True) for _ in range(2))
    timings = {}
    for stage, function in (("raw_kernel", raw), ("full_mass", full), ("full_passive", passive)):
        rows = {name: [] for name in workspaces}
        for iteration in range(10 + repeats):
            names = list(workspaces) if iteration % 2 == 0 else list(reversed(workspaces))
            for name in names:
                workspace = workspaces[name]
                if stage == "raw_kernel":
                    workspace.M.zero_()  # Required by reference; excluded from raw-kernel timing only.
                wp.synchronize_device(model.device)
                wall_start = time.perf_counter()
                wp.record_event(start)
                function(workspace)
                wp.record_event(end)
                wp.synchronize_event(end)
                wall_ms = (time.perf_counter() - wall_start) * 1000
                if iteration >= 10:
                    rows[name].append(
                        {
                            "cuda_event_ms": wp.get_event_elapsed_time(start, end, synchronize=False),
                            "completed_wall_ms": wall_ms,
                        }
                    )
        for workspace in workspaces.values():
            np.testing.assert_allclose(workspace.M.numpy(), initial_mass, rtol=1e-5, atol=1e-8)
        np.testing.assert_array_equal(state.joint_q.numpy(), q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), qd)
        timings[stage] = {
            name: {
                "samples": values,
                "percentiles": {
                    field: dict(
                        zip(
                            ("p50", "p95"),
                            np.percentile([row[field] for row in values], (50, 95)).tolist(),
                            strict=True,
                        )
                    )
                    for field in ("cuda_event_ms", "completed_wall_ms")
                },
            }
            for name, values in rows.items()
        }
    for name in ("g", "C"):
        np.testing.assert_allclose(getattr(reference, name).numpy(), getattr(owned, name).numpy(), rtol=1e-5, atol=1e-8)
    np.savez(output, **inputs, M_reference=initial_mass, M_owned=owned.M.numpy())
    return {"time_s": sample["time"], "input_sha256": array_fingerprint(inputs), "parity": "PASS", "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="8A compact.json with common q/qd samples")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    compact = json.loads(args.input.read_text())
    if compact["mesh"] not in ("r2", "r3"):
        parser.error("Source must be a Sharpa r2/r3 trajectory")
    args.output.mkdir(parents=True)
    snapshot_sources(args.output / "provenance")

    def device_state():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()

    record = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "source_compact": str(args.input.resolve()),
        "source_compact_sha256": sharpa.sha256(args.input),
        "repeats": args.repeats,
        "warmup_per_stage": 10,
        "scope": "Fixed joint poses only; not a history checkpoint or trajectory performance result.",
        "timing_semantics": "Raw excludes output clear; full mass includes required clear/inertia update; "
        "full passive includes FK/J/gravity/C. CUDA events include stream idle between launches; "
        "completed wall includes event markers and end-event wait. No candidate-array allocation in timed calls.",
        "poses": [],
        "start_unix_s": time.time(),
    }
    (args.output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        record["device_before"] = device_state()
        case = sharpa.GraspCase(sharpa.args_for(compact["mesh"], args.output))
        record["manifest"] = case.manifest
        samples = load_samples(compact, case.manifest)
        workspaces = {
            "reference": case.solver._articulation,
            "owned": articulation.MonolithicArticulationWorkspace(
                case.model, use_optimized_articulation_mass_matrix=True
            ),
        }
        with wp.ScopedDevice(case.model.device):
            for sample in samples:
                record["poses"].append(
                    profile_pose(case, workspaces, sample, args.output / f"pose-{sample['time']:.1f}.npz", args.repeats)
                )
                (args.output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
        record["device_after"] = device_state()
        record["status"] = "COMPLETED"
    except BaseException as error:
        record["status"] = "FAILED"
        record["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record.update(end_unix_s=time.time(), memory=_rss_bytes())
        (args.output / "profile.json").write_text(json.dumps(record, indent=2) + "\n")
        evidence_index(args.output)


if __name__ == "__main__":
    main()
