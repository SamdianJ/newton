# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one independently warmed PR-8A job with immutable provenance and GPU monitoring."""

import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import warp as wp

from scripts.monolithic_reference import run_p2_sharpa_study as sharpa
from scripts.monolithic_reference import run_p2_tet_study as tet
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources
from scripts.monolithic_reference.profile_release import _rss_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("sharpa", "sharpa-profile", "sharpa-matrix", "tet", "tet-profile", "tet-matrix"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refinement", type=int, default=3)
    parser.add_argument("--newton", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if args.kind.startswith("sharpa") and (args.device != "cuda:0" or args.refinement not in (2, 3)):
        raise ValueError("Sharpa protocol requires cuda:0 and r2/r3")
    if args.substeps not in (1, 2, 4, 5, 10, 20, 40):
        raise ValueError("Substeps must belong to the frozen P2 scan")
    args.output.mkdir(parents=True)
    snapshot_sources(args.output / "provenance")
    device = wp.get_device(args.device)
    wp.synchronize_device(device)
    record = {
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_device": str(device),
        "device_name": device.name,
        "pid": os.getpid(),
        "start_unix_s": time.time(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "status": "RUNNING",
        "warmup_steps": 100,
        "quiet_status": "PENDING_TELEMETRY_REVIEW",
        "memory_semantics": "RSS/HWM includes warmup/setup; CUDA logs sample driver usage each second; "
        "Warp pool high is allocated pool bytes, not total device memory.",
    }
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    monitors = []
    with ExitStack() as stack:
        if shutil.which("vmstat"):
            log = stack.enter_context((args.output / "host-vmstat.txt").open("w"))
            monitors.append(subprocess.Popen(["vmstat", "-w", "-t", "1"], stdout=log, stderr=subprocess.STDOUT))
        if device.is_cuda:
            for name, query in (
                (
                    "gpu",
                    "--query-gpu=timestamp,index,name,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm",
                ),
                ("compute-processes", "--query-compute-apps=timestamp,pid,process_name,used_memory"),
            ):
                log = stack.enter_context((args.output / f"{name}.csv").open("w"))
                monitors.append(
                    subprocess.Popen(
                        ["nvidia-smi", query, "--format=csv,noheader,nounits", "--loop=1"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
        try:
            warmup_start = time.perf_counter()
            if args.kind.startswith("sharpa"):
                mesh = f"r{args.refinement}"
                sharpa.warmup(mesh, args.output)
            else:
                tet.warmup(args.device, args.refinement)
            record["warmup_seconds"] = time.perf_counter() - warmup_start
            record["measurement_start_unix_s"] = time.time()
            if args.kind.startswith("sharpa"):
                if args.smoke:
                    sharpa.DURATION = 0.08
                job = {
                    "name": args.output.name,
                    "task": "PR-8A",
                    "mesh": mesh,
                    "newton_max_iterations": args.newton,
                    "physical_dt_s": sharpa.FRAME_DT / args.substeps,
                    "capture_matrices": args.kind == "sharpa-matrix",
                }
                result = sharpa.run_case(job, args.output, stage_profile=args.kind == "sharpa-profile")
            elif args.kind == "tet":
                result = tet.timing_run(args.device, args.refinement, args.output, steps=6 if args.smoke else None)
            elif args.kind == "tet-profile":
                result, _ = tet.staged_window(args.device, args.refinement, args.output)
            else:
                candidates = tet.matrix_run(
                    args.device, args.refinement, args.output / "matrices", steps=50 if args.smoke else 1750
                )
                tet.material_replay(args.device, args.refinement, candidates["loading"], args.output)
                result = {"captured": list(candidates)}
            record["status"] = "FAILED" if result.get("failure") else "COMPLETED"
            record["failure"] = result.get("failure")
        except BaseException as error:
            record["status"] = "FAILED"
            record["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record.update(end_unix_s=time.time(), memory=_rss_bytes())
            for monitor in monitors:
                monitor.terminate()
                monitor.wait(timeout=10)
            (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    evidence_index(args.output)
    if record["status"] == "FAILED":
        raise RuntimeError(record["failure"])


if __name__ == "__main__":
    main()
