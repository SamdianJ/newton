# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run or batch the predeclared PR8H cube exploration after physical calibration."""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import warp as wp

from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase, load_calibration
from newton.examples.softbody.monolithic_soft_cube import load_cube, validate_cube, validate_cube_calibration
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources
from scripts.monolithic_reference.profile_release import _rss_bytes


def offsets():
    return list(itertools.product((-10, 0, 10), (-10, 0, 10), (-5, 0, 5)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "contact-dir", "trajectory", "calibration"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--cube", default="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--position-offset-mm", nargs=3, type=float, default=(0, 0, 0))
    parser.add_argument("--friction-off", action="store_true")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="production")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.set_defaults(experiment="grasp", disable_aabb=False)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    try:
        calibration = load_calibration(args.calibration)
        validate_cube_calibration(calibration, args.cube, dt=0.001)
    except Exception as error:
        (args.output / "summary.json").write_text(
            json.dumps(
                {"status": "BLOCKED_CALIBRATION", "failure": str(error), "G6": "NOT_VALIDATED", "G7": "NOT_VALIDATED"},
                indent=2,
            )
            + "\n"
        )
        evidence_index(args.output)
        return 1
    if args.scan:
        if args.friction_off or args.pcg_mode != "production" or tuple(args.position_offset_mm) != (0, 0, 0):
            raise ValueError("First scan uses production graph, friction on and the declared offset grid")
        mesh = load_cube(args.cube)
        if validate_cube(mesh.vertices, mesh.tet_indices.reshape(-1, 4))["refinement"] != 2:
            raise ValueError("First scan requires the middle grid")
        jobs = []
        for index, offset in enumerate(offsets()):
            output = args.output / f"position-{index:02d}"
            command = [
                sys.executable,
                "-m",
                __name__ if __name__ != "__main__" else "scripts.monolithic_reference.run_cube_grasp",
            ]
            for key in ("asset_dir", "contact_dir", "trajectory", "calibration", "cube", "device"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            command.extend(["--output", str(output), "--position-offset-mm", *map(str, offset)])
            print(f"START {index + 1}/27 offset={offset}", flush=True)
            try:
                with (args.output / f"position-{index:02d}.log").open("w") as log:
                    with subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    ) as process:
                        try:
                            code = process.wait(timeout=2400)
                        except subprocess.TimeoutExpired:
                            # Stop the job and its telemetry child together; keep the partial artifacts.
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            raise
                failure = None
            except (subprocess.TimeoutExpired, OSError) as error:
                code, failure = 1, str(error)
            jobs.append({"offset_mm": offset, "output": str(output), "exit_code": code, "failure": failure})
            (args.output / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        evidence_index(args.output)
        return int(any(row["exit_code"] for row in jobs))
    start = time.perf_counter()
    case = None
    monitor = None
    try:
        with (args.output / "gpu.csv").open("w") as log:
            monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,pstate,clocks.sm,power.draw,memory.used,utilization.gpu",
                    "--format=csv",
                    "--loop=1",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            case = GraspCase(args)
            wp.synchronize_device(case.model.device)
            setup = time.perf_counter() - start
            while case.step_count < case.total_steps and case.failure is None:
                case.step()
            result = case.save(args.output)
            result.update(
                setup_seconds=setup,
                wall_plus_setup_seconds=time.perf_counter() - start,
                rss_hwm_bytes=_rss_bytes(),
                performance_status="PILOT_NOT_ADMISSION",
                warp_pool_high_bytes=wp.get_mempool_used_mem_high(case.model.device),
            )
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            return int(not result["complete_schedule"])
    except Exception as error:
        if case is not None:
            case.failure = f"{type(error).__name__}: {error}"
            case.save(args.output)
        (args.output / "failure.json").write_text(
            json.dumps({"failure": f"{type(error).__name__}: {error}"}, indent=2) + "\n"
        )
        return 1
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        evidence_index(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
