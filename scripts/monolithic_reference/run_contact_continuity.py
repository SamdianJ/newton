# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a contact-policy ablation with unchanged physical and solver tolerances."""

import argparse
import dataclasses
import functools
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton.examples.softbody import monolithic_sharpa_grasp as grasp
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources
from scripts.monolithic_reference.probe_contact_continuity import probe_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "contact-dir", "trajectory", "calibration"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=("baseline", "sdf", "history", "combined"), required=True)
    parser.add_argument("--position", choices=(18, 24), type=int, required=True)
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="production")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--cube", default="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz")
    parser.add_argument("--friction-off", action="store_true")
    parser.add_argument("--probe-failure", action="store_true")
    parser.add_argument("--probe-decomposition", action="store_true")
    parser.add_argument("--probe-balance", action="store_true")
    parser.set_defaults(device="cuda:0", experiment="grasp", disable_aabb=False)
    args = parser.parse_args()
    if args.probe_decomposition and not args.probe_failure:
        parser.error("--probe-decomposition requires --probe-failure")
    if args.probe_balance and not args.probe_failure:
        parser.error("--probe-balance requires --probe-failure")
    if not 1 <= args.steps <= 2500:
        parser.error("Close-only diagnosis requires --steps in [1, 2500]")
    args.position_offset_mm = (10, -10, -5) if args.position == 18 else (10, 10, -5)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    options = {
        "_cubic_sdf": args.variant in ("sdf", "combined"),
        "_smooth_history": args.variant in ("history", "combined"),
    }
    start = time.perf_counter()
    case, monitor = None, None
    failure = None
    try:
        with (args.output / "gpu.csv").open("w") as gpu:
            monitor = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=timestamp,pstate,clocks.sm,power.draw", "--format=csv", "-l", "1"],
                stdout=gpu,
                stderr=subprocess.STDOUT,
            )
            with patch.object(
                grasp, "MonolithicCollisionPipeline", functools.partial(MonolithicCollisionPipeline, **options)
            ):
                case = grasp.GraspCase(args)
            setup_seconds = time.perf_counter() - start
            solver = case.solver
            original = solver._iterate
            source, first = inspect.getsourcelines(original)
            markers = {
                "if self._converged(merit, reference):": "current",
                "step_rms = self._rms(self._y, scaled=False)": "pcg",
                "if trial.merit <= (1.0 - 1.0e-4 * alpha) * merit[0] + config.merit_noise:": "trial",
            }
            lines = {first + i: markers[line.strip()] for i, line in enumerate(source) if line.strip() in markers}
            if set(lines.values()) != set(markers.values()):
                raise RuntimeError("Trace markers changed; do not silently omit diagnostics")
            with (args.output / "newton.jsonl").open("w") as log:
                probed = False

                def trace(frame, event, arg):
                    nonlocal probed
                    if frame.f_code is not original.__func__.__code__:
                        return None
                    if (
                        event == "exception"
                        and args.probe_failure
                        and not probed
                        and getattr(arg[1], "status", None) == solver.Status.REGULARIZATION_EXHAUSTED
                    ):
                        probed = True
                        probe_failure(case, args.output)
                        if args.probe_decomposition:
                            # Optional SciPy oracle is a research-only dependency.
                            from scripts.monolithic_reference.probe_close_direction import (  # noqa: PLC0415
                                probe_close_direction,
                            )

                            probe_close_direction(case, args.output)
                        if args.probe_balance:
                            from scripts.monolithic_reference.probe_close_balance import (  # noqa: PLC0415
                                probe_close_balance,
                            )

                            probe_close_balance(case, args.output)
                    if event == "line" and frame.f_lineno in lines:
                        local = frame.f_locals
                        kind = lines[frame.f_lineno]
                        row = {"event": kind, "step": case.step_count + 1}
                        for name in ("iteration", "lambda_value", "retry", "alpha", "merit", "ratios", "reference"):
                            if name in local:
                                row[name] = local[name]
                        if kind == "pcg":
                            result = local["result"]
                            row.update(
                                pcg_iterations=result.iterations,
                                pcg_status=result.status.name,
                                rho=result.rho,
                                rho_q=result.rho_q,
                                rho_x=result.rho_x,
                            )
                        if kind == "trial":
                            trial = local["trial"]
                            row.update(
                                trial_merit=trial.merit,
                                trial_q=trial.merit_q,
                                trial_x=trial.merit_x,
                                threshold=(1 - 1e-4 * local["alpha"]) * local["merit"][0] + solver._config.merit_noise,
                            )
                        log.write(json.dumps(row) + "\n")
                    return trace

                def observed(dt):
                    sys.settrace(trace)
                    try:
                        return original(dt)
                    finally:
                        sys.settrace(None)

                solver._iterate = observed
                while not case.failure and case.step_count < args.steps:
                    case.step()
                    log.flush()
                    if case.step_count % 100 == 0 or case.failure:
                        print(args.variant, args.position, case.step_count, case.failure, flush=True)
            failure = case.failure
            case.save(args.output)
            stats = {f.name: getattr(solver.last_stats, f.name) for f in dataclasses.fields(solver.last_stats)}
            (args.output / "last-stats.json").write_text(json.dumps(stats, default=str, indent=2) + "\n")
            (args.output / "configuration.json").write_text(
                json.dumps(
                    {
                        "options": options,
                        "history_transition_degrees": [40, 45] if options["_smooth_history"] else None,
                        "solver": dataclasses.asdict(solver._config),
                        "fields": solver.collision_pipeline._cubic_evidence,
                        "setup_seconds": setup_seconds,
                        "performance_sample": False,
                        "calibration_status": "REVALIDATION_REQUIRED" if args.variant != "baseline" else "BASELINE",
                    },
                    default=str,
                    indent=2,
                )
                + "\n"
            )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        print(failure, flush=True)
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        (args.output / "run.json").write_text(
            json.dumps(
                {
                    "variant": args.variant,
                    "position": args.position,
                    "failure": failure,
                    "accepted_steps": case.step_count if case else 0,
                    "requested_steps": args.steps,
                    "elapsed_seconds": time.perf_counter() - start,
                    "performance_sample": False,
                    "G6": "NOT_VALIDATED",
                    "G7": "NOT_VALIDATED",
                },
                indent=2,
            )
            + "\n"
        )
        evidence_index(args.output)
    return int(failure is not None)


if __name__ == "__main__":
    sys.exit(main())
