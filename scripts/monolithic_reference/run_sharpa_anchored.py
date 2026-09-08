# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run complete anchored closures and separate instrumented holding tails."""

import gc
import json
import time
from pathlib import Path

import warp as wp

from newton.examples.softbody.example_monolithic_sharpa_soft_ball import Example
from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase
from scripts.monolithic_reference.profile_release import _StageProfile


def main():
    parser = Example.create_parser()
    parser.add_argument("--refinements", type=int, choices=(1, 2, 3), nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    if args.experiment != "anchored-close":
        parser.error("This runner measures anchored-close only")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for refinement in args.refinements:
        args.ball = str(Path(__file__).with_name("fixtures") / "soft_ball" / f"ball_r{refinement}.npz")
        start = time.perf_counter()
        case = GraspCase(args)
        setup = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(case.total_steps):
            case.step()
            if case.failure:
                break
            if case.step_count % 500 == 0:
                r = case.records[-1]
                print(
                    f"r{refinement}: {r['time']:.3f}s, force={r['finger_force_n']:.4f}N, "
                    f"penetration={1000 * r['penetration']:.3f}mm, detF={r['min_det_f']:.3f}",
                    flush=True,
                )
        wall = time.perf_counter() - start
        out = root / f"r{refinement}"
        summary = case.save(out)
        # Keep formal snapshots at 4.5s. This separate continuous tail is instrumented
        # with synchronization, so its overlapping stages are not additive timings.
        tail = {"steps": 0, "failure": None}
        record = {
            "refinement": refinement,
            "setup_seconds": setup,
            "run_wall_seconds": wall,
            "summary": summary,
            "profile_tail_passed": False,
        }
        results.append(record)
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        try:
            if summary["complete_schedule"]:
                with _StageProfile(case.solver, "actor_block") as timer:
                    for _ in range(20):
                        case.solver.step(case.state, case.state, case.control, None, case.fixture["dt"])
                        if case.solver.last_stats.rolled_back:
                            tail["failure"] = case.solver.last_stats.failure_reason
                            break
                        tail["steps"] += 1
                tail["instrumented_stage_timings"] = timer.summary()
                del timer
        except Exception as error:
            tail["failure"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["profile_tail_passed"] = tail["steps"] == 20 and tail["failure"] is None
            (out / "profile-tail.json").write_text(json.dumps(tail, indent=2) + "\n")
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        del case
        gc.collect()
        wp.synchronize_device(args.device)
    if not all(r["summary"]["stability_passed"] and r["profile_tail_passed"] for r in results):
        raise RuntimeError("Anchored stability gate failed; retain failed traces and unchanged thresholds")


if __name__ == "__main__":
    main()
