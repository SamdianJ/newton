# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run two A/A baselines, three or five independent AB/BA pairs and a final A bridge.

A is current-source legacy, B is current-source B3; both use default lambda zero,
FP64 CUDA graph and suppressed host pair reports. This measures Q2 cost, not
qualification against the historical P6a source or a new physical acceptance.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.run_performance import ROOT, performance_case, sha


def run_order(*, pair_count=5):
    """Freeze independent process order before observing candidate results."""
    if pair_count not in (3, 5):
        raise ValueError("Pair count must be 3 or 5")
    return (
        ["AA1-A", "AA2-A"]
        + [f"P{i}-{side}" for i in range(1, pair_count + 1) for side in ("AB" if i % 2 else "BA")]
        + ["bridge-A"]
    )


def paired_summary(receipts, *, pair_count=5):
    """Reject partial or mixed evidence before applying the frozen timing rules."""
    order = run_order(pair_count=pair_count)
    if len(receipts) != len(order):
        raise ValueError("Incomplete paired batch")
    first = receipts[0]
    for label, row in zip(order, receipts, strict=True):
        if (
            row["status"] != "COMPLETE_NOT_AUDITED"
            or row["final_state_checks"] != "FINITE_HISTORY_FORCE_GENERATION_PASS"
        ):
            raise ValueError("Incomplete or failed timing run")
        if row["contact_dataflow"] != ("legacy" if label.endswith("A") else "spatial"):
            raise ValueError("Run order/contact route differs")
        for field in ("source_hashes", "case", "controls_sha256", "numerical_policy", "python", "numpy", "warp"):
            if row[field] != first[field]:
                raise ValueError(f"Mixed paired identity: {field}")
        for field in ("input_manifest_sha256", "fixture_sha256", "control_shapes"):
            if row.get(field) != first.get(field):
                raise ValueError(f"Mixed paired identity: {field}")
        if row["device"] != "cuda:0" or not row["graph"] or row["contact_pair_diagnostics"] or not row["gc_enabled"]:
            raise ValueError("Formal execution/diagnostic mode differs")
        if row["numerical_policy"]["regularization_values"] != [0.0]:
            raise ValueError("Formal batch requires lambda zero on both sides")
        if row["counts"].get("HARD_FAILURE", 0) or row["counts"].get("STOPPED", 0):
            raise ValueError("Timing batch contains failure or Stopped steps")
        if row["published_steps"] != row["case"]["steps"]:
            raise ValueError("Incomplete physical clock")
    totals, tails, pairs = [], [], []
    for i in range(pair_count):
        pair = receipts[2 + 2 * i : 4 + 2 * i]
        a, b = sorted(pair, key=lambda row: row["contact_dataflow"])
        total, tail = b["solve_seconds"] / a["solve_seconds"], b["step_p95_seconds"] / a["step_p95_seconds"]
        totals.append(total)
        tails.append(tail)
        pairs.append(
            {"pair": i + 1, "total_B_over_A": total, "p95_B_over_A": tail, "A_work": a["totals"], "B_work": b["totals"]}
        )
    aa = receipts[1]["solve_seconds"] / first["solve_seconds"]
    bridge = receipts[-1]["solve_seconds"] / np.median([r["solve_seconds"] for r in receipts[:2]])
    environment = bool(max(aa, 1 / aa, bridge, 1 / bridge) <= 1.05)
    no_regression = float(np.median(totals)) <= 1.05 and float(np.median(tails)) <= 1.10
    return {
        "status": "ENVIRONMENT_DRIFT"
        if not environment
        else "PASS_TIMING_NO_REGRESSION"
        if no_regression
        else "FAIL_TIMING_REGRESSION",
        "pair_count": pair_count,
        "total_ratio_median": float(np.median(totals)),
        "p95_ratio_median": float(np.median(tails)),
        "timing_benefit_observed": bool(
            environment and no_regression and all(r < 1 for r in totals) and np.median(totals) < 0.95
        ),
        "AA_ratio": aa,
        "bridge_ratio": float(bridge),
        "pairs": pairs,
        "physical_qualification": "SEPARATE_REVIEW_REQUIRED",
        "qualified_speedup": None,
        "historical_P5_I2": "NOT_TESTED_BY_THIS_SAME_SOURCE_COMPARISON",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("franka-cube", "actor-coupling"), nargs="+", default=["franka-cube", "actor-coupling"]
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the protocol and commands without constructing or running scenes",
    )
    parser.add_argument("--pairs", type=int, choices=(3, 5), default=5, help="Independent AB/BA pairs per scene")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for scene in args.scene:
        case = performance_case(scene, "fp64")
        for label in run_order(pair_count=args.pairs):
            command = [
                sys.executable,
                "-B",
                "-m",
                "scripts.monolithic_reference.run_performance",
                "--scene",
                scene,
                "--precision",
                "fp64",
                "--no-contact-diagnostics",
                "--contact-dataflow",
                "legacy" if label.endswith("A") else "spatial",
                "--output",
                str((args.output / scene / label).resolve()),
            ]
            jobs.append({"scene": scene, "label": label, "steps": case["steps"], "command": command})
    plan = {
        "status": "PREPARED_NOT_RUN",
        "pair_count": args.pairs,
        "baseline": "same-source legacy/Q1/lambda0",
        "candidate": "same-source B3/Q1/lambda0",
        "total_ratio_limit": 1.05,
        "p95_ratio_limit": 1.10,
        "environment_drift_limit": 0.05,
        "warmup_steps_per_process": 20,
        "timed_steps": sum(j["steps"] for j in jobs),
        "jobs": jobs,
    }
    plan["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan["tracked_dirty"] = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    )
    plan["source_hashes"] = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ("newton", "scripts/monolithic_reference")
        for p in (ROOT / folder).rglob("*.py")
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"status": plan["status"], "processes": len(jobs), "timed_steps": plan["timed_steps"]}))
        return 0
    summaries, receipts = {}, []
    try:
        for job in jobs:
            print(json.dumps({"scene": job["scene"], "run": job["label"]}), flush=True)
            # Every timed run owns a fresh process. No profiler or recording launcher wraps it.
            log = args.output / f"{job['scene']}-{job['label']}.log"
            with log.open("w") as stream:
                subprocess.run(job["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            folder = Path(job["command"][-1])
            receipt = json.loads((folder / "receipt.json").read_text())
            if receipt["source_hashes"] != plan["source_hashes"]:
                raise ValueError("Source changed after the paired protocol was saved")
            receipts.append(receipt)
            if job["label"] == "AA2-A":
                ratio = receipts[-1]["solve_seconds"] / receipts[-2]["solve_seconds"]
                if max(ratio, 1 / ratio) > 1.05:
                    raise ValueError("A/A noise exceeds 5%; start a separate batch after inspecting the environment")
            if job["label"] == "bridge-A":
                summaries[job["scene"]] = paired_summary(receipts, pair_count=args.pairs)
                receipts = []
    except (subprocess.CalledProcessError, ValueError) as error:
        summaries["error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return int("error" in summaries or any(r["status"] != "PASS_TIMING_NO_REGRESSION" for r in summaries.values()))


if __name__ == "__main__":
    raise SystemExit(main())
