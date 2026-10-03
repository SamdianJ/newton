# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize observed host costs and verified, selected hotspot measurements."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.workload_io import digest, load_arrays, open_bundle, write_json


def summarize(bundle, receipts):
    """Keep diagnostic full-scene accounting separate from selected microbenchmarks."""
    manifest = open_bundle(bundle)
    trace = [json.loads(line) for line in (bundle / "trace.jsonl").read_text().splitlines()]
    if [r["step"] for r in trace] != list(range(1, manifest["steps"] + 1)):
        raise ValueError("Incomplete trace")
    costs, calls, counts = Counter(), Counter(), Counter()
    for row in trace:
        stats = row["stats"]
        counts["HARD_FAILURE" if stats["rolled_back"] else "CONVERGED" if stats["converged"] else "STOPPED"] += 1
        for call in row["calls"]:
            costs[call["boundary"]] += call["host_seconds_with_recording"]
            calls[call["boundary"]] += 1
    total = sum(costs.values())
    hotspots, profiles = [], []
    for receipt in receipts:
        data = json.loads(receipt.read_text())
        if data["status"] != "PASS" or data.get("bundle_manifest_sha256") != digest(bundle / "manifest.json"):
            raise ValueError("Unverified or foreign replay receipt")
        for row in data["results"]:
            if "snapshot" in row:
                times = np.asarray(row["seconds"])
                if len(times) < 5:
                    raise ValueError("Need at least five hotspot repetitions")
                hotspots.append(
                    {
                        "snapshot": row["snapshot"],
                        "boundary": row["boundary"],
                        "samples": len(times),
                        "p50_ms": float(np.median(times) * 1e3),
                        "p95_ms": float(np.percentile(times, 95) * 1e3),
                        "capture_affected": row.get("capture_affected", [False] * len(times)),
                    }
                )
            if row.get("profile"):
                profiles.append(
                    {
                        "window": row.get("window"),
                        "snapshot": row.get("snapshot"),
                        "modules": row["profile"]["modules"],
                        "opaque": row["profile"]["opaque"],
                    }
                )
    activity = defaultdict(list)
    for first, last in manifest["windows"]:
        for row in trace[first - 1 : last]:
            name = f"state-{row['step']}.npz"
            state = load_arrays(bundle / name, manifest["array_schemas"][name])
            activity[f"{first}-{last}"].append(
                {
                    "step": row["step"],
                    "contact_particle_force_norm": float(np.linalg.norm(state["history.contact_particle_force"])),
                    "contact_body_force_norm": float(np.linalg.norm(state["history.contact_body_force"])),
                    "active_samples": None,
                    "pcg_iterations": row["stats"]["linear_iterations"],
                    "nonlinear_iterations": row["stats"]["nonlinear_iterations"],
                }
            )
    return {
        "status": "BASELINE_RECORDED",
        "scene": manifest["scene"],
        "steps": len(trace),
        "counts": counts,
        "qualification": "Tool correctness and selected timings; not full-scene physics certification or speedup",
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "receipt_hashes": {str(p): digest(p) for p in receipts},
        "host_cost_scope": "Observed boundary host costs during recording; excludes unobserved step work; not GPU busy time",
        "host_modules": {
            name: {
                "calls": calls[name],
                "seconds": costs[name],
                "share_of_observed": costs[name] / total if total else None,
            }
            for name in costs
        },
        "window_activity": activity,
        "hotspots": hotspots,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    write_json(args.output, summarize(args.bundle, args.receipt))


if __name__ == "__main__":
    main()
