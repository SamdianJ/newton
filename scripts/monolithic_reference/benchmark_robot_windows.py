# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure warmed robot solver steps on the unchanged frozen quick windows."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import serial
from scripts.monolithic_reference.benchmark_workload import check_gpu_idle
from scripts.monolithic_reference.profiling import capture_count, provenance
from scripts.monolithic_reference.quick_gates import evaluate
from scripts.monolithic_reference.quick_manifest import load_manifest
from scripts.monolithic_reference.quick_scenes import WINDOWS
from scripts.monolithic_reference.retained_assets import validate_inputs
from scripts.monolithic_reference.run_quick_regression import MANIFEST, SNAPSHOT_ROOT
from scripts.monolithic_reference.state_snapshots import load_snapshot


def span_graphs(solver):
    """Snapshot dynamic-span caches outside the timed solver calls."""
    caches = {}
    if solver._contact_pool:
        caches["pool"] = solver._contact.pool.graphs
        caches.update({f"bsr_{i}": owner.graphs for i, owner in enumerate(solver._linear._fixed_bsr)})
    if solver._contact_spatial:
        caches["query"] = solver._contact.query_graphs
    result = {}
    for owner, cache in caches.items():
        for name, segment in cache.summary()["segments"].items():
            index = {"contact_candidate_order": 0, "contact_stream_finish": -1, "fixed_scalar_bsr": 2}.get(name)
            if index is not None:
                result[f"{owner}.{name}"] = {
                    **segment,
                    "resident_spans": [key[1][index] for key in cache.entries[name]],
                }
    return result


def measure_window(window, case, values, model_values, precision):
    """Warm a complete window, restore its snapshot, and time solver calls only."""
    solver = window.solver
    device = solver.model.device
    original = solver.step
    samples = []
    for phase in ("warmup", "measured"):
        check_gpu_idle(device)
        window.start(values, model_values, case["snapshot"]["step"])
        captures = capture_count(solver)
        spans_before = span_graphs(solver)
        rows, seconds = [], []

        def timed_step(*args, seconds=seconds, **kwargs):
            wp.synchronize_device(device)
            start = time.perf_counter()
            result = original(*args, **kwargs)
            wp.synchronize_device(device)
            seconds.append(time.perf_counter() - start)
            return result

        with patch.object(solver, "step", timed_step):
            for _ in range(case["steps"]):
                row = window.advance()
                stats = solver.last_stats
                row.update(
                    seconds=seconds[-1],
                    pcg=int(stats.linear_iterations),
                    scheduled_slots=int(stats.linear_scheduled_slots),
                    trials=int(stats.line_search_iterations),
                    regularization_retries=int(stats.regularization_retries),
                )
                rows.append(row)
                if row["publication"] == "HARD_FAILURE":
                    break
        check_gpu_idle(device)
        gates = evaluate(rows, case["gates"][precision], weight=window.weight, expected_steps=case["steps"])
        samples.append(
            {
                "phase": phase,
                "status": "PASS" if all(g["passed"] for g in gates.values()) else "FAIL",
                "gates": gates,
                "rows": rows,
                "counts": dict(Counter(r["publication"] for r in rows)),
                "total_seconds": sum(seconds),
                "median_ms": float(np.median(seconds) * 1000),
                "p95_ms": float(np.percentile(seconds, 95) * 1000),
                "captures": capture_count(solver) - captures,
                "span_graphs_before": spans_before,
                "span_graphs_after": span_graphs(solver),
            }
        )
    return {
        "status": "PASS" if all(sample["status"] == "PASS" for sample in samples) else "FAIL",
        "samples": samples,
    }


def run(output, *, precision="fp32", device="cuda:0", only=()):
    """Collect every selected window even when a scene fails its frozen gates."""
    manifest = load_manifest(MANIFEST)
    cases = [case for case in manifest["cases"] if precision in case["gates"]]
    unknown = set(only) - {case["id"] for case in cases}
    if unknown:
        raise ValueError(f"Unknown or inapplicable {precision} cases: {sorted(unknown)}")
    cases = [case for case in cases if not only or case["id"] in only]
    before = provenance()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=before["root"], text=True
    ).strip():
        raise ValueError("Commit tracked changes before measuring")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "source": before,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_manifest_sha256": validate_inputs(),
        "quick_manifest_sha256": manifest["sha256"],
        "precision": precision,
        "device": device,
        "planned": [case["id"] for case in cases],
        "measurement": "One full warmup window, then the same snapshot and controls. Synchronized solver.step wall time includes validation, H2 and Stats; excludes setup, snapshot restoration, controls and offline contact measurement. Recurring captures remain timed. Frozen physical gates are unchanged; Stopped is separate from Converged.",
        "cases": [],
    }
    receipt = output / "receipt.json"
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        row = {"id": case["id"], "scene": case["scene"], "snapshot": case["snapshot"], "steps": case["steps"]}
        try:
            values, model_values = load_snapshot(SNAPSHOT_ROOT / case["snapshot"]["path"], case["snapshot"]["sha256"])
            window = WINDOWS[case["scene"]](precision, device)
            row.update(measure_window(window, case, values, model_values, precision))
            row.update(
                dt=window.dt,
                physical_digest=window.solver._kinematics.digest.hex(),
                config=serial(window.solver._config),
                model_counts={
                    name: int(getattr(window.solver.model, name))
                    for name in ("body_count", "joint_dof_count", "particle_count", "tet_count", "shape_count")
                },
                scene_checks=window.scene_checks,
                import_report=window.import_report,
                graphs=window.solver._graphs.summary(),
            )
        except Exception as error:
            row.update(status="ERROR", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        report["cases"].append(row)
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        print(case["id"], row["status"], flush=True)
    after = provenance()
    report["source_unchanged"] = all(before[key] == after[key] for key in ("commit", "source_hashes"))
    report["status"] = (
        "PASS" if report["source_unchanged"] and all(row["status"] == "PASS" for row in report["cases"]) else "FAIL"
    )
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output, precision=args.precision, device=args.device, only=args.case)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
