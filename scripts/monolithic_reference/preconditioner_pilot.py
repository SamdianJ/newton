# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""PR8F frozen-matrix pilot. Run each candidate in a separate process; no admission or nightly."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference import preconditioner_study as study
from scripts.monolithic_reference.linear_evidence import array_fingerprint, evidence_index, snapshot_sources


def timed(call, device=None):
    if device is not None:
        wp.synchronize_device(device)
    start = time.perf_counter()
    result = call()
    if device is not None:
        wp.synchronize_device(device)
    return result, 1000 * (time.perf_counter() - start)


def gpu_case(case, candidate, mode, config, repeats, device):
    reference = benchmark.Reference(case, config)
    hierarchy, hierarchy_ms = timed(lambda: study.build_hierarchy(case.arrays, candidate))
    runner, setup_ms = timed(
        lambda: benchmark.Runner(
            case, mode, device, config, study_hierarchy=None if candidate == "actor_block" else hierarchy
        ),
        device,
    )
    w = runner.workspace
    cold = benchmark._timed_solve(runner, "reuse")
    cold["accuracy"] = reference.measure(runner.y.numpy())
    output = wp.zeros_like(runner.rhs)

    def apply():
        w.preconditioner.matvec(runner.rhs, output, output, 1.0, 0.0)

    apply()
    apply_graph = None
    if w.device.is_cuda:
        with wp.ScopedCapture(device=w.device) as capture:
            for _ in range(20):
                apply()
        apply_graph = capture.graph
        event_start = wp.Event(w.device, enable_timing=True)
        event_end = wp.Event(w.device, enable_timing=True)
    samples = []
    for repeat in range(repeats):
        _, apply_ms = timed(apply, device)
        apply_gpu_ms = None
        if apply_graph is not None:
            wp.record_event(event_start)
            wp.capture_launch(apply_graph)
            wp.record_event(event_end)
            apply_gpu_ms = wp.get_event_elapsed_time(event_start, event_end) / 20
        sample = benchmark._timed_solve(runner, "new_generation")
        sample.update(
            repeat=repeat,
            apply_wall_ms=apply_ms,
            apply_graph_gpu_ms=apply_gpu_ms,
            accuracy=reference.measure(runner.y.numpy()),
        )
        samples.append(sample)
    # Observation uses a separate diagnostic workspace so host reads never enter graph timings.
    observer = benchmark.Runner(
        case, "diagnostic", device, config, study_hierarchy=None if candidate == "actor_block" else hierarchy
    )
    convergence = benchmark.observe_convergence(observer, reference)
    convergence.pop("iterate_y")
    block = w._study_soft
    arrays = (
        {}
        if block is None
        else {a.ptr: a for a in vars(block).values() if isinstance(a, wp.array) and a is not block.scale}
    )
    return {
        "hierarchy_sha256": hierarchy.fingerprint,
        "level_counts": hierarchy.level_counts,
        "hierarchy_ms": hierarchy_ms,
        "workspace_conversion_and_initial_factor_ms": setup_ms,
        "extra_preallocated_bytes": sum(a.capacity for a in arrays.values()),
        "cold": cold,
        "samples": samples,
        "reference": reference.describe(),
        "diagnostic_convergence": convergence,
        "gpu_state": benchmark._gpu_sample(w.device),
    }, runner.y.numpy()


def amg_case(case, config, repeats):
    from scripts.monolithic_reference.preconditioner_amg import AmgOracle  # noqa: PLC0415 - optional CPU research

    reference = benchmark.Reference(case, config)
    oracle, build_ms = timed(lambda: AmgOracle(reference, case.arrays))
    samples = []
    for repeat in range(repeats):
        _, update_ms = timed(lambda: oracle.update(reference))
        _, apply_ms = timed(lambda: oracle.apply(reference.rhs))
        result, pcg_ms = timed(lambda: study.solve_reference_pcg(reference, oracle.apply))
        y = result.pop("y")
        samples.append(
            {
                "repeat": repeat,
                "numeric_update_ms": update_ms,
                "apply_wall_ms": apply_ms,
                "pcg_wall_ms": pcg_ms,
                **result,
            }
        )
    return {
        "build_ms": build_ms,
        "samples": samples,
        "reference": reference.describe(),
        "candidate_rank": oracle.candidates.shape[1],
        "gpu": "NOT_MEASURED",
        "levels": [] if oracle.hierarchy is None else [level.A.shape[0] for level in oracle.hierarchy.levels],
        "cost_scope": "CPU FP64 research PCG including independent per-iteration observations",
    }, y


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--matrix-list", type=Path)
    parser.add_argument("--candidate", choices=(*study.CANDIDATES, "sa_amg_cpu"), required=True)
    parser.add_argument("--mode", choices=("diagnostic", "production_graph"), default="production_graph")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--linear-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    paths = args.snapshot + (
        [] if args.matrix_list is None else [Path(p) for p in json.loads(args.matrix_list.read_text())]
    )
    if not paths:
        raise ValueError("Provide authenticated matrix snapshots")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output / "provenance")
    config = benchmark.pcg_config(linear_absolute_tolerance=args.linear_absolute_tolerance)
    rows, solutions = [], {}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "warp-lang")}
    if args.candidate == "sa_amg_cpu":
        versions["pyamg"] = importlib.metadata.version("pyamg")
    for index, path in enumerate(paths):
        row = {
            "snapshot": str(path),
            "candidate": args.candidate,
            "mode": args.mode,
            "admission": "BUDGET_REVIEW_REQUIRED",
            "versions": versions,
        }
        try:
            case = benchmark.load_snapshot(path)
            row.update(case=case.name, input_sha256=array_fingerprint(case.arrays), source=case.metadata)
            if args.candidate == "sa_amg_cpu":
                result, solution = amg_case(case, config, args.repeats)
            else:
                result, solution = gpu_case(case, args.candidate, args.mode, config, args.repeats, args.device)
            row.update(result, solution_key=str(index))
            row["numerical_validation"] = (
                "PASS"
                if all(
                    sample["status"] == "SUCCESS" and sample["accuracy"]["independent_residual_pass"]
                    for sample in result["samples"]
                )
                else "FAIL"
            )
            solutions[str(index)] = solution
        except Exception as error:  # Retain a failed case and continue unattended research.
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        (args.output / "results.json").write_text(
            json.dumps(benchmark._json_safe(rows), indent=2, allow_nan=False) + "\n"
        )
        print(path.stem, args.candidate, row.get("error", "recorded"), flush=True)
    np.savez_compressed(args.output / "solutions.npz", **solutions)
    evidence_index(args.output)
    return int(any("error" in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
