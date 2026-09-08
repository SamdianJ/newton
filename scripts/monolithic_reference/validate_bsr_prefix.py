# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the production builder with a full-capacity oracle on Sharpa."""

import ctypes
import json
import time
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton.examples.softbody.example_monolithic_sharpa_soft_ball import Example
from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase
from scripts.monolithic_reference.profile_sharpa_bsr import _AssemblyProfile, _matrix_digest


def full_capacity_builder(workspace, original):
    """Keep the old builder input only in this offline oracle."""

    def build(matrix, rows, columns, values, **kwargs):
        for dest, buffer in (
            (workspace.k_global_scalar_bsr, workspace._global),
            (workspace.ax_internal_bsr3, workspace._internal),
        ):
            if matrix is dest:
                return original(matrix, buffer.rows, buffer.columns, buffer.values, **kwargs)
        return original(matrix, rows, columns, values, **kwargs)

    return build


def main():
    parser = Example.create_parser()
    parser.set_defaults(ball="scripts/monolithic_reference/fixtures/soft_ball/ball_r3.npz")
    parser.add_argument("--builder", choices=("production", "capacity"), default="production")
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--nsys-capture", action="store_true")
    args = parser.parse_args()
    if args.experiment != "anchored-close" or not 1 <= args.steps <= 4500 or args.samples < 1:
        parser.error("Require anchored-close, 1..4500 steps and positive samples")
    args.output.mkdir(parents=True, exist_ok=True)
    case = GraspCase(args)
    ws = case.solver._linear
    original = sparse.bsr_set_from_triplets
    context = (
        patch.object(sparse, "bsr_set_from_triplets", new=full_capacity_builder(ws, original))
        if args.builder == "capacity"
        else nullcontext()
    )
    with context:
        for _ in range(args.steps):
            case.step()
            if case.failure:
                raise RuntimeError(case.failure)
            if case.step_count % 500 == 0 or case.step_count == args.steps:
                history = case.solver._contact.committed
                np.savez(
                    args.output / f"history_{case.step_count}.npz",
                    valid=history.valid.numpy(),
                    xi=history.xi_local.numpy(),
                    normal=history.normal_local.numpy(),
                )
                print(f"{args.builder}: {case.step_count}/{args.steps}", flush=True)
        summary = case.save(args.output / "trajectory")
        report = {
            "builder": args.builder,
            "steps": args.steps,
            "summary": summary,
            "samples": args.samples,
            "trajectory_mempool_used_bytes": wp.get_mempool_used_mem_current(ws.device),
            "trajectory_mempool_high_bytes": wp.get_mempool_used_mem_high(ws.device),
        }
        for detail in (False, True):
            with _AssemblyProfile(ws, detail=detail) as timer:
                for _ in range(args.samples):
                    case.solver.step(case.state, case.state, case.control, None, case.fixture["dt"])
                    if case.solver.last_stats.rolled_back:
                        raise RuntimeError(case.solver.last_stats.failure_reason)
            report["detail" if detail else "baseline"] = timer.summary()
    matrices = ((ws.k_global_scalar_bsr, ws._global), (ws.ax_internal_bsr3, ws._internal))
    before = [_matrix_digest(matrix) for matrix, _ in matrices]
    inputs = {}
    for mode in ("capacity", "prefix"):
        inputs[mode] = []
        for matrix, buffer in matrices:
            n = int(buffer.count.numpy()[0]) if mode == "prefix" else buffer.capacity
            arrays = (buffer.rows, buffer.columns, buffer.values)
            if n < buffer.capacity:
                arrays = tuple(a[:n] for a in arrays)
            inputs[mode].append((matrix, arrays, buffer.count))
    report["matrices"] = [
        {**sig, "count": int(buffer.count.numpy()[0]), "capacity": buffer.capacity}
        for sig, (_, buffer) in zip(before, matrices, strict=True)
    ]
    report["replay_groups_ms"] = {mode: [] for mode in inputs}
    # Both policies consume the same frozen triplets, with alternating order.
    for group in range(6):
        for mode in ("capacity", "prefix") if group % 2 == 0 else ("prefix", "capacity"):
            samples = []
            for _ in range(args.samples):
                wp.synchronize_device(ws.device)
                start = time.perf_counter()
                for matrix, arrays, count in inputs[mode]:
                    original(matrix, *arrays, count=count)
                wp.synchronize_device(ws.device)
                samples.append((time.perf_counter() - start) * 1000.0)
            if group:
                report["replay_groups_ms"][mode].append(samples)
            if [_matrix_digest(matrix) for matrix, _ in matrices] != before:
                raise RuntimeError(f"{mode} changed the frozen matrix")
    report["replay_bitwise_equal"] = True
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.nsys_capture:
        nvtx = ctypes.CDLL("libnvToolsExt.so.1")
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        wp.cuda_profiler_start(ws.device)
        try:
            for mode in ("capacity", "prefix"):
                for _ in range(args.samples):
                    for label, (matrix, arrays, count) in zip(
                        ("global_scalar", "internal_block3"), inputs[mode], strict=True
                    ):
                        nvtx.nvtxRangePushA(f"{mode}/{label}".encode())
                        try:
                            original(matrix, *arrays, count=count)
                        finally:
                            nvtx.nvtxRangePop()
                    wp.synchronize_device(ws.device)
        finally:
            wp.cuda_profiler_stop(ws.device)
    if args.steps == 4500 and not summary["stability_passed"]:
        raise RuntimeError("Anchored stability gate failed")


if __name__ == "__main__":
    main()
