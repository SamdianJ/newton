# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile the existing r3 assembly after a continuous anchored closure.

Use Nsight Systems with --capture-range=cudaProfilerApi and --nsys-capture
for a separate, unsynchronized replay of the final two BSR builds. This replay
keeps the exact triplets and does not advance State or friction history.
"""

import ctypes
import hashlib
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import numpy as np
import warp as wp
import warp._src.sparse as sparse_impl
from warp import sparse

from newton.examples.softbody.example_monolithic_sharpa_soft_ball import Example
from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase


@wp.kernel
def _count_nonzero(values: wp.array[float], count: wp.array[int]):
    i = wp.tid()
    if values[i] != 0.0:
        wp.atomic_add(count, 0, 1)


class _AssemblyProfile:
    """Time nested calls without retaining array arguments or changing assembly."""

    def __init__(self, workspace, *, detail=True):
        self.workspace = workspace
        self.detail = detail
        self.samples = defaultdict(list)
        self.frames = []
        self.matrix_label = None
        self.stack = ExitStack()

    def timed(self, name, function, *, nested=False):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if nested and not self.frames:
                return function(*args, **kwargs)
            key = f"{self.matrix_label}/{name}" if self.matrix_label and name != self.matrix_label else name
            wp.synchronize_device(self.workspace.device)
            frame = [time.perf_counter(), 0.0]
            self.frames.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                wp.synchronize_device(self.workspace.device)
                elapsed = (time.perf_counter() - frame[0]) * 1000.0
                self.frames.pop()
                self.samples[key].append({"inclusive_ms": elapsed, "exclusive_ms": elapsed - frame[1]})
                if self.frames:
                    self.frames[-1][1] += elapsed

        return wrapped

    def __enter__(self):
        ws = self.workspace
        try:
            self.stack.enter_context(
                patch.object(ws, "finalize_assembly", new=self.timed("finalize", ws.finalize_assembly))
            )
            if not self.detail:
                return self
            original_build = sparse.bsr_set_from_triplets

            def build(dest, *args, **kwargs):
                name = "global_scalar" if dest is ws.k_global_scalar_bsr else "internal_block3"
                previous = self.matrix_label
                self.matrix_label = name
                try:
                    return self.timed(name, original_build)(dest, *args, **kwargs)
                finally:
                    self.matrix_label = previous

            self.stack.enter_context(patch.object(sparse, "bsr_set_from_triplets", new=build))
            for name in ("_bsr_set_from_triplets_native", "_bsr_ensure_fits", "_bsr_set_compact_row_counts"):
                self.stack.enter_context(
                    patch.object(sparse_impl, name, new=self.timed(name, getattr(sparse_impl, name), nested=True))
                )
            self.stack.enter_context(
                patch.object(wp, "empty", new=self.timed("python_scratch_allocation", wp.empty, nested=True))
            )
            original_launch = wp.launch

            def launch(kernel, *args, **kwargs):
                if self.frames:
                    return self.timed(kernel.key, original_launch)(kernel, *args, **kwargs)
                return original_launch(kernel, *args, **kwargs)

            self.stack.enter_context(patch.object(wp, "launch", new=launch))
            original_numpy = wp.array.numpy

            def numpy(array, *args, **kwargs):
                if self.frames:
                    return self.timed("host_readback", original_numpy)(array, *args, **kwargs)
                return original_numpy(array, *args, **kwargs)

            self.stack.enter_context(patch.object(wp.array, "numpy", new=numpy))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def summary(self):
        result = {}
        for name, samples in self.samples.items():
            result[name] = {"calls": len(samples)}
            for metric in ("inclusive_ms", "exclusive_ms"):
                values = [s[metric] for s in samples]
                result[name][metric] = {
                    "mean": float(np.mean(values)),
                    "p50": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "sum": float(np.sum(values)),
                }
        return result


def _matrix_digest(matrix):
    offsets = matrix.offsets.numpy()
    nnz = int(offsets[-1])
    digest = hashlib.sha256()
    for array in (offsets, matrix.columns[:nnz].numpy(), matrix.values[:nnz].numpy()):
        digest.update(array.tobytes())
    return {"rows": matrix.nrow, "nnz_blocks": nnz, "sha256": digest.hexdigest()}


def main():
    parser = Example.create_parser()
    parser.set_defaults(ball="scripts/monolithic_reference/fixtures/soft_ball/ball_r3.npz")
    parser.add_argument("--warmup-steps", type=int, default=4500)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--nsys-capture", action="store_true")
    args = parser.parse_args()
    if args.experiment != "anchored-close" or not 1 <= args.warmup_steps <= 4500 or args.samples < 1:
        parser.error("Require anchored-close, 1..4500 warmup steps, and positive samples")
    args.output.mkdir(parents=True, exist_ok=True)
    case = GraspCase(args)
    for _ in range(args.warmup_steps):
        case.step()
        if case.failure:
            raise RuntimeError(case.failure)
        if case.step_count % 500 == 0:
            print(f"Warmup {case.step_count}/{args.warmup_steps}", flush=True)
    case.save(args.output / "trajectory")
    ws = case.solver._linear
    report = {"warmup_steps": args.warmup_steps, "warp_version": wp.__version__, "continuous_tail": {}}
    # The target stays frozen at the last trajectory sample throughout this tail.
    for label, detail in (("baseline", False), ("detail", True)):
        with _AssemblyProfile(ws, detail=detail) as timer:
            for _ in range(args.samples):
                case.solver.step(case.state, case.state, case.control, None, case.fixture["dt"])
                if case.solver.last_stats.rolled_back:
                    raise RuntimeError(case.solver.last_stats.failure_reason)
        report["continuous_tail"][label] = timer.summary()
    matrices = ((ws.k_global_scalar_bsr, ws._global), (ws.ax_internal_bsr3, ws._internal))
    before = [_matrix_digest(m) for m, _ in matrices]
    report["matrices"] = []
    for (_matrix, triplets), signature in zip(matrices, before, strict=True):
        nonzero = wp.zeros(1, dtype=int, device=ws.device)
        values = triplets.values.view(dtype=wp.float32).flatten()
        wp.launch(_count_nonzero, values.size, [values, nonzero], device=ws.device)
        report["matrices"].append(
            {
                **signature,
                "triplet_count": int(triplets.count.numpy()[0]),
                "triplet_capacity": triplets.capacity,
                "nonzero_input_scalars": int(nonzero.numpy()[0]),
            }
        )
    # Replay isolates assembly from changing q/x/history. No new topology backend.
    report["replay_ms"] = []
    for _ in range(args.samples):
        wp.synchronize_device(ws.device)
        start = time.perf_counter()
        for matrix, triplets in matrices:
            sparse.bsr_set_from_triplets(matrix, triplets.rows, triplets.columns, triplets.values, count=triplets.count)
        wp.synchronize_device(ws.device)
        report["replay_ms"].append((time.perf_counter() - start) * 1000.0)
    if args.nsys_capture:
        # Native NVTX avoids a new Python dependency. Only needed for Nsight.
        nvtx = ctypes.CDLL("libnvToolsExt.so.1")
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        wp.cuda_profiler_start(ws.device)
        try:
            for _ in range(args.samples):
                for label, (matrix, triplets) in zip((b"global_scalar", b"internal_block3"), matrices, strict=True):
                    nvtx.nvtxRangePushA(label)
                    try:
                        sparse.bsr_set_from_triplets(
                            matrix, triplets.rows, triplets.columns, triplets.values, count=triplets.count
                        )
                    finally:
                        nvtx.nvtxRangePop()
            wp.synchronize_device(ws.device)
        finally:
            wp.cuda_profiler_stop(ws.device)
    report["replay_bitwise_equal"] = before == [_matrix_digest(m) for m, _ in matrices]
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["replay_bitwise_equal"]:
        raise RuntimeError("Assembly replay changed the matrix")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
