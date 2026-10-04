# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed FP32 duplicate reduction for the monolithic scalar operator."""

import warp as wp
from warp import sparse
from warp.utils import radix_sort_pairs

from .linear_types import MonolithicLinearStatus
from .segment_graph import _SegmentGraphCache


@wp.kernel
def _topology_rows(rows: wp.array[int], values: wp.array[float], masked: wp.array[int]):
    i = wp.tid()
    # Preserve Warp's zero-input pruning without performing its value reduction.
    masked[i] = wp.where(values[i] == 0.0, -1, rows[i])


@wp.kernel
def _value_keys(bits: wp.array[wp.uint32], keys: wp.array[wp.uint32], indices: wp.array[int]):
    i = wp.tid()
    keys[i] = bits[i]
    indices[i] = i


@wp.kernel
def _position_keys(
    rows: wp.array[int], columns: wp.array[int], indices: wp.array[int], size: int, keys: wp.array[wp.uint64]
):
    i = wp.tid()
    source = indices[i]
    keys[i] = wp.uint64(rows[source]) * wp.uint64(size) + wp.uint64(columns[source])


@wp.func
def _lower_bound(keys: wp.array[wp.uint64], key: wp.uint64, count: int):
    lo = int(0)
    hi = count
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def _gather(
    values: wp.array[float],
    indices: wp.array[int],
    keys: wp.array[wp.uint64],
    count: int,
    starts: wp.array[int],
    work: wp.array[float],
):
    i = wp.tid()
    starts[i] = _lower_bound(keys, keys[i], count)
    work[i] = values[indices[i]]


@wp.kernel
def _pair_level(keys: wp.array[wp.uint64], starts: wp.array[int], work: wp.array[float], stride: int, count: int):
    i = wp.tid()
    if (i - starts[i]) % (2 * stride) == 0:
        right = i + stride
        if right < count:
            if keys[right] == keys[i]:
                # Right operands are not written at this level; odd tails carry.
                work[i] = work[i] + work[right]


@wp.kernel
def _write_bsr(
    keys: wp.array[wp.uint64],
    work: wp.array[float],
    count: int,
    size: int,
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    status: wp.array[int],
):
    row = wp.tid()
    for entry in range(offsets[row], offsets[row + 1]):
        key = wp.uint64(row) * wp.uint64(size) + wp.uint64(columns[entry])
        first = _lower_bound(keys, key, count)
        if first >= count:
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.BSR_BUILD_FAILURE))
        elif keys[first] != key:
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.BSR_BUILD_FAILURE))
        else:
            values[entry] = work[first]
            if not wp.isfinite(work[first]):
                wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))


class _ScalarBsrReduction:
    """Own scratch for validated scalar triplets; leave generation guards to the workspace."""

    def __init__(self, capacity, device, *, use_cuda_graph):
        self.device = device
        capacity = max(1, capacity)
        self.rows = wp.empty(capacity, dtype=int, device=device)
        self.bits = wp.empty(2 * capacity, dtype=wp.uint32, device=device)
        self.keys = wp.empty(2 * capacity, dtype=wp.uint64, device=device)
        self.indices = wp.empty(2 * capacity, dtype=int, device=device)
        self.starts = wp.empty(capacity, dtype=int, device=device)
        self.work = wp.empty(capacity, dtype=float, device=device)
        self.graphs = _SegmentGraphCache(device, enabled=use_cuda_graph)

    def build(self, matrix, triplets, count, *, layout_identity, graphs):
        if count == 0:
            sparse.bsr_set_zero(matrix)
            return
        wp.launch(_topology_rows, count, [triplets.rows, triplets.values, self.rows], device=self.device)
        # Negative rows are ignored by Warp's topology builder. Zero-only entries
        # disappear, while cancellation of nonzero inputs retains its stored zero.
        sparse.bsr_set_from_triplets(matrix, self.rows[:count], triplets.columns[:count], count=triplets.count)
        buffers = [
            triplets.rows,
            triplets.columns,
            triplets.values,
            triplets.status,
            self.bits,
            self.keys,
            self.indices,
            self.starts,
            self.work,
            matrix.offsets,
            matrix.columns,
            matrix.values,
        ]
        cache = self.graphs if graphs is None else graphs
        cache.run(
            "scalar_reduction",
            (layout_identity, count, matrix.nrow),
            buffers,
            lambda: self._reduce(matrix, triplets, count),
        )

    def _reduce(self, matrix, triplets, count):
        wp.launch(_value_keys, count, [triplets.values.view(wp.uint32), self.bits, self.indices], device=self.device)
        # Stable least-significant-key-first sorts reproduce (row, column, bits)
        # without squeezing the coordinate key and 32 value bits into one word.
        radix_sort_pairs(self.bits, self.indices, count)
        wp.launch(
            _position_keys,
            count,
            [triplets.rows, triplets.columns, self.indices, matrix.nrow, self.keys],
            device=self.device,
        )
        radix_sort_pairs(self.keys, self.indices, count)
        wp.launch(
            _gather,
            count,
            [triplets.values, self.indices, self.keys, count, self.starts, self.work],
            device=self.device,
        )
        stride = 1
        while stride < count:
            wp.launch(_pair_level, count, [self.keys, self.starts, self.work, stride, count], device=self.device)
            stride *= 2
        wp.launch(
            _write_bsr,
            matrix.nrow,
            [self.keys, self.work, count, matrix.nrow, matrix.offsets, matrix.columns, matrix.values, triplets.status],
            device=self.device,
        )
