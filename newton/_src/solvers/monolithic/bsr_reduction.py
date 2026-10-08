# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed FP32 duplicate reduction for the monolithic scalar operator."""

from itertools import pairwise
from typing import Any

import numpy as np
import warp as wp
from warp import sparse
from warp.utils import radix_sort_pairs

from .contact_pool import _merge_contact_keys
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


@wp.kernel
def _fixed_bsr_begin(count: wp.array[int], incoming: wp.array[int], status: wp.array[int], capacity: int):
    status[0] = incoming[0]
    if count[0] < 0 or count[0] > capacity:
        wp.atomic_max(status, 0, int(MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW))


@wp.kernel
def _fixed_bsr_value_keys(
    rows: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    bits: wp.array[wp.uint32],
    count: wp.array[int],
    size: int,
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    key = wp.uint64(0) - wp.uint64(1)
    if count[0] >= 0 and count[0] <= keys.shape[0] and i < count[0]:
        if rows[i] < 0 or rows[i] >= size or columns[i] < 0 or columns[i] >= size:
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE))
        if not wp.isfinite(values[i]):
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))
        key = wp.uint64(bits[i])
    keys[i] = key
    indices[i] = i


@wp.kernel
def _fixed_bsr_position_keys(
    rows: wp.array[int],
    columns: wp.array[int],
    count: wp.array[int],
    size: int,
    indices: wp.array[int],
    keys: wp.array[wp.uint64],
):
    i = wp.tid()
    source = indices[i]
    key = wp.uint64(0) - wp.uint64(1)
    if count[0] >= 0 and count[0] <= keys.shape[0] and source < count[0]:
        if rows[source] >= 0 and rows[source] < size and columns[source] >= 0 and columns[source] < size:
            key = wp.uint64(rows[source]) * wp.uint64(size) + wp.uint64(columns[source])
    keys[i] = key


@wp.kernel
def _fixed_bsr_work(
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    values: wp.array[float],
    starts: wp.array[int],
    work: wp.array[float],
    live: wp.array[int],
):
    i = wp.tid()
    first = i
    value = float(0)
    if keys[i] != wp.uint64(0) - wp.uint64(1):
        first = _lower_bound(keys, keys[i], keys.shape[0])
        value = values[indices[i]]
        if value != 0.0:
            wp.atomic_or(live, first, 1)
    starts[i] = first
    work[i] = value


@wp.kernel
def _fixed_bsr_pair(
    keys: wp.array[wp.uint64], starts: wp.array[int], work: wp.array[float], stride: int, status: wp.array[int]
):
    i = wp.tid()
    if keys[i] != wp.uint64(0) - wp.uint64(1) and (i - starts[i]) % (2 * stride) == 0:
        right = i + stride
        if right < keys.shape[0]:
            if keys[right] == keys[i]:
                # Preserve the existing odd-tail carry, including zero inputs.
                value = work[i] + work[right]
                work[i] = value
                if not wp.isfinite(value):
                    wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))


@wp.kernel
def _fixed_bsr_scan_seed(starts: wp.array[int], live: wp.array[int], prefix: wp.array[int]):
    i = wp.tid()
    value = int(0)
    if i < starts.shape[0]:
        if starts[i] == i and live[i] != 0:
            value = 1
    prefix[i] = value


@wp.kernel
def _fixed_bsr_scan_up(prefix: wp.array[int], distance: int):
    right = (wp.tid() + 1) * 2 * distance - 1
    prefix[right] = prefix[right] + prefix[right - distance]


@wp.kernel
def _fixed_bsr_scan_root(prefix: wp.array[int]):
    prefix[prefix.shape[0] - 1] = 0


@wp.kernel
def _fixed_bsr_scan_down(prefix: wp.array[int], distance: int):
    right = (wp.tid() + 1) * 2 * distance - 1
    left = right - distance
    value = prefix[left]
    prefix[left] = prefix[right]
    prefix[right] = prefix[right] + value


@wp.kernel
def _fixed_bsr_offsets(keys: wp.array[wp.uint64], prefix: wp.array[int], size: int, offsets: wp.array[int]):
    row = wp.tid()
    first = _lower_bound(keys, wp.uint64(row) * wp.uint64(size), keys.shape[0])
    offsets[row] = prefix[first]


@wp.kernel
def _fixed_bsr_entries(
    keys: wp.array[wp.uint64],
    starts: wp.array[int],
    live: wp.array[int],
    work: wp.array[float],
    prefix: wp.array[int],
    size: int,
    columns: wp.array[int],
    values: wp.array[float],
):
    i = wp.tid()
    if starts[i] == i and live[i] != 0:
        entry = prefix[i]
        columns[entry] = int(keys[i] % wp.uint64(size))
        values[entry] = work[i]


@wp.kernel
def _fixed_bsr_clear_failed(values: wp.array[Any], status: wp.array[int]):
    if status[0] != 0:
        values[wp.tid()] = values.dtype(0)


@wp.kernel
def _fixed_bsr_publish_status(status: wp.array[int], incoming: wp.array[int]):
    wp.atomic_max(incoming, 0, status[0])


class _FixedScalarBsrReduction:
    """Build a fixed-capacity scalar matrix with the existing FP32 value tree.

    Device effective counts change contents, not graph bindings or allocation.
    All input zeros remain in the tree; only zero-only coordinates are pruned.
    No sparse topology builder, native sort/scan or floating atomic sum runs.
    Generation/token guards stay with the assembly workspace. This private
    component owns its compact matrix; nnz is a capacity bound until nnz_sync.
    """

    def __init__(self, size, capacity, device, *, use_cuda_graph):
        for name, value, maximum in (("size", size, 2**31 - 2), ("capacity", capacity, 2**30 - 1)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"Invalid fixed scalar BSR {name}")
        size, capacity = int(size), int(capacity)
        output_capacity = min(capacity, size * size)
        p = 1 << capacity.bit_length()  # Always has a terminal prefix[capacity].
        byte_count = 36 * capacity + 4 * p + 4 + 4 * (size + 1) + 8 * output_capacity
        if byte_count >= 2**63:
            raise ValueError("Fixed scalar BSR byte capacity exceeds int64")
        self.size, self.capacity, self.output_capacity = size, capacity, output_capacity
        self.device = wp.get_device(device)
        self._spec = dict.fromkeys(("keys", "key_scratch"), (capacity, wp.uint64))
        self._spec.update(dict.fromkeys(("indices", "index_scratch", "starts", "live"), (capacity, wp.int32)))
        self._spec.update(work=(capacity, wp.float32), prefix=(p, wp.int32), status=(1, wp.int32))
        for name, (length, scalar) in self._spec.items():
            setattr(self, name, wp.zeros(length, dtype=scalar, device=self.device))
        self.matrix = sparse.bsr_zeros(size, size, wp.float32, device=self.device, nnz_capacity=output_capacity)
        # Construction also establishes Warp's fixed pinned nnz readback/event.
        self.matrix.notify_nnz_changed(nnz=output_capacity, nnz_capacity=output_capacity)
        self.payload_bytes = byte_count
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)

    def buffers(self):
        return (
            *(getattr(self, name) for name in self._spec),
            self.matrix.offsets,
            self.matrix.columns,
            self.matrix.values,
        )

    def build(self, triplets, *, layout_identity=""):
        matrix = self.matrix
        if matrix.nrow != self.size or matrix.ncol != self.size or matrix.row_counts is not None:
            raise ValueError("Invalid fixed scalar BSR matrix layout")
        arrays = list(self.buffers())
        specs = [
            *self._spec.values(),
            (self.size + 1, wp.int32),
            (self.output_capacity, wp.int32),
            (self.output_capacity, wp.float32),
        ]
        for name, spec in (
            ("rows", (self.capacity, wp.int32)),
            ("columns", (self.capacity, wp.int32)),
            ("values", (self.capacity, wp.float32)),
            ("count", (1, wp.int32)),
            ("status", (1, wp.int32)),
        ):
            arrays.append(getattr(triplets, name))
            specs.append(spec)
        for array, (length, scalar) in zip(arrays, specs, strict=True):
            if (
                array.shape != (length,)
                or array.dtype != scalar
                or array.device != self.device
                or not array.is_contiguous
            ):
                raise ValueError("Invalid fixed scalar BSR storage")
        intervals = sorted((a.ptr, a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)) for a in arrays if a.size)
        if any(right[0] < left[1] for left, right in pairwise(intervals)):
            raise ValueError("Fixed scalar BSR storage aliases")
        # The upper bound is fixed even if a caller read the last exact nnz.
        matrix.nnz = self.output_capacity
        self.graphs.run(
            "fixed_scalar_bsr",
            (self.size, self.capacity, layout_identity),
            arrays,
            lambda: self._build(triplets),
        )

    def _sort(self, keys, indices):
        out_keys, out_indices = self.key_scratch, self.index_scratch
        if keys.ptr == out_keys.ptr:
            out_keys, out_indices = self.keys, self.indices
        run = 1
        while run < self.capacity:
            wp.launch(
                _merge_contact_keys,
                self.capacity,
                [keys, indices, out_keys, out_indices, run],
                device=self.device,
            )
            keys, out_keys = out_keys, keys
            indices, out_indices = out_indices, indices
            run *= 2
        return keys, indices

    def _build(self, triplets):
        matrix, n = self.matrix, self.capacity
        for array in (self.live, matrix.offsets, matrix.columns, matrix.values):
            array.zero_()
        wp.launch(_fixed_bsr_begin, 1, [triplets.count, triplets.status, self.status, n], device=self.device)
        keys, indices = self.keys, self.indices
        if n:
            wp.launch(
                _fixed_bsr_value_keys,
                n,
                [
                    triplets.rows,
                    triplets.columns,
                    triplets.values,
                    triplets.values.view(wp.uint32),
                    triplets.count,
                    self.size,
                    keys,
                    indices,
                    self.status,
                ],
                device=self.device,
            )
            keys, indices = self._sort(keys, indices)
            wp.launch(
                _fixed_bsr_position_keys,
                n,
                [triplets.rows, triplets.columns, triplets.count, self.size, indices, keys],
                device=self.device,
            )
            keys, indices = self._sort(keys, indices)
            wp.launch(
                _fixed_bsr_work,
                n,
                [keys, indices, triplets.values, self.starts, self.work, self.live],
                device=self.device,
            )
            stride = 1
            while stride < n:
                wp.launch(_fixed_bsr_pair, n, [keys, self.starts, self.work, stride, self.status], device=self.device)
                stride *= 2
        wp.launch(_fixed_bsr_scan_seed, self.prefix.size, [self.starts, self.live, self.prefix], device=self.device)
        distance = 1
        while distance < self.prefix.size:
            wp.launch(
                _fixed_bsr_scan_up, self.prefix.size // (2 * distance), [self.prefix, distance], device=self.device
            )
            distance *= 2
        wp.launch(_fixed_bsr_scan_root, 1, [self.prefix], device=self.device)
        distance //= 2
        while distance:
            wp.launch(
                _fixed_bsr_scan_down, self.prefix.size // (2 * distance), [self.prefix, distance], device=self.device
            )
            distance //= 2
        wp.launch(_fixed_bsr_offsets, self.size + 1, [keys, self.prefix, self.size, matrix.offsets], device=self.device)
        if n:
            wp.launch(
                _fixed_bsr_entries,
                n,
                [keys, self.starts, self.live, self.work, self.prefix, self.size, matrix.columns, matrix.values],
                device=self.device,
            )
        for array in (matrix.offsets, matrix.columns, matrix.values):
            if array.size:
                wp.launch(_fixed_bsr_clear_failed, array.size, [array, self.status], device=self.device)
        wp.launch(_fixed_bsr_publish_status, 1, [self.status, triplets.status], device=self.device)
        matrix.notify_nnz_changed(nnz=self.output_capacity, nnz_capacity=self.output_capacity)
