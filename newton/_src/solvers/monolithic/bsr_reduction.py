# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed FP32 duplicate reduction for the monolithic scalar operator."""

from itertools import pairwise
from typing import Any

import numpy as np
import warp as wp
from warp import sparse
from warp.utils import radix_sort_pairs

from .contact_pool import _CONTACT_WORKERS, _merge_contact_keys
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
def _fixed_bsr_begin(
    count: wp.array[int], incoming: wp.array[int], status: wp.array[int], scan_size: wp.array[int], capacity: int
):
    status[0] = incoming[0]
    span = int(1)
    if count[0] < 0 or count[0] > capacity:
        wp.atomic_max(status, 0, int(MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW))
    else:
        # Include the terminal prefix[count] without reading count on the host.
        while span <= count[0]:
            span *= 2
    scan_size[0] = span


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
    live: wp.array[int],
    workers: int,
    status: wp.array[int],
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0]:
        return
    while i < n:
        if rows[i] < 0 or rows[i] >= size or columns[i] < 0 or columns[i] >= size:
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE))
        if not wp.isfinite(values[i]):
            wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))
        keys[i] = wp.uint64(bits[i])
        indices[i] = i
        live[i] = 0
        i += workers


@wp.kernel
def _fixed_bsr_position_keys(
    rows: wp.array[int],
    columns: wp.array[int],
    count: wp.array[int],
    size: int,
    indices: wp.array[int],
    keys: wp.array[wp.uint64],
    workers: int,
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0]:
        return
    while i < n:
        key = wp.uint64(0) - wp.uint64(1)
        source = indices[i]
        if source >= 0 and source < n:
            if rows[source] >= 0 and rows[source] < size and columns[source] >= 0 and columns[source] < size:
                key = wp.uint64(rows[source]) * wp.uint64(size) + wp.uint64(columns[source])
        keys[i] = key
        i += workers


@wp.kernel
def _fixed_bsr_work(
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    values: wp.array[float],
    starts: wp.array[int],
    work: wp.array[float],
    live: wp.array[int],
    count: wp.array[int],
    workers: int,
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0]:
        return
    while i < n:
        first = i
        value = float(0)
        if keys[i] != wp.uint64(0) - wp.uint64(1):
            first = _lower_bound(keys, keys[i], n)
            value = values[indices[i]]
            if value != 0.0:
                wp.atomic_or(live, first, 1)
        starts[i] = first
        work[i] = value
        i += workers


@wp.kernel
def _fixed_bsr_pair(
    keys: wp.array[wp.uint64],
    starts: wp.array[int],
    work: wp.array[float],
    count: wp.array[int],
    stride: int,
    workers: int,
    status: wp.array[int],
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0] or stride >= n:
        return
    while i < n:
        if keys[i] != wp.uint64(0) - wp.uint64(1) and (i - starts[i]) % (2 * stride) == 0:
            right = i + stride
            if right < n:
                if keys[right] == keys[i]:
                    # Same disjoint pairs and odd carries at every tree level.
                    value = work[i] + work[right]
                    work[i] = value
                    if not wp.isfinite(value):
                        wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))
        i += workers


@wp.kernel
def _fixed_bsr_scan_seed(
    starts: wp.array[int],
    live: wp.array[int],
    prefix: wp.array[int],
    count: wp.array[int],
    scan_size: wp.array[int],
    workers: int,
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > starts.shape[0]:
        n = 0
    while i < scan_size[0]:
        value = int(0)
        if i < n:
            if starts[i] == i and live[i] != 0:
                value = 1
        prefix[i] = value
        i += workers


@wp.kernel
def _fixed_bsr_scan_up(prefix: wp.array[int], scan_size: wp.array[int], fixed_size: int, distance: int, workers: int):
    span = fixed_size
    if span == 0:
        span = scan_size[0]
    right = (wp.tid() + 1) * 2 * distance - 1
    while right < span:
        prefix[right] = prefix[right] + prefix[right - distance]
        right += workers * 2 * distance


@wp.kernel
def _fixed_bsr_scan_root(prefix: wp.array[int], scan_size: wp.array[int], fixed_size: int):
    span = fixed_size
    if span == 0:
        span = scan_size[0]
    prefix[span - 1] = 0


@wp.kernel
def _fixed_bsr_scan_down(prefix: wp.array[int], scan_size: wp.array[int], fixed_size: int, distance: int, workers: int):
    span = fixed_size
    if span == 0:
        span = scan_size[0]
    right = (wp.tid() + 1) * 2 * distance - 1
    while right < span:
        left = right - distance
        value = prefix[left]
        prefix[left] = prefix[right]
        prefix[right] = prefix[right] + value
        right += workers * 2 * distance


@wp.kernel
def _fixed_bsr_offsets(
    keys: wp.array[wp.uint64], prefix: wp.array[int], count: wp.array[int], size: int, offsets: wp.array[int]
):
    row = wp.tid()
    offsets[row] = 0
    if count[0] >= 0 and count[0] <= keys.shape[0]:
        first = _lower_bound(keys, wp.uint64(row) * wp.uint64(size), count[0])
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
    count: wp.array[int],
    workers: int,
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0]:
        return
    while i < n:
        if starts[i] == i and live[i] != 0:
            entry = prefix[i]
            columns[entry] = int(keys[i] % wp.uint64(size))
            values[entry] = work[i]
        i += workers


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

    MAX_WORKERS = _CONTACT_WORKERS

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
        byte_count = 36 * capacity + 4 * p + 8 + 4 * (size + 1) + 8 * output_capacity
        if byte_count >= 2**63:
            raise ValueError("Fixed scalar BSR byte capacity exceeds int64")
        self.size, self.capacity, self.output_capacity = size, capacity, output_capacity
        self.device = wp.get_device(device)
        self._spec = dict.fromkeys(("keys", "key_scratch"), (capacity, wp.uint64))
        self._spec.update(dict.fromkeys(("indices", "index_scratch", "starts", "live"), (capacity, wp.int32)))
        self._spec.update(
            work=(capacity, wp.float32), prefix=(p, wp.int32), scan_size=(1, wp.int32), status=(1, wp.int32)
        )
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

    def _sort(self, keys, indices, count):
        out_keys, out_indices = self.key_scratch, self.index_scratch
        if keys.ptr == out_keys.ptr:
            out_keys, out_indices = self.keys, self.indices
        run = 1
        workers = min(self.capacity, self.MAX_WORKERS)
        while run < self.capacity:
            wp.launch(
                _merge_contact_keys,
                workers,
                [keys, indices, out_keys, out_indices, count, run, workers],
                device=self.device,
            )
            keys, out_keys = out_keys, keys
            indices, out_indices = out_indices, indices
            run *= 2
        return keys, indices

    def _build(self, triplets):
        matrix, n = self.matrix, self.capacity
        for array in (matrix.offsets, matrix.columns, matrix.values):
            array.zero_()
        wp.launch(
            _fixed_bsr_begin, 1, [triplets.count, triplets.status, self.status, self.scan_size, n], device=self.device
        )
        keys, indices = self.keys, self.indices
        workers = min(n, self.MAX_WORKERS)
        if n:
            wp.launch(
                _fixed_bsr_value_keys,
                workers,
                [
                    triplets.rows,
                    triplets.columns,
                    triplets.values,
                    triplets.values.view(wp.uint32),
                    triplets.count,
                    self.size,
                    keys,
                    indices,
                    self.live,
                    workers,
                    self.status,
                ],
                device=self.device,
            )
            keys, indices = self._sort(keys, indices, triplets.count)
            wp.launch(
                _fixed_bsr_position_keys,
                workers,
                [triplets.rows, triplets.columns, triplets.count, self.size, indices, keys, workers],
                device=self.device,
            )
            keys, indices = self._sort(keys, indices, triplets.count)
            wp.launch(
                _fixed_bsr_work,
                workers,
                [keys, indices, triplets.values, self.starts, self.work, self.live, triplets.count, workers],
                device=self.device,
            )
            stride = 1
            while stride < n:
                wp.launch(
                    _fixed_bsr_pair,
                    workers,
                    [keys, self.starts, self.work, triplets.count, stride, workers, self.status],
                    device=self.device,
                )
                stride *= 2
        workers = min(self.prefix.size, self.MAX_WORKERS)
        wp.launch(
            _fixed_bsr_scan_seed,
            workers,
            [self.starts, self.live, self.prefix, triplets.count, self.scan_size, workers],
            device=self.device,
        )
        distance = 1
        while distance < self.prefix.size:
            workers = min(self.prefix.size // (2 * distance), self.MAX_WORKERS)
            wp.launch(
                _fixed_bsr_scan_up,
                workers,
                [self.prefix, self.scan_size, 0, distance, workers],
                device=self.device,
            )
            distance *= 2
        wp.launch(_fixed_bsr_scan_root, 1, [self.prefix, self.scan_size, 0], device=self.device)
        distance //= 2
        while distance:
            workers = min(self.prefix.size // (2 * distance), self.MAX_WORKERS)
            wp.launch(
                _fixed_bsr_scan_down,
                workers,
                [self.prefix, self.scan_size, 0, distance, workers],
                device=self.device,
            )
            distance //= 2
        wp.launch(
            _fixed_bsr_offsets,
            self.size + 1,
            [keys, self.prefix, triplets.count, self.size, matrix.offsets],
            device=self.device,
        )
        if n:
            workers = min(n, self.MAX_WORKERS)
            wp.launch(
                _fixed_bsr_entries,
                workers,
                [
                    keys,
                    self.starts,
                    self.live,
                    self.work,
                    self.prefix,
                    self.size,
                    matrix.columns,
                    matrix.values,
                    triplets.count,
                    workers,
                ],
                device=self.device,
            )
        for array in (matrix.offsets, matrix.columns, matrix.values):
            if array.size:
                wp.launch(_fixed_bsr_clear_failed, array.size, [array, self.status], device=self.device)
        wp.launch(_fixed_bsr_publish_status, 1, [self.status, triplets.status], device=self.device)
        matrix.notify_nnz_changed(nnz=self.output_capacity, nnz_capacity=self.output_capacity)


@wp.kernel
def _fixed_internal_begin(count: wp.array[int], incoming: wp.array[int], status: wp.array[int], capacity: int):
    status[0] = incoming[0]
    if count[0] != capacity:
        wp.atomic_max(status, 0, int(MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW))


@wp.kernel
def _fixed_internal_validate(
    rows: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    count: wp.array[int],
    expected_rows: wp.array[int],
    expected_columns: wp.array[int],
    active: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    active[i] = 0
    if count[0] != active.shape[0]:
        return
    if rows[i] != expected_rows[i] or columns[i] != expected_columns[i]:
        wp.atomic_max(status, 0, int(MonolithicLinearStatus.INVALID_ARGUMENT))
    nonzero = False
    for row in range(3):
        for column in range(3):
            value = values[i][row, column]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))
            nonzero = nonzero or value != 0.0
    if nonzero:
        active[i] = 1


@wp.kernel
def _fixed_internal_values(
    inputs: wp.array[wp.mat33],
    indices: wp.array[int],
    offsets: wp.array[int],
    active: wp.array[int],
    live: wp.array[int],
    work: wp.array3d[float],
    status: wp.array[int],
):
    group, row, column = wp.tid()
    value = float(0)
    found = int(0)
    for slot in range(offsets[group], offsets[group + 1]):
        source = indices[slot]
        if active[source] != 0:
            term = inputs[source][row, column]
            if not found:
                # Native GPU builder initializes from the first nonzero block;
                # zero components within that block must still participate.
                value = term
                found = 1
            else:
                value += term
                if not wp.isfinite(value):
                    wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))
    work[group, row, column] = value
    if row == 0 and column == 0:
        live[group] = wp.where(found, 1, 0)


@wp.kernel
def _fixed_internal_seed(live: wp.array[int], prefix: wp.array[int]):
    i = wp.tid()
    value = int(0)
    if i < live.shape[0]:
        value = live[i]
    prefix[i] = value


@wp.kernel
def _fixed_internal_offsets(row_groups: wp.array[int], prefix: wp.array[int], offsets: wp.array[int]):
    row = wp.tid()
    offsets[row] = prefix[row_groups[row]]


@wp.kernel
def _fixed_internal_entries(
    live: wp.array[int],
    prefix: wp.array[int],
    columns: wp.array[int],
    work: wp.array[wp.mat33],
    out_columns: wp.array[int],
    out_values: wp.array[wp.mat33],
):
    i = wp.tid()
    if live[i] != 0:
        entry = prefix[i]
        out_columns[entry] = columns[i]
        out_values[entry] = work[i]


@wp.kernel
def _fixed_internal_finite(offsets: wp.array[int], values: wp.array[wp.mat33], status: wp.array[int]):
    row = wp.tid()
    for entry in range(offsets[row], offsets[row + 1]):
        for i in range(3):
            for j in range(3):
                if not wp.isfinite(values[entry][i, j]):
                    wp.atomic_max(status, 0, int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))


class _FixedInternalBsrReduction:
    """Reserve immutable owner-block groups and preserve native backend arithmetic.

    CUDA's stable coordinate sort followed by serial nonzero-block addition is
    reproduced from original slot order. CPU keeps the installed native builder:
    its std::sort ties are not stable and cannot be silently replaced by a tree.
    GPU evaluation uses fixed arrays and device counts, with no native builder.
    This component is only for the complete static tet owner pattern, not contact.
    """

    def __init__(self, size, rows, columns, device, *, use_cuda_graph):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or not 0 <= size <= 2**31 - 2:
            raise ValueError("Invalid fixed internal BSR size")
        rows, columns = np.asarray(rows), np.asarray(columns)
        if (
            rows.ndim != 1
            or columns.shape != rows.shape
            or not np.issubdtype(rows.dtype, np.integer)
            or not np.issubdtype(columns.dtype, np.integer)
            or rows.size >= 2**30
            or 9 * rows.size >= 2**31
            or np.any(rows < 0)
            or np.any(rows >= size)
            or np.any(columns < 0)
            or np.any(columns >= size)
        ):
            raise ValueError("Invalid fixed internal BSR pattern")
        rows, columns = rows.astype(np.int32), columns.astype(np.int32)
        capacity = len(rows)
        keys = rows.astype(np.uint64) * int(size) + columns.astype(np.uint64)
        order = np.argsort(keys, kind="stable").astype(np.int32)
        starts = (
            np.r_[0, np.flatnonzero(np.diff(keys[order]) != 0) + 1].astype(np.int32)
            if capacity
            else np.empty(0, np.int32)
        )
        group_rows, group_columns = rows[order[starts]], columns[order[starts]]
        groups = len(starts)
        row_groups = np.r_[0, np.cumsum(np.bincount(group_rows, minlength=size))].astype(np.int32)
        p = 1 << groups.bit_length()
        self.device = wp.get_device(device)
        self.size, self.capacity, self.group_count = int(size), capacity, groups
        self.output_capacity = groups if self.device.is_cuda else capacity
        byte_count = 16 * capacity + 48 * groups + 8 * (int(size) + 1) + 4 * p + 8 + 40 * self.output_capacity
        if byte_count >= 2**63:
            raise ValueError("Fixed internal BSR byte capacity exceeds int64")
        self._spec = dict.fromkeys(("expected_rows", "expected_columns", "indices", "active"), (capacity, wp.int32))
        self._spec.update(dict.fromkeys(("group_columns", "live"), (groups, wp.int32)))
        self._spec.update(
            group_offsets=(groups + 1, wp.int32),
            row_groups=(int(size) + 1, wp.int32),
            work=(groups, wp.mat33),
            prefix=(p, wp.int32),
            status=(1, wp.int32),
        )
        for name, (length, scalar) in self._spec.items():
            setattr(self, name, wp.zeros(length, dtype=scalar, device=self.device))
        for name, data in (
            ("expected_rows", rows),
            ("expected_columns", columns),
            ("indices", order),
            ("group_columns", group_columns),
            ("group_offsets", np.r_[starts, capacity].astype(np.int32)),
            ("row_groups", row_groups),
        ):
            getattr(self, name).assign(data)
        self.matrix = sparse.bsr_zeros(size, size, wp.mat33, device=self.device, nnz_capacity=self.output_capacity)
        self.matrix.notify_nnz_changed(nnz=self.output_capacity, nnz_capacity=self.output_capacity)
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)
        self.payload_bytes = byte_count

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
            raise ValueError("Invalid fixed internal BSR matrix layout")
        arrays = list(self.buffers())
        specs = [
            *self._spec.values(),
            (self.size + 1, wp.int32),
            (self.output_capacity, wp.int32),
            (self.output_capacity, wp.mat33),
        ]
        for name, scalar, length in (
            ("rows", wp.int32, self.capacity),
            ("columns", wp.int32, self.capacity),
            ("values", wp.mat33, self.capacity),
            ("count", wp.int32, 1),
            ("status", wp.int32, 1),
        ):
            arrays.append(getattr(triplets, name))
            specs.append((length, scalar))
        for array, (length, scalar) in zip(arrays, specs, strict=True):
            if (
                array.shape != (length,)
                or array.dtype != scalar
                or array.device != self.device
                or not array.is_contiguous
            ):
                raise ValueError("Invalid fixed internal BSR storage")
        intervals = sorted((a.ptr, a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)) for a in arrays if a.size)
        if any(right[0] < left[1] for left, right in pairwise(intervals)):
            raise ValueError("Fixed internal BSR storage aliases")
        matrix.nnz = self.output_capacity
        self.graphs.run(
            "fixed_internal_bsr", (self.size, self.capacity, layout_identity), arrays, lambda: self._build(triplets)
        )

    def _build(self, triplets):
        matrix = self.matrix
        for array in (matrix.offsets, matrix.columns, matrix.values):
            array.zero_()
        wp.launch(
            _fixed_internal_begin, 1, [triplets.count, triplets.status, self.status, self.capacity], device=self.device
        )
        if self.capacity:
            wp.launch(
                _fixed_internal_validate,
                self.capacity,
                [
                    triplets.rows,
                    triplets.columns,
                    triplets.values,
                    triplets.count,
                    self.expected_rows,
                    self.expected_columns,
                    self.active,
                    self.status,
                ],
                device=self.device,
            )
        if self.device.is_cpu:
            if self.status.numpy()[0] == 0:
                # Preserve native CPU std::sort and numerical-zero pruning.
                sparse.bsr_set_from_triplets(
                    matrix, triplets.rows, triplets.columns, triplets.values, count=triplets.count
                )
                if self.size:
                    wp.launch(
                        _fixed_internal_finite,
                        self.size,
                        [matrix.offsets, matrix.values, self.status],
                        device=self.device,
                    )
        else:
            if self.group_count:
                wp.launch(
                    _fixed_internal_values,
                    (self.group_count, 3, 3),
                    [
                        triplets.values,
                        self.indices,
                        self.group_offsets,
                        self.active,
                        self.live,
                        wp.array(
                            ptr=self.work.ptr,
                            capacity=self.work.capacity,
                            dtype=wp.float32,
                            shape=(self.group_count, 3, 3),
                            device=self.device,
                        ),
                        self.status,
                    ],
                    device=self.device,
                )
            wp.launch(_fixed_internal_seed, self.prefix.size, [self.live, self.prefix], device=self.device)
            distance = 1
            while distance < self.prefix.size:
                workers = self.prefix.size // (2 * distance)
                wp.launch(
                    _fixed_bsr_scan_up,
                    workers,
                    [self.prefix, triplets.count, self.prefix.size, distance, workers],
                    device=self.device,
                )
                distance *= 2
            wp.launch(_fixed_bsr_scan_root, 1, [self.prefix, triplets.count, self.prefix.size], device=self.device)
            distance //= 2
            while distance:
                workers = self.prefix.size // (2 * distance)
                wp.launch(
                    _fixed_bsr_scan_down,
                    workers,
                    [self.prefix, triplets.count, self.prefix.size, distance, workers],
                    device=self.device,
                )
                distance //= 2
            wp.launch(
                _fixed_internal_offsets,
                self.size + 1,
                [self.row_groups, self.prefix, matrix.offsets],
                device=self.device,
            )
            if self.group_count:
                wp.launch(
                    _fixed_internal_entries,
                    self.group_count,
                    [self.live, self.prefix, self.group_columns, self.work, matrix.columns, matrix.values],
                    device=self.device,
                )
        for array in (matrix.offsets, matrix.columns, matrix.values):
            if array.size:
                wp.launch(_fixed_bsr_clear_failed, array.size, [array, self.status], device=self.device)
        wp.launch(_fixed_bsr_publish_status, 1, [self.status, triplets.status], device=self.device)
        matrix.notify_nnz_changed(nnz=self.output_capacity, nnz_capacity=self.output_capacity)
