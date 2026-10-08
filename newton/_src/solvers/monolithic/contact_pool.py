# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed storage and native row trees for contact contributions."""

from typing import Any

import numpy as np
import warp as wp

from .segment_graph import _SegmentGraphCache


@wp.kernel
def _reset_rows(count: wp.array[int], capacity: int, flags: wp.array[int]):
    flags[0] = wp.where(count[0] < 0 or count[0] > capacity, 1, 0)


@wp.kernel
def _contact_keys(
    rows: wp.array[int],
    identities: wp.array[wp.uint64],
    local_ids: wp.array[int],
    values: wp.array[Any],
    count: wp.array[int],
    row_count: int,
    identity_count: wp.uint64,
    local_count: int,
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    flags: wp.array[int],
):
    i = wp.tid()
    key = wp.uint64(0) - wp.uint64(1)
    if i < count[0]:
        valid = (
            rows[i] >= 0
            and rows[i] < row_count
            and identities[i] < identity_count
            and local_ids[i] >= 0
            and local_ids[i] < local_count
        )
        if not valid:
            wp.atomic_or(flags, 0, 2)
        if not wp.isfinite(values[i]):
            wp.atomic_or(flags, 0, 4)
        if valid:
            key = (wp.uint64(rows[i]) * identity_count + identities[i]) * wp.uint64(local_count)
            key += wp.uint64(local_ids[i])
    keys[i] = key
    indices[i] = i


@wp.kernel
def _merge_contact_keys(
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    out_keys: wp.array[wp.uint64],
    out_indices: wp.array[int],
    run: int,
):
    i = wp.tid()
    n = keys.shape[0]
    # int64 avoids 2*run overflowing at the last int32-capacity merge.
    base = int((wp.int64(i) // (wp.int64(2) * wp.int64(run))) * (wp.int64(2) * wp.int64(run)))
    middle = int(wp.min(wp.int64(base) + wp.int64(run), wp.int64(n)))
    end = int(wp.min(wp.int64(base) + wp.int64(2) * wp.int64(run), wp.int64(n)))
    left = i < middle
    lo = middle
    hi = end
    own_begin = base
    if not left:
        lo = base
        hi = middle
        own_begin = middle
    first = lo
    key = keys[i]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        advance = keys[mid] < key
        if not left:
            # Keep the left run first for equal keys, including inactive tails.
            advance = keys[mid] <= key
        if advance:
            lo = mid + 1
        else:
            hi = mid
    destination = base + (i - own_begin) + (lo - first)
    out_keys[destination] = key
    out_indices[destination] = indices[i]


@wp.kernel
def _contact_offsets(keys: wp.array[wp.uint64], stride: wp.uint64, offsets: wp.array[int]):
    row = wp.tid()
    key = wp.uint64(row) * stride
    lo = int(0)
    hi = keys.shape[0]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    offsets[row] = lo


@wp.kernel
def _contact_work(
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    values: wp.array[Any],
    work: wp.array[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    value = work.dtype(0)
    if keys[i] != wp.uint64(0) - wp.uint64(1):
        value = values[indices[i]]
        if i > 0:
            if keys[i] == keys[i - 1]:
                # A duplicated local contribution has no valid stable order.
                wp.atomic_or(flags, 0, 8)
    work[i] = value


@wp.kernel
def _contact_tree(
    keys: wp.array[wp.uint64],
    stride: wp.uint64,
    offsets: wp.array[int],
    work: wp.array[Any],
    distance: int,
    flags: wp.array[int],
):
    i = wp.tid()
    if keys[i] != wp.uint64(0) - wp.uint64(1):
        row = int(keys[i] // stride)
        relative = i - offsets[row]
        if relative % distance == 0 and (relative // distance) % 2 == 0:
            # The right operand is never written at the same tree level.
            if distance < offsets[row + 1] - offsets[row]:
                right = work.dtype(0)
                if distance < offsets[row + 1] - i:
                    right = work[i + distance]
                value = work[i] + right
                work[i] = value
                if not wp.isfinite(value):
                    wp.atomic_or(flags, 0, 4)


@wp.kernel
def _contact_result(offsets: wp.array[int], work: wp.array[Any], result: wp.array[Any]):
    row = wp.tid()
    value = result.dtype(0)
    if offsets[row] < offsets[row + 1]:
        value = work[offsets[row]]
    result[row] = value


@wp.kernel
def _clear_failed_rows(offsets: wp.array[int], result: wp.array[Any], flags: wp.array[int]):
    row = wp.tid()
    if flags[0] != 0:
        offsets[row] = 0
        if row < result.shape[0]:
            result[row] = result.dtype(0)


class _ContactRows:
    """Own bounded contact-only rows; leave contact admission and physics to the caller.

    Identities are checked Model ordinals, never allocation slots. Packing reserves
    UINT64_MAX for inactive entries. A local id distinguishes repeated row entries
    from the same contact; exact duplicate keys are hard errors. Flags are device
    data: 1=count, 2=key, 4=nonfinite input/sum, 8=duplicate key. Inputs must be
    contiguous and disjoint from owned scratch. Errors publish zero
    rows and offsets, without disguising invalid data as a capacity overflow.
    """

    BLOCK_SIZE = 256

    def __init__(self, row_count, capacity, identity_count, local_count, dtype, device, *, use_cuda_graph=False):
        for name, value, minimum, upper in (
            ("row_count", row_count, 0, 2**31 - 2),
            ("capacity", capacity, 0, 2**31 - 1),
            ("identity_count", identity_count, 1, 2**64 - 1),
            ("local_count", local_count, 1, 2**31 - 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not minimum <= value <= upper
            ):
                raise ValueError(f"Invalid contact row {name}")
        stride = int(identity_count) * int(local_count)
        if stride > 2**64 - 1 or int(row_count) * stride > 2**64 - 1:
            raise ValueError("Contact row key exceeds uint64 sentinel capacity")
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("Contact rows require native scalar precision")
        native_bytes = 4 if dtype == wp.float32 else 8
        byte_count = int(capacity) * (24 + native_bytes) + (int(row_count) + 1) * 4 + int(row_count) * native_bytes + 4
        if byte_count > 2**63 - 1:
            raise ValueError("Contact row byte capacity exceeds int64")
        self.row_count = int(row_count)
        self.capacity = int(capacity)
        self.identity_count = int(identity_count)
        self.local_count = int(local_count)
        self.stride = wp.uint64(stride)
        self.dtype = dtype
        self.device = wp.get_device(device)
        self.keys = wp.empty(capacity, dtype=wp.uint64, device=self.device)
        self.key_scratch = wp.empty_like(self.keys)
        self.indices = wp.empty(capacity, dtype=int, device=self.device)
        self.index_scratch = wp.empty_like(self.indices)
        self.work = wp.empty(capacity, dtype=dtype, device=self.device)
        self.offsets = wp.empty(row_count + 1, dtype=int, device=self.device)
        self.result = wp.empty(row_count, dtype=dtype, device=self.device)
        self.flags = wp.empty(1, dtype=int, device=self.device)
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)

    def buffers(self):
        return (
            self.keys,
            self.key_scratch,
            self.indices,
            self.index_scratch,
            self.work,
            self.offsets,
            self.result,
            self.flags,
        )

    @property
    def storage_bytes(self):
        """Report current arrays, excluding inputs and graph-retained previous bindings."""
        return sum(a.capacity for a in self.buffers())

    def reduce(self, rows, identities, local_ids, values, count):
        for array, dtype, size in (
            (rows, wp.int32, self.capacity),
            (identities, wp.uint64, self.capacity),
            (local_ids, wp.int32, self.capacity),
            (values, self.dtype, self.capacity),
            (count, wp.int32, 1),
        ):
            if array.shape != (size,) or array.dtype != dtype or array.device != self.device or not array.is_contiguous:
                raise ValueError("Invalid contact row input storage")
            if array.size:
                end = array.ptr + size * wp.types.type_size_in_bytes(dtype)
                for owned in self.buffers():
                    if owned.size and array.ptr < owned.ptr + owned.capacity and owned.ptr < end:
                        raise ValueError("Contact row inputs alias owned scratch")
        inputs = (rows, identities, local_ids, values, count)
        self.graphs.run(
            "contact_rows",
            (self.row_count, self.capacity, self.identity_count, self.local_count, self.dtype, self.BLOCK_SIZE),
            (*inputs, *self.buffers()),
            lambda: self._reduce(*inputs),
        )

    def _reduce(self, rows, identities, local_ids, values, count):
        wp.launch(_reset_rows, 1, [count, self.capacity, self.flags], device=self.device)
        keys, temp_keys = self.keys, self.key_scratch
        indices, temp_indices = self.indices, self.index_scratch
        if self.capacity:
            wp.launch(
                _contact_keys,
                self.capacity,
                [
                    rows,
                    identities,
                    local_ids,
                    values,
                    count,
                    self.row_count,
                    wp.uint64(self.identity_count),
                    self.local_count,
                    keys,
                    indices,
                    self.flags,
                ],
                device=self.device,
            )
            run = 1
            while run < self.capacity:
                wp.launch(
                    _merge_contact_keys,
                    self.capacity,
                    [keys, indices, temp_keys, temp_indices, run],
                    device=self.device,
                )
                keys, temp_keys = temp_keys, keys
                indices, temp_indices = temp_indices, indices
                run *= 2
        wp.launch(_contact_offsets, self.row_count + 1, [keys, self.stride, self.offsets], device=self.device)
        if self.capacity:
            wp.launch(_contact_work, self.capacity, [keys, indices, values, self.work, self.flags], device=self.device)
            # First reduce fixed 256-entry blocks relative to each row start,
            # then combine their roots using the same specified binary tree.
            distance = 1
            while distance < min(self.capacity, self.BLOCK_SIZE):
                wp.launch(
                    _contact_tree,
                    self.capacity,
                    [keys, self.stride, self.offsets, self.work, distance, self.flags],
                    device=self.device,
                )
                distance *= 2
            while distance < self.capacity:
                wp.launch(
                    _contact_tree,
                    self.capacity,
                    [keys, self.stride, self.offsets, self.work, distance, self.flags],
                    device=self.device,
                )
                distance *= 2
        if self.row_count:
            wp.launch(_contact_result, self.row_count, [self.offsets, self.work, self.result], device=self.device)
        wp.launch(_clear_failed_rows, self.row_count + 1, [self.offsets, self.result, self.flags], device=self.device)
