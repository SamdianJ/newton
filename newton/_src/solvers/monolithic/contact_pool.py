# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed storage and native row trees for contact contributions."""

from functools import cache
from itertools import pairwise
from typing import Any

import numpy as np
import warp as wp

from .segment_graph import _SegmentGraphCache

_CONTACT_WORKERS = 262144


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
    scratch_keys: wp.array[wp.uint64],
    scratch_indices: wp.array[int],
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
    # Prefix merge leaves the tails untouched, including recorded scratch.
    scratch_keys[i] = wp.uint64(0) - wp.uint64(1)
    scratch_indices[i] = i


@wp.kernel
def _merge_contact_keys(
    keys: wp.array[wp.uint64],
    indices: wp.array[int],
    out_keys: wp.array[wp.uint64],
    out_indices: wp.array[int],
    count: wp.array[int],
    run: int,
    workers: int,
):
    i = wp.tid()
    n = count[0]
    if n < 0 or n > keys.shape[0] or i >= n:
        return
    while i < n:
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
                # Keep the left run first for equal stable keys.
                advance = keys[mid] <= key
            if advance:
                lo = mid + 1
            else:
                hi = mid
        destination = base + (i - own_begin) + (lo - first)
        out_keys[destination] = key
        out_indices[destination] = indices[i]
        # Avoid wrapping after the last int32-capacity entry.
        i += wp.min(workers, n - i)


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
        self._validate_storage()
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

    def _validate_storage(self):
        types = (wp.uint64, wp.uint64, wp.int32, wp.int32, self.dtype, wp.int32, self.dtype, wp.int32)
        sizes = (self.capacity,) * 5 + (self.row_count + 1, self.row_count, 1)
        for array, dtype, size in zip(self.buffers(), types, sizes, strict=True):
            if array.shape != (size,) or array.dtype != dtype or array.device != self.device or not array.is_contiguous:
                raise ValueError("Invalid contact row scratch storage")

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
                    temp_keys,
                    temp_indices,
                    self.flags,
                ],
                device=self.device,
            )
            run = 1
            workers = min(self.capacity, _CONTACT_WORKERS)
            while run < self.capacity:
                wp.launch(
                    _merge_contact_keys,
                    workers,
                    [keys, indices, temp_keys, temp_indices, count, run, workers],
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
        return keys, indices


@cache
def _pool_types(dtype):
    @wp.struct
    class Candidates:
        count: wp.array[int]
        overflow: wp.array[int]
        identities: wp.array[wp.uint64]
        support_counts: wp.array[int]
        columns: wp.array2d[int]
        residual: wp.array2d[dtype]
        weights: wp.array2d[float]
        factors: wp.array3d[float]
        energy: wp.array2d[dtype]
        force_counts: wp.array[int]
        force_rows: wp.array2d[int]
        forces: wp.array2d[dtype]
        pairs: wp.array3d[dtype]

    @wp.struct
    class Records:
        flags: wp.array[int]
        count: wp.array[int]
        used: wp.array[int]
        requested: wp.array[int]
        dropped: wp.array[int]
        accepted: wp.array[int]
        identities: wp.array[wp.uint64]
        source_slots: wp.array[int]
        support_count: wp.array[int]
        support_offsets: wp.array[int]
        r_rows: wp.array[int]
        r_ids: wp.array[wp.uint64]
        r_local: wp.array[int]
        r_values: wp.array[dtype]
        energy_count: wp.array[int]
        e_rows: wp.array[int]
        e_ids: wp.array[wp.uint64]
        e_local: wp.array[int]
        e_values: wp.array[dtype]
        force_count: wp.array[int]
        f_rows: wp.array[int]
        f_ids: wp.array[wp.uint64]
        f_local: wp.array[int]
        f_values: wp.array[dtype]
        factor_count: wp.array[int]
        factor_offsets: wp.array[int]
        factor_columns: wp.array[int]
        factor_local: wp.array[int]
        factor_values: wp.array[float]
        factor_weights: wp.array[float]
        factor_kind: wp.array[int]
        factor_contact: wp.array[int]
        triplet_count: wp.array[int]
        triplet_offsets: wp.array[int]
        triplet_rows: wp.array[int]
        triplet_columns: wp.array[int]
        triplet_values: wp.array[float]
        pairs: wp.array3d[dtype]

    return Candidates, Records


@wp.kernel
def _begin_pool(candidates: Any, records: Any, identity_flags: wp.array[int]):
    records.flags[0] = identity_flags[0]
    if candidates.overflow[0] < 0 or candidates.overflow[0] > 2147483647 - wp.max(candidates.count[0], 0):
        wp.atomic_or(records.flags, 0, 1)


@wp.kernel
def _validate_contacts(candidates: Any, records: Any, row_count: int, force_row_count: int, active: wp.array[int]):
    i = wp.tid()
    active[i] = 0
    if i >= candidates.count[0]:
        return
    k, f = candidates.support_counts[i], candidates.force_counts[i]
    if k < 0 or k > candidates.columns.shape[1] or f < 0 or f > candidates.force_rows.shape[1]:
        wp.atomic_or(records.flags, 0, 2)
        return
    nonzero = False
    for j in range(2):
        energy_value = candidates.energy[i, j] + candidates.energy.dtype(0)
        if not wp.isfinite(energy_value):
            wp.atomic_or(records.flags, 0, 4)
        nonzero = nonzero or energy_value != type(energy_value)(0)
        for axis in range(6):
            force = candidates.pairs[i, j, axis]
            if not wp.isfinite(force):
                wp.atomic_or(records.flags, 0, 4)
            nonzero = nonzero or force != type(force)(0)
    for j in range(k):
        residual_row = candidates.columns[i, j] + 0
        residual_value = candidates.residual[i, j] + candidates.residual.dtype(0)
        if residual_row < 0 or residual_row >= row_count:
            wp.atomic_or(records.flags, 0, 2)
        if not wp.isfinite(residual_value):
            wp.atomic_or(records.flags, 0, 4)
        nonzero = nonzero or residual_value != type(residual_value)(0)
    for component in range(3):
        weight = candidates.weights[i, component]
        if not wp.isfinite(weight) or weight < 0.0:
            wp.atomic_or(records.flags, 0, 4)
        maximum = float(0)
        for j in range(k):
            factor_value = candidates.factors[i, component, j] + 0.0
            if not wp.isfinite(factor_value):
                wp.atomic_or(records.flags, 0, 4)
            maximum = wp.max(maximum, wp.abs(factor_value))
        # Monotonic FP32 absolute products bound every (weight*gi)*gj without
        # quadratic validation. Keep the original left-associated arithmetic.
        bound = (weight * maximum) * maximum
        if not wp.isfinite(bound):
            wp.atomic_or(records.flags, 0, 4)
        nonzero = nonzero or (weight > 0.0 and maximum != 0.0)
    for j in range(f):
        force_row = candidates.force_rows[i, j] + 0
        force_value = candidates.forces[i, j] + candidates.forces.dtype(0)
        if force_row < 0 or force_row >= force_row_count:
            wp.atomic_or(records.flags, 0, 2)
        if not wp.isfinite(force_value):
            wp.atomic_or(records.flags, 0, 4)
        nonzero = nonzero or force_value != type(force_value)(0)
    if nonzero:
        active[i] = 1


@wp.kernel
def _reserve_contacts(
    candidates: Any,
    records: Any,
    indices: wp.array[int],
    active: wp.array[int],
    capacity: int,
    selected: wp.array[int],
    demands: wp.array[wp.vec4i],
):
    index = wp.tid()
    selected[index] = 0
    demands[index] = wp.vec4i(0)
    if records.flags[0] != 0 or index >= candidates.count[0]:
        return
    i = indices[index]
    if active[i] == 0:
        return
    k = candidates.support_counts[i]
    units = wp.max(1, k)
    wp.atomic_add(records.requested, 0, units)
    before = wp.atomic_add(records.used, 0, 0)
    while before <= capacity - units:
        actual = wp.atomic_cas(records.used, 0, before, before + units)
        if actual == before:
            selected[index] = 1
            components = int(0)
            for axis in range(3):
                if candidates.weights[i, axis] > 0.0:
                    components += 1
            # Retain zero products within positive factors, but preserve the
            # original omission of entire zero-weight factors from B's tree.
            demands[index] = wp.vec4i(1, k, components * k * k, candidates.force_counts[i])
            return
        before = actual
    # Failure does not increment any resource counter or publish a partial row.
    wp.atomic_add(records.dropped, 0, 1)


@wp.kernel
def _scan_seed(demands: wp.array[wp.vec4i], prefix: wp.array[wp.vec4i]):
    i = wp.tid()
    value = wp.vec4i(0)
    if i < demands.shape[0]:
        value = demands[i]
    prefix[i] = value


@wp.kernel
def _scan_up(prefix: wp.array[wp.vec4i], distance: int):
    right = (wp.tid() + 1) * 2 * distance - 1
    prefix[right] = prefix[right] + prefix[right - distance]


@wp.kernel
def _scan_root(prefix: wp.array[wp.vec4i], records: Any, append: bool):
    last = prefix.shape[0] - 1
    base = wp.vec4i(0)
    if append:
        base = wp.vec4i(records.count[0], records.support_count[0], records.triplet_count[0], records.force_count[0])
    total = prefix[last] + base
    records.count[0] = total[0]
    records.support_count[0] = total[1]
    records.triplet_count[0] = total[2]
    records.force_count[0] = total[3]
    records.energy_count[0] = 4 * total[0]
    records.factor_count[0] = 3 * total[0]
    records.support_offsets[total[0]] = total[1]
    records.triplet_offsets[total[0]] = total[2]
    records.factor_offsets[3 * total[0]] = 3 * total[1]
    prefix[last] = base


@wp.kernel
def _scan_down(prefix: wp.array[wp.vec4i], distance: int):
    right = (wp.tid() + 1) * 2 * distance - 1
    left = right - distance
    value = prefix[left]
    prefix[left] = prefix[right]
    prefix[right] = prefix[right] + value


@wp.kernel
def _write_contacts(
    candidates: Any, records: Any, indices: wp.array[int], selected: wp.array[int], prefix: wp.array[wp.vec4i]
):
    index = wp.tid()
    if selected[index] == 0:
        return
    i = indices[index]
    offset = prefix[index]
    contact, r_begin, f_begin = offset[0], offset[1], offset[3]
    identity = candidates.identities[i]
    k = candidates.support_counts[i]
    records.identities[contact] = identity
    records.source_slots[contact] = i
    records.support_offsets[contact] = r_begin
    records.triplet_offsets[contact] = offset[2]
    for component in range(3):
        factor = 3 * contact + component
        begin = 3 * r_begin + component * k
        records.factor_offsets[factor] = begin
        records.factor_weights[factor] = candidates.weights[i, component]
        records.factor_kind[factor] = wp.where(component == 0, 1, 2)
        records.factor_contact[factor] = contact
        for j in range(k):
            records.factor_columns[begin + j] = candidates.columns[i, j]
            records.factor_local[begin + j] = j
            records.factor_values[begin + j] = candidates.factors[i, component, j]
    for j in range(k):
        slot = r_begin + j
        records.r_rows[slot] = candidates.columns[i, j]
        records.r_ids[slot] = identity
        records.r_local[slot] = j
        records.r_values[slot] = candidates.residual[i, j]
    for j in range(4):
        slot = 4 * contact + j
        if j < 2:
            records.e_values[slot] = candidates.energy[i, j]
        else:
            records.e_values[slot] = wp.abs(candidates.energy[i, j % 2])
        records.e_rows[slot] = j
        records.e_ids[slot] = identity
        records.e_local[slot] = 0
    for j in range(candidates.force_counts[i]):
        slot = f_begin + j
        records.f_rows[slot] = candidates.force_rows[i, j]
        records.f_ids[slot] = identity
        records.f_local[slot] = j
        records.f_values[slot] = candidates.forces[i, j]
    for end in range(2):
        for axis in range(6):
            records.pairs[contact, end, axis] = candidates.pairs[i, end, axis]


@wp.kernel
def _write_contact_triplets(records: Any, workers: int):
    i = wp.tid()
    while i < records.triplet_count[0]:
        lo = int(0)
        hi = records.count[0]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if records.triplet_offsets[mid + 1] <= i:
                lo = mid + 1
            else:
                hi = mid
        k = records.support_offsets[lo + 1] - records.support_offsets[lo]
        local = i - records.triplet_offsets[lo]
        remaining = local // (k * k)
        component = int(0)
        for axis in range(3):
            if records.factor_weights[3 * lo + axis] > 0.0:
                if remaining == 0:
                    component = axis
                remaining -= 1
        begin = records.factor_offsets[3 * lo + component]
        row = begin + (local % (k * k)) // k
        column = begin + local % k
        records.triplet_rows[i] = records.factor_columns[row]
        records.triplet_columns[i] = records.factor_columns[column]
        records.triplet_values[i] = (
            records.factor_weights[3 * lo + component] * records.factor_values[row]
        ) * records.factor_values[column]
        if workers >= records.triplet_count[0] - i:
            break
        i += workers


@wp.kernel
def _pool_row_flags(records: Any, energy: wp.array[int], residual: wp.array[int], force: wp.array[int]):
    records.flags[0] = records.flags[0] | energy[0] | residual[0] | force[0]


@wp.kernel
def _publish_contacts(records: Any, indices: wp.array[int], selected: wp.array[int]):
    i = wp.tid()
    if records.flags[0] == 0 and selected[i] != 0:
        records.accepted[indices[i]] = 1


@wp.kernel
def _clear_failed_pool(values: wp.array[Any], flags: wp.array[int], workers: int):
    if flags[0] != 0:
        i = wp.tid()
        while i < values.shape[0]:
            values[i] = values.dtype(0)
            if workers >= values.shape[0] - i:
                break
            i += workers


@wp.kernel
def _begin_contact_batch(candidates: Any, records: Any, count: wp.array[int], base: wp.array[int], capacity: int):
    n = candidates.count[0]
    base[0] = -1
    if n < 0 or n > candidates.identities.shape[0] or count[0] > capacity - n:
        wp.atomic_or(records.flags, 0, 1)
    elif candidates.overflow[0] < 0 or candidates.overflow[0] > 2147483647 - n:
        wp.atomic_or(records.flags, 0, 1)
    else:
        base[0] = count[0]
        count[0] += n


@wp.kernel
def _contact_batch_identities(
    candidates: Any,
    records: Any,
    identities: wp.array[wp.uint64],
    base: wp.array[int],
    identity_count: wp.uint64,
    indices: wp.array[int],
):
    i = wp.tid()
    indices[i] = i
    if base[0] >= 0 and i < candidates.count[0]:
        identity = candidates.identities[i]
        identities[base[0] + i] = identity
        if identity >= identity_count:
            wp.atomic_or(records.flags, 0, 2)


@wp.kernel
def _contact_batch_sources(records: Any, selected: wp.array[int], prefix: wp.array[wp.vec4i], base: wp.array[int]):
    i = wp.tid()
    if selected[i] != 0:
        records.source_slots[prefix[i][0]] = base[0] + i


@wp.kernel
def _contact_stream_flags(records: Any, flags: wp.array[int]):
    records.flags[0] = records.flags[0] | flags[0]


@wp.kernel
def _publish_contact_stream(records: Any):
    i = wp.tid()
    if records.flags[0] == 0 and i < records.count[0]:
        records.accepted[records.source_slots[i]] = 1


class _ContactPool:
    """Stage and commit complete local contact records within one bounded budget.

    All live inputs are validated before reservation, even at zero capacity.
    One CAS reserves u=max(1,k); a fixed exclusive integer scan derives addresses
    after all reservations. Publish accepted only after every local E/r/B/force
    write and native row reduction. No physical state/history is changed here.
    Staging overflow is supplied by the producer separately from written count.
    With batch_capacity set, reuse only batch x support candidate storage and
    call begin/append/finish. Batches share one budget and concatenate exclusive
    scans. The O(M) identity ledger checks even dropped/inert inputs at finish;
    row keys use identities, never these storage slots. Outputs are unpublished
    until all batches finish. Determinism is required only without overflow.
    Three FP32 factors and all their COO records retain the existing arithmetic;
    fixed BSR topology/duplicate reduction remain with linear assembly.
    """

    FORCE_TERMS = 15  # Three tet vertices plus one body; joint force comes from r.

    def __init__(
        self,
        row_count,
        force_row_count,
        candidate_capacity,
        capacity,
        max_support,
        identity_count,
        dtype,
        device,
        *,
        contact_factor_capacity=None,
        contact_triplet_capacity=None,
        use_cuda_graph=False,
        batch_capacity=None,
    ):
        for name, value, minimum, maximum in (
            ("row_count", row_count, 0, 2**31 - 2),
            ("force_row_count", force_row_count, 0, 2**31 - 2),
            ("candidate_capacity", candidate_capacity, 0, 2**30 - 1),
            ("capacity", capacity, 0, 2**31 - 1),
            ("max_support", max_support, 1, 2**31 - 1),
            ("identity_count", identity_count, 1, 2**64 - 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"Invalid contact pool {name}")
        if dtype not in (wp.float32, wp.float64):
            raise ValueError("Contact pool requires native scalar precision")
        m, c, k = int(candidate_capacity), int(capacity), int(max_support)
        b = m if batch_capacity is None else batch_capacity
        if (
            isinstance(b, (bool, np.bool_))
            or not isinstance(b, (int, np.integer))
            or not (0 if m == 0 else 1) <= b <= m
        ):
            raise ValueError("Invalid contact batch capacity")
        b = int(b)
        if m * k >= 2**31:
            raise ValueError("Contact requested budget exceeds int32")
        requested_capacity = c
        for name, limit in (("factor", contact_factor_capacity), ("triplet", contact_triplet_capacity)):
            if limit is None:
                continue
            if (
                isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, (int, np.integer))
                or not 0 <= limit < 2**31
            ):
                raise ValueError(f"Invalid contact {name} capacity")
            if name == "factor" and 3 * min(m, c) > limit:
                c = min(c, int(limit) // 3)
            if name == "triplet" and 3 * k * min(m * k, c) > limit:
                c = min(c, int(limit) // (3 * k))
        n, r, f = min(m, c), min(m * k, c), self.FORCE_TERMS
        t = 3 * k * r
        p = 1 << b.bit_length()  # Power of two >= batch size+1, including zero.
        if 3 * k * k >= 2**31:
            raise ValueError("Contact triplet demand exceeds int32")
        for rows, local in ((1, 1), (4, 1), (row_count, k), (force_row_count, f)):
            stride = int(identity_count) * local
            if stride > 2**64 - 1 or int(rows) * stride > 2**64 - 1:
                raise ValueError("Contact pool key exceeds uint64 sentinel capacity")
        candidate_spec = {
            "count": ((1,), wp.int32),
            "overflow": ((1,), wp.int32),
            "identities": ((b,), wp.uint64),
            "support_counts": ((b,), wp.int32),
            "columns": ((b, k), wp.int32),
            "residual": ((b, k), dtype),
            "weights": ((b, 3), wp.float32),
            "factors": ((b, 3, k), wp.float32),
            "energy": ((b, 2), dtype),
            "force_counts": ((b,), wp.int32),
            "force_rows": ((b, f), wp.int32),
            "forces": ((b, f), dtype),
            "pairs": ((b, 2, 6), dtype),
        }
        record_spec = dict.fromkeys(
            (
                "flags",
                "count",
                "used",
                "requested",
                "dropped",
                "support_count",
                "energy_count",
                "force_count",
                "factor_count",
                "triplet_count",
            ),
            ((1,), wp.int32),
        )
        record_spec.update(
            {
                "accepted": ((m,), wp.int32),
                "identities": ((n,), wp.uint64),
                "source_slots": ((n,), wp.int32),
                "support_offsets": ((n + 1,), wp.int32),
                "factor_offsets": ((3 * n + 1,), wp.int32),
                "factor_columns": ((3 * r,), wp.int32),
                "factor_local": ((3 * r,), wp.int32),
                "factor_values": ((3 * r,), wp.float32),
                "factor_weights": ((3 * n,), wp.float32),
                "factor_kind": ((3 * n,), wp.int32),
                "factor_contact": ((3 * n,), wp.int32),
                "triplet_offsets": ((n + 1,), wp.int32),
                "triplet_rows": ((t,), wp.int32),
                "triplet_columns": ((t,), wp.int32),
                "triplet_values": ((t,), wp.float32),
                "pairs": ((n, 2, 6), dtype),
            }
        )
        for prefix, size in (("r", r), ("e", 4 * n), ("f", f * n)):
            for name, scalar in (("rows", wp.int32), ("ids", wp.uint64), ("local", wp.int32), ("values", dtype)):
                record_spec[f"{prefix}_{name}"] = ((size,), scalar)
        native_bytes = wp.types.type_size_in_bytes(dtype)
        # One schema defines validation, arithmetic checks, allocation and ledger.
        sizes = [int(np.prod(shape, dtype=object)) for shape, _ in (*candidate_spec.values(), *record_spec.values())]
        if any(size >= 2**31 for size in sizes):
            raise ValueError("Contact pool array capacity exceeds int32")
        byte_count = sum(
            size * wp.types.type_size_in_bytes(scalar)
            for size, (_, scalar) in zip(sizes, (*candidate_spec.values(), *record_spec.values()), strict=True)
        )
        byte_count += 24 * b + 16 * p + (4 + native_bytes) * m
        if batch_capacity is not None:
            byte_count += 8 * m + 4 * b + 8
        for rows, size in ((1, m), (4, 4 * n), (row_count, r), (force_row_count, f * n)):
            byte_count += size * (24 + native_bytes) + 4 * (int(rows) + 1) + int(rows) * native_bytes + 4
        if byte_count >= 2**63:
            raise ValueError("Contact pool byte capacity exceeds int64")
        self.row_count, self.force_row_count = int(row_count), int(force_row_count)
        self.capacity, self.requested_capacity = c, requested_capacity
        self.candidate_capacity, self.max_support, self.batch_capacity = m, k, b
        self.contact_capacity, self.support_capacity, self.triplet_capacity = n, r, t
        self.identity_count, self.dtype, self.device = int(identity_count), dtype, wp.get_device(device)
        self._specs = (candidate_spec, record_spec)
        self.candidates, self.records = (kind() for kind in _pool_types(dtype))
        for struct, spec in zip((self.candidates, self.records), self._specs, strict=True):
            for name, (shape, scalar) in spec.items():
                setattr(struct, name, wp.zeros(shape, dtype=scalar, device=self.device))
        self._active, self._selected = (wp.zeros(b, dtype=int, device=self.device) for _ in range(2))
        self._demands = wp.zeros(b, dtype=wp.vec4i, device=self.device)
        self._prefix = wp.zeros(p, dtype=wp.vec4i, device=self.device)
        self._zero_rows = wp.zeros(m, dtype=int, device=self.device)
        self._zero_values = wp.zeros(m, dtype=dtype, device=self.device)
        self._scratch_spec = ((b, wp.int32), (b, wp.int32), (b, wp.vec4i), (p, wp.vec4i), (m, wp.int32), (m, dtype))
        self._identity = _ContactRows(1, m, identity_count, 1, dtype, self.device)
        self.energy = _ContactRows(4, 4 * n, identity_count, 1, dtype, self.device)
        self.residual = _ContactRows(row_count, r, identity_count, k, dtype, self.device)
        self.force = _ContactRows(force_row_count, f * n, identity_count, f, dtype, self.device)
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)
        self.payload_bytes = byte_count
        self._stream = None
        self._stream_started = False
        if batch_capacity is not None:
            self._stream = (
                wp.zeros(m, dtype=wp.uint64, device=self.device),
                wp.zeros(1, dtype=int, device=self.device),
                wp.zeros(1, dtype=int, device=self.device),
                wp.zeros(b, dtype=int, device=self.device),
            )

    def buffers(self):
        buffers = [
            getattr(struct, name)
            for struct, spec in zip((self.candidates, self.records), self._specs, strict=True)
            for name in spec
        ]
        buffers.extend((self._active, self._selected, self._demands, self._prefix, self._zero_rows, self._zero_values))
        for owner in (self._identity, self.energy, self.residual, self.force):
            buffers.extend(owner.buffers())
        if self._stream is not None:
            buffers.extend(self._stream)
        return tuple(buffers)

    def _validate_storage(self):
        for struct, spec in zip((self.candidates, self.records), self._specs, strict=True):
            for name, (shape, dtype) in spec.items():
                array = getattr(struct, name)
                if (
                    array.shape != shape
                    or array.dtype != dtype
                    or array.device != self.device
                    or not array.is_contiguous
                ):
                    raise ValueError(f"Invalid contact pool storage: {name}")
        scratch = (self._active, self._selected, self._demands, self._prefix, self._zero_rows, self._zero_values)
        for array, (size, dtype) in zip(scratch, self._scratch_spec, strict=True):
            if array.shape != (size,) or array.dtype != dtype or array.device != self.device or not array.is_contiguous:
                raise ValueError("Invalid contact pool scratch storage")
        for owner in (self._identity, self.energy, self.residual, self.force):
            owner._validate_storage()
        if self._stream is not None:
            for array, (size, dtype) in zip(
                self._stream,
                ((self.candidate_capacity, wp.uint64), (1, wp.int32), (1, wp.int32), (self.batch_capacity, wp.int32)),
                strict=True,
            ):
                if (
                    array.shape != (size,)
                    or array.dtype != dtype
                    or array.device != self.device
                    or not array.is_contiguous
                ):
                    raise ValueError("Invalid contact stream storage")
        buffers = self.buffers()
        intervals = sorted((a.ptr, a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)) for a in buffers if a.size)
        if any(right[0] < left[1] for left, right in pairwise(intervals)):
            raise ValueError("Contact pool arrays alias")
        return buffers

    def evaluate(self):
        if self._stream is not None:
            raise ValueError("Batched contact pools require begin/append/finish")
        buffers = self._validate_storage()
        self.graphs.run(
            "contact_pool",
            (
                self.row_count,
                self.force_row_count,
                self.candidate_capacity,
                self.capacity,
                self.max_support,
                self.identity_count,
                self.dtype,
                self.energy.BLOCK_SIZE,
            ),
            buffers,
            self._evaluate,
        )

    def _evaluate(self):
        a, out = self.candidates, self.records
        for name in self._specs[1]:
            getattr(out, name).zero_()
        _, indices = self._identity._reduce(self._zero_rows, a.identities, self._zero_rows, self._zero_values, a.count)
        wp.launch(_begin_pool, 1, [a, out, self._identity.flags], device=self.device)
        m = self.candidate_capacity
        if m:
            wp.launch(
                _validate_contacts, m, [a, out, self.row_count, self.force_row_count, self._active], device=self.device
            )
            wp.launch(
                _reserve_contacts,
                m,
                [a, out, indices, self._active, self.capacity, self._selected, self._demands],
                device=self.device,
            )
        wp.launch(_scan_seed, self._prefix.size, [self._demands, self._prefix], device=self.device)
        distance = 1
        while distance < self._prefix.size:
            wp.launch(_scan_up, self._prefix.size // (2 * distance), [self._prefix, distance], device=self.device)
            distance *= 2
        wp.launch(_scan_root, 1, [self._prefix, out, False], device=self.device)
        distance //= 2
        while distance:
            wp.launch(_scan_down, self._prefix.size // (2 * distance), [self._prefix, distance], device=self.device)
            distance //= 2
        if m:
            wp.launch(_write_contacts, m, [a, out, indices, self._selected, self._prefix], device=self.device)
        self._finish_records(indices)

    def _finish_records(self, indices=None):
        out = self.records
        if self.triplet_capacity:
            workers = min(self.triplet_capacity, _CONTACT_WORKERS)
            wp.launch(_write_contact_triplets, workers, [out, workers], device=self.device)
        for owner, prefix, count in (
            (self.energy, "e", out.energy_count),
            (self.residual, "r", out.support_count),
            (self.force, "f", out.force_count),
        ):
            owner._reduce(*(getattr(out, f"{prefix}_{name}") for name in ("rows", "ids", "local", "values")), count)
        wp.launch(
            _pool_row_flags, 1, [out, self.energy.flags, self.residual.flags, self.force.flags], device=self.device
        )
        if indices is None:
            if self.contact_capacity:
                wp.launch(_publish_contact_stream, self.contact_capacity, [out], device=self.device)
        elif self.candidate_capacity:
            wp.launch(_publish_contacts, self.candidate_capacity, [out, indices, self._selected], device=self.device)
        # A row-sum overflow is a hard error for the complete evaluation, not a
        # capacity warning or a partially usable force/B record.
        outputs = [getattr(out, name) for name in self._specs[1] if name not in ("flags", "requested")]
        for owner in (self.energy, self.residual, self.force):
            outputs.extend((owner.result, owner.offsets))
        for values in outputs:
            if values.size:
                workers = min(values.size, _CONTACT_WORKERS)
                wp.launch(_clear_failed_pool, workers, [values.flatten(), out.flags, workers], device=self.device)

    def begin(self):
        """Reset one evaluation; retain only bounded batch x support staging."""
        if self._stream is None:
            raise ValueError("Contact pool has no bounded batch storage")
        self._validate_storage()
        for name in self._specs[1]:
            # COO consumers use triplet_count; generation overwrites that prefix.
            if name not in ("triplet_rows", "triplet_columns", "triplet_values"):
                getattr(self.records, name).zero_()
        for array in self._stream[:3]:
            array.zero_()
        for owner in (self.energy, self.residual, self.force):
            owner.result.zero_()
            owner.offsets.zero_()
        self._stream_started = True

    def append(self):
        """Validate a whole batch and append once against the shared CAS budget.

        This additive chain must not use the overwrite-only graph cache: its
        warmup and capture would reserve/append the same contacts multiple times.
        The producer may use a cached graph to overwrite candidates before here.
        """
        if not self._stream_started:
            raise ValueError("Begin a contact evaluation before appending")
        self._validate_storage()
        identities, count, base, indices = self._stream
        a, out = self.candidates, self.records
        wp.launch(_begin_contact_batch, 1, [a, out, count, base, self.candidate_capacity], device=self.device)
        b = self.batch_capacity
        if b:
            wp.launch(
                _contact_batch_identities,
                b,
                [a, out, identities, base, wp.uint64(self.identity_count), indices],
                device=self.device,
            )
            wp.launch(
                _validate_contacts, b, [a, out, self.row_count, self.force_row_count, self._active], device=self.device
            )
            wp.launch(
                _reserve_contacts,
                b,
                [a, out, indices, self._active, self.capacity, self._selected, self._demands],
                device=self.device,
            )
        wp.launch(_scan_seed, self._prefix.size, [self._demands, self._prefix], device=self.device)
        distance = 1
        while distance < self._prefix.size:
            wp.launch(_scan_up, self._prefix.size // (2 * distance), [self._prefix, distance], device=self.device)
            distance *= 2
        wp.launch(_scan_root, 1, [self._prefix, out, True], device=self.device)
        distance //= 2
        while distance:
            wp.launch(_scan_down, self._prefix.size // (2 * distance), [self._prefix, distance], device=self.device)
            distance //= 2
        if b:
            wp.launch(_write_contacts, b, [a, out, indices, self._selected, self._prefix], device=self.device)
            wp.launch(_contact_batch_sources, b, [out, self._selected, self._prefix, base], device=self.device)

    def finish(self):
        """Check identities across all batches, then derive and publish one evaluation."""
        if not self._stream_started:
            raise ValueError("Begin a contact evaluation before finishing")
        buffers = self._validate_storage()
        self.graphs.run(
            "contact_stream_finish",
            (self.candidate_capacity, self.capacity, self.max_support, self.identity_count, self.dtype),
            buffers,
            self._finish_stream,
        )
        self._stream_started = False

    def _finish_stream(self):
        identities, count, _, _ = self._stream
        self._identity._reduce(self._zero_rows, identities, self._zero_rows, self._zero_values, count)
        wp.launch(_contact_stream_flags, 1, [self.records, self._identity.flags], device=self.device)
        self._finish_records()
