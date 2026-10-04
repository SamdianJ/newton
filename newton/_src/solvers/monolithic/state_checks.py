# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device checks for mutable state, history and published force arrays."""

from typing import Any

import warp as wp


def scalar_view(array):
    """Retain strided input support while exposing scalar components for checks."""
    dtype = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
    return array.contiguous().view(dtype).flatten()


@wp.kernel
def finite(values: wp.array[Any], flags: wp.array[int], slot: int):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_max(flags, slot, 1)


@wp.kernel
def history_check(
    version: wp.array[int],
    valid: wp.array[int],
    dt: wp.array[Any],
    digest: wp.array[wp.uint8],
    expected: wp.array[wp.uint8],
    expected_version: int,
    q: wp.array[Any],
    qd: wp.array[Any],
    bound_q: wp.array[Any],
    bound_qd: wp.array[Any],
    flags: wp.array[int],
    base: int,
):
    i = wp.tid()
    if i == 0:
        if version[0] != expected_version or valid[0] != 1:
            flags[base] = 1
        if dt[0] <= type(dt[0])(0.0):
            wp.atomic_max(flags, base + 1, 1)
    if i < digest.shape[0] and digest[i] != expected[i]:
        wp.atomic_max(flags, base + 1, 1)
    if i < q.shape[0]:
        if q[i] != bound_q[i] or qd[i] != bound_qd[i]:
            wp.atomic_max(flags, base + 2, 1)


@wp.kernel
def particle_check(
    x: wp.array[Any],
    v: wp.array[Any],
    ids: wp.array[int],
    fixed: wp.array[Any],
    bound_x: wp.array[Any],
    bound_v: wp.array[Any],
    precision: wp.array[int],
    bits: int,
    valid: wp.array[int],
    generation: wp.array[wp.int64],
    bound: bool,
    flags: wp.array[int],
):
    i = wp.tid()
    if i < ids.shape[0]:
        j = ids[i]
        for axis in range(3):
            if x[j][axis] != fixed[i][axis] or v[j][axis] != type(v[0][0])(0.0):
                wp.atomic_max(flags, 0, 1)
    if bound:
        if i == 0:
            if bits != 0 and precision[0] != bits:
                flags[1] = 1
            if (
                (valid[0] != 0 and valid[0] != 1)
                or generation[0] < wp.int64(0)
                or (valid[0] != 0 and generation[0] == wp.int64(0))
            ):
                flags[3] = 1
        if i < x.shape[0]:
            for axis in range(3):
                if x[i][axis] != bound_x[i][axis] or v[i][axis] != bound_v[i][axis]:
                    wp.atomic_max(flags, 2, 1)
