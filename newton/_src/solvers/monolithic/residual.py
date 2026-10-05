# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static row gathers for native contact and element residual contributions."""

from typing import Any

import numpy as np
import warp as wp


def _row_gather_map(columns, row_count, device):
    """Group contribution slots by row, preserving source order and excluding fixed rows."""
    if not 0 <= row_count < 2**31 - 1 or columns.size >= 2**31:
        raise ValueError("Residual gather capacity exceeds int32")
    if columns.ndim != 1 or np.any(columns < -1) or np.any(columns >= row_count):
        raise ValueError("Invalid residual gather rows")
    slots = np.flatnonzero(columns >= 0)
    rows = columns[slots]
    slots = slots[np.argsort(rows, kind="stable")]
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=row_count))]
    return (
        wp.array(offsets, dtype=int, device=device),
        wp.array(slots, dtype=int, device=device),
    )


@wp.kernel
def _accumulate_rows(offsets: wp.array[int], slots: wp.array[int], terms: wp.array[Any], residual: wp.array[Any]):
    row = wp.tid()
    value = residual[row]
    for index in range(offsets[row], offsets[row + 1]):
        value += terms[slots[index]]
    residual[row] = value
