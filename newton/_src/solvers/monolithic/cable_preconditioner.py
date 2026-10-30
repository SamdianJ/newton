# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-cable band Cholesky in vertex/edge order; the global operator stays coupled."""

from typing import Any

import numpy as np
import warp as wp

from .cable_operator import _band


@wp.kernel
def _factor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    scale: wp.array[Any],
    rows: wp.array[int],
    inverse: wp.array[int],
    spans: wp.array2d[int],
    gauge_rows: wp.array[int],
    band: wp.array[Any],
    shifts: wp.array[Any],
    regularization: Any,
    tolerance: Any,
    factor: wp.array2d[Any],
    used: wp.array[Any],
    status: wp.array[int],
):
    actor = wp.tid()
    start, end = spans[actor, 0], spans[actor, 1]
    # FP32 sums of stiff stretch entries can leave the band indefinite in a general
    # orientation. Only the PRECONDITIONER is then shifted by a growing multiple of
    # its own diagonal; the operator PCG applies keeps the requested regularization.
    for attempt in range(shifts.shape[0]):
        shift = shifts[attempt]
        failed = int(0)
        for i in range(start, end):
            if failed == 0:
                row = rows[i]
                for distance in range(11):
                    factor[i, distance] = band[11 * i + distance]
                for slot in range(offsets[row], offsets[row + 1]):
                    column = columns[slot]
                    j = inverse[column]
                    if j >= start and j <= i:
                        if i - j <= 10:
                            factor[i, i - j] = factor[i, i - j] + values[slot] * scale[row] * scale[column]
                        elif values[slot] != values.dtype(0.0):
                            wp.atomic_max(status, 0, 1)
                            return
                factor[i, 0] = factor[i, 0] + regularization + shift * wp.abs(factor[i, 0])
                # Lift only a verified instantaneous null direction in the PRECONDITIONER.
                # The physical matrix/residual are unchanged; the delivered theta increment
                # is projected to zero mean by the common-angle constraint.
                if row == gauge_rows[actor]:
                    factor[i, 0] = factor[i, 0] + factor.dtype(1.0)
                for j in range(wp.max(start, i - 10), i + 1):
                    if failed == 0:
                        value = factor[i, i - j]
                        for k in range(wp.max(start, i - 10), j):
                            value -= factor[i, i - k] * factor[j, j - k]
                        if not wp.isfinite(value):
                            wp.atomic_max(status, 0, 16)
                            return
                        if i == j:
                            if value <= tolerance:
                                failed = 1
                            else:
                                factor[i, 0] = wp.sqrt(value)
                        else:
                            factor[i, i - j] = value / factor[j, 0]
        if failed == 0:
            used[actor] = shift
            return
    wp.atomic_max(status, 0, 12)


@wp.kernel
def _apply(
    rows: wp.array[int],
    spans: wp.array2d[int],
    factor: wp.array2d[Any],
    rhs: wp.array[Any],
    result: wp.array[Any],
    active: wp.array[int],
):
    actor = wp.tid()
    if active[0] == 0:
        return
    start, end = spans[actor, 0], spans[actor, 1]
    for i in range(start, end):
        value = rhs[rows[i]]
        for j in range(wp.max(start, i - 10), i):
            value -= factor[i, i - j] * result[rows[j]]
        result[rows[i]] = value / factor[i, 0]
    for reverse in range(end - start):
        i = end - 1 - reverse
        value = result[rows[i]]
        for j in range(i + 1, wp.min(end, i + 11)):
            value -= factor[j, j - i] * result[rows[j]]
        result[rows[i]] = value / factor[i, 0]


class CablePreconditioner:
    """Own fixed band storage and an authenticated per-actor scalar permutation."""

    #: Preconditioner-only diagonal shifts, tried in order until the band factors.
    SHIFTS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

    def __init__(self, layout, order, *, device, dtype=wp.float32, operator=None):
        expected, spans = [], []
        count = 0
        for kind, _, start, xyz, theta in layout.material_ranges:
            if kind != "cable":
                continue
            spans.append((count, count + xyz + theta))
            expected.extend(range(start, start + xyz + theta))
            count += xyz + theta
        order = np.asarray(order)
        if order.shape != (count,) or order.dtype.kind not in "iu":
            raise ValueError("Cable preconditioning requires an integer scalar permutation")
        for start, end in spans:
            if sorted(order[start:end]) != sorted(expected[start:end]):
                raise ValueError("Cable preconditioner permutation must preserve each actor")
        inverse = np.full(layout.scalar_dof_count, -1, dtype=np.int32)
        inverse[order] = np.arange(count)
        self.device = wp.get_device(device)
        self.rows = wp.array(order, dtype=int, device=device)
        self.inverse = wp.array(inverse, dtype=int, device=device)
        self.spans = wp.array(np.asarray(spans, dtype=np.int32).reshape(-1, 2), dtype=int, device=device)
        self.gauge_rows = wp.full(len(spans), -1, dtype=int, device=device)
        self.dtype = dtype
        self.factor_values = wp.zeros((count, 11), dtype=dtype, device=device)
        self.active = wp.ones(1, dtype=int, device=device)
        self.shifts = wp.array(self.SHIFTS, dtype=dtype, device=device)
        self.used_shift = wp.zeros(len(spans), dtype=dtype, device=device)
        self.band = wp.zeros(11 * count, dtype=dtype, device=device)
        self._operator = operator
        self._band_table = None if operator is None else self._build_band_table(operator, inverse, count, device)

    @staticmethod
    def _build_band_table(operator, inverse, count, device):
        """Map each cable's lower band slots to its element entries, in element order."""
        tables = []
        for cable in operator.cables:
            per_slot = {}
            for kind, element, a, b, row_a, row_b in cable.entries():
                i, j = int(inverse[row_a]), int(inverse[row_b])
                if i < 0 or j < 0:
                    raise ValueError("Cable element rows lie outside the cable preconditioner")
                if j > i:
                    continue
                if i - j > 10:
                    raise ValueError("Cable element coupling exceeds the preconditioner band")
                per_slot.setdefault(11 * i + i - j, []).append((kind, element, a, b, row_a, row_b))
            slots = sorted(per_slot)
            offsets = np.zeros(len(slots) + 1, dtype=np.int32)
            offsets[1:] = np.cumsum([len(per_slot[slot]) for slot in slots])
            flat = [entry for slot in slots for entry in per_slot[slot]]
            columns = [np.asarray(c, dtype=np.int32) for c in zip(*flat, strict=True)] if flat else [[]] * 6
            arrays = [np.asarray(slots, dtype=np.int32), offsets, *columns]
            tables.append([wp.array(np.asarray(a, dtype=np.int32), dtype=int, device=device) for a in arrays])
        return tables

    def buffers(self):
        """Return every owned array for generation and graph binding."""
        arrays = (
            self.rows,
            self.inverse,
            self.spans,
            self.gauge_rows,
            self.factor_values,
            self.active,
            self.shifts,
            self.used_shift,
            self.band,
        )
        if self._band_table is not None:
            arrays += tuple(a for table in self._band_table for a in table)
        return arrays

    def signature(self):
        """Bind cached plans to owned topology and numerical storage."""
        return tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in self.buffers())

    def factor(self, workspace, tolerance):
        """Factor each cable principal block without dropping its xyz/theta entries."""
        matrix = workspace.k_global_scalar_bsr
        self.band.zero_()
        if self._band_table is not None:
            for cable, table in zip(self._operator.cables, self._band_table, strict=True):
                if table[0].size:
                    wp.launch(
                        _band,
                        table[0].size,
                        [*table, cable.stretch_matrix, cable.bend_matrix, workspace.scale, self.band],
                        device=self.device,
                    )
        wp.launch(
            _factor,
            self.spans.shape[0],
            [
                matrix.offsets,
                matrix.columns,
                matrix.values,
                workspace.scale,
                self.rows,
                self.inverse,
                self.spans,
                self.gauge_rows,
                self.band,
                self.shifts,
                self.dtype(workspace._lambda),
                self.dtype(tolerance),
                self.factor_values,
                self.used_shift,
                workspace._status,
            ],
            device=self.device,
        )

    def apply(self, rhs, result, *, active=None):
        """Apply the cable factors to disjoint rows of an existing preconditioned vector."""
        wp.launch(
            _apply,
            self.spans.shape[0],
            [
                self.rows,
                self.spans,
                self.factor_values,
                rhs,
                result,
                self.active if active is None else active,
            ],
            device=self.device,
        )
