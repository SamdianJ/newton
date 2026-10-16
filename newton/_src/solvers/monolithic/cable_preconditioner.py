# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-cable band Cholesky in vertex/edge order; the global operator stays coupled."""

import numpy as np
import warp as wp


@wp.kernel
def _factor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    scale: wp.array[float],
    rows: wp.array[int],
    inverse: wp.array[int],
    spans: wp.array2d[int],
    gauge_rows: wp.array[int],
    regularization: float,
    tolerance: float,
    factor: wp.array2d[float],
    status: wp.array[int],
):
    actor = wp.tid()
    start, end = spans[actor, 0], spans[actor, 1]
    for i in range(start, end):
        row = rows[i]
        for distance in range(11):
            factor[i, distance] = float(0.0)
        for slot in range(offsets[row], offsets[row + 1]):
            column = columns[slot]
            j = inverse[column]
            if j >= start and j <= i:
                if i - j <= 10:
                    factor[i, i - j] = values[slot] * scale[row] * scale[column]
                elif values[slot] != 0.0:
                    wp.atomic_max(status, 0, 1)
                    return
        factor[i, 0] += regularization
        # Lift only a verified instantaneous null direction in the PRECONDITIONER.
        # The physical matrix/residual are unchanged; the delivered theta increment
        # is projected to zero mean by the common-angle constraint.
        if row == gauge_rows[actor]:
            factor[i, 0] += float(1.0)
        for j in range(wp.max(start, i - 10), i + 1):
            value = factor[i, i - j]
            for k in range(wp.max(start, i - 10), j):
                value -= factor[i, i - k] * factor[j, j - k]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, 16)
                return
            if i == j:
                if value <= tolerance:
                    wp.atomic_max(status, 0, 12)
                    return
                factor[i, 0] = wp.sqrt(value)
            else:
                factor[i, i - j] = value / factor[j, 0]


@wp.kernel
def _apply(
    rows: wp.array[int],
    spans: wp.array2d[int],
    factor: wp.array2d[float],
    rhs: wp.array[float],
    result: wp.array[float],
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

    def __init__(self, layout, order, *, device):
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
        self.factor_values = wp.zeros((count, 11), dtype=float, device=device)
        self.active = wp.ones(1, dtype=int, device=device)

    def buffers(self):
        """Return every owned array for generation and graph binding."""
        return (self.rows, self.inverse, self.spans, self.gauge_rows, self.factor_values, self.active)

    def signature(self):
        """Bind cached plans to owned topology and numerical storage."""
        return tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in self.buffers())

    def factor(self, workspace, tolerance):
        """Factor each cable principal block without dropping its xyz/theta entries."""
        matrix = workspace.k_global_scalar_bsr
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
                workspace._lambda,
                tolerance,
                self.factor_values,
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
