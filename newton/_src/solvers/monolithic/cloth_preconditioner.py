# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-vertex cloth blocks; all inter-vertex terms remain in the global matrix."""

from typing import Any

import warp as wp


@wp.kernel
def _factor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    rows: wp.array[int],
    scale: wp.array[Any],
    regularization: Any,
    tolerance: Any,
    factors: wp.array[Any],
    status: wp.array[int],
):
    p = wp.tid()
    start = rows[p]
    matrix, lower = factors.dtype(), factors.dtype()
    for i in range(3):
        row = start + i
        for k in range(offsets[row], offsets[row + 1]):
            column = columns[k]
            if column >= start and column < start + 3:
                matrix[i, column - start] = values[k] * scale[row] * scale[column]
        matrix[i, i] += regularization
    for i in range(3):
        for j in range(i + 1):
            value = matrix[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, 16)
                return
            if i == j:
                if value <= tolerance:
                    wp.atomic_max(status, 0, 12)
                    return
                lower[i, i] = wp.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    factors[p] = lower


@wp.kernel
def _apply(
    rows: wp.array[int],
    factors: wp.array[Any],
    rhs: wp.array[Any],
    result: wp.array[Any],
    active: wp.array[int],
):
    p = wp.tid()
    if active[0] == 0:
        return
    start, lower = rows[p], factors[p]
    x = wp.vector(rhs[start], rhs[start + 1], rhs[start + 2])
    for i in range(3):
        value = rhs[start + i]
        for j in range(i):
            value -= lower[i, j] * x[j]
        x[i] = value / lower[i, i]
    for k in range(3):
        i = 2 - k
        value = x[i]
        for j in range(i + 1, 3):
            value -= lower[j, i] * x[j]
        x[i] = value / lower[i, i]
    for i in range(3):
        result[start + i] = x[i]


class ClothPreconditioner:
    """Bind fixed cloth row maps and numerical storage to PCG generations."""

    def __init__(self, layout, *, device, dtype=wp.float32):
        self.device = wp.get_device(device)
        rows = [
            row
            for kind, _, start, xyz, _ in layout.material_ranges
            if kind == "cloth"
            for row in range(start, start + xyz, 3)
        ]
        self.rows = wp.array(rows, dtype=int, device=device)
        self.dtype = dtype
        matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
        self.factors = wp.zeros(len(rows), dtype=matrix, device=device)
        self.active = wp.ones(1, dtype=int, device=device)

    def buffers(self):
        """Return all owned graph-bound storage."""
        return self.rows, self.factors, self.active

    def signature(self):
        """Authenticate row maps and factor allocation."""
        return tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in self.buffers())

    def factor(self, workspace, tolerance):
        """Factor the full three-coordinate principal block for each vertex."""
        b = workspace.k_global_scalar_bsr
        wp.launch(
            _factor,
            self.rows.size,
            [
                b.offsets,
                b.columns,
                b.values,
                self.rows,
                workspace.scale,
                self.dtype(workspace._lambda),
                self.dtype(tolerance),
                self.factors,
                workspace._status,
            ],
            device=self.device,
        )

    def apply(self, rhs, result, *, active=None):
        """Apply factors only while the common PCG solve remains active."""
        wp.launch(
            _apply,
            self.rows.size,
            [self.rows, self.factors, rhs, result, self.active if active is None else active],
            device=self.device,
        )
