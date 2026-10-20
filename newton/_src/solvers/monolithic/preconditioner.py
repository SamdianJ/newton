# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actor block assembly, Cholesky factors and preconditioner application."""

from typing import Any

import numpy as np
import warp as wp

from .linear_types import MonolithicLinearGeneration, MonolithicLinearStatus


@wp.kernel
def _assemble_rigid_preconditioner(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    rows: wp.array2d[int],
    scale: wp.array[Any],
    regularization: Any,
    result: wp.array3d[Any],
):
    body, i, j = wp.tid()
    row, column = rows[body, i], rows[body, j]
    value = result.dtype(0.0)
    for k in range(offsets[row], offsets[row + 1]):
        if columns[k] == column:
            value = values[k] * scale[row] * scale[column]
    if i == j:
        value += regularization
    result[body, i, j] = value


@wp.kernel
def _factor_rigid_cholesky(matrix: wp.array3d[Any], factor: wp.array3d[Any], tolerance: Any, status: wp.array[int]):
    body = wp.tid()
    for i in range(6):
        for j in range(i + 1):
            value = matrix[body, i, j]
            for k in range(j):
                value -= factor[body, i, k] * factor[body, j, k]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, 16)
                return
            if i == j:
                if value <= tolerance:
                    wp.atomic_max(status, 0, 12)
                    return
                factor[body, i, j] = wp.sqrt(value)
            else:
                factor[body, i, j] = value / factor[body, j, j]


@wp.kernel
def _apply_rigid_preconditioner(factor: wp.array3d[Any], rows: wp.array2d[int], x: wp.array[Any], out: wp.array[Any]):
    body = wp.tid()
    for i in range(6):
        value = x[rows[body, i]]
        for j in range(i):
            value -= factor[body, i, j] * out[rows[body, j]]
        out[rows[body, i]] = value / factor[body, i, i]
    for k in range(6):
        i = 5 - k
        value = out[rows[body, i]]
        for j in range(i + 1, 6):
            value -= factor[body, j, i] * out[rows[body, j]]
        out[rows[body, i]] = value / factor[body, i, i]


@wp.kernel
def _assemble_monolithic_q_preconditioner(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    q_start: int,
    scale: wp.array[Any],
    regularization: Any,
    result: wp.array2d[Any],
):
    i, j = wp.tid()
    row, column = q_start + i, q_start + j
    value = result.dtype(0.0)
    for index in range(offsets[row], offsets[row + 1]):
        if columns[index] == column:
            value = values[index]
    value *= scale[row] * scale[column]
    if i == j:
        value += regularization
    result[i, j] = value


@wp.kernel
def _assemble_monolithic_x_preconditioner(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    x_start: int,
    scale: wp.array[Any],
    regularization: Any,
    result: wp.array[Any],
):
    particle = wp.tid()
    base = x_start + 3 * particle
    block = result.dtype()
    for i in range(3):
        row = base + i
        for index in range(offsets[row], offsets[row + 1]):
            column = columns[index]
            if column >= base and column < base + 3:
                block[i, column - base] = values[index] * scale[row] * scale[column]
        block[i, i] += regularization
    result[particle] = block


@wp.kernel
def _factor_monolithic_q_cholesky(
    matrix: wp.array2d[Any],
    factor: wp.array2d[Any],
    tolerance: Any,
    status: wp.array[int],
):
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            value = matrix[i, j]
            for k in range(j):
                value -= factor[i, k] * factor[j, k]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, 16)
                return
            if i == j:
                if value <= tolerance:
                    wp.atomic_max(status, 0, 12)
                    return
                factor[i, j] = wp.sqrt(value)
            else:
                factor[i, j] = value / factor[j, j]


@wp.kernel
def _factor_monolithic_x_cholesky(
    matrix: wp.array[Any],
    factor: wp.array[Any],
    tolerance: Any,
    status: wp.array[int],
):
    particle = wp.tid()
    a, lower = matrix[particle], factor.dtype()
    for i in range(3):
        for j in range(i + 1):
            value = a[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            if not wp.isfinite(value):
                wp.atomic_max(status, 0, 16)
                return
            if i == j:
                if value <= tolerance:
                    wp.atomic_max(status, 0, 12)
                    return
                lower[i, j] = wp.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    factor[particle] = lower


@wp.kernel
def _apply_q_preconditioner(factor: wp.array2d[Any], q_start: int, x: wp.array[Any], temporary: wp.array[Any]):
    n = factor.shape[0]
    for i in range(n):
        value = x[q_start + i]
        for j in range(i):
            value -= factor[i, j] * temporary[q_start + j]
        temporary[q_start + i] = value / factor[i, i]
    for reverse_i in range(n):
        i = n - 1 - reverse_i
        value = temporary[q_start + i]
        for j in range(i + 1, n):
            value -= factor[j, i] * temporary[q_start + j]
        temporary[q_start + i] = value / factor[i, i]


@wp.kernel
def _apply_x_preconditioner(
    factor: wp.array[Any],
    x_start: int,
    x: wp.array[Any],
    temporary: wp.array[Any],
):
    particle = wp.tid()
    lower = factor[particle]
    base = x_start + 3 * particle
    result = wp.vector(x[base], x[base + 1], x[base + 2])
    for i in range(3):
        value = x[base + i]
        for j in range(i):
            value -= lower[i, j] * result[j]
        result[i] = value / lower[i, i]
    for reverse_i in range(3):
        i = 2 - reverse_i
        value = result[i]
        for j in range(i + 1, 3):
            value -= lower[j, i] * result[j]
        result[i] = value / lower[i, i]
    for i in range(3):
        temporary[base + i] = result[i]


@wp.kernel
def _combine_vectors(x: wp.array[Any], y: wp.array[Any], z: wp.array[Any], alpha: float, beta: float):
    i = wp.tid()
    value = z.dtype(alpha) * x[i]
    if beta != 0.0:
        value += z.dtype(beta) * y[i]
    z[i] = value


def _monolithic_preconditioner_matvec(workspace, x, y, z, alpha, beta) -> None:
    if not workspace._factor_is_current() or workspace.layout != workspace._layout_identity:
        raise ValueError("Preconditioner must be refactored for current scaling and lambda")
    workspace._require_precision(x, y, z)
    wp.launch(
        _apply_q_preconditioner,
        1,
        [workspace._q_cholesky, workspace.layout.q_scalar_offset, x, workspace._preconditioned],
        device=workspace.device,
    )
    wp.launch(
        _apply_rigid_preconditioner,
        len(workspace.layout.rigid_ranges),
        [workspace._rigid_cholesky, workspace.rigid_rows, x, workspace._preconditioned],
        device=workspace.device,
    )
    if workspace.layout.dynamic_particle_count:
        wp.launch(
            _apply_x_preconditioner,
            workspace.layout.dynamic_particle_count,
            [workspace._x_cholesky, workspace.layout.x_scalar_offset, x, workspace._preconditioned],
            device=workspace.device,
        )
    if workspace._cloth_preconditioner is not None:
        workspace._cloth_preconditioner.apply(x, workspace._preconditioned)
    if workspace._cable_preconditioner is not None:
        workspace._cable_preconditioner.apply(x, workspace._preconditioned)
    wp.launch(_combine_vectors, x.size, [workspace._preconditioned, y, z, alpha, beta], device=workspace.device)


def factor_actor_preconditioner(
    self, *, generation: MonolithicLinearGeneration, pivot_tolerance: float
) -> MonolithicLinearStatus:
    if not self._current(generation) or not self._scaling_valid:
        return MonolithicLinearStatus.STALE_GENERATION
    self._factor_valid = False
    if not np.isfinite(pivot_tolerance) or pivot_tolerance < 0:
        return MonolithicLinearStatus.INVALID_ARGUMENT
    self.factor_setup_count += 1
    if self._segment_graphs is None:
        self._factor_actor_device(pivot_tolerance)
    else:
        matrix = self.k_global_scalar_bsr
        self._segment_graphs.run(
            "factor",
            (
                self.layout.identity,
                matrix.nrow,
                self.layout.q_scalar_offset,
                self.layout.x_scalar_offset,
                self._lambda,
                pivot_tolerance,
            ),
            (
                self._status,
                self.aq_actor_dense,
                self.scale,
                self._q_preconditioner,
                self._x_preconditioner,
                self._q_cholesky,
                self._x_cholesky,
                self._rigid_preconditioner,
                self._rigid_cholesky,
                self.rigid_rows,
                matrix.offsets,
                matrix.columns,
                matrix.values,
            )
            + (() if self._cable_preconditioner is None else self._cable_preconditioner.buffers())
            + (() if self._cloth_preconditioner is None else self._cloth_preconditioner.buffers()),
            lambda: self._factor_actor_device(pivot_tolerance),
        )
    status = self._read_status()
    self._factor_valid = status == MonolithicLinearStatus.SUCCESS
    self._factor_binding = self._factor_storage_binding() if self._factor_valid else None
    self.factor_last_status = status
    self.factor_failure_count += int(not self._factor_valid)
    return status


def _factor_actor_device(self, pivot_tolerance):
    self._status.zero_()
    matrix = self.k_global_scalar_bsr
    regularization, tolerance = self.dtype(self._lambda), self.dtype(pivot_tolerance)
    if self._cloth_preconditioner is not None:
        self._cloth_preconditioner.factor(self, pivot_tolerance)
    if self._cable_preconditioner is not None:
        self._cable_preconditioner.factor(self, pivot_tolerance)
    wp.launch(
        _assemble_rigid_preconditioner,
        (len(self.layout.rigid_ranges), 6, 6),
        [
            matrix.offsets,
            matrix.columns,
            matrix.values,
            self.rigid_rows,
            self.scale,
            regularization,
            self._rigid_preconditioner,
        ],
        device=self.device,
    )
    wp.launch(
        _factor_rigid_cholesky,
        len(self.layout.rigid_ranges),
        [self._rigid_preconditioner, self._rigid_cholesky, tolerance, self._status],
        device=self.device,
    )
    wp.launch(
        _assemble_monolithic_q_preconditioner,
        self.aq_actor_dense.shape,
        [
            matrix.offsets,
            matrix.columns,
            matrix.values,
            self.layout.q_scalar_offset,
            self.scale,
            regularization,
            self._q_preconditioner,
        ],
        device=self.device,
    )
    if self.layout.dynamic_particle_count:
        wp.launch(
            _assemble_monolithic_x_preconditioner,
            self.layout.dynamic_particle_count,
            [
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self.layout.x_scalar_offset,
                self.scale,
                regularization,
                self._x_preconditioner,
            ],
            device=self.device,
        )
    wp.launch(
        _factor_monolithic_q_cholesky,
        1,
        [self._q_preconditioner, self._q_cholesky, tolerance, self._status],
        device=self.device,
    )
    if self.layout.dynamic_particle_count:
        wp.launch(
            _factor_monolithic_x_cholesky,
            self.layout.dynamic_particle_count,
            [self._x_preconditioner, self._x_cholesky, tolerance, self._status],
            device=self.device,
        )
