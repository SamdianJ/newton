# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actor block assembly, Cholesky factors and preconditioner application."""

import numpy as np
import warp as wp

from .bsr import _MonolithicContactFactors
from .linear_types import MonolithicLinearGeneration, MonolithicLinearStatus


@wp.kernel
def _assemble_monolithic_q_preconditioner(
    aq: wp.array2d[float],
    factors: _MonolithicContactFactors,
    scale: wp.array[float],
    regularization: float,
    result: wp.array2d[float],
):
    i, j = wp.tid()
    value = aq[i, j]
    for c in range(factors.count[0]):
        value += factors.weights[c] * factors.gq[c, i] * factors.gq[c, j]
    value *= scale[i] * scale[j]
    if i == j:
        value += regularization
    result[i, j] = value


@wp.kernel
def _assemble_monolithic_x_preconditioner(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    factors: _MonolithicContactFactors,
    q_count: int,
    scale: wp.array[float],
    regularization: float,
    result: wp.array[wp.mat33],
):
    particle = wp.tid()
    block = wp.mat33(0.0)
    for index in range(offsets[particle], offsets[particle + 1]):
        if columns[index] == particle:
            block = values[index]
    for c in range(factors.count[0]):
        g = wp.vec3(0.0)
        for slot in range(9):
            column = factors.gx_columns[c, slot]
            if column >= 3 * particle and column < 3 * particle + 3:
                g[column - 3 * particle] += factors.gx_values[c, slot]
        block += factors.weights[c] * wp.outer(g, g)
    for i in range(3):
        for j in range(3):
            block[i, j] *= scale[q_count + 3 * particle + i] * scale[q_count + 3 * particle + j]
        block[i, i] += regularization
    result[particle] = block


@wp.kernel
def _factor_monolithic_q_cholesky(
    matrix: wp.array2d[float],
    factor: wp.array2d[float],
    tolerance: float,
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
    matrix: wp.array[wp.mat33],
    factor: wp.array[wp.mat33],
    tolerance: float,
    status: wp.array[int],
):
    particle = wp.tid()
    a, lower = matrix[particle], wp.mat33(0.0)
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
def _apply_q_preconditioner(factor: wp.array2d[float], x: wp.array[float], temporary: wp.array[float]):
    n = factor.shape[0]
    for i in range(n):
        value = x[i]
        for j in range(i):
            value -= factor[i, j] * temporary[j]
        temporary[i] = value / factor[i, i]
    for reverse_i in range(n):
        i = n - 1 - reverse_i
        value = temporary[i]
        for j in range(i + 1, n):
            value -= factor[j, i] * temporary[j]
        temporary[i] = value / factor[i, i]


@wp.kernel
def _apply_x_preconditioner(
    factor: wp.array[wp.mat33],
    q_count: int,
    x: wp.array[float],
    temporary: wp.array[float],
):
    particle = wp.tid()
    lower = factor[particle]
    base = q_count + 3 * particle
    result = wp.vec3(0.0)
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
def _combine_vectors(x: wp.array[float], y: wp.array[float], z: wp.array[float], alpha: float, beta: float):
    i = wp.tid()
    value = alpha * x[i]
    if beta != 0.0:
        value += beta * y[i]
    z[i] = value


def _monolithic_preconditioner_matvec(workspace, x, y, z, alpha, beta) -> None:
    if not workspace._factor_valid:
        raise ValueError("Preconditioner must be refactored for current scaling and lambda")
    wp.launch(
        _apply_q_preconditioner, 1, [workspace._q_cholesky, x, workspace._preconditioned], device=workspace.device
    )
    if workspace._study_soft is not None:
        workspace._study_soft.apply(x, workspace._preconditioned)
    elif workspace.layout.dynamic_particle_count:
        wp.launch(
            _apply_x_preconditioner,
            workspace.layout.dynamic_particle_count,
            [workspace._x_cholesky, workspace.layout.q_dof_count, x, workspace._preconditioned],
            device=workspace.device,
        )
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
    if self._segment_graphs is None or self._study_soft is not None:
        self._factor_actor_device(pivot_tolerance)
    else:
        matrix, factors = self.ax_internal_bsr3, self._factors
        self._segment_graphs.run(
            "factor",
            (matrix.nrow, self.layout.q_dof_count, self._lambda, pivot_tolerance),
            (
                self._status,
                self.aq_actor_dense,
                self.scale,
                self._q_preconditioner,
                self._x_preconditioner,
                self._q_cholesky,
                self._x_cholesky,
                matrix.offsets,
                matrix.columns,
                matrix.values,
                factors.count,
                factors.weights,
                factors.gq,
                factors.gx_columns,
                factors.gx_values,
            ),
            lambda: self._factor_actor_device(pivot_tolerance),
        )
    status = self._read_status()
    self._factor_valid = status == MonolithicLinearStatus.SUCCESS
    self.factor_last_status = status
    self.factor_failure_count += int(not self._factor_valid)
    return status


def _factor_actor_device(self, pivot_tolerance):
    self._status.zero_()
    wp.launch(
        _assemble_monolithic_q_preconditioner,
        self.aq_actor_dense.shape,
        [self.aq_actor_dense, self._factors, self.scale, self._lambda, self._q_preconditioner],
        device=self.device,
    )
    matrix = self.ax_internal_bsr3
    if matrix.nrow and self._study_soft is None:
        wp.launch(
            _assemble_monolithic_x_preconditioner,
            matrix.nrow,
            [
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self._factors,
                self.layout.q_dof_count,
                self.scale,
                self._lambda,
                self._x_preconditioner,
            ],
            device=self.device,
        )
    wp.launch(
        _factor_monolithic_q_cholesky,
        1,
        [self._q_preconditioner, self._q_cholesky, pivot_tolerance, self._status],
        device=self.device,
    )
    if self._study_soft is not None:
        self._study_soft.factor(self, pivot_tolerance)
    elif matrix.nrow:
        wp.launch(
            _factor_monolithic_x_cholesky,
            matrix.nrow,
            [self._x_preconditioner, self._x_cholesky, pivot_tolerance, self._status],
            device=self.device,
        )
