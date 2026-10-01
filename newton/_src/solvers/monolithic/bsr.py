# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded BSR assembly, symmetric scaling and matrix-vector products."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import warp.sparse as sparse

from .linear_types import MonolithicLinearGeneration, MonolithicLinearStatus


@wp.struct
class _MonolithicScalarTriplets:
    rows: wp.array[int]
    columns: wp.array[int]
    values: wp.array[float]
    count: wp.array[int]
    status: wp.array[int]
    capacity: int
    row_count: int
    column_count: int
    active_token: wp.array[int]
    token: int


@wp.struct
class _MonolithicMat33Triplets:
    rows: wp.array[int]
    columns: wp.array[int]
    values: wp.array[wp.mat33]
    count: wp.array[int]
    status: wp.array[int]
    capacity: int
    row_count: int
    column_count: int
    active_token: wp.array[int]
    token: int


@wp.struct
class _MonolithicContactFactors:
    offsets: wp.array[int]
    columns: wp.array[int]
    values: wp.array[float]
    local_indices: wp.array[int]
    weights: wp.array[float]
    kind: wp.array[int]
    candidate_tid: wp.array[int]
    count: wp.array[int]
    status: wp.array[int]
    capacity: int
    scalar_dof_count: int
    active_token: wp.array[int]
    token: int


@wp.func
def _append_monolithic_scalar_triplet(triplets: _MonolithicScalarTriplets, row: int, column: int, value: float):
    if triplets.active_token[0] != triplets.token:
        return
    if row < 0 or row >= triplets.row_count or column < 0 or column >= triplets.column_count:
        wp.atomic_max(triplets.status, 0, 3)
        return
    if not wp.isfinite(value):
        wp.atomic_max(triplets.status, 0, 7)
        return
    slot = wp.atomic_add(triplets.count, 0, 1)
    if slot >= triplets.capacity:
        wp.atomic_max(triplets.status, 0, 4)
        return
    triplets.rows[slot] = row
    triplets.columns[slot] = column
    triplets.values[slot] = value


@wp.func
def _accumulate_monolithic_actor_q_entry(
    aq_actor_dense: wp.array2d[float],
    global_triplets: _MonolithicScalarTriplets,
    q_row: int,
    q_column: int,
    value: float,
):
    if global_triplets.active_token[0] != global_triplets.token:
        return
    if q_row < 0 or q_row >= aq_actor_dense.shape[0] or q_column < 0 or q_column >= aq_actor_dense.shape[1]:
        wp.atomic_max(global_triplets.status, 0, 3)
        return
    if not wp.isfinite(value):
        wp.atomic_max(global_triplets.status, 0, 7)
        return
    wp.atomic_add(aq_actor_dense, q_row, q_column, value)
    _append_monolithic_scalar_triplet(global_triplets, q_row, q_column, value)


@wp.func
def _accumulate_monolithic_internal_x_block(
    ax_internal_triplets: _MonolithicMat33Triplets,
    global_triplets: _MonolithicScalarTriplets,
    dynamic_particle_row: int,
    dynamic_particle_column: int,
    value: wp.mat33,
    x_scalar_offset: int,
):
    if (
        ax_internal_triplets.active_token[0] != ax_internal_triplets.token
        or global_triplets.active_token[0] != global_triplets.token
    ):
        return
    if (
        dynamic_particle_row < 0
        or dynamic_particle_row >= ax_internal_triplets.row_count
        or dynamic_particle_column < 0
        or dynamic_particle_column >= ax_internal_triplets.column_count
    ):
        wp.atomic_max(ax_internal_triplets.status, 0, 3)
        return
    for i in range(3):
        for j in range(3):
            if not wp.isfinite(value[i, j]):
                wp.atomic_max(ax_internal_triplets.status, 0, 7)
                return
    slot = wp.atomic_add(ax_internal_triplets.count, 0, 1)
    if slot >= ax_internal_triplets.capacity:
        wp.atomic_max(ax_internal_triplets.status, 0, 5)
        return
    ax_internal_triplets.rows[slot] = dynamic_particle_row
    ax_internal_triplets.columns[slot] = dynamic_particle_column
    ax_internal_triplets.values[slot] = value
    for i in range(3):
        for j in range(3):
            _append_monolithic_scalar_triplet(
                global_triplets,
                x_scalar_offset + 3 * dynamic_particle_row + i,
                x_scalar_offset + 3 * dynamic_particle_column + j,
                value[i, j],
            )


@wp.func
def _reserve_monolithic_contact_factor(factors: _MonolithicContactFactors) -> int:
    if factors.active_token[0] != factors.token:
        return -1
    slot = wp.atomic_add(factors.count, 0, 1)
    if slot >= factors.capacity:
        wp.atomic_max(factors.status, 0, 6)
        return -1
    return slot


@wp.func
def _append_monolithic_contact_factor_triplets(
    factors: _MonolithicContactFactors,
    factor_index: int,
    global_triplets: _MonolithicScalarTriplets,
):
    if factors.active_token[0] != factors.token or global_triplets.active_token[0] != global_triplets.token:
        return
    if factor_index < 0 or factor_index >= factors.capacity:
        wp.atomic_max(factors.status, 0, 3)
        return
    weight = factors.weights[factor_index]
    if not wp.isfinite(weight) or weight < 0.0:
        wp.atomic_max(factors.status, 0, 8)
        return
    begin, end = factors.offsets[factor_index], factors.offsets[factor_index + 1]
    if begin < 0 or end < begin or end > factors.values.shape[0] or end > factors.columns.shape[0]:
        wp.atomic_max(factors.status, 0, 6)
        return
    for i in range(begin, end):
        if factors.columns[i] < 0 or factors.columns[i] >= factors.scalar_dof_count:
            wp.atomic_max(factors.status, 0, 3)
            return
        if not wp.isfinite(factors.values[i]):
            wp.atomic_max(factors.status, 0, 7)
            return
    if weight == 0.0:
        return
    for i in range(begin, end):
        row, gi = factors.columns[i], factors.values[i]
        for j in range(begin, end):
            column, gj = factors.columns[j], factors.values[j]
            _append_monolithic_scalar_triplet(global_triplets, row, column, weight * gi * gj)


@dataclass(frozen=True, slots=True)
class MonolithicLinearAssembly:
    generation: MonolithicLinearGeneration
    aq_actor_dense: wp.array2d[float]
    ax_internal_triplets: _MonolithicMat33Triplets
    contact_factors: _MonolithicContactFactors
    global_scalar_triplets: _MonolithicScalarTriplets


def _create_triplets(struct_type, capacity, size, dtype, device, active_token, token):
    triplets = struct_type()
    triplets.rows = wp.zeros(capacity, dtype=int, device=device)
    triplets.columns = wp.zeros(capacity, dtype=int, device=device)
    triplets.values = wp.zeros(capacity, dtype=dtype, device=device)
    triplets.count = wp.zeros(1, dtype=int, device=device)
    triplets.status = wp.zeros(1, dtype=int, device=device)
    triplets.capacity = capacity
    triplets.row_count = size
    triplets.column_count = size
    triplets.active_token = active_token
    triplets.token = token
    return triplets


@wp.kernel
def _validate_triplets(
    rows: wp.array[int], columns: wp.array[int], values: wp.array[Any], size: int, status: wp.array[int]
):
    index = wp.tid()
    if rows[index] < 0 or rows[index] >= size or columns[index] < 0 or columns[index] >= size:
        wp.atomic_max(status, 0, 3)
    if not wp.isfinite(values[index]):
        wp.atomic_max(status, 0, 7)


@wp.kernel
def _validate_bsr_finite(offsets: wp.array[int], values: wp.array[Any], status: wp.array[int]):
    row = wp.tid()
    for index in range(offsets[row], offsets[row + 1]):
        if not wp.isfinite(values[index]):
            wp.atomic_max(status, 0, 7)


@wp.kernel
def _validate_aq_finite(values: wp.array2d[float], status: wp.array[int]):
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_max(status, 0, 7)


@wp.kernel
def _extract_scalar_diagonal(
    offsets: wp.array[int], columns: wp.array[int], values: wp.array[float], diagonal: wp.array[float]
):
    row = wp.tid()
    value = float(0.0)
    for index in range(offsets[row], offsets[row + 1]):
        if columns[index] == row:
            value = values[index]
    diagonal[row] = value


@wp.kernel
def _build_monolithic_symmetric_scaling(
    matrix_diagonal: wp.array[float],
    dynamic_diagonal: wp.array[float],
    epsilon_d: float,
    diagonal: wp.array[float],
    scale: wp.array[float],
    floor_count: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    raw, dynamic = matrix_diagonal[i], dynamic_diagonal[i]
    floor = epsilon_d * dynamic
    if not wp.isfinite(raw) or not wp.isfinite(dynamic) or not wp.isfinite(floor) or raw <= 0.0 or dynamic <= 0.0:
        wp.atomic_max(status, 0, 10)
        return
    d = wp.max(raw, floor)
    s = 1.0 / wp.sqrt(d)
    if not wp.isfinite(s):
        wp.atomic_max(status, 0, 10)
        return
    diagonal[i] = d
    scale[i] = s
    if floor > raw:
        wp.atomic_add(floor_count, 0, 1)


@wp.kernel
def _scale_vector(scale: wp.array[float], x: wp.array[float], result: wp.array[float]):
    i = wp.tid()
    result[i] = scale[i] * x[i]


@wp.kernel
def _scaled_matvec_finish(
    scale: wp.array[float],
    kx: wp.array[float],
    x: wp.array[float],
    y: wp.array[float],
    result: wp.array[float],
    regularization: float,
    alpha: float,
    beta: float,
):
    i = wp.tid()
    value = alpha * (scale[i] * kx[i] + regularization * x[i])
    if beta != 0.0:
        value += beta * y[i]
    result[i] = value


def _monolithic_scaled_matvec(workspace, x, y, z, alpha, beta) -> None:
    if not workspace._scaling_valid or workspace.layout != workspace._layout_identity:
        raise ValueError("Scaled operator requires current valid scaling")
    wp.launch(_scale_vector, x.size, [workspace.scale, x, workspace._scaled_x], device=workspace.device)
    sparse.bsr_mv(workspace.k_global_scalar_bsr, workspace._scaled_x, workspace._k_scaled)
    wp.launch(
        _scaled_matvec_finish,
        x.size,
        [workspace.scale, workspace._k_scaled, x, y, z, workspace._lambda, alpha, beta],
        device=workspace.device,
    )


def build_scaling(
    self, dynamic_diagonal: wp.array[float], *, epsilon_d: float, generation: MonolithicLinearGeneration
) -> MonolithicLinearStatus:
    """Freeze D and S for the current unregularized assembly."""
    if not self._current(generation):
        return MonolithicLinearStatus.STALE_GENERATION
    self._scaling_valid = False
    self._factor_valid = False
    self._warm_start_key = None
    if not self._vector_valid(dynamic_diagonal) or not np.isfinite(epsilon_d) or epsilon_d <= 0:
        return MonolithicLinearStatus.INVALID_ARGUMENT
    if self._segment_graphs is None:
        self._build_scaling_device(dynamic_diagonal, epsilon_d)
    else:
        matrix = self.k_global_scalar_bsr
        self._segment_graphs.run(
            "scaling",
            (self.layout.identity, matrix.nrow, epsilon_d),
            (
                dynamic_diagonal,
                self.dynamic_diagonal,
                self._status,
                self.floor_count,
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self._matrix_diagonal,
                self.diagonal,
                self.scale,
            ),
            lambda: self._build_scaling_device(dynamic_diagonal, epsilon_d),
        )
    status = self._read_status()
    self._scaling_valid = status == MonolithicLinearStatus.SUCCESS
    self._lambda = 0.0
    return status


def _build_scaling_device(self, dynamic_diagonal, epsilon_d):
    wp.copy(self.dynamic_diagonal, dynamic_diagonal)
    self._status.zero_()
    self.floor_count.zero_()
    matrix = self.k_global_scalar_bsr
    wp.launch(
        _extract_scalar_diagonal,
        matrix.nrow,
        [matrix.offsets, matrix.columns, matrix.values, self._matrix_diagonal],
        device=self.device,
    )
    wp.launch(
        _build_monolithic_symmetric_scaling,
        matrix.nrow,
        [
            self._matrix_diagonal,
            self.dynamic_diagonal,
            epsilon_d,
            self.diagonal,
            self.scale,
            self.floor_count,
            self._status,
        ],
        device=self.device,
    )
