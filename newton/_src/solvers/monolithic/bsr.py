# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded BSR assembly, symmetric scaling and matrix-vector products."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .linear_types import MonolithicLinearGeneration, MonolithicLinearStatus


def _struct(name, fields):
    return wp.struct(type(name, (), {"__annotations__": fields, "__module__": __name__, "__qualname__": name}))


def _triplet_struct(name, value):
    return _struct(
        name,
        {
            "rows": wp.array[int],
            "columns": wp.array[int],
            "values": wp.array[value],
            "count": wp.array[int],
            "status": wp.array[int],
            "capacity": int,
            "row_count": int,
            "column_count": int,
            "active_token": wp.array[int],
            "token": int,
        },
    )


def _factor_struct(name, value):
    return _struct(
        name,
        {
            "offsets": wp.array[int],
            "columns": wp.array[int],
            "values": wp.array[value],
            "local_indices": wp.array[int],
            "weights": wp.array[value],
            "kind": wp.array[int],
            "candidate_tid": wp.array[int],
            "count": wp.array[int],
            "status": wp.array[int],
            "capacity": int,
            "scalar_dof_count": int,
            "active_token": wp.array[int],
            "token": int,
        },
    )


@dataclass(frozen=True, slots=True)
class AssemblyStructs:
    """Assembly struct types whose values are stored in one scalar precision."""

    scalar: Any
    """Scalar COO triplets."""
    mat33: Any
    """Particle 3x3 block COO triplets."""
    factors: Any
    """Contact rank-one factors."""


ASSEMBLY_STRUCTS = {
    scalar: AssemblyStructs(
        _triplet_struct(f"_ScalarTriplets{bits}", scalar),
        _triplet_struct(f"_Mat33Triplets{bits}", matrix),
        _factor_struct(f"_ContactFactors{bits}", scalar),
    )
    for scalar, matrix, bits in ((wp.float32, wp.mat33, 32), (wp.float64, wp.mat33d, 64))
}
"""Assembly structs keyed by the solver scalar type."""


@wp.kernel
def _pack_assembly_counts(
    global_triplets: Any,
    internal_triplets: Any,
    factors: Any,
    packet: wp.array[int],
):
    packet[0] = global_triplets.count[0]
    packet[1] = internal_triplets.count[0]
    packet[2] = factors.count[0]
    packet[3] = global_triplets.status[0]
    packet[4] = internal_triplets.status[0]
    packet[5] = factors.status[0]


@wp.kernel
def _pack_assembly_result(
    status: wp.array[int], global_offsets: wp.array[int], internal_offsets: wp.array[int], packet: wp.array[int]
):
    packet[0] = status[0]
    packet[1] = global_offsets[global_offsets.shape[0] - 1]
    packet[2] = internal_offsets[internal_offsets.shape[0] - 1]


@wp.func
def _append_monolithic_scalar_triplet(triplets: Any, row: int, column: int, value: Any):
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
    aq_actor_dense: wp.array2d[Any],
    global_triplets: Any,
    q_row: int,
    q_column: int,
    value: Any,
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
    ax_internal_triplets: Any,
    global_triplets: Any,
    dynamic_particle_row: int,
    dynamic_particle_column: int,
    value: Any,
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
def _reserve_monolithic_contact_factor(factors: Any) -> int:
    if factors.active_token[0] != factors.token:
        return -1
    slot = wp.atomic_add(factors.count, 0, 1)
    if slot >= factors.capacity:
        wp.atomic_max(factors.status, 0, 6)
        return -1
    return slot


@wp.func
def _append_monolithic_contact_factor_triplets(
    factors: Any,
    factor_index: int,
    global_triplets: Any,
):
    if factors.active_token[0] != factors.token or global_triplets.active_token[0] != global_triplets.token:
        return
    if factor_index < 0 or factor_index >= factors.capacity:
        wp.atomic_max(factors.status, 0, 3)
        return
    weight = factors.weights[factor_index]
    zero = factors.weights.dtype(0.0)
    if not wp.isfinite(weight) or weight < zero:
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
    if weight == zero:
        return
    for i in range(begin, end):
        row, gi = factors.columns[i], factors.values[i]
        for j in range(begin, end):
            column, gj = factors.columns[j], factors.values[j]
            _append_monolithic_scalar_triplet(global_triplets, row, column, weight * gi * gj)


@dataclass(frozen=True, slots=True)
class MonolithicLinearAssembly:
    generation: MonolithicLinearGeneration
    aq_actor_dense: wp.array2d[Any]
    ax_internal_triplets: Any
    contact_factors: Any
    global_scalar_triplets: Any


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
def _validate_aq_finite(values: wp.array2d[Any], status: wp.array[int]):
    row, column = wp.tid()
    if not wp.isfinite(values[row, column]):
        wp.atomic_max(status, 0, 7)


@wp.kernel
def _extract_scalar_diagonal(
    offsets: wp.array[int], columns: wp.array[int], values: wp.array[Any], diagonal: wp.array[Any]
):
    row = wp.tid()
    value = values.dtype(0.0)
    for index in range(offsets[row], offsets[row + 1]):
        if columns[index] == row:
            value = values[index]
    diagonal[row] = value


@wp.func
def _diagonal_scale(
    raw: Any, dynamic: Any, epsilon_d: Any, i: int, diagonal: Any, scale: Any, floor_count: Any, status: Any
):
    zero = type(raw)(0.0)
    floor = epsilon_d * dynamic
    if not wp.isfinite(raw) or not wp.isfinite(dynamic) or not wp.isfinite(floor) or raw <= zero or dynamic <= zero:
        wp.atomic_max(status, 0, 10)
        return
    d = wp.max(raw, floor)
    s = type(raw)(1.0) / wp.sqrt(d)
    if not wp.isfinite(s):
        wp.atomic_max(status, 0, 10)
        return
    diagonal[i] = d
    scale[i] = s
    if floor > raw:
        wp.atomic_add(floor_count, 0, 1)


@wp.kernel
def _build_monolithic_symmetric_scaling(
    matrix_diagonal: wp.array[Any],
    dynamic_diagonal: wp.array[Any],
    epsilon_d: Any,
    diagonal: wp.array[Any],
    scale: wp.array[Any],
    floor_count: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    _diagonal_scale(matrix_diagonal[i], dynamic_diagonal[i], epsilon_d, i, diagonal, scale, floor_count, status)


@wp.kernel
def _build_mixed_scaling(
    matrix_diagonal: wp.array[Any],
    dynamic_input: wp.array[Any],
    coordinate_scale: wp.array[Any],
    coordinate_mask: wp.array[wp.int32],
    epsilon_d: Any,
    dynamic_diagonal: wp.array[Any],
    diagonal: wp.array[Any],
    scale: wp.array[Any],
    floor_count: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    if coordinate_mask[i] != 0:
        # The same explicit change of units as build_coordinate_scaling.
        s = coordinate_scale[i]
        zero = type(s)(0.0)
        d = type(s)(1.0) / (s * s)
        if not wp.isfinite(s) or s <= zero or not wp.isfinite(d) or d <= zero:
            wp.atomic_max(status, 0, 10)
            return
        dynamic_diagonal[i] = d
        diagonal[i] = d
        scale[i] = s
        return
    dynamic_diagonal[i] = dynamic_input[i]
    _diagonal_scale(matrix_diagonal[i], dynamic_input[i], epsilon_d, i, diagonal, scale, floor_count, status)


@wp.kernel
def _scaled_matvec(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    scale: wp.array[Any],
    x: wp.array[Any],
    y: wp.array[Any],
    z: wp.array[Any],
    regularization: Any,
    alpha: float,
    beta: float,
):
    # The same row product as the production PCG operator.
    i = wp.tid()
    total = z.dtype(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        total += values[k] * (scale[j] * x[j])
    value = z.dtype(alpha) * (scale[i] * total + regularization * x[i])
    if beta != 0.0:
        value += z.dtype(beta) * y[i]
    z[i] = value


def _monolithic_scaled_matvec(workspace, x, y, z, alpha, beta) -> None:
    if not workspace._scaling_valid or workspace.layout != workspace._layout_identity:
        raise ValueError("Scaled operator requires current valid scaling")
    workspace._require_precision(x, y, z)
    b = workspace.k_global_scalar_bsr
    wp.launch(
        _scaled_matvec,
        x.size,
        [b.offsets, b.columns, b.values, workspace.scale, x, y, z, workspace.dtype(workspace._lambda), alpha, beta],
        device=workspace.device,
    )


def build_scaling(
    self, dynamic_diagonal: wp.array, *, epsilon_d: float, generation: MonolithicLinearGeneration
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


def build_mixed_scaling(
    self,
    dynamic_diagonal: wp.array,
    coordinate_scale: wp.array,
    coordinate_mask: wp.array[wp.int32],
    *,
    epsilon_d: float,
    generation: MonolithicLinearGeneration,
) -> MonolithicLinearStatus:
    """Freeze D and S with diagonal rows from the assembly and fixed coordinate rows."""
    if not self._current(generation):
        return MonolithicLinearStatus.STALE_GENERATION
    self._scaling_valid = False
    self._factor_valid = False
    self._warm_start_key = None
    if (
        not self._vector_valid(dynamic_diagonal)
        or not self._vector_valid(coordinate_scale)
        or not isinstance(coordinate_mask, wp.array)
        or coordinate_mask.dtype != wp.int32
        or coordinate_mask.device != self.device
        or coordinate_mask.shape != (self.layout.scalar_dof_count,)
        or not np.isfinite(epsilon_d)
        or epsilon_d <= 0
    ):
        return MonolithicLinearStatus.INVALID_ARGUMENT
    inputs = (dynamic_diagonal, coordinate_scale, coordinate_mask)
    if self._segment_graphs is None:
        _build_mixed_scaling_device(self, *inputs, epsilon_d)
    else:
        matrix = self.k_global_scalar_bsr
        self._segment_graphs.run(
            "mixed_scaling",
            (self.layout.identity, matrix.nrow, epsilon_d),
            (
                *inputs,
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
            lambda: _build_mixed_scaling_device(self, *inputs, epsilon_d),
        )
    status = self._read_status()
    self._scaling_valid = status == MonolithicLinearStatus.SUCCESS
    self._lambda = 0.0
    return status


def _build_mixed_scaling_device(self, dynamic_diagonal, coordinate_scale, coordinate_mask, epsilon_d):
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
        _build_mixed_scaling,
        matrix.nrow,
        [
            self._matrix_diagonal,
            dynamic_diagonal,
            coordinate_scale,
            coordinate_mask,
            self.dtype(epsilon_d),
            self.dynamic_diagonal,
            self.diagonal,
            self.scale,
            self.floor_count,
            self._status,
        ],
        device=self.device,
    )


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
            self.dtype(epsilon_d),
            self.diagonal,
            self.scale,
            self.floor_count,
            self._status,
        ],
        device=self.device,
    )
