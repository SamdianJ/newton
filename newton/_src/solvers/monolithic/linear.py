# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded assembly storage for the experimental monolithic solver.

This module assembles supplied contributions only. Scaling, preconditioning and
numerical solving belong to the subsequent solver phase.
"""

import enum
from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.sparse as sparse


class MonolithicLinearStatus(enum.IntEnum):
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    STALE_GENERATION = 2
    TRIPLET_INDEX_OUT_OF_RANGE = 3
    GLOBAL_TRIPLET_OVERFLOW = 4
    INTERNAL_TRIPLET_OVERFLOW = 5
    CONTACT_FACTOR_OVERFLOW = 6
    NONFINITE_CONTRIBUTION = 7
    INVALID_CONTACT_WEIGHT = 8
    BSR_BUILD_FAILURE = 9


@dataclass(frozen=True, slots=True)
class MonolithicLinearLayout:
    q_dof_count: int
    dynamic_particle_count: int

    def __post_init__(self):
        if self.q_dof_count < 0 or self.dynamic_particle_count < 0:
            raise ValueError("DoF counts must be nonnegative")

    @property
    def x_scalar_offset(self) -> int:
        return self.q_dof_count

    @property
    def scalar_dof_count(self) -> int:
        return self.q_dof_count + 3 * self.dynamic_particle_count


@dataclass(frozen=True, slots=True)
class MonolithicLinearCapacities:
    global_scalar_triplet_count: int
    internal_mat33_triplet_count: int
    contact_factor_count: int
    pcg_max_iterations: int
    pcg_true_residual_check_count: int

    def __post_init__(self):
        if (
            min(
                self.global_scalar_triplet_count,
                self.internal_mat33_triplet_count,
                self.contact_factor_count,
                self.pcg_max_iterations,
                self.pcg_true_residual_check_count,
            )
            < 0
        ):
            raise ValueError("Capacities must be nonnegative")


@dataclass(frozen=True, slots=True)
class MonolithicLinearGeneration:
    step_index: int
    nonlinear_iteration: int
    contact_generation: int
    assembly_sequence: int


@dataclass(frozen=True, slots=True)
class MonolithicLinearBuildResult:
    status: MonolithicLinearStatus
    generation: MonolithicLinearGeneration
    global_triplet_count: int
    internal_triplet_count: int
    contact_factor_count: int
    global_nnz: int
    internal_nnz: int


@dataclass(frozen=True, slots=True)
class MonolithicDenseLinearOracle:
    raw_matrix: np.ndarray
    dynamic_diagonal: np.ndarray | None = None
    scale: np.ndarray | None = None
    regularized_matrix: np.ndarray | None = None
    scaled_matrix: np.ndarray | None = None


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
    gq: wp.array2d[float]
    gx_columns: wp.array2d[int]
    gx_values: wp.array2d[float]
    weights: wp.array[float]
    count: wp.array[int]
    status: wp.array[int]
    capacity: int
    q_dof_count: int
    dynamic_particle_count: int
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
    x_scalar_offset: int,
):
    if factors.active_token[0] != factors.token or global_triplets.active_token[0] != global_triplets.token:
        return
    if factor_index < 0 or factor_index >= factors.capacity or factor_index >= factors.count[0]:
        wp.atomic_max(factors.status, 0, 3)
        return
    weight = factors.weights[factor_index]
    if not wp.isfinite(weight) or weight <= 0.0:
        wp.atomic_max(factors.status, 0, 8)
        return
    for i in range(factors.q_dof_count + 9):
        row = i
        gi = float(0.0)
        if i < factors.q_dof_count:
            gi = factors.gq[factor_index, i]
        else:
            local_row = factors.gx_columns[factor_index, i - factors.q_dof_count]
            if local_row == -1:
                continue
            if local_row < 0 or local_row >= 3 * factors.dynamic_particle_count:
                wp.atomic_max(factors.status, 0, 3)
                return
            row = x_scalar_offset + local_row
            gi = factors.gx_values[factor_index, i - factors.q_dof_count]
        for j in range(factors.q_dof_count + 9):
            column = j
            gj = float(0.0)
            if j < factors.q_dof_count:
                gj = factors.gq[factor_index, j]
            else:
                local_column = factors.gx_columns[factor_index, j - factors.q_dof_count]
                if local_column == -1:
                    continue
                if local_column < 0 or local_column >= 3 * factors.dynamic_particle_count:
                    wp.atomic_max(factors.status, 0, 3)
                    return
                column = x_scalar_offset + local_column
                gj = factors.gx_values[factor_index, j - factors.q_dof_count]
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


class MonolithicLinearWorkspace:
    """Own one current assembly; only guarded contribution helpers may mutate it.

    Direct writes to device arrays bypass the assembly contract. Producers must
    use the scatter helpers and reserve contact slots before filling them.
    """

    def __init__(
        self,
        layout: MonolithicLinearLayout,
        capacities: MonolithicLinearCapacities,
        particle_to_dynamic: wp.array[int],
        *,
        device: wp.DeviceLike,
    ):
        self.layout = layout
        self.capacities = capacities
        self.device = wp.get_device(device)
        if (
            particle_to_dynamic.device != self.device
            or particle_to_dynamic.dtype != wp.int32
            or particle_to_dynamic.ndim != 1
        ):
            raise ValueError("particle_to_dynamic must be an int array on the workspace device")
        mapping = particle_to_dynamic.numpy()
        if np.any(mapping < -1) or not np.array_equal(
            np.sort(mapping[mapping >= 0]), np.arange(layout.dynamic_particle_count)
        ):
            raise ValueError("particle_to_dynamic must map each dynamic particle exactly once")
        self.particle_to_dynamic = particle_to_dynamic
        self.aq_actor_dense = wp.zeros((layout.q_dof_count, layout.q_dof_count), dtype=float, device=device)
        self.k_global_scalar_bsr = sparse.bsr_zeros(
            layout.scalar_dof_count, layout.scalar_dof_count, float, device=device
        )
        self.ax_internal_bsr3 = sparse.bsr_zeros(
            layout.dynamic_particle_count, layout.dynamic_particle_count, wp.mat33, device=device
        )
        self._active_token = wp.zeros(1, dtype=int, device=device)
        self._token = 0
        self._generation = None
        self._sealed = True
        self._valid = False
        self._global = _create_triplets(
            _MonolithicScalarTriplets,
            capacities.global_scalar_triplet_count,
            layout.scalar_dof_count,
            float,
            device,
            self._active_token,
            0,
        )
        self._internal = _create_triplets(
            _MonolithicMat33Triplets,
            capacities.internal_mat33_triplet_count,
            layout.dynamic_particle_count,
            wp.mat33,
            device,
            self._active_token,
            0,
        )
        factors = _MonolithicContactFactors()
        factors.gq = wp.zeros((capacities.contact_factor_count, layout.q_dof_count), dtype=float, device=device)
        factors.gx_columns = wp.full((capacities.contact_factor_count, 9), -1, dtype=int, device=device)
        factors.gx_values = wp.zeros((capacities.contact_factor_count, 9), dtype=float, device=device)
        factors.weights = wp.zeros(capacities.contact_factor_count, dtype=float, device=device)
        factors.count = wp.zeros(1, dtype=int, device=device)
        factors.status = wp.zeros(1, dtype=int, device=device)
        factors.capacity = capacities.contact_factor_count
        factors.q_dof_count = layout.q_dof_count
        factors.dynamic_particle_count = layout.dynamic_particle_count
        factors.active_token = self._active_token
        factors.token = 0
        self._factors = factors

    def begin_assembly(self, generation: MonolithicLinearGeneration) -> MonolithicLinearAssembly:
        if self._generation is not None and (generation.step_index, generation.assembly_sequence) <= (
            self._generation.step_index,
            self._generation.assembly_sequence,
        ):
            raise ValueError("Assembly sequence must increase within each step")
        self._token += 1
        self._active_token.fill_(self._token)
        self._generation = generation
        self._sealed = False
        self._valid = False
        self.aq_actor_dense.zero_()
        # Copy only struct descriptors: old views retain their captured token.
        for name, struct_type, fields in [
            (
                "_global",
                _MonolithicScalarTriplets,
                (
                    "rows",
                    "columns",
                    "values",
                    "count",
                    "status",
                    "capacity",
                    "row_count",
                    "column_count",
                    "active_token",
                ),
            ),
            (
                "_internal",
                _MonolithicMat33Triplets,
                (
                    "rows",
                    "columns",
                    "values",
                    "count",
                    "status",
                    "capacity",
                    "row_count",
                    "column_count",
                    "active_token",
                ),
            ),
            (
                "_factors",
                _MonolithicContactFactors,
                (
                    "gq",
                    "gx_columns",
                    "gx_values",
                    "weights",
                    "count",
                    "status",
                    "capacity",
                    "q_dof_count",
                    "dynamic_particle_count",
                    "active_token",
                ),
            ),
        ]:
            previous = getattr(self, name)
            current = struct_type()
            for field in fields:
                setattr(current, field, getattr(previous, field))
            current.token = self._token
            current.count.zero_()
            current.status.zero_()
            setattr(self, name, current)
        self._factors.gq.zero_()
        self._factors.gx_columns.fill_(-1)
        self._factors.gx_values.zero_()
        self._factors.weights.zero_()
        return MonolithicLinearAssembly(generation, self.aq_actor_dense, self._internal, self._factors, self._global)

    def finalize_assembly(self, *, generation: MonolithicLinearGeneration) -> MonolithicLinearBuildResult:
        if generation != self._generation or self._sealed:
            return MonolithicLinearBuildResult(MonolithicLinearStatus.STALE_GENERATION, generation, 0, 0, 0, 0, 0)
        self._sealed = True
        self._active_token.zero_()
        counts = [int(buffer.count.numpy()[0]) for buffer in (self._global, self._internal, self._factors)]
        status = MonolithicLinearStatus.SUCCESS
        for buffer, overflow in [
            (self._global, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW),
            (self._internal, MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW),
            (self._factors, MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW),
        ]:
            code = int(buffer.status.numpy()[0])
            if code:
                status = MonolithicLinearStatus(code)
                break
            if int(buffer.count.numpy()[0]) > buffer.capacity:
                status = overflow
                break
        if status == MonolithicLinearStatus.SUCCESS:
            try:
                for matrix, triplets in [
                    (self.k_global_scalar_bsr, self._global),
                    (self.ax_internal_bsr3, self._internal),
                ]:
                    sparse.bsr_set_from_triplets(
                        matrix, triplets.rows, triplets.columns, triplets.values, count=triplets.count
                    )
            except RuntimeError:
                status = MonolithicLinearStatus.BSR_BUILD_FAILURE
        self._valid = status == MonolithicLinearStatus.SUCCESS
        return MonolithicLinearBuildResult(
            status,
            generation,
            *counts,
            int(self.k_global_scalar_bsr.offsets.numpy()[-1]) if self._valid else 0,
            int(self.ax_internal_bsr3.offsets.numpy()[-1]) if self._valid else 0,
        )

    def densify_for_test(self, *, generation: MonolithicLinearGeneration) -> MonolithicDenseLinearOracle:
        """Copy the finalized production BSR to a host dense debug oracle."""
        if generation != self._generation or not self._sealed or not self._valid:
            raise ValueError("Dense oracle requires the current valid sealed generation")
        matrix = self.k_global_scalar_bsr
        offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
        dense = np.zeros((matrix.nrow, matrix.ncol), dtype=np.float64)
        for row in range(matrix.nrow):
            start, end = offsets[row], offsets[row + 1]
            dense[row, columns[start:end]] = values[start:end]
        return MonolithicDenseLinearOracle(dense)
