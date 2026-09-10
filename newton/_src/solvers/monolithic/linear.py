# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded assembly, symmetric scaling and private PCG for monolithic stepping."""

import enum
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import warp as wp
import warp.sparse as sparse
from warp.optim.linear import LinearOperator


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
    INVALID_DIAGONAL = 10
    PRECONDITIONER_NOT_FACTORED = 11
    NON_POSITIVE_PIVOT = 12
    NON_POSITIVE_CURVATURE = 13
    NEAR_ZERO_CURVATURE = 14
    NON_POSITIVE_PRECONDITIONED_RESIDUAL = 15
    NONFINITE_ITERATION = 16
    MAX_ITERATIONS = 17
    STAGNATION = 18


class MonolithicPcgWarmStart(enum.IntEnum):
    ZERO = 0
    SAME_GENERATION = 1


class MonolithicContactFactorKind(enum.IntEnum):
    """Private scalar factor tags for normal and tangential contributions."""

    NONE = 0
    NORMAL = 1
    TANGENT = 2


@dataclass(frozen=True, slots=True)
class MonolithicPcgConfig:
    maximum_iterations: int
    true_residual_interval: int
    linear_tolerance: float
    residual_floor_global: float
    residual_floor_q: float
    residual_floor_x: float
    curvature_absolute_tolerance: float
    curvature_relative_tolerance: float
    preconditioner_positive_tolerance: float
    stagnation_window: int
    stagnation_minimum_reduction: float

    def __post_init__(self):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.maximum_iterations, self.true_residual_interval, self.stagnation_window)
        ):
            raise ValueError("PCG iteration counts and stagnation window must be positive integers")
        positive = (self.linear_tolerance, self.residual_floor_global, self.residual_floor_q, self.residual_floor_x)
        nonnegative = (
            self.curvature_absolute_tolerance,
            self.curvature_relative_tolerance,
            self.preconditioner_positive_tolerance,
            self.stagnation_minimum_reduction,
        )
        if (
            any(
                not np.isfinite(value)
                or value < np.finfo(np.float32).smallest_subnormal
                or value > np.finfo(np.float32).max
                for value in positive
            )
            or any(not np.isfinite(value) or value < 0 or value > np.finfo(np.float32).max for value in nonnegative)
            or self.stagnation_minimum_reduction >= 1
        ):
            raise ValueError("PCG tolerances must be finite with positive residual floors")


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
        for value in (
            self.global_scalar_triplet_count,
            self.internal_mat33_triplet_count,
            self.contact_factor_count,
            self.pcg_max_iterations,
            self.pcg_true_residual_check_count,
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value < 2**31
            ):
                raise ValueError("Capacities must be nonnegative int32 integers")

    @classmethod
    def for_p1(cls, layout, *, tet_count, static_pair_count, friction, pcg_max_iterations):
        """Bound P1 storage arithmetically before allocating contact physics.

        Each pair has three history slots and each slot has one normal factor
        plus two tangential factors when friction is enabled. Python integer
        arithmetic avoids overflow before the final int32 capacity check.
        """
        counts = (layout.q_dof_count, layout.dynamic_particle_count, tet_count, static_pair_count, pcg_max_iterations)
        if type(friction) is not bool or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in counts
        ):
            raise ValueError("P1 capacity counts must be nonnegative integers and friction must be bool")
        q, n, t, pairs, iterations = map(int, counts)
        factors = 3 * pairs * (3 if friction else 1)
        internal = n + 16 * t
        return cls(q * q + 9 * internal + factors * (q + 9) ** 2, internal, factors, iterations, iterations + 2)


@dataclass(frozen=True, slots=True)
class MonolithicLinearGeneration:
    step_index: int
    nonlinear_iteration: int
    contact_generation: int
    assembly_sequence: int
    config_generation: int = 0
    history_epoch: int = 0


@dataclass(frozen=True, slots=True)
class MonolithicLinearSolveResult:
    status: MonolithicLinearStatus
    generation: MonolithicLinearGeneration
    iterations: int
    true_residual_checks: int
    residual_replacements: int
    rho: float
    rho_q: float
    rho_x: float
    min_p_ap: float
    min_r_z: float
    warm_start: MonolithicPcgWarmStart | None = None
    initial_guess_norm: float = float("nan")
    """Measured L2 norm of the initial scaled unknown y; NaN before a solve starts."""
    recursive_true_residual_gap: tuple[float, float, float] = (float("nan"),) * 3
    """Maximum scaled L2 recursive-minus-true gap in global/q/x slices before replacement."""
    stagnation_window: int | None = None
    """Configured stagnation window in true-residual checks; None before a solve starts."""


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
    kind: wp.array[int]
    candidate_tid: wp.array[int]
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
    if not workspace._scaling_valid:
        raise ValueError("Scaled operator requires current valid scaling")
    wp.launch(_scale_vector, x.size, [workspace.scale, x, workspace._scaled_x], device=workspace.device)
    sparse.bsr_mv(workspace.k_global_scalar_bsr, workspace._scaled_x, workspace._k_scaled)
    wp.launch(
        _scaled_matvec_finish,
        x.size,
        [workspace.scale, workspace._k_scaled, x, y, z, workspace._lambda, alpha, beta],
        device=workspace.device,
    )


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
    if workspace.layout.dynamic_particle_count:
        wp.launch(
            _apply_x_preconditioner,
            workspace.layout.dynamic_particle_count,
            [workspace._x_cholesky, workspace.layout.q_dof_count, x, workspace._preconditioned],
            device=workspace.device,
        )
    wp.launch(_combine_vectors, x.size, [workspace._preconditioned, y, z, alpha, beta], device=workspace.device)


@wp.kernel
def _pcg_products(a: wp.array[float], b: wp.array[float], products: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    av, bv = a[i], b[i]
    if not wp.isfinite(av) or not wp.isfinite(bv):
        wp.atomic_max(status, 0, 16)
    wp.atomic_add(products, 0, av * bv)
    wp.atomic_add(products, 1, av * av)
    wp.atomic_add(products, 2, bv * bv)


@wp.kernel
def _validate_monolithic_preconditioned_residual(
    products: wp.array[float],
    tolerance: float,
    r_z: wp.array[float],
    status: wp.array[int],
):
    value = products[0]
    if not wp.isfinite(value) or not wp.isfinite(products[1]) or not wp.isfinite(products[2]):
        status[0] = 16
    elif value <= tolerance:
        status[0] = 15
    else:
        r_z[0] = value


@wp.kernel
def _compute_monolithic_pcg_alpha(
    products: wp.array[float],
    r_z: wp.array[float],
    absolute_tolerance: float,
    relative_tolerance: float,
    alpha: wp.array[float],
    status: wp.array[int],
):
    curvature = products[0]
    if not wp.isfinite(curvature) or not wp.isfinite(products[1]) or not wp.isfinite(products[2]):
        status[0] = 16
    elif curvature <= 0.0:
        status[0] = 13
    elif curvature <= absolute_tolerance + relative_tolerance * wp.sqrt(products[1]) * wp.sqrt(products[2]):
        status[0] = 14
    else:
        value = r_z[0] / curvature
        if not wp.isfinite(value):
            status[0] = 16
        else:
            alpha[0] = value


@wp.kernel
def _update_monolithic_pcg_solution_residual(
    y: wp.array[float],
    r: wp.array[float],
    p: wp.array[float],
    ap: wp.array[float],
    alpha: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    yi, ri = y[i] + alpha[0] * p[i], r[i] - alpha[0] * ap[i]
    if not wp.isfinite(yi) or not wp.isfinite(ri):
        wp.atomic_max(status, 0, 16)
        return
    y[i] = yi
    r[i] = ri


@wp.kernel
def _compute_monolithic_pcg_beta(
    r_z: wp.array[float],
    old_r_z: wp.array[float],
    beta: wp.array[float],
    status: wp.array[int],
):
    if status[0] != 0:
        return
    if not wp.isfinite(old_r_z[0]) or old_r_z[0] <= 0.0 or not wp.isfinite(r_z[0]) or r_z[0] <= 0.0:
        status[0] = 15
        return
    value = r_z[0] / old_r_z[0]
    if not wp.isfinite(value):
        status[0] = 16
    else:
        beta[0] = value


@wp.kernel
def _pcg_update_direction(z: wp.array[float], p: wp.array[float], beta: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    if status[0] != 0:
        return
    value = z[i] + beta[0] * p[i]
    if not wp.isfinite(value):
        wp.atomic_max(status, 0, 16)
    p[i] = value


@wp.struct
class _MonolithicPcgPacket:
    status: int
    check: int
    ratios: wp.vec3


@wp.kernel
def _pack_monolithic_pcg_residual(
    status: wp.array[int], ratios: wp.array[float], packet: wp.array[_MonolithicPcgPacket]
):
    value = _MonolithicPcgPacket()
    value.status = status[0]
    value.ratios = wp.vec3(ratios[0], ratios[1], ratios[2])
    packet[0] = value


@wp.kernel
def _cache_monolithic_pcg_denominator(norms: wp.array[float], floor: wp.float64, denominator: wp.array[wp.float64]):
    denominator[0] = wp.max(wp.float64(norms[3]), floor)


@wp.kernel
def _pack_monolithic_pcg_check(
    products: wp.array[float],
    status: wp.array[int],
    denominator: wp.array[wp.float64],
    tolerance: wp.float64,
    scheduled: bool,
    packet: wp.array[_MonolithicPcgPacket],
):
    value = _MonolithicPcgPacket()
    value.status = status[0]
    # The old host decision used Python doubles, including the configured floor.
    threshold = tolerance * denominator[0]
    value.check = int(scheduled or wp.float64(products[0]) <= threshold * threshold)
    packet[0] = value


@wp.kernel
def _record_monolithic_pcg_product(products: wp.array[float], summary: wp.array[float], index: int, first: bool):
    value = products[0]
    old = summary[index]
    # Preserve Python min's NaN ordering and the old curvature initialization.
    if first or (index == 0 and wp.isnan(old)) or value < old:
        summary[index] = value


@wp.kernel
def _record_monolithic_pcg_norms(norms: wp.array[float], summary: wp.array[float], initial: bool, first: bool):
    if initial:
        summary[2] = norms[0]
    else:
        for i in range(3):
            old = float(summary[3 + i])
            value = float(norms[i])
            if first:
                summary[3 + i] = value
            elif not wp.isfinite(old) or not wp.isfinite(value):
                summary[3 + i] = wp.nan
            elif value > old:
                summary[3 + i] = value


@wp.kernel
def _true_residual_maxima(
    ay: wp.array[float],
    rhs: wp.array[float],
    q_count: int,
    residual: wp.array[float],
    maxima: wp.array[float],
    status: wp.array[int],
    compute_rhs: bool,
):
    i = wp.tid()
    value = rhs[i] - ay[i]
    residual[i] = value
    if not wp.isfinite(value) or not wp.isfinite(rhs[i]):
        wp.atomic_max(status, 0, 16)
        return
    block = int(1)
    if i >= q_count:
        block = 2
    wp.atomic_max(maxima, 0, wp.abs(value))
    wp.atomic_max(maxima, block, wp.abs(value))
    if compute_rhs:
        wp.atomic_max(maxima, 3, wp.abs(rhs[i]))
        wp.atomic_max(maxima, block + 3, wp.abs(rhs[i]))


@wp.kernel
def _true_residual_sums(
    residual: wp.array[float],
    rhs: wp.array[float],
    q_count: int,
    maxima: wp.array[float],
    sums: wp.array[float],
    status: wp.array[int],
    compute_rhs: bool,
):
    i = wp.tid()
    if status[0] != 0:
        return
    block = int(1)
    if i >= q_count:
        block = 2
    # Each slice has its own scale so a large x block cannot erase a tiny q block.
    for local in range(2):
        index = int(0)
        if local == 1:
            index = block
        if maxima[index] > 0.0:
            value = residual[i] / maxima[index]
            wp.atomic_add(sums, index, value * value)
        if compute_rhs and maxima[index + 3] > 0.0:
            value = rhs[i] / maxima[index + 3]
            wp.atomic_add(sums, index + 3, value * value)


@wp.kernel
def _compute_true_residual_norms(
    maxima: wp.array[float],
    sums: wp.array[float],
    norms: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    value = maxima[i] * wp.sqrt(sums[i])
    if not wp.isfinite(value) or not wp.isfinite(sums[i]):
        wp.atomic_max(status, 0, 16)
        return
    norms[i] = value


@wp.kernel
def _compute_monolithic_true_residual_ratios(
    norms: wp.array[float],
    floor_global: float,
    floor_q: float,
    floor_x: float,
    ratios: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    floor = floor_global
    if i == 1:
        floor = floor_q
    elif i == 2:
        floor = floor_x
    value = norms[i] / wp.max(norms[i + 3], floor)
    ratios[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(status, 0, 16)


class MonolithicLinearWorkspace:
    """Own one current assembly with fixed producer slots and guarded appends.

    Fixed-slot producers write the current assembly's reserved prefix. Contact
    producers reserve slots with guarded helpers. Raw array writes require the
    caller to enforce the current generation; only helper writes are token-guarded.
    """

    def __init__(
        self,
        layout: MonolithicLinearLayout,
        capacities: MonolithicLinearCapacities,
        particle_to_dynamic: wp.array[int],
        *,
        device: wp.DeviceLike,
        pcg_mode: str = "diagnostic",
    ):
        if pcg_mode not in ("diagnostic", "production"):
            raise ValueError("pcg_mode must be diagnostic or production")
        self._pcg_mode = pcg_mode
        self._pcg_active = False
        self._rhs_norms_cached = False
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
        self._fixed_patterns = None
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
        factors.kind = wp.zeros(capacities.contact_factor_count, dtype=int, device=device)
        factors.candidate_tid = wp.full(capacities.contact_factor_count, -1, dtype=int, device=device)
        factors.count = wp.zeros(1, dtype=int, device=device)
        factors.status = wp.zeros(1, dtype=int, device=device)
        factors.capacity = capacities.contact_factor_count
        factors.q_dof_count = layout.q_dof_count
        factors.dynamic_particle_count = layout.dynamic_particle_count
        factors.active_token = self._active_token
        factors.token = 0
        self._factors = factors
        n = layout.scalar_dof_count
        for name in (
            "dynamic_diagonal",
            "diagonal",
            "scale",
            "_matrix_diagonal",
            "_scaled_x",
            "_k_scaled",
            "_preconditioned",
            "_r",
            "_z",
            "_p",
            "_ap",
            "_true_r",
        ):
            setattr(self, name, wp.zeros(n, dtype=float, device=device))
        for name in ("_r_z", "_old_r_z", "_alpha", "_beta"):
            setattr(self, name, wp.zeros(1, dtype=float, device=device))
        self._products = wp.zeros(3, dtype=float, device=device)
        self._true_sums = wp.zeros(6, dtype=float, device=device)
        self._true_maxima = wp.zeros(6, dtype=float, device=device)
        self._true_norms = wp.zeros(6, dtype=float, device=device)
        self._true_residual_norms = self._true_norms[:3]
        if self.pcg_mode == "diagnostic":
            self._diagnostic_vector = wp.zeros(n, dtype=float, device=device)
            self._diagnostic_zero = wp.zeros(n, dtype=float, device=device)
            self._diagnostic_maxima = wp.zeros(6, dtype=float, device=device)
            self._diagnostic_sums = wp.zeros(6, dtype=float, device=device)
            self._diagnostic_norms = wp.zeros(6, dtype=float, device=device)
            self._diagnostic_status = wp.zeros(1, dtype=int, device=device)
            self._diagnostic_summary = wp.zeros(6, dtype=float, device=device)
        self._pcg_packet = wp.zeros(1, dtype=_MonolithicPcgPacket, device=device)
        self._rhs_denominator = wp.zeros(1, dtype=wp.float64, device=device)
        self.factor_setup_count = 0
        self.factor_failure_count = 0
        self.factor_last_status = None
        self._ratios = wp.zeros(3, dtype=float, device=device)
        self._status = wp.zeros(1, dtype=int, device=device)
        self.floor_count = wp.zeros(1, dtype=int, device=device)
        self._q_preconditioner = wp.zeros_like(self.aq_actor_dense)
        self._q_cholesky = wp.zeros_like(self.aq_actor_dense)
        self._x_preconditioner = wp.zeros(layout.dynamic_particle_count, dtype=wp.mat33, device=device)
        self._x_cholesky = wp.zeros_like(self._x_preconditioner)
        self._scaling_valid = False
        self._factor_valid = False
        self._lambda = 0.0
        self._warm_start_key = None
        self.operator = LinearOperator((n, n), wp.float32, self.device, partial(_monolithic_scaled_matvec, self))
        self.preconditioner = LinearOperator(
            (n, n), wp.float32, self.device, partial(_monolithic_preconditioner_matvec, self)
        )

    def _set_fixed_triplet_pattern(
        self,
        *,
        global_rows: np.ndarray,
        global_columns: np.ndarray,
        internal_rows: np.ndarray,
        internal_columns: np.ndarray,
    ) -> None:
        """Reserve immutable owner patterns once, before the first assembly."""
        if self._generation is not None or self._fixed_patterns is not None:
            raise ValueError("Fixed triplet patterns must be configured once before assembly")
        patterns = []
        for row_indices, column_indices, buffer in [
            (global_rows, global_columns, self._global),
            (internal_rows, internal_columns, self._internal),
        ]:
            rows, columns = np.asarray(row_indices), np.asarray(column_indices)
            if (
                rows.ndim != 1
                or columns.shape != rows.shape
                or not np.issubdtype(rows.dtype, np.integer)
                or not np.issubdtype(columns.dtype, np.integer)
                or rows.size > buffer.capacity
                or np.any(rows < 0)
                or np.any(rows >= buffer.row_count)
                or np.any(columns < 0)
                or np.any(columns >= buffer.column_count)
            ):
                raise ValueError("Fixed triplet pattern must contain in-range integer pairs within capacity")
            patterns.append((rows.astype(np.int32, copy=True), columns.astype(np.int32, copy=True)))
        self._fixed_patterns = tuple(
            tuple(wp.array(indices, dtype=int, device=self.device) for indices in pattern) for pattern in patterns
        )

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
        self.factor_setup_count = 0
        self.factor_failure_count = 0
        self.factor_last_status = None
        self._scaling_valid = False
        self._factor_valid = False
        self._warm_start_key = None
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
                    "kind",
                    "candidate_tid",
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
        for index, buffer in enumerate((self._global, self._internal)):
            buffer.values.zero_()
            if self._fixed_patterns is not None:
                rows, columns = self._fixed_patterns[index]
                count = rows.size
                if count:
                    wp.copy(buffer.rows, rows, count=count)
                    wp.copy(buffer.columns, columns, count=count)
                buffer.count.fill_(count)
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
        for index, (buffer, overflow) in enumerate(
            [
                (self._global, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW),
                (self._internal, MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW),
                (self._factors, MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW),
            ]
        ):
            code = int(buffer.status.numpy()[0])
            if code:
                status = MonolithicLinearStatus(code)
                break
            minimum = self._fixed_patterns[index][0].size if self._fixed_patterns is not None and index < 2 else 0
            if counts[index] < minimum:
                status = MonolithicLinearStatus.INVALID_ARGUMENT
                break
            if counts[index] > buffer.capacity:
                status = overflow
                break
        if status == MonolithicLinearStatus.SUCCESS:
            # Fixed-slot kernels write raw arrays, bypassing append-time validation.
            for buffer, count in zip((self._global, self._internal), counts[:2], strict=True):
                if count:
                    wp.launch(
                        _validate_triplets,
                        count,
                        [buffer.rows, buffer.columns, buffer.values, buffer.row_count, self._global.status],
                        device=self.device,
                    )
            status = MonolithicLinearStatus(int(self._global.status.numpy()[0]))
        if status == MonolithicLinearStatus.SUCCESS:
            try:
                for (matrix, triplets), count in zip(
                    [(self.k_global_scalar_bsr, self._global), (self.ax_internal_bsr3, self._internal)],
                    counts[:2],
                    strict=True,
                ):
                    arrays = (triplets.rows, triplets.columns, triplets.values)
                    # Warp sorts by array length; device count only masks the tail.
                    # Keep empty-capacity arrays intact: slicing them is unsupported.
                    if count < triplets.capacity:
                        arrays = tuple(array[:count] for array in arrays)
                    sparse.bsr_set_from_triplets(matrix, *arrays, count=triplets.count)
            except RuntimeError:
                status = MonolithicLinearStatus.BSR_BUILD_FAILURE
        if status == MonolithicLinearStatus.SUCCESS:
            # Finite local terms can overflow during owner or duplicate reductions.
            for matrix in (self.k_global_scalar_bsr, self.ax_internal_bsr3):
                if matrix.nrow:
                    wp.launch(
                        _validate_bsr_finite,
                        matrix.nrow,
                        [matrix.offsets, matrix.values, self._global.status],
                        device=self.device,
                    )
            wp.launch(
                _validate_aq_finite,
                self.aq_actor_dense.shape,
                [self.aq_actor_dense, self._global.status],
                device=self.device,
            )
            status = MonolithicLinearStatus(int(self._global.status.numpy()[0]))
        self._valid = status == MonolithicLinearStatus.SUCCESS
        return MonolithicLinearBuildResult(
            status,
            generation,
            *counts,
            int(self.k_global_scalar_bsr.offsets.numpy()[-1]) if self._valid else 0,
            int(self.ax_internal_bsr3.offsets.numpy()[-1]) if self._valid else 0,
        )

    def _current(self, generation) -> bool:
        return generation == self._generation and self._sealed and self._valid

    def _vector_valid(self, vector) -> bool:
        return (
            isinstance(vector, wp.array)
            and vector.dtype == wp.float32
            and vector.device == self.device
            and vector.shape == (self.layout.scalar_dof_count,)
            and vector.is_contiguous
        )

    def _read_status(self) -> MonolithicLinearStatus:
        return MonolithicLinearStatus(int(self._status.numpy()[0]))

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
        status = self._read_status()
        self._scaling_valid = status == MonolithicLinearStatus.SUCCESS
        self._lambda = 0.0
        return status

    def set_regularization(
        self, lambda_value: float, *, generation: MonolithicLinearGeneration
    ) -> MonolithicLinearStatus:
        if not self._current(generation) or not self._scaling_valid:
            return MonolithicLinearStatus.STALE_GENERATION
        if not np.isfinite(lambda_value) or lambda_value < 0 or lambda_value > np.finfo(np.float32).max:
            return MonolithicLinearStatus.INVALID_ARGUMENT
        if lambda_value != self._lambda:
            self._factor_valid = False
        self._lambda = float(lambda_value)
        return MonolithicLinearStatus.SUCCESS

    def factor_actor_preconditioner(
        self, *, generation: MonolithicLinearGeneration, pivot_tolerance: float
    ) -> MonolithicLinearStatus:
        if not self._current(generation) or not self._scaling_valid:
            return MonolithicLinearStatus.STALE_GENERATION
        self._factor_valid = False
        if not np.isfinite(pivot_tolerance) or pivot_tolerance < 0:
            return MonolithicLinearStatus.INVALID_ARGUMENT
        self.factor_setup_count += 1
        self._status.zero_()
        wp.launch(
            _assemble_monolithic_q_preconditioner,
            self.aq_actor_dense.shape,
            [self.aq_actor_dense, self._factors, self.scale, self._lambda, self._q_preconditioner],
            device=self.device,
        )
        matrix = self.ax_internal_bsr3
        if matrix.nrow:
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
        if matrix.nrow:
            wp.launch(
                _factor_monolithic_x_cholesky,
                matrix.nrow,
                [self._x_preconditioner, self._x_cholesky, pivot_tolerance, self._status],
                device=self.device,
            )
        status = self._read_status()
        self._factor_valid = status == MonolithicLinearStatus.SUCCESS
        self.factor_last_status = status
        self.factor_failure_count += int(not self._factor_valid)
        return status

    @property
    def pcg_mode(self) -> str:
        """Return the immutable execution mode selected at construction."""
        return self._pcg_mode

    def _diagnostic_block_norms(self, vector) -> None:
        """Measure scaled vector norms without altering iterative solver status or scratch."""
        if vector is not self._diagnostic_vector:
            wp.copy(self._diagnostic_vector, vector)
        self._diagnostic_status.zero_()
        self._diagnostic_maxima.zero_()
        self._diagnostic_sums.zero_()
        self._diagnostic_norms.fill_(float("nan"))
        wp.launch(
            _true_residual_maxima,
            vector.size,
            [
                self._diagnostic_zero,
                self._diagnostic_vector,
                self.layout.q_dof_count,
                self._diagnostic_vector,
                self._diagnostic_maxima,
                self._diagnostic_status,
                True,
            ],
            device=self.device,
        )
        wp.launch(
            _true_residual_sums,
            vector.size,
            [
                self._diagnostic_vector,
                self._diagnostic_vector,
                self.layout.q_dof_count,
                self._diagnostic_maxima,
                self._diagnostic_sums,
                self._diagnostic_status,
                True,
            ],
            device=self.device,
        )
        wp.launch(
            _compute_true_residual_norms,
            6,
            [self._diagnostic_maxima, self._diagnostic_sums, self._diagnostic_norms, self._diagnostic_status],
            device=self.device,
        )

    def _products_of(self, a, b):
        self._products.zero_()
        wp.launch(_pcg_products, a.size, [a, b, self._products, self._status], device=self.device)

    def _true_residual(self, rhs, y, config):
        self._status.zero_()
        self.operator.matvec(y, self._ap, self._ap, 1.0, 0.0)
        self._true_sums.zero_()
        self._true_maxima.zero_()
        # Always recompute RHS norms for numerical consistency (fix tet r5 regression)
        compute_rhs = True
        if compute_rhs:
            self._true_norms.fill_(float("nan"))
        else:
            self._true_residual_norms.fill_(float("nan"))
        self._ratios.fill_(float("nan"))
        wp.launch(
            _true_residual_maxima,
            y.size,
            [self._ap, rhs, self.layout.q_dof_count, self._true_r, self._true_maxima, self._status, compute_rhs],
            device=self.device,
        )
        wp.launch(
            _true_residual_sums,
            y.size,
            [self._true_r, rhs, self.layout.q_dof_count, self._true_maxima, self._true_sums, self._status, compute_rhs],
            device=self.device,
        )
        wp.launch(
            _compute_true_residual_norms,
            6 if compute_rhs else 3,
            [self._true_maxima, self._true_sums, self._true_norms, self._status],
            device=self.device,
        )
        wp.launch(
            _compute_monolithic_true_residual_ratios,
            3,
            [
                self._true_norms,
                config.residual_floor_global,
                config.residual_floor_q,
                config.residual_floor_x,
                self._ratios,
                self._status,
            ],
            device=self.device,
        )
        if compute_rhs:
            wp.launch(
                _cache_monolithic_pcg_denominator,
                1,
                [self._true_norms, config.residual_floor_global, self._rhs_denominator],
                device=self.device,
            )
            self._rhs_norms_cached = True
        wp.launch(
            _pack_monolithic_pcg_residual,
            1,
            [self._status, self._ratios, self._pcg_packet],
            device=self.device,
        )
        packet = self._pcg_packet.numpy()[0]
        self._pcg_boundary_status = MonolithicLinearStatus(int(packet["status"]))
        return tuple(float(value) for value in packet["ratios"])

    def solve_pcg(
        self,
        rhs_hat: wp.array[float],
        y: wp.array[float],
        *,
        generation: MonolithicLinearGeneration,
        warm_start: MonolithicPcgWarmStart,
        config: MonolithicPcgConfig,
    ) -> MonolithicLinearSolveResult:
        """Solve Khat*y=rhs_hat using the current operator and actor preconditioner.

        SAME_GENERATION reuses only the last successful solution array for this
        unchanged K/D/S. Lambda may change, but requires refactorization first.
        Every solve recomputes its initial true residual, including warm starts.
        """
        iterations, checks, replacements = 0, 0, 0
        ratios = (float("nan"),) * 3
        min_p_ap, min_r_z = float("nan"), float("nan")
        applied_warm_start, initial_guess_norm, stagnation_window = None, float("nan"), None
        residual_gap, gap_samples = (float("nan"),) * 3, 0
        diagnostic = self.pcg_mode == "diagnostic"
        diagnostic_started = False

        def result(status):
            nonlocal min_p_ap, min_r_z, initial_guess_norm, residual_gap
            self._pcg_active = False
            if diagnostic_started:
                summary = self._diagnostic_summary.numpy()
                min_p_ap, min_r_z, initial_guess_norm = map(float, summary[:3])
                residual_gap = tuple(map(float, summary[3:]))
            return MonolithicLinearSolveResult(
                status,
                generation,
                iterations,
                checks,
                replacements,
                *ratios,
                min_p_ap,
                min_r_z,
                applied_warm_start,
                initial_guess_norm,
                residual_gap,
                stagnation_window,
            )

        def record_product(index, first=False):
            if diagnostic:
                wp.launch(
                    _record_monolithic_pcg_product,
                    1,
                    [self._products, self._diagnostic_summary, index, first],
                    device=self.device,
                )

        def record_gap():
            nonlocal gap_samples
            if not diagnostic:
                return
            wp.launch(
                _combine_vectors,
                self._r.size,
                [self._r, self._true_r, self._diagnostic_vector, 1.0, -1.0],
                device=self.device,
            )
            self._diagnostic_block_norms(self._diagnostic_vector)
            wp.launch(
                _record_monolithic_pcg_norms,
                1,
                [self._diagnostic_norms, self._diagnostic_summary, False, gap_samples == 0],
                device=self.device,
            )
            gap_samples += 1

        if not self._current(generation) or not self._scaling_valid:
            return result(MonolithicLinearStatus.STALE_GENERATION)
        if not self._factor_valid:
            return result(MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
        if (
            not isinstance(config, MonolithicPcgConfig)
            or not self._vector_valid(rhs_hat)
            or not self._vector_valid(y)
            or (rhs_hat.size > 0 and rhs_hat.ptr < y.ptr + y.size * 4 and y.ptr < rhs_hat.ptr + rhs_hat.size * 4)
            or warm_start not in (MonolithicPcgWarmStart.ZERO, MonolithicPcgWarmStart.SAME_GENERATION)
            or config.maximum_iterations > self.capacities.pcg_max_iterations
            or config.maximum_iterations + 2 > self.capacities.pcg_true_residual_check_count
        ):
            return result(MonolithicLinearStatus.INVALID_ARGUMENT)
        if warm_start == MonolithicPcgWarmStart.SAME_GENERATION and self._warm_start_key != (generation, y.ptr):
            return result(MonolithicLinearStatus.STALE_GENERATION)
        self._warm_start_key = None
        self._status.zero_()
        if warm_start == MonolithicPcgWarmStart.ZERO:
            y.zero_()
        applied_warm_start = MonolithicPcgWarmStart(warm_start)
        self._pcg_active = True
        self._rhs_norms_cached = False
        if diagnostic:
            self._diagnostic_summary.fill_(float("nan"))
            diagnostic_started = True
            self._diagnostic_block_norms(y)
            wp.launch(
                _record_monolithic_pcg_norms,
                1,
                [self._diagnostic_norms, self._diagnostic_summary, True, True],
                device=self.device,
            )
        stagnation_window = config.stagnation_window
        ratios = self._true_residual(rhs_hat, y, config)
        checks += 1
        status = self._pcg_boundary_status
        if status != MonolithicLinearStatus.SUCCESS:
            return result(status)
        if max(ratios) <= config.linear_tolerance:
            self._warm_start_key = (generation, y.ptr)
            return result(MonolithicLinearStatus.SUCCESS)
        wp.copy(self._r, self._true_r)
        self.preconditioner.matvec(self._r, self._z, self._z, 1.0, 0.0)
        self._products_of(self._r, self._z)
        wp.launch(
            _validate_monolithic_preconditioned_residual,
            1,
            [self._products, config.preconditioner_positive_tolerance, self._r_z, self._status],
            device=self.device,
        )
        record_product(1, first=True)
        status = self._read_status()
        if status != MonolithicLinearStatus.SUCCESS:
            return result(status)
        wp.copy(self._p, self._z)
        history = [max(ratios)]
        # A check can occur on every iteration when a recursive residual is tiny.
        # The configured check capacity therefore bounds that worst case.
        for iteration in range(1, config.maximum_iterations + 1):
            self.operator.matvec(self._p, self._ap, self._ap, 1.0, 0.0)
            self._products_of(self._p, self._ap)
            wp.launch(
                _compute_monolithic_pcg_alpha,
                1,
                [
                    self._products,
                    self._r_z,
                    config.curvature_absolute_tolerance,
                    config.curvature_relative_tolerance,
                    self._alpha,
                    self._status,
                ],
                device=self.device,
            )
            record_product(0)
            status = self._read_status()
            if status != MonolithicLinearStatus.SUCCESS:
                break
            wp.launch(
                _update_monolithic_pcg_solution_residual,
                y.size,
                [y, self._r, self._p, self._ap, self._alpha, self._status],
                device=self.device,
            )
            iterations = iteration
            self._products_of(self._r, self._r)
            wp.launch(
                _pack_monolithic_pcg_check,
                1,
                [
                    self._products,
                    self._status,
                    self._rhs_denominator,
                    config.linear_tolerance,
                    iteration % config.true_residual_interval == 0 or iteration == config.maximum_iterations,
                    self._pcg_packet,
                ],
                device=self.device,
            )
            packet = self._pcg_packet.numpy()[0]
            status = MonolithicLinearStatus(int(packet["status"]))
            if status != MonolithicLinearStatus.SUCCESS:
                break
            check = bool(packet["check"])
            if check:
                ratios = self._true_residual(rhs_hat, y, config)
                record_gap()
                checks += 1
                status = self._pcg_boundary_status
                if status != MonolithicLinearStatus.SUCCESS:
                    break
                if max(ratios) <= config.linear_tolerance:
                    self._warm_start_key = (generation, y.ptr)
                    return result(MonolithicLinearStatus.SUCCESS)
                history.append(max(ratios))
                if (
                    len(history) > config.stagnation_window
                    and history[-1]
                    >= (1.0 - config.stagnation_minimum_reduction) * history[-1 - config.stagnation_window]
                ):
                    status = MonolithicLinearStatus.STAGNATION
                    break
                wp.copy(self._r, self._true_r)
                replacements += 1
            if iteration == config.maximum_iterations:
                status = MonolithicLinearStatus.MAX_ITERATIONS
                break
            wp.copy(self._old_r_z, self._r_z)
            self.preconditioner.matvec(self._r, self._z, self._z, 1.0, 0.0)
            self._products_of(self._r, self._z)
            wp.launch(
                _validate_monolithic_preconditioned_residual,
                1,
                [self._products, config.preconditioner_positive_tolerance, self._r_z, self._status],
                device=self.device,
            )
            record_product(1)
            status = self._read_status()
            if status != MonolithicLinearStatus.SUCCESS:
                break
            if check:
                # Residual replacement restarts conjugacy against the actual operator.
                wp.copy(self._p, self._z)
            else:
                wp.launch(
                    _compute_monolithic_pcg_beta,
                    1,
                    [self._r_z, self._old_r_z, self._beta, self._status],
                    device=self.device,
                )
                wp.launch(
                    _pcg_update_direction, y.size, [self._z, self._p, self._beta, self._status], device=self.device
                )
                status = self._read_status()
                if status != MonolithicLinearStatus.SUCCESS:
                    break
        # Failure diagnostics also come from an extra actual-operator matvec.
        ratios = self._true_residual(rhs_hat, y, config)
        if iterations:
            record_gap()
        checks += 1
        if self._pcg_boundary_status == MonolithicLinearStatus.NONFINITE_ITERATION:
            status = MonolithicLinearStatus.NONFINITE_ITERATION
        return result(status)

    def recover_delta(
        self, y: wp.array[float], delta: wp.array[float], *, generation: MonolithicLinearGeneration
    ) -> MonolithicLinearStatus:
        if not self._current(generation) or not self._scaling_valid:
            return MonolithicLinearStatus.STALE_GENERATION
        if not self._vector_valid(y) or not self._vector_valid(delta):
            return MonolithicLinearStatus.INVALID_ARGUMENT
        wp.launch(_scale_vector, y.size, [self.scale, y, delta], device=self.device)
        self._status.zero_()
        self._products_of(delta, delta)
        return self._read_status()

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
        if not self._scaling_valid:
            return MonolithicDenseLinearOracle(dense)
        diagonal, scale = self.diagonal.numpy().astype(np.float64), self.scale.numpy().astype(np.float64)
        regularized = dense + self._lambda * np.diag(diagonal)
        # Match the production S*K*S + lambda*I operator, avoiding a second assembly.
        scaled = scale[:, None] * dense * scale[None, :] + self._lambda * np.eye(matrix.nrow)
        return MonolithicDenseLinearOracle(dense, self.dynamic_diagonal.numpy(), scale, regularized, scaled)
