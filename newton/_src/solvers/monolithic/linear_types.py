# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared layouts, capacities, configuration and results for linear solves."""

import enum
from dataclasses import dataclass

import numpy as np


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
    residual_replacement_threshold: float = 0.1
    """Relative gap threshold for replacing recursive residual with true residual.
    Replace when max(eta_global, eta_q, eta_x) >= threshold, where each eta is
    ||gap||_block / max(||r_true||_block, floor_block). Checking per-block drift
    prevents articulation drift from being masked by larger deformable residuals.
    Default 0.1 (replace when any block drifts >= 10%). Set to 0.0 for always-replace."""
    linear_absolute_tolerance: float = 1.0e-6
    """Absolute scaled residual norm tolerance for every block; provisional, pending calibration.
    Zero retains the legacy relative-with-floor acceptance rule."""
    residual_norm_limits: tuple[float, float, float] | None = None
    """Optional outer-solve caps on global/q/x scaled L2 residuals, applied after rtol/atol.
    None leaves standalone PCG acceptance unchanged. These are per-solve numeric data."""

    def __post_init__(self):
        if self.residual_norm_limits is not None:
            if not isinstance(self.residual_norm_limits, tuple) or len(self.residual_norm_limits) != 3:
                raise ValueError("residual_norm_limits must be a global/q/x tuple")
            for limit in self.residual_norm_limits:
                if (
                    isinstance(limit, (bool, np.bool_))
                    or not isinstance(limit, (int, float, np.integer, np.floating))
                    or not np.isfinite(limit)
                    or not float(np.finfo(np.float32).smallest_subnormal) <= limit <= float(np.finfo(np.float32).max)
                ):
                    raise ValueError("residual_norm_limits must be positive and finite in float32")
        value = self.linear_absolute_tolerance
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 0
            or value > float(np.finfo(np.float32).max)
            or (0 < value < np.finfo(np.float32).smallest_subnormal)
        ):
            raise ValueError("linear_absolute_tolerance must be nonnegative and representable in float32")
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
            self.residual_replacement_threshold,
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

    pcg_policy: str = "dynamic_smart_restart_v2"
    scheduled_slots: int = 0
    converged_at_slot: int = -1
    last_checked_slot: int = -1
    failure_slot: int = -1
    residual_current: bool = False
    output_valid: bool = False
    checkpoint_ratios: tuple[float, float, float] = (float("nan"),) * 3
    prepare_seconds: float = 0.0
    capture_seconds: float = 0.0
    graph_rebuilt: bool = False


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
