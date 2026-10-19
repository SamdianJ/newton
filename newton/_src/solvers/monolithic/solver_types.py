# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver definitions shared by the coupled solver without importing a solver implementation."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .convergence import ConvergencePolicy


@dataclass(frozen=True, slots=True, kw_only=True)
class ClothInput:
    """Experimental monolithic shell input; arrays are copied during construction.

    Triangle IDs select Model triangles. Bending is (alpha, beta) in N m;
    thickness is in m and surface density in kg/m^2. Native membrane
    coefficients come from Model.tri_materials. Contact-free shells share the
    global solve with other supported actors; material contact is unavailable.
    """

    triangle_ids: tuple[int, ...]
    bending: tuple[float, float]
    thickness: float
    surface_density: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CableInput:
    """Experimental finite-EA open DER chain with quasi-static material angles.

    EA is in N; bending is an SPD 2x2 material tensor and GJ is in N m^2.
    Radius is a circular contact envelope in m, density is in kg/m, and
    rest_theta is unwrapped radians. Each rest_director is a unit vector
    perpendicular to its edge. Contact-free cables share the global nonlinear
    solve; material contacts, closed loops and branching are not available.
    """

    particle_ids: tuple[int, ...]
    EA: float
    bending: tuple[tuple[float, float], tuple[float, float]]
    GJ: float
    radius: float
    linear_density: float
    rest_theta: tuple[float, ...]
    rest_director: tuple[tuple[float, float, float], ...]
    fixed_theta: tuple[bool, ...]


class Status(enum.Enum):
    NOT_RUN = "not_run"
    SUCCESS = "success"
    CONTACT_OVERFLOW = "contact_overflow"
    NONFINITE = "nonfinite"
    TET_INVERSION = "tet_inversion"
    TET_PROJECTION_FAILURE = "tet_projection_failure"
    LINEAR_NON_POSITIVE_CURVATURE = "linear_non_positive_curvature"
    LINEAR_PRECONDITIONER_NOT_SPD = "linear_preconditioner_not_spd"
    PRECONDITIONER_FACTORIZATION_FAILED = "preconditioner_factorization_failed"
    LINEAR_MAX_ITERATIONS = "linear_max_iterations"
    LINE_SEARCH_EXHAUSTED = "line_search_exhausted"
    REGULARIZATION_EXHAUSTED = "regularization_exhausted"
    NONLINEAR_STAGNATION = "nonlinear_stagnation"
    NONLINEAR_MAX_ITERATIONS = "nonlinear_max_iterations"


class MaterialModel(str, enum.Enum):
    """Experimental solver-wide P1 material selection."""

    KIM_STABLE_NO_LOG = "kim_stable_no_log"
    SMITH_LOG_STABILIZED = "smith_log_stabilized"


class MassMode(str, enum.Enum):
    """Experimental solver-wide P1 mass discretization."""

    LUMPED = "lumped"
    CONSISTENT = "consistent"


class Precision(enum.Enum):
    """Construction-time precision of nonlinear state and decisions."""

    FP64 = "fp64"
    FP32 = "fp32"


class PCGMode(str, enum.Enum):
    """Experimental PCG observation level, fixed at construction.

    Both modes run the same masked PCG. ``DIAGNOSTIC`` additionally records
    every checkpoint's true-residual group norms; ``PRODUCTION`` permits CUDA
    graph capture.
    """

    DIAGNOSTIC = "diagnostic"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class JointTerms:
    """Opt in to scalar joint physics; omitted terms retain V0.1 rejection.

    Gains, bounds, friction magnitudes and effort/velocity limits come from
    Model and are copied at solver construction. Rebuild after changing
    them. Position/velocity targets come from Control each step. Target
    trajectory generation and speed limiting remain the caller's job.
    """

    implicit_pd: bool = False
    """Use POSITION_VELOCITY targets and saturate the total PD output."""
    limits: bool = False
    """Enable lower/upper penalty and outward-only limit damping."""
    friction: bool = False
    """Enable regularized Coulomb joint friction."""
    limit_width: tuple[float, ...] | None = None
    """Positive activation widths per DOF [rad or m]; required for limits."""
    friction_velocity_scale: tuple[float, ...] | None = None
    """Positive smoothing speeds per DOF [rad/s or m/s]; required for friction."""


@dataclass(frozen=True, slots=True)
class Stats:
    """Immutable diagnostics; unmeasured numerical fields contain NaN.

    Linear residual ratios are measured by actual-operator post-checks;
    they remain NaN if the step does not invoke a linear solve.
    """

    status: Status
    material_model: str | None = None
    mass_mode: str | None = None
    tet_config_identity: str | None = None
    failure_reason: str | None = None
    converged: bool = False
    variational: Mapping | None = None
    """Internal tet-only diagnostics; None on the legacy coupled path."""
    rolled_back: bool = False
    step_generation: int = 0
    accepted_generation: int = 0
    nonlinear_iterations: int = 0
    line_search_iterations: int = 0
    linear_iterations: int = 0
    linear_scheduled_slots: int = 0
    """Sum of scheduled production slots; conditional graphs stop after terminal batches."""
    pcg_policy: str | None = None
    """Most recent linear solve's numerical scheduling and restart policy."""
    linear_converged_at_slot: int = -1
    """Most recent production solve's acceptance slot, or -1."""
    linear_last_checked_slot: int = -1
    """Most recent production solve's last valid true-residual checkpoint."""
    linear_failure_slot: int = -1
    """Most recent production solve's first numerical failure slot, or -1."""
    linear_output_valid: bool = False
    """Whether the most recent production solve published a finite direction."""
    regularization_retries: int = 0
    matrix_assembly_count: int = 0
    matrix_nnz: int = 0
    triplet_count: int = 0
    triplet_capacity: int = 0
    owner_generation: int = -1
    preconditioner_kind: str = "none"
    factor_setup_count: int = 0
    """Total actual factor attempts across all current assemblies in this step."""
    factor_failure_count: int = 0
    """Total failed factor attempts across all current assemblies in this step."""
    q_block_count: int = 0
    """Number of articulation Cholesky blocks in the selected preconditioner."""
    particle_block_count: int = 0
    """Number of dynamic-particle 3x3 blocks in the selected preconditioner."""
    warm_start: str | None = None
    """Most recent solve's initial-guess mode; None when no solve started."""
    linear_scale_generation: int = -1
    """Assembly sequence of the most recent solve, which can precede the current scale."""
    rho: float = math.nan
    """Normalized global scaled residual [dimensionless]; NaN when unmeasured."""
    rho_q: float = math.nan
    """Normalized joint scaled residual [dimensionless]; NaN when unmeasured."""
    rho_x: float = math.nan
    """Normalized particle scaled residual [dimensionless]; NaN when unmeasured."""
    merit_initial: float = math.nan
    """First current merit using its initial scale; not a convergence denominator."""
    merit_q_initial: float = math.nan
    """Joint merit at the first current evaluation, using its initial scale."""
    merit_x_initial: float = math.nan
    """Particle merit at the first current evaluation, using its initial scale."""
    merit_final: float = math.nan
    """Last evaluated current merit; on rollback this can describe a discarded state."""
    merit_q_final: float = math.nan
    """Joint block of the last evaluated current merit."""
    merit_x_final: float = math.nan
    """Particle block of the last evaluated current merit."""
    merit_reference: float = math.nan
    """Initial residual RMS re-evaluated with the last current scale."""
    merit_q_reference: float = math.nan
    """Joint reference RMS using the last current scale."""
    merit_x_reference: float = math.nan
    """Particle reference RMS using the last current scale."""
    convergence_ratio: float = math.nan
    """Current merit / (absolute gate + relative gate * reference); must be <= 1."""
    convergence_ratio_q: float = math.nan
    """Joint convergence ratio; must be <= 1 independently of the global gate."""
    convergence_ratio_x: float = math.nan
    """Particle convergence ratio; must be <= 1 independently of the global gate."""
    scale_generation: int = -1
    """Current assembly sequence for the reported merit/reference/gates."""
    raw_residual_q_norm: float = math.nan
    """Unscaled joint L2 norm [mixed N, N*m]; diagnostic only."""
    raw_residual_x_norm: float = math.nan
    """Unscaled particle L2 norm [N]; diagnostic only."""
    scaled_step: float = math.nan
    """RMS of the last accepted alpha*y in its solve scale; NaN before acceptance."""
    scaled_step_q: float = math.nan
    """Joint RMS of the last accepted alpha*y in its solve scale."""
    scaled_step_x: float = math.nan
    """Particle RMS of the last accepted alpha*y in its solve scale."""
    merit_noise: float = math.nan
    """Actual absolute allowance in the trial merit decrease condition."""
    residual_floor_global: float = math.nan
    """Actual global scaled linear residual denominator floor."""
    residual_floor_q: float = math.nan
    """Actual joint scaled linear residual denominator floor."""
    residual_floor_x: float = math.nan
    """Actual particle scaled linear residual denominator floor."""
    accepted_alpha: float = math.nan
    lambda_value: float = math.nan
    true_residual_recomputations: int = 0
    residual_replacements: int = 0
    active_sample_count: int = 0
    soft_contact_pair_count: int = 0
    published_contact_generation: int = -1
    max_penetration: float = math.nan
    """Maximum contact penetration in the returned state [m]."""
    min_det_f: float = math.nan
    """Minimum evaluated tet deformation determinant [dimensionless]."""
    contact_force_imbalance: float = math.nan
    contact_moment_imbalance: float = math.nan
    generalized_projection_error: float = math.nan
    contact_sign_error: float = math.nan
    contact_history: Mapping[str, float | int] | None = None
    """Final per-step contact/history audit; None in the q-only diagnostic."""
    joint_force_generation: int = -1
    """Step generation of returned-state joint diagnostics; -1 when absent."""
    joint_pd_force: tuple[float, ...] | None = None
    """Actual saturated PD forces [N*m or N] per joint DOF."""
    joint_limit_force: tuple[float, ...] | None = None
    """Physical lower/upper limit forces in the returned state."""
    joint_friction_force: tuple[float, ...] | None = None
    """Physical joint friction forces in the returned state."""
    joint_pd_saturated: tuple[bool, ...] | None = None
    """Whether each returned-state PD output is at its effort bound."""
    joint_limit_damping_power: tuple[float, ...] | None = None
    """Limit damping mechanical power [W]; nonpositive."""
    joint_friction_power: tuple[float, ...] | None = None
    """Regularized joint friction mechanical power [W]; nonpositive."""
    timings: Mapping[str, float | None] = field(default_factory=lambda: MappingProxyType({}))
    """Measured durations [ms], or None for unmeasured entries."""

    def __post_init__(self):
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))


@dataclass(frozen=True, slots=True)
class _SolverMonolithicInternalConfig:
    """Portable CPU/CUDA settings from the versioned P0 component and trajectory calibration.

    The historical component, float32 reduction, coordinate-ULP and trajectory
    calibration driver is preserved in the P4 legacy archive.
    """

    epsilon_d: float = 1.0e-6
    residual_floor_global: float = 1.7210128930635914e-4
    residual_floor_q: float = 2.1090331703277343e-5
    residual_floor_x: float = 1.7099733184977595e-4
    convergence: ConvergencePolicy = field(default_factory=ConvergencePolicy)
    """Per-row physical convergence tolerances."""
    det_f_guard: float = 0.2
    regularization_values: tuple[float, ...] = (0.0,)


class _StepFailure(RuntimeError):
    def __init__(self, status: Status, reason: str):
        self.status = status
        super().__init__(reason)
