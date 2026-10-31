# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared layouts, capacities, configuration and results for linear solves."""

import enum
import itertools
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
    residual_replacement_threshold: float = 0.1
    """Relative gap threshold for replacing recursive residual with true residual.
    Replace when max(eta_global, eta_q, eta_x) >= threshold, where each eta is
    ||gap||_block / max(||r_true||_block, floor_block). Checking per-block drift
    prevents articulation drift from being masked by larger deformable residuals.
    Default 0.1 (replace when any block drifts >= 10%). Set to 0.0 for always-replace."""
    linear_absolute_tolerance: float = 1.0e-6
    """Absolute scaled residual norm tolerance for every block.
    The nonlinear solver sets it to the largest group limit, so each group stops at its own limit."""
    residual_norm_limits: tuple[float, float, float] | None = None
    """Optional outer-solve caps on global/q/x scaled L2 residuals, applied after rtol/atol.
    None leaves standalone PCG acceptance unchanged. These are per-solve numeric data."""
    group_norm_limits: tuple[float, ...] | None = None
    """Private outer-solve caps ordered by the complete actor convergence groups."""
    group_residual_floors: tuple[float, ...] | None = None
    """Explicit floors in each group's units, required with material layouts."""

    def __post_init__(self):
        for values in (self.group_norm_limits, self.group_residual_floors):
            if values is None:
                continue
            if not isinstance(values, tuple) or not values:
                raise ValueError("Group residual limits/floors must be a nonempty tuple")
            for limit in values:
                if (
                    isinstance(limit, (bool, np.bool_))
                    or not isinstance(limit, (int, float, np.integer, np.floating))
                    or not np.isfinite(limit)
                    or not float(np.finfo(np.float32).smallest_subnormal) <= limit <= float(np.finfo(np.float32).max)
                ):
                    raise ValueError("Group residual limits/floors must be positive and finite in float32")
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
            for value in (self.maximum_iterations, self.true_residual_interval)
        ):
            raise ValueError("PCG iteration counts must be positive integers")
        positive = (self.linear_tolerance, self.residual_floor_global, self.residual_floor_q, self.residual_floor_x)
        if any(
            not np.isfinite(value)
            or value < np.finfo(np.float32).smallest_subnormal
            or value > np.finfo(np.float32).max
            for value in positive
        ) or not (
            np.isfinite(self.residual_replacement_threshold)
            and 0 <= self.residual_replacement_threshold <= np.finfo(np.float32).max
        ):
            raise ValueError("PCG tolerances must be finite with positive residual floors")


@dataclass(frozen=True, slots=True)
class MonolithicLinearLayout:
    q_dof_count: int
    dynamic_particle_count: int
    q_scalar_offset: int = 0
    tet_scalar_offset: int | None = None
    identity: str = ""
    actor_ranges: tuple[tuple[str, int, int], ...] = ()
    rigid_ranges: tuple[tuple[int, int], ...] = ()
    material_ranges: tuple[tuple[str, int, int, int, int], ...] = ()
    # World segment boundaries: world w owns scalar rows [world_segments[w], world_segments[w + 1]).
    world_segments: tuple[int, ...] = ()
    articulation_ranges: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        for value in (self.q_dof_count, self.dynamic_particle_count, self.q_scalar_offset, self.x_scalar_offset):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not 0 <= value < 2**31
            ):
                raise ValueError("Layout counts and offsets must be nonnegative int32 integers")
        if self.scalar_dof_count >= 2**31:
            raise ValueError("Actor layout capacity exceeds int32")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**31
            for pair in self.rigid_ranges
            for value in pair
        ) or len({source for source, _ in self.rigid_ranges}) != len(self.rigid_ranges):
            raise ValueError("Rigid ranges require unique nonnegative int32 source identities")
        if not self.actor_ranges:
            object.__setattr__(
                self,
                "actor_ranges",
                (
                    ("articulation", self.q_scalar_offset, self.q_dof_count),
                    ("tet", self.x_scalar_offset, 3 * self.dynamic_particle_count),
                ),
            )
        if not self.articulation_ranges:
            object.__setattr__(
                self,
                "articulation_ranges",
                tuple(
                    (i, start, count)
                    for i, (kind, start, count) in enumerate(self.actor_ranges)
                    if kind == "articulation" and count
                ),
            )
        if (
            any(
                isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v < 2**31
                for block in self.articulation_ranges
                for v in block
            )
            or sum(count for _, _, count in self.articulation_ranges) != self.q_dof_count
            or len({source for source, _, _ in self.articulation_ranges}) != len(self.articulation_ranges)
            or sorted((start, count) for _, start, count in self.articulation_ranges)
            != sorted((start, count) for kind, start, count in self.actor_ranges if kind == "articulation" and count)
        ):
            raise ValueError("Articulation ranges require unique identities and exact actor blocks")
        cursor = 0
        for _, start, count in sorted(self.actor_ranges, key=lambda item: item[1]):
            if count < 0 or start != cursor:
                raise ValueError("Actor ranges must partition the scalar layout")
            cursor += count
        if cursor != self.scalar_dof_count:
            raise ValueError("Actor ranges must partition the scalar layout")
        if not self.world_segments:
            object.__setattr__(self, "world_segments", (0, self.scalar_dof_count))
        object.__setattr__(self, "world_segments", tuple(int(value) for value in self.world_segments))
        boundaries = {0, *(start + count for _, start, count in self.actor_ranges)}
        segments = self.world_segments
        if (
            len(segments) < 2
            or segments[0] != 0
            or segments[-1] != self.scalar_dof_count
            or (
                any(end <= start for start, end in itertools.pairwise(segments))
                if self.scalar_dof_count
                else segments != (0, 0)
            )
            or not set(segments) <= boundaries
        ):
            raise ValueError("World segments must partition the scalar layout at actor boundaries")
        if sorted((start, count) for kind, start, count in self.actor_ranges if kind == "rigid") != sorted(
            (start, 6) for _, start in self.rigid_ranges
        ):
            raise ValueError("Rigid ranges must match six-dimensional actor blocks")
        identities = set()
        for kind, source, start, xyz, theta in self.material_ranges:
            if (
                kind not in ("cloth", "cable")
                or (kind, source) in identities
                or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v < 2**31
                    for v in (source, start, xyz, theta)
                )
                or xyz % 3
                or (kind == "cloth" and theta)
            ):
                raise ValueError("Material ranges require unique owners and exact xyz/theta counts")
            identities.add((kind, source))
        if sorted(
            (kind, start, count) for kind, start, count in self.actor_ranges if kind in ("cloth", "cable")
        ) != sorted((kind, start, xyz + theta) for kind, _, start, xyz, theta in self.material_ranges):
            raise ValueError("Material ranges must match actor blocks")

    @classmethod
    def from_actors(cls, table):
        """Derive the supported linear view without imposing an actor order."""
        arms = [actor for actor in table.actors if actor.kind == "articulation"]
        tets = [actor for actor in table.actors if actor.kind == "tet"]
        if len(tets) > 1:
            raise ValueError("Coupled entry permits at most one tet")
        arm, tet = next(iter(arms), None), next(iter(tets), None)
        return cls(
            sum(a.dof_count for a in arms),
            0 if tet is None else len(tet.dynamic_particle_indices),
            0 if arm is None else arm.dof_start,
            0 if tet is None else tet.dof_start,
            table.identity,
            tuple((actor.kind, actor.dof_start, actor.dof_count) for actor in table.actors),
            tuple((actor.source_index, actor.dof_start) for actor in table.actors if actor.kind == "rigid"),
            tuple(
                (a.kind, a.source_index, a.dof_start, 3 * len(a.dynamic_particle_indices), len(a.free_theta_indices))
                for a in table.actors
                if a.kind in ("cloth", "cable")
            ),
            articulation_ranges=tuple((a.source_index, a.dof_start, a.dof_count) for a in arms),
        )

    @property
    def q_matrix_shape(self):
        """Dense single-tree view, or packed independent blocks for multiple trees."""
        return (
            (self.q_dof_count, self.q_dof_count)
            if len(self.articulation_ranges) <= 1
            else (sum(count * count for _, _, count in self.articulation_ranges), 1)
        )

    @property
    def q_blocks(self):
        """Global row start, DoF count and packed matrix offset for each tree."""
        offset, blocks = 0, []
        for _, start, count in self.articulation_ranges:
            blocks.append((start, count, offset))
            offset += count * count
        return tuple(blocks)

    def articulation_indices(self):
        """Return all mechanical rows in the packed tree order."""
        return np.array(
            [row for _, start, count in self.articulation_ranges for row in range(start, start + count)], dtype=np.int32
        )

    @property
    def x_scalar_offset(self) -> int:
        return self.q_dof_count if self.tet_scalar_offset is None else self.tet_scalar_offset

    @property
    def world_count(self) -> int:
        return len(self.world_segments) - 1

    @property
    def groups(self) -> tuple[tuple[str, int, int], ...]:
        """Return the global group followed by each nonempty actor range."""
        groups = [("global", 0, self.scalar_dof_count)]
        rigid_ids = {start: source for source, start in self.rigid_ranges}
        material_ids = {(kind, start): (source, xyz, theta) for kind, source, start, xyz, theta in self.material_ranges}
        arm_ids = {start: source for source, start, _ in self.articulation_ranges}
        for kind, start, count in self.actor_ranges:
            if kind == "rigid":
                groups.extend(
                    ((f"rigid_{rigid_ids[start]}_t", start, 3), (f"rigid_{rigid_ids[start]}_r", start + 3, 3))
                )
            elif kind in ("cloth", "cable") and count:
                source, xyz, theta = material_ids[kind, start]
                if xyz:
                    groups.append((f"{kind}_{source}_x", start, xyz))
                if theta:
                    groups.append((f"{kind}_{source}_theta", start + xyz, theta))
            elif kind == "articulation" and count and len(arm_ids) > 1:
                groups.append((f"articulation_{arm_ids[start]}", start, count))
            elif count:
                groups.append((kind, start, count))
        return tuple(groups)

    def indices(self, kind, source_index=None):
        """Return local-to-global scalar rows for one supported actor type."""
        if kind in ("cloth", "cable"):
            matches = [
                (start, xyz + theta)
                for k, source, start, xyz, theta in self.material_ranges
                if k == kind and (source_index is None or source == source_index)
            ]
            if len(matches) != 1:
                raise ValueError("Material indices require one unambiguous actor")
            start, count = matches[0]
        elif kind == "articulation":
            matches = [
                (start, count)
                for source, start, count in self.articulation_ranges
                if source_index is None or source == source_index
            ]
            if not matches and not self.q_dof_count and source_index is None:
                return np.empty(0, dtype=np.int32)
            if len(matches) != 1:
                raise ValueError("Articulation indices require one unambiguous actor")
            start, count = matches[0]
        elif kind == "tet":
            start, count = self.x_scalar_offset, 3 * self.dynamic_particle_count
        else:
            raise ValueError(f"Unsupported actor kind: {kind}")
        return np.arange(start, start + count, dtype=np.int32)

    @property
    def scalar_dof_count(self) -> int:
        return (
            self.q_dof_count
            + 3 * self.dynamic_particle_count
            + 6 * len(self.rigid_ranges)
            + sum(xyz + theta for _, _, _, xyz, theta in self.material_ranges)
        )


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
    warm_start: MonolithicPcgWarmStart | None = None
    checkpoint_history: tuple[tuple[int, tuple[float, ...]], ...] = ()
    """Diagnostic observation only: (slot, true-residual norm per convergence group) at each checkpoint."""

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
