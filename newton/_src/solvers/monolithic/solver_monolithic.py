# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import warp as wp

from ...sim import Contacts, Control, JointType, Model, State, eval_fk
from ..solver import SolverBase
from .articulation import (
    MonolithicArticulationWorkspace,
    _validate_joint_coordinate_layout,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
    project_articulation_body_wrenches,
    recover_articulation_candidate_rates,
    scatter_articulation_actor_tangent,
)
from .collision import MonolithicCollisionPipeline
from .contact import (
    MonolithicContactWorkspace,
    _MonolithicReturnedStateRole,
    assemble_current_contacts,
    evaluate_final_contacts,
    evaluate_trial_contacts,
)
from .linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .tet import (
    TetEvaluationStatus,
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
    evaluate_tet_residual,
    validate_tet_scope,
)


@dataclass(frozen=True, slots=True)
class _SolverMonolithicInternalConfig:
    """Initial tiny-fixture settings; full asset/trajectory calibration is pending.

    See ``scripts/monolithic_reference/calibrate_components.py`` for the
    float32 reduction and coordinate-ULP measurements behind these draft gates.
    """

    epsilon_d: float = 1.0e-6
    residual_floor_global: float = 1.11e-6
    residual_floor_q: float = 1.1e-6
    residual_floor_x: float = 5.0e-8
    merit_noise: float = 0.0
    merit_absolute_global: float = 1.1e-6
    merit_absolute_q: float = 2.8e-6
    merit_absolute_x: float = 7.1e-8
    merit_relative_global: float = 1.0e-4
    merit_relative_q: float = 1.0e-4
    merit_relative_x: float = 1.0e-4
    step_tolerance_global: float = 2.6e-7
    step_tolerance_q: float = 6.9e-7
    step_tolerance_x: float = 1.8e-8
    det_f_guard: float = 0.2
    regularization_values: tuple[float, ...] = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0)


@dataclass(frozen=True, slots=True)
class _TrialResult:
    status: int
    merit: float
    merit_q: float
    merit_x: float
    min_det_f: float
    maximum_penetration: float


class _StepFailure(RuntimeError):
    def __init__(self, status: SolverMonolithic.Status, reason: str):
        self.status = status
        super().__init__(reason)


@wp.kernel
def _scale_vector(values: wp.array[float], scale: wp.array[float], sign: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = sign * scale[i] * values[i]


@wp.kernel
def _copy_particle_residual(values: wp.array[wp.vec3], offset: int, out: wp.array[float]):
    i = wp.tid()
    for axis in range(3):
        out[offset + 3 * i + axis] = values[i][axis]


@wp.kernel
def _raw_residual_sums(values: wp.array[float], nq: int, out: wp.array[wp.float64]):
    i = wp.tid()
    value = wp.float64(values[i])
    block = int(i >= nq)
    wp.atomic_add(out, block, value * value)


@wp.kernel
def _build_dynamic_diagonal(
    mass: wp.array3d[float],
    particle_mass: wp.array[float],
    particle_ids: wp.array[int],
    nq: int,
    inv_dt_sq: float,
    out: wp.array[float],
):
    i = wp.tid()
    if i < nq:
        out[i] = mass[0, i, i] * inv_dt_sq
    else:
        out[i] = particle_mass[particle_ids[(i - nq) // 3]] * inv_dt_sq


@dataclass(frozen=True, slots=True)
class _MonolithicLayout:
    q_dof_count: int
    dynamic_particle_count: int
    scalar_dof_count: int
    q_dof_to_joint_q: wp.array[int]
    q_dof_to_joint_qd: wp.array[int]
    dynamic_particle_ids: wp.array[int]
    particle_to_dynamic: wp.array[int]
    body_to_articulation: wp.array[int]
    body_to_link_index: wp.array[int]


def _build_layout(model: Model) -> _MonolithicLayout:
    if model.articulation_count != 1 or model.tet_count == 0:
        raise ValueError("Monolithic requires one articulation and one connected tet body")
    starts = model.articulation_start.numpy()
    ends = model.articulation_end.numpy()
    if starts.tolist() != [0, model.joint_count] or ends.tolist() != [model.joint_count]:
        raise ValueError("Monolithic does not support unowned joints or loop closures")
    types = model.joint_type.numpy()
    if not np.isin(types, [JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC]).all():
        raise ValueError("Monolithic supports only FIXED, REVOLUTE and PRISMATIC joints")
    children, parents = model.joint_child.numpy(), model.joint_parent.numpy()
    seen = set()
    roots = 0
    body_to_link = np.full(model.body_count, -1, dtype=np.int32)
    for link, (parent, child) in enumerate(zip(parents, children, strict=True)):
        if child < 0 or child >= model.body_count or child in seen or (parent != -1 and parent not in seen):
            raise ValueError("Monolithic requires a world-anchored articulation tree")
        roots += int(parent == -1)
        seen.add(int(child))
        body_to_link[child] = link
    if roots != 1 or len(seen) != model.body_count:
        raise ValueError("Monolithic requires one world-anchored tree owning all bodies")
    tets = model.tet_indices.numpy()
    if np.any(tets < 0) or np.any(tets >= model.particle_count) or any(len(set(t)) != 4 for t in tets):
        raise ValueError("Invalid tet topology")
    target = np.unique(tets)
    connected = set(tets[0])
    remaining = list(tets[1:])
    while remaining:
        pending = []
        for tet in remaining:
            if connected.intersection(tet):
                connected.update(tet)
            else:
                pending.append(tet)
        if len(pending) == len(remaining):
            raise ValueError("Monolithic requires one connected tet body")
        remaining = pending
    inv_mass = model.particle_inv_mass.numpy()
    if not np.isfinite(inv_mass[target]).all() or np.any(inv_mass[target] < 0.0):
        raise ValueError("Target inverse masses must be finite and nonnegative")
    dynamic = target[inv_mass[target] > 0.0]
    particle_to_dynamic = np.full(model.particle_count, -1, dtype=np.int32)
    particle_to_dynamic[dynamic] = np.arange(len(dynamic), dtype=np.int32)
    moving = types != JointType.FIXED
    q_map = _validate_joint_coordinate_layout(model)
    qd_map = model.joint_qd_start.numpy()[:-1][moving]
    nq = len(q_map)

    def array(values):
        return wp.array(values, dtype=wp.int32, device=model.device)

    layout = _MonolithicLayout(
        nq,
        len(dynamic),
        nq + 3 * len(dynamic),
        array(q_map),
        array(qd_map),
        array(dynamic),
        array(particle_to_dynamic),
        array(np.where(body_to_link >= 0, 0, -1)),
        array(body_to_link),
    )
    _validate_scope(model, layout)
    return layout


def _validate_scope(model: Model, layout: _MonolithicLayout) -> None:
    """Validate participating world labels; collision asset gates belong to PR-4A."""
    bodies = np.flatnonzero(layout.body_to_articulation.numpy() >= 0)
    shapes = np.isin(model.shape_body.numpy(), bodies)
    target = np.unique(model.tet_indices.numpy())
    labels = np.concatenate(
        (model.body_world.numpy()[bodies], model.shape_world.numpy()[shapes], model.particle_world.numpy()[target])
    )
    if len(np.unique(labels)) != 1 or np.any(labels < -1) or np.any(labels >= model.world_count):
        raise ValueError("Monolithic requires exactly one participating world label")


def _validate_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt <= 0.0 or dt > np.finfo(np.float32).max or np.float32(dt) <= 0.0:
        raise ValueError("dt must be finite and positive in float32")


def _validate_state(model: Model, state: State) -> None:
    """Reject optional/custom state arrays until their transaction semantics exist."""
    attributes = (
        ("joint_q", model.joint_coord_count, wp.float32),
        ("joint_qd", model.joint_dof_count, wp.float32),
        ("body_q", model.body_count, wp.transform),
        ("body_qd", model.body_count, wp.spatial_vector),
        ("body_f", model.body_count, wp.spatial_vector),
        ("particle_q", model.particle_count, wp.vec3),
        ("particle_qd", model.particle_count, wp.vec3),
        ("particle_f", model.particle_count, wp.vec3),
    )
    for name, count, dtype in attributes:
        value = getattr(state, name)
        if value is None or value.shape != (count,) or value.dtype != dtype or value.device != model.device:
            raise ValueError(f"Invalid state array shape, dtype or device: {name}")
    supported = {name for name, _, _ in attributes}
    for name, value in vars(state).items():
        if isinstance(value, wp.array) and name not in supported:
            raise ValueError(f"Unsupported PR-0 state array: {name}")
        if isinstance(value, Model.AttributeNamespace) and any(
            isinstance(attribute, wp.array) for attribute in vars(value).values()
        ):
            raise ValueError(f"Unsupported PR-0 state array namespace: {name}")


def _validate_dirichlet(model: Model, layout: _MonolithicLayout, state: State) -> None:
    target = np.unique(model.tet_indices.numpy())
    fixed = target[layout.particle_to_dynamic.numpy()[target] < 0]
    if np.any(state.particle_qd.numpy()[fixed] != 0.0):
        raise ValueError("Static Dirichlet particles require exactly zero input velocity")


class _FrozenStepInputs:
    def __init__(self, model: Model):
        self.joint_f = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        self.body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.particle_f = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)

    def snapshot(self, state_in: State, control: Control) -> None:
        for target, source in (
            (self.joint_f, control.joint_f),
            (self.body_f, state_in.body_f),
            (self.particle_f, state_in.particle_f),
        ):
            if (
                source is None
                or target.shape != source.shape
                or target.dtype != source.dtype
                or target.device != source.device
            ):
                raise ValueError("Invalid frozen input shape, dtype or device")
        wp.copy(self.joint_f, control.joint_f)
        wp.copy(self.body_f, state_in.body_f)
        wp.copy(self.particle_f, state_in.particle_f)


@wp.kernel
def _pack_monolithic_unknowns(
    joint_q: wp.array[float],
    particle_q: wp.array[wp.vec3],
    q_dof_to_joint_q: wp.array[int],
    dynamic_particle_ids: wp.array[int],
    q_dof_count: int,
    out_z: wp.array[float],
):
    i = wp.tid()
    if i < q_dof_count:
        out_z[i] = joint_q[q_dof_to_joint_q[i]]
    else:
        p = (i - q_dof_count) // 3
        a = (i - q_dof_count) % 3
        out_z[i] = particle_q[dynamic_particle_ids[p]][a]


@wp.kernel
def _unpack_monolithic_unknowns(
    z: wp.array[float],
    q_dof_to_joint_q: wp.array[int],
    dynamic_particle_ids: wp.array[int],
    q_dof_count: int,
    out_joint_q: wp.array[float],
    out_particle_q: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < q_dof_count:
        out_joint_q[q_dof_to_joint_q[i]] = z[i]
    else:
        p = i - q_dof_count
        offset = q_dof_count + 3 * p
        out_particle_q[dynamic_particle_ids[p]] = wp.vec3(z[offset], z[offset + 1], z[offset + 2])


@wp.kernel
def _form_monolithic_trial(
    accepted_z: wp.array[float], delta_z: wp.array[float], alpha: float, out_trial_z: wp.array[float]
):
    i = wp.tid()
    out_trial_z[i] = accepted_z[i] + alpha * delta_z[i]


@wp.kernel
def _predict_coordinates(
    joint_qd: wp.array[float],
    particle_qd: wp.array[wp.vec3],
    q_map: wp.array[int],
    particle_ids: wp.array[int],
    nq: int,
    dt: float,
    z: wp.array[float],
):
    i = wp.tid()
    if i < nq:
        z[i] += dt * joint_qd[q_map[i]]
    else:
        z[i] += dt * particle_qd[particle_ids[(i - nq) // 3]][(i - nq) % 3]


@wp.kernel
def _recover_particle_kinematics(
    x: wp.array[wp.vec3],
    x0: wp.array[wp.vec3],
    v0: wp.array[wp.vec3],
    particle_ids: wp.array[int],
    dt: float,
    velocity: wp.array[wp.vec3],
    acceleration: wp.array[wp.vec3],
):
    i = wp.tid()
    p = particle_ids[i]
    v = (x[p] - x0[p]) / dt
    velocity[p] = v
    acceleration[p] = (v - v0[p]) / dt


class _Candidate:
    def __init__(self, model: Model, layout: _MonolithicLayout):
        self._model = model
        self._layout = layout
        self._origin = model.state()
        self._dt = None
        self._trial_source = None
        self._trial_source_generation = -1
        self.state = model.state()
        _validate_state(model, self.state)
        self.z = wp.zeros(layout.scalar_dof_count, device=model.device)
        self.qdd = wp.zeros(layout.q_dof_count, device=model.device)
        self.particle_acceleration = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.residual = wp.zeros(layout.scalar_dof_count, device=model.device)
        self.generation = 0

    def load_predictor(self, state_in: State, dt: float) -> None:
        _validate_dt(dt)
        _validate_state(self._model, state_in)
        _validate_dirichlet(self._model, self._layout, state_in)
        self._origin.assign(state_in)
        self.state.assign(state_in)
        self._dt = dt
        layout = self._layout
        wp.launch(
            _pack_monolithic_unknowns,
            layout.scalar_dof_count,
            inputs=[
                state_in.joint_q,
                state_in.particle_q,
                layout.q_dof_to_joint_q,
                layout.dynamic_particle_ids,
                layout.q_dof_count,
                self.z,
            ],
            device=self._model.device,
        )
        wp.launch(
            _predict_coordinates,
            layout.scalar_dof_count,
            inputs=[
                state_in.joint_qd,
                state_in.particle_qd,
                layout.q_dof_to_joint_qd,
                layout.dynamic_particle_ids,
                layout.q_dof_count,
                dt,
                self.z,
            ],
            device=self._model.device,
        )
        self._recover(dt)
        self.generation = 0
        self._trial_source = None

    def form_trial(self, accepted: _Candidate, delta: wp.array[float], alpha: float, dt: float) -> None:
        _validate_dt(dt)
        if accepted is self or accepted._model is not self._model or accepted._layout is not self._layout:
            raise ValueError("Trial requires a distinct candidate with the same model and layout")
        if accepted._dt != dt:
            raise ValueError("Trial dt must equal the frozen predictor dt")
        if delta.shape != self.z.shape or delta.dtype != self.z.dtype or delta.device != self.z.device:
            raise ValueError("Invalid trial delta shape, dtype or device")
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("Trial alpha must be finite and in (0, 1]")
        self._origin.assign(accepted._origin)
        self.state.assign(accepted.state)
        self._dt = dt
        wp.launch(
            _form_monolithic_trial,
            self._layout.scalar_dof_count,
            inputs=[accepted.z, delta, alpha, self.z],
            device=self._model.device,
        )
        self._recover(dt)
        self.generation += 1
        self._trial_source = accepted
        self._trial_source_generation = accepted.generation

    def _recover(self, dt: float) -> None:
        layout, state, origin = self._layout, self.state, self._origin
        wp.launch(
            _unpack_monolithic_unknowns,
            layout.q_dof_count + layout.dynamic_particle_count,
            inputs=[
                self.z,
                layout.q_dof_to_joint_q,
                layout.dynamic_particle_ids,
                layout.q_dof_count,
                state.joint_q,
                state.particle_q,
            ],
            device=self._model.device,
        )
        wp.launch(
            recover_articulation_candidate_rates,
            layout.q_dof_count,
            inputs=[
                layout.q_dof_to_joint_q,
                0,
                layout.q_dof_count,
                1.0 / dt,
                state.joint_q,
                origin.joint_q,
                origin.joint_qd,
                state.joint_qd,
                self.qdd,
            ],
            device=self._model.device,
        )
        self.particle_acceleration.zero_()
        wp.launch(
            _recover_particle_kinematics,
            layout.dynamic_particle_count,
            inputs=[
                state.particle_q,
                origin.particle_q,
                origin.particle_qd,
                layout.dynamic_particle_ids,
                dt,
                state.particle_qd,
                self.particle_acceleration,
            ],
            device=self._model.device,
        )
        self.residual.zero_()
        eval_fk(self._model, state.joint_q, state.joint_qd, state)

    def commit(self, state_out: State) -> None:
        _validate_state(self._model, self.state)
        _validate_state(self._model, state_out)
        state_out.assign(self.state)


class SolverMonolithic(SolverBase):
    """Solve articulated rigid/tet motion with a common implicit position update.

    This experimental implementation supports normal-only P1Q3 contact and
    static fixed particles. Numerical defaults currently use tiny-fixture
    development calibration; broader asset and trajectory validation is pending.

    .. experimental::
        This class may change without the normal deprecation period.
    """

    supports_collision_pipeline = True

    class Status(enum.Enum):
        NOT_RUN = "not_run"
        SUCCESS = "success"
        CONTACT_OVERFLOW = "contact_overflow"
        NONFINITE = "nonfinite"
        TET_INVERSION = "tet_inversion"
        LINEAR_NON_POSITIVE_CURVATURE = "linear_non_positive_curvature"
        LINEAR_PRECONDITIONER_NOT_SPD = "linear_preconditioner_not_spd"
        PRECONDITIONER_FACTORIZATION_FAILED = "preconditioner_factorization_failed"
        LINEAR_STAGNATION = "linear_stagnation"
        LINEAR_MAX_ITERATIONS = "linear_max_iterations"
        LINE_SEARCH_EXHAUSTED = "line_search_exhausted"
        REGULARIZATION_EXHAUSTED = "regularization_exhausted"
        NONLINEAR_STAGNATION = "nonlinear_stagnation"
        NONLINEAR_MAX_ITERATIONS = "nonlinear_max_iterations"

    @dataclass(frozen=True, slots=True)
    class Stats:
        """Immutable diagnostics; unmeasured numerical fields contain NaN.

        Linear residual ratios are measured by actual-operator post-checks;
        they remain NaN if the step does not invoke a linear solve.
        """

        status: SolverMonolithic.Status
        failure_reason: str | None = None
        converged: bool = False
        rolled_back: bool = False
        step_generation: int = 0
        accepted_generation: int = 0
        nonlinear_iterations: int = 0
        line_search_iterations: int = 0
        linear_iterations: int = 0
        regularization_retries: int = 0
        matrix_assembly_count: int = 0
        matrix_nnz: int = 0
        triplet_count: int = 0
        triplet_capacity: int = 0
        owner_generation: int = -1
        preconditioner_kind: str = "none"
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
        min_p_ap: float = math.nan
        min_r_z: float = math.nan
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
        timings: Mapping[str, float | None] = field(default_factory=lambda: MappingProxyType({}))
        """Measured durations [ms], or None for unmeasured entries."""

        def __post_init__(self):
            object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))

    def __init__(
        self,
        model: Model,
        *,
        collision_pipeline: MonolithicCollisionPipeline,
        contact_stiffness: float,
        newton_max_iterations: int = 10,
        line_search_max_iterations: int = 8,
        linear_max_iterations: int = 200,
        linear_tolerance: float = 1.0e-4,
    ) -> None:
        """Validate the model and allocate solver-owned candidate/contact buffers.

        Args:
            model: Model containing one articulation and one connected tet body.
            collision_pipeline: Fixed P1Q3 pipeline constructed for this model.
            contact_stiffness: Normal penalty stiffness [N/m^3].
            newton_max_iterations: Positive nonlinear iteration limit.
            line_search_max_iterations: Positive backtracking iteration limit.
            linear_max_iterations: Positive linear iteration limit.
            linear_tolerance: Positive relative linear residual tolerance.
        """
        if not isinstance(collision_pipeline, MonolithicCollisionPipeline):
            raise ValueError("Monolithic requires a MonolithicCollisionPipeline")
        if collision_pipeline.model is not model:
            raise ValueError("collision_pipeline and solver must use the same model")
        for name, value in (("contact_stiffness", contact_stiffness), ("linear_tolerance", linear_tolerance)):
            if (
                not math.isfinite(value)
                or value <= 0.0
                or value > float(np.finfo(np.float32).max)
                or np.float32(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive in float32")
        for name, value in (
            ("newton_max_iterations", newton_max_iterations),
            ("line_search_max_iterations", line_search_max_iterations),
            ("linear_max_iterations", linear_max_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._layout = _build_layout(model)
        validate_tet_scope(
            model,
            dynamic_particle_ids=self._layout.dynamic_particle_ids.numpy(),
            particle_to_dynamic=self._layout.particle_to_dynamic.numpy(),
        )
        self._articulation = MonolithicArticulationWorkspace(model)
        super().__init__(
            model,
            collision_pipeline=collision_pipeline,
            collision_frequency_type=dict.fromkeys(self.CollisionSlot, self.CollisionFrequencyType.NONE),
        )
        self._trial_contacts = collision_pipeline.contacts()
        self._transaction = _StepTransaction(model, self._layout)
        self.contact_stiffness = contact_stiffness
        self.newton_max_iterations = newton_max_iterations
        self.line_search_max_iterations = line_search_max_iterations
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance = linear_tolerance
        self._config = _SolverMonolithicInternalConfig()
        layout = self._layout
        nq, nx, size = layout.q_dof_count, layout.dynamic_particle_count, layout.scalar_dof_count
        pattern = build_tet_triplet_pattern(
            model.tet_indices.numpy(),
            layout.particle_to_dynamic.numpy(),
            x_dof_start=nq,
        )
        contact_capacity = collision_pipeline.soft_contact_max
        self._linear = MonolithicLinearWorkspace(
            MonolithicLinearLayout(nq, nx),
            MonolithicLinearCapacities(
                nq * nq + len(pattern.global_rows) + contact_capacity * (nq + 9) ** 2,
                len(pattern.ax_rows),
                contact_capacity,
                linear_max_iterations,
                linear_max_iterations + 2,
            ),
            layout.particle_to_dynamic,
            device=model.device,
        )
        self._linear._set_fixed_triplet_pattern(
            global_rows=np.concatenate((np.repeat(np.arange(nq), nq), pattern.global_rows)).astype(np.int32),
            global_columns=np.concatenate((np.tile(np.arange(nq), nq), pattern.global_columns)).astype(np.int32),
            internal_rows=pattern.ax_rows,
            internal_columns=pattern.ax_columns,
        )
        self._tet_workspace = create_tet_assembly_workspace(pattern, tet_count=model.tet_count, device=model.device)
        self._tet_triplet_count = len(pattern.global_rows)
        self._contact = MonolithicContactWorkspace(
            model,
            collision_pipeline,
            self._linear,
            contact_stiffness=contact_stiffness,
        )
        self._final_evaluation_state = model.state()
        self._particle_residual = wp.zeros(nx, dtype=wp.vec3, device=model.device)
        self._dynamic_diagonal = wp.zeros(size, dtype=float, device=model.device)
        self._residual_ref = wp.zeros(size, dtype=float, device=model.device)
        self._metric_vector = wp.zeros(size, dtype=float, device=model.device)
        self._metric_slices = (self._metric_vector, self._metric_vector[:nq], self._metric_vector[nq:])
        self._norm_values = wp.zeros(3, dtype=float, device=model.device)
        self._raw_norm_sums = wp.zeros(2, dtype=wp.float64, device=model.device)
        self._norm_outputs = tuple(self._norm_values[i : i + 1] for i in range(3))
        self._rhs_hat = wp.zeros(size, dtype=float, device=model.device)
        self._y = wp.zeros(size, dtype=float, device=model.device)
        self._delta = wp.zeros(size, dtype=float, device=model.device)
        self._final_residual = wp.zeros(size, dtype=float, device=model.device)
        self._contact_status = wp.zeros(1, dtype=int, device=model.device)
        self._default_control = model.control(clone_variables=False)
        self._generation = None
        self._assembly_sequence = 0

    @property
    def last_stats(self) -> SolverMonolithic.Stats:
        """Return diagnostics from the last completed step."""
        return self._transaction.last_stats

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        """Advance by ``dt`` [s], or restore the input on a hard numerical failure.

        Exhausting nonlinear iterations commits the last safe accepted state
        with an unconverged status. If contact publication also fails on the
        rollback state, restore the input, invalidate forces and raise RuntimeError.
        """
        self._resolve_step_contacts(contacts)
        _validate_dt(dt)
        _validate_state(self.model, state_in)
        _validate_state(self.model, state_out)
        _validate_dirichlet(self.model, self._layout, state_in)
        pipeline = self.collision_pipeline
        pipeline.validate_contacts(self.contacts)
        pipeline.validate_contacts(self._trial_contacts)
        control = self._default_control if control is None else control
        self._articulation.validate_candidate(
            self.model,
            state_in,
            self._transaction.accepted.qdd,
            control.joint_f,
            state_in.body_f,
        )
        start = time.perf_counter()
        self._transaction.begin(state_in, state_out, control, dt)
        self._contact.invalidate_final_force()
        self._assembly_sequence = 0
        self._metrics = {
            "nonlinear_iterations": 0,
            "line_search_iterations": 0,
            "linear_iterations": 0,
            "regularization_retries": 0,
            "matrix_assembly_count": 0,
            "true_residual_recomputations": 0,
            "residual_replacements": 0,
            "preconditioner_kind": "actor_block",
            "triplet_capacity": self._linear.capacities.global_scalar_triplet_count,
            "merit_noise": self._config.merit_noise,
            "residual_floor_global": self._config.residual_floor_global,
            "residual_floor_q": self._config.residual_floor_q,
            "residual_floor_x": self._config.residual_floor_x,
        }
        reason = None
        try:
            try:
                status = self._iterate(dt)
            except _StepFailure as error:
                status, reason = error.status, str(error)
            self._finish_step(state_out, status, reason, start)
        except Exception:
            # A programming/configuration exception must not leave a reusable
            # solver in an open transaction or retain a partially published state.
            state_out.assign(self._transaction._original)
            self._contact.invalidate_final_force()
            self._transaction._finished = True
            raise

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Idempotently publish cached forces for the last returned state."""
        if contacts is not self.contacts:
            raise ValueError("update_contacts requires solver.contacts")
        generation = self._contact.final_force_generation
        if generation is None:
            raise ValueError("No valid final contact publication")
        state = generation.returned_state if state is None else state
        self._contact._publish_final_forces(
            state,
            contacts,
            step_generation=self._transaction.step_generation,
            returned_state_role=generation.returned_state_role,
        )

    def _rms(self, values: wp.array[float], *, scaled: bool = True) -> tuple[float, float, float]:
        if scaled:
            wp.launch(
                _scale_vector,
                values.size,
                [values, self._linear.scale, 1.0, self._metric_vector],
                device=self.model.device,
            )
        else:
            wp.copy(self._metric_vector, values)
        self._norm_values.zero_()
        for vector, output in zip(self._metric_slices, self._norm_outputs, strict=True):
            if vector.size:
                wp.utils.array_inner(vector, vector, out=output)
        norms = self._norm_values.numpy()
        if not np.isfinite(norms).all() or np.any(norms < 0):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite scaled reduction")
        return tuple(
            math.sqrt(float(value) / max(vector.size, 1))
            for value, vector in zip(norms, self._metric_slices, strict=True)
        )

    def _converged(self, merit: tuple[float, float, float], reference: tuple[float, float, float]) -> bool:
        config = self._config
        return all(
            value <= getattr(config, f"merit_absolute_{block}") + getattr(config, f"merit_relative_{block}") * ref
            for block, value, ref in zip(("global", "q", "x"), merit, reference, strict=True)
        )

    def _collide(self, state: State, contacts: Contacts) -> None:
        try:
            self.collision_pipeline.collide(state, contacts, dt=0.0)
        except MonolithicCollisionPipeline.Error as error:
            if error.status == self.collision_pipeline.Status.CONTACT_CAPACITY_OVERFLOW:
                raise _StepFailure(self.Status.CONTACT_OVERFLOW, str(error)) from error
            if error.status == self.collision_pipeline.Status.INVALID_SDF_GRADIENT:
                raise _StepFailure(self.Status.NONFINITE, str(error)) from error
            raise

    def _actor_residual(self, candidate: _Candidate) -> None:
        model, articulation, inputs = self.model, self._articulation, self._transaction.inputs
        eval_articulation_passive_candidate(model, candidate.state, articulation)
        nq = self._layout.q_dof_count
        wp.launch(
            project_articulation_body_wrenches,
            nq,
            [
                0,
                nq,
                model.articulation_start,
                model.articulation_end,
                model.joint_child,
                articulation.scratch.J,
                inputs.body_f,
                articulation.generalized_body_force,
            ],
            device=model.device,
        )
        wp.launch(
            eval_articulation_actor_residual,
            nq,
            [
                0,
                0,
                nq,
                articulation.M,
                candidate.qdd,
                articulation.C,
                articulation.g,
                inputs.joint_f,
                articulation.generalized_body_force,
                candidate.residual,
            ],
            device=model.device,
        )

    def _tet_arguments(self, candidate: _Candidate, dt: float) -> dict:
        return {
            "candidate_particle_q": candidate.state.particle_q,
            "particle_q_n": self._transaction._original.particle_q,
            "particle_qd_n": self._transaction._original.particle_qd,
            "frozen_particle_f": self._transaction.inputs.particle_f,
            "dynamic_particle_ids": self._layout.dynamic_particle_ids,
            "particle_to_dynamic": self._layout.particle_to_dynamic,
            "residual_x": self._particle_residual,
            "dt": dt,
            "min_det_f_guard": self._config.det_f_guard,
            "workspace": self._tet_workspace,
        }

    def _check_tet(self) -> float:
        code = TetEvaluationStatus(int(self._tet_workspace.failure_flags.numpy()[0]))
        if code == TetEvaluationStatus.DET_F_GUARD:
            raise _StepFailure(self.Status.TET_INVERSION, "Tet determinant guard")
        if code == TetEvaluationStatus.NONFINITE:
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite tet evaluation")
        if code != TetEvaluationStatus.SUCCESS:
            raise ValueError(f"Invalid tet evaluation contract: {code.name}")
        return float(self._tet_workspace.min_det_f.numpy()[0])

    def _check_contact(self) -> None:
        code = int(self._contact_status.numpy()[0])
        if code == int(MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW):
            raise _StepFailure(self.Status.CONTACT_OVERFLOW, "Contact factor capacity")
        if code == int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite contact evaluation")
        if code:
            raise ValueError(f"Invalid contact evaluation contract: {code}")

    def _evaluate_current(self, candidate: _Candidate, dt: float) -> _TrialResult:
        self._actor_residual(candidate)
        self._collide(candidate.state, self._trial_contacts)
        self._assembly_sequence += 1
        generation = MonolithicLinearGeneration(
            self._transaction.step_generation,
            self._metrics["nonlinear_iterations"],
            int(self._trial_contacts.contact_generation.numpy()[0]),
            self._assembly_sequence,
        )
        self._generation = generation
        assembly = self._linear.begin_assembly(generation)
        nq, model = self._layout.q_dof_count, self.model
        triplets = assembly.global_scalar_triplets
        wp.launch(
            scatter_articulation_actor_tangent,
            (nq, nq),
            [
                0,
                nq,
                0,
                0,
                1.0 / dt**2,
                self._articulation.M,
                assembly.aq_actor_dense,
                triplets.rows,
                triplets.columns,
                triplets.values,
            ],
            device=model.device,
        )
        assemble_tet_residual_tangent(
            model,
            **self._tet_arguments(candidate, dt),
            scatter=TetScatterBuffers(
                assembly.ax_internal_triplets.values, triplets.values[nq * nq : nq * nq + self._tet_triplet_count]
            ),
        )
        min_det = self._check_tet()
        wp.launch(
            _copy_particle_residual,
            self._layout.dynamic_particle_count,
            [self._particle_residual, nq, candidate.residual],
            device=model.device,
        )
        assemble_current_contacts(
            model,
            candidate.state,
            self._trial_contacts,
            self.collision_pipeline,
            self._articulation,
            self._contact,
            assembly,
            candidate.residual,
            generation=generation,
        )
        result = self._linear.finalize_assembly(generation=generation)
        self._metrics["matrix_assembly_count"] += 1
        self._metrics.update(
            matrix_nnz=result.global_nnz,
            triplet_count=result.global_triplet_count,
            owner_generation=self._assembly_sequence,
        )
        self._require_linear(result.status)
        wp.launch(
            _build_dynamic_diagonal,
            self._layout.scalar_dof_count,
            [
                self._articulation.M,
                model.particle_mass,
                self._layout.dynamic_particle_ids,
                nq,
                1.0 / dt**2,
                self._dynamic_diagonal,
            ],
            device=model.device,
        )
        self._require_linear(
            self._linear.build_scaling(self._dynamic_diagonal, epsilon_d=self._config.epsilon_d, generation=generation)
        )
        return _TrialResult(
            0, *self._rms(candidate.residual), min_det, self._contact._diagnostics(0)["max_penetration"]
        )

    def _evaluate_trial(self, candidate: _Candidate, dt: float) -> _TrialResult:
        self._actor_residual(candidate)
        self._collide(candidate.state, self._trial_contacts)
        evaluate_tet_residual(self.model, **self._tet_arguments(candidate, dt))
        min_det = self._check_tet()
        wp.launch(
            _copy_particle_residual,
            self._layout.dynamic_particle_count,
            [self._particle_residual, self._layout.q_dof_count, candidate.residual],
            device=self.model.device,
        )
        self._contact_status.zero_()
        evaluate_trial_contacts(
            self.model,
            candidate.state,
            self._trial_contacts,
            self.collision_pipeline,
            self._articulation,
            self._contact,
            candidate.residual,
            self._contact_status,
            owner_generation=self._generation,
            trial_generation=candidate.generation,
        )
        self._check_contact()
        return _TrialResult(
            0, *self._rms(candidate.residual), min_det, self._contact._diagnostics(1)["max_penetration"]
        )

    def _require_linear(self, status: MonolithicLinearStatus) -> None:
        if status == MonolithicLinearStatus.SUCCESS:
            return
        mapped = {
            "NON_POSITIVE_PIVOT": self.Status.PRECONDITIONER_FACTORIZATION_FAILED,
            "NON_POSITIVE_CURVATURE": self.Status.LINEAR_NON_POSITIVE_CURVATURE,
            "NEAR_ZERO_CURVATURE": self.Status.LINEAR_NON_POSITIVE_CURVATURE,
            "NON_POSITIVE_PRECONDITIONED_RESIDUAL": self.Status.LINEAR_PRECONDITIONER_NOT_SPD,
            "STAGNATION": self.Status.LINEAR_STAGNATION,
            "MAX_ITERATIONS": self.Status.LINEAR_MAX_ITERATIONS,
            "GLOBAL_TRIPLET_OVERFLOW": self.Status.CONTACT_OVERFLOW,
            "CONTACT_FACTOR_OVERFLOW": self.Status.CONTACT_OVERFLOW,
        }
        if status.name in ("STALE_GENERATION", "INVALID_ARGUMENT", "PRECONDITIONER_NOT_FACTORED"):
            raise ValueError(f"Invalid linear workspace contract: {status.name}")
        raise _StepFailure(mapped.get(status.name, self.Status.NONFINITE), status.name)

    def _iterate(self, dt: float) -> SolverMonolithic.Status:
        config, transaction = self._config, self._transaction
        small_step = False
        for iteration in range(self.newton_max_iterations + 1):
            self._metrics["nonlinear_iterations"] = iteration
            current = self._evaluate_current(transaction.accepted, dt)
            if iteration == 0:
                wp.copy(self._residual_ref, transaction.accepted.residual)
                self._metrics.update(
                    merit_initial=current.merit, merit_q_initial=current.merit_q, merit_x_initial=current.merit_x
                )
            merit = (current.merit, current.merit_q, current.merit_x)
            reference = self._rms(self._residual_ref)
            self._raw_norm_sums.zero_()
            wp.launch(
                _raw_residual_sums,
                self._layout.scalar_dof_count,
                [transaction.accepted.residual, self._layout.q_dof_count, self._raw_norm_sums],
                device=self.model.device,
            )
            raw_norms = np.sqrt(self._raw_norm_sums.numpy())
            ratios = tuple(
                value / (getattr(config, f"merit_absolute_{block}") + getattr(config, f"merit_relative_{block}") * ref)
                for block, value, ref in zip(("global", "q", "x"), merit, reference, strict=True)
            )
            self._metrics.update(
                merit_final=merit[0],
                merit_q_final=merit[1],
                merit_x_final=merit[2],
                min_det_f=current.min_det_f,
                merit_reference=reference[0],
                merit_q_reference=reference[1],
                merit_x_reference=reference[2],
                convergence_ratio=ratios[0],
                convergence_ratio_q=ratios[1],
                convergence_ratio_x=ratios[2],
                scale_generation=self._generation.assembly_sequence,
                raw_residual_q_norm=float(raw_norms[0]),
                raw_residual_x_norm=float(raw_norms[1]),
            )
            if self._converged(merit, reference):
                return self.Status.SUCCESS
            if small_step:
                raise _StepFailure(self.Status.NONLINEAR_STAGNATION, "Small scaled step with unresolved residual")
            if iteration == self.newton_max_iterations:
                return self.Status.NONLINEAR_MAX_ITERATIONS
            wp.launch(
                _scale_vector,
                self._layout.scalar_dof_count,
                [transaction.accepted.residual, self._linear.scale, -1.0, self._rhs_hat],
                device=self.model.device,
            )
            pcg_config = MonolithicPcgConfig(
                maximum_iterations=self.linear_max_iterations,
                true_residual_interval=10,
                linear_tolerance=self.linear_tolerance,
                residual_floor_global=config.residual_floor_global,
                residual_floor_q=config.residual_floor_q,
                residual_floor_x=config.residual_floor_x,
                curvature_absolute_tolerance=0.0,
                curvature_relative_tolerance=1.0e-12,
                preconditioner_positive_tolerance=0.0,
                stagnation_window=10,
                stagnation_minimum_reduction=1.0e-3,
            )
            accepted = False
            last_reason = "No acceptable trial"
            for retry, lambda_value in enumerate(config.regularization_values):
                self._metrics["regularization_retries"] += int(retry > 0)
                self._metrics["lambda_value"] = lambda_value
                try:
                    self._require_linear(self._linear.set_regularization(lambda_value, generation=self._generation))
                    self._require_linear(
                        self._linear.factor_actor_preconditioner(generation=self._generation, pivot_tolerance=0.0)
                    )
                    result = self._linear.solve_pcg(
                        self._rhs_hat,
                        self._y,
                        generation=self._generation,
                        warm_start=MonolithicPcgWarmStart.ZERO,
                        config=pcg_config,
                    )
                    self._metrics["linear_iterations"] += result.iterations
                    self._metrics["true_residual_recomputations"] += result.true_residual_checks
                    self._metrics["residual_replacements"] += result.residual_replacements
                    self._metrics.update(
                        rho=result.rho,
                        rho_q=result.rho_q,
                        rho_x=result.rho_x,
                        min_p_ap=result.min_p_ap,
                        min_r_z=result.min_r_z,
                    )
                    self._require_linear(result.status)
                    self._require_linear(self._linear.recover_delta(self._y, self._delta, generation=self._generation))
                except _StepFailure as error:
                    last_reason = str(error)
                    continue
                step_rms = self._rms(self._y, scaled=False)
                for search in range(self.line_search_max_iterations):
                    alpha = 0.5**search
                    self._metrics["line_search_iterations"] += 1
                    transaction.trial.form_trial(transaction.accepted, self._delta, alpha, dt)
                    try:
                        trial = self._evaluate_trial(transaction.trial, dt)
                    except _StepFailure as error:
                        last_reason = str(error)
                        continue
                    if trial.merit <= (1.0 - 1.0e-4 * alpha) * merit[0] + config.merit_noise:
                        transaction.accept_trial()
                        self._metrics["accepted_alpha"] = alpha
                        self._metrics.update(
                            scaled_step=alpha * step_rms[0],
                            scaled_step_q=alpha * step_rms[1],
                            scaled_step_x=alpha * step_rms[2],
                        )
                        small_step = all(
                            alpha * value <= getattr(config, f"step_tolerance_{block}")
                            for block, value in zip(("global", "q", "x"), step_rms, strict=True)
                        )
                        accepted = True
                        break
                    last_reason = "Actual trial merit failed the decrease condition"
                if accepted:
                    break
            if not accepted:
                raise _StepFailure(self.Status.REGULARIZATION_EXHAUSTED, last_reason)
        raise AssertionError("Unreachable nonlinear loop exit")

    def _publish_final(self, state: State, status: SolverMonolithic.Status) -> None:
        roles = {
            self.Status.SUCCESS: _MonolithicReturnedStateRole.STATE_OUT_CONVERGED,
            self.Status.NONLINEAR_MAX_ITERATIONS: _MonolithicReturnedStateRole.STATE_OUT_SOFT_STOP,
        }
        role = roles.get(status, _MonolithicReturnedStateRole.STATE_IN_ROLLBACK)
        # FK refresh must not overwrite the original body caches on rollback.
        self._final_evaluation_state.assign(state)
        eval_articulation_passive_candidate(self.model, self._final_evaluation_state, self._articulation)
        self._collide(state, self.contacts)
        self._final_residual.zero_()
        self._contact_status.zero_()
        evaluate_final_contacts(
            self.model,
            state,
            self.contacts,
            self.collision_pipeline,
            self._articulation,
            self._contact,
            self._final_residual,
            self._contact_status,
            step_generation=self._transaction.step_generation,
            returned_state_role=role,
        )
        self._check_contact()
        if self.contacts.force is not None:
            self._contact._publish_final_forces(
                state, self.contacts, step_generation=self._transaction.step_generation, returned_state_role=role
            )

    def _finish_step(self, state_out: State, status: SolverMonolithic.Status, reason: str | None, start: float) -> None:
        transaction = self._transaction
        commit = status in (self.Status.SUCCESS, self.Status.NONLINEAR_MAX_ITERATIONS)
        if commit:
            transaction.accepted.commit(state_out)
        else:
            state_out.assign(transaction._original)
        publication_error = None
        try:
            self._publish_final(state_out, status)
        except _StepFailure as error:
            status, reason, commit = error.status, str(error), False
            state_out.assign(transaction._original)
            try:
                self._publish_final(state_out, status)
            except _StepFailure as second_error:
                publication_error = second_error
                self._contact.invalidate_final_force()
        diagnostics = self._contact._diagnostics(2) if publication_error is None else {}
        transaction.last_stats = self.Stats(
            status=status,
            failure_reason=reason,
            converged=status == self.Status.SUCCESS,
            rolled_back=not commit,
            step_generation=transaction.step_generation,
            accepted_generation=transaction.accepted.generation,
            soft_contact_pair_count=self.collision_pipeline.soft_contact_pair_count,
            published_contact_generation=int(self.contacts.contact_generation.numpy()[0])
            if publication_error is None
            else -1,
            active_sample_count=diagnostics.get("active_sample_count", 0),
            max_penetration=diagnostics.get("max_penetration", math.nan),
            contact_force_imbalance=diagnostics.get("force_imbalance", math.nan),
            contact_moment_imbalance=diagnostics.get("moment_imbalance", math.nan),
            generalized_projection_error=diagnostics.get("generalized_projection_error", math.nan),
            contact_sign_error=diagnostics.get("contact_sign_error", math.nan),
            timings={"step": 1000.0 * (time.perf_counter() - start)},
            **self._metrics,
        )
        transaction._finished = True
        if status != self.Status.SUCCESS:
            stats = transaction.last_stats
            logging.getLogger(__name__).warning(
                "Monolithic status=%s failure_reason=%s converged=%s rolled_back=%s step_generation=%d nonlinear_iterations=%d linear_iterations=%d rho=%s",
                status.value,
                reason,
                stats.converged,
                stats.rolled_back,
                stats.step_generation,
                stats.nonlinear_iterations,
                stats.linear_iterations,
                stats.rho,
            )
        if publication_error is not None:
            raise RuntimeError(
                "Contact publication failed on the rollback state; forces are invalid"
            ) from publication_error


class _StepTransaction:
    """Stage candidate state and frozen inputs for physical evaluation.

    This helper does not collide or publish final forces. Its caller supplies
    evaluated safety and terminal diagnostics. The solver owns physical
    evaluation and the final contact transaction.
    Optional and custom state arrays are unsupported.
    Input/output schemas are validated before copying, including at terminal
    publication in case the caller changed the destination after ``begin``.
    """

    def __init__(self, model: Model, layout: _MonolithicLayout):
        self._model, self._layout = model, layout
        self.accepted = _Candidate(model, layout)
        self.trial = _Candidate(model, layout)
        self.inputs = _FrozenStepInputs(model)
        self._original = model.state()
        self._state_out = None
        self._finished = True
        self.step_generation = 0
        self.last_stats = SolverMonolithic.Stats(SolverMonolithic.Status.NOT_RUN)

    def begin(self, state_in: State, state_out: State, control: Control | None, dt: float) -> None:
        if not self._finished:
            raise RuntimeError("The previous transaction has not finished")
        _validate_dt(dt)
        _validate_state(self._model, state_in)
        _validate_state(self._model, state_out)
        _validate_dirichlet(self._model, self._layout, state_in)
        self.inputs.snapshot(state_in, self._model.control() if control is None else control)
        self._original.assign(state_in)
        self.accepted.load_predictor(state_in, dt)
        self.trial.generation = 0
        self.trial._trial_source = None
        self._state_out = state_out
        self.step_generation += 1
        self._finished = False

    def accept_trial(self) -> None:
        if (
            self._finished
            or self.trial._trial_source is not self.accepted
            or self.trial._trial_source_generation != self.accepted.generation
        ):
            raise RuntimeError("No current trial to accept")
        generation = self.accepted.generation + 1
        self.accepted, self.trial = self.trial, self.accepted
        self.accepted.generation = generation

    def finish_state(
        self,
        status: SolverMonolithic.Status,
        *,
        accepted_safe: bool,
        failure_reason: str | None = None,
        rho: float = math.nan,
        nonlinear_iterations: int = 0,
        linear_iterations: int = 0,
    ) -> SolverMonolithic.Stats:
        if self._finished:
            raise RuntimeError("Transaction has already finished")
        if not isinstance(status, SolverMonolithic.Status) or status == SolverMonolithic.Status.NOT_RUN:
            raise ValueError("A terminal status is required")
        commit = status in (SolverMonolithic.Status.SUCCESS, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
        if commit and not accepted_safe:
            raise ValueError("Committing a candidate requires a safe accepted evaluation")
        _validate_state(self._model, self._state_out)
        if commit:
            self.accepted.commit(self._state_out)
        else:
            _validate_state(self._model, self._original)
            self._state_out.assign(self._original)
        self.last_stats = SolverMonolithic.Stats(
            status=status,
            failure_reason=failure_reason,
            converged=status == SolverMonolithic.Status.SUCCESS,
            rolled_back=not commit,
            step_generation=self.step_generation,
            accepted_generation=self.accepted.generation,
            nonlinear_iterations=nonlinear_iterations,
            linear_iterations=linear_iterations,
            rho=rho,
        )
        self._finished = True
        if status != SolverMonolithic.Status.SUCCESS:
            logging.getLogger(__name__).warning(
                "Monolithic status=%s failure_reason=%s converged=%s rolled_back=%s step_generation=%d nonlinear_iterations=%d linear_iterations=%d rho=%s",
                status.value,
                failure_reason,
                self.last_stats.converged,
                self.last_stats.rolled_back,
                self.step_generation,
                nonlinear_iterations,
                linear_iterations,
                rho,
            )
        return self.last_stats
