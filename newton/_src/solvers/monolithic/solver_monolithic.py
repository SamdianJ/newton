# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import warp as wp

from ...sim import Contacts, Control, JointType, Model, State, eval_fk
from ..solver import SolverBase


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
    q_map = model.joint_q_start.numpy()[:-1][moving]
    qd_map = model.joint_qd_start.numpy()[:-1][moving]
    nq = len(q_map)
    if nq != model.joint_dof_count or nq != model.joint_coord_count:
        raise ValueError("Inconsistent revolute/prismatic coordinate layout")

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


# Pure BE kinematics for PR-0. Move/reuse in articulation/tet owners when those
# modules land; no inertia, force or material formulas belong to the shell.
@wp.kernel
def _recover_joint_kinematics(
    z: wp.array[float],
    q0: wp.array[float],
    v0: wp.array[float],
    q_map: wp.array[int],
    v_map: wp.array[int],
    dt: float,
    velocity: wp.array[float],
    acceleration: wp.array[float],
):
    i = wp.tid()
    displacement = z[i] - q0[q_map[i]]
    velocity[v_map[i]] = displacement / dt
    acceleration[i] = (displacement / dt - v0[v_map[i]]) / dt


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
            _recover_joint_kinematics,
            layout.q_dof_count,
            inputs=[
                self.z,
                origin.joint_q,
                origin.joint_qd,
                layout.q_dof_to_joint_q,
                layout.q_dof_to_joint_qd,
                dt,
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
    """Experimental monolithic solver shell; simulation stepping is not implemented.

    The constructor, :meth:`step`, and :meth:`update_contacts` currently raise
    :class:`NotImplementedError`. The articulation, tet, contact, and nonlinear
    solve implementations must land before this experimental API can run.

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

        Contact publication is absent in the PR-0 transaction scaffold, so its
        generation is -1 and its contact diagnostics remain unmeasured.
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
        merit_final: float = math.nan
        merit_q_final: float = math.nan
        merit_x_final: float = math.nan
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
        """Maximum contact penetration [m]; unmeasured in PR-0."""
        min_det_f: float = math.nan
        """Minimum tet deformation determinant [dimensionless]; unmeasured in PR-0."""
        contact_force_imbalance: float = math.nan
        contact_moment_imbalance: float = math.nan
        generalized_projection_error: float = math.nan
        contact_sign_error: float = math.nan
        timings: Mapping[str, float | None] = field(default_factory=lambda: MappingProxyType({}))
        """Measured durations [ms], or None for unmeasured entries; empty in PR-0."""

        def __post_init__(self):
            object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))

    def __init__(
        self,
        model: Model,
        *,
        collision_pipeline,
        contact_stiffness: float,
        newton_max_iterations: int = 10,
        line_search_max_iterations: int = 8,
        linear_max_iterations: int = 200,
        linear_tolerance: float = 1.0e-4,
    ) -> None:
        raise NotImplementedError("Monolithic physics and collision pipeline are not implemented")

    @property
    def last_stats(self) -> SolverMonolithic.Stats:
        """Return diagnostics from the last completed step."""
        raise NotImplementedError("Monolithic stepping is not implemented")

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        """Advance by ``dt`` [s] once the monolithic physics implementation exists."""
        raise NotImplementedError("Monolithic physics and collision pipeline are not implemented")

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Publish final physical contact forces once contact evaluation exists."""
        raise NotImplementedError("Monolithic final contact publication is not implemented")


class _StepTransaction:
    """Stage state-only transactions for integration with future physical evaluation.

    This helper does not collide or publish final forces. Its caller supplies
    evaluated safety and terminal diagnostics. It cannot implement a solver step
    until the final contact transaction and physics modules have landed.
    Optional and custom state arrays are unsupported in this PR-0 scaffold.
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
