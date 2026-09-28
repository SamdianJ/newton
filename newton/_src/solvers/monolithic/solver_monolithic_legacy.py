# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Historical solver; retained as the transitional P3 parent until the T2 transaction merge."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import Contacts, Control, JointType, Model, State, eval_fk
from ..solver import SolverBase
from .articulation import (
    MonolithicArticulationWorkspace,
    MonolithicJointTermsWorkspace,
    _validate_joint_coordinate_layout,
    add_joint_term_residual,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
    project_articulation_body_wrenches,
    recover_articulation_candidate_rates,
    scatter_articulation_actor_tangent,
    scatter_joint_term_tangent,
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
from .segment_graph import _SegmentGraphCache
from .solver_types import (
    JointTerms,
    MassMode,
    MaterialModel,
    PCGMode,
    Stats,
    Status,
    _SolverMonolithicInternalConfig,
    _StepFailure,
)
from .tet import (
    TetEvaluationStatus,
    TetPhysicsWorkspace,
    TetScatterBuffers,
    _validate_evaluation,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
    evaluate_tet_residual,
    validate_tet_scope,
)
from .tet_variational import TetObjectiveWorkspace, acceptance


@dataclass(frozen=True, slots=True)
class _TrialResult:
    status: int
    merit: float
    merit_q: float
    merit_x: float
    min_det_f: float
    maximum_penetration: float
    objective: tuple[float, ...] | None = None
    objective_abs_sum: float = 0.0


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


def _build_layout(model: Model, *, joint_diagnostic: bool = False, tet_only: bool = False) -> _MonolithicLayout:
    if tet_only and (model.articulation_count or model.joint_count or model.body_count or model.shape_count):
        raise ValueError("Tet-only requires zero articulations, joints, bodies and shapes")
    if joint_diagnostic:
        flags = model.shape_flags.numpy()
        if (
            model.particle_count
            or model.tet_count
            or np.any(flags & int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES))
        ):
            raise ValueError("Joint diagnostic requires zero particles/tets and collision-disabled shapes")
    if (not tet_only and model.articulation_count != 1) or (model.tet_count == 0 and not joint_diagnostic):
        raise ValueError("Monolithic requires one articulation and one connected tet body")
    starts = model.articulation_start.numpy()
    ends = model.articulation_end.numpy()
    if not tet_only and (starts.tolist() != [0, model.joint_count] or ends.tolist() != [model.joint_count]):
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
    if not tet_only and (roots != 1 or len(seen) != model.body_count):
        raise ValueError("Monolithic requires one world-anchored tree owning all bodies")
    tets = model.tet_indices.numpy() if model.tet_count else np.empty((0, 4), dtype=np.int32)
    if np.any(tets < 0) or np.any(tets >= model.particle_count) or any(len(set(t)) != 4 for t in tets):
        raise ValueError("Invalid tet topology")
    target = np.unique(tets)
    connected = set(tets[0]) if len(tets) else set()
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
    if tet_only and (not len(dynamic) or len(target) != model.particle_count):
        raise ValueError("Tet-only requires dynamic nodes and all particles owned by the connected tet body")
    particle_to_dynamic = np.full(model.particle_count, -1, dtype=np.int32)
    particle_to_dynamic[dynamic] = np.arange(len(dynamic), dtype=np.int32)
    moving = types != JointType.FIXED
    q_map = np.empty(0, dtype=np.int32) if tet_only else _validate_joint_coordinate_layout(model)
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
    target = np.unique(model.tet_indices.numpy()) if model.tet_count else np.empty(0, dtype=np.int32)
    labels = np.concatenate(
        (model.body_world.numpy()[bodies], model.shape_world.numpy()[shapes], model.particle_world.numpy()[target])
    )
    if len(np.unique(labels)) != 1 or np.any(labels < -1) or np.any(labels >= model.world_count):
        raise ValueError("Monolithic requires exactly one participating world label")


def _validate_dt(dt: float) -> None:
    if not math.isfinite(dt) or dt <= 0.0 or dt > np.finfo(np.float32).max or np.float32(dt) <= 0.0:
        raise ValueError("dt must be finite and positive in float32")


def _validate_state(model: Model, state: State, *, tet_only: bool = False) -> None:
    """Reject optional/custom state arrays until their transaction semantics exist."""
    attributes = (
        ("joint_q", model.joint_coord_count, wp.float32),
        ("joint_qd", model.joint_dof_count, wp.float32),
        ("body_q", model.body_count, wp.transform),
        ("body_qd", model.body_count, wp.spatial_vector),
        ("body_f", model.body_count, wp.spatial_vector),
        ("particle_q", model.particle_count, wp.vec3d if tet_only else wp.vec3),
        ("particle_qd", model.particle_count, wp.vec3d if tet_only else wp.vec3),
        ("particle_f", model.particle_count, wp.vec3),
    )
    for name, count, dtype in attributes:
        value = getattr(state, name)
        if count == 0 and value is None:
            continue
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


def _validate_dirichlet(model: Model, layout: _MonolithicLayout, state: State, *, tet_only: bool = False) -> None:
    if not model.tet_count:
        return
    target = np.arange(model.particle_count) if tet_only else np.unique(model.tet_indices.numpy())
    fixed = target[layout.particle_to_dynamic.numpy()[target] < 0]
    if np.any(state.particle_qd.numpy()[fixed] != 0.0):
        raise ValueError("Static Dirichlet particles require exactly zero input velocity")


class _FrozenStepInputs:
    def __init__(self, model: Model):
        self.joint_f = wp.zeros(model.joint_dof_count, dtype=float, device=model.device) if model.joint_count else None
        self.body_f = (
            wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device) if model.body_count else None
        )
        self.gravity = wp.clone(model.gravity)
        self._model = model
        self.particle_f = (
            wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device) if model.particle_count else None
        )

    def snapshot(self, state_in: State, control: Control) -> None:
        for target, source in (
            (self.joint_f, control.joint_f),
            (self.body_f, state_in.body_f),
            (self.particle_f, state_in.particle_f),
        ):
            if target is None and (source is None or source.size == 0):
                continue
            if (
                target is None
                or source is None
                or target.shape != source.shape
                or target.dtype != source.dtype
                or target.device != source.device
            ):
                raise ValueError("Invalid frozen input shape, dtype or device")
        if self.joint_f is not None:
            wp.copy(self.joint_f, control.joint_f)
        if self.body_f is not None:
            wp.copy(self.body_f, state_in.body_f)
        if self.particle_f is not None:
            wp.copy(self.particle_f, state_in.particle_f)
        wp.copy(self.gravity, self._model.gravity)


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
def _initialize_joint_displacement(
    velocity: wp.array[float],
    qd_map: wp.array[int],
    dt: float,
    displacement: wp.array[float],
):
    i = wp.tid()
    displacement[i] = dt * velocity[qd_map[i]]


@wp.kernel
def _recover_joint_displacement(
    displacement: wp.array[float],
    q_map: wp.array[int],
    qd_map: wp.array[int],
    q_n: wp.array[float],
    v_n: wp.array[float],
    dt: float,
    q: wp.array[float],
    v: wp.array[float],
    acceleration: wp.array[float],
    z: wp.array[float],
):
    i = wp.tid()
    # Rebase BE at q_n: subtracting rounded absolute angles loses sub-ULP updates.
    velocity = displacement[i] / dt
    position = q_n[q_map[i]] + displacement[i]
    q[q_map[i]] = position
    z[i] = position
    v[qd_map[i]] = velocity
    acceleration[i] = (velocity - v_n[qd_map[i]]) / dt


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


def _new_candidate_state(model: Model, tet_only: bool) -> State:
    state = model.state()
    if tet_only:
        for name in ("particle_q", "particle_qd"):
            setattr(
                state,
                name,
                wp.array(getattr(model, name).numpy().astype(np.float64), dtype=wp.vec3d, device=model.device),
            )
    return state


@wp.kernel
def _predict_tet_coordinates(
    x: wp.array[wp.vec3d], v: wp.array[wp.vec3d], ids: wp.array[int], dt: wp.float64, z: wp.array[wp.float64]
):
    i = wp.tid()
    z[i] = x[ids[i // 3]][i % 3] + dt * v[ids[i // 3]][i % 3]


@wp.kernel
def _form_tet_trial(x: wp.array[wp.float64], p: wp.array[float], alpha: wp.float64, out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = x[i] + alpha * wp.float64(p[i])


@wp.kernel
def _recover_tet_candidate(
    z: wp.array[wp.float64],
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    ids: wp.array[int],
    dt: wp.float64,
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    acceleration: wp.array[wp.vec3d],
):
    i = wp.tid()
    p = ids[i]
    position = wp.vec3d(z[3 * i], z[3 * i + 1], z[3 * i + 2])
    velocity = (position - xn[p]) / dt
    x[p] = position
    v[p] = velocity
    acceleration[p] = (velocity - vn[p]) / dt


class _Candidate:
    def __init__(self, model: Model, layout: _MonolithicLayout, *, tet_only: bool = False):
        self._model = model
        self._layout = layout
        self._tet_only = tet_only
        self._origin = _new_candidate_state(model, tet_only)
        self._dt = None
        self._trial_source = None
        self._trial_source_generation = -1
        self.state = _new_candidate_state(model, tet_only)
        _validate_state(model, self.state, tet_only=tet_only)
        self.z = wp.zeros(layout.scalar_dof_count, dtype=wp.float64 if tet_only else float, device=model.device)
        self.qdd = wp.zeros(layout.q_dof_count, device=model.device)
        self.joint_displacement = wp.zeros(layout.q_dof_count, device=model.device)
        self.particle_acceleration = (
            wp.zeros(model.particle_count, dtype=wp.vec3d if tet_only else wp.vec3, device=model.device)
            if model.particle_count
            else None
        )
        self.residual = wp.zeros(layout.scalar_dof_count, device=model.device)
        self.generation = 0

    def load_predictor(self, state_in: State, dt: float) -> None:
        _validate_dt(dt)
        _validate_state(self._model, state_in, tet_only=self._tet_only)
        _validate_dirichlet(self._model, self._layout, state_in, tet_only=self._tet_only)
        self._origin.assign(state_in)
        self.state.assign(state_in)
        self._dt = dt
        layout = self._layout
        if self._tet_only:
            wp.launch(
                _predict_tet_coordinates,
                layout.scalar_dof_count,
                [state_in.particle_q, state_in.particle_qd, layout.dynamic_particle_ids, dt, self.z],
                device=self._model.device,
            )
            self._recover(dt)
            self.generation = 0
            self._trial_source = None
            return
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
        wp.launch(
            _initialize_joint_displacement,
            layout.q_dof_count,
            [state_in.joint_qd, layout.q_dof_to_joint_qd, dt, self.joint_displacement],
            device=self._model.device,
        )
        self._recover(dt, from_displacement=True)
        self.generation = 0
        self._trial_source = None

    def form_trial(self, accepted: _Candidate, delta: wp.array[float], alpha: float, dt: float) -> None:
        _validate_dt(dt)
        if accepted is self or accepted._model is not self._model or accepted._layout is not self._layout:
            raise ValueError("Trial requires a distinct candidate with the same model and layout")
        if accepted._dt != dt:
            raise ValueError("Trial dt must equal the frozen predictor dt")
        if delta.shape != self.z.shape or delta.dtype != wp.float32 or delta.device != self.z.device:
            raise ValueError("Invalid trial delta shape, dtype or device")
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("Trial alpha must be finite and in (0, 1]")
        self._origin.assign(accepted._origin)
        self.state.assign(accepted.state)
        self._dt = dt
        if self._tet_only:
            wp.launch(_form_tet_trial, self.z.size, [accepted.z, delta, alpha, self.z], device=self._model.device)
            self._recover(dt)
            self.generation += 1
            self._trial_source = accepted
            self._trial_source_generation = accepted.generation
            return
        wp.launch(
            _form_monolithic_trial,
            self._layout.scalar_dof_count,
            inputs=[accepted.z, delta, alpha, self.z],
            device=self._model.device,
        )
        wp.launch(
            _form_monolithic_trial,
            self._layout.q_dof_count,
            [accepted.joint_displacement, delta, alpha, self.joint_displacement],
            device=self._model.device,
        )
        self._recover(dt, from_displacement=True)
        self.generation += 1
        self._trial_source = accepted
        self._trial_source_generation = accepted.generation

    def _recover(self, dt: float, *, from_displacement: bool = False) -> None:
        layout, state, origin = self._layout, self.state, self._origin
        if self._tet_only:
            self.particle_acceleration.zero_()
            wp.launch(
                _recover_tet_candidate,
                layout.dynamic_particle_count,
                [
                    self.z,
                    origin.particle_q,
                    origin.particle_qd,
                    layout.dynamic_particle_ids,
                    dt,
                    state.particle_q,
                    state.particle_qd,
                    self.particle_acceleration,
                ],
                device=self._model.device,
            )
            self.residual.zero_()
            return
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
        if from_displacement:
            wp.launch(
                _recover_joint_displacement,
                layout.q_dof_count,
                [
                    self.joint_displacement,
                    layout.q_dof_to_joint_q,
                    layout.q_dof_to_joint_qd,
                    origin.joint_q,
                    origin.joint_qd,
                    dt,
                    state.joint_q,
                    state.joint_qd,
                    self.qdd,
                    self.z,
                ],
                device=self._model.device,
            )
        else:
            # Offline absolute-coordinate probes intentionally measure stored-q quantization.
            wp.launch(
                recover_articulation_candidate_rates,
                layout.q_dof_count,
                [
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
            wp.launch(
                _initialize_joint_displacement,
                layout.q_dof_count,
                [state.joint_qd, layout.q_dof_to_joint_qd, dt, self.joint_displacement],
                device=self._model.device,
            )
        if layout.dynamic_particle_count:
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
        if self._model.body_count:
            eval_fk(self._model, state.joint_q, state.joint_qd, state)

    def commit(self, state_out: State) -> None:
        _validate_state(self._model, self.state, tet_only=self._tet_only)
        _validate_state(self._model, state_out, tet_only=self._tet_only)
        state_out.assign(self.state)


class SolverMonolithic(SolverBase):
    """Solve articulated rigid/tet motion with a common implicit position update.

    This experimental implementation supports bilateral P1Q3 contact and
    static fixed particles. Numerical defaults are calibrated on the declared
    component and normal-loading fixtures; other asset ranges are not certified.

    .. experimental::
        This class may change without the normal deprecation period.
    """

    supports_collision_pipeline = True

    Status = Status
    MaterialModel = MaterialModel
    MassMode = MassMode
    PCGMode = PCGMode
    JointTerms = JointTerms
    Stats = Stats

    def __init__(
        self,
        model: Model,
        *,
        collision_pipeline: MonolithicCollisionPipeline,
        contact_stiffness: float,
        joint_terms: SolverMonolithic.JointTerms | None = None,
        material_model: SolverMonolithic.MaterialModel | str = MaterialModel.KIM_STABLE_NO_LOG,
        mass_mode: SolverMonolithic.MassMode | str = MassMode.LUMPED,
        tet_rest_density: wp.array[float] | None = None,
        normal_smoothing_width: float = 0.0,
        friction_coefficient: float = 0.0,
        tangential_stiffness: float | None = None,
        newton_max_iterations: int = 10,
        line_search_max_iterations: int = 8,
        linear_max_iterations: int = 200,
        linear_tolerance: float = 1.0e-4,
        linear_absolute_tolerance: float = 1.0e-6,
        use_optimized_articulation_mass_matrix: bool = False,
        pcg_mode: SolverMonolithic.PCGMode | str = PCGMode.DIAGNOSTIC,
        use_cuda_graph: bool = False,
    ) -> None:
        """Validate the model and allocate solver-owned candidate/contact buffers.

        Args:
            model: Model containing one articulation and one connected tet body.
            collision_pipeline: Fixed P1Q3 pipeline constructed for this model.
            contact_stiffness: Normal penalty stiffness [N/m^3].
            joint_terms: Optional scalar joint physics. None keeps V0.1 behavior.
            material_model: Experimental Kim no-log or Smith log-stabilized material.
            mass_mode: Experimental lumped or consistent P1 mass.
            tet_rest_density: Same-device float32 per-tet density [kg/m^3], required
                for consistent mass and copied at construction; otherwise None.
            normal_smoothing_width: Experimental PolyReLU half-width [m]; zero retains quadratic contact.
            friction_coefficient: Experimental isotropic contact friction coefficient; defaults to zero.
            tangential_stiffness: Experimental tangential penalty [N/m^3]; required for friction.
            newton_max_iterations: Positive nonlinear iteration limit.
            line_search_max_iterations: Positive backtracking iteration limit.
            linear_max_iterations: Positive linear iteration limit.
            linear_tolerance: Positive relative linear residual tolerance.
            linear_absolute_tolerance: Nonnegative absolute scaled linear residual
                norm tolerance, applied to global/q/x blocks. Provisional default
                1e-6 pending calibration; zero restores relative-with-floor acceptance.
                Both tolerance branches are capped at half of each current outer
                Newton residual goal, converted from RMS to L2.
            use_optimized_articulation_mass_matrix: Experimental private parallel
                articulation mass evaluation; fixed at construction, defaults to False.
            pcg_mode: Experimental dynamic diagnostic or fixed masked production PCG policy.
            use_cuda_graph: Capture production PCG and stable device segments on CUDA; default False.
                Production retains mandatory numerical checks and final force publication.
        """
        self._initialize(
            model,
            collision_pipeline=collision_pipeline,
            contact_stiffness=contact_stiffness,
            joint_terms=joint_terms,
            material_model=material_model,
            mass_mode=mass_mode,
            tet_rest_density=tet_rest_density,
            normal_smoothing_width=normal_smoothing_width,
            friction_coefficient=friction_coefficient,
            tangential_stiffness=tangential_stiffness,
            newton_max_iterations=newton_max_iterations,
            line_search_max_iterations=line_search_max_iterations,
            linear_max_iterations=linear_max_iterations,
            linear_tolerance=linear_tolerance,
            linear_absolute_tolerance=linear_absolute_tolerance,
            use_optimized_articulation_mass_matrix=use_optimized_articulation_mass_matrix,
            pcg_mode=pcg_mode,
            use_cuda_graph=use_cuda_graph,
            joint_diagnostic=False,
        )

    @classmethod
    def _create_articulation_only(
        cls, model: Model, *, globalization, joint_terms=None, publish_policy="strict", **options
    ):
        """Create the private FP64 PR2 articulation validation path."""
        from .articulation_solver import _ArticulationSolver  # noqa: PLC0415 -- private subclass imports this base

        return _ArticulationSolver(
            model, globalization=globalization, joint_terms=joint_terms, publish_policy=publish_policy, **options
        )

    @classmethod
    def _create_coupled_variational(cls, model, *, contact_model, globalization, friction_velocity_scale, **options):
        """Create the private strict PR3 joint/particle path with an explicit law."""
        from .coupled_solver import _CoupledSolver  # noqa: PLC0415 -- private subclass imports this base

        return _CoupledSolver(
            model,
            contact_model=contact_model,
            globalization=globalization,
            friction_velocity_scale=friction_velocity_scale,
            **options,
        )

    @classmethod
    def _create_joint_diagnostic(cls, model: Model, *, joint_terms: SolverMonolithic.JointTerms, **options):
        """Create the internal G1H test entry; not a supported rigid-only solver API."""
        solver = cls.__new__(cls)
        solver._initialize(
            model,
            collision_pipeline=None,
            contact_stiffness=1.0,
            joint_terms=joint_terms,
            joint_diagnostic=True,
            **options,
        )
        return solver

    @classmethod
    def _create_tet_only(cls, model: Model, *, globalization, publish_policy="strict", **options):
        """Create the internal P3 validation path with an explicit acceptance policy."""
        if globalization not in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            raise ValueError("Unknown tet globalization mode")
        if publish_policy not in ("strict", "legacy"):
            raise ValueError("Unknown tet publish policy")
        solver = cls.__new__(cls)
        solver._initialize(
            model,
            collision_pipeline=None,
            contact_stiffness=1.0,
            joint_terms=None,
            joint_diagnostic=False,
            tet_only=True,
            **options,
        )
        solver._globalization = globalization
        solver._publish_policy = publish_policy
        return solver

    def _create_tet_state(self) -> State:
        """Create the internal FP64 position/velocity State required by the P3 path.

        The returned arrays are the physical solve and publication state, not
        display copies. A float32 State is rejected rather than silently rounded.
        """
        if not self._tet_only:
            raise ValueError("FP64 tet states belong only to the private P3 path")
        return _new_candidate_state(self.model, True)

    @property
    def tet_config_identity(self) -> str | None:
        """SHA-256 of construction tet configuration; None for the joint diagnostic."""
        return self._tet_physics.identity if self._tet_physics is not None else None

    @property
    def use_optimized_articulation_mass_matrix(self) -> bool:
        """Return the immutable experimental articulation execution policy."""
        return self._use_optimized_articulation_mass_matrix

    @property
    def use_cuda_graph(self) -> bool:
        """Return the immutable experimental PCG and stable-segment capture option."""
        return self._use_cuda_graph

    @property
    def pcg_mode(self) -> SolverMonolithic.PCGMode:
        """Return the immutable experimental PCG observation policy."""
        return self._pcg_mode

    def _initialize(
        self,
        model,
        *,
        collision_pipeline,
        contact_stiffness,
        joint_terms,
        joint_diagnostic,
        tet_only=False,
        material_model=MaterialModel.KIM_STABLE_NO_LOG,
        mass_mode=MassMode.LUMPED,
        tet_rest_density=None,
        normal_smoothing_width=0.0,
        friction_coefficient=0.0,
        tangential_stiffness=None,
        newton_max_iterations=10,
        line_search_max_iterations=8,
        linear_max_iterations=200,
        linear_tolerance=1.0e-4,
        linear_absolute_tolerance=1.0e-6,
        use_optimized_articulation_mass_matrix=False,
        pcg_mode=PCGMode.DIAGNOSTIC,
        use_cuda_graph=False,
    ):
        # Freeze names now; accepting an option must never silently run different physics.
        if (
            isinstance(linear_absolute_tolerance, (bool, np.bool_))
            or not isinstance(linear_absolute_tolerance, (int, float, np.integer, np.floating))
            or not math.isfinite(linear_absolute_tolerance)
            or linear_absolute_tolerance < 0
            or linear_absolute_tolerance > float(np.finfo(np.float32).max)
            or (0 < linear_absolute_tolerance < np.finfo(np.float32).smallest_subnormal)
        ):
            raise ValueError("linear_absolute_tolerance must be nonnegative and representable in float32")
        if type(use_optimized_articulation_mass_matrix) is not bool:
            raise ValueError("use_optimized_articulation_mass_matrix must be a bool")
        self._use_optimized_articulation_mass_matrix = use_optimized_articulation_mass_matrix
        self._pcg_mode = self.PCGMode(pcg_mode)
        if type(use_cuda_graph) is not bool or (use_cuda_graph and self._pcg_mode != self.PCGMode.PRODUCTION):
            raise ValueError("use_cuda_graph requires production PCG and a bool")
        if use_cuda_graph and not model.device.is_cuda:
            raise ValueError("use_cuda_graph requires a CUDA device")
        self._use_cuda_graph = use_cuda_graph
        material_model = self.MaterialModel(material_model)
        mass_mode = self.MassMode(mass_mode)
        for name, value in (
            ("normal_smoothing_width", normal_smoothing_width),
            ("friction_coefficient", friction_coefficient),
            ("tangential_stiffness", tangential_stiffness),
        ):
            if value is None and name == "tangential_stiffness":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value)
                or value < 0
                or value > np.finfo(np.float32).max
            ):
                raise ValueError(f"{name} must be nonnegative finite float32")
        if joint_diagnostic and (
            material_model != self.MaterialModel.KIM_STABLE_NO_LOG
            or mass_mode != self.MassMode.LUMPED
            or tet_rest_density is not None
        ):
            raise ValueError("Tet options do not apply to the q-only diagnostic")
        if mass_mode == self.MassMode.LUMPED and tet_rest_density is not None:
            raise ValueError("tet_rest_density requires consistent mass")
        if normal_smoothing_width != 0 or friction_coefficient != 0 or tangential_stiffness is not None:
            if joint_diagnostic:
                raise ValueError("Contact options do not apply to the q-only diagnostic")
        if friction_coefficient > 0 and (tangential_stiffness is None or tangential_stiffness <= 0):
            raise ValueError("Contact friction requires positive tangential_stiffness")
        if tangential_stiffness is not None and (tangential_stiffness <= 0 or np.float32(tangential_stiffness) <= 0):
            raise ValueError("tangential_stiffness must be positive in float32")
        if friction_coefficient > 0 and np.float32(friction_coefficient) <= 0:
            raise ValueError("friction_coefficient must be positive in float32")
        if normal_smoothing_width > 0 and np.float32(normal_smoothing_width) <= 0:
            raise ValueError("normal_smoothing_width must be positive in float32")
        self._config_generation = 0  # Immutable per solver; rebuild instead of reconfiguring.
        self._history_epoch = 0
        self._joint_diagnostic = joint_diagnostic
        self._tet_only = tet_only
        self._globalization = None
        self._publish_policy = "legacy"
        has_contact = not joint_diagnostic and not tet_only
        if tet_only and (
            joint_terms is not None or normal_smoothing_width or friction_coefficient or tangential_stiffness
        ):
            raise ValueError("Tet-only does not support joint or contact physics")
        if has_contact:
            if not isinstance(collision_pipeline, MonolithicCollisionPipeline):
                raise ValueError("Monolithic requires a MonolithicCollisionPipeline")
            if normal_smoothing_width + collision_pipeline._sdf_query_error > collision_pipeline._soft_contact_gap:
                raise ValueError("Detection gap must cover normal_smoothing_width and SDF query error")
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
        self._layout = _build_layout(model, joint_diagnostic=joint_diagnostic, tet_only=tet_only)
        if not joint_diagnostic:
            validate_tet_scope(
                model,
                dynamic_particle_ids=self._layout.dynamic_particle_ids.numpy(),
                particle_to_dynamic=self._layout.particle_to_dynamic.numpy(),
            )
        self._tet_physics = (
            TetPhysicsWorkspace(
                model, material_model=material_model.value, mass_mode=mass_mode.value, density=tet_rest_density
            )
            if not joint_diagnostic
            else None
        )
        if joint_terms is not None and not isinstance(joint_terms, self.JointTerms):
            raise ValueError("joint_terms must be SolverMonolithic.JointTerms or None")
        self._joint_terms = (
            MonolithicJointTermsWorkspace(
                model,
                implicit_pd=joint_terms.implicit_pd,
                limits=joint_terms.limits,
                friction=joint_terms.friction,
                limit_width=joint_terms.limit_width,
                friction_velocity_scale=joint_terms.friction_velocity_scale,
            )
            if joint_terms is not None
            else None
        )
        self._articulation = (
            None
            if tet_only
            else MonolithicArticulationWorkspace(
                model,
                joint_terms=self._joint_terms,
                use_optimized_articulation_mass_matrix=self.use_optimized_articulation_mass_matrix,
            )
        )
        super().__init__(
            model,
            collision_pipeline=collision_pipeline,
            collision_frequency_type=dict.fromkeys(self.CollisionSlot, self.CollisionFrequencyType.NONE),
        )
        self._trial_contacts = collision_pipeline.contacts() if has_contact else None
        self._transaction = _StepTransaction(model, self._layout, tet_only=tet_only)
        self.contact_stiffness = contact_stiffness
        self.newton_max_iterations = newton_max_iterations
        self.line_search_max_iterations = line_search_max_iterations
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance = linear_tolerance
        self.linear_absolute_tolerance = linear_absolute_tolerance
        self._config = _SolverMonolithicInternalConfig()
        if tet_only:
            # The private FP64 path resolves corrections below the legacy stopping floor.
            # See the fixed-physics study in scripts/monolithic_reference/study_p3_tet_stopping.py.
            self._config = replace(
                self._config,
                merit_absolute_global=1.0e-7,
                merit_absolute_x=1.0e-7,
                merit_relative_global=1.0e-6,
                merit_relative_x=1.0e-6,
                step_tolerance_global=1.0e-10,
                step_tolerance_x=1.0e-10,
            )
        layout = self._layout
        nq, nx, size = layout.q_dof_count, layout.dynamic_particle_count, layout.scalar_dof_count
        pattern = build_tet_triplet_pattern(
            model.tet_indices.numpy() if model.tet_count else np.empty((0, 4), dtype=np.int32),
            layout.particle_to_dynamic.numpy(),
            x_dof_start=nq,
        )
        contact_capacity = (
            collision_pipeline.soft_contact_max * (3 if friction_coefficient > 0 else 1) if has_contact else 0
        )
        # Include triplets, BSR build scratch and factors before requesting device storage.
        self._contact_storage_bound_bytes = contact_capacity * ((nq + 9) ** 2 * 64 + (nq + 32) * 4)
        if (
            friction_coefficient > 0
            and model.device.is_cuda
            and self._contact_storage_bound_bytes > model.device.free_memory
        ):
            raise ValueError("Contact capacity exceeds available device memory")
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
            pcg_mode=self.pcg_mode.value,
            use_cuda_graph=self.use_cuda_graph,
        )
        self._linear._set_fixed_triplet_pattern(
            global_rows=np.concatenate((np.repeat(np.arange(nq), nq), pattern.global_rows)).astype(np.int32),
            global_columns=np.concatenate((np.tile(np.arange(nq), nq), pattern.global_columns)).astype(np.int32),
            internal_rows=pattern.ax_rows,
            internal_columns=pattern.ax_columns,
        )
        self._tet_workspace = (
            create_tet_assembly_workspace(
                pattern, tet_count=model.tet_count, device=model.device, physics=self._tet_physics
            )
            if not joint_diagnostic
            else None
        )
        self._tet_objective = TetObjectiveWorkspace(model, layout, self._tet_physics) if tet_only else None
        self._variational = None
        self._tet_triplet_count = len(pattern.global_rows)
        self._contact = (
            None
            if not has_contact
            else MonolithicContactWorkspace(
                model,
                collision_pipeline,
                self._linear,
                contact_stiffness=contact_stiffness,
                normal_smoothing_width=normal_smoothing_width,
                friction_coefficient=friction_coefficient,
                tangential_stiffness=tangential_stiffness,
                physics_identity=self._tet_physics.identity
                + (
                    hashlib.sha256(self._joint_terms.params.numpy().tobytes()).hexdigest()
                    if self._joint_terms is not None
                    else ""
                ),
            )
        )
        self._final_evaluation_state = model.state()
        self._particle_residual = wp.zeros(nx, dtype=wp.vec3, device=model.device) if not joint_diagnostic else None
        self._dynamic_diagonal = wp.zeros(size, dtype=float, device=model.device)
        self._residual_ref = wp.zeros(size, dtype=float, device=model.device)
        self._metric_vector = wp.zeros(size, dtype=float, device=model.device)
        self._metric_slices = (
            self._metric_vector,
            self._metric_vector[:nq] if nq else wp.empty(0, dtype=float, device=model.device),
            self._metric_vector[nq:] if nx else wp.empty(0, dtype=float, device=model.device),
        )
        self._norm_values = wp.zeros(3, dtype=float, device=model.device)
        self._raw_norm_sums = wp.zeros(2, dtype=wp.float64, device=model.device)
        self._norm_outputs = tuple(self._norm_values[i : i + 1] for i in range(3))
        self._rhs_hat = wp.zeros(size, dtype=float, device=model.device)
        self._y = wp.zeros(size, dtype=float, device=model.device)
        self._delta = wp.zeros(size, dtype=float, device=model.device)
        self._final_residual = wp.zeros(size, dtype=float, device=model.device)
        self._contact_status = wp.zeros(1, dtype=int, device=model.device)
        self._segment_graphs = _SegmentGraphCache(model.device, enabled=use_cuda_graph)
        if self._articulation is not None:
            self._articulation._segment_graphs = self._segment_graphs
        self._linear._segment_graphs = self._segment_graphs
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
        if self._contact is None and contacts is not None:
            raise ValueError("Contact-free solver does not accept contacts")
        self._resolve_step_contacts(contacts)
        _validate_dt(dt)
        _validate_state(self.model, state_in, tet_only=self._tet_only)
        _validate_state(self.model, state_out, tet_only=self._tet_only)
        _validate_dirichlet(self.model, self._layout, state_in, tet_only=self._tet_only)
        if self._tet_physics is not None:
            self._tet_physics.validate_identity()
        pipeline = self.collision_pipeline
        if pipeline is not None:
            pipeline.validate_contacts(self.contacts)
            pipeline.validate_contacts(self._trial_contacts)
            self._contact.validate_contract(self.model, pipeline, self.contacts)
        control = self._default_control if control is None else control
        if self._articulation is not None:
            self._articulation.validate_candidate(
                self.model,
                state_in,
                self._transaction.accepted.qdd,
                control.joint_f,
                state_in.body_f,
            )
        if self._joint_terms is not None:
            self._joint_terms.snapshot_targets(control)
        start = time.perf_counter()
        self._transaction.begin(state_in, state_out, control, dt)
        self._variational = (
            {
                "mode": self._globalization,
                "publish_policy": self._publish_policy,
                "objective_evaluations": 0,
                "residual_evaluations": 0,
                "trials": [],
            }
            if self._tet_only
            else None
        )
        if self._joint_terms is not None:
            self._joint_terms.final_generation = -1
        if self._contact is not None:
            self._contact.invalidate_final_force()
            self._contact.begin_step(self._transaction._original, dt)
        self._assembly_sequence = 0
        self._metrics = {
            "nonlinear_iterations": 0,
            "line_search_iterations": 0,
            "linear_iterations": 0,
            "linear_scheduled_slots": 0,
            "regularization_retries": 0,
            "matrix_assembly_count": 0,
            "true_residual_recomputations": 0,
            "residual_replacements": 0,
            "preconditioner_kind": "actor_block",
            "factor_setup_count": 0,
            "factor_failure_count": 0,
            "q_block_count": int(self._layout.q_dof_count > 0),
            "particle_block_count": self._layout.dynamic_particle_count,
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
            if self._contact is not None:
                self._contact.discard_history()
                self._history_epoch = self._contact.history_epoch
                self._contact.invalidate_final_force()
            if self._joint_terms is not None:
                self._joint_terms.final_generation = -1
            if self._contact is not None and self._contact.friction_coefficient > 0:
                try:
                    self._publish_final(state_out, self.Status.NONFINITE)
                except Exception:
                    self._contact.invalidate_final_force()
                    if self._joint_terms is not None:
                        self._joint_terms.final_generation = -1
            self._transaction._finished = True
            raise

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Idempotently publish cached forces for the last returned state."""
        if self._contact is None:
            raise ValueError("Contact-free solver has no contact publication")
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
        self._segment_graphs.run(
            "norms",
            (scaled,),
            (
                values,
                self._linear.scale,
                self._metric_vector,
                *self._metric_slices,
                self._norm_values,
                *self._norm_outputs,
            ),
            lambda: self._reduce_norms(values, scaled=scaled),
        )
        norms = self._norm_values.numpy()
        if not np.isfinite(norms).all() or np.any(norms < 0):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite scaled reduction")
        return tuple(
            math.sqrt(float(value) / max(vector.size, 1))
            for value, vector in zip(norms, self._metric_slices, strict=True)
        )

    def _reduce_norms(self, values, *, scaled):
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

    def _converged(self, merit: tuple[float, float, float], reference: tuple[float, float, float]) -> bool:
        config = self._config
        return all(
            value <= getattr(config, f"merit_absolute_{block}") + getattr(config, f"merit_relative_{block}") * ref
            for block, value, ref in zip(("global", "q", "x"), merit, reference, strict=True)
            if not (self._tet_only and block == "q")
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
        if self._articulation is None:
            return
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

        if self._joint_terms is not None:
            self._evaluate_joint_terms(candidate.state, candidate._dt)
            wp.launch(
                add_joint_term_residual, nq, [self._joint_terms.residual, candidate.residual], device=model.device
            )

    def _evaluate_joint_terms(self, state: State, dt: float) -> None:
        self._joint_terms.evaluate(state, dt)
        if int(self._joint_terms.status.numpy()[0]):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite joint term evaluation")

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
            "frozen_gravity": self._transaction.inputs.gravity if self._tet_only else None,
        }

    def _evaluate_objective(self, candidate: _Candidate, dt: float):
        if self._tet_objective is None or self._transaction._finished or candidate._dt != dt:
            raise ValueError("Objective requires a live tet-only step and its frozen dt")
        if candidate is not self._transaction.accepted and (
            candidate is not self._transaction.trial
            or candidate._trial_source is not self._transaction.accepted
            or candidate._trial_source_generation != self._transaction.accepted.generation
        ):
            raise ValueError("Objective candidate generation is stale")
        status, parts, absolute_sum = self._tet_objective.evaluate(
            candidate,
            self._transaction._original,
            self._transaction.inputs,
            dt,
            self._config.det_f_guard,
            self._segment_graphs,
        )
        if status:
            raise _StepFailure(
                self.Status.TET_INVERSION if status == 1 else self.Status.NONFINITE, "Invalid tet objective"
            )
        if self._variational is not None:
            self._variational["objective_evaluations"] += 1
        return parts, absolute_sum

    def _evaluation_result(self, candidate, dt, min_det, penetration):
        parts, absolute_sum = (None, 0.0)
        if self._tet_only:
            parts, absolute_sum = self._evaluate_objective(candidate, dt)
            if self._variational is not None:
                self._variational["residual_evaluations"] += 1
        return _TrialResult(0, *self._rms(candidate.residual), min_det, penetration, parts, absolute_sum)

    def _check_tet(self) -> float:
        code = TetEvaluationStatus(int(self._tet_workspace.failure_flags.numpy()[0]))
        if code == TetEvaluationStatus.DET_F_GUARD:
            raise _StepFailure(self.Status.TET_INVERSION, "Tet determinant guard")
        if code == TetEvaluationStatus.NONFINITE:
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite tet evaluation")
        if code == TetEvaluationStatus.PSD_PROJECTION_FAILURE:
            raise _StepFailure(self.Status.TET_PROJECTION_FAILURE, "Element spectral projection did not converge")
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
        if self._contact is not None:
            self._collide(candidate.state, self._trial_contacts)
        self._assembly_sequence += 1
        generation = MonolithicLinearGeneration(
            self._transaction.step_generation,
            self._metrics["nonlinear_iterations"],
            int(self._trial_contacts.contact_generation.numpy()[0]) if self._contact is not None else 0,
            self._assembly_sequence,
            config_generation=self._config_generation,
            history_epoch=self._history_epoch,
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
                self._articulation.M if self._articulation is not None else None,
                assembly.aq_actor_dense,
                triplets.rows,
                triplets.columns,
                triplets.values,
            ],
            device=model.device,
        )
        if self._joint_terms is not None:
            wp.launch(
                scatter_joint_term_tangent,
                nq,
                [self._joint_terms.tangent, assembly.aq_actor_dense, triplets.values, nq],
                device=model.device,
            )
        min_det = math.nan
        if not self._joint_diagnostic:
            assemble_tet_residual_tangent(
                model,
                **self._tet_arguments(candidate, dt),
                scatter=TetScatterBuffers(
                    assembly.ax_internal_triplets.values,
                    triplets.values[nq * nq : nq * nq + self._tet_triplet_count]
                    if self._tet_triplet_count
                    else self._metric_slices[2],
                ),
            )
            min_det = self._check_tet()
            wp.launch(
                _copy_particle_residual,
                self._layout.dynamic_particle_count,
                [self._particle_residual, nq, candidate.residual],
                device=model.device,
            )
        if self._contact is not None:
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
                self._articulation.M if self._articulation is not None else None,
                self._tet_physics.dynamic_mass_diagonal if self._tet_physics is not None else model.particle_mass,
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
        return self._evaluation_result(
            candidate,
            dt,
            min_det,
            self._contact._diagnostics(0)["max_penetration"] if self._contact is not None else math.nan,
        )

    def _evaluate_trial(self, candidate: _Candidate, dt: float) -> _TrialResult:
        self._actor_residual(candidate)
        if self._joint_diagnostic:
            return _TrialResult(0, *self._rms(candidate.residual), math.nan, math.nan)
        if self._contact is not None:
            self._collide(candidate.state, self._trial_contacts)
        args = self._tet_arguments(candidate, dt)
        if self._segment_graphs.enabled and self._tet_workspace.physics is not None:
            _validate_evaluation(self.model, args)
            physics, workspace = self._tet_workspace.physics, self._tet_workspace
            buffers = [value for value in args.values() if isinstance(value, wp.array)]
            buffers += [
                physics.indices,
                self.model.gravity,
                physics.worlds,
                physics.poses,
                physics.materials,
                physics.mass,
                physics.row_mass,
                physics.coefficients,
                workspace.elastic_block_slots,
                workspace.tet_energy,
                workspace.min_det_f,
                workspace.failure_flags,
            ]
            self._segment_graphs.run(
                "tet_residual",
                (dt, self._config.det_f_guard, physics.material_model, physics.mass_mode),
                buffers,
                lambda: evaluate_tet_residual(self.model, **args),
            )
        else:
            evaluate_tet_residual(self.model, **args)
        min_det = self._check_tet()
        wp.launch(
            _copy_particle_residual,
            self._layout.dynamic_particle_count,
            [self._particle_residual, self._layout.q_dof_count, candidate.residual],
            device=self.model.device,
        )
        if self._contact is None:
            return self._evaluation_result(candidate, dt, min_det, math.nan)
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
            if self._tet_only:
                self._variational.update(
                    objective=current.objective,
                    objective_total=sum(current.objective),
                    iterate_generation=transaction.accepted.generation,
                )
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
                linear_absolute_tolerance=self.linear_absolute_tolerance,
                # Leave half the outer residual budget for nonlinear error. Convert
                # each block's RMS goal to L2 in the current scaling, including q-only.
                residual_norm_limits=tuple(
                    0.5
                    * math.sqrt(max(vector.size, 1))
                    * (getattr(config, f"merit_absolute_{block}") + getattr(config, f"merit_relative_{block}") * ref)
                    for block, ref, vector in zip(("global", "q", "x"), reference, self._metric_slices, strict=True)
                ),
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
            last_error = None
            evaluated_trial = False
            for retry, lambda_value in enumerate(config.regularization_values):
                self._metrics["regularization_retries"] += int(retry > 0)
                self._metrics["lambda_value"] = lambda_value
                try:
                    self._require_linear(self._linear.set_regularization(lambda_value, generation=self._generation))
                    setups, failures = self._linear.factor_setup_count, self._linear.factor_failure_count
                    factor_status = self._linear.factor_actor_preconditioner(
                        generation=self._generation, pivot_tolerance=0.0
                    )
                    self._metrics["factor_setup_count"] += self._linear.factor_setup_count - setups
                    self._metrics["factor_failure_count"] += self._linear.factor_failure_count - failures
                    self._require_linear(factor_status)
                    result = self._linear.solve_pcg(
                        self._rhs_hat,
                        self._y,
                        generation=self._generation,
                        warm_start=MonolithicPcgWarmStart.ZERO,
                        config=pcg_config,
                        delta=self._delta,
                    )
                    self._metrics["linear_iterations"] += result.iterations
                    self._metrics["linear_scheduled_slots"] += result.scheduled_slots
                    self._metrics.update(
                        pcg_policy=result.pcg_policy,
                        linear_converged_at_slot=result.converged_at_slot,
                        linear_last_checked_slot=result.last_checked_slot,
                        linear_failure_slot=result.failure_slot,
                        linear_output_valid=result.output_valid,
                    )
                    self._metrics["true_residual_recomputations"] += result.true_residual_checks
                    self._metrics["residual_replacements"] += result.residual_replacements
                    self._metrics.update(
                        rho=result.rho,
                        rho_q=result.rho_q,
                        rho_x=result.rho_x,
                        min_p_ap=result.min_p_ap,
                        min_r_z=result.min_r_z,
                        warm_start=result.warm_start.name.lower() if result.warm_start is not None else None,
                        initial_guess_norm=result.initial_guess_norm,
                        recursive_true_residual_gap=result.recursive_true_residual_gap[0],
                        recursive_true_residual_gap_q=result.recursive_true_residual_gap[1],
                        recursive_true_residual_gap_x=result.recursive_true_residual_gap[2],
                        stagnation_window=result.stagnation_window,
                        linear_scale_generation=result.generation.assembly_sequence,
                    )
                    self._require_linear(result.status)
                    if self.pcg_mode == self.PCGMode.PRODUCTION:
                        if not result.output_valid:
                            raise _StepFailure(self.Status.NONFINITE, "PCG returned an invalid direction")
                    else:
                        self._require_linear(
                            self._linear.recover_delta(self._y, self._delta, generation=self._generation)
                        )
                except _StepFailure as error:
                    last_reason = str(error)
                    last_error = error
                    continue
                step_rms = self._rms(self._y, scaled=False)
                slope, slope_abs = (math.nan, 0.0)
                if self._tet_only:
                    slope, slope_abs = self._tet_objective.slope(transaction.accepted.residual, self._delta)
                    if not math.isfinite(slope):
                        raise _StepFailure(self.Status.NONFINITE, "Nonfinite objective slope")
                for search in range(self.line_search_max_iterations):
                    alpha = 0.5**search
                    self._metrics["line_search_iterations"] += 1
                    transaction.trial.form_trial(transaction.accepted, self._delta, alpha, dt)
                    try:
                        trial = self._evaluate_trial(transaction.trial, dt)
                    except _StepFailure as error:
                        last_reason = str(error)
                        last_error = error
                        continue
                    evaluated_trial = True
                    passes = trial.merit <= (1.0 - 1.0e-4 * alpha) * merit[0] + config.merit_noise
                    if self._tet_only:
                        # FP64 reduction bound is distinct from the calibrated FP32 residual noise.
                        energy_noise = (
                            32 * np.finfo(np.float64).eps * (current.objective_abs_sum + trial.objective_abs_sum)
                        )
                        # The dot uses FP64, but its residual input is assembled in FP32.
                        slope_noise = 32 * np.finfo(np.float32).eps * slope_abs
                        residual_noise = max(
                            config.merit_noise, 32 * np.finfo(np.float32).eps * (merit[0] + trial.merit)
                        )
                        passes, a, r, improved = acceptance(
                            self._globalization,
                            sum(current.objective),
                            sum(trial.objective),
                            slope,
                            alpha,
                            merit[0],
                            trial.merit,
                            energy_noise=energy_noise,
                            residual_noise=residual_noise,
                            slope_noise=slope_noise,
                        )
                        self._variational["trials"].append(
                            {
                                "iteration": iteration,
                                "retry": retry,
                                "alpha": alpha,
                                "slope": slope,
                                "current_objective": current.objective,
                                "trial_objective": trial.objective,
                                "current_merit": merit[0],
                                "trial_merit": trial.merit,
                                "A": a,
                                "R": r,
                                "accepted": passes,
                                "improved": improved,
                                "accepted_by": tuple(
                                    name
                                    for name, enabled in (
                                        ("Armijo", a and self._globalization != "ResNorm"),
                                        ("ResNorm", r and self._globalization != "Armijo"),
                                    )
                                    if passes and enabled
                                ),
                                "energy_noise": energy_noise,
                                "residual_noise": residual_noise,
                                "slope_noise": slope_noise,
                            }
                        )
                    if passes:
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
                if self._tet_only and not evaluated_trial and last_error is not None:
                    raise last_error
                raise _StepFailure(self.Status.REGULARIZATION_EXHAUSTED, last_reason)
        raise AssertionError("Unreachable nonlinear loop exit")

    def _publish_final(self, state: State, status: SolverMonolithic.Status) -> None:
        if self._tet_only:
            _validate_state(self.model, state, tet_only=True)
            return
        if self._joint_terms is not None:
            self._joint_terms.final_generation = -1
        roles = {
            self.Status.SUCCESS: _MonolithicReturnedStateRole.STATE_OUT_CONVERGED,
            self.Status.NONLINEAR_MAX_ITERATIONS: _MonolithicReturnedStateRole.STATE_OUT_SOFT_STOP,
        }
        role = roles.get(status, _MonolithicReturnedStateRole.STATE_IN_ROLLBACK)
        # FK refresh must not overwrite the original body caches on rollback.
        self._final_evaluation_state.assign(state)
        eval_articulation_passive_candidate(self.model, self._final_evaluation_state, self._articulation)
        if not self._joint_diagnostic:
            self._collide(state, self.contacts)
            self._final_residual.zero_()
            self._contact_status.zero_()
            evaluate_final_contacts(
                self.model,
                self._final_evaluation_state,
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
            self._contact.bind_returned_state(state)
            if self.contacts.force is not None:
                self._contact._publish_final_forces(
                    state, self.contacts, step_generation=self._transaction.step_generation, returned_state_role=role
                )

        if self._joint_terms is not None:
            self._evaluate_joint_terms(state, self._transaction.accepted._dt)
            self._joint_terms.publish(self._transaction.step_generation)

    def _finish_step(self, state_out: State, status: SolverMonolithic.Status, reason: str | None, start: float) -> None:
        transaction = self._transaction
        commit = status == self.Status.SUCCESS or (
            self._publish_policy == "legacy" and status == self.Status.NONLINEAR_MAX_ITERATIONS
        )
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
            if self._contact is not None:
                self._contact.discard_history()
            try:
                self._publish_final(state_out, status)
            except _StepFailure as second_error:
                publication_error = second_error
                if self._contact is not None:
                    self._contact.invalidate_final_force()
        diagnostics = self._contact._diagnostics(2) if publication_error is None and self._contact is not None else {}
        joint_diagnostics = {}
        terms = self._joint_terms
        if publication_error is None and terms is not None and terms.final_generation == transaction.step_generation:
            # One final-state readback, never in the candidate kernel loop.
            forces, power = terms.final_force.numpy(), terms.final_power.numpy()
            joint_diagnostics = {
                "joint_force_generation": terms.final_generation,
                "joint_pd_force": tuple(float(x) for x in forces[:, 0]),
                "joint_limit_force": tuple(float(x) for x in forces[:, 1]),
                "joint_friction_force": tuple(float(x) for x in forces[:, 2]),
                "joint_pd_saturated": tuple(bool(x) for x in terms.final_saturated.numpy()),
                "joint_limit_damping_power": tuple(float(x) for x in power[:, 1]),
                "joint_friction_power": tuple(float(x) for x in power[:, 2]),
            }
        if self._joint_diagnostic:
            # Empty blocks are not measured or certified; report N/A as NaN.
            for key in tuple(self._metrics):
                if key.startswith(("merit_x", "raw_residual_x")) or key.endswith("_x"):
                    self._metrics[key] = math.nan
        if self._tet_only:
            for key in tuple(self._metrics):
                if key.startswith(("merit_q", "raw_residual_q")) or key.endswith("_q"):
                    self._metrics[key] = math.nan
            self._variational.update(
                tangent_evaluations=self._metrics["matrix_assembly_count"],
                returned_state="accepted" if commit else "step_input",
                published=commit,
                termination={
                    self.Status.SUCCESS: "CONVERGED",
                    self.Status.NONLINEAR_MAX_ITERATIONS: "STOPPED_MAX_ITERS",
                    self.Status.NONLINEAR_STAGNATION: "STOPPED_SMALL_STEP",
                    self.Status.REGULARIZATION_EXHAUSTED: "GLOBALIZATION_FAILED",
                    self.Status.LINE_SEARCH_EXHAUSTED: "GLOBALIZATION_FAILED",
                    self.Status.NONFINITE: "DIVERGED",
                    self.Status.TET_INVERSION: "DIVERGED",
                    self.Status.TET_PROJECTION_FAILURE: "DIVERGED",
                }.get(status, "LINEAR_FAILED"),
            )
        transaction.last_stats = self.Stats(
            status=status,
            material_model=self._tet_physics.material_model if self._tet_physics is not None else None,
            mass_mode=self._tet_physics.mass_mode if self._tet_physics is not None else None,
            tet_config_identity=self._tet_physics.identity if self._tet_physics is not None else None,
            failure_reason=reason,
            converged=status == self.Status.SUCCESS,
            variational=MappingProxyType(self._variational) if self._variational is not None else None,
            rolled_back=not commit,
            step_generation=transaction.step_generation,
            accepted_generation=transaction.accepted.generation,
            soft_contact_pair_count=self.collision_pipeline.soft_contact_pair_count if self._contact is not None else 0,
            published_contact_generation=int(self.contacts.contact_generation.numpy()[0])
            if publication_error is None and self._contact is not None
            else -1,
            active_sample_count=diagnostics.get("active_sample_count", 0),
            max_penetration=diagnostics.get("max_penetration", math.nan),
            contact_force_imbalance=diagnostics.get("force_imbalance", math.nan),
            contact_moment_imbalance=diagnostics.get("moment_imbalance", math.nan),
            generalized_projection_error=diagnostics.get("generalized_projection_error", math.nan),
            contact_sign_error=diagnostics.get("contact_sign_error", math.nan),
            contact_history=MappingProxyType(
                {
                    **self._contact.history_diagnostics(),
                    "history_epoch": self._history_epoch + int(commit and self._contact.friction_coefficient > 0),
                    "history_commit_count": int(commit and self._contact.friction_coefficient > 0),
                    "history_rollback_count": int(not commit and self._contact.friction_coefficient > 0),
                }
            )
            if self._contact is not None and publication_error is None
            else None,
            timings={"step": 1000.0 * (time.perf_counter() - start)},
            **self._metrics,
            **joint_diagnostics,
        )
        if self._contact is not None:
            if commit:
                self._contact.commit_history()
            else:
                self._contact.discard_history()
            self._history_epoch = self._contact.history_epoch
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

    def __init__(self, model: Model, layout: _MonolithicLayout, *, tet_only: bool = False):
        self._model, self._layout = model, layout
        self._tet_only = tet_only
        self.accepted = _Candidate(model, layout, tet_only=tet_only)
        self.trial = _Candidate(model, layout, tet_only=tet_only)
        self.inputs = _FrozenStepInputs(model)
        self._original = _new_candidate_state(model, tet_only)
        self._state_out = None
        self._finished = True
        self.step_generation = 0
        self.last_stats = SolverMonolithic.Stats(SolverMonolithic.Status.NOT_RUN)

    def begin(self, state_in: State, state_out: State, control: Control | None, dt: float) -> None:
        if not self._finished:
            raise RuntimeError("The previous transaction has not finished")
        _validate_dt(dt)
        _validate_state(self._model, state_in, tet_only=self._tet_only)
        _validate_state(self._model, state_out, tet_only=self._tet_only)
        _validate_dirichlet(self._model, self._layout, state_in, tet_only=self._tet_only)
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
        _validate_state(self._model, self._state_out, tet_only=self._tet_only)
        if commit:
            self.accepted.commit(self._state_out)
        else:
            _validate_state(self._model, self._original, tet_only=self._tet_only)
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
