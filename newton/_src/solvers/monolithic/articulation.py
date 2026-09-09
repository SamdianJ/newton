# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned storage and thin orchestration of Newton articulation dynamics."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import JointTargetMode, JointType, Model, State, eval_fk, eval_jacobian, eval_mass_matrix
from ...sim.articulation import compute_body_spatial_inertia
from ...sim.inverse_dynamics import _compute_coriolis_force, _compute_gravity_force, _InverseDynamicsScratchBuffer


@wp.kernel
def _eval_owned_mass_matrix(
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_child: wp.array[int],
    joint_qd_start: wp.array[int],
    body_I_s: wp.array[wp.spatial_matrix],
    J: wp.array3d[float],
    M: wp.array3d[float],
):
    dof_i, dof_j = wp.tid()
    joint_start = articulation_start[0]
    joint_end = articulation_end[0]
    dof_count = joint_qd_start[joint_end] - joint_qd_start[joint_start]
    value = float(0.0)
    if dof_i < dof_count and dof_j < dof_count:
        for link_idx in range(joint_end - joint_start):
            I_s = body_I_s[joint_child[joint_start + link_idx]]
            row_start = link_idx * 6
            # Retain the reference's per-link partial sums and k/l ordering.
            sum_val = float(0.0)
            for k in range(6):
                for l in range(6):
                    sum_val += J[0, row_start + k, dof_i] * I_s[k, l] * J[0, row_start + l, dof_j]
            value += sum_val
    M[0, dof_i, dof_j] = value


def _validate_joint_coordinate_layout(model: Model) -> np.ndarray:
    """Validate the complete supported joint prefix layout and return the DOF-to-coordinate map."""
    types = model.joint_type.numpy()
    if not np.isin(types, [JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC]).all():
        raise ValueError("Monolithic supports only FIXED, REVOLUTE and PRISMATIC joints")
    moving = types != JointType.FIXED
    expected = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(moving, dtype=np.int32)))
    if expected[-1] != model.joint_coord_count or expected[-1] != model.joint_dof_count:
        raise ValueError("Invalid revolute/prismatic joint coordinate counts")
    for name in ("joint_q_start", "joint_qd_start"):
        starts = getattr(model, name)
        if starts.dtype != wp.int32 or not np.array_equal(starts.numpy(), expected):
            raise ValueError(f"Invalid joint coordinate prefix layout: {name}")
    dimensions = np.stack((types == JointType.PRISMATIC, types == JointType.REVOLUTE), axis=1)
    if not np.array_equal(model.joint_dof_dim.numpy(), dimensions):
        raise ValueError("Invalid joint_dof_dim for FIXED/REVOLUTE/PRISMATIC joints")
    return expected[:-1][moving]


class MonolithicArticulationWorkspace:
    """Reuse passive dynamics scratch on one model and one execution stream.

    Model array identities and layouts are fixed for this workspace's lifetime.
    Rebuild the workspace after replacing topology or model storage. Updating
    candidate coordinates and velocities in their existing arrays is supported.
    """

    _array_names = (
        "articulation_start",
        "articulation_end",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_ancestor",
        "joint_articulation",
        "joint_q_start",
        "joint_qd_start",
        "joint_target_q_start",
        "joint_axis",
        "joint_dof_dim",
        "joint_X_p",
        "joint_X_c",
        "joint_enabled",
        "joint_target_q",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_armature",
        "joint_damping",
        "joint_friction",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_target_ke",
        "joint_target_kd",
        "joint_target_mode",
        "body_com",
        "body_mass",
        "body_inertia",
        "body_flags",
        "body_world",
        "gravity",
        "_fk_articulation_level_start",
        "_fk_level_joint_start",
        "_fk_level_joints",
        "_fk_level_parent_pos",
    )
    _count_names = (
        "body_count",
        "joint_count",
        "articulation_count",
        "joint_coord_count",
        "joint_dof_count",
        "max_dofs_per_articulation",
        "max_joints_per_articulation",
        "world_count",
        "constraint_mimic_count",
        "_fk_level_capacity",
        "_has_rod_joints",
    )

    def __init__(
        self,
        model: Model,
        *,
        stream: wp.Stream | None = None,
        joint_terms: MonolithicJointTermsWorkspace | None = None,
        use_optimized_articulation_mass_matrix: bool = False,
    ) -> None:
        if type(use_optimized_articulation_mass_matrix) is not bool:
            raise TypeError("use_optimized_articulation_mass_matrix must be a bool")
        self._use_optimized_articulation_mass_matrix = use_optimized_articulation_mass_matrix
        if model.articulation_count != 1:
            raise ValueError("Monolithic requires one articulation")
        joint_count, dof_count, body_count = model.joint_count, model.joint_dof_count, model.body_count
        for names, shape, dtype in (
            (("articulation_start",), (2,), wp.int32),
            (("articulation_end",), (1,), wp.int32),
            (
                ("joint_type", "joint_parent", "joint_child", "joint_ancestor", "joint_articulation"),
                (joint_count,),
                wp.int32,
            ),
            (("joint_q_start", "joint_qd_start", "joint_target_q_start"), (joint_count + 1,), wp.int32),
            (("joint_X_p", "joint_X_c"), (joint_count,), wp.transform),
            (("joint_enabled",), (joint_count,), wp.bool),
            (("joint_dof_dim",), (joint_count, 2), wp.int32),
            (("joint_axis",), (dof_count,), wp.vec3),
            (("joint_target_q",), (model.joint_coord_count,), wp.float32),
            (("joint_target_mode",), (dof_count,), wp.int32),
            (
                (
                    "joint_limit_lower",
                    "joint_limit_upper",
                    "joint_armature",
                    "joint_damping",
                    "joint_friction",
                    "joint_limit_ke",
                    "joint_limit_kd",
                    "joint_target_ke",
                    "joint_target_kd",
                ),
                (dof_count,),
                wp.float32,
            ),
            (("body_mass",), (body_count,), wp.float32),
            (("body_inertia",), (body_count,), wp.mat33),
            (("body_com",), (body_count,), wp.vec3),
            (("body_flags", "body_world"), (body_count,), wp.int32),
            (
                ("gravity",),
                (1,) if model.world_count == 1 and model.gravity.shape == (1,) else (model.world_count + 1,),
                wp.vec3,
            ),
        ):
            for name in names:
                array = getattr(model, name)
                if array is None or (array.shape, array.dtype, array.device) != (shape, dtype, model.device):
                    raise ValueError(f"Invalid articulation model array shape, dtype or device: {name}")
        for name in self._array_names:
            array = getattr(model, name)
            if array is not None and array.device != model.device:
                raise ValueError(f"Invalid articulation model array device: {name}")
            if name.startswith("_fk_") and array is not None and (array.ndim != 1 or array.dtype != wp.int32):
                raise ValueError(f"Invalid articulation FK cache shape or dtype: {name}")
        if model.constraint_mimic_count:
            raise ValueError("Monolithic does not support mimic constraints")
        if model.actuators:
            raise ValueError("Monolithic does not support model actuators")
        allowed = joint_terms.allowed_properties if joint_terms is not None else frozenset()
        if joint_terms is not None and joint_terms.model is not model:
            raise ValueError("Joint terms workspace belongs to another model")
        for name in (
            "joint_armature",
            "joint_damping",
            "joint_friction",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
        ):
            if name not in allowed and np.any(getattr(model, name).numpy() != 0.0):
                raise ValueError(f"Monolithic requires zero {name}")
        mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
        if not np.isfinite(mass).all() or np.any(mass < 0.0):
            raise ValueError("Articulation body mass must be finite and nonnegative")
        if (
            not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.transpose(0, 2, 1), rtol=1e-6, atol=0.0)
            or np.any(np.linalg.eigvalsh(inertia.astype(np.float64)) < 0.0)
        ):
            raise ValueError("Articulation body inertia must be finite, symmetric and positive semidefinite")
        coord = _validate_joint_coordinate_layout(model)
        starts, ends = model.articulation_start.numpy(), model.articulation_end.numpy()
        if starts.tolist() != [0, model.joint_count] or ends.tolist() != [model.joint_count]:
            raise ValueError("Monolithic requires a tree without unowned joints or loop closures")
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        body_to_link = np.full(model.body_count, -1, dtype=np.int32)
        seen = set()
        roots = 0
        for link, (parent, child) in enumerate(zip(parents, children, strict=True)):
            if child < 0 or child >= model.body_count or child in seen or (parent != -1 and parent not in seen):
                raise ValueError("Monolithic requires a world-anchored joint tree")
            seen.add(int(child))
            roots += int(parent == -1)
            body_to_link[child] = link
        if roots != 1 or len(seen) != model.body_count:
            raise ValueError("Monolithic requires one world-anchored tree owning every body")
        if not model.joint_enabled.numpy().all():
            raise ValueError("Monolithic requires enabled joints")
        if stream is not None and stream.device != model.device:
            raise ValueError("Workspace stream must belong to the model device")
        self.model = model
        self.device = model.device
        self.stream = stream if stream is not None else (wp.get_stream(model.device) if model.device.is_cuda else None)
        self._arrays = {name: getattr(model, name) for name in self._array_names}
        self._array_layouts = {
            name: (value.shape, value.dtype, value.device) for name, value in self._arrays.items() if value is not None
        }
        self._counts = {name: getattr(model, name) for name in self._count_names}
        with wp.ScopedStream(self.stream):
            self.scratch = _InverseDynamicsScratchBuffer(
                model.body_count,
                model.articulation_count,
                model.joint_dof_count,
                model.joint_target_q.shape[0],
                model.max_dofs_per_articulation,
                model.max_joints_per_articulation,
                model.world_count,
                device=model.device,
            )
            self.M = wp.empty(
                (1, model.max_dofs_per_articulation, model.max_dofs_per_articulation), dtype=float, device=self.device
            )
            self.g = wp.empty(model.joint_dof_count, dtype=float, device=self.device)
            self.C = wp.empty_like(self.g)
            self.generalized_body_force = wp.empty_like(self.g)
            self.dof_to_coord = wp.array(coord, dtype=int, device=self.device)
            self.body_to_articulation = wp.zeros(model.body_count, dtype=int, device=self.device)
            self.body_to_link_index = wp.array(body_to_link, dtype=int, device=self.device)
            # Warm the public FK dispatch on independent state, not model initial arrays.
            eval_articulation_passive_candidate(model, model.state(), self)
            mass_matrix = self.M.numpy()[0].astype(np.float64)
            if (
                not np.isfinite(mass_matrix).all()
                or not np.allclose(mass_matrix, mass_matrix.T, rtol=1e-5, atol=1e-8)
                or (mass_matrix.size and np.linalg.eigvalsh(mass_matrix).min() <= 0.0)
            ):
                raise ValueError("Articulation inertia must be positive definite on every free DOF")

    @property
    def use_optimized_articulation_mass_matrix(self) -> bool:
        """Return the mass dispatch fixed at workspace construction."""
        return self._use_optimized_articulation_mass_matrix

    def _validate_state(self, model: Model, state: State) -> None:
        if model is not self.model or model.device != self.device or model.actuators:
            raise ValueError("Articulation workspace is stale")
        for name, value in self._arrays.items():
            current = getattr(model, name)
            if current is not value or (
                current is not None and (current.shape, current.dtype, current.device) != self._array_layouts[name]
            ):
                raise ValueError(f"Articulation workspace is stale: {name}")
        for name, value in self._counts.items():
            if getattr(model, name) != value:
                raise ValueError(f"Articulation workspace is stale: {name}")
        for name, count, dtype in (
            ("joint_q", model.joint_coord_count, wp.float32),
            ("joint_qd", model.joint_dof_count, wp.float32),
            ("body_q", model.body_count, wp.transform),
            ("body_qd", model.body_count, wp.spatial_vector),
        ):
            self._validate_array(getattr(state, name), name, count, dtype)

    def _validate_array(self, value: wp.array, name: str, count: int, dtype: type) -> None:
        if value is None or value.shape != (count,) or value.dtype != dtype or value.device != self.device:
            raise ValueError(f"Invalid articulation array shape, dtype or device: {name}")

    def validate_candidate(
        self,
        model: Model,
        state: State,
        joint_qdd: wp.array[float],
        frozen_joint_f: wp.array[float],
        frozen_body_f: wp.array[wp.spatial_vector],
    ) -> None:
        """Reject incompatible candidate and frozen-force storage before any launch."""
        self._validate_state(model, state)
        self._validate_array(joint_qdd, "joint_qdd", model.joint_dof_count, wp.float32)
        self._validate_array(frozen_joint_f, "frozen_joint_f", model.joint_dof_count, wp.float32)
        self._validate_array(frozen_body_f, "frozen_body_f", model.body_count, wp.spatial_vector)


def _eval_passive_dynamics(model: Model, state: State, workspace: MonolithicArticulationWorkspace) -> None:
    scratch = workspace.scratch
    eval_fk(model, state.joint_q, state.joint_qd, state)
    eval_jacobian(model, state, J=scratch.J, joint_S_s=scratch.joint_S_s)
    if workspace.use_optimized_articulation_mass_matrix:
        wp.launch(
            compute_body_spatial_inertia,
            model.body_count,
            inputs=[model.body_inertia, model.body_mass, state.body_q, scratch.body_I_s],
            device=model.device,
        )
        wp.launch(
            _eval_owned_mass_matrix,
            (workspace.M.shape[1], workspace.M.shape[2]),
            inputs=[
                model.articulation_start,
                model.articulation_end,
                model.joint_child,
                model.joint_qd_start,
                scratch.body_I_s,
                scratch.J,
                workspace.M,
            ],
            device=model.device,
        )
    else:
        eval_mass_matrix(
            model, state, H=workspace.M, J=scratch.J, body_I_s=scratch.body_I_s, joint_S_s=scratch.joint_S_s
        )
    _compute_gravity_force(model, state, workspace.g, scratch)
    _compute_coriolis_force(model, state, workspace.C, scratch)


def eval_articulation_passive_candidate(
    model: Model,
    candidate_state: State,
    workspace: MonolithicArticulationWorkspace,
) -> None:
    """Overwrite FK, Jacobian and passive dynamics using preallocated storage."""
    workspace._validate_state(model, candidate_state)
    with wp.ScopedStream(workspace.stream):
        _eval_passive_dynamics(model, candidate_state, workspace)


@wp.kernel
def recover_articulation_candidate_rates(
    dof_to_coord: wp.array[int],
    articulation_dof_start: int,
    dof_count: int,
    inv_dt: float,
    joint_q: wp.array[float],
    joint_q_n: wp.array[float],
    joint_qd_n: wp.array[float],
    out_joint_qd: wp.array[float],
    out_joint_qdd: wp.array[float],
):
    i = wp.tid()
    if i < dof_count:
        d = articulation_dof_start + i
        q = dof_to_coord[i]
        velocity = (joint_q[q] - joint_q_n[q]) * inv_dt
        out_joint_qd[d] = velocity
        out_joint_qdd[d] = (velocity - joint_qd_n[d]) * inv_dt


@wp.func
def articulation_point_jacobian_column(
    spatial_jacobian: wp.array3d[float],
    articulation_index: int,
    link_index: int,
    dof_index: int,
    body_q: wp.transform,
    body_com: wp.vec3,
    point_world: wp.vec3,
) -> wp.vec3:
    row = 6 * link_index
    v = wp.vec3(
        spatial_jacobian[articulation_index, row, dof_index],
        spatial_jacobian[articulation_index, row + 1, dof_index],
        spatial_jacobian[articulation_index, row + 2, dof_index],
    )
    w = wp.vec3(
        spatial_jacobian[articulation_index, row + 3, dof_index],
        spatial_jacobian[articulation_index, row + 4, dof_index],
        spatial_jacobian[articulation_index, row + 5, dof_index],
    )
    return v + wp.cross(w, point_world - wp.transform_point(body_q, body_com))


@wp.kernel
def project_articulation_body_wrenches(
    articulation_index: int,
    dof_count: int,
    articulation_start: wp.array[int],
    articulation_end: wp.array[int],
    joint_child: wp.array[int],
    spatial_jacobian: wp.array3d[float],
    body_f_world_com: wp.array[wp.spatial_vector],
    out_generalized_body_force: wp.array[float],
):
    d = wp.tid()
    if d < dof_count:
        value = float(0.0)
        start = articulation_start[articulation_index]
        for j in range(start, articulation_end[articulation_index]):
            wrench = body_f_world_com[joint_child[j]]
            for k in range(6):
                value += spatial_jacobian[articulation_index, 6 * (j - start) + k, d] * wrench[k]
        out_generalized_body_force[d] = value


@wp.kernel
def eval_articulation_actor_residual(
    articulation_index: int,
    articulation_dof_start: int,
    dof_count: int,
    mass_matrix: wp.array3d[float],
    joint_qdd: wp.array[float],
    coriolis_force: wp.array[float],
    gravity_force: wp.array[float],
    frozen_joint_f: wp.array[float],
    generalized_body_force: wp.array[float],
    out_actor_residual_q: wp.array[float],
):
    i = wp.tid()
    if i < dof_count:
        d = articulation_dof_start + i
        value = coriolis_force[d] + gravity_force[d] - frozen_joint_f[d] - generalized_body_force[i]
        for j in range(dof_count):
            value += mass_matrix[articulation_index, i, j] * joint_qdd[articulation_dof_start + j]
        out_actor_residual_q[i] = value


@wp.kernel
def scatter_articulation_actor_tangent(
    articulation_index: int,
    dof_count: int,
    global_q_offset: int,
    triplet_offset: int,
    inv_dt_sq: float,
    mass_matrix: wp.array3d[float],
    out_aq_actor_dense: wp.array2d[float],
    out_triplet_rows: wp.array[int],
    out_triplet_columns: wp.array[int],
    out_triplet_values: wp.array[float],
):
    i, j = wp.tid()
    if i < dof_count and j < dof_count:
        value = mass_matrix[articulation_index, i, j] * inv_dt_sq
        out_aq_actor_dense[i, j] = value
        slot = triplet_offset + i * dof_count + j
        out_triplet_rows[slot] = global_q_offset + i
        out_triplet_columns[slot] = global_q_offset + j
        out_triplet_values[slot] = value


@wp.kernel(enable_backward=False)
def _validate_joint_term_inputs(
    q_target: wp.array[float],
    v_target: wp.array[float],
    joint_f: wp.array[float],
    target_map: wp.array[int],
    driven: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    if not wp.isfinite(q_target[target_map[i]]) or not wp.isfinite(v_target[i]) or not wp.isfinite(joint_f[i]):
        wp.atomic_max(status, 0, 1)
    if driven[i] != 0 and joint_f[i] != 0.0:
        wp.atomic_max(status, 0, 2)


@wp.func
def _limit_side(d: float, speed: float, ke: float, kd: float, width: float, inv_h: float):
    activation = float(0.0)
    derivative = float(0.0)
    if d >= width:
        activation = 1.0
    elif d > 0.0:
        t = d / width
        activation = t * t * (3.0 - 2.0 * t)
        derivative = 6.0 * t * (1.0 - t) / width
    outward = wp.max(speed, 0.0)
    elastic = ke * wp.max(d, 0.0)
    damping = kd * activation * outward
    tangent = float(0.0)
    if d > 0.0:
        tangent = ke + kd * derivative * outward
    if speed > 0.0:
        tangent += kd * activation * inv_h
    return elastic + damping, tangent, 0.5 * ke * wp.max(d, 0.0) * wp.max(d, 0.0), -damping * speed


@wp.kernel(enable_backward=False)
def _evaluate_joint_terms(
    q: wp.array[float],
    velocity: wp.array[float],
    q_target: wp.array[float],
    v_target: wp.array[float],
    target_map: wp.array[int],
    params: wp.array2d[float],
    inv_h: float,
    residual: wp.array2d[float],
    force: wp.array2d[float],
    tangent: wp.array2d[float],
    potential: wp.array2d[float],
    dissipation_power: wp.array2d[float],
    saturated: wp.array[int],
    out_status: wp.array[int],
):
    i = wp.tid()
    v = velocity[i]
    kp, kd, effort = params[i, 0], params[i, 1], params[i, 2]
    rp, tp, ep = float(0.0), float(0.0), float(0.0)
    saturated[i] = 0
    a = kp + kd * inv_h
    if a > 0.0:
        raw = kp * (q[i] - q_target[target_map[i]]) + kd * (v - v_target[i])
        rp = wp.clamp(raw, -effort, effort)
        if wp.abs(raw) < effort:
            tp = a
            ep = 0.5 * raw * raw / a
        else:
            saturated[i] = 1
            ep = effort * (wp.abs(raw) - 0.5 * effort) / a
        if not wp.isfinite(raw) or not wp.isfinite(a):
            wp.atomic_max(out_status, 0, 1)
    ke, limit_kd, width = params[i, 5], params[i, 6], params[i, 7]
    upper, ku, eu, pu = _limit_side(q[i] - params[i, 4], v, ke, limit_kd, width, inv_h)
    lower, kl, el, pl = _limit_side(params[i, 3] - q[i], -v, ke, limit_kd, width, inv_h)
    f, eps = params[i, 8], params[i, 9]
    rf, tf, ef, pf = float(0.0), float(0.0), float(0.0), float(0.0)
    if f > 0.0:
        denom = wp.sqrt(v * v + eps * eps)
        rf = f * v / denom
        tf = f * inv_h * (eps / denom) * (eps / denom) / denom
        ef = f / inv_h * (denom - eps)
        pf = -rf * v
    residual[i, 0] = rp
    residual[i, 1] = upper - lower
    residual[i, 2] = rf
    tangent[i, 0] = tp
    tangent[i, 1] = ku + kl
    tangent[i, 2] = tf
    potential[i, 0] = ep
    potential[i, 1] = eu + el
    potential[i, 2] = ef
    dissipation_power[i, 0] = 0.0
    dissipation_power[i, 1] = pu + pl
    dissipation_power[i, 2] = pf
    for term in range(3):
        force[i, term] = -residual[i, term]
        if (
            not wp.isfinite(residual[i, term])
            or not wp.isfinite(tangent[i, term])
            or tangent[i, term] < 0.0
            or not wp.isfinite(potential[i, term])
            or not wp.isfinite(dissipation_power[i, term])
        ):
            wp.atomic_max(out_status, 0, 1)
    if not wp.isfinite(q[i]) or not wp.isfinite(v):
        wp.atomic_max(out_status, 0, 1)


@wp.kernel(enable_backward=False)
def add_joint_term_residual(terms: wp.array2d[float], residual: wp.array[float]):
    i = wp.tid()
    residual[i] += terms[i, 0] + terms[i, 1] + terms[i, 2]


@wp.kernel(enable_backward=False)
def scatter_joint_term_tangent(
    terms: wp.array2d[float], aq: wp.array2d[float], triplet_values: wp.array[float], nq: int
):
    i = wp.tid()
    value = terms[i, 0] + terms[i, 1] + terms[i, 2]
    aq[i, i] += value
    triplet_values[i * nq + i] += value


class MonolithicJointTermsWorkspace:
    """Frozen scalar joint parameters and allocation-free candidate evaluation.

    Columns of residual/force/tangent/potential are PD, limit, joint friction.
    Potential contains the PD/friction incremental potentials (up to constants)
    and only the elastic limit energy. Power contains limit damping and joint
    friction power; PD power is not included because targets can inject work.
    Parameters are copied at construction; rebuild after editing model values.
    """

    def __init__(
        self,
        model: Model,
        *,
        implicit_pd: bool = False,
        limits: bool = False,
        friction: bool = False,
        limit_width: tuple[float, ...] | None = None,
        friction_velocity_scale: tuple[float, ...] | None = None,
    ):
        for value in (implicit_pd, limits, friction):
            if type(value) is not bool:
                raise ValueError("Joint term switches must be bool")
        _validate_joint_coordinate_layout(model)
        self.model, self.device, self.count = model, model.device, model.joint_dof_count
        self.implicit_pd = implicit_pd
        nq = self.count
        allowed = set()
        self._sources = {}

        def read(name):
            array = getattr(model, name)
            if array is None or (array.shape, array.dtype, array.device) != ((nq,), wp.float32, model.device):
                raise ValueError(f"Invalid joint parameter array: {name}")
            values = array.numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite joint parameter: {name}")
            self._sources[name] = array
            return values

        def scale(values, enabled, name):
            if not enabled:
                if values is not None:
                    raise ValueError(f"{name} requires its joint term enabled")
                return np.ones(nq, dtype=np.float32)
            if values is None:
                raise ValueError(f"{name} must specify each joint DOF")
            result = np.asarray(values, dtype=np.float32)
            if result.shape != (nq,) or not np.isfinite(result).all() or np.any(result <= 0):
                raise ValueError(f"{name} must be positive finite per-DOF values")
            return result

        parameters = np.zeros((nq, 10), dtype=np.float32)
        parameters[:, 7] = scale(limit_width, limits, "limit_width")
        parameters[:, 9] = scale(friction_velocity_scale, friction, "friction_velocity_scale")
        driven = np.zeros(nq, dtype=np.int32)
        if implicit_pd:
            kp, kd = read("joint_target_ke"), read("joint_target_kd")
            if np.any(kp < 0) or np.any(kd < 0):
                raise ValueError("PD gains must be nonnegative")
            driven = ((kp > 0) | (kd > 0)).astype(np.int32)
            modes = model.joint_target_mode
            if modes is None or (modes.shape, modes.dtype, modes.device) != ((nq,), wp.int32, model.device):
                raise ValueError("Invalid joint_target_mode")
            mode = modes.numpy()
            if np.any(mode[driven != 0] != int(JointTargetMode.POSITION_VELOCITY)) or np.any(
                ~np.isin(
                    mode[driven == 0], [JointTargetMode.NONE, JointTargetMode.EFFORT, JointTargetMode.POSITION_VELOCITY]
                )
            ):
                raise ValueError("Implicit PD requires POSITION_VELOCITY mode")
            self._sources["joint_target_mode"] = modes
            effort, speed = read("joint_effort_limit"), read("joint_velocity_limit")
            if np.any(effort[driven != 0] <= 0) or np.any(speed[driven != 0] <= 0):
                raise ValueError("Driven joint effort/velocity limits must be positive")
            parameters[:, 0], parameters[:, 1], parameters[:, 2] = kp, kd, effort
            self.velocity_limit = wp.array(speed, dtype=float, device=model.device)
            allowed.update(("joint_target_ke", "joint_target_kd"))
        if limits:
            low, high = read("joint_limit_lower"), read("joint_limit_upper")
            ke, kd = read("joint_limit_ke"), read("joint_limit_kd")
            if np.any(low > high) or np.any(ke < 0) or np.any(kd < 0):
                raise ValueError("Invalid joint limit bounds or gains")
            parameters[:, 3], parameters[:, 4], parameters[:, 5], parameters[:, 6] = low, high, ke, kd
            allowed.update(("joint_limit_ke", "joint_limit_kd"))
        if friction:
            f = read("joint_friction")
            if np.any(f < 0):
                raise ValueError("Joint friction must be nonnegative")
            parameters[:, 8] = f
            allowed.add("joint_friction")
        self.allowed_properties = frozenset(allowed)
        starts = model.joint_target_q_start
        if starts is None or (starts.shape, starts.dtype, starts.device) != (
            (model.joint_count + 1,),
            wp.int32,
            model.device,
        ):
            raise ValueError("Invalid joint_target_q_start")
        moving = model.joint_type.numpy() != int(JointType.FIXED)
        target_ids = starts.numpy()[:-1][moving]
        if not np.array_equal(starts.numpy(), model.joint_q_start.numpy()):
            raise ValueError("Invalid scalar joint target layout")
        self._sources["joint_target_q_start"] = starts
        self.target_map = wp.array(target_ids, dtype=int, device=model.device)
        self.params = wp.array(parameters, dtype=float, device=model.device)
        self.driven = wp.array(driven, dtype=int, device=model.device)
        self.target_q = wp.zeros(model.joint_coord_count, dtype=float, device=model.device)
        self.target_qd = wp.zeros(nq, dtype=float, device=model.device)
        self.residual = wp.zeros((nq, 3), dtype=float, device=model.device)
        self.force = wp.zeros_like(self.residual)
        self.tangent = wp.zeros_like(self.residual)
        self.potential = wp.zeros_like(self.residual)
        self.dissipation_power = wp.zeros_like(self.residual)
        self.saturated = wp.zeros(nq, dtype=int, device=model.device)
        self.status = wp.zeros(1, dtype=int, device=model.device)
        self.final_force = wp.zeros_like(self.force)
        self.final_residual = wp.zeros_like(self.residual)
        self.final_tangent = wp.zeros_like(self.tangent)
        self.final_power = wp.zeros_like(self.dissipation_power)
        self.final_saturated = wp.zeros_like(self.saturated)
        self.final_generation = -1

    def snapshot_targets(self, control) -> None:
        """Validate before copying inputs; no public/candidate state is mutated."""
        for name, array in self._sources.items():
            if getattr(self.model, name) is not array:
                raise ValueError(f"Stale joint parameter storage: {name}")
        if not self.implicit_pd:
            return
        for name, count in (
            ("joint_target_q", self.model.joint_coord_count),
            ("joint_target_qd", self.count),
            ("joint_f", self.count),
        ):
            array = getattr(control, name)
            if array is None or (array.shape, array.dtype, array.device) != ((count,), wp.float32, self.device):
                raise ValueError(f"Invalid joint control array: {name}")
        self.status.zero_()
        wp.launch(
            _validate_joint_term_inputs,
            self.count,
            [
                control.joint_target_q,
                control.joint_target_qd,
                control.joint_f,
                self.target_map,
                self.driven,
                self.status,
            ],
            device=self.device,
        )
        code = int(self.status.numpy()[0])
        if code:
            raise ValueError("Duplicate driven joint_f" if code == 2 else "Nonfinite joint control")
        wp.copy(self.target_q, control.joint_target_q)
        wp.copy(self.target_qd, control.joint_target_qd)

    def evaluate(self, state: State, dt: float) -> None:
        """Evaluate candidate terms into reusable scratch; caller checks status."""
        self.status.zero_()
        wp.launch(
            _evaluate_joint_terms,
            self.count,
            [
                state.joint_q,
                state.joint_qd,
                self.target_q,
                self.target_qd,
                self.target_map,
                self.params,
                1.0 / dt,
                self.residual,
                self.force,
                self.tangent,
                self.potential,
                self.dissipation_power,
                self.saturated,
                self.status,
            ],
            device=self.device,
        )

    def publish(self, generation: int) -> None:
        """Copy only a validated returned state's terms to the final cache."""
        for dst, src in (
            (self.final_force, self.force),
            (self.final_residual, self.residual),
            (self.final_tangent, self.tangent),
            (self.final_power, self.dissipation_power),
            (self.final_saturated, self.saturated),
        ):
            wp.copy(dst, src)
        self.final_generation = generation
