# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned storage and thin orchestration of Newton articulation dynamics."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import JointType, Model, State, eval_fk, eval_jacobian, eval_mass_matrix
from ...sim.inverse_dynamics import _compute_coriolis_force, _compute_gravity_force, _InverseDynamicsScratchBuffer


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

    def __init__(self, model: Model, *, stream: wp.Stream | None = None) -> None:
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
        for name in (
            "joint_armature",
            "joint_damping",
            "joint_friction",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
        ):
            if np.any(getattr(model, name).numpy() != 0.0):
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
        types = model.joint_type.numpy()
        if not np.isin(types, [JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC]).all():
            raise ValueError("Monolithic supports only FIXED, REVOLUTE and PRISMATIC joints")
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
        moving = types != JointType.FIXED
        coord = model.joint_q_start.numpy()[:-1][moving]
        dofs = model.joint_qd_start.numpy()[:-1][moving]
        if (
            len(coord) != model.joint_dof_count
            or len(coord) != model.joint_coord_count
            or not np.array_equal(dofs, np.arange(model.joint_dof_count))
        ):
            raise ValueError("Invalid revolute/prismatic joint coordinate layout")
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
    eval_mass_matrix(model, state, H=workspace.M, J=scratch.J, body_I_s=scratch.body_I_s, joint_S_s=scratch.joint_S_s)
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
