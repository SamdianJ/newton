# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched native COM dynamics for single-body world-root FREE actors."""

from typing import Any

import numpy as np
import warp as wp

from .articulation import _inner
from .articulation_state import _initial_symmetric, _predict, _recover_history, _skew


@wp.kernel
def _initialize(
    q: wp.array[Any],
    qd: wp.array[Any],
    qs: wp.array[int],
    vs: wp.array[int],
    xp: wp.array[Any],
    xc: wp.array[Any],
    local_com: wp.array[Any],
    com: wp.array[Any],
    rotation: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
):
    i = wp.tid()
    s, d = qs[i], vs[i]
    quat = wp.normalize(type(rotation[0])(q[s + 3], q[s + 4], q[s + 5], q[s + 6]))
    transform = xp[i] * type(xp[0])(type(com[0])(q[s], q[s + 1], q[s + 2]), quat) * wp.transform_inverse(xc[i])
    rotation[i] = wp.normalize(wp.transform_get_rotation(transform))
    com[i] = wp.transform_get_translation(transform) + wp.quat_rotate(rotation[i], local_com[i])
    parent_rotation = wp.transform_get_rotation(xp[i])
    v[i] = wp.quat_rotate(parent_rotation, type(com[0])(qd[d], qd[d + 1], qd[d + 2]))
    omega[i] = wp.quat_rotate(parent_rotation, type(com[0])(qd[d + 3], qd[d + 4], qd[d + 5]))


@wp.kernel
def _matrices(quats: wp.array[Any], rotations: wp.array[Any]):
    i = wp.tid()
    rotations[i] = wp.quat_to_matrix(quats[i])


@wp.kernel
def _mirrors(
    com: wp.array[Any],
    rotation: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
    xp: wp.array[Any],
    xc: wp.array[Any],
    local_com: wp.array[Any],
    q: wp.array2d[Any],
    qd: wp.array2d[Any],
    poses: wp.array[Any],
):
    i = wp.tid()
    poses[i] = type(xp[0])(com[i] - wp.quat_rotate(rotation[i], local_com[i]), rotation[i])
    free = wp.transform_inverse(xp[i]) * poses[i] * xc[i]
    p, r = wp.transform_get_translation(free), wp.normalize(wp.transform_get_rotation(free))
    linear = wp.quat_rotate_inv(wp.transform_get_rotation(xp[i]), v[i])
    angular = wp.quat_rotate_inv(wp.transform_get_rotation(xp[i]), omega[i])
    for j in range(3):
        q[i, j] = p[j]
        qd[i, j] = linear[j]
        qd[i, j + 3] = angular[j]
    for j in range(4):
        q[i, j + 3] = r[j]


@wp.kernel
def _publish(
    qlocal: wp.array2d[Any],
    vlocal: wp.array2d[Any],
    poses: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
    qs: wp.array[int],
    vs: wp.array[int],
    bodies: wp.array[int],
    q: wp.array[Any],
    qd: wp.array[Any],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    for j in range(7):
        q[qs[i] + j] = qlocal[i, j]
    for j in range(6):
        qd[vs[i] + j] = vlocal[i, j]
    p, r = wp.transform_get_translation(poses[i]), wp.transform_get_rotation(poses[i])
    body_q[bodies[i]] = wp.transform(wp.vec3(p), wp.quat(r))
    body_qd[bodies[i]] = wp.spatial_vector(wp.vec3(v[i]), wp.vec3(omega[i]))


@wp.kernel
def _trial(
    com: wp.array[Any],
    rotation: wp.array[Any],
    rows: wp.array2d[int],
    direction: wp.array[float],
    alpha: Any,
    out_com: wp.array[Any],
    out_rotation: wp.array[Any],
):
    i = wp.tid()
    dc = type(com[0])(
        type(alpha)(direction[rows[i, 0]]), type(alpha)(direction[rows[i, 1]]), type(alpha)(direction[rows[i, 2]])
    )
    dr = alpha * type(com[0])(
        type(alpha)(direction[rows[i, 3]]), type(alpha)(direction[rows[i, 4]]), type(alpha)(direction[rows[i, 5]])
    )
    angle = wp.length(dr)
    increment = type(rotation[0])(type(alpha)(0.0), type(alpha)(0.0), type(alpha)(0.0), type(alpha)(1.0))
    if angle != type(alpha)(0.0):
        increment = wp.quat_from_axis_angle(dr / angle, angle)
    out_com[i] = com[i] + alpha * dc
    out_rotation[i] = wp.normalize(increment * rotation[i])


@wp.kernel
def _terms(
    com: wp.array[Any],
    rotation: wp.array[Any],
    rotations: wp.array[Any],
    tn: wp.array[Any],
    rn: wp.array[Any],
    predicted_t: wp.array[Any],
    predicted_r: wp.array[Any],
    mass: wp.array[Any],
    second: wp.array[Any],
    force: wp.array[Any],
    torque: wp.array[Any],
    h: Any,
    tangent: bool,
    energy: wp.array[Any],
    residual: wp.array2d[Any],
    b: wp.array3d[float],
    diagonal: wp.array2d[Any],
):
    i = wp.tid()
    r = wp.quat_to_matrix(rotation[i])
    rotations[i] = r
    dt, dr = com[i] - predicted_t[i], r - predicted_r[i]
    rs = dr * second[i]
    energy[i] = (mass[i] * wp.dot(dt, dt) + _inner(rs, dr)) / (type(h)(2.0) * h * h) - wp.dot(force[i], com[i] - tn[i])
    energy[i] += type(h)(0.5) * wp.trace(_skew(torque[i]) * r * wp.transpose(rn[i]))
    a = rs * wp.transpose(r) / (h * h)
    rt = mass[i] * dt / (h * h) - force[i]
    rr = type(com[0])(a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]) - torque[i]
    inertia = wp.trace(second[i]) * wp.identity(n=3, dtype=type(h)) - second[i]
    world = r * inertia * wp.transpose(r)
    for j in range(3):
        residual[i, j] = rt[j]
        residual[i, j + 3] = rr[j]
        diagonal[i, j] = mass[i] / (h * h)
        diagonal[i, j + 3] = world[j, j] / (h * h)
    if tangent:
        c = r * second[i] * wp.transpose(predicted_r[i])
        raw = (wp.trace(c) * wp.identity(n=3, dtype=type(h)) - (c + wp.transpose(c)) * type(h)(0.5)) / (h * h)
        vectors, values = wp.eig3(raw)
        positive = type(com[0])(
            wp.max(values[0], type(h)(0.0)), wp.max(values[1], type(h)(0.0)), wp.max(values[2], type(h)(0.0))
        )
        projected = vectors * wp.diag(positive) * wp.transpose(vectors)
        for j in range(6):
            for k in range(6):
                value = type(h)(0.0)
                if j < 3 and k == j:
                    value = mass[i] / (h * h)
                elif j >= 3 and k >= 3:
                    value = projected[j - 3, k - 3]
                b[i, j, k] = float(value)


@wp.kernel
def _scatter_residual(rows: wp.array2d[int], local: wp.array2d[Any], out: wp.array[Any]):
    i, j = wp.tid()
    out[rows[i, j]] += local[i, j]


@wp.kernel
def _scatter_tangent(
    b: wp.array3d[float],
    rows: wp.array2d[int],
    diagonal: wp.array2d[Any],
    offset: int,
    values: wp.array[float],
    out_diagonal: wp.array[float],
):
    i, j, k = wp.tid()
    values[offset + 36 * i + 6 * j + k] = b[i, j, k]
    if j == k:
        out_diagonal[rows[i, j]] = float(diagonal[i, j])


class RigidObjective:
    """Own fixed rigid maps, native objective scratch and publication conversion."""

    def __init__(self, model, actors, *, dtype):
        self.model, self.device, self.dtype = model, model.device, dtype
        self.actors = tuple(a for a in actors.actors if a.kind == "rigid")
        self.count = len(self.actors)
        self.numpy_dtype = np.float64 if dtype == wp.float64 else np.float32
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
        self.transform = wp.transformd if dtype == wp.float64 else wp.transform
        self.body_indices = [a.body_indices[0] for a in self.actors]
        joint_indices = [a.joint_indices[0] for a in self.actors]
        self.coord_indices = np.array([a.coord_indices for a in self.actors], dtype=np.int32).reshape(-1, 7)
        self.dof_indices = np.array([a.dof_indices for a in self.actors], dtype=np.int32).reshape(-1, 6)
        self.rows_numpy = np.array([range(a.dof_start, a.dof_start + 6) for a in self.actors], dtype=np.int32).reshape(
            -1, 6
        )
        self.rows = wp.array(self.rows_numpy, dtype=int, device=self.device)
        self.qs = wp.array(self.coord_indices[:, 0], dtype=int, device=self.device)
        self.vs = wp.array(self.dof_indices[:, 0], dtype=int, device=self.device)
        self.bodies = wp.array(self.body_indices, dtype=int, device=self.device)
        self.sources = {}
        if self.count:
            for name in ("shape_flags", "shape_contact_pairs"):
                self.sources[name] = getattr(model, name)
            self.values = {name: value.numpy().copy() for name, value in self.sources.items()}
            for name in (
                "joint_target_ke",
                "joint_target_kd",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_friction",
                "joint_damping",
                "joint_armature",
            ):
                if np.any(getattr(model, name).numpy()[self.dof_indices] != 0):
                    raise ValueError(f"Unsupported FREE joint term: {name}")
        else:
            self.values = {}
        for name in ("joint_X_p", "joint_X_c"):
            value = getattr(model, name).numpy()[joint_indices].astype(self.numpy_dtype)
            if self.count:
                norm = np.linalg.norm(value[:, 3:], axis=1)
                if not np.isfinite(value).all() or np.any(abs(norm - 1) > 1e-5):
                    raise ValueError("Rigid joint anchors require finite unit rotations")
                value[:, 3:] /= norm[:, None]
            setattr(
                self, "xp" if name == "joint_X_p" else "xc", wp.array(value, dtype=self.transform, device=self.device)
            )
        self.mass = wp.array(model.body_mass.numpy()[self.body_indices], dtype=dtype, device=self.device)
        self.local_com = wp.array(model.body_com.numpy()[self.body_indices], dtype=self.vector, device=self.device)
        inertia = model.body_inertia.numpy()[self.body_indices].astype(self.numpy_dtype)
        second = np.trace(inertia, axis1=1, axis2=2)[:, None, None] / 2 * np.eye(3) - inertia
        self.second = wp.array(second, dtype=self.matrix, device=self.device)
        for name in ("tn", "predicted_t", "force", "torque"):
            setattr(self, name, wp.zeros(self.count, dtype=self.vector, device=self.device))
        for name in ("rn", "predicted_r", "rotations"):
            setattr(self, name, wp.zeros(self.count, dtype=self.matrix, device=self.device))
        self.energy = wp.zeros(self.count, dtype=dtype, device=self.device)
        self.residual = wp.zeros((self.count, 6), dtype=dtype, device=self.device)
        self.tangent = wp.zeros((self.count, 6, 6), dtype=float, device=self.device)
        self.diagonal = wp.zeros((self.count, 6), dtype=dtype, device=self.device)
        self.mirror_q = wp.zeros((self.count, 7), dtype=dtype, device=self.device)
        self.mirror_v = wp.zeros((self.count, 6), dtype=dtype, device=self.device)
        self.poses = wp.zeros(self.count, dtype=self.transform, device=self.device)
        self._maps = {name: getattr(self, name) for name in ("rows", "qs", "vs", "bodies")}
        self._map_values = {name: value.numpy().copy() for name, value in self._maps.items()}
        self.has_torque = False

    def validate_identity(self):
        for name, source in self._maps.items():
            if getattr(self, name) is not source or not np.array_equal(source.numpy(), self._map_values[name]):
                raise ValueError(f"Rigid mapping changed: {name}")
        for name, source in self.sources.items():
            if getattr(self.model, name) is not source or not np.array_equal(source.numpy(), self.values[name]):
                raise ValueError(f"Rigid contact eligibility changed: {name}")

    def initialize(self, state, h):
        history = state._monolithic_coupled
        q = state.joint_q.numpy()[self.coord_indices]
        if np.any(abs(np.linalg.norm(q[:, 3:], axis=1) - 1) > 1e-5):
            raise ValueError("FREE coordinates require unit quaternions")
        wp.launch(
            _initialize,
            self.count,
            [
                state.joint_q,
                state.joint_qd,
                self.qs,
                self.vs,
                self.xp,
                self.xc,
                self.local_com,
                history.rigid_com,
                history.rigid_rotation,
                history.rigid_v_com,
                history.rigid_omega,
            ],
            device=self.device,
        )
        if np.any(h * np.linalg.norm(history.rigid_omega.numpy(), axis=1) >= 1):
            raise ValueError("Initial rigid angular velocity requires dt * |omega| < 1")
        wp.launch(
            _initial_symmetric, self.count, [history.rigid_omega, self.dtype(h), history.rigid_vsym], device=self.device
        )
        self.publish(state)

    def mirrors(self, history):
        wp.launch(
            _mirrors,
            self.count,
            [
                history.rigid_com,
                history.rigid_rotation,
                history.rigid_v_com,
                history.rigid_omega,
                self.xp,
                self.xc,
                self.local_com,
                self.mirror_q,
                self.mirror_v,
                self.poses,
            ],
            device=self.device,
        )

    def publish(self, state):
        history = state._monolithic_coupled
        self.mirrors(history)
        wp.launch(
            _publish,
            self.count,
            [
                self.mirror_q,
                self.mirror_v,
                self.poses,
                history.rigid_v_com,
                history.rigid_omega,
                self.qs,
                self.vs,
                self.bodies,
                state.joint_q,
                state.joint_qd,
                state.body_q,
                state.body_qd,
            ],
            device=self.device,
        )

    def validate(self, state):
        history = state._monolithic_coupled
        tolerance = 1e-10 if self.dtype == wp.float64 else 1e-6
        if np.any(abs(np.linalg.norm(history.rigid_rotation.numpy(), axis=1) - 1) > tolerance):
            raise ValueError("Invalid native rigid quaternion")
        # A recovered discrete rate reconstructs an orthogonal relative
        # rotation. Reset's forward predictor uses the opposite sign.
        sym = history.rigid_vsym.numpy()
        omega = history.rigid_omega.numpy()
        rate = np.zeros((self.count, 3, 3), dtype=self.numpy_dtype)
        rate[:, 0, 0], rate[:, 1, 1], rate[:, 2, 2] = sym[:, 0], sym[:, 1], sym[:, 2]
        rate[:, 0, 1], rate[:, 1, 0] = sym[:, 3] - omega[:, 2], sym[:, 3] + omega[:, 2]
        rate[:, 0, 2], rate[:, 2, 0] = sym[:, 4] + omega[:, 1], sym[:, 4] - omega[:, 1]
        rate[:, 1, 2], rate[:, 2, 1] = sym[:, 5] - omega[:, 0], sym[:, 5] + omega[:, 0]
        sign = -1 if history.force_valid.numpy()[0] else 1
        relative = np.eye(3) + sign * history.last_dt.numpy()[0] * rate
        if not np.allclose(relative @ np.swapaxes(relative, 1, 2), np.eye(3), atol=tolerance, rtol=tolerance):
            raise ValueError("Invalid rigid discrete integration history")
        for name in ("com", "rotation"):
            if not np.array_equal(
                getattr(history, "rigid_" + name).numpy(), getattr(history, "rigid_bound_" + name).numpy()
            ):
                raise ValueError("Rigid pose changed; explicitly reset history")
        self.mirrors(history)
        q, expected = state.joint_q.numpy()[self.coord_indices], self.mirror_q.numpy()
        # Quaternion signs encode the same rotation; other native values remain bound exactly.
        if self.count:
            sign = np.where(np.sum(q[:, 3:] * expected[:, 3:], axis=1) < 0, -1, 1)
            expected[:, 3:] *= sign[:, None]
        if not np.allclose(q, expected, atol=tolerance, rtol=tolerance) or not np.allclose(
            state.joint_qd.numpy()[self.dof_indices], self.mirror_v.numpy(), atol=tolerance, rtol=tolerance
        ):
            raise ValueError("FREE mirror disagrees with native rigid authority")

    def prepare(self, state, control, dt):
        if not self.count:
            return
        history = state._monolithic_coupled
        force = state.body_f.numpy()[self.body_indices].astype(self.numpy_dtype) + control.joint_f.numpy()[
            self.dof_indices
        ].astype(self.numpy_dtype)
        gravity = self.model.gravity.numpy()[self.model.body_world.numpy()[self.body_indices]]
        self.force.assign(force[:, :3] + self.mass.numpy()[:, None] * gravity)
        self.torque.assign(force[:, 3:])
        self.has_torque = bool(np.any(force[:, 3:] != 0))
        wp.copy(self.tn, history.rigid_com)
        wp.launch(_matrices, self.count, [history.rigid_rotation, self.rn], device=self.device)
        wp.launch(
            _predict,
            self.count,
            [
                self.tn,
                self.rn,
                history.rigid_v_com,
                history.rigid_omega,
                history.rigid_vsym,
                self.dtype(dt),
                self.predicted_t,
                self.predicted_r,
            ],
            device=self.device,
        )

    def evaluate(self, state, dt, tangent):
        if not self.count:
            return
        history = state._monolithic_coupled
        wp.launch(
            _terms,
            self.count,
            [
                history.rigid_com,
                history.rigid_rotation,
                self.rotations,
                self.tn,
                self.rn,
                self.predicted_t,
                self.predicted_r,
                self.mass,
                self.second,
                self.force,
                self.torque,
                self.dtype(dt),
                tangent,
                self.energy,
                self.residual,
                self.tangent,
                self.diagonal,
            ],
            device=self.device,
        )

    def trial(self, source, target, direction, alpha):
        if not self.count:
            return
        a, b = source._monolithic_coupled, target._monolithic_coupled
        wp.launch(
            _trial,
            self.count,
            [a.rigid_com, a.rigid_rotation, self.rows, direction, self.dtype(alpha), b.rigid_com, b.rigid_rotation],
            device=self.device,
        )

    def recover(self, state, dt):
        if not self.count:
            return
        history = state._monolithic_coupled
        wp.launch(_matrices, self.count, [history.rigid_rotation, self.rotations], device=self.device)
        wp.launch(
            _recover_history,
            self.count,
            [
                self.tn,
                self.rn,
                history.rigid_com,
                self.rotations,
                self.dtype(dt),
                history.rigid_v_com,
                history.rigid_omega,
                history.rigid_vsym,
            ],
            device=self.device,
        )
        self.publish(state)

    def domain(self, state):
        if not self.count:
            return True
        self.validate_identity()
        wp.launch(_matrices, self.count, [state._monolithic_coupled.rigid_rotation, self.rotations], device=self.device)
        return bool(np.all(np.einsum("bij,bij->b", self.rotations.numpy(), self.rn.numpy()) > 1))
