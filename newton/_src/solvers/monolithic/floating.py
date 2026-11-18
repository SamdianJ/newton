# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native poses of floating-base roots: FREE world-root joints of articulated trees.

A root's unknowns are the first six rows of its tree's block: the world COM increment and the
left rotation increment, as for a FREE rigid actor. Its COM and quaternion are native coupled
state; the exact increments of every internal state are kept beside the scalar joint steps,
whose root rows stay zero. ``joint_q``/``joint_qd`` are mirrors in Newton's FREE convention.

A root may carry an implicit 6D pose drive and an external world COM wrench (``joint_f``). The
drive acts on u_t = k_t (c_n - c_t + dc) + d_t (dc / h - v_t) and
u_r = k_r (Log(R_n R_t^T) + phi) + d_r (phi / h - w_t), phi = Log(R R_n^T), each with the Huber
energy of the scalar joint PD truncated by the norm of u (force and torque limits): a convex C1
energy of (dc, phi). Rotation rows are left perturbations, so the rotational residual is
J_l^-T(phi) dE/dphi and its tangent the Gauss-Newton part J_l^-T H J_l^-1.
"""

from typing import Any

import numpy as np
import warp as wp

from ...sim import JointTargetMode
from .articulation import _rotation_step
from .convergence import abs3
from .numeric_parameters import scalar_value
from .rigid import _initialize, _mirrors
from .rigid_pose import _derive_steps, _pose_coordinates, _pose_difference, _predict_pose, _trial


@wp.kernel
def _gather_root_prediction(
    bodies: wp.array[int],
    predicted_t: wp.array[Any],
    step_t: wp.array[Any],
    omega: wp.array[Any],
    out_predicted: wp.array[Any],
    out_step: wp.array[Any],
    out_omega: wp.array[Any],
):
    f = wp.tid()
    out_predicted[f] = predicted_t[bodies[f]]
    out_step[f] = step_t[bodies[f]]
    out_omega[f] = omega[bodies[f]]


@wp.kernel
def _gather_root_rates(
    bodies: wp.array[int], v: wp.array[Any], omega: wp.array[Any], out_v: wp.array[Any], out_omega: wp.array[Any]
):
    f = wp.tid()
    out_v[f] = v[bodies[f]]
    out_omega[f] = omega[bodies[f]]


@wp.kernel
def _scatter_mirrors(
    qlocal: wp.array2d[Any],
    vlocal: wp.array2d[Any],
    qs: wp.array[int],
    vs: wp.array[int],
    rates: int,
    q: wp.array[Any],
    qd: wp.array[Any],
):
    f = wp.tid()
    for j in range(7):
        q[qs[f] + j] = qlocal[f, j]
    if rates != 0:
        for j in range(6):
            qd[vs[f] + j] = vlocal[f, j]


@wp.func
def _quat_log(q: Any):
    """Rotation vector of a unit quaternion, using the short representative (angle <= pi)."""
    v = wp.vector(q[0], q[1], q[2])
    w = q[3]
    if w < type(w)(0.0):
        v = -v
        w = -w
    length = wp.length(v)
    factor = type(w)(2.0)
    if length > type(w)(0.0):
        factor = type(w)(2.0) * wp.atan2(length, w) / length
    return factor * v


@wp.func
def _left_jacobian_inverse(phi: Any):
    """J_l^-1(phi): d Log(Exp(d) Exp(phi)) / d d at d = 0."""
    theta = wp.length(phi)
    k = wp.skew(phi)
    one = type(theta)(1.0)
    # Series below 1e-2 rad: 1/12 + theta^2/720 + theta^4/30240 (relative error < 1e-12).
    c = (
        one / type(theta)(12.0)
        + theta * theta / type(theta)(720.0)
        + theta * theta * theta * theta / type(theta)(30240.0)
    )
    if theta > type(theta)(1.0e-2):
        # 1/theta^2 - cot(theta/2)/(2 theta), equal to the (1 + cos)/sin form without its 0/0 at pi.
        c = one / (theta * theta) - wp.cos(theta * type(theta)(0.5)) / (
            type(theta)(2.0) * theta * wp.sin(theta * type(theta)(0.5))
        )
    return wp.identity(n=3, dtype=type(theta)) - type(theta)(0.5) * k + c * (k * k)


@wp.func
def _huber(u: Any, a: Any, limit: Any):
    """Norm-truncated Huber energy of u = a x + b: value, gradient and Hessian with respect to x."""
    norm = wp.length(u)
    energy = norm * norm / (type(a)(2.0) * a)
    gradient = u
    hessian = a * wp.identity(n=3, dtype=type(a))
    if norm >= limit:
        unit = u / norm
        energy = limit * (norm - limit * type(a)(0.5)) / a
        gradient = limit * unit
        hessian = a * (limit / norm) * (wp.identity(n=3, dtype=type(a)) - wp.outer(unit, unit))
    return energy, gradient, hessian


# Per-root snapshot status bits.
_NONFINITE_TARGET_Q = wp.constant(1)
_NONFINITE_TARGET_QD = wp.constant(2)
_NONFINITE_JOINT_F = wp.constant(4)
_NONUNIT_TARGET = wp.constant(8)
_ROOT_TORQUE = wp.constant(16)


@wp.kernel
def _snapshot_controls(
    target_q: wp.array[float],
    target_qd: wp.array[float],
    joint_f: wp.array[float],
    target_starts: wp.array[int],
    dof_starts: wp.array[int],
    local_starts: wp.array[int],
    params: wp.array2d[Any],
    pose: wp.array2d[Any],
    rate: wp.array2d[Any],
    force: wp.array[Any],
    torque: wp.array[Any],
    joint_force: wp.array[Any],
    status: wp.array[int],
):
    f = wp.tid()
    t, d = target_starts[f], dof_starts[f]
    bits = int(0)
    for k in range(7):
        value = target_q[t + k]
        if not wp.isfinite(value):
            bits = bits | _NONFINITE_TARGET_Q
        pose[f, k] = type(params[0, 0])(value)
    for k in range(6):
        if not wp.isfinite(target_qd[d + k]):
            bits = bits | _NONFINITE_TARGET_QD
        if not wp.isfinite(joint_f[d + k]):
            bits = bits | _NONFINITE_JOINT_F
        rate[f, k] = type(params[0, 0])(target_qd[d + k])
    norm = wp.sqrt(
        pose[f, 3] * pose[f, 3] + pose[f, 4] * pose[f, 4] + pose[f, 5] * pose[f, 5] + pose[f, 6] * pose[f, 6]
    )
    # Stiff triples read the quaternion (the COM target through xc and the COM offset, the rotation error
    # directly); damping-only triples do not, but any driven root is checked (Newton's builder sets unit
    # free-joint targets).
    driven = params[f, 0] + params[f, 1] + params[f, 3] + params[f, 4] > type(norm)(0.0)
    if driven and not (wp.abs(norm - type(norm)(1.0)) <= type(norm)(1.0e-5)):
        bits = bits | _NONUNIT_TARGET
    f3, t3 = type(force[0])(), type(torque[0])()
    for k in range(3):
        f3[k] = type(params[0, 0])(joint_f[d + k])
        t3[k] = type(params[0, 0])(joint_f[d + 3 + k])
        if joint_f[d + 3 + k] != 0.0:
            bits = bits | _ROOT_TORQUE
    force[f] = f3
    torque[f] = t3
    status[f] = bits
    # The wrench enters the root term; the scalar joint rows of the root carry no force.
    for k in range(6):
        joint_force[local_starts[f] + k] = type(params[0, 0])(0.0)


@wp.kernel
def _drive_start(
    pose: wp.array2d[Any],
    rate: wp.array2d[Any],
    xp: wp.array[Any],
    xc: wp.array[Any],
    local_com: wp.array[Any],
    tn: wp.array[Any],
    quat_n: wp.array[Any],
    error_t: wp.array[Any],
    error_r: wp.array[Any],
    target_v: wp.array[Any],
    target_w: wp.array[Any],
):
    f = wp.tid()
    q = wp.normalize(type(quat_n[0])(pose[f, 3], pose[f, 4], pose[f, 5], pose[f, 6]))
    body = xp[f] * type(xp[0])(type(tn[0])(pose[f, 0], pose[f, 1], pose[f, 2]), q) * wp.transform_inverse(xc[f])
    rotation = wp.normalize(wp.transform_get_rotation(body))
    target_c = wp.transform_get_translation(body) + wp.quat_rotate(rotation, local_com[f])
    error_t[f] = tn[f] - target_c
    # Rotation from the target to the step start: Log(R_n R_t^T).
    error_r[f] = _quat_log(quat_n[f] * wp.quat_inverse(rotation))
    parent = wp.transform_get_rotation(xp[f])
    target_v[f] = wp.quat_rotate(parent, type(tn[0])(rate[f, 0], rate[f, 1], rate[f, 2]))
    target_w[f] = wp.quat_rotate(parent, type(tn[0])(rate[f, 3], rate[f, 4], rate[f, 5]))


@wp.kernel
def _drive_terms(
    steps: wp.array[Any],
    rotation_steps: wp.array[Any],
    params: wp.array2d[Any],
    error_t: wp.array[Any],
    error_r: wp.array[Any],
    target_v: wp.array[Any],
    target_w: wp.array[Any],
    force: wp.array[Any],
    torque: wp.array[Any],
    rows: wp.array2d[int],
    entries: wp.array2d[int],
    h_arg: Any,
    build_tangent: bool,
    energy: wp.array[Any],
    residual: wp.array[Any],
    absolute: wp.array[Any],
    tangent: wp.array[Any],
):
    h = scalar_value(h_arg)
    f = wp.tid()
    zero = type(h)(0.0)
    dc = steps[f]
    phi = _quat_log(rotation_steps[f])
    e = -wp.dot(force[f], dc) + type(h)(0.5) * wp.trace(wp.skew(torque[f]) * _rotation_step(rotation_steps[f]))
    gt, gr = -force[f], -torque[f]
    at, ar = abs3(force[f]), abs3(torque[f])
    ht = wp.matrix(dtype=type(h), shape=(3, 3))
    hr = wp.matrix(dtype=type(h), shape=(3, 3))
    a = params[f, 0] + params[f, 1] / h
    if a > zero:
        u = params[f, 0] * (error_t[f] + dc) + params[f, 1] * (dc / h - target_v[f])
        ee, g, hh = _huber(u, a, params[f, 2])
        e += ee
        gt = gt + g
        at = at + abs3(g)
        ht = hh
    a = params[f, 3] + params[f, 4] / h
    if a > zero:
        u = params[f, 3] * (error_r[f] + phi) + params[f, 4] * (phi / h - target_w[f])
        ee, g, hh = _huber(u, a, params[f, 5])
        jinv = _left_jacobian_inverse(phi)
        left = wp.transpose(jinv) * g
        e += ee
        gr = gr + left
        ar = ar + abs3(left)
        hr = wp.transpose(jinv) * hh * jinv
    energy[f] = e
    for k in range(3):
        residual[rows[f, k]] = residual[rows[f, k]] + gt[k]
        residual[rows[f, k + 3]] = residual[rows[f, k + 3]] + gr[k]
        absolute[rows[f, k]] = absolute[rows[f, k]] + at[k]
        absolute[rows[f, k + 3]] = absolute[rows[f, k + 3]] + ar[k]
    if build_tangent:
        for i in range(3):
            for j in range(3):
                tangent[entries[f, 6 * i + j]] = tangent[entries[f, 6 * i + j]] + ht[i, j]
                tangent[entries[f, 6 * (i + 3) + j + 3]] = tangent[entries[f, 6 * (i + 3) + j + 3]] + hr[i, j]


class FloatingRoots:
    """Own the maps, step-start snapshot and mirrors of every floating root of the trees."""

    def __init__(self, kinematics, model, *, implicit_pd=False):
        k = kinematics
        self.kinematics, self.model, self.device, self.dtype = k, model, k.device, k.dtype
        self.count = k.floating_count
        self.vector, self.quaternion, self.transform = k.vector, k.quaternion, k.transform
        joints = np.asarray(k.joint_ids, dtype=np.int32)[k.root_joints.numpy()]
        self.joint_indices = joints
        self.coord_indices = np.array(
            [range(int(s), int(s) + 7) for s in model.joint_q_start.numpy()[joints]], dtype=np.int32
        ).reshape(-1, 7)
        self.dof_indices = np.array(
            [range(int(s), int(s) + 6) for s in model.joint_qd_start.numpy()[joints]], dtype=np.int32
        ).reshape(-1, 6)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.qs = array(self.coord_indices[:, 0], int)
        self.vs = array(self.dof_indices[:, 0], int)
        self.bodies = k.root_bodies
        root_joints = k.root_joints.numpy()
        self.xp = array(k.xp.numpy()[root_joints], self.transform)
        self.xc = array(k.xc.numpy()[root_joints], self.transform)
        self.local_com = array(k.local_com.numpy()[self.bodies.numpy()], self.vector)
        # Global rows are bound by the solver once its linear layout exists.
        self.rows = wp.zeros((self.count, 6), dtype=int, device=self.device)
        # Step-start poses and the inertial predictor's root entries.
        for name in ("tn", "predicted_t", "step_t", "omega", "v", "w"):
            setattr(self, name, wp.zeros(self.count, dtype=self.vector, device=self.device))
        self.quat_n = wp.zeros(self.count, dtype=self.quaternion, device=self.device)
        # Increments of states the solver does not own, derived from their poses.
        self.external_steps = (
            wp.zeros(self.count, dtype=self.vector, device=self.device),
            wp.zeros(self.count, dtype=self.quaternion, device=self.device),
        )
        self.mirror_q = wp.zeros((self.count, 7), dtype=self.dtype, device=self.device)
        self.mirror_v = wp.zeros((self.count, 6), dtype=self.dtype, device=self.device)
        self.poses = wp.zeros(self.count, dtype=self.transform, device=self.device)
        self.params = array(self._drive_parameters(model, implicit_pd), self.dtype)
        # Local tree rows and flat tangent entries of each root's 6 x 6 block.
        local = k.root_rows.numpy().reshape(-1, 6)
        self.local_rows = array(local, int)
        self.local_starts = array(local[:, 0] if self.count else np.empty(0, dtype=np.int32), int)
        offsets, starts, sizes = [], [], []
        offset = start = 0
        for _, _, size in k._tree_sizes:
            offsets.append(offset), starts.append(start), sizes.append(size)
            offset, start = offset + size * size, start + size
        entries = np.zeros((self.count, 36), dtype=np.int32)
        for f, row in enumerate(local):
            tree = int(np.searchsorted(np.asarray(starts), row[0], side="right") - 1)
            for i in range(6):
                for j in range(6):
                    entries[f, 6 * i + j] = (
                        offsets[tree] + (row[i] - starts[tree]) * sizes[tree] + row[j] - starts[tree]
                    )
        self.entries = array(entries, int)
        self.target_starts = array(model.joint_target_q_start.numpy()[joints], int)
        self.pose = wp.zeros((self.count, 7), dtype=self.dtype, device=self.device)
        self.rate = wp.zeros((self.count, 6), dtype=self.dtype, device=self.device)
        for name in ("force", "torque", "error_t", "error_r", "target_v", "target_w"):
            setattr(self, name, wp.zeros(self.count, dtype=self.vector, device=self.device))
        self.energy = wp.zeros(self.count, dtype=self.dtype, device=self.device)
        self.status = wp.zeros(self.count, dtype=int, device=self.device)
        self.has_torque = False

    def _root_name(self, f):
        joint = int(self.joint_indices[f])
        labels = getattr(self.model, "joint_label", None)
        label = labels[joint] if labels is not None and joint < len(labels) else None
        return repr(label) if label else str(joint)

    def _drive_parameters(self, model, implicit_pd):
        """Per root [k_t, d_t, F_max, k_r, d_r, tau_max]; gains and limits are isotropic per triple."""
        params = np.zeros((self.count, 6), dtype=np.float64)
        if not self.count:
            return params
        if model.joint_target_q.size != model.joint_coord_count:
            raise ValueError(
                "Floating roots require coordinate-layout joint targets (target_q holds the 7 root pose "
                "coordinates); set newton.use_coord_layout_targets = True before building the model"
            )
        if not implicit_pd:
            return params
        values = {
            name: getattr(model, name).numpy()
            for name in ("joint_target_ke", "joint_target_kd", "joint_effort_limit", "joint_target_mode")
        }
        for f, dofs in enumerate(self.dof_indices):
            name = self._root_name(f)
            for part, rows in (("linear", dofs[:3]), ("angular", dofs[3:])):
                kp, kd, limit = (values[n][rows] for n in ("joint_target_ke", "joint_target_kd", "joint_effort_limit"))
                for field, triple in (("target_ke", kp), ("target_kd", kd), ("effort_limit", limit)):
                    if not np.isfinite(triple).all() or np.any(triple != triple[0]):
                        raise ValueError(
                            f"Floating root joint {name}: {part} {field} must be finite and equal on its three "
                            f"DOFs (isotropic 6D drive), got {triple.tolist()}"
                        )
                if kp[0] < 0 or kd[0] < 0:
                    raise ValueError(f"Floating root joint {name}: {part} drive gains must be nonnegative")
                if kp[0] > 0 or kd[0] > 0:
                    if np.any(values["joint_target_mode"][rows] != int(JointTargetMode.POSITION_VELOCITY)):
                        raise ValueError(
                            f"Floating root joint {name}: a driven {part} triple requires POSITION_VELOCITY mode"
                        )
                    if limit[0] <= 0:
                        raise ValueError(
                            f"Floating root joint {name}: a driven {part} triple requires a positive effort limit"
                        )
                offset = 0 if part == "linear" else 3
                params[f, offset : offset + 3] = kp[0], kd[0], limit[0]
        return params

    def snapshot(self, control, objective):
        """Freeze root targets and the root wrench; zero the root's scalar joint forces."""
        if not self.count:
            return
        wp.launch(
            _snapshot_controls,
            self.count,
            [
                control.joint_target_q,
                control.joint_target_qd,
                control.joint_f,
                self.target_starts,
                self.vs,
                self.local_starts,
                self.params,
                self.pose,
                self.rate,
                self.force,
                self.torque,
                objective.joint_force,
                self.status,
            ],
            device=self.device,
        )
        # One blocking read per step (floating scenes only): the per-root status bits.
        status = self.status.numpy()
        bad = _NONFINITE_TARGET_Q | _NONFINITE_TARGET_QD | _NONFINITE_JOINT_F | _NONUNIT_TARGET
        for f in np.flatnonzero(status & bad):
            bits, name = int(status[f]), self._root_name(int(f))
            for bit, field in (
                (_NONFINITE_TARGET_Q, "joint_target_q"),
                (_NONFINITE_TARGET_QD, "joint_target_qd"),
                (_NONFINITE_JOINT_F, "joint_f"),
            ):
                if bits & bit:
                    raise ValueError(f"Floating root joint {name}: nonfinite {field}")
            if bits & _NONUNIT_TARGET:
                raise ValueError(
                    f"Floating root joint {name}: a driven root's target quaternion must be unit (within 1e-5)"
                )
        self.has_torque = bool(np.any(status & _ROOT_TORQUE))

    def begin_drive(self):
        """Fix the drive's start errors and world target rates at the step start."""
        if not self.count:
            return
        wp.launch(
            _drive_start,
            self.count,
            [
                self.pose,
                self.rate,
                self.xp,
                self.xc,
                self.local_com,
                self.tn,
                self.quat_n,
                self.error_t,
                self.error_r,
                self.target_v,
                self.target_w,
            ],
            device=self.device,
        )

    def drive_terms(self, steps, objective, dt_buffer, build_tangent):
        """Add the root drive and wrench to the tree objective's energy, residual and tangent."""
        if not self.count:
            return
        wp.launch(
            _drive_terms,
            self.count,
            [
                *steps,
                self.params,
                self.error_t,
                self.error_r,
                self.target_v,
                self.target_w,
                self.force,
                self.torque,
                self.local_rows,
                self.entries,
                dt_buffer,
                build_tangent,
                self.energy,
                objective.residual,
                objective.absolute,
                objective.tangent.flatten(),
            ],
            device=self.device,
        )

    def bind_rows(self, local_rows):
        """Bind each root's six global rows from the tree-local row map."""
        rows = np.asarray(local_rows)[self.kinematics.root_rows.numpy().reshape(-1)].reshape(-1, 6)
        self.rows.assign(rows.astype(np.int32))

    def initialize(self, state):
        """Set native root poses from ``joint_q``; rates stay in the articulation history."""
        if not self.count:
            return
        q = state.joint_q.numpy()[self.coord_indices]
        qd = state.joint_qd.numpy()[self.dof_indices]
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Floating root coordinates and rates must be finite")
        if np.any(abs(np.linalg.norm(q[:, 3:], axis=1) - 1) > 1e-5):
            raise ValueError("Floating root coordinates require unit quaternions")
        history = state._monolithic_coupled
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
                history.floating_com,
                history.floating_rotation,
                self.v,
                self.w,
            ],
            device=self.device,
        )
        self.publish(state, rates=False)

    def publish(self, state, *, rates=True):
        """Write the ``joint_q`` (and, after a step, ``joint_qd``) mirrors of the native roots."""
        if not self.count:
            return
        history = state._monolithic_coupled
        if rates:
            wp.launch(
                _gather_root_rates,
                self.count,
                [self.bodies, history.link_v_com, history.link_omega, self.v, self.w],
                device=self.device,
            )
        wp.launch(
            _mirrors,
            self.count,
            [
                history.floating_com,
                history.floating_rotation,
                self.v,
                self.w,
                self.xp,
                self.xc,
                self.local_com,
                self.mirror_q,
                self.mirror_v,
                self.poses,
            ],
            device=self.device,
        )
        wp.launch(
            _scatter_mirrors,
            self.count,
            [self.mirror_q, self.mirror_v, self.qs, self.vs, int(rates), state.joint_q, state.joint_qd],
            device=self.device,
        )

    def validate(self, state):
        """Require unit native rotations, unchanged bound poses and agreeing ``joint_q`` mirrors."""
        if not self.count:
            return
        history = state._monolithic_coupled
        tolerance = 1e-10 if self.dtype == wp.float64 else 1e-6
        if np.any(abs(np.linalg.norm(history.floating_rotation.numpy(), axis=1) - 1) > tolerance):
            raise ValueError("Invalid native floating-root quaternion")
        for name in ("com", "rotation"):
            if not np.array_equal(
                getattr(history, "floating_" + name).numpy(), getattr(history, "floating_bound_" + name).numpy()
            ):
                raise ValueError("Floating root pose changed; explicitly reset history")
        wp.launch(
            _mirrors,
            self.count,
            [
                history.floating_com,
                history.floating_rotation,
                self.v,
                self.w,
                self.xp,
                self.xc,
                self.local_com,
                self.mirror_q,
                self.mirror_v,
                self.poses,
            ],
            device=self.device,
        )
        q, expected = state.joint_q.numpy()[self.coord_indices], self.mirror_q.numpy()
        # Quaternion signs encode the same rotation; the pose is bound exactly by the native state.
        sign = np.where(np.sum(q[:, 3:] * expected[:, 3:], axis=1) < 0, -1, 1)
        expected[:, 3:] *= sign[:, None]
        if not np.allclose(q, expected, atol=tolerance, rtol=tolerance):
            raise ValueError("Floating-root joint_q mirror disagrees with native root pose")

    def begin_step(self, state_in, accepted, steps, objective, dt_buffer):
        """Freeze the step start and predict each root as a rigid body from its link history."""
        if not self.count:
            return
        start, history = state_in._monolithic_coupled, accepted._monolithic_coupled
        wp.copy(self.tn, start.floating_com)
        wp.copy(self.quat_n, start.floating_rotation)
        wp.launch(
            _gather_root_prediction,
            self.count,
            [
                self.bodies,
                objective.predicted_t,
                objective.step_t,
                start.link_omega,
                self.predicted_t,
                self.step_t,
                self.omega,
            ],
            device=self.device,
        )
        wp.launch(
            _predict_pose,
            self.count,
            [
                self.predicted_t,
                self.step_t,
                self.quat_n,
                self.omega,
                dt_buffer,
                history.floating_com,
                history.floating_rotation,
                *steps,
            ],
            device=self.device,
        )

    def trial(self, source_steps, target, target_steps, direction, alpha_buffer):
        if not self.count:
            return
        history = target._monolithic_coupled
        wp.launch(
            _trial,
            self.count,
            [
                *source_steps,
                self.rows,
                direction,
                alpha_buffer,
                self.tn,
                self.quat_n,
                *target_steps,
                history.floating_com,
                history.floating_rotation,
            ],
            device=self.device,
        )

    def derive_steps(self, state):
        """Return the increments of a state the solver does not own from its rounded pose."""
        if self.count:
            history = state._monolithic_coupled
            wp.launch(
                _derive_steps,
                self.count,
                [history.floating_com, history.floating_rotation, self.tn, self.quat_n, *self.external_steps],
                device=self.device,
            )
        return self.external_steps

    def difference(self, steps_a, steps_b, out):
        """Overwrite the root rows of ``out`` with the increment from ``a`` to ``b``."""
        if self.count:
            wp.launch(_pose_difference, self.count, [*steps_a, *steps_b, self.rows, out], device=self.device)

    def coordinates(self, steps, out):
        """Overwrite the root rows of ``out`` with coordinate magnitudes of the increments."""
        if self.count:
            wp.launch(_pose_coordinates, (self.count, 6), [*steps, self.rows, out], device=self.device)
