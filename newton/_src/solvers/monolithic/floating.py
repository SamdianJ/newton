# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native poses of floating-base roots: FREE world-root joints of articulated trees.

A root's unknowns are the first six rows of its tree's block: the world COM increment and the
left rotation increment, as for a FREE rigid actor. Its COM and quaternion are native coupled
state; the exact increments of every internal state are kept beside the scalar joint steps,
whose root rows stay zero. ``joint_q``/``joint_qd`` are mirrors in Newton's FREE convention.
"""

from typing import Any

import numpy as np
import warp as wp

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


class FloatingRoots:
    """Own the maps, step-start snapshot and mirrors of every floating root of the trees."""

    def __init__(self, kinematics, model):
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
