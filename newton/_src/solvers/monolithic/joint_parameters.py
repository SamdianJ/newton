# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen scalar joint parameters and control snapshots shared by the coupled solver."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import JointTargetMode, JointType, Model
from .actors import _joint_layout


def _validate_joint_coordinate_layout(model: Model, actor=None) -> np.ndarray:
    """Return mechanical DOF-to-Model coordinates, keeping FREE outside scalar terms."""
    types, starts, _, _ = _joint_layout(model)
    joints = np.arange(model.joint_count) if actor is None else np.asarray(actor.joint_indices, dtype=np.int32)
    if not np.isin(types[joints], [JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC]).all():
        raise ValueError("Monolithic supports only FIXED, REVOLUTE and PRISMATIC joints")
    return starts[joints[types[joints] != JointType.FIXED]].astype(np.int32)


@wp.kernel(enable_backward=False)
def _validate_joint_term_inputs(
    q_target: wp.array[float],
    v_target: wp.array[float],
    joint_f: wp.array[float],
    target_map: wp.array[int],
    dof_map: wp.array[int],
    driven: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    dof = dof_map[i]
    if not wp.isfinite(q_target[target_map[i]]) or not wp.isfinite(v_target[dof]) or not wp.isfinite(joint_f[dof]):
        wp.atomic_max(status, 0, 1)
    if driven[i] != 0 and joint_f[dof] != 0.0:
        wp.atomic_max(status, 0, 2)


@wp.kernel(enable_backward=False)
def _gather_joint_targets(
    q: wp.array[float],
    v: wp.array[float],
    targets: wp.array[int],
    dofs: wp.array[int],
    local_q: wp.array[float],
    local_v: wp.array[float],
):
    i = wp.tid()
    local_q[i] = q[targets[i]]
    local_v[i] = v[dofs[i]]


class JointParameters:
    """Validate and freeze model parameters and per-step scalar joint targets."""

    def __init__(
        self,
        model: Model,
        *,
        actor=None,
        implicit_pd: bool = False,
        limits: bool = False,
        friction: bool = False,
        limit_width: tuple[float, ...] | None = None,
        friction_velocity_scale: tuple[float, ...] | None = None,
    ):
        for value in (implicit_pd, limits, friction):
            if type(value) is not bool:
                raise ValueError("Joint term switches must be bool")
        coordinates = _validate_joint_coordinate_layout(model, actor)
        dofs = tuple(range(model.joint_dof_count)) if actor is None else actor.dof_indices
        self.model, self.device, self.count = model, model.device, len(dofs)
        self.dof_map = wp.array(dofs, dtype=int, device=model.device)
        self.implicit_pd = implicit_pd
        nq = self.count
        allowed = set()

        def read(name):
            array = getattr(model, name)
            if array is None or (array.shape, array.dtype, array.device) != (
                (model.joint_dof_count,),
                wp.float32,
                model.device,
            ):
                raise ValueError(f"Invalid joint parameter array: {name}")
            values = array.numpy()[list(dofs)]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite joint parameter: {name}")
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
            if modes is None or (modes.shape, modes.dtype, modes.device) != (
                (model.joint_dof_count,),
                wp.int32,
                model.device,
            ):
                raise ValueError("Invalid joint_target_mode")
            mode = modes.numpy()[list(dofs)]
            if np.any(mode[driven != 0] != int(JointTargetMode.POSITION_VELOCITY)) or np.any(
                ~np.isin(
                    mode[driven == 0], [JointTargetMode.NONE, JointTargetMode.EFFORT, JointTargetMode.POSITION_VELOCITY]
                )
            ):
                raise ValueError("Implicit PD requires POSITION_VELOCITY mode")
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
        target_ids = coordinates if actor is None else actor.target_indices
        self.target_map = wp.array(target_ids, dtype=int, device=model.device)
        self.params = wp.array(parameters, dtype=float, device=model.device)
        self.driven = wp.array(driven, dtype=int, device=model.device)
        self.target_q = wp.zeros(nq, dtype=float, device=model.device)
        self.target_qd = wp.zeros(nq, dtype=float, device=model.device)
        self.status = wp.zeros(1, dtype=int, device=model.device)

    def snapshot_targets(self, control) -> None:
        """Validate before copying inputs; no public/candidate state is mutated."""
        if not self.implicit_pd:
            return
        for name, count in (
            ("joint_target_q", self.model.joint_target_q.size),
            ("joint_target_qd", self.model.joint_dof_count),
            ("joint_f", self.model.joint_dof_count),
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
                self.dof_map,
                self.driven,
                self.status,
            ],
            device=self.device,
        )
        code = int(self.status.numpy()[0])
        if code:
            raise ValueError("Duplicate driven joint_f" if code == 2 else "Nonfinite joint control")
        wp.launch(
            _gather_joint_targets,
            self.count,
            [
                control.joint_target_q,
                control.joint_target_qd,
                self.target_map,
                self.dof_map,
                self.target_q,
                self.target_qd,
            ],
            device=self.device,
        )
