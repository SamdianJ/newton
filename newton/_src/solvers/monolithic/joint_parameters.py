# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen scalar joint parameters and control snapshots shared by the P3 solver."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import JointTargetMode, JointType, Model


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


class JointParameters:
    """Validate and freeze model parameters and per-step scalar joint targets."""

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
        self.status = wp.zeros(1, dtype=int, device=model.device)

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
