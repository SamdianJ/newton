# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read the fixed-orientation, vertical-palm teleop trajectory for PR3."""

import math

import numpy as np


class RecordedGrasp:
    """Preserve keyframe timing; reject motion unsupported by the P3 carriage."""

    dt = 0.001

    def __init__(self, payload):
        if payload.get("format") != "mujoco_teleop.target_trajectory" or payload.get("version") != 1:
            raise ValueError("Unsupported teleop trajectory format")
        frame_dt = payload.get("frame_dt", 0)
        if not np.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError("Require a positive finite frame interval")
        rows = payload.get("keyframes", [])
        if len(rows) < 2:
            raise ValueError("Require at least two recorded keyframes")
        frames = [row["frame"] for row in rows]
        if any(type(frame) is not int for frame in frames) or frames[0] != 0 or np.any(np.diff(frames) <= 0):
            raise ValueError("Require strictly increasing integer frames starting at zero")
        self.time = np.asarray(frames, dtype=float) * frame_dt
        self.position = np.asarray([row["position"] for row in rows], dtype=float)
        rotation = np.asarray([row["quaternion_xyzw"] for row in rows], dtype=float)
        self.gripper = np.asarray([row["gripper_q"] for row in rows], dtype=float)
        if self.position.shape != (len(rows), 3) or rotation.shape != (len(rows), 4):
            raise ValueError("Invalid pose shape")
        if not all(np.isfinite(value).all() for value in (self.time, self.position, rotation, self.gripper)):
            raise ValueError("Nonfinite recorded target")
        norms = np.linalg.norm(rotation, axis=1)
        if np.any(norms == 0) or np.any(self.gripper < 0) or np.any(self.gripper > 1):
            raise ValueError("Invalid quaternion or gripper range")
        rotation /= norms[:, None]
        if np.any(abs(abs(rotation @ rotation[0]) - 1) > 1e-12):
            raise ValueError("Recorded rotation changes; a vertical carriage cannot replay this trajectory")
        if not np.all(self.position[:, :2] == self.position[0, :2]):
            raise ValueError("Recorded horizontal translation requires a different palm fixture")
        self.quaternion = rotation[0]
        self.duration = float(self.time[-1])
        self.steps = math.ceil(self.duration / self.dt)

    def sample(self, time_s):
        """Return position, gripper, velocity, gripper rate and recorded phase."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Target time must be finite and nonnegative")
        index = min(int(np.searchsorted(self.time, time_s, side="right") - 1), len(self.time) - 2)
        duration = self.time[index + 1] - self.time[index]
        alpha = min((time_s - self.time[index]) / duration, 1.0)
        delta = self.position[index + 1] - self.position[index]
        grip_delta = self.gripper[index + 1] - self.gripper[index]
        position = self.position[index] + alpha * delta
        grip = float(self.gripper[index] + alpha * grip_delta)
        phase = "approach" if delta[2] < 0 else "lift" if delta[2] > 0 else "close" if grip_delta < 0 else "other"
        if time_s >= self.duration:
            return position, grip, np.zeros(3), 0.0, phase
        return position, grip, delta / duration, float(grip_delta / duration), phase
