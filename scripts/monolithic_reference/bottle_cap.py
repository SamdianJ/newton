# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hollow cap and compliant thread surrogate against a fixed bottle-neck fixture."""

from dataclasses import dataclass

import numpy as np


def cap_geometry():
    """Return a closed cup mesh, mass [kg], COM inertia [kg m²] and seated COM [m]."""
    outer, inner, height, lid, density = 0.014, 0.012, 0.016, 0.002, 950.0
    side_mass = density * np.pi * (outer**2 - inner**2) * (height - lid)
    lid_mass = density * np.pi * outer**2 * lid
    mass = side_mass + lid_mass
    centers = np.array([(height - lid) / 2, height - lid / 2])
    com = float(np.array([side_mass, lid_mass]) @ centers / mass)
    axial = side_mass * (outer**2 + inner**2) / 2 + lid_mass * outer**2 / 2
    transverse = (
        side_mass * (3 * (outer**2 + inner**2) + (height - lid) ** 2) / 12
        + lid_mass * (3 * outer**2 + lid**2) / 12
        + np.array([side_mass, lid_mass]) @ (centers - com) ** 2
    )
    n = 48
    angle = np.arange(n) * 2 * np.pi / n
    profile = [(outer, 0), (outer, height), (inner, height - lid), (inner, 0)]
    points = np.concatenate(
        [np.c_[r * np.cos(angle), np.full(n, y - com), r * np.sin(angle)] for r, y in profile]
        + [np.array([[0, height - com, 0], [0, height - lid - com, 0]])]
    )
    faces = []
    for first, second in ((0, 1), (2, 3), (3, 0)):
        for j in range(n):
            a, b, c, d = first * n + j, first * n + (j + 1) % n, second * n + (j + 1) % n, second * n + j
            faces.extend(((a, c, b), (a, d, c)))
    for j in range(n):
        faces.extend(((4 * n, n + (j + 1) % n, n + j), (4 * n + 1, 2 * n + j, 2 * n + (j + 1) % n)))
    return points, np.array(faces, dtype=np.int32), mass, np.array([transverse, axial, transverse]), 0.068 + com


def rotation(quaternion):
    """Convert an xyzw quaternion to a rotation matrix without a Warp dependency."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class CapThread:
    """Track turns and remove finite thread forces after two turns and 6 mm travel.

    Forces are explicit, frozen external wrenches for one solver step. This is
    a fixture approximation, not an implicit screw joint or resolved tooth contact.
    Positive rotation about +Y unscrews the cap. Detached caps do not re-engage.
    """

    seated_y: float
    theta: float = 0.0
    previous_yaw: float = 0.0
    released: bool = False
    lead: float = 0.003
    turns: float = 2.0
    stiffness: float = 300.0
    damping: float = 0.5

    @staticmethod
    def orientation(pose):
        """Return yaw, its world angular-velocity gradient, and the cap axis."""
        r = rotation(pose[3:])
        x, axis = r[:, 0], r[:, 1]
        denominator = x[0] ** 2 + x[2] ** 2
        if denominator < 0.1:
            raise ValueError("Cap tilt exceeds the thread surrogate's orientation domain")
        gradient = np.array([-x[0] * x[1] / denominator, 1.0, -x[2] * x[1] / denominator])
        return float(np.arctan2(-x[2], x[0])), gradient, axis

    def observe(self, pose):
        """Commit a valid published pose; unwrap rotations and latch disengagement."""
        pose = np.asarray(pose)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 0.5:
            raise ValueError("Expected a finite rigid pose with a nonzero quaternion")
        if self.released:
            return
        yaw, _, _ = self.orientation(pose)
        delta = float(np.arctan2(np.sin(yaw - self.previous_yaw), np.cos(yaw - self.previous_yaw)))
        if abs(delta) >= np.pi / 2:
            raise ValueError("Cap rotation sampling is too coarse to unwrap safely")
        self.theta += delta
        self.previous_yaw = yaw
        if self.theta >= 2 * np.pi * self.turns and pose[1] - self.seated_y >= self.lead * self.turns - 0.0002:
            self.released = True

    def wrench(self, pose, velocity):
        """Return cap COM force/torque [N, N m]; fixture receives the opposite wrench."""
        if self.released:
            return np.zeros(6)
        _, gradient, axis = self.orientation(pose)
        pitch = self.lead / (2 * np.pi)
        error = pose[1] - self.seated_y - pitch * self.theta
        error_rate = velocity[1] - pitch * (gradient @ velocity[3:])
        force = -100 * np.asarray(pose[:3]) - 0.4 * np.asarray(velocity[:3])
        force[1] = -self.stiffness * error - self.damping * error_rate
        torque = -pitch * force[1] * gradient
        torque += 0.01 * np.cross(axis, [0, 1, 0]) - 0.00005 * (velocity[3:] - axis * (axis @ velocity[3:]))
        torque -= 0.00002 * np.tanh((gradient @ velocity[3:]) / 0.1) * gradient
        if self.theta < 0:
            torque -= (0.01 * self.theta + 0.0001 * min(0.0, gradient @ velocity[3:])) * gradient
        return np.r_[force, torque]
