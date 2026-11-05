# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental, power-conjugate finger curl admittance for HumanHand20DOF."""

import numpy as np

from scripts.monolithic_reference.admittance_control import AxisAdmittance


class HandAdmittance:
    """Map four virtual tendon travels [m] to flexion angles [rad].

    q = q_open + A*x, where A [rad/m] uses a 50 mm virtual lever, not a
    measured fingertip Jacobian. The resisting force is -A.T @ tau_contact
    [N], so tau_contact @ qdot = -load @ xdot exactly. Target forces therefore
    refer to virtual tendons, not fingertip normal-force magnitudes.
    """

    def __init__(self, lower, upper):
        self.opening = np.zeros(20)
        self.opening[17] = -0.65  # Retract the thumb clear of both bottle and table.
        self.mapping = np.zeros((20, 4))
        for finger in range(4):
            self.mapping[4 * finger + 1 : 4 * finger + 4, finger] = 20.0
        lower, upper = np.asarray(lower), np.asarray(upper)
        if lower.shape != (20,) or upper.shape != (20,) or not np.all(np.isfinite([lower, upper])):
            raise ValueError("Require finite HumanHand20DOF limits")
        if np.any(lower > self.opening) or np.any(upper < self.opening):
            raise ValueError("Open hand lies outside joint limits")
        self.axes = []
        for column in self.mapping.T:
            active = column > 0
            travel = min(0.065, float(np.min(upper[active] / column[active])))
            self.axes.append(AxisAdmittance(mass=0.2, damping=100.0, position_limits=(0, travel), max_speed=0.02))
        self.last_generation = -1

    def snapshot(self):
        """Return integrator state and last consumed publication for rollback."""
        return self.last_generation, tuple((axis.position, axis.velocity) for axis in self.axes)

    def restore(self, snapshot):
        """Restore the state preceding a rejected physics step."""
        self.last_generation, states = snapshot
        for axis, state in zip(self.axes, states, strict=True):
            axis.reset(*state)

    def loads(self, forces):
        """Project published contact torques into resisting tendon loads [N]."""
        tau = np.asarray(forces.joint_force, dtype=float)
        if not forces.force_valid or tau.shape != (20,) or not np.all(np.isfinite(tau)):
            raise ValueError("Require valid finite hand contact torques")
        return -self.mapping.T @ tau

    def command(self, phase, forces, dt, *, target_force):
        """Advance from fresh contact feedback; release explicitly returns to open."""
        loads = self.loads(forces)
        if (
            forces.force_generation <= self.last_generation
            or not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(target_force)
            or target_force < 0
            or phase not in ("settle", "squeeze", "hold", "release")
        ):
            raise ValueError("Require fresh feedback, valid phase, positive dt and nonnegative target")
        for axis, load in zip(self.axes, loads, strict=True):
            if phase == "settle":
                axis.reset(0.0)
            else:
                release = phase == "release"
                axis.advance(
                    float((0 if release else target_force) - load),
                    dt,
                    velocity_bounds=(-0.02, -0.02) if release else None,
                )
        self.last_generation = forces.force_generation
        return (
            self.opening + self.mapping @ [axis.position for axis in self.axes],
            self.mapping @ [axis.velocity for axis in self.axes],
        )
