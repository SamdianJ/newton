# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental per-finger force feedback for the nine-DOF Franka scene."""

import math

import numpy as np

from scripts.monolithic_reference.admittance_control import AxisAdmittance


class FrankaFingerAdmittance:
    """Convert published opening-axis forces [N] into finger PD references [m, m/s].

    The fixed Franka layout has seven arm DOFs followed by two prismatic fingers.
    x = q_open - q is positive inward, so the resisting load is +joint_force.
    Each finger targets 1 N while grasping. Release requests an outward velocity;
    zero force alone would leave a pure mass-damper admittance at its last position.
    """

    def __init__(self, lower, upper):
        lower, self.opening = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float).copy()
        if lower.shape != (2,) or self.opening.shape != (2,):
            raise ValueError("Require two Franka finger limits")
        self.axes = [
            AxisAdmittance(mass=0.2, damping=20.0, position_limits=(0.0, float(hi - lo)), max_speed=0.05)
            for lo, hi in zip(lower, self.opening, strict=True)
        ]
        self.last_generation = -1

    def snapshot(self):
        """Return virtual state and consumed publication identity for rollback."""
        return self.last_generation, tuple((axis.position, axis.velocity) for axis in self.axes)

    def restore(self, snapshot):
        """Restore the state preceding an unsuccessful physics step."""
        self.last_generation, states = snapshot
        for axis, state in zip(self.axes, states, strict=True):
            axis.reset(*state)

    def command(self, phase, forces, dt):
        """Consume one fresh observation and return two opening references [m, m/s]."""
        loads = np.asarray(forces.joint_force)[7:]
        if (
            not forces.force_valid
            or forces.force_generation <= self.last_generation
            or loads.shape != (2,)
            or not np.all(np.isfinite(loads))
            or not math.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Require fresh finite published Franka contact feedback and positive dt")
        if phase not in ("settle", "approach", "close", "lift", "hold", "lower", "release", "retract"):
            raise ValueError("Unknown Franka task phase")
        for axis, measured_load in zip(self.axes, loads, strict=True):
            # NumPy FP32 scalar promotion must not change the virtual integrator precision.
            load = float(measured_load)
            if phase in ("settle", "approach"):
                axis.reset(0.0)
            else:
                release = phase in ("release", "retract")
                axis.advance(
                    0.0 - load if release else 1.0 - load, dt, velocity_bounds=(-0.05, -0.05) if release else None
                )
        self.last_generation = forces.force_generation
        return self.opening - [axis.position for axis in self.axes], -np.array([axis.velocity for axis in self.axes])
