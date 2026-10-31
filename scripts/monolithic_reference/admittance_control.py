# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental translational admittance for task-level reference generation."""

import math


class AxisAdmittance:
    """Integrate M*xdd + B*xd + K*(x-reference) = force_error along one axis.

    Position and velocity are commanded motion [m, m/s], not measured motion.
    The caller supplies a signed force error [N] conjugate to this coordinate.
    Backward Euler treats virtual dynamics implicitly and the force as held;
    stability of a delayed physical contact loop must be checked separately.

    Args:
        mass: Virtual mass [kg], strictly positive.
        damping: Virtual damping [N*s/m], strictly positive.
        position_limits: Allowed reference interval [m].
        max_speed: Symmetric reference speed limit [m/s].
        stiffness: Virtual spring stiffness [N/m], nonnegative.
        reference: Fixed spring rest position [m].
        position: Initial commanded position [m]; initial velocity is zero.
    """

    def __init__(self, *, mass, damping, position_limits, max_speed, stiffness=0.0, reference=0.0, position=0.0):
        lower, upper = position_limits
        values = (mass, damping, stiffness, reference, lower, upper, max_speed, position)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Admittance parameters must be finite")
        if mass <= 0 or damping <= 0 or stiffness < 0 or max_speed <= 0 or lower >= upper:
            raise ValueError("Require positive mass/damping/speed, nonnegative stiffness and ordered limits")
        self.mass, self.damping, self.stiffness = mass, damping, stiffness
        self.reference, self.position_limits, self.max_speed = reference, (lower, upper), max_speed
        self.reset(position)

    def reset(self, position, velocity=0.0):
        """Set the applied command state [m, m/s] without changing the spring reference."""
        if not all(math.isfinite(value) for value in (position, velocity)):
            raise ValueError("Command state must be finite")
        if not self.position_limits[0] <= position <= self.position_limits[1] or abs(velocity) > self.max_speed:
            raise ValueError("Command state exceeds configured limits")
        self.position, self.velocity = float(position), float(velocity)

    def advance(self, force_error, dt, *, velocity_bounds=None):
        """Return the next bounded command [m, m/s] from force error [N] and dt [s].

        Optional velocity_bounds [m/s] intersect the configured speed limit.
        A task can prohibit closing or request retreat regardless of force error.
        Position limits take priority over velocity bounds at the end of travel.
        Saturation replaces internal state; it never accumulates hidden motion.
        """
        if not math.isfinite(force_error) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Require finite force error and positive finite dt")
        low, high = (-self.max_speed, self.max_speed) if velocity_bounds is None else velocity_bounds
        if not all(math.isfinite(value) for value in (low, high)):
            raise ValueError("Velocity bounds must be finite")
        low, high = max(low, -self.max_speed), min(high, self.max_speed)
        if low > high:
            raise ValueError("Velocity bounds do not intersect the configured speed range")
        velocity = (
            self.mass * self.velocity + dt * (force_error - self.stiffness * (self.position - self.reference))
        ) / (self.mass + dt * self.damping + dt * dt * self.stiffness)
        if not math.isfinite(velocity):
            raise ValueError("Admittance update overflowed")
        velocity = min(max(velocity, low), high)
        position = min(max(self.position + dt * velocity, self.position_limits[0]), self.position_limits[1])
        # Integrate only applied motion, including partial steps at a travel stop.
        velocity = (position - self.position) / dt
        # Subtraction can round past max_speed; keep emitted states restorable.
        velocity = min(max(velocity, -self.max_speed), self.max_speed)
        self.position, self.velocity = position, velocity
        return position, velocity
