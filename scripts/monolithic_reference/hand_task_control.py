# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-responsive hand targets driven only by previously published observations."""

import copy
from dataclasses import dataclass

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def finger_index(name):
    """Map a retained Sharpa link/joint label to its finger, or None for the palm."""
    return next((i for i, finger in enumerate(FINGERS) if f"_{finger}_" in name), None)


def contact_loads(pairs, hand_shapes, object_shapes, ground_shapes, *, material, generation):
    """Separate published object grip loads from world support reactions [N].

    Finger loads sum pair-force norms, not certified normal loads or force closure.
    """
    fingers, floor = np.zeros(5), np.zeros(5)
    support = hand_object = 0.0
    for pair in pairs:
        if pair.get("force_generation") != generation:
            raise ValueError("Contact feedback requires matching published force generations")
        shapes = (pair["source"], pair["target"])
        wrench = np.asarray(pair["world_wrench"], dtype=float)
        if wrench.shape != (2, 6) or not np.isfinite(wrench).all():
            raise ValueError("Contact feedback requires finite pair wrenches")
        for side, shape in enumerate(shapes):
            other = shapes[1 - side]
            is_object = shape in object_shapes or (material and shape == -1)
            if is_object and other in ground_shapes:
                support += max(0.0, float(wrench[side, 2]))
            if shape in hand_shapes and (other in object_shapes or (material and other == -1)):
                hand_object += float(np.linalg.norm(wrench[side, :3]))
            finger = hand_shapes.get(shape)
            if finger is not None:
                load = float(np.linalg.norm(wrench[side, :3]))
                if other in object_shapes or (material and other == -1):
                    fingers[finger] += load
                elif other in ground_shapes:
                    floor[finger] += load
    return fingers, floor, support, hand_object


@dataclass
class HandObservation:
    """Published feedback in world coordinates, with distances [m] and forces [N]."""

    generation: int
    finger_force: np.ndarray
    finger_ground_force: np.ndarray
    finger_penetration: np.ndarray
    support_force: float
    object_position: np.ndarray
    object_velocity: np.ndarray
    object_bottom: float
    valid: bool = True
    open_error_rad: float = 0.0
    hand_object_force: float = 0.0
    hand_object_penetration: float = 0.0

    def snapshot(self):
        """Return the observation needed to replay the next controller update."""
        return {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in vars(self).items()}

    @classmethod
    def from_snapshot(cls, payload):
        """Restore a previously serialized published observation."""
        values = dict(payload)
        for name in ("finger_force", "finger_ground_force", "finger_penetration", "object_position", "object_velocity"):
            values[name] = np.asarray(values[name], dtype=float)
        return cls(**values)


class ContactHandControl:
    """Close fingers independently, verify grip, move, unload on support, and release."""

    def __init__(self, task, *, mass, friction, source_rates):
        if task not in ("grasp-rigid", "grasp-soft", "pull-slider"):
            raise ValueError("Unknown hand task")
        if mass <= 0 or friction < 0 or not np.isfinite([mass, friction]).all():
            raise ValueError("Require positive mass and nonnegative finite friction")
        self.task, self.mass, self.friction = task, mass, friction
        self.max_rate = np.minimum(np.asarray(source_rates, dtype=float), 1.2)
        if self.max_rate.shape != (5,) or not np.isfinite(self.max_rate).all() or np.any(self.max_rate <= 0):
            raise ValueError("Require five positive finite closure-source speed limits")
        self.required_load = 1.5 * (0.1 if task == "pull-slider" else mass * 9.81) / max(friction, 1e-12)
        self.sources, self.rates = np.zeros(5), np.zeros(5)
        self.carriage = self.carriage_velocity = 0.0
        self.phase, self.phase_start = "settle", 0.0
        self.last_time, self.last_generation = 0.0, 0
        self.stable_time = 0.0
        self.lost_time = 0.0
        self.move_start = 0.0
        self.move_end = 0.05
        self.brake_position = self.brake_velocity = 0.0
        self.failure = None
        self.completed = False
        self.floor_blocked = np.zeros(5, dtype=bool)
        self.initial_position = None
        self.events = []

    @property
    def parameters(self):
        return {
            "required_pair_load_N": self.required_load,
            "load_factor": 1.5,
            "finger_penetration_guard_m": 0.001,
            "contact_detection_N": 0.01,
            "support_weight_fraction": 0.25,
            "stable_duration_s": 0.1,
            "release_pair_load_N": 0.01,
            "release_penetration_m": 0.00025,
            "release_open_error_rad": 0.02,
            "close_deadline_s": 2.75,
            "source_acceleration_s_inv": 24.0,
            "source_max_rates": self.max_rate.tolist(),
            "load_interpretation": "sum of hand-object pair-force norms; physical task gates remain independent",
        }

    def transition(self, phase, t):
        self.phase, self.phase_start = phase, t
        self.stable_time = 0.0
        self.lost_time = 0.0
        self.events.append({"time": t, "phase": phase, "failure": self.failure})

    def snapshot(self):
        """Save phase, debounce, target progress and the last consumed generation."""
        return {
            name: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for name, value in vars(self).items()
        }

    @classmethod
    def from_snapshot(cls, payload):
        """Restore exact controller state without advancing its clock."""
        result = cls(
            payload["task"], mass=payload["mass"], friction=payload["friction"], source_rates=payload["max_rate"]
        )
        if set(payload) != set(vars(result)):
            raise ValueError("Controller snapshot has missing or unexpected fields")
        for name, raw_value in payload.items():
            value = copy.deepcopy(raw_value)
            current = getattr(result, name)
            if isinstance(current, np.ndarray):
                value = np.asarray(value, dtype=current.dtype)
                if value.shape != current.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid controller snapshot field {name}")
            if name == "initial_position" and value is not None:
                value = np.asarray(value, dtype=float)
            setattr(result, name, value)
        return result

    @staticmethod
    def blend(elapsed, duration):
        u = float(np.clip(elapsed / duration, 0.0, 1.0))
        return u**3 * (10 - 15 * u + 6 * u**2), 30 * u**2 * (1 - u) ** 2 / duration

    def advance(self, t, observation):
        """Advance once before the next simulation step; never consumes a trial state."""
        dt = t - self.last_time
        if not np.isfinite(t) or not 0 < dt <= 0.01:
            raise ValueError("Feedback control requires increasing times with steps at most 10 ms")
        self.last_time = t
        if observation is None:
            if t > 0.25:
                self.failure = "missing-feedback"
                self.transition("feedback-invalid", t)
            self.rates.fill(0)
            self.carriage_velocity = 0.0
            return
        arrays = (
            observation.finger_force,
            observation.finger_ground_force,
            observation.finger_penetration,
            observation.object_position,
            observation.object_velocity,
        )
        valid = (
            observation.valid
            and observation.generation > self.last_generation
            and all(
                np.shape(value) == shape for value, shape in zip(arrays, ((5,), (5,), (5,), (3,), (3,)), strict=True)
            )
            and all(np.isfinite(value).all() for value in arrays)
            and all(np.min(value) >= 0 for value in arrays[:3])
            and observation.support_force >= 0
            and np.isfinite(
                [
                    observation.support_force,
                    observation.object_bottom,
                    observation.open_error_rad,
                    observation.hand_object_force,
                    observation.hand_object_penetration,
                ]
            ).all()
            and observation.open_error_rad >= 0
            and observation.hand_object_force >= 0
            and observation.hand_object_penetration >= 0
        )
        if not valid or self.phase == "feedback-invalid":
            self.failure = self.failure or "stale-or-invalid-feedback"
            if self.phase != "feedback-invalid":
                self.transition("feedback-invalid", t)
            self.rates.fill(0)
            self.carriage_velocity = 0.0
            return
        self.last_generation = observation.generation
        if self.initial_position is None:
            self.initial_position = np.array(observation.object_position, copy=True)
        force = np.asarray(observation.finger_force)
        near = np.asarray(observation.finger_penetration) >= 0.001
        self.floor_blocked |= np.asarray(observation.finger_ground_force) >= 0.01
        grip = (
            self.friction > 0
            and force[0] >= 0.2 * self.required_load
            and force[1:].sum() >= 0.2 * self.required_load
            and force.sum() >= self.required_load
            and not near.any()
        )
        if self.phase in ("move", "hold"):
            contact_remains = force[0] >= 0.01 and force[1:].sum() >= 0.01
            self.lost_time = 0.0 if contact_remains else self.lost_time + dt
            if self.lost_time >= 0.1:
                self.failure = "grip-lost"
                self.brake_position, self.brake_velocity = self.carriage, self.carriage_velocity
                self.transition("abort-move", t)
        if self.phase == "settle" and t >= 0.25:
            self.transition("close", t)
        if self.phase == "close":
            stable = grip and np.linalg.norm(observation.object_velocity) <= 0.01
            self.stable_time = self.stable_time + dt if stable else 0.0
            if self.stable_time >= 0.1:
                self.move_start = self.carriage
                if self.task == "pull-slider":
                    self.move_end = self.carriage + self.initial_position[0] + 0.05 - observation.object_position[0]
                self.transition("move", t)
            elif t >= 2.75:
                self.failure = "grip-not-established"
                self.transition("release", t)
        elif self.phase == "move" and t - self.phase_start >= 1.0:
            self.transition("hold", t)
        elif self.phase == "hold" and t - self.phase_start >= 0.5:
            if self.task == "pull-slider":
                at_target = abs(observation.object_position[0] - self.initial_position[0] - 0.05) <= 0.01
                if at_target and abs(observation.object_velocity[0]) <= 0.01:
                    self.transition("release", t)
                elif t - self.phase_start >= 1.0:
                    self.failure = "slider-not-stable-at-target"
                    self.transition("release", t)
            else:
                if observation.object_position[2] - self.initial_position[2] < 0.03:
                    self.failure = "object-not-lifted"
                self.move_start = self.carriage
                self.transition("lower", t)
        elif self.phase == "lower":
            supported = (
                observation.support_force >= 0.25 * self.mass * 9.81
                and observation.object_bottom <= 0.0005
                and abs(observation.object_velocity[2]) <= 0.01
            )
            if supported:
                self.brake_position, self.brake_velocity = self.carriage, self.carriage_velocity
                self.transition("unload", t)
            elif t - self.phase_start >= 1.5:
                self.failure = self.failure or "support-not-established"
                self.transition("support-wait", t)
        elif self.phase == "unload" and t - self.phase_start >= 0.2:
            if observation.support_force >= 0.25 * self.mass * 9.81:
                self.transition("release", t)
        elif self.phase == "abort-move" and t - self.phase_start >= 0.08:
            self.transition("release", t)
        elif self.phase == "release":
            released = (
                np.max(self.sources) <= 1e-12
                and max(force.sum(), observation.hand_object_force) <= 0.01
                and max(np.max(observation.finger_penetration), observation.hand_object_penetration) <= 0.00025
                and observation.open_error_rad <= 0.02
                and np.linalg.norm(observation.object_velocity) <= 0.01
            )
            self.stable_time = self.stable_time + dt if released else 0.0
            if self.stable_time >= 0.1:
                self.completed = self.failure is None
                self.transition("observe", t)

        elapsed = t - self.phase_start
        if self.phase == "move":
            b, v = self.blend(elapsed, 1.0)
            self.carriage, self.carriage_velocity = (
                self.move_start + (self.move_end - self.move_start) * b,
                (self.move_end - self.move_start) * v,
            )
        elif self.phase == "lower":
            b, v = self.blend(elapsed, 1.0)
            self.carriage, self.carriage_velocity = self.move_start * (1 - b), -self.move_start * v
        elif self.phase in ("unload", "abort-move"):
            u = min(elapsed / 0.08, 1.0)
            self.carriage = self.brake_position + self.brake_velocity * 0.08 * (u - 0.5 * u**2)
            self.carriage_velocity = self.brake_velocity * (1 - u)
        else:
            self.carriage_velocity = 0.0

        desired = np.zeros(5)
        if self.phase in ("close", "move", "hold", "lower"):
            target = max(0.025, 0.5 * self.required_load)
            desired = self.max_rate * np.clip(1 - force / target, -0.25, 1.0)
            desired = np.where(force >= 0.01, np.minimum(desired, 0.25), desired)
            desired = np.where(self.floor_blocked, np.minimum(desired, 0.0), desired)
            desired = np.where(near | (np.asarray(observation.finger_ground_force) > 0.02), -0.25, desired)
        elif self.phase in ("unload", "release"):
            desired = -self.max_rate if self.phase == "release" else np.full(5, -0.25)
        desired = np.clip(desired, -np.sqrt(48 * self.sources), np.sqrt(48 * (3 - self.sources)))
        self.rates += np.clip(desired - self.rates, -24 * dt, 24 * dt)
        previous = self.sources.copy()
        self.sources = np.clip(self.sources + dt * self.rates, 0.0, 3.0)
        self.rates = (self.sources - previous) / dt
