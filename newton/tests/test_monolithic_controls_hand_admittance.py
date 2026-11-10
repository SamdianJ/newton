# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check virtual tendon units, signed feedback, and transactional updates."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.monolithic_reference.hand_admittance import HandAdmittance
from scripts.monolithic_reference.run_hand_bottle import finger_feedback, lift_target


class TestHandAdmittance(unittest.TestCase):
    def test_lift_force_is_not_finger_torque(self):
        """A vertical support force must not alter the finger admittance input."""
        outputs = []
        for lift_force in (0, 100):
            forces = self.feedback(np.r_[lift_force, np.linspace(-0.02, 0, 20)])
            outputs.append(self.controller().command("hold", finger_feedback(forces), 0.005, target_force=2))
        np.testing.assert_array_equal(*outputs)
        for t in np.linspace(4.1, 5.9, 9):
            p, v = lift_target(t)
            numerical = (lift_target(t + 1e-5)[0] - lift_target(t - 1e-5)[0]) / 2e-5
            self.assertAlmostEqual(v, numerical, places=8)
            self.assertGreater(p, 0)
            self.assertGreater(v, 0)
        self.assertEqual(lift_target(0), (0, 0))
        np.testing.assert_allclose(lift_target(7), (0.08, 0), atol=1e-15)

    def controller(self):
        """Construct a hand with limits covering its open pose and curl."""
        return HandAdmittance(np.full(20, -1.0), np.full(20, 1.5))

    def feedback(self, torque=None, *, generation=1, valid=True):
        """Make an independent generalized contact observation."""
        return SimpleNamespace(
            joint_force=np.zeros(20) if torque is None else torque, force_valid=valid, force_generation=generation
        )

    def test_power_and_independent_opposing_feedback(self):
        """Preserve virtual power and reverse only the overloaded finger."""
        controller = self.controller()
        torque = np.linspace(-0.02, 0.03, 20)
        velocity = np.array([0.01, -0.02, 0.005, 0.03])
        self.assertAlmostEqual(
            torque @ (controller.mapping @ velocity), -controller.loads(self.feedback(torque)) @ velocity, places=14
        )
        for generation in range(1, 51):
            controller.command("squeeze", self.feedback(generation=generation), 0.005, target_force=2)
        torque = np.zeros(20)
        torque[1:4] = -0.1  # Opposing torques exceed the target even during deceleration.
        _, qd = controller.command("hold", self.feedback(torque, generation=51), 0.005, target_force=2)
        self.assertLess(qd[1], 0)
        self.assertGreater(qd[5], 0)
        np.testing.assert_array_equal(qd[[0, 4, 8, 12, 16, 17, 18, 19]], 0)

    def test_invalid_feedback_and_restore_are_atomic(self):
        """Reject stale or malformed feedback without changing any virtual state."""
        controller = self.controller()
        before = controller.snapshot()
        expected = controller.command("squeeze", self.feedback(), 0.005, target_force=2)
        controller.restore(before)
        np.testing.assert_array_equal(expected, controller.command("squeeze", self.feedback(), 0.005, target_force=2))
        before = controller.snapshot()
        for observation in (
            self.feedback(),
            self.feedback(valid=False, generation=2),
            self.feedback(np.full(20, np.nan), generation=2),
        ):
            with self.assertRaises(ValueError):
                controller.command("squeeze", observation, 0.005, target_force=2)
            self.assertEqual(controller.snapshot(), before)

    def test_release_and_precision(self):
        """Release to open without windup and keep integration independent of sensor dtype."""
        controllers = [self.controller(), self.controller()]
        for generation in range(1, 101):
            outputs = [
                c.command("squeeze", self.feedback(np.zeros(20, dtype=d), generation=generation), 0.005, target_force=2)
                for c, d in zip(controllers, (np.float32, np.float64), strict=True)
            ]
            np.testing.assert_array_equal(*outputs)
        controller = controllers[0]
        for generation in range(101, 501):
            q, qd = controller.command("release", self.feedback(generation=generation), 0.005, target_force=0)
        np.testing.assert_array_equal(q, controller.opening)
        np.testing.assert_array_equal(qd, 0)


if __name__ == "__main__":
    unittest.main()
