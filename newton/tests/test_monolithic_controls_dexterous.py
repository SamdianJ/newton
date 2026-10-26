# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check task controls and physical success metrics without long robot runs."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.monolithic_reference.run_dexterous_tasks import TASKS, TaskScene, targets, task_metrics


class TestDexterousControls(unittest.TestCase):
    def test_velocity_matches_positions(self):
        """Compare commanded velocities against independent central differences."""
        for task in (*TASKS, "pinch-place"):
            for t in np.linspace(0.01, 3.99, 201):
                with self.subTest(task=task, time=t):
                    h = 1e-6
                    _, velocity, _ = targets(task, t)
                    difference = (targets(task, t + h)[0] - targets(task, t - h)[0]) / (2 * h)
                    np.testing.assert_allclose(velocity, difference, atol=1e-8, rtol=1e-6)

    def test_release_and_end_hold(self):
        """Require fingers to reopen and settle at the end of each task."""
        for task in (*TASKS, "pinch-place"):
            initial = targets(task, 0)[0]
            held = targets(task, 2.5)[0]
            final, velocity, _ = targets(task, 4.0)
            self.assertGreater(np.linalg.norm(final[1] - final[0]), np.linalg.norm(held[1] - held[0]))
            self.assertAlmostEqual(np.linalg.norm(initial[1] - initial[0]), np.linalg.norm(final[1] - final[0]))
            np.testing.assert_array_equal(velocity, 0)

    def test_static_object_fails_tasks(self):
        """Reject a stationary object even when actuator contact forces are present."""
        for task in (*TASKS, "pinch-place"):
            row = {
                "finger_contact_force_N": 1.0,
                "object_pose": [0, 0, 0.015, 0, 0, 0, 1],
                "knob_angle_rad": 0.0,
                "grab_displacement_m": 0.0,
                "edge_stretch_max": 1.0,
                "phase": "move",
                "knob_contact_torque_Nm": 1e-4,
                "sample_penetration_m": 0.0005,
                "pair_force_imbalance_N": 0.0,
                "pair_moment_imbalance_Nm": 0.0,
                "knob_active_force_Nm": 0.0,
            }
            _, gates = task_metrics(SimpleNamespace(task=task), [row] * 2000)
            self.assertFalse(all(gates.values()))

    def test_knob_turn_variations(self):
        """Respect signed rotations and initial orientation in the fingertip trajectory."""
        for angle, yaw in ((-np.pi / 2, 0), (np.pi / 2, np.pi / 6)):
            position, _, _ = targets("turn-knob", 2.5, turn_angle=angle, initial_yaw=yaw)
            direction = position[1] - position[0]
            direction /= np.linalg.norm(direction)
            np.testing.assert_allclose(direction, [-np.sin(angle + yaw), np.cos(angle + yaw), 0], atol=1e-15)
            t, h = 1.7, 1e-6
            _, velocity, _ = targets("turn-knob", t, turn_angle=angle, initial_yaw=yaw)
            upper = targets("turn-knob", t + h, turn_angle=angle, initial_yaw=yaw)[0]
            lower = targets("turn-knob", t - h, turn_angle=angle, initial_yaw=yaw)[0]
            np.testing.assert_allclose(velocity, (upper - lower) / (2 * h), rtol=1e-6, atol=1e-8)

    def test_independent_passive_knob(self):
        """Keep the hand and passive rotor in different fixed-base actors without a FREE intermediary."""
        scene = TaskScene("turn-knob", device="cpu")
        actors = scene.solver._articulation_actors
        self.assertEqual(len(actors), 2)
        self.assertEqual(scene.solver._rigid.count, 0)
        self.assertTrue(scene.wrist)
        self.assertNotIn(scene.object_body, actors[0].body_indices)
        self.assertIn(scene.object_body, actors[1].body_indices)
        self.assertTrue(set(scene.fingers).issubset(actors[0].body_indices))
        self.assertEqual(scene.model.joint_target_ke.numpy()[scene.knob_dof], 0)
        self.assertEqual(scene.model.joint_target_kd.numpy()[scene.knob_dof], 0)
        self.assertIn("A-A", scene.solver._contact.actor_samples.pair_kinds.values())
        scene.step(0.002)
        self.assertTrue(scene.solver.last_stats.converged)
        self.assertEqual(scene.control.joint_f.numpy()[scene.knob_dof], 0)
        self.assertEqual(scene.solver.last_stats.joint_pd_force[scene.knob_dof], 0)

    def test_knob_physical_gates(self):
        """Reject penetration, pair imbalance or active actuation even at the correct final angle."""
        scene = SimpleNamespace(task="turn-knob", initial_yaw=0.0, turn_angle=np.pi / 2)
        row = {
            "phase": "move",
            "finger_contact_force_N": 0.1,
            "knob_angle_rad": np.pi / 2,
            "knob_contact_torque_Nm": 1e-4,
            "sample_penetration_m": 0.0005,
            "pair_force_imbalance_N": 0.0,
            "pair_moment_imbalance_Nm": 0.0,
            "knob_active_force_Nm": 0.0,
        }
        self.assertTrue(all(task_metrics(scene, [row])[1].values()))
        for name, invalid, gate in (
            ("sample_penetration_m", 0.0011, "penetration"),
            ("pair_force_imbalance_N", 1e-7, "pair_force_balance"),
            ("pair_moment_imbalance_Nm", 1e-7, "pair_moment_balance"),
            ("knob_active_force_Nm", 1e-7, "passive_knob"),
            ("knob_contact_torque_Nm", 1e-6, "contact_torque"),
        ):
            self.assertFalse(task_metrics(scene, [{**row, name: invalid}])[1][gate])


if __name__ == "__main__":
    unittest.main(verbosity=2)
