# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check task controls and physical success metrics without long robot runs."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.monolithic_reference.run_dexterous_tasks import TASKS, targets, task_metrics


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
            }
            _, gates = task_metrics(SimpleNamespace(task=task), [row] * 2000)
            self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
