# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent controls and physical acceptance checks for full-hand task probes."""

import unittest

import numpy as np

from scripts.monolithic_reference.run_hand_tasks import TASKS, physical_gates, schedule


class TestHandTaskControls(unittest.TestCase):
    def test_schedule_derivatives(self):
        for task in TASKS:
            for t in np.linspace(0.01, 6.99, 231):
                h = 1e-6
                actual = schedule(task, t)
                left, right = schedule(task, t - h), schedule(task, t + h)
                np.testing.assert_allclose(
                    [actual[1], actual[3]], [(right[0] - left[0]) / (2 * h), (right[2] - left[2]) / (2 * h)], atol=1e-8
                )

    def test_release_and_carriage_return(self):
        for task in TASKS:
            final = schedule(task, 7)
            self.assertEqual(final[0], 0)
            self.assertEqual(final[1], 0)
            self.assertEqual(final[3], 0)
            self.assertEqual(final[2], 0.05 if task == "pull-slider" else 0)
            self.assertEqual(schedule(task, 4.25)[2], 0.05)

    @staticmethod
    def good_rows(task):
        base = {
            "time": 7,
            "contact_force_N": 0.5,
            "joint_limit_violation_rad": 0,
            "speed_ratio": 0.5,
            "effort_ratio": 0.5,
            "object_active_force_N": 0,
            "min_det_f": 0.8,
            "sample_penetration_m": 0.001,
            "object_position": [0.05 if task == "pull-slider" else 0, 0, 0.02],
        }
        return [{**base, "time": 4, "object_position": [0, 0, 0.07]}, base]

    def test_static_object_rejected(self):
        for task in TASKS:
            rows = [{**r, "object_position": [0, 0, 0.0202]} for r in self.good_rows(task)]
            self.assertFalse(physical_gates(task, rows)[1]["motion"])

    def test_unphysical_success_rejected(self):
        for task in TASKS:
            rows = self.good_rows(task)
            self.assertTrue(all(physical_gates(task, rows)[1].values()))
            for key, value, gate in (
                ("sample_penetration_m", 0.002, "penetration"),
                ("joint_limit_violation_rad", 0.03, "joint_limits"),
                ("effort_ratio", 1.3, "effort"),
                ("speed_ratio", 1.3, "speed"),
                ("object_active_force_N", 0.01, "passive_object"),
                ("min_det_f", -0.1, "material"),
            ):
                bad = [{**r, key: value} for r in rows]
                self.assertFalse(physical_gates(task, bad)[1][gate])

    def test_missing_observation_or_geometry_is_not_success(self):
        rows = self.good_rows("grasp-rigid")
        self.assertFalse(physical_gates("grasp-rigid", rows[:1])[1]["released_stable"])
        for row in rows:
            del row["sample_penetration_m"]
        self.assertFalse(physical_gates("grasp-rigid", rows)[1]["penetration"])

    def test_slider_gate_follows_elevated_target(self):
        rows = self.good_rows("pull-slider")
        raised = [{**r, "object_position": np.add(r["object_position"], (0, 0, 0.08)).tolist()} for r in rows]
        self.assertTrue(all(physical_gates("pull-slider", raised, height=0.08)[1].values()))
        self.assertFalse(physical_gates("pull-slider", rows, height=0.08)[1]["placement"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
