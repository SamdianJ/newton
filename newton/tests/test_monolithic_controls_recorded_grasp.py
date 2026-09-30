# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check recorded grasp timing before interpreting a physical rollout."""

import copy
import unittest

import numpy as np

from newton.examples.softbody.monolithic_grasp_trajectory import RecordedGrasp


def payload():
    return {
        "format": "mujoco_teleop.target_trajectory",
        "version": 1,
        "frame_dt": 1 / 60,
        "keyframes": [
            {"frame": f, "position": [0.1, -0.015, z], "quaternion_xyzw": [1, 0, 0, 0], "gripper_q": g}
            for f, z, g in ((0, 0.06, 1), (276, 0.06, 0), (490, 0.09, 0))
        ],
    }


class TestRecordedGrasp(unittest.TestCase):
    def test_prepend_approach_preserves_closure_and_lift(self):
        """Keep fingers open during descent and retain subsequent relative motion."""
        original = payload()
        adjusted = copy.deepcopy(original)
        first = copy.deepcopy(original["keyframes"][0])
        for row in adjusted["keyframes"]:
            row["frame"] += 30
            row["position"][2] -= 0.005
        adjusted["keyframes"].insert(0, first)
        baseline, trajectory = RecordedGrasp(original), RecordedGrasp(adjusted)
        self.assertEqual(trajectory.steps, 8667)
        sample = trajectory.sample(0.25)
        self.assertEqual(sample[4], "approach")
        self.assertEqual(sample[1], 1)
        self.assertEqual(sample[3], 0)
        np.testing.assert_allclose(sample[2], [0, 0, -0.01])
        for time in (0, 1, 4.6, 5, baseline.duration):
            old, new = baseline.sample(time), trajectory.sample(time + 0.5)
            np.testing.assert_allclose(new[0], old[0] - [0, 0, 0.005], atol=1e-15)
            self.assertAlmostEqual(new[1], old[1])
        self.assertAlmostEqual(trajectory.position[-1, 2] - trajectory.position[1, 2], 0.03)

    def test_clock_and_stages(self):
        """Preserve the recorded clock without inventing hold or release."""
        trajectory = RecordedGrasp(payload())
        self.assertEqual(trajectory.steps, 8167)
        self.assertAlmostEqual(trajectory.duration, 490 / 60)
        np.testing.assert_allclose(trajectory.sample(2.3)[0], [0.1, -0.015, 0.06])
        self.assertAlmostEqual(trajectory.sample(2.3)[1], 0.5)
        self.assertEqual(trajectory.sample(2.3)[4], "close")
        self.assertEqual(trajectory.sample(6)[4], "lift")
        end = trajectory.sample(8.167)
        np.testing.assert_allclose(end[0], [0.1, -0.015, 0.09])
        self.assertEqual(end[1], 0)
        np.testing.assert_array_equal(end[2], np.zeros(3))
        self.assertEqual(end[3], 0)

    def test_segment_derivatives(self):
        """Use the target interpolation derivative in physical seconds."""
        trajectory = RecordedGrasp(payload())
        for time in (1, 5.5):
            p, g, v, rate, _ = trajectory.sample(time)
            del p, g
            lo, hi = trajectory.sample(time - 1e-4), trajectory.sample(time + 1e-4)
            np.testing.assert_allclose((hi[0] - lo[0]) / 2e-4, v, atol=1e-12)
            self.assertAlmostEqual((hi[1] - lo[1]) / 2e-4, rate)

    def test_reject_invalid_or_unsupported_motion(self):
        """Reject invalid clocks and motion a vertical carriage cannot replay."""
        original = payload()
        variants = []
        for key, value in (
            ("frame", 276),
            ("gripper_q", float("nan")),
            ("position", [0.2, 0, 0.09]),
            ("quaternion_xyzw", [0, 0, 0, 1]),
        ):
            bad = copy.deepcopy(original)
            bad["keyframes"][-1][key] = value
            variants.append(bad)
        bad = copy.deepcopy(original)
        bad["frame_dt"] = 0
        variants.append(bad)
        for bad in variants:
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                RecordedGrasp(bad)


if __name__ == "__main__":
    unittest.main()
