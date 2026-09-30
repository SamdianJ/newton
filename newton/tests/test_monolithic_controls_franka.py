# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the Franka trajectory."""

import unittest

import numpy as np

from newton.examples.softbody.monolithic_franka import TIMES, targets


class TestFrankaGrasp(unittest.TestCase):
    def setUp(self):
        self.high = np.array([0, -0.4, 0, -2.0, 0, 1.7, 0.78, 0.04, 0.04])
        self.low = self.high + np.array([0, -0.1, 0, -0.2, 0, 0.1, 0, 0, 0])

    def test_endpoint_timing_and_opening(self):
        """Preserve phases, finger opening and zero velocity at segment boundaries."""
        openings = [0.04, 0.04, 0.04, 0.014, 0.014, 0.014, 0.014, 0.04, 0.04]
        for t, opening in zip(TIMES, openings, strict=True):
            q, v, _ = targets(self.high, self.low, t)
            np.testing.assert_allclose(q[7:], opening, rtol=0, atol=1e-15)
            np.testing.assert_allclose(v, 0, rtol=0, atol=1e-15)
        q, v, phase = targets(self.high, self.low, TIMES[-1] + 1.0)
        np.testing.assert_allclose(q, self.high)
        np.testing.assert_array_equal(v, np.zeros(9))
        self.assertEqual(phase, "retract")

    def test_velocity_matches_target_derivative(self):
        """Match endpoint-controller velocities to the analytic position derivative."""
        for t in (TIMES[:-1] + TIMES[1:]) / 2:
            q, v, _ = targets(self.high, self.low, t)
            h = 1e-6
            forward = targets(self.high, self.low, t + h)[0]
            backward = targets(self.high, self.low, t - h)[0]
            np.testing.assert_allclose(v, (forward - backward) / (2 * h), rtol=1e-7, atol=1e-9)
            self.assertEqual(q[7], q[8])

    def test_reject_invalid_time(self):
        """Reject nonfinite or negative trajectory times."""
        for t in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                targets(self.high, self.low, t)


if __name__ == "__main__":
    unittest.main()
