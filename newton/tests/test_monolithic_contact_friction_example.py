# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the frozen friction-demo schedule and acceptance controls."""

import unittest

import numpy as np

from newton.examples.softbody.monolithic_contact_friction import FIXTURE, target, validate_comparison


class TestFrictionExample(unittest.TestCase):
    def test_schedule(self):
        """Check stage endpoints, hold velocity and the derivative of each smooth ramp."""
        for t, q in zip(FIXTURE["target_times"], FIXTURE["target_positions"], strict=True):
            actual, velocity = target(t)
            np.testing.assert_allclose(actual, q, atol=1e-12)
            np.testing.assert_array_equal(velocity, 0)
        for t in (0.4, 1.0, 2.0, 2.75, 3.5, 3.9):
            _, velocity = target(t)
            fd = (target(t + 1e-5)[0] - target(t - 1e-5)[0]) / 2e-5
            np.testing.assert_allclose(fd, velocity, atol=1e-10)

    def test_incomplete_run(self):
        """Prevent a partial or empty demonstration from claiming acceptance."""

        class EmptyCase:
            records = ()

            def summary(self):
                return {
                    "steps": 0,
                    "converged_fraction": 1.0,
                    "peak_displacement_x": 0.0,
                    "peak_tangent_force": 0.0,
                    "stick_samples": 0,
                    "slide_samples": 0,
                    "lost_samples": 0,
                    "final_history_energy": 0.0,
                }

        result = validate_comparison([EmptyCase(), EmptyCase()])
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["complete"])
        self.assertFalse(result["gates"]["deformation"])


if __name__ == "__main__":
    unittest.main()
