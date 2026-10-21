# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check production rod derivatives against the independent complex-step oracle."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable import CableObjective
from scripts.monolithic_reference.cable_oracle import CableFrame, CableReference, cable_derivatives, cable_energy


class TestMonolithicCableObjective(unittest.TestCase):
    device = "cpu"

    def test_transport_chart_and_unwrapped_history(self):
        """Differentiate away from the committed chart origin without wrapping theta."""
        x = np.array([[0.0, 0, 0], [0.5, 0, 0], [1, 0, 0]])
        d = np.tile([0.0, 1.0, 0.0], (2, 1))
        theta = np.array([8.0, 8.1])
        stiffness = (10.0, np.array([[0.03, 0.007], [0.007, 0.05]]), 0.02)
        rest = CableReference(x, theta, d, stiffness)
        objective = CableObjective(x, theta, d, stiffness, dtype=wp.float64, device=self.device)
        rotated = d.copy()
        rotated[1] = [0, np.cos(0.3), np.sin(0.3)]
        frame = CableFrame(x, rotated, reference_twist=np.array([0.3 + 2 * np.pi]))
        trial = x + np.array([[0, 0, 0], [0.01, -0.1, 0.07], [-0.01, 0.08, 0.04]])
        energy, residual, tangent = objective.evaluate_host(trial, theta + np.array([0.13, -0.3]), frame)
        r, b = cable_derivatives(rest, trial, theta + np.array([0.13, -0.3]), frame)
        np.testing.assert_allclose(energy, cable_energy(rest, trial, theta + np.array([0.13, -0.3]), frame), atol=1e-12)
        np.testing.assert_allclose(residual, r, atol=1e-11)
        np.testing.assert_allclose(tangent, b, atol=2e-6, rtol=2e-5)
        self.assertGreater(objective.phi.numpy()[0], np.pi)

    def test_single_edge_and_instantaneous_common_angle_mode(self):
        """Keep theta massless and distinguish straight from bent anisotropic rods."""
        for vertices in (2, 3):
            x = np.column_stack((np.arange(vertices, dtype=float), np.zeros((vertices, 2))))
            d = np.tile([0.0, 1.0, 0.0], (vertices - 1, 1))
            theta = np.zeros(vertices - 1)
            stiffness = (10.0, np.diag([0.02, 0.06]), 0.03)
            objective = CableObjective(x, theta, d, stiffness, dtype=wp.float64, device=self.device)
            frame = CableFrame(x, d)
            _, residual, matrix = objective.evaluate_host(x, theta, frame)
            common = np.r_[np.zeros(x.size), np.ones(theta.size)]
            np.testing.assert_allclose(matrix @ common, 0, atol=1e-12)
            np.testing.assert_array_equal(residual, 0)
            if vertices == 3:
                trial = x.copy()
                trial[-1, 1:] = [0.3, 0.1]
                _, residual, matrix = objective.evaluate_host(trial, theta, frame)
                self.assertGreater(np.linalg.norm(matrix @ common), 1e-4)
                self.assertGreater(abs(residual @ common), 1e-4)

    def test_anisotropic_curved_reference(self):
        """Preserve every xyz/theta block and the natural curved, twisted shape."""
        x0 = np.array([[0, 0, 0], [0.4, 0, 0], [0.7, 0.2, 0], [1, 0.4, 0]])
        d0 = np.tile([0.0, 0.0, 1.0], (3, 1))
        theta0 = np.array([0.1, -0.2, 0.4])
        stiffness = (12.0, np.array([[0.08, 0.02], [0.02, 0.04]]), 0.03)
        rest = CableReference(x0, theta0, d0, stiffness)
        frame = CableFrame(x0, d0)
        for dtype in (wp.float64, wp.float32):
            objective = CableObjective(x0, theta0, d0, stiffness, dtype=dtype, device=self.device)
            for deformed in (False, True):
                with self.subTest(dtype=dtype, deformed=deformed):
                    x, theta = x0.copy(), theta0.copy()
                    if deformed:
                        x[1:] += [[0.01, 0.02, -0.04], [-0.02, 0.03, 0.04], [0.02, -0.01, 0.05]]
                        theta += [0.03, -0.09, 0.13]
                    energy, residual, tangent = objective.evaluate_host(x, theta, frame)
                    expected_r, expected_b = cable_derivatives(rest, x, theta, frame)
                    tolerance = 2e-5 if dtype == wp.float32 else 1e-11
                    np.testing.assert_allclose(
                        energy, cable_energy(rest, x, theta, frame), rtol=tolerance, atol=tolerance
                    )
                    np.testing.assert_allclose(residual, expected_r, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(tangent, expected_b, rtol=2e-5, atol=2e-6)
                    self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -2e-6 * np.linalg.norm(tangent))
                    np.testing.assert_allclose(residual[: x.size].reshape(-1, 3).sum(axis=0), 0, atol=tolerance)

    def test_geometry_rejection_does_not_change_anchor(self):
        """Reject collapse and antiparallel transport without modifying committed frames."""
        x = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
        d = np.tile([0.0, 1.0, 0.0], (2, 1))
        theta = np.zeros(2)
        frame = CableFrame(x, d)
        objective = CableObjective(x, theta, d, (10, np.eye(2), 1), dtype=wp.float64, device=self.device)
        for bad in (np.array([[0.0, 0, 0], [0, 0, 0], [2, 0, 0]]), -x):
            with self.assertRaises(ValueError):
                objective.evaluate_host(bad, theta, frame)
        np.testing.assert_array_equal(frame.positions, x)
        np.testing.assert_array_equal(frame.directors, d)
        energy, residual, _ = objective.evaluate_host(x, theta, frame)
        np.testing.assert_array_equal(energy, 0)
        np.testing.assert_array_equal(residual, 0)

    def test_far_step_in_float32(self):
        """Resolve a small FP32 step of a far, much-twisted rod from its exact increments."""
        x0 = np.array([[0, 0, 0], [0.4, 0, 0], [0.7, 0.2, 0], [1, 0.4, 0]])
        d0 = np.tile([0.0, 0.0, 1.0], (3, 1))
        theta0 = np.array([0.1, -0.2, 0.4]) + 200.0
        stiffness = (12.0, np.array([[0.08, 0.02], [0.02, 0.04]]), 0.03)
        rest = CableReference(x0, theta0, d0, stiffness)
        # At 1 km, FP32 positions round at 6e-5 m and angles near 200 rad at 2e-5.
        offset = np.array([1000.0, -700.0, 400.0])
        start = (x0 + offset).astype(np.float32).astype(float)
        theta = theta0.astype(np.float32).astype(float)
        step = (1e-4 * np.sin(np.arange(12).reshape(4, 3) * [1.3, 2.1, 2.9])).astype(np.float32).astype(float)
        theta_step = np.array([1e-4, -2e-4, 1.5e-4]).astype(np.float32).astype(float)
        frame = CableFrame(start, d0)
        objective = CableObjective(x0, theta0, d0, stiffness, dtype=wp.float32, device=self.device)
        energy, residual, _ = objective.evaluate_host(start, theta, frame, step=step, theta_step=theta_step)
        expected = cable_energy(rest, start + step, theta + theta_step, frame)
        self.assertLess(np.abs(energy - expected).sum(), 2e-3 * np.abs(expected).sum())
        expected, _ = cable_derivatives(rest, start + step, theta + theta_step, frame)
        np.testing.assert_allclose(residual, expected, atol=2e-3 * np.abs(expected).max())

    def test_near_rest_noise_in_float32(self):
        """Keep FP32 stretch forces near rest free of length-size cancellation noise."""
        x0 = np.array([[0, 0, 0], [0.4, 0, 0], [0.7, 0.2, 0], [1, 0.4, 0]])
        d0 = np.tile([0.0, 0.0, 1.0], (3, 1))
        theta0 = np.array([0.1, -0.2, 0.4])
        stiffness = (12.0, np.array([[0.08, 0.02], [0.02, 0.04]]), 0.03)
        rest = CableReference(x0, theta0, d0, stiffness)
        start = x0.astype(np.float32).astype(float)
        theta = theta0.astype(np.float32).astype(float)
        frame = CableFrame(start, d0)
        objective = CableObjective(x0, theta0, d0, stiffness, dtype=wp.float32, device=self.device)
        step = (1e-6 * np.sin(np.arange(12).reshape(4, 3) * [1.3, 2.1, 2.9])).astype(np.float32).astype(float)
        none = np.zeros(3)

        def residual(increment):
            return objective.evaluate_host(start, theta, frame, step=increment, theta_step=none)[1]

        def exact(increment):
            return cable_derivatives(rest, start + increment, theta, frame)[0]

        base, expected = residual(step), exact(step)
        rng = np.random.default_rng(1)
        noise = 0.0
        for _ in range(20):
            jittered = (step + 1e-9 * rng.standard_normal(step.shape)).astype(np.float32).astype(float)
            noise = max(noise, np.abs((residual(jittered) - base) - (exact(jittered) - expected)).max())
        # An extension formed as |e| - L0 varies by ~EA eps32 here.
        self.assertLess(noise, 0.05 * stiffness[0] * np.finfo(np.float32).eps)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMonolithicCableObjectiveCUDA(TestMonolithicCableObjective):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
