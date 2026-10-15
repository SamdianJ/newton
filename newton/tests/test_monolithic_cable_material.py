# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check DER energy, time-parallel frames and coupled position/twist tangents on CPU."""

import unittest

import numpy as np

from scripts.monolithic_reference.cable_oracle import (
    CableFrame,
    CableReference,
    advance_frame,
    cable_derivatives,
    cable_energy,
)


def straight_cable(count=5):
    """Create a unit-spaced open cable with directors along the y-axis."""
    points = np.column_stack((np.arange(count), np.zeros((count, 2))))
    return points, np.zeros(count - 1), np.tile([0.0, 1.0, 0.0], (count - 1, 1))


class TestMonolithicCableMaterial(unittest.TestCase):
    stiffness = (100.0, 2.0, 3.0)

    def test_axial_and_twist_analytics(self):
        """Recover uniform stretch and twist energies with end reactions and zero interior torque."""
        points, theta, directors = straight_cable()
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        positions = points * 1.1
        theta = np.linspace(0, 0.6, len(theta))
        expected = [0.5 * 100 * 4 * 0.1**2, 0, 0.5 * 3 * 0.6**2 / 3]
        np.testing.assert_allclose(cable_energy(rest, positions, theta, frame), expected, atol=1e-12)
        residual, tangent = cable_derivatives(rest, positions, theta, frame)
        np.testing.assert_allclose(residual[:15].reshape(-1, 3)[:, 0], [-10, 0, 0, 0, 10], atol=1e-12)
        np.testing.assert_allclose(residual[15:], [-0.6, 0, 0, 0.6], atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -1e-12)
        # Gauge-free twist block is the exact weighted chain Laplacian.
        laplacian = np.diff(np.eye(4), axis=0)
        np.testing.assert_allclose(tangent[15:, 15:], 3 * laplacian.T @ laplacian, atol=1e-12)

    def test_planar_bending(self):
        """Match the discrete circular-arc energy and angular derivative analytically."""
        points, theta, directors = straight_cable(3)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        for angle in (0.1, 0.5, -0.6):
            positions = np.array([[0.0, 0, 0], [1, 0, 0], [1 + np.cos(angle), np.sin(angle), 0]])
            energy = cable_energy(rest, positions, theta, frame)
            expected = 2 * self.stiffness[1] * np.tan(angle / 2) ** 2
            np.testing.assert_allclose(energy, [0, expected, 0], atol=1e-12)
            residual, _ = cable_derivatives(rest, positions, theta, frame)
            derivative = residual[6:9] @ [-np.sin(angle), np.cos(angle), 0]
            self.assertAlmostEqual(
                derivative, 2 * self.stiffness[1] * np.tan(angle / 2) / np.cos(angle / 2) ** 2, places=11
            )

    def test_curved_reference_and_coupling(self):
        """Keep a naturally curved twisted reference stress free and retain position-theta coupling."""
        points = np.array([[0.0, 0, 0], [1, 0, 0], [1.8, 0.6, 0], [2.4, 1.4, 0]])
        theta = np.array([0.1, 0.3, -0.2])
        directors = np.tile([0.0, 0, 1.0], (3, 1))
        rest = CableReference(points, theta, directors, (100.0, np.array([[2.0, 0.4], [0.4, 6.0]]), 3.0))
        frame = CableFrame(points, directors)
        np.testing.assert_allclose(cable_energy(rest, points, theta, frame), 0, atol=1e-28)
        residual, tangent = cable_derivatives(rest, points, theta, frame)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(tangent[:12, 12:]), 0.1)
        self.assertGreater(cable_energy(rest, points, theta + 0.4, frame)[1], 0.01)
        # At rest Gauss-Newton equals the exact Hessian, including all mixed blocks.
        state = np.r_[points.ravel(), theta]
        numerical = np.empty_like(tangent)
        for index in range(len(state)):
            offset = np.zeros_like(state)
            offset[index] = 1e-6
            plus, minus = state + offset, state - offset
            numerical[:, index] = (
                cable_derivatives(rest, plus[:12].reshape(-1, 3), plus[12:], frame)[0]
                - cable_derivatives(rest, minus[:12].reshape(-1, 3), minus[12:], frame)[0]
            ) / 2e-6
        np.testing.assert_allclose(tangent, numerical, atol=3e-8, rtol=2e-7)

    def test_derivatives_balance_and_gauge(self):
        """Check every coupled residual coordinate, translation balance and the straight-rod theta symmetry."""
        points, theta, directors = straight_cable(4)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        positions = points + np.random.default_rng(32).normal(scale=0.08, size=points.shape)
        theta = np.array([0.2, -0.1, 0.4])
        residual, tangent = cable_derivatives(rest, positions, theta, frame)
        state = np.r_[positions.ravel(), theta]
        numerical = np.empty_like(state)
        for index in range(len(state)):
            offset = np.zeros_like(state)
            offset[index] = 1e-6
            plus, minus = state + offset, state - offset
            numerical[index] = (
                cable_energy(rest, plus[:12].reshape(-1, 3), plus[12:], frame).sum()
                - cable_energy(rest, minus[:12].reshape(-1, 3), minus[12:], frame).sum()
            ) / 2e-6
        np.testing.assert_allclose(residual, numerical, atol=2e-8, rtol=2e-7)
        np.testing.assert_allclose(residual[:12].reshape(-1, 3).sum(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(sum(residual[12:]), 0, places=12)
        np.testing.assert_allclose(
            cable_energy(rest, positions, theta + 0.7, frame), cable_energy(rest, positions, theta, frame), atol=1e-12
        )
        np.testing.assert_allclose(tangent, tangent.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -1e-11)

    def test_frame_transport_and_recovery(self):
        """Evaluate trials from a fixed anchor and reproduce continuation from copied frame history."""
        points, theta, directors = straight_cable(4)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        original = frame.directors.copy()
        first = points.copy()
        first[2:, 2] += 0.2
        committed = advance_frame(frame, first)
        other = points.copy()
        other[2:, 1] -= 0.3
        advance_frame(frame, other)
        np.testing.assert_array_equal(frame.directors, original)
        np.testing.assert_array_equal(advance_frame(frame, first).directors, committed.directors)
        recovered = CableFrame(committed.positions, committed.directors, reference_twist=committed.reference_twist)
        second = first.copy()
        second[1, 1] += 0.1
        np.testing.assert_array_equal(
            cable_energy(rest, second, theta, recovered), cable_energy(rest, second, theta, committed)
        )
        np.testing.assert_allclose(
            cable_energy(rest, first, theta, frame), cable_energy(rest, first, theta, committed), atol=1e-12
        )
        directions = np.diff(first, axis=0)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        np.testing.assert_allclose(np.sum(directions * committed.directors, axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(committed.directors, axis=1), 1, atol=1e-14)

    def test_rigid_motion(self):
        """Rotate the reference and its history together without changing material energies."""
        points, theta, directors = straight_cable(4)
        positions = points.copy()
        positions[1:3, 2] += 0.2
        theta += [0, 0.1, 0.3]
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2, 1, 1], [1, -1, 2]]))
        frame = CableFrame(points, directors)
        rest = CableReference(points, np.zeros(3), directors, self.stiffness)
        rotated_rest = CableReference(points @ rotation.T, np.zeros(3), directors @ rotation.T, self.stiffness)
        rotated_frame = CableFrame(points @ rotation.T, directors @ rotation.T)
        np.testing.assert_allclose(
            cable_energy(rest, positions, theta, frame),
            cable_energy(rotated_rest, positions @ rotation.T, theta, rotated_frame),
            atol=1e-12,
        )
        residual, tangent = cable_derivatives(rest, positions, theta, frame)
        rotated_r, rotated_b = cable_derivatives(rotated_rest, positions @ rotation.T, theta, rotated_frame)
        mapping = np.eye(15)
        mapping[:12, :12] = np.kron(np.eye(4), rotation)
        np.testing.assert_allclose(rotated_r, mapping @ residual, atol=1e-12)
        np.testing.assert_allclose(rotated_b, mapping @ tangent @ mapping.T, atol=1e-11)

    def test_invalid_input_and_ownership(self):
        """Reject collapsed or reversing geometry and copy all caller-owned reference/history arrays."""
        points, theta, directors = straight_cable(3)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        points[1] += [0, 0.2, 0]
        theta[:] = 1
        directors[:] = 0
        original, angles, axes = straight_cable(3)
        np.testing.assert_array_equal(frame.positions, original)
        np.testing.assert_array_equal(frame.directors, axes)
        np.testing.assert_allclose(cable_energy(rest, original, angles, frame), 0, atol=1e-28)
        for bad in (original * 0, np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 0]]), original * np.nan):
            with self.assertRaises(ValueError):
                cable_energy(rest, bad, angles, frame)
        with self.assertRaises(ValueError):
            advance_frame(frame, -original)
        with self.assertRaises(ValueError):
            CableReference(original, angles, axes, (100, -2, 3))
        with self.assertRaises(ValueError):
            CableFrame(original, axes * 2)

    def test_twist_branch_and_single_edge(self):
        """Retain unwrapped twist history across pi and expose the one-edge theta zero mode."""
        points, theta, _ = straight_cable(3)
        angles = np.array([0.0, 3.2])
        directors = np.column_stack((np.zeros(2), np.cos(angles), np.sin(angles)))
        frame = CableFrame(points, directors, reference_twist=np.array([3.2]))
        np.testing.assert_allclose(frame.reference_twist, [3.2], atol=1e-14)
        np.testing.assert_allclose(advance_frame(frame, points).reference_twist, [3.2], atol=1e-14)
        with self.assertRaises(ValueError):
            CableFrame(points, directors, reference_twist=np.array([3.1]))
        points, theta, directors = straight_cable(2)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        residual, tangent = cable_derivatives(rest, points, theta + 8 * np.pi, frame)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        np.testing.assert_array_equal(tangent[-1], 0)

    def test_force_and_torque_balance(self):
        """Balance translational moments and theta torques in the current frame chart."""
        points, theta, directors = straight_cable(4)
        rest = CableReference(points, theta, directors, self.stiffness)
        positions = points + np.random.default_rng(82).normal(scale=0.1, size=points.shape)
        frame = advance_frame(CableFrame(points, directors), positions)
        residual, _ = cable_derivatives(rest, positions, np.array([0.1, 0.3, 0.5]), frame)
        tangents = np.diff(positions, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        moment = np.cross(positions, residual[:12].reshape(-1, 3)).sum(axis=0)
        moment += (residual[12:, None] * tangents).sum(axis=0)
        np.testing.assert_allclose(moment, 0, atol=1e-12)

    def test_quasistatic_torque_cycle(self):
        """Solve loading and unloading torque without theta mass and recover the omitted gauge equation."""
        points, theta, directors = straight_cable(5)
        rest = CableReference(points, theta, directors, self.stiffness)
        frame = CableFrame(points, directors)
        for torque in (0.2, 0.6, 0.6, 0.2, 0.0):
            applied = np.array([-torque, 0, 0, torque])
            residual, tangent = cable_derivatives(rest, points, theta, frame)
            # Only a gauge is eliminated; its original equilibrium equation is checked below.
            theta[1:] += np.linalg.solve(tangent[16:, 16:], applied[1:] - residual[16:])
            residual, _ = cable_derivatives(rest, points, theta, frame)
            np.testing.assert_allclose(residual[15:], applied, atol=1e-12)
            np.testing.assert_allclose(theta, np.arange(4) * torque / self.stiffness[2], atol=1e-12)

    def test_anisotropic_material_orientation(self):
        """Retain the physical common-theta response of an anisotropic bent rod."""
        points, theta, directors = straight_cable(3)
        bending = np.array([[2.0, 0.4], [0.4, 6.0]])
        rest = CableReference(points, theta, directors, (100.0, bending, 3.0))
        frame = CableFrame(points, directors)
        angle = 0.4
        positions = np.array([[0.0, 0, 0], [1, 0, 0], [1 + np.cos(angle), np.sin(angle), 0]])
        for orientation in (0.0, 0.3, np.pi / 2):
            strain = 2 * np.tan(angle / 2) * np.array([np.cos(orientation), -np.sin(orientation)])
            expected = 0.5 * strain @ bending @ strain
            energy = cable_energy(rest, positions, theta + orientation, frame)
            np.testing.assert_allclose(energy, [0, expected, 0], atol=1e-12)
        residual, tangent = cable_derivatives(rest, positions, theta + 0.3, frame)
        self.assertGreater(abs(sum(residual[9:])), 0.01)
        direction = np.array([np.cos(0.3), -np.sin(0.3)])
        derivative = np.array([-np.sin(0.3), -np.cos(0.3)])
        expected_torque = (2 * np.tan(angle / 2)) ** 2 * derivative @ bending @ direction
        self.assertAlmostEqual(sum(residual[9:]), expected_torque, places=12)
        self.assertGreater(np.linalg.norm(tangent[:9, 9:]), 0.01)
        self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -1e-12)
        bending[:] = 0  # material input is a snapshot, not a borrowed array
        self.assertGreater(cable_energy(rest, positions, theta, frame)[1], 0.1)
        for bad in (np.diag([2.0, 0.0]), np.array([[2.0, 1], [0, 3]]), np.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):
                CableReference(points, theta, directors, (100.0, bad, 3.0))

    def test_anisotropic_neutral_straight_mode(self):
        """Expose a temporary straight-rod null direction without imposing a permanent orientation clamp."""
        points, theta, directors = straight_cable(3)
        rest = CableReference(points, theta, directors, (100.0, np.diag([2.0, 6.0]), 3.0))
        frame = CableFrame(points, directors)
        residual, tangent = cable_derivatives(rest, points, theta, frame)
        null = np.r_[np.zeros(9), 1.0, 1.0]
        np.testing.assert_allclose(tangent @ null, 0, atol=1e-12)
        self.assertAlmostEqual(residual @ null, 0, places=12)
        positions = points.copy()
        positions[2, 1] = 0.3
        bent_r, bent_b = cable_derivatives(rest, positions, theta + 0.3, frame)
        self.assertGreater(abs(bent_r @ null), 0.01)
        self.assertGreater(np.linalg.norm(bent_b @ null), 0.01)


if __name__ == "__main__":
    unittest.main()
