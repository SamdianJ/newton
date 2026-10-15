# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the selected shell energy with CPU-only independent mathematical fixtures."""

import unittest

import numpy as np

from scripts.monolithic_reference.cloth_oracle import (
    shell_bending_tangent,
    shell_energy,
    shell_membrane_derivatives,
    shell_membrane_tangent,
    shell_residual,
)


def flat_stencil():
    """Build one triangle and the opposite vertex across each of its three edges."""
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0]], dtype=float)


def folded_stencil(angle):
    """Rotate the neighbor across the x-axis without stretching the central triangle."""
    points = flat_stencil()
    points[5] = [1, -np.cos(angle), np.sin(angle)]
    return points


class TestMonolithicClothMaterial(unittest.TestCase):
    material = (30.0, 40.0, 0.02, 0.07)

    def test_projected_membrane_tangent(self):
        """Preserve rest stiffness and full coupling while projecting compressed material modes."""
        reference = flat_stencil()
        _, exact = shell_membrane_derivatives(reference, reference, self.material)
        np.testing.assert_allclose(shell_membrane_tangent(reference, reference, self.material), exact, atol=1e-12)
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2, 1, 1], [1, -1, 2]]))
        block_rotation = np.kron(np.eye(3), rotation)
        for stretch in (0.3, 0.9, 1.2, 2.0):
            positions = reference * [stretch, 0.8, 1]
            tangent = shell_membrane_tangent(reference, positions, self.material)
            np.testing.assert_allclose(tangent, tangent.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -1e-12)
            np.testing.assert_allclose(tangent @ np.tile(np.eye(3), (3, 1)), 0, atol=1e-12)
            self.assertGreater(np.linalg.norm(tangent[:3, 3:6]), 1)
            np.testing.assert_allclose(
                shell_membrane_tangent(reference, positions @ rotation.T, self.material),
                block_rotation @ tangent @ block_rotation.T,
                atol=1e-12,
            )
            _, exact = shell_membrane_derivatives(reference, positions, self.material)
            self.assertGreaterEqual(np.linalg.eigvalsh(tangent - exact).min(), -1e-12)

    def test_bending_material_tangent(self):
        """Verify full curvature material stiffness and eliminate missing-neighbor coordinates."""
        reference = folded_stencil(0.3)
        for boundary in ((False, False, False), (False, False, True), (True, True, True)):
            positions = folded_stencil(0.8)
            tangent = shell_bending_tangent(reference, positions, self.material, boundary=boundary)
            np.testing.assert_allclose(tangent, tangent.T, atol=1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), -1e-12)
            np.testing.assert_allclose(tangent @ np.tile(np.eye(3), (6, 1)), 0, atol=1e-12)
            for i, missing in enumerate(boundary):
                if missing:
                    np.testing.assert_array_equal(tangent[9 + 3 * i : 12 + 3 * i], 0)
            if not any(boundary):
                derivative = np.zeros(18)
                derivative[15:] = [0, np.sin(0.8), np.cos(0.8)]
                expected = (self.material[2] + self.material[3]) / (2 * np.cos(0.4) ** 4)
                self.assertAlmostEqual(derivative @ tangent @ derivative, expected, places=12)
                self.assertGreater(np.linalg.norm(tangent[:3, 15:]), 0.01)
        # At rest the omitted geometric term vanishes: compare all columns to d(r)/dx.
        membrane = shell_membrane_tangent(reference, reference, self.material)
        full = shell_bending_tangent(reference, reference, self.material)
        full[:9, :9] += membrane
        numerical = np.empty((18, 18))
        for index in range(18):
            offset = np.zeros((6, 3))
            offset.flat[index] = 1e-6
            numerical[:, index] = (
                shell_residual(reference, reference + offset, self.material)
                - shell_residual(reference, reference - offset, self.material)
            ).ravel() / 2e-6
        np.testing.assert_allclose(full, numerical, atol=2e-8, rtol=2e-7)

    def test_reference_shape(self):
        """Keep flat and pre-folded reference shapes stress free."""
        for angle in (0.0, np.pi / 2, -np.pi / 3):
            with self.subTest(angle=angle):
                reference = folded_stencil(angle)
                np.testing.assert_allclose(shell_energy(reference, reference, self.material), 0, atol=1e-28)
                np.testing.assert_allclose(shell_residual(reference, reference, self.material), 0, atol=1e-12)

    def test_affine_membrane(self):
        """Match stable Neo-Hookean energy and stress for stretch, shear and compression."""
        transforms = (np.diag([1.2, 0.9, 1]), np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]]), np.diag([0.7, 0.8, 1]))
        references = (flat_stencil(), flat_stencil() @ np.array([[1, 0.2, 0], [0.4, 1.3, 0], [0, 0, 1]]))
        for reference, transform in ((r, t) for r in references for t in transforms):
            with self.subTest(reference=reference.tolist(), transform=transform.tolist()):
                positions = reference @ transform.T
                gradient = transform[:, :2]
                metric = gradient.T @ gradient
                jacobian = np.linalg.det(transform[:2, :2])
                lam, mu, _, _ = self.material
                rest_edges = (reference[1:3, :2] - reference[0, :2]).T
                area = np.linalg.det(rest_edges) / 2
                expected = area * (
                    mu / 2 * (np.trace(metric) - 2) - mu * (jacobian - 1) + (lam + mu) / 2 * (jacobian - 1) ** 2
                )
                energy = shell_energy(reference, positions, self.material)
                np.testing.assert_allclose(energy, [expected, 0], atol=1e-14, rtol=1e-13)
                pressure = (lam + mu) * (jacobian - 1) - mu
                stress = mu * gradient + pressure * jacobian * gradient @ np.linalg.inv(metric)
                stress = stress @ np.linalg.inv(rest_edges).T
                expected_r = np.zeros((6, 3))
                expected_r[:3] = area * np.array([-stress.sum(axis=1), stress[:, 0], stress[:, 1]])
                np.testing.assert_allclose(shell_residual(reference, positions, self.material), expected_r, atol=1e-12)

    def test_area_preserving_shear(self):
        """Recover the exact quadratic simple-shear energy at small and large strains."""
        reference = flat_stencil()
        for shear in (1e-7, 0.3, 2.0):
            with self.subTest(shear=shear):
                positions = reference.copy()
                positions[:, 0] += shear * reference[:, 1]
                energy = shell_energy(reference, positions, self.material)[0]
                self.assertAlmostEqual(energy / (self.material[1] * shear**2 / 4), 1, places=8)

    def test_membrane_compression_domain(self):
        """Keep positive area ratios unclamped and reject collapsed triangles."""
        reference = flat_stencil()
        lam, mu, _, _ = self.material
        for stretch in (1e-8, 0.01, 0.5, 1.5):
            with self.subTest(stretch=stretch):
                positions = reference * [stretch, 1, 1]
                expected = (lam + 2 * mu) * (stretch - 1) ** 2 / 4
                np.testing.assert_allclose(shell_energy(reference, positions, self.material), [expected, 0], atol=1e-13)
                residual = shell_residual(reference, positions, self.material)
                self.assertAlmostEqual(residual[1, 0], (lam + 2 * mu) * (stretch - 1) / 2, places=11)
        positions = reference * [0, 1, 1]
        for evaluate in (shell_energy, shell_residual, shell_membrane_derivatives):
            with self.assertRaises(ValueError):
                evaluate(reference, positions, self.material)

    def test_negative_lame_parameter(self):
        """Admit negative lambda when the two-dimensional bulk modulus remains positive."""
        reference = flat_stencil()
        material = (-30.0, 40.0, 0.02, 0.07)
        positions = reference * [0.8, 1.2, 1]
        self.assertGreater(shell_energy(reference, positions, material)[0], 0)
        np.testing.assert_allclose(shell_residual(reference, reference, material), 0, atol=1e-12)

    def test_membrane_hessian(self):
        """Check every analytic membrane block against independent residual differences."""
        boundary = (True, True, True)
        references = (flat_stencil(), flat_stencil() @ np.array([[1, 0.2, 0], [0.4, 1.3, 0], [0, 0, 1]]))
        for reference in references:
            for stretch in (0.6, 1.0, 1.4):
                with self.subTest(reference=reference.tolist(), stretch=stretch):
                    positions = reference * [stretch, 0.9, 1]
                    positions[1, 2] += 0.15
                    residual, hessian = shell_membrane_derivatives(reference, positions, self.material)
                    np.testing.assert_allclose(
                        residual, shell_residual(reference, positions, self.material, boundary=boundary)[:3], atol=1e-12
                    )
                    numerical = np.empty((9, 9))
                    for index in range(9):
                        offset = np.zeros_like(positions)
                        offset.flat[index] = 1e-6
                        numerical[:, index] = (
                            shell_residual(reference, positions + offset, self.material, boundary=boundary)[:3]
                            - shell_residual(reference, positions - offset, self.material, boundary=boundary)[:3]
                        ).ravel() / 2e-6
                    np.testing.assert_allclose(hessian, numerical, atol=2e-8, rtol=1e-7)
                    np.testing.assert_allclose(hessian, hessian.T, atol=1e-13)
                    np.testing.assert_allclose(hessian @ np.tile(np.eye(3), (3, 1)), 0, atol=1e-13)
                    self.assertGreater(np.linalg.norm(hessian[:3, 3:6]), 1)

    def test_membrane_hessian_geometry(self):
        """Distinguish the exact compressed Hessian from a PSD production tangent."""
        reference = flat_stencil()
        _, rest_hessian = shell_membrane_derivatives(reference, reference, self.material)
        self.assertGreaterEqual(np.linalg.eigvalsh(rest_hessian).min(), -1e-12)
        positions = reference * [0.6, 0.7, 1]
        residual, hessian = shell_membrane_derivatives(reference, positions, self.material)
        self.assertLess(np.linalg.eigvalsh(hessian).min(), -1)
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2, 1, 1], [1, -1, 2]]))
        block_rotation = np.kron(np.eye(3), rotation)
        rotated_r, rotated_h = shell_membrane_derivatives(reference, positions @ rotation.T, self.material)
        np.testing.assert_allclose(rotated_r, residual @ rotation.T, atol=1e-12)
        np.testing.assert_allclose(rotated_h, block_rotation @ hessian @ block_rotation.T, atol=1e-12)
        for scale in (1e-3, 10.0):
            scaled_r, scaled_h = shell_membrane_derivatives(reference * scale, positions * scale, self.material)
            np.testing.assert_allclose(scaled_r, scale * residual, atol=1e-12)
            np.testing.assert_allclose(scaled_h, hessian, atol=1e-12)

    def test_folded_bending(self):
        """Match the analytic single-fold curvature energy and angular derivative."""
        stiffness = self.material[2] + self.material[3]
        for rest_angle, angle in ((0.0, 0.6), (np.pi / 2, 1.3), (-0.5, 0.2)):
            with self.subTest(rest_angle=rest_angle, angle=angle):
                reference, positions = folded_stencil(rest_angle), folded_stencil(angle)
                change = np.tan(angle / 2) - np.tan(rest_angle / 2)
                energy = shell_energy(reference, positions, self.material)
                np.testing.assert_allclose(energy, [0, stiffness * change**2], atol=1e-14, rtol=1e-13)
                derivative = shell_residual(reference, positions, self.material)[5] @ [0, np.sin(angle), np.cos(angle)]
                self.assertAlmostEqual(derivative, stiffness * change / np.cos(angle / 2) ** 2, places=13)

    def test_rigid_motion(self):
        """Preserve energy and rotate residuals under independent rigid frame changes."""
        reference, positions = folded_stencil(0.4), folded_stencil(0.8)
        positions[1] += [0.03, 0.02, -0.01]
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [-2, 1, 1], [1, -1, 2]]))
        rotation[:, -1] *= np.linalg.det(rotation)
        energy = shell_energy(reference, positions, self.material)
        residual = shell_residual(reference, positions, self.material)
        transformed = positions @ rotation.T + [2, -3, 1]
        np.testing.assert_allclose(shell_energy(reference, transformed, self.material), energy, atol=1e-12)
        np.testing.assert_allclose(
            shell_residual(reference, transformed, self.material), residual @ rotation.T, atol=1e-11
        )
        np.testing.assert_allclose(
            shell_energy(reference @ rotation.T + [4, 1, -2], positions, self.material), energy, atol=1e-12
        )

    def test_energy_gradient_and_balance(self):
        """Compare complex-step residuals with central differences and internal balance."""
        reference = folded_stencil(0.5)
        positions = reference + np.random.default_rng(14).normal(scale=0.03, size=(6, 3))
        residual = shell_residual(reference, positions, self.material)
        numerical = np.zeros_like(positions)
        for index in range(18):
            offset = np.zeros_like(positions)
            offset.flat[index] = 1e-6
            numerical.flat[index] = (
                sum(shell_energy(reference, positions + offset, self.material))
                - sum(shell_energy(reference, positions - offset, self.material))
            ) / 2e-6
        np.testing.assert_allclose(residual, numerical, atol=2e-9, rtol=2e-7)
        np.testing.assert_allclose(residual.sum(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.cross(positions, residual).sum(axis=0), 0, atol=1e-12)

    def test_boundary_extrapolation(self):
        """Eliminate missing hinge vertices and retain their chain-rule contribution."""
        reference = flat_stencil()
        positions = reference.copy()
        positions[1] += [0.1, 0.03, 0.04]
        boundary = (True, True, True)
        residual = shell_residual(reference, positions, self.material, boundary=boundary)
        np.testing.assert_array_equal(residual[3:], 0)
        self.assertAlmostEqual(shell_energy(reference, positions, self.material, boundary=boundary)[1], 0, places=25)
        changed = positions.copy()
        changed[3:] += 100
        np.testing.assert_array_equal(
            shell_energy(reference, changed, self.material, boundary=boundary),
            shell_energy(reference, positions, self.material, boundary=boundary),
        )
        boundary = (False, False, True)
        positions[3, 2] += 0.2
        offset = np.zeros_like(positions)
        offset[1, 2] = 1e-6
        numerical = (
            sum(shell_energy(reference, positions + offset, self.material, boundary=boundary))
            - sum(shell_energy(reference, positions - offset, self.material, boundary=boundary))
        ) / 2e-6
        self.assertAlmostEqual(
            shell_residual(reference, positions, self.material, boundary=boundary)[1, 2], numerical, places=8
        )

    def test_length_scaling(self):
        """Scale membrane energy by area while preserving curvature energy."""
        reference, positions = folded_stencil(0.2), folded_stencil(0.7)
        positions[1, 0] += 0.1
        membrane, bending = shell_energy(reference, positions, self.material)
        for scale in (1e-3, 10.0):
            np.testing.assert_allclose(
                shell_energy(reference * scale, positions * scale, self.material),
                [membrane * scale**2, bending],
                rtol=1e-12,
                atol=1e-14,
            )

    def test_invalid_inputs(self):
        """Reject invalid material, topology, finite data and singular geometry."""
        reference = flat_stencil()
        bad_materials = (
            (30, 0, 0.02, 0.07),
            (-40, 40, 0.02, 0.07),
            (30, 40, 0.02, 0),
            (30, 40, -0.035, 0.07),
            (np.nan, 40, 0.02, 0.07),
        )
        for material in bad_materials:
            for evaluate in (shell_energy, shell_residual, shell_membrane_derivatives):
                with self.subTest(material=material, evaluate=evaluate.__name__), self.assertRaises(ValueError):
                    evaluate(reference, reference, material)
        invalid = [reference[:3], reference * np.nan, folded_stencil(np.pi)]
        collapsed = reference.copy()
        collapsed[2] = collapsed[1]
        invalid.append(collapsed)
        for points in invalid:
            with self.subTest(points=points.tolist()):
                for source, target in ((reference, points), (points, reference)):
                    with self.assertRaises(ValueError):
                        shell_energy(source, target, self.material)
        for boundary in ((False,), (0, 0, 1)):
            with self.assertRaises(ValueError):
                shell_energy(reference, reference, self.material, boundary=boundary)


if __name__ == "__main__":
    unittest.main()
