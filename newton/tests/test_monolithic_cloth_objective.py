# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare production shell elements with independent continuum derivatives."""

import itertools
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cloth import ClothObjective
from newton.tests.test_monolithic_cloth_material import flat_stencil, folded_stencil
from scripts.monolithic_reference.cloth_oracle import (
    shell_bending_tangent,
    shell_energy,
    shell_membrane_tangent,
    shell_residual,
)


class TestClothObjective(unittest.TestCase):
    def test_complete_element_derivatives(self):
        """Match native energy/residual and full projected tangents across boundaries and strains."""
        devices = ["cpu"] + ([wp.get_cuda_device()] if wp.is_cuda_available() else [])
        for device in devices:
            for dtype in (wp.float64, wp.float32):
                for boundary in itertools.product((False, True), repeat=3):
                    for stretch in (0.6, 1.0, 1.3):
                        with self.subTest(device=str(device), dtype=dtype, boundary=boundary, stretch=stretch):
                            rest = folded_stencil(0.4)
                            x = folded_stencil(0.7) * [stretch, 0.9, 1.0]
                            x[:, 0] += 0.2 * x[:, 1]
                            material = (-15.0, 40.0, 0.02, 0.07)
                            stencil = np.arange(6)[None, :]
                            stencil[0, 3:][list(boundary)] = -1
                            obj = ClothObjective(rest, stencil, [material], dtype=dtype, device=device)
                            energy, residual, tangent = obj.evaluate_host(x)
                            expected = shell_bending_tangent(rest, x, material, boundary=boundary)
                            expected[:9, :9] += shell_membrane_tangent(rest, x, material)
                            tol = 3e-5 if dtype == wp.float32 else 2e-10
                            np.testing.assert_allclose(
                                energy[0], shell_energy(rest, x, material, boundary=boundary), rtol=tol, atol=tol
                            )
                            np.testing.assert_allclose(
                                residual[0].reshape(6, 3),
                                shell_residual(rest, x, material, boundary=boundary),
                                rtol=tol,
                                atol=tol,
                            )
                            self.assertLess(np.linalg.norm(tangent[0] - expected) / np.linalg.norm(expected), 2e-5)
                            self.assertGreaterEqual(
                                np.linalg.eigvalsh(tangent[0].astype(float)).min(),
                                -2e-6 * max(1, np.linalg.norm(expected, ord=2)),
                            )
                            np.testing.assert_allclose(tangent[0], tangent[0].T, atol=1e-6)
                            for a, b in itertools.product(range(6), repeat=2):
                                block = np.s_[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
                                error = np.linalg.norm(tangent[0][block] - expected[block])
                                self.assertLessEqual(error, 2e-5 * max(np.linalg.norm(expected[block]), 1e-8))

    def test_rigid_motion_and_translation_nullspace(self):
        """Preserve natural folds under rotations and retain all rigid translations in B."""
        rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])
        rest = folded_stencil(0.4)
        obj = ClothObjective(rest, np.arange(6)[None, :], [(30, 40, 0.02, 0.07)], device="cpu")
        e, r, tangent = obj.evaluate_host(rest @ rotation.T + [2, 3, 4])
        np.testing.assert_allclose(e, 0, atol=1e-25)
        np.testing.assert_allclose(r, 0, atol=1e-11)
        for axis in np.eye(3):
            direction = np.tile(axis, 6)
            self.assertLess(
                np.linalg.norm(tangent[0] @ direction),
                2e-6 * np.linalg.norm(tangent[0], ord=2) * np.linalg.norm(direction),
            )
        self.assertGreater(np.linalg.norm(tangent[0, :3, 3:6]), 1)
        self.assertGreater(np.linalg.norm(tangent[0, 9:12, 12:15]), 1e-4)

    def test_rest_and_geometry(self):
        """Keep natural folds stress free and reject collapsed or opposing stencil normals."""
        for angle in (0.0, 0.5, np.pi / 2):
            rest = folded_stencil(angle)
            obj = ClothObjective(rest, np.arange(6)[None, :], [(30, 40, 0.02, 0.07)], device="cpu")
            e, r, _ = obj.evaluate_host(rest)
            np.testing.assert_allclose(e, 0, atol=1e-25)
            np.testing.assert_allclose(r, 0, atol=1e-11)
        obj = ClothObjective(flat_stencil(), np.arange(6)[None, :], [(30, 40, 0.02, 0.07)], device="cpu")
        for x in (flat_stencil() * [0, 1, 1], folded_stencil(np.pi), flat_stencil() * np.nan):
            with self.assertRaises(ValueError):
                obj.evaluate_host(x)


if __name__ == "__main__":
    unittest.main()
