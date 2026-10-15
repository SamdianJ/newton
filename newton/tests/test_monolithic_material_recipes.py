# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate physical rest geometry and analytic material scales of frozen acceptance recipes."""

import unittest

import numpy as np

from scripts.monolithic_reference.cable_oracle import CableFrame, CableReference, cable_energy
from scripts.monolithic_reference.cloth_oracle import shell_energy, shell_residual
from scripts.monolithic_reference.material_recipes import build_case, shell_coefficients, shell_stencils


class TestMonolithicMaterialRecipes(unittest.TestCase):
    def test_shell_rest_and_mass(self):
        """Verify exact recipe areas, lumped mass and zero rest force including folded box edges."""
        for name, expected_area in (("cloth-press", 0.04), ("precreased-box", 0.0224)):
            case = build_case(name)
            recipe = case["recipe"]
            self.assertAlmostEqual(case["masses"].sum(), expected_area * recipe["surface_density_kg_m2"], places=14)
            self.assertTrue(np.all(case["masses"] > 0))
            material = shell_coefficients(recipe)
            for stencil, missing in shell_stencils(case["positions"], case["triangles"]):
                np.testing.assert_allclose(shell_energy(stencil, stencil, material, boundary=missing), 0, atol=1e-20)
                np.testing.assert_allclose(shell_residual(stencil, stencil, material, boundary=missing), 0, atol=1e-8)

    def test_cable_scale_and_mass(self):
        """Verify the frozen cable mass, axial stiffness and unwrapped twist span analytically."""
        case = build_case("cable-bend-twist")
        recipe, points = case["recipe"], case["positions"]
        theta = np.zeros(len(points) - 1)
        directors = np.tile([0.0, 1, 0], (len(theta), 1))
        rest = CableReference(points, theta, directors, (recipe["EA_N"], recipe["bending_Nm2"], recipe["GJ_Nm2"]))
        frame = CableFrame(points, directors)
        self.assertAlmostEqual(case["masses"].sum(), recipe["length_m"] * recipe["linear_density_kg_m"], places=15)
        theta = np.linspace(0, recipe["end_theta_load_rad"], len(theta))
        energy = cable_energy(rest, points * 1.001, theta, frame)
        expected_twist = 0.5 * recipe["GJ_Nm2"] * theta[-1] ** 2 / rest.dual_lengths.sum()
        np.testing.assert_allclose(
            energy, [0.5 * recipe["EA_N"] * recipe["length_m"] * 0.001**2, 0, expected_twist], atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
