# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

from scripts.monolithic_reference.tet_analytic import (
    fixed_free_frequency,
    homogeneous_traction,
    implicit_euler_oscillation,
    oscillation_period_and_decay,
    timoshenko_tip_deflection,
    young_poisson,
)
from scripts.monolithic_reference.tet_benchmarks import TetBenchmark


class TestTetAnalytic(unittest.TestCase):
    def test_lame_conversion(self):
        self.assertEqual(young_poisson(1.0, 0.0), (2.0, 0.0))
        young, nu = young_poisson(10000 / 2.6, 10000 * 0.3 / (1.3 * 0.4))
        self.assertAlmostEqual(young, 10000.0, places=8)
        self.assertAlmostEqual(nu, 0.3, places=12)
        with self.assertRaises(ValueError):
            young_poisson(0.0, 1.0)

    def test_rest_is_stress_free_and_small_strain_is_linear(self):
        for smith in (False, True):
            np.testing.assert_allclose(homogeneous_traction(np.eye(3), 2.0, 3.0, smith=smith), 0.0, atol=1e-14)
            eps = 1e-6
            stress = homogeneous_traction(np.diag([1 + eps, 1, 1]), 1.0, 0.0, smith=smith)
            self.assertLess(abs(stress[0, 0] / (2 * eps) - 1), 1e-4)
            self.assertLess(abs(stress[1, 1]), 1e-10)
        with self.assertRaises(ValueError):
            homogeneous_traction(np.diag([1.0, 1.0, -1.0]), 1.0, 0.0, smith=True)

    def test_timoshenko_tip_deflection(self):
        value = timoshenko_tip_deflection(1.0, 1.0, 0.1, 0.1, 1e6, 0.0)
        self.assertAlmostEqual(value, 0.02 + 1.2e-4, delta=1e-12)

    def test_fixed_free_frequency(self):
        self.assertAlmostEqual(fixed_free_frequency(1.0, 1.0, 0.0, 2.0), math.pi / 2, places=14)
        with self.assertRaises(ValueError):
            fixed_free_frequency(1.0, 1.0, 1.0, 2.0)

    def test_implicit_euler_oscillation(self):
        period, ratio = implicit_euler_oscillation(1.0, 1.0)
        self.assertAlmostEqual(period, 8.0, places=12)
        self.assertAlmostEqual(ratio, 1 / 16, places=12)
        period, ratio = implicit_euler_oscillation(100.0, 1e-5)
        self.assertLess(abs(period / (2 * math.pi / 100) - 1), 1e-6)
        self.assertLess(abs(ratio / math.exp(-math.pi * 1e-3) - 1), 1e-3)

    def test_period_and_decay_measurement(self):
        dt, omega, decay = 1e-3, 2 * math.pi, 0.9
        t = np.arange(4000) * dt
        signal = decay ** (t / 1.0) * np.sin(omega * t)
        period, ratio = oscillation_period_and_decay(signal, dt)
        self.assertLess(abs(period - 1.0), 2e-3)
        self.assertLess(abs(ratio - decay), 2e-3)
        with self.assertRaises(ValueError):
            oscillation_period_and_decay(np.ones(10), dt)


class TestTetBenchmarkSmoke(unittest.TestCase):
    """Advance each closed-form tet experiment a few public solver steps on the CPU."""

    def test_nodal_mass_matches_body_mass(self):
        case = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped")
        expected = case.recipe["density"] * np.prod(case.size)
        self.assertAlmostEqual(case.nodal_mass.sum() / expected, 1.0, delta=1e-6)
        dynamic = ~case.fixed
        np.testing.assert_allclose(case.model.particle_mass.numpy()[dynamic], case.nodal_mass[dynamic], rtol=1e-6)

    def test_patch_returns_toward_homogeneous_state(self):
        case = TetBenchmark("patch", case="tension", cells=(2, 2, 2), material_model="kim_stable_no_log")
        before = case.interior_error()
        for _ in range(3):
            case.step()
        self.assertLess(case.interior_error(), 0.1 * before)

    def test_cantilever_bends_downward(self):
        case = TetBenchmark("cantilever", cells=(10, 1, 1))
        for _ in range(3):
            case.step()
        self.assertLess(case.history[-1][2], 0.0)

    def test_vibration_starts_outward(self):
        case = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped")
        for _ in range(5):
            case.step()
        self.assertGreater(case.history[-1][0], 0.0)


if __name__ == "__main__":
    unittest.main()
