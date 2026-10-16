# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate analytic benchmark invariants before comparing any production output."""

import unittest

import numpy as np

from scripts.monolithic_reference.cable_analytic import cantilever_deflection, localized_helix
from scripts.monolithic_reference.cable_benchmarks import CableBenchmark


class TestMonolithicCableAnalytic(unittest.TestCase):
    def test_cantilever_load_and_rigidity(self):
        """Check the clamp, tip compliance and linear force/rigidity scaling."""
        s = np.linspace(0, 1, 17)
        w = cantilever_deflection(s, length=1, force=3e-6, bending=1e-3)
        self.assertEqual(w[0], 0)
        self.assertAlmostEqual(w[-1], 1e-3, places=15)
        np.testing.assert_allclose(cantilever_deflection(s, length=1, force=6e-6, bending=2e-3), w)

    def test_localized_geometry_and_boundary_relations(self):
        """Recover the published tilt and verify inextensibility and twist/writhe balance."""
        s = np.linspace(-9.29 / 2, 9.29 / 2, 10001)
        result = localized_helix(s, length=9.29, bending=1.345, twisting=0.789, rotation=54 * np.pi, shortening=0.3)
        self.assertAlmostEqual(result["max_tilt"], 0.919, delta=0.002)
        np.testing.assert_allclose(np.linalg.norm(result["tangents"], axis=1), 1, atol=2e-15)
        x = result["positions"]
        derivative = (x[:-4] - 8 * x[1:-3] + 8 * x[3:-1] - x[4:]) / (12 * (s[1] - s[0]))
        np.testing.assert_allclose(derivative, result["tangents"][2:-2], atol=1e-9)
        cosine = result["tangents"][:, 0]
        envelope = (cosine - np.cos(result["max_tilt"])) / (1 - np.cos(result["max_tilt"]))
        np.testing.assert_allclose(envelope, result["envelope"], atol=2e-15)
        self.assertLess(abs(result["finite_shortening"] - 0.3), 1e-9)
        self.assertLess(abs(result["finite_end_rotation"] - 54 * np.pi), 1e-8)
        self.assertAlmostEqual(
            result["twist_density"] * 9.29 + 2 * np.pi * result["writhe_turns"],
            result["finite_end_rotation"],
            places=12,
        )

    def test_mesh_samples_still_require_physical_gates(self):
        """Reject moving mesh samples and retain analytical gates for low EA."""
        case = CableBenchmark.__new__(CableBenchmark)
        case.kind, case.ea_multiplier = "localized_helix", 0.1
        case.steps = case.total_steps = 3000
        case.recipe = {"gates": {"relative_centerline_error": 0.03, "velocity": 1e-5}}
        result = {"complete": True, "relative_centerline_error": 0.1, "velocity": 1e-3}
        case.metrics = lambda: result
        with self.assertRaises(AssertionError):
            case.check_final(accuracy=False)
        result["velocity"] = 0
        case.check_final(accuracy=False)
        with self.assertRaises(AssertionError):
            case.check_final()
        result["relative_centerline_error"] = 0.01
        case.check_final()
        result["complete"] = False
        with self.assertRaises(AssertionError):
            case.check_final(accuracy=False)

    def test_invalid_or_unselected_branch(self):
        """Reject inputs for which this analytic branch is not an applicable reference."""
        with self.assertRaises(ValueError):
            localized_helix([0], length=1, bending=1, twisting=1, rotation=100, shortening=0.5)
        with self.assertRaises(ValueError):
            cantilever_deflection([-0.1], length=1, force=1, bending=1)

    def test_nonfinite_metrics_reject(self):
        """Reject nonfinite physical, approximation and diagnostic metrics on every mesh."""
        case = CableBenchmark.__new__(CableBenchmark)
        case.kind, case.steps, case.total_steps = "cantilever", 600, 600
        case.recipe = {"gates": {"relative_centerline_error": 0.03, "velocity": 1e-5}}
        valid = {"complete": True, "relative_centerline_error": 0.01, "velocity": 0, "transverse_phase": 0}
        for name in ("relative_centerline_error", "velocity", "transverse_phase"):
            for value in (float("nan"), float("inf"), -float("inf")):
                for accuracy in (False, True):
                    with self.subTest(name=name, value=value, accuracy=accuracy), self.assertRaises(AssertionError):
                        case.metrics = lambda name=name, value=value: {**valid, name: value}
                        case.check_final(accuracy=accuracy)


if __name__ == "__main__":
    unittest.main()
