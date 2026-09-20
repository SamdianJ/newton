# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check physical time and available evidence in the PR1 accuracy diagnostic."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import load_fraction
from scripts.monolithic_reference.diagnose_p3_tet_accuracy import diagnose
from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.verify_p3_tet import save


class TestP3AccuracyDiagnostic(unittest.TestCase):
    def source_data(self, dt, *, scene="fall"):
        """Create exact free-fall samples on the common 1 ms output grid."""
        rest = build_model("cpu", cantilever=scene == "transverse").particle_q.numpy().astype(float)
        times = np.arange(1, 5) * 0.001
        steps = np.rint(times / dt).astype(int)
        return {
            "scene": scene,
            "dt": dt,
            "material": "kim_stable_no_log",
            "mass": "consistent",
            "mode": "ArmijoOrResNorm",
            "records": [{"time": t, "newton": 0, "linear_tip": [0, 0, 0.001]} for t in times],
            "states": [rest + np.array([0, 0, -9.81]) * dt**2 * n * (n + 1) / 2 for n in steps],
            "gates": {"analytic_error": 0.0, "linear_error": 0.0},
        }

    def run_diagnostic(self, data):
        """Pass a serialized trajectory through the real diagnostic entry point."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectory.json"
            save(source, data)
            return diagnose(source)

    def test_refined_free_fall(self):
        """Compare sampled free fall at the actual final physical time."""
        for dt in (0.0005, 0.00025):
            with self.subTest(dt=dt):
                result = self.run_diagnostic(self.source_data(dt))
                np.testing.assert_allclose(result["production_center_of_mass_error"], 0, atol=1e-14)
                self.assertLess(result["independent_analytic_error"], 1e-8)
                self.assertEqual(result["steps"], round(0.004 / dt))
                self.assertIsNone(result["production_residual_sum"])
                self.assertIsNone(result["residual_integrated_center_of_mass_error"])
                self.assertIsNone(result["production_zero_iteration_steps"])
                self.assertIn("missing", result["production_residual_unavailable_reason"])

    def test_baseline_residual_diagnostic(self):
        """Retain per-step residual reconstruction for the dense 1 ms baseline."""
        result = self.run_diagnostic(self.source_data(0.001))
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["production_zero_iteration_steps"], 4)
        self.assertEqual(len(result["production_residual_sum"]), 4)
        np.testing.assert_allclose(result["production_center_of_mass_error"], 0, atol=1e-14)
        np.testing.assert_allclose(result["residual_integrated_center_of_mass_error"], 0, atol=1e-8)

    def test_refined_load_times(self):
        """Freeze the transverse force at every physical substep time."""
        for dt in (0.0005, 0.00025):
            with self.subTest(dt=dt):
                with patch(
                    "scripts.monolithic_reference.diagnose_p3_tet_accuracy.load_fraction", wraps=load_fraction
                ) as load:
                    result = self.run_diagnostic(self.source_data(dt, scene="transverse"))
                expected = np.arange(1, round(0.004 / dt) + 1) * dt
                np.testing.assert_allclose([call.args[0] for call in load.call_args_list], expected, atol=1e-15)
                self.assertEqual(result["steps"], len(expected))
                self.assertTrue(np.isfinite(result["independent_linear_relative_error"]))

    def test_invalid_sample_times(self):
        """Reject nonfinite, nonincreasing and off-step sample times."""
        for times in ([0.001, 0.001], [0.001, 0.0005], [0.001, 0.00125], [0.001, float("inf")], [1e-14, 0.001]):
            with self.subTest(times=times):
                data = self.source_data(0.0005)
                data["records"] = [dict(data["records"][0], time=t) for t in times]
                data["states"] = data["states"][:2]
                with self.assertRaisesRegex(ValueError, "sample times"):
                    self.run_diagnostic(data)

    def test_invalid_step_and_state_count(self):
        """Reject invalid physical dt and unmatched sampled states before reconstruction."""
        for dt in (0, -0.001, float("inf")):
            with self.subTest(dt=dt):
                data = self.source_data(0.001)
                data["dt"] = dt
                with self.assertRaisesRegex(ValueError, "physical dt"):
                    self.run_diagnostic(data)
        data = self.source_data(0.001)
        data["states"].pop()
        with self.assertRaisesRegex(ValueError, "one production state"):
            self.run_diagnostic(data)


if __name__ == "__main__":
    unittest.main()
