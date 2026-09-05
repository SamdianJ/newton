# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that offline calibration records measured terms and bounded scope."""

import unittest
from dataclasses import replace

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.calibrate_components import CalibrationCase, calibrate_case, calibration_cases


class TestCalibrationInputs(unittest.TestCase):
    """Check the declared matrix without running a benchmark in ordinary tests."""

    def test_matrix_and_validation(self):
        """Record deterministic discrete coverage and reject invalid parameters."""
        cases = calibration_cases(seed=87)
        self.assertEqual(len(cases), 28)
        self.assertEqual(cases, calibration_cases(seed=87))
        self.assertEqual(len({case.case_id for case in cases}), 28)
        self.assertEqual({case.young_modulus for case in cases}, {1e3, 1e4, 1e5})
        self.assertEqual({case.poisson_ratio for case in cases}, {0.2, 0.3, 0.45})
        self.assertEqual({case.dt for case in cases}, {0.001, 0.01})
        self.assertEqual({case.coordinate_scale for case in cases}, {0.5, 1.0, 2.0})
        self.assertEqual({case.world_offset for case in cases}, {0.0, 10.0})
        for change in ({"dt": 0}, {"poisson_ratio": 0.5}, {"coordinate_scale": -1}, {"world_offset": np.nan}):
            with self.assertRaises(ValueError):
                replace(cases[0], **change)


def test_measured_active_and_inactive(test, device):
    """Measure actual forces and keep coordinate sensitivity separate from reduction noise."""
    for active in (False, True):
        case = CalibrationCase("test", 1e4, 0.3, 0.001, 1.0, 0.0, active, 87)
        result = calibrate_case(device, case=case, repeats=3)
        test.assertEqual(result["status"], "DRAFT")
        test.assertEqual(result["parameters"]["seed"], 87)
        test.assertEqual(len(result["fixture_sha256"]), 64)
        test.assertEqual(result["block_order"], ["global", "q", "x"])
        test.assertEqual(len(result["coordinate_quantization"]), 28)
        test.assertEqual(result["force_noise"]["active_sample_count"] > 0, active)
        test.assertEqual(np.linalg.norm(result["force_noise"]["physical_world_force_or_wrench"]) > 0, active)
        test.assertGreaterEqual(result["force_noise"]["recommended_force_detection_floor"], 0)
        test.assertEqual(len(result["reduction_noise"]["merit_samples"]), 3)
        test.assertEqual(len(result["linear_cancellation"]), 9)
        test.assertTrue(np.isfinite(result["diagonal"]).all())
        test.assertEqual(result["finite_difference"]["contact"]["status"] == "NOT_APPLICABLE", not active)
        for name in ("tet", "fk") + (("contact",) if active else ()):
            test.assertGreaterEqual(len(result["finite_difference"][name]["records"]), 5)
        test.assertEqual(result["finite_difference"]["tet"]["status"], "PASS")
        test.assertEqual(result["finite_difference"]["fk"]["status"], "PASS")
        if active:
            test.assertEqual(result["finite_difference"]["contact"]["status"], "PASS")
            test.assertGreater(result["force_noise"]["coordinate_ulp_resultant_change_max"], 0)


class TestMonolithicCalibration(unittest.TestCase):
    """Run a small production-path smoke test on each available device."""


for device in get_test_devices():
    add_function_test(
        TestMonolithicCalibration,
        test_measured_active_and_inactive.__name__,
        test_measured_active_and_inactive,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
