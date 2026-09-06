# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the opt-in monolithic release profiler contracts."""

import unittest

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.profile_release import (
    _select_stiffness_lower_bound,
    profile_collision_kernels,
    run_profiled_normal,
)


class TestReleaseProfileHelpers(unittest.TestCase):
    def test_stiffness_lower_bound_is_fail_closed(self):
        rows = [
            {"stiffness_n_m3": 2.0, "physical_numerical_gate": "PASS"},
            {"stiffness_n_m3": 1.0, "physical_numerical_gate": "FAIL"},
            {"stiffness_n_m3": 4.0, "physical_numerical_gate": "PASS"},
        ]
        self.assertEqual(_select_stiffness_lower_bound(rows), 2.0)
        self.assertIsNone(_select_stiffness_lower_bound([*rows[:1], {"stiffness_n_m3": 1.0}]))
        self.assertIsNone(_select_stiffness_lower_bound([]))


def test_collision_profile(test, device):
    result = profile_collision_kernels(device, repeats=2, medium_level=1)
    test.assertEqual({row["asset"] for row in result}, {"formal_single_finger", "refined_tet_level_1"})
    for row in result:
        test.assertGreater(row["face_pair_count"], 0)
        test.assertEqual(row["near_field_sdf_queries_per_pair"]["p1q3"], 4)
        test.assertEqual(row["near_field_sdf_queries_per_pair"]["adaptive_face"], 482)
        for method in ("p1q3", "adaptive_face"):
            test.assertEqual(len(row[method]["samples_ms"]), 2)
            test.assertGreaterEqual(row[method]["record_count"], 0)


def test_short_profiled_normal(test, device):
    result = run_profiled_normal(device, variant="actor_block", substeps=3, allocation_audit=True)
    test.assertEqual(result["variant"], "actor_block")
    test.assertEqual(result["substeps"], 3)
    test.assertEqual(len(result["step_ms_samples"]), 3)
    test.assertTrue(result["all_finite"])
    audit = result["allocation_audit"]
    test.assertEqual(audit["allocation_gate"], "PASS")
    test.assertEqual(audit["unexpected_python_device_allocations"], 0)
    test.assertIn("collision_candidate", result["stages"])
    test.assertIn("whole_step", result["percentiles_ms"])


class TestReleaseProfileDevices(unittest.TestCase):
    """Exercise actual CPU/CUDA kernels without running the long release matrix."""


add_function_test(
    TestReleaseProfileDevices, "test_collision_profile", test_collision_profile, devices=get_test_devices()
)
add_function_test(
    TestReleaseProfileDevices,
    "test_short_profiled_normal",
    test_short_profiled_normal,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
