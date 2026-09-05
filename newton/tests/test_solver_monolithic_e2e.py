# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify DRAFT normal-loading measurements without claiming a frozen E2E gate."""

import copy
import unittest

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import assess_run, command_at, load_fixture, run_loading


class TestNormalLoadingContract(unittest.TestCase):
    def test_drive_and_draft_contract(self):
        """Keep the scene draft and sample continuous free/loading/settle commands."""
        fixture = load_fixture()
        self.assertEqual(fixture["status"], "DRAFT")
        self.assertEqual(fixture["contact"]["stiffness_n_m3"], 1.0e7)
        self.assertEqual(fixture["acceptance"]["delta_soft_min_m"], 0.005)
        for time_s, expected, phase in ((0.0, 0.0, "free_space"), (0.2, 0.003, "loading"), (0.7, 0.012, "settle")):
            q, qd, actual_phase = command_at(time_s, fixture)
            self.assertAlmostEqual(q, expected)
            self.assertAlmostEqual(qd, 0.0)
            self.assertEqual(actual_phase, phase)

    def test_soft_metric_excludes_rigid_motion(self):
        """Measure apex-to-surface shortening independently of a rigid transform."""
        fixture = load_fixture()
        positions = np.asarray(fixture["soft"]["rest_positions_m"])
        distance = np.linalg.norm(positions[3] - positions[:3].mean(axis=0))
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transformed = positions @ rotation.T + [0.7, -0.4, 0.2]
        self.assertAlmostEqual(np.linalg.norm(transformed[3] - transformed[:3].mean(axis=0)), distance)


def test_short_normal_loading(test, device):
    """Measure a short real trajectory and reject incomplete or failed evidence without changing thresholds."""
    fixture = load_fixture()
    result = run_loading(fixture, device=device, substeps=12)
    test.assertEqual(len(result["records"]), 12)
    test.assertFalse(result["summary"]["gates"]["minimum_steps"])
    test.assertFalse(result["summary"]["v01_exit"])
    test.assertEqual(result["metadata"]["status"], "DRAFT")
    for i, record in enumerate(result["records"]):
        test.assertEqual(record["step"], i)
        test.assertEqual(record["phase"], "free_space")
        test.assertTrue(record["finite_state"])
        test.assertEqual(record["active_sample_count"], 0)
        test.assertEqual(record["normal_compressive_force_n"], 0.0)
        np.testing.assert_array_equal(
            record["node_positions_m"][3], np.asarray(fixture["soft"]["rest_positions_m"][3], dtype=np.float32)
        )
    invalid = copy.deepcopy(result["records"])
    for record in invalid[:3]:
        record.update(converged=False, rolled_back=True, convergence_status="nonfinite")
    invalid[0]["finite_state"] = False
    summary = assess_run(invalid, fixture)
    test.assertFalse(summary["gates"]["finite_state"])
    test.assertFalse(summary["gates"]["converged_ratio"])
    test.assertFalse(summary["gates"]["consecutive_non_success"])


class TestMonolithicE2E(unittest.TestCase):
    """Run short physical CPU/CUDA smoke trajectories for the offline measurement tool."""


add_function_test(TestMonolithicE2E, "test_short_normal_loading", test_short_normal_loading, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
