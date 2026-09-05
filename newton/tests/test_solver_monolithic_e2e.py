# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify DRAFT normal-loading measurements without claiming a frozen E2E gate."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import (
    LEGACY_FIXTURE,
    SolverMonolithic,
    _v01_exit,
    assess_run,
    command_at,
    load_fixture,
    run_loading,
)


class TestNormalLoadingContract(unittest.TestCase):
    def test_drive_and_versioned_status_contract(self):
        """Keep independent evidence axes and sample the declared command."""
        fixture = load_fixture()
        self.assertEqual(fixture["status"], "DRAFT")
        self.assertEqual(fixture["schema_version"], "normal_loading/v2")
        self.assertEqual(fixture["calibration_status"], "UNFROZEN")
        self.assertEqual(fixture["support_status"], "UNFROZEN")
        self.assertEqual(fixture["reference_status"], "BLOCKED")
        self.assertIn("active-contact", fixture["reference_provenance"]["reason"])
        self.assertEqual(fixture["contact"]["stiffness_n_m3"], 1.0e7)
        self.assertEqual(fixture["acceptance"]["delta_soft_min_m"], 0.005)
        for time_s, expected, phase in ((0.0, 0.0, "free_space"), (0.2, 0.003, "loading"), (0.7, 0.012, "settle")):
            q, qd, actual_phase = command_at(time_s, fixture)
            self.assertAlmostEqual(q, expected)
            self.assertAlmostEqual(qd, 0.0)
            self.assertEqual(actual_phase, phase)

    def test_legacy_draft_remains_loadable(self):
        fixture = load_fixture(LEGACY_FIXTURE)
        self.assertEqual(fixture["schema_version"], "normal_loading_draft/v1")
        self.assertEqual(fixture["calibration_status"], "UNFROZEN")

    def test_frozen_axis_requires_versioned_hash_but_is_not_globally_blocked(self):
        fixture = load_fixture()
        fixture["calibration_status"] = "FROZEN"
        fixture["calibration_provenance"] = {
            "version": "monolithic_calibration/v1",
            "sha256": "a" * 64,
            "candidate_private_config": {"cpu": {"residual_floor_global": 1.0e-6}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture))
            loaded = load_fixture(path)
            self.assertEqual(loaded["calibration_status"], "FROZEN")
            for invalid in (None, {"version": "v1", "sha256": None}, {"version": "v1", "sha256": "bad"}):
                changed = copy.deepcopy(fixture)
                changed["calibration_provenance"] = invalid
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_fixture(path)

    def test_candidate_and_blocked_statuses_do_not_count_as_frozen(self):
        fixture = load_fixture()
        fixture["calibration_status"] = "CANDIDATE"
        fixture["support_status"] = "CANDIDATE"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture))
            load_fixture(path)
            summary = assess_run([], fixture)
            self.assertFalse(summary["gates"]["frozen_calibration"])
            self.assertFalse(summary["gates"]["frozen_support"])
            self.assertFalse(summary["gates"]["frozen_reference"])
            changed = copy.deepcopy(fixture)
            changed["reference_provenance"] = None
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_fixture(path)

    def test_v01_exit_requires_numerics_and_all_three_frozen_axes(self):
        gates = {
            "e2e_numerical_pass": True,
            "frozen_calibration": True,
            "frozen_support": True,
            "frozen_reference": True,
        }
        self.assertTrue(_v01_exit(gates))
        for name in gates:
            with self.subTest(name=name):
                changed = dict(gates)
                changed[name] = False
                self.assertFalse(_v01_exit(changed))

    def test_soft_metric_excludes_rigid_motion(self):
        """Measure apex-to-surface shortening independently of a rigid transform."""
        fixture = load_fixture()
        positions = np.asarray(fixture["soft"]["rest_positions_m"])
        distance = np.linalg.norm(positions[3] - positions[:3].mean(axis=0))
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transformed = positions @ rotation.T + [0.7, -0.4, 0.2]
        self.assertAlmostEqual(np.linalg.norm(transformed[3] - transformed[:3].mean(axis=0)), distance)

    def test_reject_unsupported_or_invalid_fixture(self):
        """Reject declarations this exact single-tet Z-plane runner cannot implement."""
        fixture = load_fixture()
        invalid = (
            ("rigid", "shape_type", "sphere"),
            ("rigid", "joint_type", "REVOLUTE"),
            ("contact", "quadrature", "P1Q1"),
            ("rigid", "approach_axis_world", [1, 0, 0]),
            ("rigid", "mass_kg", -1),
            ("rigid", "inertia_kg_m2", [1, 1]),
            ("soft", "fixed_nodes", [False] * 4),
            ("soft", "probe_nodes", [9]),
            ("soft", "rest_positions_m", [[0, 0, 0]] * 4),
            ("soft", "particle_radius_m", -1),
            ("contact", "stiffness_n_m3", float("nan")),
            ("drive", "free_end_s", 0),
            ("drive", "loading_end_s", 0.1),
            ("acceptance", "minimum_converged_ratio", 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for section, name, value in invalid:
                with self.subTest(field=f"{section}.{name}"):
                    changed = copy.deepcopy(fixture)
                    changed[section][name] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_fixture(path)
            for name, value in (("dt_s", 0), ("substeps", -1)):
                changed = copy.deepcopy(fixture)
                changed[name] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_fixture(path)


def test_short_normal_loading(test, device):
    """Measure a short real trajectory and reject incomplete or failed evidence without changing thresholds."""
    fixture = load_fixture()
    fixture["calibration_status"] = "FROZEN"
    fixture["calibration_provenance"] = {
        "version": "monolithic_calibration/test-v1",
        "sha256": "b" * 64,
        "candidate_private_config": {"cpu": {"residual_floor_global": 1.0e-6}},
    }
    result = run_loading(fixture, device=device, substeps=12)
    test.assertEqual(len(result["records"]), 12)
    test.assertFalse(result["summary"]["gates"]["minimum_steps"])
    test.assertFalse(result["summary"]["v01_exit"])
    test.assertEqual(result["metadata"]["status"], "DRAFT")
    test.assertEqual(result["metadata"]["calibration_status"], "FROZEN")
    test.assertEqual(result["metadata"]["support_status"], "UNFROZEN")
    test.assertEqual(result["metadata"]["reference_status"], "BLOCKED")
    test.assertEqual(result["metadata"]["calibration_version"], "monolithic_calibration/test-v1")
    test.assertEqual(result["metadata"]["calibration_sha256"], "b" * 64)
    test.assertEqual(result["metadata"]["candidate_solver_internal_config"], {"cpu": {"residual_floor_global": 1.0e-6}})
    test.assertIsNone(result["metadata"]["requested_solver_internal_config"])
    default_config = result["metadata"]["solver_internal_config"]
    test.assertNotEqual(default_config["residual_floor_global"], 1.0e-6)

    candidate = dict(default_config)
    candidate["merit_noise"] = 1.0e-12
    applied = run_loading(fixture, device=device, substeps=1, candidate_internal_config=candidate)
    test.assertEqual(applied["metadata"]["requested_solver_internal_config"], candidate)
    test.assertEqual(applied["metadata"]["solver_internal_config"], candidate)
    invalid = dict(candidate)
    invalid["unknown_field"] = 1.0
    with test.assertRaises(ValueError):
        run_loading(fixture, device=device, substeps=0, candidate_internal_config=invalid)
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

    for field in ("q_sign_projection_error", "x_sign_projection_error"):
        for value in (1.0, None, float("nan"), float("inf"), -float("inf")):
            with test.subTest(field=field, value=value):
                invalid = copy.deepcopy(result["records"])
                invalid[0][field] = value
                test.assertFalse(assess_run(invalid, fixture)["gates"]["contact_balance_and_sign"])
    # A complete-looking prefix must never erase a subsequent execution failure.
    prefix = result["records"] * 100
    failed = assess_run(prefix, fixture, execution_error={"step": 1200, "reason": "interrupted"})
    test.assertTrue(failed["gates"]["minimum_steps"])
    test.assertFalse(failed["gates"]["execution_completed"])
    test.assertFalse(failed["draft_numerical_pass"])


def test_loading_failure_preserves_prefix(test, device):
    """Keep a real successful state record when the following substep aborts."""
    original = SolverMonolithic.step
    for failure in (RuntimeError("publication failed"), KeyboardInterrupt()):
        calls = 0

        def fail_second(solver, *args, failure=failure, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original(solver, *args, **kwargs)

        with patch.object(SolverMonolithic, "step", fail_second):
            result = run_loading(load_fixture(), device=device, substeps=3)
        test.assertEqual(len(result["records"]), 1)
        test.assertTrue(result["records"][0]["converged"])
        test.assertEqual(result["metadata"]["execution_error"]["step"], 1)
        test.assertFalse(result["summary"]["gates"]["execution_completed"])
        test.assertFalse(result["summary"]["draft_numerical_pass"])


class TestMonolithicE2E(unittest.TestCase):
    """Run short physical CPU/CUDA smoke trajectories for the offline measurement tool."""


add_function_test(TestMonolithicE2E, "test_short_normal_loading", test_short_normal_loading, devices=get_test_devices())

add_function_test(
    TestMonolithicE2E,
    "test_loading_failure_preserves_prefix",
    test_loading_failure_preserves_prefix,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
