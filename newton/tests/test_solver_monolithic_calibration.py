# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that offline calibration records measured terms and bounded scope."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.calibrate_components import (
    CalibrationCase,
    CalibrationFreezeError,
    _coordinate_envelope,
    _fd_status,
    aggregate_calibration_evidence,
    calibrate_case,
    calibration_cases,
    freeze_calibration_cases,
)


class TestCalibrationInputs(unittest.TestCase):
    """Check the declared matrix without running a benchmark in ordinary tests."""

    def test_fd_requires_adjacent_steps(self):
        """A single favorable step cannot establish a reproducible FD plateau."""
        records = [
            {"step": h, "derivative": {"relative_error": error}}
            for h, error in ((0.1, 0.02), (0.01, 0.001), (0.001, 0.02))
        ]
        self.assertEqual(_fd_status(records, ("derivative",), "cpu")["status"], "OUTSIDE_MEASURED_GATE")
        records[-1]["derivative"]["relative_error"] = 0.002
        self.assertEqual(_fd_status(records, ("derivative",), "cpu")["status"], "PASS")
        records[-1]["fixed_active_set"] = False
        self.assertEqual(_fd_status(records, ("derivative",), "cpu")["status"], "OUTSIDE_MEASURED_GATE")

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

    def test_freeze_matrix_declares_required_discrete_coverage(self):
        """The full profile names every required axis without claiming intervals."""
        cases = freeze_calibration_cases()
        self.assertEqual({case.seed for case in cases}, {20260905, 20260917, 20260929})
        self.assertEqual({case.direction_index for case in cases}, {0, 1, 2})
        self.assertEqual({case.tet_count for case in cases}, {1, 8, 64})
        self.assertEqual({case.requested_contact_cardinality for case in cases}, {0, 3, 12, 48})
        self.assertEqual({case.boundary_mode for case in cases}, {"all_dynamic", "fixed_opposite"})
        self.assertEqual({case.contact_stiffness for case in cases}, {2e5, 1e7})
        self.assertEqual({case.young_modulus for case in cases}, {1e3, 1e4, 1e5})
        self.assertEqual({case.poisson_ratio for case in cases}, {0.2, 0.3, 0.45})
        self.assertEqual({case.dt for case in cases}, {0.001, 0.01})
        self.assertEqual({case.coordinate_scale for case in cases}, {0.5, 1.0, 2.0})
        sweep = [case for case in cases if case.coordinate_role == "coordinate_sweep"]
        self.assertEqual({case.world_offset for case in sweep}, {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0})
        self.assertEqual({case.coordinate_pattern for case in sweep}, {"positive_axis", "negative_axis", "mixed"})
        outside = [case for case in cases if case.coordinate_role == "outside_control"]
        self.assertTrue(outside)
        self.assertEqual({case.world_offset for case in outside}, {10.0})
        self.assertEqual(len(cases), 26)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))

    def test_freeze_aggregator_fails_closed(self):
        """Labels alone cannot freeze an incomplete or failed evidence set."""
        case = freeze_calibration_cases()[0]
        record = {
            "status": "RAW_MEASUREMENT",
            "device": "cpu",
            "parameters": asdict(case),
            "fixture_sha256": "a" * 64,
            "observed": {"tet_count": case.tet_count, "active_sample_count": case.requested_contact_cardinality},
            "gates": {"finite_difference": "PASS", "reduction": "PASS", "linear": "PASS", "trajectory": "PASS"},
        }
        raw = {
            "artifact_kind": "raw_measurement",
            "calibration_schema_version": 2,
            "profile": "freeze",
            "git_dirty": False,
            "source_set_sha256": "b" * 64,
            "devices": ["cpu", "cuda"],
            "evidence": [record],
        }
        candidate = aggregate_calibration_evidence([raw])
        self.assertEqual(candidate["calibration_status"], "INCOMPLETE")
        self.assertEqual(candidate["v01_status"], "DRAFT")
        self.assertTrue(candidate["missing_case_keys"])
        with self.assertRaises(CalibrationFreezeError):
            aggregate_calibration_evidence([raw], freeze=True)
        raw["git_dirty"] = True
        with self.assertRaises(CalibrationFreezeError):
            aggregate_calibration_evidence([raw])

    def test_coordinate_envelope_stops_at_first_common_failure(self):
        """Later passing points cannot recover a failed CPU/CUDA common prefix."""
        records = []
        for device in ("cpu", "cuda"):
            for case in freeze_calibration_cases():
                if case.coordinate_role != "coordinate_sweep":
                    continue
                passed = case.world_offset != 0.5
                records.append(
                    {
                        "device_class": device,
                        "case": case,
                        "raw": {
                            "coordinate_support_status": "PASS" if passed else "OUTSIDE_MEASURED_GATE",
                            "gates": {
                                "finite_difference": "PASS" if passed else "FAIL",
                                "reduction": "PASS",
                                "linear": "PASS",
                            },
                        },
                    }
                )
        result = _coordinate_envelope(records)
        self.assertEqual(result["max_supported_abs_coordinate_m"], 0.25)
        self.assertEqual(result["first_failed_magnitude_m"], 0.5)
        self.assertEqual(result["passing_points_after_first_failure_not_used"], [1.0, 2.0, 5.0])

    def test_complete_passing_evidence_freezes_only_calibration_substate(self):
        """A complete manifest freezes calibration but never V0.1 acceptance."""
        records = []
        for device in ("cpu", "cuda:0"):
            for case in freeze_calibration_cases():
                outside = case.coordinate_role == "outside_control"
                records.append(
                    {
                        "status": "RAW_MEASUREMENT",
                        "device": device,
                        "parameters": asdict(case),
                        "fixture_sha256": "a" * 64,
                        "observed": {
                            "tet_count": case.tet_count,
                            "active_sample_count": case.requested_contact_cardinality,
                        },
                        "gates": {
                            "finite_difference": "FAIL" if outside else "PASS",
                            "reduction": "PASS",
                            "linear": "PASS",
                            "trajectory": "PASS",
                        },
                        "coordinate_support_status": "OUTSIDE_MEASURED_GATE" if outside else "PASS",
                        "draft_parameters": {
                            "residual_floors": [3.0, 1.0, 2.0],
                            "merit_absolute": [0.3, 0.1, 0.2],
                            "small_scaled_step": [0.03, 0.01, 0.02],
                            "merit_noise": 0.0,
                        },
                        "force_noise": {"recommended_force_detection_floor": 4e-6},
                    }
                )
        raw = {
            "artifact_kind": "raw_measurement",
            "calibration_schema_version": 2,
            "profile": "freeze",
            "git_dirty": False,
            "source_set_sha256": "b" * 64,
            "evidence": records,
        }
        trajectories = {
            "artifact_kind": "trajectory_evidence",
            "calibration_schema_version": 2,
            "profile": "freeze",
            "git_dirty": False,
            "source_set_sha256": "b" * 64,
            "evidence": [
                {
                    "device": device,
                    "role": role,
                    "status": "PASS",
                    "fixture_sha256": "c" * 64,
                    "finite_state": True,
                    "convergence_gate": "PASS",
                    "representative_states": ["free", "onset", "loading", "peak", "settled"],
                    "candidate_vs_strict_baseline": "PASS",
                    "false_convergence_count": 0,
                    "support_artifact_sha256": "d" * 64,
                }
                for device in ("cpu", "cuda:0")
                for role in ("normal_loading_1000", "c4_supported_motion")
            ],
        }
        result = aggregate_calibration_evidence([raw, trajectories], freeze=True)
        self.assertEqual(result["calibration_status"], "FROZEN")
        self.assertEqual(result["v01_status"], "DRAFT")
        self.assertEqual(result["coordinate_envelope"]["max_supported_abs_coordinate_m"], 5.0)
        self.assertEqual(
            result["candidate_private_config"]["cpu"]["solver_internal_config"]["residual_floor_global"], 3.0
        )
        self.assertEqual(result["portable_solver_internal_config"]["residual_floor_global"], 3.0)
        self.assertEqual(result["portable_acceptance"]["force_detection_floor_n"], 4e-6)

    def test_versioned_candidate_is_not_runtime_authority(self):
        """The checked-in proposal is machine-readable but explicitly unfrozen."""
        path = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/calibration_candidate_v2.json"
        candidate = json.loads(path.read_text())
        self.assertEqual(candidate["calibration_schema_version"], 2)
        self.assertEqual(candidate["calibration_status"], "INCOMPLETE")
        self.assertEqual(candidate["v01_status"], "DRAFT")
        self.assertTrue(candidate["runtime_use"].startswith("PROHIBITED"))
        self.assertEqual(set(candidate["candidate_private_config"]), {"cpu", "cuda"})
        fields = {
            "epsilon_d",
            "residual_floor_global",
            "residual_floor_q",
            "residual_floor_x",
            "merit_noise",
            "merit_absolute_global",
            "merit_absolute_q",
            "merit_absolute_x",
            "merit_relative_global",
            "merit_relative_q",
            "merit_relative_x",
            "step_tolerance_global",
            "step_tolerance_q",
            "step_tolerance_x",
            "det_f_guard",
            "regularization_values",
        }
        self.assertEqual(set(candidate["portable_solver_internal_config"]), fields)
        self.assertNotIn("force_detection_floor_n", candidate["portable_solver_internal_config"])
        self.assertEqual(candidate["portable_acceptance"]["force_detection_floor_n"], 4e-6)


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
        for block in ("q", "x"):
            test.assertLess(result["force_noise"][f"scaled_projection_error_{block}"]["relative_error"], 1e-4)
        test.assertEqual(len(result["reduction_noise"]["merit_samples"]), 3)
        test.assertEqual(len(result["linear_cancellation"]), 9)
        test.assertTrue(np.isfinite(result["diagonal"]).all())
        test.assertEqual(result["finite_difference"]["contact"]["status"] == "NOT_APPLICABLE", not active)
        for name in ("tet", "fk") + (("contact",) if active else ()):
            test.assertGreaterEqual(len(result["finite_difference"][name]["records"]), 5)
        test.assertEqual(result["finite_difference"]["tet"]["status"], "PASS")
        test.assertEqual(np.asarray(result["finite_difference"]["tet"]["physical_world_force_or_wrench"]).shape, (4, 3))
        test.assertEqual(result["finite_difference"]["fk"]["status"], "PASS")
        if active:
            test.assertEqual(result["finite_difference"]["contact"]["status"], "PASS")
            test.assertGreater(result["force_noise"]["coordinate_ulp_resultant_change_max"], 0)


def test_freeze_profile_smoke(test, device):
    """Exercise a refined fixed-node/contact-rich raw record without freezing it."""
    case = next(
        case
        for case in freeze_calibration_cases()
        if case.mesh_level == 1
        and case.boundary_mode == "fixed_opposite"
        and case.contact_stiffness == 1e7
        and case.active_contact
    )
    result = calibrate_case(device, case=case, repeats=2, profile="freeze")
    test.assertEqual(result["status"], "RAW_MEASUREMENT")
    test.assertEqual(result["observed"], {"tet_count": 8, "active_sample_count": 12})
    test.assertEqual(result["gates"]["contact_cardinality"], "PASS")
    test.assertEqual(result["gates"]["trajectory"], "NOT_MEASURED")


class TestMonolithicCalibration(unittest.TestCase):
    """Run a small production-path smoke test on each available device."""


for device in get_test_devices():
    add_function_test(
        TestMonolithicCalibration,
        test_measured_active_and_inactive.__name__,
        test_measured_active_and_inactive,
        devices=[device],
    )
    add_function_test(
        TestMonolithicCalibration,
        test_freeze_profile_smoke.__name__,
        test_freeze_profile_smoke,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
