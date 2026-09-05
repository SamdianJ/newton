# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify that offline calibration records measured terms and bounded scope."""

import copy
import hashlib
import json
import subprocess
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.solver_monolithic import _SolverMonolithicInternalConfig
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.calibrate_components import (
    CalibrationCase,
    CalibrationFreezeError,
    _coordinate_envelope,
    _fd_status,
    _hash_json,
    aggregate_calibration_evidence,
    build_trajectory_evidence,
    calibrate_case,
    calibration_cases,
    freeze_calibration_cases,
)

_ROOT = Path(__file__).parents[2]


def _committed_source_identity(paths):
    git_sha = subprocess.check_output(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], text=True).strip()
    sources = {
        path: hashlib.sha256(
            subprocess.check_output(["git", "-C", str(_ROOT), "show", f"{git_sha}:{path}"])
        ).hexdigest()
        for path in paths
    }
    return {
        "git_sha": git_sha,
        "git_dirty": False,
        "source_sha256": sources,
        "source_set_sha256": _hash_json(sources),
    }


def _synthetic_raw_record(case, device):
    """Build structurally valid raw arrays; integrity/gate tests mutate these."""

    def fd(metrics):
        records = [{"step": step, **{name: {"relative_error": 0.0} for name in metrics}} for step in range(5)]
        return {"status": "PASS", "records": records}

    stored = {
        "case_id": case.case_id,
        "tet_indices": [[0, 1, 2, 3] for _ in range(case.tet_count)],
        "particle_q": [[case.world_offset, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
        "initial_particle_q": [
            [case.world_offset, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ],
        "joint_X_p": [[case.world_offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "joint_X_c": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "shape_transform": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        "deformation_seed_input": [0.0, 0.0, 0.0],
    }
    quantization = [
        {
            "scaled_step_rms": [6e-4, 2e-4, 4e-4],
            "scaled_residual_change_rms": [1.5e-4, 5e-9, 1e-4],
            "force_resultant_change": 2e-6,
        }
    ]
    record = {
        "status": "RAW_MEASUREMENT",
        "device": device,
        "parameters": asdict(case),
        "stored_fixture": stored,
        "fixture_sha256": _hash_json({"parameters": asdict(case), "stored": stored}),
        "observed": {"tet_count": case.tet_count, "active_sample_count": case.requested_contact_cardinality},
        "finite_difference": {
            "tet": fd(("energy_gradient", "physical_force_raw_derivative")),
            "fk": fd(("q_radians", "q_metres")),
            "contact": (
                fd(("gap_q", "gap_x", "energy_q", "energy_x"))
                if case.requested_contact_cardinality
                else {"status": "NOT_APPLICABLE", "records": []}
            ),
        },
        "reduction_noise": {"merit_samples": [[0.0, 0.0, 0.0] for _ in range(100)]},
        "linear_cancellation": [
            {
                "lambda": regularization,
                "zero_rhs_block": block,
                "true_norm_max": [0.0, 1e-8, 1e-8],
                "true_norm_peak_to_peak": [0.0, 0.0, 0.0],
            }
            for regularization in (0.0, 0.001, 1.0)
            for block in ("none", "q", "x")
        ],
        "coordinate_quantization": quantization,
        "force_noise": {
            "recommended_force_detection_floor": 4e-6,
            "resultant_samples": [[0.0] * 6 for _ in range(100)],
            "physical_rigid_sample_forces": [[0.0, 0.0, 0.0]],
            "physical_world_force_or_wrench": [0.0] * 6,
        },
        "draft_parameters": {
            "residual_floors": [2.8284271247461903e-4, 2e-4, 2e-4],
            "merit_absolute": [3e-4, 1e-8, 2e-4],
            "small_scaled_step": [3e-4, 1e-4, 2e-4],
            "merit_noise": 0.0,
            "linear_tolerance_target": 1e-4,
        },
    }
    record["gates"] = {
        "finite_difference": "PASS",
        "contact_cardinality": "PASS",
        "reduction": "PASS",
        "linear": "PASS",
        "trajectory": "NOT_MEASURED",
    }
    record["coordinate_support_status"] = "PASS"
    return record


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
        record = _synthetic_raw_record(case, "cpu")
        source = _committed_source_identity(["newton/_src/solvers/monolithic/contact.py"])
        raw = {
            "artifact_kind": "raw_measurement",
            "calibration_schema_version": 2,
            "profile": "freeze",
            "git_dirty": False,
            "repeats": 100,
            **source,
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
        self.assertEqual(result["maximum_passing_translation_offset_m"], 0.25)
        self.assertEqual(result["first_failing_translation_offset_m"], 0.5)
        self.assertEqual(result["passing_points_after_first_failure_not_used"], [1.0, 2.0, 5.0])

    def test_complete_passing_evidence_freezes_only_calibration_substate(self):
        """A complete manifest freezes calibration but never V0.1 acceptance."""
        records = []
        for device in ("cpu", "cuda:0"):
            for case in freeze_calibration_cases():
                records.append(_synthetic_raw_record(case, device))
        source = _committed_source_identity(["newton/_src/solvers/monolithic/contact.py"])
        raw = {
            "artifact_kind": "raw_measurement",
            "calibration_schema_version": 2,
            "profile": "freeze",
            "git_dirty": False,
            "repeats": 100,
            **source,
            "evidence": records,
        }
        portable = aggregate_calibration_evidence([raw])["portable_solver_internal_config"]

        def loading_run(device, *, config):
            return {
                "metadata": {
                    "device": device,
                    "worktree_dirty": False,
                    "input_sha256": "c" * 64,
                    "newton_sha": source["git_sha"],
                    "solver_internal_config": config if config is not None else {"legacy_default": True},
                    "requested_solver_internal_config": config,
                    "actual_parameters": {
                        "acceptance": {
                            "minimum_substeps": 1000,
                            "minimum_converged_ratio": 0.99,
                            "maximum_consecutive_non_success": 2,
                            "linear_tolerance": 1e-4,
                        }
                    },
                },
                "summary": {
                    "gates": {
                        "finite_state": True,
                        "converged_ratio": True,
                        "converged_linear_gates": True,
                    },
                    "e2e_numerical_pass": True,
                    "substeps": 1000,
                    "converged_ratio": 1.0,
                    "maximum_consecutive_non_success": 0,
                    "contact_onset_step": 200,
                    "peak_force_n": 1.0,
                    "settled_force_n": 0.9,
                    "secant_stiffness_n_m": 100.0,
                    "curve_work_j": 0.01,
                },
                "records": [
                    {
                        "finite_state": True,
                        "converged": True,
                        "nonlinear_convergence_ratios": [0.5, 0.5, 0.5],
                        "linear_iterations": 2,
                        "stats": {"rho": 5e-5, "rho_q": 5e-5, "rho_x": 5e-5},
                        "normal_compressive_force_n": index / 1000,
                    }
                    for index in range(1000)
                ],
                "artifact_sha256": ("e" if config is not None else "f") * 64,
            }

        probe_runs = [loading_run(device, config=portable) for device in ("cpu", "cuda:0")]
        for run in probe_runs:
            run["artifact_sha256"] = "b" * 64
            run["summary"].update(
                e2e_numerical_pass=False,
                converged_ratio=0.0,
                maximum_consecutive_non_success=1000,
            )
            run["summary"]["gates"].update(converged_ratio=False, consecutive_non_success=False)
            for record in run["records"]:
                record.update(
                    converged=False,
                    committed_unconverged=True,
                    rolled_back=False,
                    nonlinear_convergence_ratios=[0.5, 2.0, 0.5],
                )
                record["stats"]["merit_q_final"] = 1.62e-7
        final_config = dict(portable)
        final_config["merit_absolute_q"] = 2.5e-7

        c4_source = _committed_source_identity(["newton/_src/solvers/monolithic/contact.py"])
        c4 = {
            "status": "FROZEN",
            **c4_source,
            "evidence": [
                {
                    "device": device,
                    "c4_gate": "PASS",
                    "supported_local_motion_gate": "PASS",
                    "support_envelope": {"status": "FROZEN", "sha256": "d" * 64},
                    "static": [{"solver_internal_config": final_config}],
                }
                for device in ("cpu", "cuda:0")
            ],
        }
        c4_bytes = json.dumps(c4, sort_keys=True).encode()
        compact_bytes = json.dumps(
            {"calibration_status": "FROZEN", "solver_internal_config": final_config}, sort_keys=True
        ).encode()
        candidate_runs = [loading_run(device, config=final_config) for device in ("cpu", "cuda:0")]
        for run in candidate_runs:
            run["metadata"].update(
                calibration_status="FROZEN",
                calibration_sha256=hashlib.sha256(compact_bytes).hexdigest(),
                support_status="FROZEN",
                support_sha256=hashlib.sha256(c4_bytes).hexdigest(),
            )
        producer = _committed_source_identity(["scripts/monolithic_reference/calibrate_components.py"])
        with patch(
            "scripts.monolithic_reference.calibrate_components._current_producer_identity",
            return_value=producer,
        ):
            verified = build_trajectory_evidence(
                raw,
                probe_runs,
                candidate_runs,
                [loading_run(device, config=None) for device in ("cpu", "cuda:0")],
                c4_bytes,
                compact_bytes,
            )
        result = aggregate_calibration_evidence([raw], freeze=True, verified_trajectory=verified)
        self.assertEqual(
            verified.audit_artifact["upstream"]["compact_calibration"]["artifact_sha256"],
            candidate_runs[0]["metadata"]["calibration_sha256"],
        )
        self.assertEqual(
            verified.audit_artifact["upstream"]["c4"]["artifact_sha256"],
            candidate_runs[0]["metadata"]["support_sha256"],
        )
        self.assertEqual(result["calibration_status"], "FROZEN")
        self.assertEqual(result["v01_status"], "DRAFT")
        coordinate = result["exact_fixture_translation_offset_evidence"]
        self.assertEqual(coordinate["maximum_passing_translation_offset_m"], 5.0)
        self.assertIn("not a general world-coordinate range", coordinate["scope"])
        self.assertEqual(
            result["candidate_private_config"]["cpu"]["solver_internal_config"]["residual_floor_global"],
            2.8284271247461903e-4,
        )
        self.assertEqual(result["portable_solver_internal_config"]["residual_floor_global"], 2.8284271247461903e-4)
        self.assertEqual(result["portable_acceptance"]["force_detection_floor_n"], 4e-6)

        tampered = copy.deepcopy(raw)
        for fd_record in tampered["evidence"][0]["finite_difference"]["tet"]["records"]:
            fd_record["energy_gradient"]["relative_error"] = 1.0
        rejected = aggregate_calibration_evidence([tampered], verified_trajectory=verified)
        self.assertTrue(rejected["failed_records"])
        with self.assertRaises(CalibrationFreezeError):
            aggregate_calibration_evidence([tampered], freeze=True, verified_trajectory=verified)

        audit_only = copy.deepcopy(verified.audit_artifact)
        audit_only["upstream"]["normal_loading"]["cpu"]["candidate_artifact_sha256"] = "0" * 64
        normal_row = next(
            row for row in audit_only["evidence"] if row["device"] == "cpu" and row["role"] == "normal_loading_1000"
        )
        normal_row["candidate_artifact_sha256"] = "0" * 64
        audit_result = aggregate_calibration_evidence([raw, audit_only])
        self.assertEqual(audit_result["calibration_status"], "INCOMPLETE")
        with self.assertRaisesRegex(CalibrationFreezeError, "audit-only"):
            aggregate_calibration_evidence([raw, audit_only], freeze=True)

        for field in ("calibration_sha256", "support_sha256"):
            wrong_provenance = copy.deepcopy(candidate_runs)
            wrong_provenance[0]["metadata"][field] = "0" * 64
            with patch(
                "scripts.monolithic_reference.calibrate_components._current_producer_identity",
                return_value=producer,
            ):
                with self.assertRaisesRegex(CalibrationFreezeError, "provenance hash differs"):
                    build_trajectory_evidence(
                        raw,
                        probe_runs,
                        wrong_provenance,
                        [loading_run(device, config=None) for device in ("cpu", "cuda:0")],
                        c4_bytes,
                        compact_bytes,
                    )

        wrong_source = copy.deepcopy(raw)
        wrong_source["source_sha256"]["newton/_src/solvers/monolithic/contact.py"] = "d" * 64
        wrong_source["source_set_sha256"] = _hash_json(wrong_source["source_sha256"])
        with self.assertRaises(CalibrationFreezeError):
            aggregate_calibration_evidence([wrong_source])

        missing_parameter = copy.deepcopy(raw)
        del missing_parameter["evidence"][0]["draft_parameters"]["residual_floors"]
        rejected = aggregate_calibration_evidence([missing_parameter])
        self.assertTrue(rejected["failed_records"])
        self.assertTrue(rejected["artifact_failures"])

        no_pcg = copy.deepcopy(candidate_runs)
        for run in no_pcg:
            for record in run["records"]:
                record["linear_iterations"] = 0
                record["stats"] = {}
        with patch(
            "scripts.monolithic_reference.calibrate_components._current_producer_identity",
            return_value=producer,
        ):
            with self.assertRaisesRegex(CalibrationFreezeError, "did not exercise runtime PCG"):
                build_trajectory_evidence(
                    raw,
                    probe_runs,
                    no_pcg,
                    [loading_run(device, config=None) for device in ("cpu", "cuda:0")],
                    c4_bytes,
                    compact_bytes,
                )

    def test_versioned_candidate_is_not_runtime_authority(self):
        """The checked-in proposal is machine-readable but explicitly unfrozen."""
        path = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/calibration_candidate_v2.json"
        candidate = json.loads(path.read_text())
        self.assertEqual(candidate["calibration_schema_version"], 2)
        self.assertEqual(candidate["calibration_status"], "INCOMPLETE")
        self.assertEqual(candidate["v01_status"], "DRAFT")
        self.assertTrue(candidate["runtime_use"].startswith("PROHIBITED"))
        coordinate = candidate["exact_fixture_translation_offset_evidence"]
        self.assertEqual(coordinate["maximum_passing_translation_offset_m"], 2.0)
        self.assertIn("not a general world-coordinate range", coordinate["scope"])
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
        self.assertEqual(candidate["portable_acceptance"]["force_detection_floor_n"], 1.2695789400826758e-05)

    def test_frozen_calibration_is_the_runtime_authority(self):
        """Keep production defaults identical to the reviewed compact freeze manifest."""
        path = (
            Path(__file__).parents[2]
            / "scripts/monolithic_reference/fixtures/monolithic_calibration_frozen_v1.json"
        )
        frozen = json.loads(path.read_text())
        self.assertEqual(frozen["schema_version"], "monolithic_calibration_frozen/v1")
        self.assertEqual(frozen["calibration_status"], "FROZEN")
        self.assertEqual(frozen["v01_status"], "DRAFT")
        self.assertTrue(frozen["runtime_authority"])
        runtime = json.loads(json.dumps(asdict(_SolverMonolithicInternalConfig())))
        self.assertEqual(runtime, frozen["solver_internal_config"])
        self.assertEqual(frozen["acceptance"]["force_detection_floor_n"], 1.2695789400826758e-05)
        for digest in frozen["evidence"].values():
            self.assertEqual(len(digest), 64)


def test_measured_active_and_inactive(test, device):
    """Measure actual forces and keep coordinate sensitivity separate from reduction noise."""
    for active in (False, True):
        case = CalibrationCase("test", 1e4, 0.3, 0.001, 1.0, 0.0, active, 87)
        result = calibrate_case(device, case=case, repeats=3)
        json.dumps(result, allow_nan=False)
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
    json.dumps(result, allow_nan=False)
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
