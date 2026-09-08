# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the opt-in monolithic release profiler contracts."""

import copy
import gc
import hashlib
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import warp as wp

import scripts.monolithic_reference.profile_release as release_profile
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import assess_run, build_scene, load_fixture
from scripts.monolithic_reference.profile_release import (
    _save_trajectories,
    _select_stiffness_lower_bound,
    profile_collision_kernels,
    run_profiled_medium,
    run_profiled_normal,
    summarize_records,
)


class TestReleaseProfileHelpers(unittest.TestCase):
    def test_formal_fixture_and_stiffness_candidates_are_separate(self):
        """Use pinned release evidence only for the unchanged formal fixture."""
        formal = release_profile._normal_fixture()
        candidate = release_profile._normal_fixture(200000.0)
        self.assertEqual(formal["status"], "FROZEN")
        self.assertEqual(formal["schema_version"], "normal_loading/v3")
        self.assertEqual(candidate["status"], "DRAFT")
        self.assertEqual(candidate["schema_version"], "normal_loading/v2")
        self.assertEqual(candidate["contact"]["stiffness_n_m3"], 200000.0)
        self.assertNotEqual(formal["contact"]["stiffness_n_m3"], candidate["contact"]["stiffness_n_m3"])
        self.assertEqual(release_profile._normal_fixture(), formal)

    def test_stiffness_lower_bound_is_fail_closed(self):
        """Reject a lower bound without a measured failing predecessor."""
        rows = [
            {"stiffness_n_m3": 2.0, "physical_numerical_gate": "PASS"},
            {"stiffness_n_m3": 1.0, "physical_numerical_gate": "FAIL"},
            {"stiffness_n_m3": 4.0, "physical_numerical_gate": "PASS"},
        ]
        self.assertEqual(_select_stiffness_lower_bound(rows), 2.0)
        self.assertIsNone(_select_stiffness_lower_bound([*rows[:1], {"stiffness_n_m3": 1.0}]))
        self.assertIsNone(_select_stiffness_lower_bound([]))


def test_profile_enter_cleanup(test, device):
    """Restore every patch when profile entry fails before or after inverse replacement."""
    for failure in ("construct_inverse", "after_inverse_replacement"):
        _model, _state, _control, solver = build_scene(load_fixture(), device=device)
        workspace = solver._linear
        profile = release_profile._StageProfile(solver, "diagonal")
        originals = [(wp, "launch", wp.launch)]
        for owner, names in (
            (solver, ("_collide", "_evaluate_current", "_evaluate_trial")),
            (
                release_profile.solver_module,
                ("assemble_current_contacts", "evaluate_trial_contacts", "evaluate_final_contacts"),
            ),
            (workspace, ("finalize_assembly", "factor_actor_preconditioner", "solve_pcg", "preconditioner")),
        ):
            originals.extend((owner, name, getattr(owner, name)) for name in names)
        enter_context = profile.stack.enter_context

        def fail_after_replacement(context, _enter_context=enter_context):
            if context.attribute == "solve_pcg":
                raise RuntimeError("injected entry failure")
            return _enter_context(context)

        injection = (
            patch.object(release_profile, "_DiagonalInverse", side_effect=RuntimeError("injected entry failure"))
            if failure == "construct_inverse"
            else patch.object(profile.stack, "enter_context", side_effect=fail_after_replacement)
        )
        try:
            with injection, test.assertRaisesRegex(RuntimeError, "injected entry failure"):
                profile.__enter__()
            for owner, name, original in originals:
                test.assertEqual(getattr(owner, name), original, (failure, name))
        finally:
            profile.__exit__(None, None, None)


@wp.kernel
def _touch_profile_scratch(scratch: wp.array[float]):
    scratch[wp.tid()] = 1.0


def test_profile_releases_launch_arguments(test, device):
    """Profiling must not retain temporary sparse-builder buffers across launches."""
    _model, _state, _control, solver = build_scene(load_fixture(), device=device)
    with release_profile._StageProfile(solver, "actor_block"):
        scratch = wp.zeros(1024, dtype=float, device=device)
        reference = weakref.ref(scratch)
        wp.launch(_touch_profile_scratch, dim=1024, inputs=[scratch], device=device)
        wp.synchronize_device(device)
        del scratch
        gc.collect()
        test.assertIsNone(reference(), "Profiler retained a completed launch's temporary buffer")


def test_collision_profile(test, device):
    """Run both collision kernels on matching candidate pairs."""
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
    """Preserve full trajectory evidence and recomputable diagnostics."""
    result = run_profiled_normal(device, variant="actor_block", substeps=3, allocation_audit=True)
    test.assertEqual(result["variant"], "actor_block")
    test.assertEqual(result["substeps"], 3)
    test.assertEqual(len(result["step_ms_samples"]), 3)
    test.assertTrue(result["all_finite"])
    test.assertEqual(result["fixture"]["status"], "FROZEN")
    test.assertFalse(result["assessment"]["v01_exit"])
    records = result["records"]
    test.assertEqual(len(records), 3)
    test.assertEqual(result["diagnostics"], summarize_records(records))
    test.assertGreater(result["diagnostics"]["matrix_nnz"]["p50"], 0)
    for name in ("rho", "rho_q", "rho_x"):
        test.assertEqual(
            result["diagnostics"][name]["measured_count"], sum(r["stats"][name] is not None for r in records)
        )
    changed = copy.deepcopy(records)
    changed[0]["rolled_back"] = True
    test.assertEqual(summarize_records(changed)["rollback_count"], 1)
    with tempfile.TemporaryDirectory() as directory:
        report = {"run": copy.deepcopy(result)}
        hashes = _save_trajectories(report, Path(directory))
        artifact = report["run"]["trajectory"]
        saved = Path(directory) / artifact["path"]
        test.assertEqual(hashes[artifact["path"]], hashlib.sha256(saved.read_bytes()).hexdigest())
        restored = [json.loads(line) for line in saved.read_text().splitlines()]
        test.assertEqual(restored, records)
        test.assertEqual(summarize_records(restored), result["diagnostics"])
        test.assertEqual(assess_run(restored, result["fixture"]), result["assessment"])
    audit = result["allocation_audit"]
    test.assertEqual(audit["allocation_gate"], "PASS")
    test.assertEqual(audit["unexpected_python_device_allocations"], 0)
    test.assertIn("collision_candidate", result["stages"])
    test.assertIn("whole_step", result["percentiles_ms"])
    for name in (
        "preconditioner_q_setup",
        "preconditioner_q_factor",
        "preconditioner_x_setup",
        "preconditioner_x_factor",
        "preconditioner_q_apply",
        "preconditioner_x_apply",
    ):
        test.assertGreater(result["stages"][name]["calls"], 0)


def test_medium_records(test, device):
    """Retain medium trajectory states and independent quality diagnostics."""
    result = run_profiled_medium(device, level=1, substeps=2, profile_stages=False)
    test.assertEqual(result["diagnostics"]["record_count"], 2)
    test.assertEqual(result["diagnostics"]["rollback_count"], 0)
    test.assertEqual(result["diagnostics"]["nonfinite_state_count"], 0)
    test.assertGreater(result["diagnostics"]["matrix_nnz"]["p50"], 0)
    test.assertEqual(len(result["records"][0]["node_positions_m"]), result["particle_count"])


def test_uninstrumented_normal(test, device):
    """Measure a production step without stage patches or allocation hooks."""
    with (
        patch("scripts.monolithic_reference.profile_release._StageProfile", side_effect=AssertionError),
        patch("scripts.monolithic_reference.profile_release._AllocationAudit", side_effect=AssertionError),
    ):
        result = run_profiled_normal(device, substeps=2, profile_stages=False)
    test.assertFalse(result["measurement_mode"]["stage_instrumentation"])
    test.assertFalse(result["measurement_mode"]["allocation_tracker"])
    test.assertEqual(result["stages"], {})
    test.assertIsNone(result["allocation_audit"])
    with test.assertRaises(ValueError):
        run_profiled_normal(device, variant="diagonal", substeps=1, profile_stages=False)


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


add_function_test(TestReleaseProfileDevices, "test_medium_records", test_medium_records, devices=get_test_devices())
add_function_test(
    TestReleaseProfileDevices, "test_uninstrumented_normal", test_uninstrumented_normal, devices=get_test_devices()
)


add_function_test(
    TestReleaseProfileDevices, "test_profile_enter_cleanup", test_profile_enter_cleanup, devices=get_test_devices()
)


add_function_test(
    TestReleaseProfileDevices,
    "test_profile_releases_launch_arguments",
    test_profile_releases_launch_arguments,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main()
