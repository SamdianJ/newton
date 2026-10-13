# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard timing summaries against incomplete or capture-contaminated claims."""

import copy
import json
import unittest

import numpy as np

from scripts.monolithic_reference import run_performance as performance
from scripts.monolithic_reference.run_actor_coupling import build_model, load_case
from scripts.monolithic_reference.run_paired_performance import paired_summary, run_order
from scripts.monolithic_reference.run_performance import summarize


class TestPerformanceSummary(unittest.TestCase):
    def test_completed_stopped_is_not_qualified(self):
        """Keep completed Stopped runs distinct from qualified acceleration."""
        result = summarize([1.0, 2.0, 3.0], [False, True, False], {"STOPPED": 3}, 3)
        self.assertEqual(result["status"], "COMPLETE_NOT_AUDITED")
        self.assertEqual(result["solve_seconds"], 6.0)
        self.assertEqual(result["capture_free_seconds"], 4.0)
        self.assertEqual(result["capture_free_steps"], 2)
        self.assertEqual(result["qualified_speedup"], None)

    def test_partial_and_failure(self):
        """Reject incomplete advancement as a full performance result."""
        self.assertEqual(summarize([1.0], [False], {"STOPPED": 1}, 3)["status"], "PARTIAL")
        self.assertEqual(summarize([1.0], [False], {"HARD_FAILURE": 1}, 3)["status"], "FAIL")

    def test_actor_performance_recipe(self):
        """Build both actor routes with frozen controls, lambda zero and explicit B3 resources."""
        case = performance.performance_case("actor-coupling", "fp64")
        samples = []
        for route in ("legacy", "spatial"):
            model, run, state, control = performance.build_performance_scene(
                case, contact_dataflow=route, device="cpu", graph=False
            )
            self.assertEqual(case["steps"], 2000)
            self.assertEqual(run._config.regularization_values, (0.0,))
            self.assertEqual(control[0].shape, (2001, model.joint_target_q.size))
            self.assertEqual(control[1].shape, (2001, model.joint_dof_count))
            self.assertEqual(control[0].dtype, np.float32)
            self.assertEqual(control[1].dtype, np.float32)
            self.assertTrue(np.isfinite(state.joint_q.numpy()).all())
            unchanged = np.setdiff1d(np.arange(model.joint_target_q.size), run._articulation_actor.target_indices)
            np.testing.assert_array_equal(control[0][1000, unchanged], model.joint_target_q.numpy()[unchanged])
            if route == "spatial":
                self.assertEqual(run._contact.resource_identity[-1], 720)
                self.assertEqual(run._contact.resource_identity[-2], 2160)
                self.assertEqual(run._contact.law_margin, 0.0)
            samples.append(control)
        for field in (0, 1):
            np.testing.assert_array_equal(samples[0][field], samples[1][field])

    def test_actor_chain_scaling(self):
        """Extend only the rigid chain while retaining the frozen default geometry."""
        case = load_case()
        for count in (2, 4):
            model = build_model(device="cpu", rigid_count=count)
            self.assertEqual(model.body_count, count + 1)
            self.assertEqual(model.particle_count, 125)
            self.assertEqual(model.tet_count, 384)
            centers = model.body_q.numpy()[:, :3]
            self.assertAlmostEqual(centers[-1, 0], -0.0303)
            np.testing.assert_allclose(np.diff(centers[:, 0]), 0.0202, atol=1e-8)
            if count == 2:
                expected = [case["centers_m"][name] for name in ("pusher", "rigid_1", "rigid_2")]
                np.testing.assert_array_equal(centers, np.asarray(expected, dtype=np.float32))
        for value in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                build_model(device="cpu", rigid_count=value)

    def test_actor_fp32_rejected(self):
        """Reject a precision change to the frozen actor acceptance recipe."""
        with self.assertRaisesRegex(ValueError, "FP64"):
            performance.performance_case("actor-coupling", "fp32")

    def test_paired_identity_and_limits(self):
        """Apply paired limits only to complete, consistent production evidence."""
        receipts = []
        for label in run_order():
            candidate = label.endswith("B")
            receipts.append(
                {
                    "status": "COMPLETE_NOT_AUDITED",
                    "final_state_checks": "FINITE_HISTORY_FORCE_GENERATION_PASS",
                    "source_hashes": {"solver.py": "same"},
                    "case": {"scene": "actor-coupling", "steps": 2000},
                    "controls_sha256": "fixed",
                    "numerical_policy": {"regularization_values": [0.0]},
                    "python": "3.12",
                    "numpy": "2",
                    "warp": "1",
                    "device": "cuda:0",
                    "graph": True,
                    "contact_pair_diagnostics": False,
                    "gc_enabled": True,
                    "counts": {"CONVERGED": 2000},
                    "published_steps": 2000,
                    "totals": {"linear_iterations": 3000},
                    "contact_dataflow": "spatial" if candidate else "legacy",
                    "solve_seconds": 90.0 if candidate else 100.0,
                    "step_p95_seconds": 0.09 if candidate else 0.1,
                }
            )
        result = paired_summary(receipts)
        self.assertEqual(result["status"], "PASS_TIMING_NO_REGRESSION")
        self.assertTrue(result["timing_benefit_observed"])
        self.assertIsNone(result["qualified_speedup"])
        json.dumps(result)
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            paired_summary(receipts[:-1])
        for field, value in (
            ("source_hashes", {}),
            ("controls_sha256", "changed"),
            ("numerical_policy", {"regularization_values": [0.0, 1.0]}),
            ("status", "PARTIAL"),
            ("counts", {"STOPPED": 1}),
            ("gc_enabled", False),
        ):
            bad = copy.deepcopy(receipts)
            bad[3][field] = value
            with self.assertRaises(ValueError):
                paired_summary(bad)
        three = receipts[:8] + receipts[-1:]
        result = paired_summary(three, pair_count=3)
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(len(result["pairs"]), 3)
        self.assertAlmostEqual(result["total_ratio_median"], 0.9)
        self.assertTrue(result["timing_benefit_observed"])
        self.assertEqual(run_order(pair_count=3), [*run_order()[:8], "bridge-A"])
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            paired_summary(three)
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            paired_summary(three[:-1], pair_count=3)
        bad_three = copy.deepcopy(three)
        bad_three[-1]["solve_seconds"] = 106.0
        self.assertEqual(paired_summary(bad_three, pair_count=3)["status"], "ENVIRONMENT_DRIFT")
        for count in (0, 1, 4):
            with self.assertRaises(ValueError):
                run_order(pair_count=count)
        bad = copy.deepcopy(receipts)
        bad[-1]["solve_seconds"] = 106.0
        self.assertEqual(paired_summary(bad)["status"], "ENVIRONMENT_DRIFT")
        for row in receipts:
            if row["contact_dataflow"] == "spatial":
                row["step_p95_seconds"] = 0.111
        self.assertEqual(paired_summary(receipts)["status"], "FAIL_TIMING_REGRESSION")


if __name__ == "__main__":
    unittest.main()
