# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard timing summaries against incomplete or capture-contaminated claims."""

import unittest

import numpy as np

from scripts.monolithic_reference import run_performance as performance
from scripts.monolithic_reference.run_actor_coupling import build_model, load_case
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
        """Build the spatial actor route with frozen controls, lambda zero and explicit B3 resources."""
        case = performance.performance_case("actor-coupling", "fp64")
        model, run, state, control = performance.build_performance_scene(
            case, contact_dataflow="spatial", device="cpu", graph=False
        )
        self.assertEqual(case["steps"], 2000)
        self.assertEqual(run._config.regularization_values, (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0))
        self.assertEqual(control[0].shape, (2001, model.joint_target_q.size))
        self.assertEqual(control[1].shape, (2001, model.joint_dof_count))
        self.assertEqual(control[0].dtype, np.float32)
        self.assertEqual(control[1].dtype, np.float32)
        self.assertTrue(np.isfinite(state.joint_q.numpy()).all())
        unchanged = np.setdiff1d(np.arange(model.joint_target_q.size), run._articulation_actor.target_indices)
        np.testing.assert_array_equal(control[0][1000, unchanged], model.joint_target_q.numpy()[unchanged])
        self.assertEqual(run._contact.resource_identity[-1], 720)
        self.assertEqual(run._contact.resource_identity[-2], 2160)
        with self.assertRaisesRegex(ValueError, "contact route"):
            performance.build_performance_scene(case, contact_dataflow="pooled", device="cpu", graph=False)

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


if __name__ == "__main__":
    unittest.main()
