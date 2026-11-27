# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA FP32 E2E. Run the rigid-mandrel and seven-wire rollouts with
NEWTON_RUN_CABLE_HITCH_E2E=1 uv run -m unittest -v newton.tests.test_monolithic_cable_self_lock
"""

import os
import unittest

import warp as wp

from scripts.monolithic_reference.cable_hitch import run_hitch
from scripts.monolithic_reference.cable_spot_tie import run_spot_tie


@unittest.skipUnless(os.environ.get("NEWTON_RUN_CABLE_HITCH_E2E") == "1", "Opt-in full cable hitch rollouts")
@unittest.skipUnless(wp.is_cuda_available(), "Cable hitch E2E requires CUDA")
class TestMonolithicCableSelfLock(unittest.TestCase):
    def assert_formed_and_loaded(self, report):
        self.assertEqual(report["passages"], [3, 2, 1])
        self.assertGreaterEqual(report["end_separation_gain_mm"], 10)
        self.assertGreater(report["simultaneous_end_tension_mN"], 3)
        self.assertLessEqual(report["rope_overlap_mm"], 0.32)
        self.assertLessEqual(report["core_overlap_mm"], 0.32)
        self.assertGreater(report["centerline_bound_mm"], 0)
        self.assertLess(report["max_released_fixture_force_N"], 1e-8)
        self.assertTrue(report["full_load_12mN"])
        self.assertGreaterEqual(report["full_load_duration_s"], 3.0)

    def test_self_lock_under_12mN(self):
        """Hold the load after releasing all guides and grippers, with submillimeter end slip."""
        report = run_hitch(rope_mu=0.35)
        self.assert_formed_and_loaded(report)
        self.assertGreaterEqual(report["min_released_turns"], 0.80)
        self.assertLessEqual(report["max_wrap_change_turns"], 0.15)
        self.assertLessEqual(report["end_excursion_last_half_s_mm"], 1.0)
        self.assertAlmostEqual(report["opposing_reaction_last_half_s_mN"], 12.0, delta=1.2)

    def test_same_hitch_slips_without_rope_friction(self):
        """Thread and draw the same hitch, then lose the wrap and slide when rope friction is zero."""
        report = run_hitch(rope_mu=0.0)
        self.assert_formed_and_loaded(report)
        self.assertLess(report["end_turns"], 0.80)
        self.assertGreater(report["end_excursion_last_half_s_mm"], 1.0)

    def test_seven_wire_clove_square_self_lock(self):
        """Hold 12 mN on seven flexible wires after three proof pulls and complete release."""
        report = run_spot_tie()
        self.assertTrue(report["formation"]["valid"], report["formation"])
        for force, plateau, unloaded in zip(
            (12, 16, 8), report["proof_plateaus_mN"], report["proof_unloaded_N"], strict=True
        ):
            for measured in plateau:
                self.assertAlmostEqual(measured, force, places=5)
            self.assertLess(unloaded, 1e-8)
        self.assertLess(report["proof_profile_error_N"], 1e-8)
        self.assertLess(report["service_profile_error_N"], 1e-8)
        self.assertLess(report["free_jaw_during_proofs_N"], 1e-8)
        self.assertLess(report["max_released_fixture_force_N"], 1e-8)
        self.assertTrue(report["no_early_service_load"])
        self.assertTrue(report["full_load_12mN"])
        self.assertGreaterEqual(report["full_load_duration_s"], 3.0)
        self.assertEqual(report["support_motion_mm"], 0)
        self.assertTrue(report["both_wraps_on_span"])
        self.assertGreaterEqual(report["min_wire_turns"], 1.80)
        self.assertLessEqual(report["max_turn_change"], 0.15)
        for tip, excursion in zip(("loaded", "free"), report["tip_excursions_mm"], strict=True):
            with self.subTest(tip=tip):
                self.assertLessEqual(excursion, 10.0)
        self.assertLessEqual(report["end_excursion_last_half_s_mm"], 1.0)
        self.assertAlmostEqual(report["opposing_reaction_mN"], 12.0, delta=1.2)
        self.assertGreaterEqual(report["terminal_tail_mm"], 10.8)
        for overlap, limit in zip(report["overlap_mm"], (0.32, 0.32, 0.20 * 25 / 6), strict=True):
            self.assertLessEqual(overlap, limit)
        self.assertGreater(report["centerline_bound_mm"], 0)
        self.assertGreater(report["closure_bound_mm"], 0)


if __name__ == "__main__":
    unittest.main()
