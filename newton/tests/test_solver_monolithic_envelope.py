# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify independently calibrated normal-response acceptance."""

import copy
import tempfile
import unittest
from itertools import pairwise
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.internal_envelope import FROZEN_FIXTURE, assess_envelope, load_envelope, static_curve
from scripts.monolithic_reference.normal_loading import _v01_exit, load_fixture


class TestInternalEnvelope(unittest.TestCase):
    def test_static_rest_and_positive_stiffness(self):
        """Recover stress-free rest and monotone compressive response analytically."""
        curve = static_curve(load_fixture())
        self.assertAlmostEqual(curve[0]["force_n"], 0.0, places=10)
        self.assertTrue(all(p["force_n"] > 0 for p in curve[1:]))
        self.assertTrue(all(b["closure_m"] > a["closure_m"] for a, b in pairwise(curve)))

    def test_frozen_fixture_binds_physics_and_acceptance(self):
        """Reject changed stiffness, trajectory, calibration and acceptance under frozen identity."""
        fixture = load_fixture(FROZEN_FIXTURE)
        self.assertEqual(load_envelope(fixture)["status"], "FROZEN")
        for section, field, value in (
            ("contact", "stiffness_n_m3", 1e6),
            ("drive", "kp_n_m", 2001),
            ("acceptance", "linear_tolerance", 1),
            ("calibration_provenance", "sha256", "a" * 64),
        ):
            changed = copy.deepcopy(fixture)
            changed[section][field] = value
            with self.assertRaises(ValueError):
                load_envelope(changed)

    def test_scalar_and_interior_curve_corruption_is_rejected(self):
        """Reject absent/nonfinite/outside metrics and an interior-only force mutation."""
        fixture = load_fixture(FROZEN_FIXTURE)
        envelope = load_envelope(fixture)
        records = [
            {"actual_closure_m": p["closure_m"], "normal_compressive_force_n": p["force_n"]}
            for p in envelope["static_curve"]
        ]
        summary = {name: sum(bounds) / 2 for name, bounds in envelope["metrics"].items()}
        summary["contact_onset_step"] = 0
        self.assertTrue(assess_envelope(records, summary, fixture)["pass"])
        for name in envelope["metrics"]:
            for value in (None, float("nan"), float("inf"), -1, envelope["metrics"][name][1] + 1):
                changed = dict(summary)
                changed[name] = value
                self.assertFalse(assess_envelope(records, changed, fixture)["pass"])
        records[100]["normal_compressive_force_n"] += 1
        self.assertFalse(assess_envelope(records, summary, fixture)["pass"])

    def test_frozen_cli_fails_for_physical_envelope_failure(self):
        """Return a failing process status when frozen numerics pass but response fails."""
        from scripts.monolithic_reference.normal_loading import main  # noqa: PLC0415

        result = {"metadata": {}, "records": [], "summary": {"draft_numerical_pass": True, "v01_exit": False}}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sys.argv", ["normal_loading", "--fixture", str(FROZEN_FIXTURE), "--output", directory]),
                patch("scripts.monolithic_reference.normal_loading.run_loading", return_value=result),
            ):
                self.assertEqual(main(), 1)

    def test_impact_modal_bound_matches_declared_discrete_model(self):
        """Bound both independent initial perturbations over actual matrix powers."""
        fixture = load_fixture(FROZEN_FIXTURE)
        budget = load_envelope(fixture)["impact_budget"]
        transition = np.asarray(budget["transition"])
        amplitudes = np.asarray(budget["modal_amplitudes_n"])
        decay = np.asarray(budget["modal_decay"])
        for sign in (-1, 1):
            state = np.asarray(
                [budget["initial_penetration_bound_m"], 0, sign * budget["initial_velocity_bound_m_s"], 0]
            )
            for step in range(100):
                force = abs(budget["contact_stiffness_n_m"] * (state[0] - state[1]))
                self.assertLessEqual(force, float(np.sum(amplitudes * decay**step)) + 1e-12)
                state = transition @ state

    def test_exit_requires_internal_envelope(self):
        """Keep a numerical-only or missing-envelope result outside release acceptance."""
        gates = {"e2e_numerical_pass": True, "frozen_calibration": True, "frozen_support": True}
        self.assertFalse(_v01_exit(gates))
        gates["internal_envelope_pass"] = True
        self.assertTrue(_v01_exit(gates))


if __name__ == "__main__":
    unittest.main()
