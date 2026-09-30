# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent joint references without retired solver diagnostics."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from newton.tests.test_monolithic_solver_contracts import model_and_solver
from scripts.monolithic_reference.p3_articulation_fixtures import specification
from scripts.monolithic_reference.p3_articulation_oracle import DiscreteOracle
from scripts.monolithic_reference.reference_p3_articulation_continuous import reference


class TestJointOracle(unittest.TestCase):
    def test_oracle_scale_and_backtrack_budget(self):
        """Preserve default roots and the 32-search failure budget with a stricter goal."""
        model, solver = model_and_solver("cpu")
        params = solver._joint_terms.params.numpy()
        results = []
        for options in ({}, {"goal_scale": 1.0}, {"goal_scale": 0.1}):
            oracle = DiscreteOracle(model, params, [0.0], [0.0], 0.001, **options)
            results.append(oracle.step(0.001, np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros(1)))
        np.testing.assert_array_equal(results[0]["q"], results[1]["q"])
        self.assertEqual(results[0]["goal"], results[1]["goal"])
        self.assertAlmostEqual(results[2]["goal"] / results[0]["goal"], 0.1)
        oracle = DiscreteOracle(model, params, [0.0], [0.0], 0.001, goal_scale=0.1)
        with patch.object(oracle.kinematics, "inertia", return_value=(0.0, np.ones(1), np.eye(1))) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "ORACLE_UNCERTIFIED"):
                oracle.step(0.001, np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros(1))
        self.assertEqual(evaluate.call_count, 33)
        np.testing.assert_array_equal(oracle.q, [0.0])

        def slow_residual(q, *args):
            return 0.0, np.array([1 + 1e-6 * q[0]]), np.eye(1)

        with patch.object(oracle.kinematics, "inertia", side_effect=slow_residual) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "iterations=100"):
                oracle.step(0.001, np.zeros(1), np.zeros(1), np.zeros((1, 3)), np.zeros(1))
        self.assertEqual(evaluate.call_count, 201)
        np.testing.assert_array_equal(oracle.q, [0.0])

    def test_reference_output_grid(self):
        """Return each native comparison time while preserving default sampled states."""
        case = copy.deepcopy(next(c for c in specification()["cases"] if c["id"] == "constant_force"))
        case["duration"] = 0.002
        old = reference(case, 64)
        fine = reference(case, 64, output_dt=0.000125)
        np.testing.assert_array_equal(np.array(fine["states"])[::8], old["states"])
        np.testing.assert_allclose(fine["times"], np.arange(17) * 0.000125, rtol=0, atol=1e-16)
        case["duration"] = 0.004
        full = reference(case, 8, output_dt=0.000125)
        self.assertEqual(len(full["states"]), 33)
        np.testing.assert_allclose(full["times"][1:], np.arange(1, 33) * 0.000125, rtol=0, atol=1e-16)
        self.assertEqual(full["times"][-1], 0.004)
        for dt in (0, float("nan"), 0.0001):
            with self.assertRaises(ValueError):
                reference(case, 64, output_dt=dt)


if __name__ == "__main__":
    unittest.main()
