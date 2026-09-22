# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify diagnostic controls without changing production defaults."""

import copy
import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.articulation_solver import _ArticulationSolver
from newton._src.solvers.monolithic.solver_monolithic import _SolverMonolithicInternalConfig
from newton.tests.test_monolithic_p3_articulation_contracts import assert_snapshot, model_and_solver, snapshot
from scripts.monolithic_reference.diagnose_p3_sharpa import configure_stopping
from scripts.monolithic_reference.p3_articulation_fixtures import specification, target
from scripts.monolithic_reference.p3_articulation_oracle import DiscreteOracle
from scripts.monolithic_reference.reference_p3_articulation_continuous import reference
from scripts.monolithic_reference.verify_p3_articulation import run_case


class TestPrecisionDiagnostics(unittest.TestCase):
    def test_held_control_sampling_and_reference(self):
        """Hold both actual control arrays from each outer start through all RK stages."""
        case = copy.deepcopy(next(c for c in specification()["cases"] if c["id"] == "driven_terms"))
        case["duration"] = 0.002
        with patch("scripts.monolithic_reference.verify_p3_articulation.target", wraps=target) as sample:
            result = run_case(
                case, "cpu", "ArmijoOrResNorm", h=0.000125, pcg_mode="diagnostic", graph=False, control_period=0.001
            )
        self.assertTrue(result["passed_same_discrete"])
        self.assertEqual([call.args[1] for call in sample.call_args_list], [0.0, 0.001])
        self.assertEqual(result["committed_outer_steps"], 2)
        for i, row in enumerate(result["trace"]):
            for key, value in zip(("q_target", "qd_target"), target(case, (i // 8) * 0.001), strict=True):
                np.testing.assert_array_equal(row[key], value.astype(np.float32).astype(float))
        with patch("scripts.monolithic_reference.reference_p3_articulation_continuous.target", wraps=target) as sample:
            held = reference(case, 64, output_dt=0.000125, control_period=0.001)
        self.assertEqual([call.args[1] for call in sample.call_args_list], [0.0, 0.001])
        self.assertEqual(len(held["states"]), 17)
        self.assertFalse(np.array_equal(held["states"], reference(case, 64, output_dt=0.000125)["states"]))

    def test_failed_substep_rolls_back_outer_history(self):
        """Reject a partial outer cycle and restore every integration history field."""
        case = copy.deepcopy(next(c for c in specification()["cases"] if c["id"] == "constant_force"))
        case["duration"] = 0.002
        advance = _ArticulationSolver.step
        inputs = []

        def fail_third(solver, state, out, control, **kwargs):
            inputs.append(snapshot(state))
            if len(inputs) == 3:
                with patch.object(solver, "_iterate_articulation", return_value=solver.Status.NONLINEAR_MAX_ITERATIONS):
                    return advance(solver, state, out, control, **kwargs)
            return advance(solver, state, out, control, **kwargs)

        with patch.object(_ArticulationSolver, "step", new=fail_third):
            result = run_case(
                case, "cpu", "ArmijoOrResNorm", h=0.000125, pcg_mode="diagnostic", graph=False, control_period=0.001
            )
        self.assertFalse(result["passed_same_discrete"])
        self.assertEqual(result["committed_outer_steps"], 0)
        self.assertEqual(len(result["trace"]), 3)
        self.assertTrue(result["trace"][0]["converged"])
        self.assertFalse(any(row["outer_committed"] for row in result["trace"]))
        restored = result["failure_state"]["outer_rollback"]
        np.testing.assert_array_equal(restored["q"], inputs[0]["joint_q"])
        np.testing.assert_array_equal(restored["qd"], inputs[0]["joint_qd"])
        for name, value in restored["history"].items():
            np.testing.assert_array_equal(value, inputs[0]["history." + name])

    def test_production_articulation_goals(self):
        """Promote the certified goals only for the articulation entry."""
        _, solver = model_and_solver("cpu")
        original = asdict(_SolverMonolithicInternalConfig())
        for name, value in original.items():
            scaled = name in ("merit_absolute_global", "merit_absolute_q", "merit_relative_global", "merit_relative_q")
            self.assertEqual(getattr(solver._config, name), value * 0.1 if scaled else value)

    def test_isolated_stopping_configuration(self):
        """Scale exactly four goals while retaining production defaults and budgets."""
        _, solver = model_and_solver("cpu")
        before = asdict(solver._config)
        self.assertEqual(configure_stopping(solver), before)
        actual = configure_stopping(solver, 0.1)
        for name, value in before.items():
            changed = name in ("merit_absolute_global", "merit_absolute_q", "merit_relative_global", "merit_relative_q")
            self.assertEqual(actual[name], value * 0.1 if changed else value)
        _, other = model_and_solver("cpu")
        self.assertEqual(asdict(other._config), before)
        self.assertEqual(
            (solver.newton_max_iterations, solver.line_search_max_iterations, solver.linear_max_iterations),
            (10, 8, 200),
        )
        for invalid in (0, -1, float("nan"), float("inf"), True, 0.5):
            with self.assertRaises(ValueError):
                configure_stopping(solver, invalid)

    def test_linear_and_final_gates(self):
        """Use scaled goals for linear caps and reject an otherwise publishable residual."""
        limits = []
        for scale in (1.0, 0.1):
            model, solver = model_and_solver("cpu")
            configure_stopping(solver, scale)
            state = solver._create_articulation_state(dt=0.001)
            state.body_f.assign([[1, 0, 0, 0, 0, 0]])
            with patch.object(solver._linear, "solve_pcg", wraps=solver._linear.solve_pcg) as solve:
                solver.step(state, state, model.control(), dt=0.001)
            self.assertTrue(solver.last_stats.converged)
            limits.append(solve.call_args.kwargs["config"].residual_norm_limits[:2])
            for row in solver.last_stats.variational["linear_postchecks"]:
                self.assertLessEqual(row["limit"], min(limits[-1]))
        np.testing.assert_allclose(limits[1], np.array(limits[0]) * 0.1, rtol=1e-12)
        model, solver = model_and_solver("cpu")
        configure_stopping(solver, 0.1)
        state = solver._create_articulation_state(dt=0.001)
        original, iterate = snapshot(state), solver._iterate_articulation

        def changed_candidate(dt):
            status = iterate(dt)
            solver._accepted.joint_q.fill_(1e-10)
            return status

        with patch.object(solver, "_iterate_articulation", side_effect=changed_candidate):
            solver.step(state, state, model.control(), dt=0.001)
        self.assertFalse(solver.last_stats.converged)
        self.assertTrue(solver.last_stats.rolled_back)
        self.assertGreater(solver.last_stats.convergence_ratio_q, 1)
        self.assertLess(solver.last_stats.merit_q_final, 2.5e-7)
        assert_snapshot(state, original)

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
        case["duration"] = 1.2
        full = reference(case, 8, output_dt=0.000125)
        self.assertEqual(len(full["states"]), 9601)
        np.testing.assert_allclose(full["times"][1:], np.arange(1, 9601) * 0.000125, rtol=0, atol=1e-16)
        self.assertEqual(full["times"][-1], 1.2)
        for dt in (0, float("nan"), 0.0001):
            with self.assertRaises(ValueError):
                reference(case, 64, output_dt=dt)

    def test_failed_oracle_is_not_pass(self):
        """Keep oracle failure and the complete last input instead of passing a prefix."""
        case = copy.deepcopy(specification()["cases"][0])
        case["duration"] = 0.001
        with patch.object(DiscreteOracle, "step", side_effect=RuntimeError("ORACLE_UNCERTIFIED")):
            result = run_case(case, "cpu", "ArmijoOrResNorm", h=0.000125, pcg_mode="diagnostic", graph=False)
        self.assertFalse(result["passed_same_discrete"])
        self.assertEqual(result["trace"], [])
        self.assertEqual(result["failure_state"]["failed_step"], 1)
        self.assertIn("link_vsym", result["failure_state"]["history"])

    def test_failed_single_step_is_not_committed(self):
        """Exclude a failed single-step cycle from the committed outer count."""
        case = copy.deepcopy(specification()["cases"][0])
        case["duration"] = 0.001
        with patch.object(
            _ArticulationSolver,
            "_iterate_articulation",
            return_value=_ArticulationSolver.Status.NONLINEAR_MAX_ITERATIONS,
        ):
            result = run_case(case, "cpu", "ArmijoOrResNorm", h=0.001, pcg_mode="diagnostic", graph=False)
        self.assertFalse(result["passed_same_discrete"])
        self.assertEqual(result["committed_outer_steps"], 0)
        self.assertFalse(result["trace"][0]["outer_committed"])


if __name__ == "__main__":
    unittest.main()
