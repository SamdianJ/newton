# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify measured nonlinear diagnostics against independently evaluated norms."""

import unittest
from unittest.mock import patch

import numpy as np

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_solver_monolithic_step import _free_scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_current_scale_diagnostics(test, device):
    """Report the actual current-scale reference and all three convergence gates."""
    fixture, solver = _free_scene(device, newton_max_iterations=1)
    positions = fixture.state.particle_q.numpy().copy()
    positions[1, 0] += 0.015
    fixture.state.particle_q.assign(positions)
    with test.assertLogs("newton._src.solvers.monolithic.solver_monolithic", level="WARNING"):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    stats = solver.last_stats
    test.assertEqual(stats.status, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
    test.assertEqual(stats.scale_generation, solver._generation.assembly_sequence)
    residual = solver._transaction.accepted.residual.numpy().astype(np.float64)
    scale = solver._linear.scale.numpy().astype(np.float64)
    reference = solver._residual_ref.numpy().astype(np.float64)
    for block, selection in (("global", slice(None)), ("q", slice(0, 2)), ("x", slice(2, None))):
        suffix = "" if block == "global" else "_" + block
        expected_reference = np.linalg.norm((scale * reference)[selection]) / np.sqrt(reference[selection].size)
        expected_merit = np.linalg.norm((scale * residual)[selection]) / np.sqrt(residual[selection].size)
        denominator = (
            getattr(solver._config, "merit_absolute_" + block)
            + getattr(solver._config, "merit_relative_" + block) * expected_reference
        )
        np.testing.assert_allclose(getattr(stats, "merit" + suffix + "_reference"), expected_reference, rtol=2e-6)
        np.testing.assert_allclose(
            getattr(stats, "convergence_ratio" + suffix), expected_merit / denominator, rtol=2e-6
        )
    test.assertGreater(max(stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x), 1.0)
    np.testing.assert_allclose(stats.raw_residual_q_norm, np.linalg.norm(residual[:2]), rtol=1e-12)
    np.testing.assert_allclose(stats.raw_residual_x_norm, np.linalg.norm(residual[2:]), rtol=1e-12)
    scaled_step = stats.accepted_alpha * solver._y.numpy().astype(np.float64)
    for suffix, selection in (("", slice(None)), ("_q", slice(0, 2)), ("_x", slice(2, None))):
        np.testing.assert_allclose(
            getattr(stats, "scaled_step" + suffix),
            np.linalg.norm(scaled_step[selection]) / np.sqrt(scaled_step[selection].size),
            rtol=2e-6,
        )


def test_unmeasured_and_effective_settings(test, device):
    """Distinguish absent solves and steps from measured zero values and expose the actual gates."""
    fixture, solver = _free_scene(device)
    test.assertEqual(solver.last_stats.scale_generation, -1)
    test.assertTrue(np.isnan(solver.last_stats.convergence_ratio))
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    stats = solver.last_stats
    test.assertTrue(stats.converged)
    for value in (stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x):
        test.assertLessEqual(value, 1.0)
    test.assertTrue(np.isnan(stats.scaled_step))
    test.assertEqual(stats.merit_q_initial, stats.merit_q_reference)
    test.assertEqual(stats.merit_x_initial, stats.merit_x_reference)
    test.assertEqual(stats.merit_noise, solver._config.merit_noise)
    test.assertEqual(stats.residual_floor_global, solver._config.residual_floor_global)
    test.assertEqual(stats.residual_floor_q, solver._config.residual_floor_q)
    test.assertEqual(stats.residual_floor_x, solver._config.residual_floor_x)


def test_linear_diagnostics_survive_final_assembly(test, device):
    """Accumulate actual factor attempts across assemblies and reset solve diagnostics on the next step."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -9.81))
    factor, solve = solver._linear.factor_actor_preconditioner, solver._linear.solve_pcg
    counts, results = [], []

    def measure_factor(*args, **kwargs):
        before = (solver._linear.factor_setup_count, solver._linear.factor_failure_count)
        result = factor(*args, **kwargs)
        counts.append((solver._linear.factor_setup_count - before[0], solver._linear.factor_failure_count - before[1]))
        return result

    def measure_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        results.append(result)
        return result

    with (
        patch.object(solver._linear, "factor_actor_preconditioner", side_effect=measure_factor),
        patch.object(solver._linear, "solve_pcg", side_effect=measure_solve),
    ):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    stats = solver.last_stats
    test.assertGreater(len(results), 0)
    test.assertEqual(stats.factor_setup_count, sum(value[0] for value in counts))
    test.assertEqual(stats.factor_failure_count, sum(value[1] for value in counts))
    test.assertEqual(solver._linear.factor_setup_count, 0)
    test.assertEqual(stats.warm_start, "zero")
    test.assertEqual(stats.initial_guess_norm, results[-1].initial_guess_norm)
    test.assertEqual(stats.stagnation_window, results[-1].stagnation_window)
    test.assertEqual(stats.linear_scale_generation, results[-1].generation.assembly_sequence)
    test.assertLess(stats.linear_scale_generation, stats.scale_generation)
    np.testing.assert_allclose(
        [stats.recursive_true_residual_gap, stats.recursive_true_residual_gap_q, stats.recursive_true_residual_gap_x],
        results[-1].recursive_true_residual_gap,
        equal_nan=True,
    )
    test.assertEqual(stats.q_block_count, 1)
    test.assertEqual(stats.particle_block_count, 4)
    fixture.model.gravity.zero_()
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    stats = solver.last_stats
    test.assertEqual(stats.factor_setup_count, 0)
    test.assertEqual(stats.factor_failure_count, 0)
    test.assertIsNone(stats.warm_start)
    test.assertIsNone(stats.stagnation_window)
    test.assertTrue(np.isnan(stats.initial_guess_norm))
    test.assertTrue(np.isnan(stats.recursive_true_residual_gap))
    test.assertEqual(stats.linear_scale_generation, -1)


class TestMonolithicDiagnostics(unittest.TestCase):
    """Exercise nonlinear diagnostics on CPU and CUDA."""


for _test in (
    test_current_scale_diagnostics,
    test_unmeasured_and_effective_settings,
    test_linear_diagnostics_survive_final_assembly,
):
    add_function_test(TestMonolithicDiagnostics, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
