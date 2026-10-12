# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the delivered direction against the equation actually solved."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp
import warp.sparse as sparse

from newton._src.solvers.monolithic.iteration import blocks
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.benchmark_pcg import pcg_config


def modes(device):
    return [("diagnostic", False), ("production", False)] + ([("production", True)] if device.is_cuda else [])


def test_default_unregularized_failure(test, device):
    """Stop after a failed default zero-lambda direction and preserve explicit retry support."""
    for mode, graph in modes(device):
        model = scene(device, contact=False)
        run = solver(model, mode=mode, graph=graph)
        state = run.create_state(dt=0.001)
        before = snapshot(state)
        for configured in (False, True):
            if configured:
                run._config = replace(run._config, regularization_values=(0.0, 0.01))
            attempted = []

            def decline(*args, attempted=attempted, **kwargs):
                attempted.append(kwargs["regularization"])
                return False

            with patch("newton._src.solvers.monolithic.iteration.solve_direction", side_effect=decline):
                run.step(state, state, model.control(), dt=0.001)
            test.assertEqual(attempted, [0.0, 0.01] if configured else [0.0])
            test.assertEqual(run.last_stats.regularization_retries, int(configured))
            test.assertEqual(run.last_stats.status, run.Status.REGULARIZATION_EXHAUSTED)
            test.assertFalse(run.last_stats.converged)
            test.assertTrue(run.last_stats.rolled_back)
            assert_snapshot(test, state, before)


def identity_workspace(device, mode, graph, *, mixed=False):
    if mixed:
        run = contact_solver(contact_scene(device), mode=mode, graph=graph)
        state = contact_state(run, "chain")
    else:
        model = scene(device, contact=False)
        run = solver(model, mode=mode, graph=graph)
        state = run.create_state(dt=0.001)
    run.step(state, state, run.model.control(), dt=0.001)
    w = run._linear
    n = run._delta.size
    rows = wp.array(np.arange(n, dtype=np.int32), device=device)
    w.k_global_scalar_bsr = sparse.bsr_from_triplets(n, n, rows, rows, wp.ones(n, dtype=float, device=device))
    w.scale.fill_(1)
    w._factor_valid = False
    w.set_regularization(0.0, generation=run._generation)
    run._require_linear(w.factor_actor_preconditioner(generation=run._generation, pivot_tolerance=0.0))
    run._variational["linear_postchecks"] = []
    run._variational["linear_direction_attempts"] = []
    return run


def qualify(run, residual, limits, *, budget=20, regularization=0.0):
    from newton._src.solvers.monolithic.iteration import solve_direction  # noqa: PLC0415

    run._linear.set_regularization(regularization, generation=run._generation)
    run._require_linear(run._linear.factor_actor_preconditioner(generation=run._generation, pivot_tolerance=0.0))
    scale = run._linear.scale.numpy().astype(np.float64)
    run._rhs.assign((-scale * residual).astype(np.float32))
    config = replace(
        pcg_config(budget),
        residual_norm_limits=tuple(limits[name] for name in ("global", "q", "x")),
        group_norm_limits=tuple(limits[name] for name, _ in blocks(run)),
    )
    if run._device_iteration:
        run._accepted_r.assign(residual)
        residual = run._accepted_r
    return solve_direction(run, residual, scale, limits, config, iteration=0, retry=0, regularization=regularization)


def test_native_correction(test, device):
    """Correct a native/internal boundary disagreement without tightening the initial solve."""
    for mode, graph in modes(device):
        for mixed in (False, True):
            run = identity_workspace(device, mode, graph, mixed=mixed)
            parts = dict(blocks(run))
            target = next(name for name in parts if name.startswith("rigid_") and name.endswith("_t")) if mixed else "q"
            gate = float(np.float32(1e-12 if mixed else 1e-7))
            limits = dict.fromkeys(parts, 1.0)
            limits[target] = gate
            residual = np.zeros(run._delta.size)
            residual[parts[target].start] = gate * (1 + 1e-9)
            original_rhs = (-residual).astype(np.float32)
            test.assertTrue(qualify(run, residual, limits))
            rows = run._variational["linear_postchecks"]
            test.assertEqual(len(rows), 2)
            test.assertGreater(rows[0]["blocks"][target]["residual"], gate)
            test.assertLessEqual(rows[0]["blocks"][target]["internal_residual"], gate)
            test.assertTrue(all(v["residual"] <= v["limit"] for v in rows[-1]["blocks"].values()))
            test.assertEqual(rows[-1]["correction"], 1)
            test.assertLessEqual(rows[-1]["total_iterations"], 20)
            np.testing.assert_array_equal(run._rhs.numpy(), original_rhs)
            first = run._delta.numpy().copy()
            test.assertTrue(qualify(run, residual, limits))
            np.testing.assert_array_equal(first, run._delta.numpy())


def test_precision_floor_and_budget(test, device):
    """Report unrepresentable directions and consume one shared iteration budget."""
    for mode, graph in modes(device):
        for budget, reason in ((1, "direction_accuracy_budget_exhausted"), (20, "direction_precision_unreachable")):
            run = identity_workspace(device, mode, graph)
            residual = np.zeros(run._delta.size)
            residual[0] = -1.0 - 2**-25
            limits = {name: 1e-9 for name, _ in blocks(run)}
            regularization = 1.0 if budget == 20 else 0.0
            test.assertFalse(qualify(run, residual, limits, budget=budget, regularization=regularization))
            attempt = run._variational["linear_direction_attempts"][-1]
            test.assertEqual(attempt["failure_reason"], reason)
            test.assertLessEqual(sum(p["iterations"] for p in attempt["passes"]), budget)
            test.assertEqual(run._delta.numpy()[0], 1.0 / (1 + regularization))
            test.assertTrue(np.isfinite(run._delta.numpy()).all())


def test_regularized_trial_and_final_gate(test, device):
    """Require real nonlinear improvement and final physical convergence after linear acceptance."""
    for mode, graph in modes(device):
        for regularization, converged in ((0.01, True), (1e10, False)):
            model = scene(device, contact=False)
            run = solver(model, mode=mode, graph=graph)
            run._config = replace(run._config, regularization_values=(regularization,))
            state = run.create_state(dt=0.001)
            run.step(state, state, model.control(), dt=0.001)
            test.assertEqual(run.last_stats.converged, converged, run.last_stats)
            posts = run.last_stats.variational["linear_postchecks"]
            test.assertTrue(posts)
            test.assertTrue(all(all(v["residual"] <= v["limit"] for v in row["blocks"].values()) for row in posts))
            test.assertTrue(
                any(row["blocks"]["global"]["original_residual"] > row["blocks"]["global"]["limit"] for row in posts)
            )
            test.assertTrue(run.last_stats.variational["trials"])
            if converged:
                test.assertTrue(
                    any(row["accepted"] and row["improved"] for row in run.last_stats.variational["trials"])
                )
            else:
                test.assertTrue(run.last_stats.rolled_back)
                test.assertGreater(run.last_stats.convergence_ratio, 1)


def test_regularized_posterior(test, device):
    """Check recovered FP32 directions and actual FP32 lambda in native precision."""
    for precision in ("fp64", "fp32"):
        model = scene(device)
        run = solver(model, precision=precision, mode="production", graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix.astype(np.float64)
        scale = run._linear.scale.numpy().astype(np.float64)
        residual = np.linspace(0.2, -0.1, len(matrix)).astype(run._numpy_dtype)
        for requested in (0.0, 1.234567891):
            actual = float(np.float32(requested))
            run._linear.set_regularization(requested, generation=run._generation)
            a = scale[:, None] * matrix * scale[None, :] + actual * np.eye(len(matrix))
            delta = (scale * np.linalg.solve(a, -scale * residual)).astype(np.float32)
            run._delta.assign(delta)
            error = run._physical_linear_error(None, delta, residual, scale)
            original = scale * (matrix @ delta.astype(float) + residual)
            regularization = actual * (delta.astype(float) / scale)
            np.testing.assert_allclose(
                error, original + regularization, atol=2e-6 if precision == "fp32" else 1e-14, rtol=1e-6
            )
            if requested:
                test.assertGreater(np.linalg.norm(original), 1e-5)
                test.assertLess(np.linalg.norm(error), np.linalg.norm(original) * 1e-4)


def test_extreme_scale_and_lambda_updates(test, device):
    """Use actual scale and lambda values on repeated graph bindings, including underflow."""
    model = scene(device)
    run = solver(model, mode="production", graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    run.step(state, state, model.control(), dt=0.001)
    matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix.astype(float)
    direction = np.linspace(-0.3, 0.7, len(matrix)).astype(np.float32)
    residual = np.linspace(0.2, -0.1, len(matrix))
    run._delta.assign(direction)
    run._linear.scale.assign(np.geomspace(1e-8, 1e8, len(matrix)).astype(np.float32))
    scale = run._linear.scale.numpy().astype(float)
    captures = None
    for requested in (0.0, 1.234567891, 1e-50, 0.1):
        run._linear.set_regularization(requested, generation=run._generation)
        actual = run._physical_linear_error(None, direction, residual, scale)
        original = scale * (matrix @ direction.astype(float) + residual)
        term = float(np.float32(requested)) * direction.astype(float) / scale
        magnitude = abs(scale) * (abs(matrix) @ abs(direction.astype(float)) + abs(residual)) + abs(term)
        test.assertTrue(np.all(abs(actual - original - term) <= 64 * np.finfo(float).eps * magnitude))
        if device.is_cuda:
            current = run._graphs.summary()["segments"]["coupled_posterior"]["captures"]
            if captures is None:
                captures = current
            test.assertEqual(current, captures)


class TestDirection(unittest.TestCase):
    """Exercise native posterior gates through the real sparse solver buffers."""


add_function_test(TestDirection, "test_regularized_posterior", test_regularized_posterior, devices=get_test_devices())
for _test in (
    test_default_unregularized_failure,
    test_native_correction,
    test_precision_floor_and_budget,
    test_regularized_trial_and_final_gate,
    test_extreme_scale_and_lambda_updates,
):
    add_function_test(TestDirection, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
