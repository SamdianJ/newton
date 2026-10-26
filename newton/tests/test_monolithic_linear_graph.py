# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture the complete fixed chain and audit its actual hot-path API calls."""

import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_monolithic_linear_production import impl, make_plan, solve
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import benchmark_pcg as benchmark


def no_host_or_allocations():
    stack = ExitStack()
    for name in ("numpy",):
        stack.enter_context(patch.object(wp.array, name, side_effect=AssertionError("hot D2H")))
    for name in ("synchronize", "synchronize_device", "synchronize_stream", "empty", "zeros", "ones", "clone"):
        stack.enter_context(patch.object(wp, name, side_effect=AssertionError("hot sync/allocation: " + name)))
    return stack


def test_hot_path(test, device):
    """Enqueue success, failure and warm starts without host reads or allocations."""
    plan = make_plan(device, np.eye(4), [1, 2, 3, 4])
    wp.load_module(module=impl._initialize_state.module, device=device)
    for value, warm in ((1, False), (-1, False), (1, False), (1, True)):
        plan.values.assign((np.eye(4) * value).astype(np.float32).reshape(-1))
        plan.set_solve_parameters(regularization=0, zero_initial_guess=not warm)
        with no_host_or_allocations():
            plan.enqueue_solve()
        result = plan.collect_result()
        test.assertEqual(
            result.status, impl.ProductionStatus.SUCCESS if value > 0 else impl.ProductionStatus.NON_POSITIVE_CURVATURE
        )


def test_graph_replay(test, device):
    """Replay A-B-A, changed RHS/lambda/scales/factors, failure and warm states."""
    cfg = impl.MaskedPcgConfig(maximum_iterations=11, true_residual_interval=5)
    graph_plan = make_plan(device, np.eye(4), [1, 2, 3, 4], config=cfg)
    reference = make_plan(device, np.eye(4), [1, 2, 3, 4], config=cfg)
    graph_plan.set_solve_parameters(regularization=0)
    wp.load_module(module=impl._initialize_state.module, device=device)
    with no_host_or_allocations():
        graph_plan.capture()
    original_graph = graph_plan.graph
    for diagonal, rhs, lam, scale, warm in (
        ([1] * 4, [1, 2, 3, 4], 0, 1, False),
        ([2, 3, 4, 5], [2, 1, -2, 3], 0.5, 2, False),
        ([-1] * 4, [1, 0, 0, 0], 0, 1, False),
        ([1] * 4, [float("nan"), 0, 0, 0], 0, 1, False),
        ([1] * 4, [1, 2, 3, 4], 0, 1, False),
        ([1] * 4, [1, 2, 3, 4], 0, 1, True),
        ([1] * 4, [0] * 4, 0, 1, False),
    ):
        for plan in (graph_plan, reference):
            plan.values.assign(np.diag(diagonal).astype(np.float32).reshape(-1))
            plan.rhs.assign(np.asarray(rhs, dtype=np.float32))
            plan.scale.fill_(scale)
            factors = np.sqrt(np.asarray(diagonal) * scale**2 + lam) if min(diagonal) > 0 else np.ones(4)
            plan.q_cholesky.assign(np.array([[factors[0]]], dtype=np.float32))
            plan.x_cholesky.assign(np.array([np.diag(factors[1:])], dtype=np.float32))
            plan.set_solve_parameters(regularization=lam, zero_initial_guess=not warm)
        with no_host_or_allocations():
            graph_plan.replay()
            reference.enqueue_solve()
        actual, expected = graph_plan.collect_result(), reference.collect_result()
        test.assertEqual(
            (
                actual.status,
                actual.iterations,
                actual.true_residual_checks,
                actual.residual_replacements,
                actual.convergence_slot,
            ),
            (
                expected.status,
                expected.iterations,
                expected.true_residual_checks,
                expected.residual_replacements,
                expected.convergence_slot,
            ),
        )
        np.testing.assert_allclose(actual.ratios, expected.ratios, rtol=1e-5, atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(graph_plan.y.numpy(), reference.y.numpy(), rtol=1e-5, atol=1e-7, equal_nan=True)
        test.assertIs(graph_plan.graph, original_graph)
    graph_plan.replay()
    with test.assertRaisesRegex(RuntimeError, "Collect"):
        graph_plan.replay()
    graph_plan.collect_result()
    with wp.ScopedStream(wp.Stream(device)):
        with test.assertRaisesRegex(ValueError, "Stream"):
            graph_plan.replay()


def test_graph_terminal_boundaries(test, device):
    """Capture maximum-budget, zero-preconditioned, overflow and recovery terminal paths."""
    cases = [
        (np.diag([1, 2, 3, 4]), [1, 2, 3, 4], {"config": impl.MaskedPcgConfig(maximum_iterations=1)}, False),
        (
            np.diag(np.linspace(1, 100, 64)),
            [1] * 64,
            {"config": impl.MaskedPcgConfig(maximum_iterations=11, true_residual_interval=5, linear_tolerance=1e-12)},
            False,
        ),
        (np.eye(4), [1, 0, 0, 0], {"lower_q": [[1e30]], "lower_x": [np.eye(3) * 1e30]}, False),
        (np.eye(4), [3e38] * 4, {}, False),
        (np.zeros((4, 4)), [0] * 4, {"scale": [1e30] * 4, "empty": True}, True),
    ]
    for matrix, rhs, kwargs, warm in cases:
        plans = [make_plan(device, matrix, rhs, **kwargs) for _ in range(2)]
        results = []
        for i, plan in enumerate(plans):
            if warm:
                plan.y.fill_(1e20)
            plan.delta.fill_(23)
            results.append(solve(plan, warm=warm, graph=bool(i)))
        test.assertNotEqual(results[0].status, impl.ProductionStatus.SUCCESS)
        for key in (
            "status",
            "iterations",
            "true_residual_checks",
            "residual_replacements",
            "last_true_slot",
            "failure_slot",
            "output_valid",
        ):
            test.assertEqual(getattr(results[0], key), getattr(results[1], key), key)
        np.testing.assert_array_equal(plans[1].delta.numpy(), [23] * plans[1].n)


class TestMonolithicPcgProductionGraph(unittest.TestCase):
    pass


def test_workspace_graph_reuse(test, device):
    """Reuse numeric generations but reject stale factors, changed topology and storage."""
    case = benchmark.synthetic_cases()[0]
    runner = benchmark.Runner(case, "production_graph", device, benchmark.pcg_config())
    test.assertTrue(runner.solve().graph_rebuilt)
    plan = runner.workspace._masked_pcg_plan
    graph = plan.graph
    old_generation = runner.workspace._generation
    for regularization, rhs_factor in ((0.25, 2), (0.0, 1)):
        runner.refresh_generation()
        test.assertEqual(runner.solve().status, impl.MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
        with test.assertRaisesRegex(ValueError, "Stale"):
            plan.replay()
        runner.workspace.set_regularization(regularization, generation=runner.workspace._generation)
        runner.factor()
        runner.rhs.assign(case.arrays["rhs_hat"] * rhs_factor)
        result = runner.solve()
        test.assertEqual(result.status, impl.MonolithicLinearStatus.SUCCESS)
        test.assertFalse(result.graph_rebuilt)
        test.assertIs(runner.workspace._masked_pcg_plan, plan)
        test.assertIs(plan.graph, graph)
        changed = replace(
            case,
            arrays={**case.arrays, "rhs_hat": case.arrays["rhs_hat"] * rhs_factor},
            metadata={**case.metadata, "lambda": regularization},
        )
        test.assertTrue(
            benchmark.Reference(changed, runner.config).measure(runner.y.numpy())["independent_residual_pass"]
        )
    stale = runner.workspace.solve_pcg(
        runner.rhs,
        runner.y,
        generation=old_generation,
        warm_start=impl.MonolithicPcgWarmStart.ZERO,
        config=runner.config,
    )
    test.assertEqual(stale.status, impl.MonolithicLinearStatus.STALE_GENERATION)
    rhs = runner.rhs
    plan.replay()
    runner.rhs = wp.clone(rhs)
    with test.assertRaisesRegex(RuntimeError, "Collect"):
        runner.solve()
    test.assertIs(runner.workspace._masked_pcg_plan, plan)
    plan.collect_result()
    runner.rhs = rhs
    matrix = runner.workspace.k_global_scalar_bsr
    matrix.values = wp.clone(matrix.values)
    with test.assertRaises(ValueError):
        plan.replay()
    test.assertTrue(runner.solve().graph_rebuilt)
    runner.refresh_generation()
    # Equivalent matrix with each row's entries reversed: valid but a different layout.
    columns = matrix.columns.numpy().reshape(case.n, case.n)[:, ::-1].copy().reshape(-1)
    values = matrix.values.numpy().reshape(case.n, case.n)[:, ::-1].copy().reshape(-1)
    matrix.columns.assign(columns)
    matrix.values.assign(values)
    runner.factor()
    test.assertTrue(runner.solve().graph_rebuilt)


def test_conditional_slot_budget(test, device):
    """Stop at checked terminal states while retaining WAIT_CHECK and partial budgets."""
    for maximum, interval, rhs, expected_slots in (
        (200, 10, [0, 0, 0, 0], 0),
        (200, 10, [1, 2, 3, 4], 10),
        (11, 5, [1, 2, 3, 4], 5),
        (3, 10, [1, 2, 3, 4], 3),
    ):
        plan = make_plan(
            device,
            np.eye(4),
            rhs,
            config=impl.MaskedPcgConfig(maximum_iterations=maximum, true_residual_interval=interval),
        )
        result = solve(plan, graph=True)
        test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
        test.assertEqual(result.iteration_slots, expected_slots)
        test.assertEqual(result.convergence_slot, expected_slots)
        test.assertLessEqual(result.iterations, result.iteration_slots)
    with patch.object(wp, "is_conditional_graph_supported", return_value=False):
        fallback = make_plan(device, np.eye(4), [1, 2, 3, 4])
    result = solve(fallback, graph=True)
    test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
    test.assertEqual(result.iteration_slots, fallback.config.maximum_iterations)


def test_fused_products(test, device):
    """Overwrite a masked dot in one launch, including partial tiles and nonfinite inputs."""
    plan = make_plan(device, np.eye(4), [1, 2, 3, 4])
    kernel = impl._make_products_kernel(256, wp.float32)
    for n in (1, 33, 2051):
        a = np.linspace(-2, 3, n, dtype=np.float32)
        b = np.sin(a).astype(np.float32)
        av, bv = (wp.array(v, device=device) for v in (a, b))
        wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 200], device=device)
        plan.products.fill_(123)
        wp.launch_tiled(kernel, dim=1, inputs=[av, bv, plan.products, plan.state], block_dim=128, device=device)
        test.assertAlmostEqual(float(plan.products.numpy()[0]), float(a.astype(float) @ b.astype(float)), delta=1e-3)
    av.fill_(float("nan"))
    wp.launch_tiled(kernel, dim=1, inputs=[av, bv, plan.products, plan.state], block_dim=128, device=device)
    test.assertFalse(np.isfinite(plan.products.numpy()[0]))
    state = plan.state.numpy()
    state["enabled"] = 0
    plan.state.assign(state)
    plan.products.fill_(23)
    wp.launch_tiled(kernel, dim=1, inputs=[av, bv, plan.products, plan.state], block_dim=128, device=device)
    test.assertEqual(plan.products.numpy()[0], 23)
    n = 2050
    large = impl.MaskedPcgPlan(
        offsets=wp.array(np.arange(n + 1, dtype=np.int32), device=device),
        columns=wp.array(np.arange(n, dtype=np.int32), device=device),
        values=wp.ones(n, dtype=float, device=device),
        scale=wp.ones(n, dtype=float, device=device),
        q_cholesky=wp.ones((1, 1), dtype=float, device=device),
        x_cholesky=wp.array(np.repeat(np.eye(3)[None], (n - 1) // 3, axis=0), dtype=wp.mat33, device=device),
        rhs_hat=wp.ones(n, dtype=float, device=device),
        y=wp.zeros(n, dtype=float, device=device),
        config=impl.MaskedPcgConfig(maximum_iterations=3, true_residual_interval=2),
    )
    test.assertIsNone(large._products_kernel)
    result = solve(large, graph=True)
    test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
    test.assertEqual(result.iteration_slots, 2)
    np.testing.assert_array_equal(large.y.numpy(), np.ones(n))


add_function_test(TestMonolithicPcgProductionGraph, "test_hot_path", test_hot_path, devices=get_test_devices())
add_function_test(
    TestMonolithicPcgProductionGraph,
    "test_workspace_graph_reuse",
    test_workspace_graph_reuse,
    devices=[d for d in get_test_devices() if d.is_cuda],
)
add_function_test(
    TestMonolithicPcgProductionGraph,
    "test_conditional_slot_budget",
    test_conditional_slot_budget,
    devices=[d for d in get_test_devices() if d.is_cuda],
)
add_function_test(
    TestMonolithicPcgProductionGraph,
    "test_fused_products",
    test_fused_products,
    devices=[d for d in get_test_devices() if d.is_cuda],
)
add_function_test(
    TestMonolithicPcgProductionGraph,
    "test_graph_replay",
    test_graph_replay,
    devices=[device for device in get_test_devices() if device.is_cuda],
)

add_function_test(
    TestMonolithicPcgProductionGraph,
    "test_graph_terminal_boundaries",
    test_graph_terminal_boundaries,
    devices=[device for device in get_test_devices() if device.is_cuda],
)
