# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Capture the complete fixed chain and audit its actual hot-path API calls."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_monolithic_pcg_production import impl, make_plan, solve
from newton.tests.unittest_utils import add_function_test, get_test_devices


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
    wp.load_module(module=impl.__name__, device=device)
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
    wp.load_module(module=impl.__name__, device=device)
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
        np.testing.assert_array_equal(plans[1].delta.numpy(), [23] * 4)


class TestMonolithicPcgProductionGraph(unittest.TestCase):
    pass


add_function_test(TestMonolithicPcgProductionGraph, "test_hot_path", test_hot_path, devices=get_test_devices())
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
