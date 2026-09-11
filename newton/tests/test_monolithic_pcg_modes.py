# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check PCG mode parity, failure guards and device-only optional summaries."""

import unittest
from dataclasses import asdict, replace
from functools import partial
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import linear
from newton.tests import test_solver_monolithic_linear as reference
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _case(device, mode, name):
    constructor = linear.MonolithicLinearWorkspace
    if mode is not None:
        constructor = partial(constructor, pcg_mode=mode)
    cross = {"negative_curvature": 2.0, "tiny_curvature": 0.5, "stagnation": 0.9}.get(name, 1e-6)
    with patch.object(reference, "MonolithicLinearWorkspace", constructor):
        workspace, generation = reference._two_block_system(device, cross)
    rhs = wp.array([1e-5, 1.0, 0.0, 0.0], dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)
    config = reference._pcg_config()
    warm = linear.MonolithicPcgWarmStart.ZERO
    if name == "zero":
        rhs.zero_()
    elif name in ("negative_curvature", "tiny_curvature"):
        rhs.assign(np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32))
        if name == "tiny_curvature":
            config = replace(config, curvature_absolute_tolerance=2.0)
    elif name == "max_iterations":
        config = replace(config, maximum_iterations=1)
    elif name == "replacement":
        config = replace(config, true_residual_interval=1)
    elif name == "stagnation":
        rhs.assign(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        config = replace(config, true_residual_interval=1, stagnation_window=1, stagnation_minimum_reduction=0.5)
    elif name == "bad_preconditioner":
        original = workspace.preconditioner._matvec
        workspace.preconditioner._matvec = lambda x, affine, out, alpha, beta: original(x, affine, out, -alpha, beta)
    elif name == "failure_nonfinite":
        original = workspace.operator._matvec
        calls = []

        def failing(x, affine, out, alpha, beta):
            calls.append(True)
            original(x, affine, out, -alpha if len(calls) == 2 else alpha, beta)
            if len(calls) == 3:
                out.fill_(float("nan"))

        workspace.operator._matvec = failing
    elif name == "invalid_argument":
        y = rhs
    elif name == "warm_stale":
        warm = linear.MonolithicPcgWarmStart.SAME_GENERATION
    elif name == "nan":
        rhs.fill_(float("nan"))
    elif name == "stale":
        generation = replace(generation, assembly_sequence=2)
    elif name == "unfactored":
        workspace._factor_valid = False
    elif name in ("warm", "lambda_retry"):
        workspace.solve_pcg(rhs, y, generation=generation, warm_start=warm, config=config)
        warm = linear.MonolithicPcgWarmStart.SAME_GENERATION
        if name == "lambda_retry":
            workspace.set_regularization(0.1, generation=generation)
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)
    result = workspace.solve_pcg(rhs, y, generation=generation, warm_start=warm, config=config)
    return workspace, result, y.numpy().copy()


_CASES = (
    "normal",
    "zero",
    "negative_curvature",
    "tiny_curvature",
    "max_iterations",
    "replacement",
    "stagnation",
    "bad_preconditioner",
    "nan",
    "stale",
    "unfactored",
    "warm",
    "lambda_retry",
    "failure_nonfinite",
    "invalid_argument",
    "warm_stale",
)


def test_modes_parity(test, device):
    """Preserve mandatory decisions and solutions across both modes and corner cases."""
    optional = {"min_p_ap", "min_r_z", "initial_guess_norm", "recursive_true_residual_gap"}
    for name in _CASES:
        with test.subTest(case=name):
            _, diagnostic, yd = _case(device, "diagnostic", name)
            production, result, yp = _case(device, "production", name)
            for key, value in asdict(diagnostic).items():
                if key in optional:
                    test.assertTrue(np.isnan(getattr(result, key)).all(), key)
                elif isinstance(value, float):
                    np.testing.assert_allclose(getattr(result, key), value, rtol=2e-6, atol=1e-10)
                else:
                    test.assertEqual(getattr(result, key), getattr(diagnostic, key), key)
            np.testing.assert_allclose(yp, yd, rtol=2e-6, atol=1e-10)
            test.assertFalse(any(name.startswith("_diagnostic") for name in vars(production)))


def test_mode_validation(test, device):
    """Reject unknown modes and keep the construction-time mode read-only."""
    for mode in (None, True, "fast", 0):
        with test.assertRaises(ValueError):
            _case(device, mode if mode is not None else "", "normal")
    workspace, _, _ = _case(device, "production", "normal")
    with test.assertRaises(AttributeError):
        workspace.pcg_mode = "diagnostic"


def test_mode_work_audit(test, device):
    """Avoid optional production work, repeated scalar reads and warmed device allocations."""
    for mode in ("diagnostic", "production"):
        with test.subTest(mode=mode):
            constructor = partial(linear.MonolithicLinearWorkspace, pcg_mode=mode)
            with patch.object(reference, "MonolithicLinearWorkspace", constructor):
                workspace, generation = reference._ready_system(device, heterogeneous=True)
            rhs = wp.array(np.linspace(-1, 2, 11).astype(np.float32), device=device)
            y = wp.zeros(11, dtype=float, device=device)
            config = reference._pcg_config(linear_tolerance=1e-5, true_residual_interval=3)

            solve = partial(
                workspace.solve_pcg,
                rhs,
                y,
                generation=generation,
                warm_start=linear.MonolithicPcgWarmStart.ZERO,
                config=config,
            )

            solve()
            reads = []
            original_numpy = wp.array.numpy

            def read(array, *args, reads=reads, original_numpy=original_numpy, **kwargs):
                reads.append(array)
                return original_numpy(array, *args, **kwargs)

            allocator = wp.get_device(device).get_allocator()
            with (
                patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate,
                patch.object(wp.array, "numpy", autospec=True, side_effect=read),
                patch.object(wp, "launch", wraps=wp.launch) as launch,
            ):
                result = solve()
            test.assertEqual(result.status, linear.MonolithicLinearStatus.SUCCESS)
            test.assertEqual(allocate.call_count, 0)
            test.assertFalse(any(array is workspace._products or array is workspace._true_norms for array in reads))
            optional_kernels = (linear._record_monolithic_pcg_product, linear._record_monolithic_pcg_norms)
            kernels = [call.args[0] if call.args else call.kwargs["kernel"] for call in launch.call_args_list]
            optional_calls = [kernel for kernel in kernels if kernel in optional_kernels]
            if mode == "production":
                test.assertEqual(optional_calls, [])
            else:
                test.assertGreater(len(optional_calls), 0)
                test.assertEqual(sum(array is workspace._diagnostic_summary for array in reads), 1)
                test.assertFalse(any(array is workspace._diagnostic_norms for array in reads))
            cache_calls = [kernel for kernel in kernels if kernel is linear._cache_monolithic_pcg_denominator]
            # After fix for tet r5 regression: always recompute RHS norms, so cache is called
            # on every true residual check (not just the first), ensuring numerical consistency
            test.assertEqual(len(cache_calls), result.true_residual_checks)
            test.assertGreater(result.true_residual_checks, 1)


def test_solve_reuse(test, device):
    """Reset RHS norm caches and optional summaries through an A to B to A solve sequence."""
    for mode in ("diagnostic", "production"):
        constructor = partial(linear.MonolithicLinearWorkspace, pcg_mode=mode)
        with patch.object(reference, "MonolithicLinearWorkspace", constructor):
            workspace, generation = reference._two_block_system(device, 1e-6)
        rhs = wp.array([1e-5, 1.0, 0.0, 0.0], dtype=float, device=device)
        y = wp.zeros(4, dtype=float, device=device)
        results, solutions = [], []
        for value in (1.0, 1000.0, 1.0):
            rhs.assign(np.array([1e-5, value, 0.0, 0.0], dtype=np.float32))
            results.append(
                workspace.solve_pcg(
                    rhs,
                    y,
                    generation=generation,
                    warm_start=linear.MonolithicPcgWarmStart.ZERO,
                    config=reference._pcg_config(),
                )
            )
            solutions.append(y.numpy().copy())
        np.testing.assert_array_equal(solutions[0], solutions[2])
        for key, value in asdict(results[0]).items():
            if isinstance(value, (float, tuple)):
                np.testing.assert_allclose(value, getattr(results[2], key), rtol=0, atol=0)
            else:
                test.assertEqual(value, asdict(results[2])[key])


def test_recursive_check_double_precision(test, device):
    """Match the original double-precision host threshold at float32 decision boundaries."""
    workspace, _, _ = _case(device, "production", "normal")
    for denominator, tolerance in ((1.0, 0.1000000007), (1e20, 1e-4), (1e-20, 1e-4)):
        threshold = (tolerance * denominator) ** 2
        nearest = np.float32(threshold)
        for value in (np.nextafter(nearest, np.float32(0)), nearest, np.nextafter(nearest, np.float32(np.inf))):
            workspace._products.assign(np.array([value, 0, 0], dtype=np.float32))
            workspace._rhs_denominator.assign(np.array([denominator], dtype=np.float64))
            workspace._status.zero_()
            wp.launch(
                linear._pack_monolithic_pcg_check,
                1,
                [
                    workspace._products,
                    workspace._status,
                    workspace._rhs_denominator,
                    tolerance,
                    False,
                    workspace._pcg_packet,
                ],
                device=device,
            )
            packet = workspace._pcg_packet.numpy()[0]
            test.assertEqual(bool(packet["check"]), float(value) <= threshold)
            test.assertEqual(packet.dtype["status"].kind, "i")


class TestMonolithicPcgModes(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _function in (
        test_modes_parity,
        test_mode_validation,
        test_mode_work_audit,
        test_solve_reuse,
        test_recursive_check_double_precision,
    ):
        add_function_test(TestMonolithicPcgModes, _function.__name__, _function, devices=[_device])

if __name__ == "__main__":
    unittest.main(verbosity=2)
