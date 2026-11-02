# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check sparse-content reuse and recovery at asynchronous PCG boundaries."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
)
from newton.tests.test_monolithic_linear_production import impl
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import benchmark_pcg as benchmark


def test_sparse_content_reuse(test, device):
    """Reuse fixed storage across A/B/A, zero pruning and full capacity in both precisions."""
    for dtype, numpy_dtype in ((wp.float32, np.float32), (wp.float64, np.float64)):
        workspace = MonolithicLinearWorkspace(
            MonolithicLinearLayout(1, 1),
            MonolithicLinearCapacities(16, 1, 0, 100, 102),
            wp.array([0], dtype=int, device=device),
            device=device,
            dtype=dtype,
            pcg_mode="production",
            use_cuda_graph=device.is_cuda,
        )
        workspace._set_fixed_triplet_pattern(
            global_rows=np.arange(4), global_columns=np.arange(4), internal_rows=[0], internal_columns=[0]
        )
        workspace._enable_fixed_bsr_storage()
        diagonal = np.array([2, 3, 4, 5], dtype=numpy_dtype)
        rhs, y = wp.ones(4, dtype=dtype, device=device), wp.zeros(4, dtype=dtype, device=device)
        dynamic = wp.array(diagonal, device=device)
        plan = graph = None
        nnz = []
        for sequence, contact in enumerate((1, 2, 1, 0, 3), 1):
            generation = MonolithicLinearGeneration(1, sequence, 1, sequence)
            assembly = workspace.begin_assembly(generation)
            assembly.aq_actor_dense.assign(diagonal[:1].reshape(1, 1))
            assembly.ax_internal_triplets.values.assign(np.diag(diagonal[1:])[None])
            triplets = assembly.global_scalar_triplets
            triplets.values[:4].assign(diagonal)
            vector = np.array([1, -1, 1, -1], dtype=numpy_dtype) if contact == 3 else np.zeros(4, numpy_dtype)
            if contact in (1, 2):
                vector[0], vector[contact] = 1, -1
            # Fill every reserved slot, including numerical zeros; the dense
            # case reaches the output capacity after duplicate reduction.
            rows, columns = np.where(~np.eye(4, dtype=bool))
            contribution = np.outer(vector, vector)
            triplets.values[:4].assign(diagonal + np.diag(contribution))
            triplets.rows[4:].assign(rows.astype(np.int32))
            triplets.columns[4:].assign(columns.astype(np.int32))
            triplets.values[4:].assign(contribution[rows, columns])
            triplets.count.fill_(16)
            built = workspace.finalize_assembly(generation=generation)
            test.assertEqual(built.status, impl.MonolithicLinearStatus.SUCCESS)
            nnz.append(int(workspace.k_global_scalar_bsr.offsets.numpy()[-1]))
            workspace.build_scaling(dynamic, epsilon_d=0.01, generation=generation)
            workspace.set_regularization(0.03, generation=generation)
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0)
            matrix = workspace.k_global_scalar_bsr
            layout_pointers = {matrix.offsets.ptr, matrix.columns.ptr}
            read = wp.array.numpy

            def guarded_read(array, *args, plan=plan, layout_pointers=layout_pointers, read=read, **kwargs):
                if plan is not None and array.ptr in layout_pointers:
                    raise AssertionError("Steady solve downloaded sparse layout")
                return read(array, *args, **kwargs)

            with patch.object(wp.array, "numpy", guarded_read):
                result = workspace.solve_pcg(
                    rhs,
                    y,
                    generation=generation,
                    warm_start=impl.MonolithicPcgWarmStart.ZERO,
                    config=benchmark.pcg_config(100),
                )
            test.assertEqual(result.status, impl.MonolithicLinearStatus.SUCCESS)
            if plan is None:
                plan = workspace._masked_pcg_plan
                graph = plan.graph
            else:
                test.assertIs(workspace._masked_pcg_plan, plan)
                test.assertIs(plan.graph, graph)
                test.assertFalse(result.graph_rebuilt)
            scale = workspace.scale.numpy().astype(float)
            expected = scale[:, None] * (np.diag(diagonal) + contribution) * scale[None, :] + np.eye(4) * 0.03
            np.testing.assert_allclose(expected @ y.numpy(), np.ones(4), rtol=3e-5, atol=3e-6)
        test.assertEqual(nnz, [6, 6, 6, 4, 16])
        generation = MonolithicLinearGeneration(1, 6, 1, 6)
        assembly = workspace.begin_assembly(generation)
        assembly.global_scalar_triplets.rows[4:5].fill_(4)
        assembly.global_scalar_triplets.count.fill_(5)
        test.assertEqual(
            workspace.finalize_assembly(generation=generation).status,
            impl.MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE,
        )
        with test.assertRaisesRegex(ValueError, "Stale"):
            plan.set_solve_parameters(regularization=0.03)


def _runner(device, *, graph=False):
    return benchmark.Runner(
        benchmark.synthetic_cases()[0], "production_graph" if graph else "production", device, benchmark.pcg_config()
    )


def _assert_recovered(test, runner, plan):
    test.assertFalse(plan._in_flight)
    test.assertIsNone(runner.workspace._warm_start_key)
    with test.assertRaises(RuntimeError):
        plan.collect_result()
    plan.set_solve_parameters(regularization=runner.workspace._lambda)
    with test.assertRaises(RuntimeError):
        plan.collect_result()
    result = runner.solve()
    test.assertEqual(result.status, impl.MonolithicLinearStatus.SUCCESS)
    case = benchmark.synthetic_cases()[0]
    test.assertTrue(benchmark.Reference(case, runner.config).measure(runner.y.numpy())["independent_residual_pass"])


def test_enqueue_and_collect_failures(test, device):
    """Recover partial enqueue and failed packet reads without accepting old results."""
    runner = _runner(device)
    runner.solve()
    plan = runner.workspace._masked_pcg_plan
    launch = plan._launch
    calls = 0

    def partial_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected enqueue")
        return launch(*args, **kwargs)

    with patch.object(plan, "_launch", partial_launch), test.assertRaisesRegex(RuntimeError, "injected"):
        runner.solve()
    _assert_recovered(test, runner, plan)
    read = wp.array.numpy
    for after_read in (False, True):

        def failed_read(array, *args, after_read=after_read, **kwargs):
            if array is plan.state:
                if after_read:
                    read(array, *args, **kwargs)
                raise RuntimeError("injected collect")
            return read(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", failed_read), test.assertRaisesRegex(RuntimeError, "injected"):
            runner.solve()
        _assert_recovered(test, runner, plan)

    def unfinished_packet(array, *args, **kwargs):
        result = read(array, *args, **kwargs)
        if array is plan.state:
            result["status"] = int(impl.ProductionStatus.RUNNING)
        return result

    with patch.object(wp.array, "numpy", unfinished_packet), test.assertRaisesRegex(RuntimeError, "terminal"):
        runner.solve()
    _assert_recovered(test, runner, plan)


def test_graph_submission_failures(test, device):
    """Drain ambiguous submissions, invalidate failed captures and preserve reentry guards."""
    runner = _runner(device, graph=True)
    runner.solve()
    plan = runner.workspace._masked_pcg_plan
    launch = wp.capture_launch
    for submitted in (False, True):

        def failed_launch(*args, submitted=submitted, **kwargs):
            if submitted:
                launch(*args, **kwargs)
            raise RuntimeError("injected replay")

        with patch.object(wp, "capture_launch", failed_launch), test.assertRaisesRegex(RuntimeError, "injected"):
            plan.replay()
        _assert_recovered(test, runner, plan)
    with patch.object(wp, "capture_launch", side_effect=RuntimeError("injected")), test.assertRaises(RuntimeError):
        plan.replay()
    plan.set_solve_parameters(regularization=runner.workspace._lambda)
    with wp.ScopedCapture(device=device) as capture:
        plan.enqueue_solve()
    wp.capture_launch(capture.graph)
    test.assertEqual(plan.collect_result().status, impl.ProductionStatus.SUCCESS)
    runner.solve()
    enqueue = plan.enqueue_solve

    def failed_capture():
        enqueue()
        raise RuntimeError("injected capture")

    with patch.object(plan, "enqueue_solve", failed_capture), test.assertRaisesRegex(RuntimeError, "injected"):
        plan.capture()
    test.assertIsNone(plan.graph)
    _assert_recovered(test, runner, plan)
    emit = plan._enqueue_solve

    def failed_external_capture(observer):
        emit(observer)
        raise RuntimeError("injected external capture")

    with (
        patch.object(plan, "_enqueue_solve", failed_external_capture),
        test.assertRaisesRegex(RuntimeError, "injected"),
    ):
        with wp.ScopedCapture(device=device):
            plan.enqueue_solve()
    _assert_recovered(test, runner, plan)
    plan.replay()
    with test.assertRaisesRegex(RuntimeError, "Collect"):
        runner.solve()
    plan.collect_result()
    with (
        patch.object(wp, "capture_launch", side_effect=RuntimeError("injected replay")),
        patch.object(wp, "synchronize_stream", side_effect=RuntimeError("injected drain")),
        test.assertRaisesRegex(RuntimeError, "injected"),
    ):
        runner.solve()
    test.assertTrue(plan._in_flight)
    test.assertIsNotNone(plan.graph)
    with test.assertRaises(RuntimeError):
        plan.collect_result()
    with test.assertRaisesRegex(RuntimeError, "Collect"):
        runner.solve()


class TestPcgPlanLifecycle(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (test_sparse_content_reuse, test_enqueue_and_collect_failures):
        add_function_test(TestPcgPlanLifecycle, function.__name__, function, devices=[device])
    if device.is_cuda:
        add_function_test(
            TestPcgPlanLifecycle, "test_graph_submission_failures", test_graph_submission_failures, devices=[device]
        )


if __name__ == "__main__":
    unittest.main()
