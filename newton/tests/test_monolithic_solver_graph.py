# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check retained graph-cache and linear segments with complete public states."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import linear_types
from newton._src.solvers.monolithic.segment_graph import _SegmentGraphCache
from newton.tests import test_monolithic_linear_assembly as linear_tests
from newton.tests import test_monolithic_solver_coupled as coupled
from newton.tests.monolithic_test_utils import build_contact_free_coupled_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.profiling import capture_count


@wp.kernel
def _copy_scaled(source: wp.array[float], result: wp.array[float], factor: float):
    i = wp.tid()
    result[i] = factor * source[i]


@wp.kernel
def _increment(result: wp.array[int]):
    result[wp.tid()] += 1


def test_segment_additive_capture(test, device):
    """Execute additive work once on first capture, replay and buffer rebinding."""
    cache = _SegmentGraphCache(device, enabled=True)
    for _ in range(2):
        result = wp.zeros(4, dtype=int, device=device)

        def execute(result=result):
            wp.launch(_increment, 4, [result], device=device)

        for expected in range(1, 4):
            cache.run("increment", (), (result,), execute, warmup=False)
            np.testing.assert_array_equal(result.numpy(), expected)
    test.assertEqual(cache.captures["increment"], 2)
    cache.enabled = False
    cache.run("increment", (), (result,), execute, warmup=False)
    np.testing.assert_array_equal(result.numpy(), 4)


def test_segments_continuous(test, device):
    """Match optional linear segments to PCG-only graphs across loads and dt changes."""
    scenes = [build_contact_free_coupled_fixture(device, pcg_mode="production", use_cuda_graph=True) for _ in range(2)]
    cache = _SegmentGraphCache(device, enabled=True)
    scenes[0][1]._linear._segment_graphs = cache
    for index in range(16):
        dt = 0.001 if index < 8 else 0.002
        for fixture, solver in scenes:
            fixture.model.gravity.assign(np.array([[index * 0.1, 0, -9.81]], dtype=np.float32))
            solver.step(fixture.state, fixture.state, fixture.control, dt=dt)
            test.assertTrue(solver.last_stats.converged)
        first, second = (coupled.snapshot(fixture.state) for fixture, _ in scenes)
        test.assertEqual(first.keys(), second.keys())
        for name in first:
            np.testing.assert_allclose(first[name], second[name], rtol=2e-5, atol=2e-6, err_msg=name)
    test.assertEqual(set(cache.captures), {"assembly_initialize", "scaling", "factor"})
    test.assertGreater(cache.replays["assembly_initialize"], cache.captures["assembly_initialize"])
    test.assertEqual(cache.captures["assembly_initialize"], 1)
    fixed = scenes[0][1]._linear._fixed_bsr[0].graphs
    test.assertEqual(fixed.captures["fixed_scalar_bsr"], 1)
    test.assertGreater(fixed.replays["fixed_scalar_bsr"], 1)


def test_segment_warm_replay(test, device):
    """Read changed buffer values on warm replay and rebind changed scalar parameters."""
    cache = _SegmentGraphCache(device, enabled=True)
    source, result = wp.ones(4, device=device), wp.zeros(4, device=device)
    for factor in (2.0, 3.0):

        def execute(factor=factor):
            wp.launch(_copy_scaled, 4, [source, result, factor], device=device)

        cache.run("copy", (factor,), (source, result), execute)
        for value in (1.0, -2.0, 0.25):
            source.fill_(value)
            with patch.object(wp, "launch", side_effect=AssertionError("uncaptured launch")):
                cache.run("copy", (factor,), (source, result), execute)
            np.testing.assert_array_equal(result.numpy(), factor * value)
    test.assertEqual(cache.captures["copy"], 2)
    test.assertEqual(cache.replays["copy"], 8)


def test_objective_candidate_reuse(test, device):
    """Retain all candidate/tangent bindings without repeating graph capture."""
    model = coupled.scene(device)
    solver = coupled.solver(model, mode="production", graph=True)
    state = solver.create_state(dt=0.001)
    solver.step(state, state, model.control(), dt=0.001)
    candidates = (solver._original, solver._accepted, solver._trial)
    expected = [
        solver._evaluate(candidate, 0.001, build_tangent=tangent)
        for candidate in candidates
        for tangent in (False, True)
    ]
    captures = capture_count(solver)
    for _ in range(3):
        with patch.object(wp, "ScopedCapture", side_effect=AssertionError("warm objective recaptured")):
            actual = [
                solver._evaluate(candidate, 0.001, build_tangent=tangent)
                for candidate in candidates
                for tangent in (False, True)
            ]
        for got, wanted in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got[0], wanted[0], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(got[1], wanted[1], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(got[2], wanted[2], rtol=2e-5, atol=2e-6)
    test.assertEqual(capture_count(solver), captures)


def test_objective_contact_rebinding(test, device):
    """Rebind contact outputs without recapturing unrelated objective work."""
    model = coupled.scene(device)
    solver = coupled.solver(model, mode="production", graph=True)
    state = solver.create_state(dt=0.001)
    solver.step(state, state, model.control(), dt=0.001)
    for device_result in (False, True):
        energy, residual, absolute = solver._evaluate(solver._accepted, 0.001, device_result=device_result)
        expected = residual.numpy().copy() if device_result else residual.copy()
        owners = solver._graphs.captures["coupled_objective_owners"]
        results = solver._graphs.captures["coupled_objective_result"]
        solver._contact.pool.residual.result = wp.clone(solver._contact.pool.residual.result)
        got, residual, total = solver._evaluate(solver._accepted, 0.001, device_result=device_result)
        np.testing.assert_array_equal(residual.numpy() if device_result else residual, expected)
        test.assertEqual(got, energy)
        test.assertEqual(total, absolute)
        test.assertEqual(solver._graphs.captures["coupled_objective_owners"], owners)
        test.assertEqual(solver._graphs.captures["coupled_objective_result"], results + 1)


def test_binding_and_stream_changes(test, device):
    """Rebind replaced buffers and streams while bounding retained graph allocations."""
    cache = _SegmentGraphCache(device, enabled=True)
    result = wp.zeros(4, device=device)
    # Keep inputs alive to prevent allocator address reuse from obscuring bindings.
    sources = [wp.full(4, float(index), device=device) for index in range(7)]
    for source in sources:

        def execute(source=source):
            wp.launch(_copy_scaled, 4, [source, result, 2.0], device=device)

        cache.run("copy", (2.0,), (source, result), execute)
        np.testing.assert_array_equal(result.numpy(), 2 * source.numpy())
    test.assertEqual(cache.captures["copy"], 7)
    test.assertEqual(len(cache.entries["copy"]), 6)
    test.assertEqual(cache.evictions["copy"], 1)
    wp.synchronize_device(device)
    with wp.ScopedStream(wp.Stream(device)):
        cache.run("copy", (2.0,), (source, result), execute)
    wp.synchronize_device(device)
    test.assertEqual(cache.captures["copy"], 8)
    test.assertEqual(cache.evictions["copy"], 2)
    np.testing.assert_array_equal(result.numpy(), 2 * source.numpy())


def test_factor_scaling_and_failures(test, device):
    """Preserve scaling, matrix rebinding and pivot failures with linear segments."""
    fixture, solver = build_contact_free_coupled_fixture(
        device, gravity=(0, 0, -9.81), pcg_mode="production", use_cuda_graph=True
    )
    cache = _SegmentGraphCache(device, enabled=True)
    solver._linear._segment_graphs = cache
    solver.step(fixture.state, fixture.state_next, fixture.control, dt=0.001)
    test.assertTrue(solver.last_stats.converged)
    workspace, generation = solver._linear, solver._generation
    for lam in (0.0, 0.1, 2.0, 0.0):
        for epsilon in (1e-6, 1e-4):
            test.assertEqual(workspace.build_scaling(solver._diagonal, epsilon_d=epsilon, generation=generation), 0)
            test.assertEqual(workspace.set_regularization(lam, generation=generation), 0)
            test.assertEqual(workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0), 0)
            expected = [a.numpy().copy() for a in (workspace.scale, workspace._q_cholesky, workspace._x_cholesky)]
            cache.enabled = False
            workspace.build_scaling(solver._diagonal, epsilon_d=epsilon, generation=generation)
            workspace.set_regularization(lam, generation=generation)
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0)
            cache.enabled = True
            for a, b in zip((workspace.scale, workspace._q_cholesky, workspace._x_cholesky), expected, strict=True):
                np.testing.assert_allclose(a.numpy(), b, rtol=1e-6, atol=1e-7)
    before = cache.captures["factor"]
    workspace.k_global_scalar_bsr.values = wp.clone(workspace.k_global_scalar_bsr.values)
    test.assertEqual(workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0), 0)
    test.assertEqual(cache.captures["factor"], before + 1)
    linear_tests._replace_q_block(workspace, np.full((workspace.layout.q_dof_count,) * 2, -1e10, dtype=np.float32))
    test.assertEqual(
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
        linear_types.MonolithicLinearStatus.NON_POSITIVE_PIVOT,
    )
    test.assertFalse(workspace._factor_valid)


def test_graph_transactions(test, device):
    """Preserve complete contact histories through failed trials and publication under graphs."""
    original = coupled.solver

    def create(*args, **kwargs):
        kwargs.update(mode="production", graph=True)
        return original(*args, **kwargs)

    with patch.object(coupled, "solver", create):
        coupled.test_failure_gates(test, device)


def test_assembly_tokens(test, device):
    """Reuse initialization while rejecting stale descriptors and restoring fixed prefixes."""
    original = linear_tests._workspace
    caches = []

    def workspace(*args, **kwargs):
        result = original(*args, **kwargs)
        result._segment_graphs = _SegmentGraphCache(device, enabled=True)
        caches.append(result._segment_graphs)
        return result

    with patch.object(linear_tests, "_workspace", workspace):
        linear_tests.test_generation_freeze(test, device)
        linear_tests.test_fixed_prefix_contact_and_reset(test, device)
    for cache in caches:
        test.assertEqual(cache.captures["assembly_initialize"], 1)
        test.assertGreater(cache.replays["assembly_initialize"], 1)


class TestMonolithicSegmentGraphs(unittest.TestCase):
    def test_cpu_non_graph(self):
        """Keep the CPU public solver path free of graph capture."""
        fixture, solver = build_contact_free_coupled_fixture("cpu", gravity=(0, 0, -9.81), pcg_mode="production")
        with patch.object(wp, "capture_launch", side_effect=AssertionError("CPU graph")):
            solver.step(fixture.state, fixture.state_next, fixture.control, dt=0.001)
        self.assertTrue(solver.last_stats.converged)
        self.assertIsNone(solver._linear._segment_graphs)


for function in (
    test_segment_additive_capture,
    test_segments_continuous,
    test_segment_warm_replay,
    test_objective_contact_rebinding,
    test_objective_candidate_reuse,
    test_binding_and_stream_changes,
    test_factor_scaling_and_failures,
    test_graph_transactions,
    test_assembly_tokens,
):
    add_function_test(
        TestMonolithicSegmentGraphs,
        function.__name__,
        function,
        devices=[device for device in get_test_devices() if device.is_cuda],
    )

if __name__ == "__main__":
    unittest.main()
