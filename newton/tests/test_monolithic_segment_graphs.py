# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check stable device segments independently of the production PCG graph."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import articulation, linear
from newton._src.solvers.monolithic.segment_graph import _SegmentGraphCache
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests import test_monolithic_owned_mass_matrix as mass_tests
from newton.tests import test_solver_monolithic_friction as friction_tests
from newton.tests import test_solver_monolithic_linear as linear_tests
from newton.tests import test_solver_monolithic_step as stepping
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p2_measurement import P2StageProfile


def test_segments_continuous(test, device):
    """Match PCG-only graphs across changing loads, dt and candidate swaps."""
    scenes = [stepping._free_scene(device, pcg_mode="production", use_cuda_graph=True) for _ in range(2)]
    scenes[1][1]._segment_graphs.enabled = False
    for index in range(16):
        dt = 0.001 if index < 8 else 0.002
        for fixture, solver in scenes:
            fixture.model.gravity.assign(np.array([[index * 0.1, 0, -9.81]], dtype=np.float32))
            solver.step(fixture.state, fixture.state, fixture.control, None, dt)
            test.assertEqual(solver.last_stats.status.name, "SUCCESS")
        for name in stepping._STATE_ARRAYS:
            np.testing.assert_allclose(
                getattr(scenes[0][0].state, name).numpy(),
                getattr(scenes[1][0].state, name).numpy(),
                rtol=2e-5,
                atol=2e-6,
            )
    cache = scenes[0][1]._segment_graphs
    test.assertEqual(
        set(cache.captures), {"articulation", "tet_residual", "norms", "assembly_initialize", "scaling", "factor"}
    )
    test.assertGreater(cache.replays["articulation"], cache.captures["articulation"])
    test.assertEqual(cache.captures["assembly_initialize"], 1)


def test_segment_warm_replay(test, device):
    """Refresh passive dynamics with new input values without dispatching Python kernels."""
    fixture, solver = stepping._free_scene(device, pcg_mode="production", use_cuda_graph=True)
    candidate = solver._transaction.accepted.state
    candidate.assign(fixture.state)
    articulation.eval_articulation_passive_candidate(fixture.model, candidate, solver._articulation)
    for angle in (0.1, -0.2, 0.3):
        candidate.joint_q.fill_(angle)
        with patch.object(wp, "launch", side_effect=AssertionError("uncaptured launch")):
            articulation.eval_articulation_passive_candidate(fixture.model, candidate, solver._articulation)
        actual = candidate.body_q.numpy().copy()
        solver._segment_graphs.enabled = False
        articulation.eval_articulation_passive_candidate(fixture.model, candidate, solver._articulation)
        solver._segment_graphs.enabled = True
        np.testing.assert_allclose(candidate.body_q.numpy(), actual, rtol=1e-6, atol=1e-7)
    test.assertEqual(solver._segment_graphs.captures["articulation"], 1)


def test_binding_and_stream_changes(test, device):
    """Rebind replaced arrays and streams, reject stale models and bound retained graphs."""
    fixture, solver = stepping._free_scene(device, pcg_mode="production", use_cuda_graph=True)
    a = solver._articulation
    state = fixture.state
    original_q = state.joint_q
    for index in range(7):
        state.joint_q = wp.clone(original_q)
        state.joint_q.fill_(0.02 * index)
        articulation.eval_articulation_passive_candidate(fixture.model, state, a)
    cache = solver._segment_graphs
    test.assertEqual(cache.captures["articulation"], 7)
    test.assertEqual(len(cache.entries["articulation"]), 4)
    test.assertEqual(cache.evictions["articulation"], 3)
    wp.synchronize_device(device)
    a.stream = wp.Stream(device)
    articulation.eval_articulation_passive_candidate(fixture.model, state, a)
    test.assertEqual(cache.captures["articulation"], 8)
    wp.synchronize_device(device)
    with patch.object(fixture.model, "body_mass", wp.clone(fixture.model.body_mass)):
        with test.assertRaisesRegex(ValueError, "stale"):
            articulation.eval_articulation_passive_candidate(fixture.model, state, a)


def test_owned_graph_dynamics(test, device):
    """Recompute mass, gravity and Coriolis for changing branched articulation poses and rates."""
    model = mass_tests._branched_model(device)
    for owned in (False, True):
        state = model.state()
        workspace = articulation.MonolithicArticulationWorkspace(model, use_optimized_articulation_mass_matrix=owned)
        cache = _SegmentGraphCache(device, enabled=True)
        workspace._segment_graphs = cache
        for i in range(6):
            state.joint_q.assign(np.linspace(-0.2, 0.1 * i, model.joint_dof_count, dtype=np.float32))
            state.joint_qd.assign(np.linspace(-0.1 * i, 0.3, model.joint_dof_count, dtype=np.float32))
            articulation.eval_articulation_passive_candidate(model, state, workspace)
            actual = [value.numpy().copy() for value in (workspace.M, workspace.g, workspace.C, workspace.scratch.J)]
            cache.enabled = False
            articulation.eval_articulation_passive_candidate(model, state, workspace)
            cache.enabled = True
            for result, expected in zip(
                (workspace.M, workspace.g, workspace.C, workspace.scratch.J), actual, strict=True
            ):
                np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-8)
        test.assertEqual(cache.captures["articulation"], 1)


def test_factor_scaling_and_failures(test, device):
    """Update lambda, scaling and matrix bindings without bypassing validity or pivot failures."""
    fixture, solver = stepping._free_scene(device, gravity=(0, 0, -9.81), pcg_mode="production", use_cuda_graph=True)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    workspace, generation = solver._linear, solver._generation
    cache = solver._segment_graphs
    for lam in (0.0, 0.1, 2.0, 0.0):
        for epsilon in (1e-6, 1e-4):
            test.assertEqual(
                workspace.build_scaling(solver._dynamic_diagonal, epsilon_d=epsilon, generation=generation), 0
            )
            test.assertEqual(workspace.set_regularization(lam, generation=generation), 0)
            test.assertEqual(workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0), 0)
            expected = [a.numpy().copy() for a in (workspace.scale, workspace._q_cholesky, workspace._x_cholesky)]
            cache.enabled = False
            workspace.build_scaling(solver._dynamic_diagonal, epsilon_d=epsilon, generation=generation)
            workspace.set_regularization(lam, generation=generation)
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0)
            cache.enabled = True
            for a, b in zip((workspace.scale, workspace._q_cholesky, workspace._x_cholesky), expected, strict=True):
                np.testing.assert_allclose(a.numpy(), b, rtol=1e-6, atol=1e-7)
    before = cache.captures["factor"]
    workspace.ax_internal_bsr3.values = wp.clone(workspace.ax_internal_bsr3.values)
    test.assertEqual(workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0), 0)
    test.assertEqual(cache.captures["factor"], before + 1)
    workspace.aq_actor_dense.fill_(-1e10)
    test.assertEqual(
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
        linear.MonolithicLinearStatus.NON_POSITIVE_PIVOT,
    )
    test.assertFalse(workspace._factor_valid)
    solver._y.fill_(float("nan"))
    with test.assertRaisesRegex(RuntimeError, "Nonfinite scaled reduction"):
        solver._rms(solver._y, scaled=False)


def test_graph_transactions(test, device):
    """Preserve contacting history, retry, rollback and failed final publication with segments enabled."""
    original = SolverMonolithic.__init__

    def initialize(self, *args, **kwargs):
        kwargs.update(pcg_mode="production", use_cuda_graph=True)
        original(self, *args, **kwargs)

    with patch.object(SolverMonolithic, "__init__", initialize):
        stepping.test_hard_failure_rollback(test, device)
        stepping.test_final_failure_in_place_republication(test, device)
        friction_tests.test_active_history_rollback(test, device)


def test_profile_cold_capture(test, device):
    """Profile cold segment capture without inserting synchronization into the graph."""
    fixture, solver = stepping._free_scene(device, gravity=(0, 0, -9.81), pcg_mode="production", use_cuda_graph=True)
    with P2StageProfile(solver, "actor_block") as profile:
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    report = profile.summary()
    test.assertEqual(solver.last_stats.status.name, "SUCCESS")
    test.assertGreater(report["preconditioner_setup_factor"]["inclusive_total_ms"], 0)
    test.assertEqual(report["preconditioner_q_setup"]["calls"], 0)


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


def test_trial_guard_updates(test, device):
    """Honor changed tet guards and invalid dt after warming a residual graph."""
    fixture, solver = stepping._free_scene(device, gravity=(0, 0, -9.81), pcg_mode="production", use_cuda_graph=True)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    candidate = solver._transaction.accepted
    solver._evaluate_trial(candidate, 0.001)
    before = solver._segment_graphs.captures["tet_residual"]
    solver._config = replace(solver._config, det_f_guard=2.0)
    with test.assertRaisesRegex(RuntimeError, "Tet determinant guard"):
        solver._evaluate_trial(candidate, 0.001)
    test.assertEqual(solver._segment_graphs.captures["tet_residual"], before + 1)
    with test.assertRaisesRegex(ValueError, "dt must"):
        solver._evaluate_trial(candidate, -0.001)


class TestMonolithicSegmentGraphs(unittest.TestCase):
    def test_cpu_non_graph(self):
        """Keep the CPU path free of graph capture."""
        fixture, solver = stepping._free_scene("cpu", gravity=(0, 0, -9.81), pcg_mode="production")
        with patch.object(wp, "capture_launch", side_effect=AssertionError("CPU graph")):
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        self.assertEqual(solver.last_stats.status.name, "SUCCESS")
        self.assertFalse(solver._segment_graphs.captures)


for function in (
    test_segments_continuous,
    test_segment_warm_replay,
    test_binding_and_stream_changes,
    test_owned_graph_dynamics,
    test_factor_scaling_and_failures,
    test_graph_transactions,
    test_profile_cold_capture,
    test_assembly_tokens,
    test_trial_guard_updates,
):
    add_function_test(
        TestMonolithicSegmentGraphs,
        function.__name__,
        function,
        devices=[device for device in get_test_devices() if device.is_cuda],
    )

if __name__ == "__main__":
    unittest.main()
