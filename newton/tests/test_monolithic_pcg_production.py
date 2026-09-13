# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise the fixed masked policy using real kernels and small dense systems."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import pcg_production as impl
from newton._src.solvers.monolithic.linear import (
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
    MonolithicPcgWarmStart,
)
from newton._src.solvers.monolithic.pcg_common import decide_restart
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests import test_monolithic_execution_options as execution
from newton.tests import test_solver_monolithic_linear as fixtures
from newton.tests import test_solver_monolithic_step as stepping
from newton.tests.monolithic_pcg_oracle import solve_fixed
from newton.tests.unittest_utils import add_function_test, get_test_devices


def make_plan(device, matrix, rhs, *, nq=1, config=None, scale=None, lower_q=None, lower_x=None, empty=False):
    matrix = np.asarray(matrix, dtype=np.float32)
    n = len(matrix)
    nx = (n - nq) // 3
    return impl.MaskedPcgPlan(
        offsets=wp.array(
            np.zeros(n + 1, dtype=np.int32) if empty else np.arange(n + 1, dtype=np.int32) * n, device=device
        ),
        columns=wp.array(np.tile(np.arange(n, dtype=np.int32), n), device=device),
        values=wp.array(matrix.reshape(-1), device=device),
        scale=wp.array(
            np.ones(n, dtype=np.float32) if scale is None else np.asarray(scale, dtype=np.float32), device=device
        ),
        q_cholesky=wp.array(
            np.eye(nq, dtype=np.float32) if lower_q is None else np.asarray(lower_q, dtype=np.float32), device=device
        ),
        x_cholesky=wp.array(
            np.tile(np.eye(3, dtype=np.float32), (nx, 1, 1))
            if lower_x is None
            else np.asarray(lower_x, dtype=np.float32),
            dtype=wp.mat33,
            device=device,
        ),
        rhs_hat=wp.array(np.asarray(rhs, dtype=np.float32), device=device),
        y=wp.zeros(n, dtype=float, device=device),
        config=config or impl.MaskedPcgConfig(maximum_iterations=11, true_residual_interval=5),
    )


def solve(plan, *, warm=False, regularization=0.0, graph=False):
    plan.set_solve_parameters(regularization=regularization, zero_initial_guess=not warm)
    if graph:
        if plan.graph is None:
            plan.capture()
        plan.replay()
    else:
        plan.enqueue_solve()
    return plan.collect_result()


def test_small_systems(test, device):
    """Solve coupled, q-only and x-only systems against a dense oracle."""
    for nq, nx in ((2, 0), (0, 1), (2, 2)):
        n = nq + 3 * nx
        rng = np.random.default_rng(42)
        v = rng.normal(size=(n, n)).astype(np.float32)
        a = v.T @ v + np.eye(n, dtype=np.float32) * 5
        rhs = np.linspace(-1, 2, n, dtype=np.float32)
        cfg = impl.MaskedPcgConfig(maximum_iterations=50, true_residual_interval=5)
        plan = make_plan(device, a, rhs, nq=nq, config=cfg)
        result = solve(plan)
        test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
        test.assertTrue(result.output_valid)
        test.assertLessEqual(max(result.ratios), cfg.linear_tolerance)
        np.testing.assert_allclose(plan.y.numpy(), np.linalg.solve(a.astype(float), rhs), rtol=3e-4, atol=2e-5)


def test_exact_wait_and_slots(test, device):
    """Freeze exact solutions until the fixed check and reset all replay counters."""
    for k, m in ((1, 10), (3, 10), (11, 5), (10, 5)):
        cfg = impl.MaskedPcgConfig(maximum_iterations=k, true_residual_interval=m)
        plan = make_plan(device, np.eye(4), [1, 2, 3, 4], config=cfg)
        result = solve(plan)
        test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
        test.assertEqual(result.iterations, 1)
        test.assertEqual(result.convergence_slot, min(k, m))
        test.assertEqual(result.true_residual_checks, 2)
        test.assertEqual(result.iteration_slots, k)
        test.assertEqual(result.residual_replacements, 0)
        before = plan.y.numpy().copy()
        result = solve(plan, warm=True)
        test.assertEqual(result.iterations, 0)
        test.assertEqual(result.convergence_slot, 0)
        test.assertEqual(result.true_residual_checks, 1)
        np.testing.assert_array_equal(plan.y.numpy(), before)


def test_curvature_and_reset(test, device):
    """Latch nonpositive curvature without extra checks and recover next solve."""
    for value in (0, -1):
        plan = make_plan(device, np.eye(4) * value, [1, 0, 0, 0])
        plan.delta.fill_(17)
        result = solve(plan)
        test.assertEqual(result.status, impl.ProductionStatus.NON_POSITIVE_CURVATURE)
        test.assertEqual(result.failure_slot, 1)
        test.assertEqual(result.true_residual_checks, 1)
        test.assertEqual(result.iterations, 0)
        test.assertFalse(result.output_valid)
        np.testing.assert_array_equal(plan.delta.numpy(), np.full(4, 17))
        plan.values.assign(np.eye(4, dtype=np.float32).reshape(-1))
        result = solve(plan)
        test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
        test.assertEqual(result.failure_slot, -1)


def test_large_finite_dot(test, device):
    """Avoid rejecting a finite required dot because an unused square overflows."""
    plan = make_plan(device, np.eye(4), [1e20, 0, 0, 0], lower_q=[[1e10]], lower_x=[np.eye(3) * 1e10])
    result = solve(plan, regularization=1e20)
    test.assertEqual(result.status, impl.ProductionStatus.SUCCESS)
    np.testing.assert_allclose(plan.y.numpy(), [1, 0, 0, 0], rtol=1e-5)


def test_nonfinite_max_and_recovery_output(test, device):
    """Preserve failure freshness and withhold nonfinite recovered directions."""
    for value in (np.nan, np.inf):
        plan = make_plan(device, np.eye(4), [value, 0, 0, 0])
        result = solve(plan)
        test.assertEqual(result.status, impl.ProductionStatus.NONFINITE_ITERATION)
        test.assertEqual(result.last_true_slot, -1)
        test.assertTrue(all(np.isnan(result.ratios)))
        test.assertFalse(result.output_valid)
    plan = make_plan(
        device,
        np.diag([1, 2, 3, 4]),
        [1, 2, 3, 4],
        config=impl.MaskedPcgConfig(maximum_iterations=1, true_residual_interval=10),
    )
    result = solve(plan)
    test.assertEqual(result.status, impl.ProductionStatus.MAX_ITERATIONS)
    test.assertEqual((result.iterations, result.last_true_slot, result.true_residual_checks), (1, 1, 2))
    test.assertTrue(result.residual_matches_y)
    plan = make_plan(device, np.zeros((4, 4)), [0, 0, 0, 0], scale=[1e30] * 4, empty=True)
    plan.y.fill_(1e20)
    plan.delta.fill_(27)
    result = solve(plan, warm=True)
    test.assertEqual(result.status, impl.ProductionStatus.NONFINITE_ITERATION)
    test.assertEqual(result.convergence_slot, 0)
    test.assertFalse(result.output_valid)
    np.testing.assert_array_equal(plan.delta.numpy(), [27] * 4)


def test_gap_recovery_and_block_gate(test, device):
    """Share per-block threshold boundaries and recover waiting states independently."""

    plan = make_plan(device, np.eye(4), [1, 0, 0, 0])
    decision = wp.zeros(1, dtype=int, device=device)
    for gap, expected in ((0.124, False), (0.125, True), (0.126, True)):
        wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
        plan.norms.assign(np.asarray([100, 1, 100, 0.01, gap, 0.01, 100, 1, 100], dtype=np.float32))
        plan.params.replacement_threshold = 0.125
        wp.launch(
            impl._checkpoint_decision,
            1,
            [plan.norms, plan.state, plan.error, plan.params, 5, False, False],
            device=device,
        )
        wp.launch(
            decide_restart,
            1,
            [plan.norms[3:6], plan.norms[:3], plan.params.floors, wp.float64(0.125), plan.error, decision],
            device=device,
        )
        test.assertEqual(bool(plan.state.numpy()[0]["restart"]), expected)
        test.assertEqual(bool(decision.numpy()[0]), expected)
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
    plan.products.zero_()
    wp.launch(
        impl._gamma_and_beta_guard,
        1,
        [plan.products, plan.scalars, plan.state, plan.error, plan.params, 1],
        device=device,
    )
    plan.norms.assign(np.asarray([1, 1, 0, 0, 0, 0, 1, 1, 0], dtype=np.float32))
    wp.launch(
        impl._checkpoint_decision, 1, [plan.norms, plan.state, plan.error, plan.params, 5, False, False], device=device
    )
    item = plan.state.numpy()[0]
    test.assertEqual(
        (item["enabled"], item["restart"], item["recovering"], item["residual_replacements"]), (1, 1, 1, 1)
    )
    # A fresh recovery that still underflows cannot be mistaken for acceptance.
    wp.launch(
        impl._gamma_and_beta_guard,
        1,
        [plan.products, plan.scalars, plan.state, plan.error, plan.params, 5],
        device=device,
    )
    test.assertEqual(plan.state.numpy()[0]["status"], 15)
    # Global passes but a small unresolved q block vetoes success.
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
    plan.norms.assign(np.asarray([1e-5, 1e-5, 0, 0, 0, 0, 1e5, 1e-5, 1e5], dtype=np.float32))
    wp.launch(
        impl._checkpoint_decision, 1, [plan.norms, plan.state, plan.error, plan.params, 11, False, True], device=device
    )
    test.assertEqual(plan.state.numpy()[0]["status"], 17)


def test_all_heavy_kernels_masked(test, device):
    """Leave terminal vectors, counters and buffers unchanged despite invalid inputs."""
    for matrix in (np.eye(4), -np.eye(4)):
        plan = make_plan(device, matrix, [1, 2, 3, 4])
        solve(plan)
        names = ("y", "r", "p", "z", "ap", "true_r", "gap", "products", "state", "error", "maxima", "sums", "norms")
        before = {name: getattr(plan, name).numpy().copy() for name in names}
        plan.values.fill_(np.nan)
        plan.q_cholesky.fill_(np.nan)
        plan.x_cholesky.fill_(wp.mat33(np.nan))
        plan._emit_matvec(plan.p, checkpoint=False)
        plan._emit_checkpoint(10)
        plan._emit_direction(10)
        wp.launch(
            impl._masked_update,
            4,
            [plan.y, plan.r, plan.p, plan.ap, plan.scalars, plan.state, plan.error],
            device=device,
        )
        for name in names:
            np.testing.assert_equal(getattr(plan, name).numpy(), before[name], err_msg=name)


def test_binding_and_adapter(test, device):
    """Validate bindings, factor generations, warm starts and exclusive dispatch."""

    original = fixtures.MonolithicLinearWorkspace
    with patch.object(
        fixtures, "MonolithicLinearWorkspace", side_effect=lambda *a, **kw: original(*a, **kw, pcg_mode="production")
    ):
        workspace, generation = fixtures._ready_system(device, heterogeneous=True)
    rhs = wp.array(np.linspace(-1, 2, 11, dtype=np.float32), device=device)
    y = wp.zeros(11, dtype=float, device=device)
    delta = wp.zeros_like(y)
    kwargs = {
        "generation": generation,
        "warm_start": MonolithicPcgWarmStart.ZERO,
        "config": fixtures._pcg_config(),
        "delta": delta,
    }
    with patch.object(MonolithicLinearWorkspace, "_solve_pcg_diagnostic", side_effect=AssertionError("old host solve")):
        result = workspace.solve_pcg(rhs, y, **kwargs)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        test.assertTrue(result.output_valid)
        np.testing.assert_allclose(delta.numpy(), workspace.scale.numpy() * y.numpy(), rtol=1e-6)
        test.assertFalse(hasattr(workspace, "_diagnostic_vector"))
        old_plan = workspace._masked_pcg_plan
        result = workspace.solve_pcg(rhs, y, **{**kwargs, "warm_start": MonolithicPcgWarmStart.SAME_GENERATION})
        test.assertEqual(result.iterations, 0)
        test.assertIs(workspace._masked_pcg_plan, old_plan)
        workspace.set_regularization(0.5, generation=generation)
        result = workspace.solve_pcg(rhs, y, **kwargs)
        test.assertEqual(result.status, MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0)
        result = workspace.solve_pcg(rhs, y, **kwargs)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        result = workspace.solve_pcg(rhs, rhs, **kwargs)
        test.assertEqual(result.status, MonolithicLinearStatus.INVALID_ARGUMENT)
        result = workspace.solve_pcg(rhs, y, **{**kwargs, "generation": replace(generation, assembly_sequence=99)})
        test.assertEqual(result.status, MonolithicLinearStatus.STALE_GENERATION)
    plan = make_plan(device, np.eye(4), [1, 2, 3, 4])
    plan.columns.assign(np.zeros(16, dtype=np.int32))
    with test.assertRaisesRegex(ValueError, "layout"):
        plan.set_solve_parameters(regularization=0)
    plan = make_plan(device, np.eye(4), [1, 2, 3, 4])
    plan.r = wp.zeros_like(plan.r)
    with test.assertRaisesRegex(ValueError, "storage"):
        plan.set_solve_parameters(regularization=0)


def test_operator_and_factor_parity(test, device):
    """Apply the masked operator and factors to the same assembled BSR as diagnostic."""

    workspace, _generation = fixtures._ready_system(device, heterogeneous=True)
    vector = wp.array(np.linspace(-1, 2, 11, dtype=np.float32), device=device)
    y = wp.zeros_like(vector)
    plan = impl.MaskedPcgPlan.from_workspace(workspace, vector, y, config=impl.MaskedPcgConfig())
    plan.set_solve_parameters(regularization=workspace._lambda)
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 200], device=device)
    plan._emit_matvec(vector, checkpoint=False)
    expected = wp.zeros_like(vector)
    workspace.operator.matvec(vector, expected, expected, 1.0, 0.0)
    np.testing.assert_allclose(plan.ap.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)
    wp.copy(plan.r, vector)
    plan._emit_direction(0)
    workspace.preconditioner.matvec(vector, expected, expected, 1.0, 0.0)
    np.testing.assert_allclose(plan.z.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)


def test_same_policy_oracle(test, device):
    """Compare state, checks, restarts and work against an independent host policy."""

    matrices = (np.eye(4), np.array([[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]]), -np.eye(4))
    for matrix in matrices:
        for k, interval, threshold in ((11, 5, 0.1), (1, 10, 0.1), (20, 2, 0.0)):
            cfg = impl.MaskedPcgConfig(
                maximum_iterations=k, true_residual_interval=interval, residual_replacement_threshold=threshold
            )
            plan = make_plan(device, matrix, [1, 2, 3, 4], config=cfg)
            result = solve(plan)
            expected = solve_fixed(matrix, [1, 2, 3, 4], np.eye(4, dtype=np.float32), 1, cfg)
            test.assertEqual(result.status.name, expected["status"])
            test.assertEqual(
                (
                    result.iterations,
                    result.true_residual_checks,
                    result.residual_replacements,
                    result.last_true_slot,
                    result.convergence_slot,
                ),
                (
                    expected["iterations"],
                    expected["checks"],
                    expected["replacements"],
                    expected["last"],
                    expected["convergence_slot"],
                ),
            )
            np.testing.assert_allclose(plan.y.numpy(), expected["y"], rtol=1e-4, atol=1e-6)


def test_solver_production_transactions(test, device):
    """Exercise actual new dispatch, q-only, retries, rollback and history publication."""

    options = [
        {"pcg_mode": "production", "use_optimized_articulation_mass_matrix": owned, "use_cuda_graph": graph}
        for owned in (False, True)
        for graph in ((False, True) if device.is_cuda else (False,))
    ]
    for choice in options:
        fixture, solver = stepping._free_scene(device, gravity=(0, 0, -9.81), **choice)
        with (
            patch.object(
                MonolithicLinearWorkspace, "_solve_pcg_diagnostic", side_effect=AssertionError("host fallback")
            ),
            patch.object(solver._linear, "recover_delta", side_effect=AssertionError("duplicate recovery")),
        ):
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        test.assertEqual(solver.last_stats.status, solver.Status.SUCCESS)
        test.assertGreater(solver.last_stats.linear_iterations, 0)
        test.assertGreaterEqual(solver.last_stats.linear_scheduled_slots, solver.last_stats.linear_iterations)
        test.assertTrue(solver.last_stats.linear_output_valid)
        test.assertEqual(solver.last_stats.pcg_policy, impl.PCG_POLICY)
        for name in ("min_p_ap", "min_r_z", "initial_guess_norm", "recursive_true_residual_gap"):
            test.assertTrue(np.isnan(getattr(solver.last_stats, name)))
        with patch.object(execution, "_options", return_value=iter([choice])):
            execution.test_q_only_policies(test, device)
    # The shared transaction assertions describe physical invariants, not old iteration parity.
    for graph in (False, True) if device.is_cuda else (False,):
        choice = {"pcg_mode": "production", "use_optimized_articulation_mass_matrix": False, "use_cuda_graph": graph}
        with patch.object(execution, "_options", return_value=iter([choice])):
            execution.test_shared_transactions(test, device)


def test_scalar_failures_and_first_slot(test, device):
    """Latch finite/positivity failures before dependent writes and retain their slot."""
    plan = make_plan(device, np.eye(4), [1, 0, 0, 0])
    for value, expected in ((-1, 15), (np.nan, 16), (np.inf, 16)):
        wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
        plan.products.fill_(value)
        wp.launch(
            impl._gamma_and_beta_guard,
            1,
            [plan.products, plan.scalars, plan.state, plan.error, plan.params, 3],
            device=device,
        )
        test.assertEqual(plan.state.numpy()[0]["status"], expected)
        wp.launch(impl._begin_slot, 1, [plan.state, plan.error, 9], device=device)
        test.assertEqual(plan.state.numpy()[0]["failure_slot"], 3)
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
    plan.products.fill_(1e-30)
    plan.scalars.assign(np.asarray([1e30, 0, 0], dtype=np.float32))
    wp.launch(
        impl._alpha_guard, 1, [plan.products, plan.scalars, plan.state, plan.error, plan.params, 2], device=device
    )
    test.assertEqual(plan.state.numpy()[0]["status"], 16)
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
    plan.products.fill_(1e30)
    plan.scalars.assign(np.asarray([1e-30, 0, 0], dtype=np.float32))
    wp.launch(
        impl._gamma_and_beta_guard,
        1,
        [plan.products, plan.scalars, plan.state, plan.error, plan.params, 4],
        device=device,
    )
    test.assertEqual(plan.state.numpy()[0]["status"], 16)
    wp.launch(impl._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11], device=device)
    plan.products.fill_(1e-20)
    plan.scalars.assign(np.asarray([1e-20, 0, 0], dtype=np.float32))
    wp.launch(
        impl._alpha_guard, 1, [plan.products, plan.scalars, plan.state, plan.error, plan.params, 2], device=device
    )
    test.assertEqual(plan.state.numpy()[0]["enabled"], 1)
    test.assertEqual(plan.scalars.numpy()[1], 1)


def test_binding_rejections(test, device):
    """Reject malformed configuration, capacity, aliased vectors and invalid graph options."""

    for name, values in (
        ("maximum_iterations", (0, -1, True, 1.5)),
        ("true_residual_interval", (0, -1, True)),
        ("residual_replacement_threshold", (-1, np.nan, np.inf)),
    ):
        for value in values:
            with test.assertRaises(ValueError):
                impl.MaskedPcgConfig(**{name: value})
            if name == "residual_replacement_threshold":
                with test.assertRaises(ValueError):
                    fixtures._pcg_config(**{name: value})
    plan = make_plan(device, np.eye(4), [1, 2, 3, 4])
    args = {
        name: getattr(plan, name)
        for name in ("offsets", "columns", "values", "scale", "q_cholesky", "x_cholesky", "y", "delta")
    }
    args["rhs_hat"] = plan.rhs
    for changes in (
        {"rhs_hat": plan.y},
        {"delta": plan.rhs},
        {"y": plan.values[:4]},
        {"offsets": wp.array([0, 4, 8, 12, 100], dtype=int, device=device)},
        {"columns": wp.full(16, 4, dtype=int, device=device)},
        {"scale": wp.ones(4, dtype=wp.float64, device=device)},
    ):
        with test.assertRaises(ValueError):
            impl.MaskedPcgPlan(**{**args, **changes})

    plan.config = replace(plan.config, maximum_iterations=3)
    with test.assertRaises(ValueError):
        plan.set_solve_parameters(regularization=0)

    for value in (0, 1, None, "true", np.bool_(True)):
        with test.assertRaises(ValueError):
            SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, use_cuda_graph=value)
    with test.assertRaises(ValueError):
        SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, use_cuda_graph=True)


class TestMonolithicPcgProduction(unittest.TestCase):
    pass


for fn in (
    test_small_systems,
    test_exact_wait_and_slots,
    test_curvature_and_reset,
    test_large_finite_dot,
    test_nonfinite_max_and_recovery_output,
    test_gap_recovery_and_block_gate,
    test_all_heavy_kernels_masked,
    test_binding_and_adapter,
    test_operator_and_factor_parity,
    test_same_policy_oracle,
    test_solver_production_transactions,
    test_scalar_failures_and_first_slot,
    test_binding_rejections,
):
    add_function_test(TestMonolithicPcgProduction, fn.__name__, fn, devices=get_test_devices())
