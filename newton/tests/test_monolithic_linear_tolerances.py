# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check explicit absolute PCG tolerance across native execution modes."""

import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import pcg, pcg_production_kernels
from newton._src.solvers.monolithic.pcg_production import MaskedPcgConfig
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.monolithic_test_utils import build_contact_free_coupled_fixture, build_tiny_cpu_fixture
from newton.tests.test_monolithic_linear_production import make_plan
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import benchmark_pcg as benchmark


def test_absolute_acceptance(test, device):
    """Accept small RHS by absolute norm, retain relative ratios and allow opting out."""
    case = benchmark.synthetic_cases()[0]
    for mode in ("diagnostic", "production", "production_graph") if device.is_cuda else ("diagnostic", "production"):
        for atol, magnitude, zero_iterations in ((1e-6, 5e-7, True), (1e-6, 2e-6, False), (0.0, 5e-7, False)):
            with test.subTest(mode=mode, atol=atol, magnitude=magnitude):
                case.arrays["rhs_hat"] = np.zeros(case.n, dtype=np.float32)
                case.arrays["rhs_hat"][0] = magnitude
                config = replace(benchmark.pcg_config(30), linear_absolute_tolerance=atol)
                runner = benchmark.Runner(case, mode, device, config)
                result = runner.solve()
                test.assertEqual(result.status.name, "SUCCESS")
                test.assertEqual(result.iterations == 0, zero_iterations)
                accuracy = benchmark.Reference(case, config).measure(runner.y.numpy())
                test.assertTrue(accuracy["independent_residual_pass"])
                if zero_iterations:
                    test.assertGreater(max(result.rho, result.rho_q, result.rho_x), config.linear_tolerance)
                    np.testing.assert_array_equal(runner.y.numpy(), 0)
                    if mode == "production_graph":
                        graph = runner.workspace._masked_pcg_plan.graph
                        runner.config = replace(config, linear_absolute_tolerance=0.0)
                        second = runner.solve()
                        test.assertGreater(second.iterations, 0)
                        test.assertTrue(second.graph_rebuilt)
                        test.assertIsNot(runner.workspace._masked_pcg_plan.graph, graph)


def test_block_acceptance(test, device):
    """Require every block to pass either its relative or absolute bound in both controllers."""
    plan = make_plan(device, np.eye(4), [1, 0, 0, 0])
    packet = wp.zeros(1, dtype=pcg._MonolithicPcgPacket, device=device)
    ratios = wp.zeros(3, dtype=float, device=device)
    unbounded = (float("inf"),) * 3
    for norms, rhs, limits, expected in (
        ((5e-7, 3e-7, 4e-7), (5e-7, 3e-7, 4e-7), unbounded, True),
        ((1e-5, 0, 1e-5), (1, 0, 1), unbounded, True),
        ((2e-6, 2e-6, 0), (10, 2e-6, 10), unbounded, False),
        ((2e-6, 0, 2e-6), (10, 10, 2e-6), unbounded, False),
        ((5e-7, 3e-7, 4e-7), (5e-7, 3e-7, 4e-7), (1e-7, 1e-6, 1e-6), False),
        ((5e-7, 3e-7, 4e-7), (5e-7, 3e-7, 4e-7), (1e-6, 1e-7, 1e-6), False),
        ((5e-7, 3e-7, 4e-7), (5e-7, 3e-7, 4e-7), (1e-6, 1e-6, 1e-7), False),
        ((1e-5, 0, 1e-5), (1, 0, 1), (1e-6, 1e-6, 1e-6), False),
    ):
        with test.subTest(norms=norms, rhs=rhs, limits=limits):
            plan.residual_norm_limits.assign(np.asarray(limits, dtype=np.float32))
            plan._launch(pcg_production_kernels._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11])
            plan.norms.assign(np.array([*norms, 0, 0, 0, *rhs], dtype=np.float32))
            plan._launch(
                pcg_production_kernels._checkpoint_decision,
                1,
                [plan.norms, plan.state, plan.error, plan.params, 0, True, False],
            )
            test.assertEqual(int(plan.state.numpy()[0]["status"]) == 0, expected)
            diagnostic_norms = wp.array([*norms, *rhs], dtype=float, device=device)
            wp.launch(
                pcg._pack_monolithic_pcg_residual,
                1,
                [
                    plan.error,
                    ratios,
                    diagnostic_norms,
                    plan.params.floors,
                    1e-4,
                    1e-6,
                    plan.residual_norm_limits,
                    plan.compat_groups,
                    packet,
                ],
                device=device,
            )
            test.assertEqual(bool(packet.numpy()[0]["check"]), expected)


def test_solver_option(test, device):
    """Pass the public provisional default and explicit overrides into actual linear solves."""
    for atol in (1e-6, 0.0, 2e-6):
        fixture, solver = build_contact_free_coupled_fixture(
            device, gravity=(0, 0, -9.81), linear_absolute_tolerance=atol
        )
        test.assertEqual(solver.linear_absolute_tolerance, atol)
        original = solver._linear.solve_pcg
        values = []

        def solve(*args, original=original, values=values, solver=solver, **kwargs):
            config = kwargs["config"]
            values.append(config.linear_absolute_tolerance)
            limits = config.residual_norm_limits
            test.assertIsNotNone(limits)
            # Each block stops at the global forcing term or at half its smallest scaled row tolerance.
            scaled = solver._linear.scale.numpy().astype(float) * solver._criterion.tau.numpy().astype(float)
            forcing = solver.linear_tolerance * np.linalg.norm(solver._rhs.numpy().astype(float))
            for i, part in enumerate((slice(None), slice(0, 2), slice(2, None))):  # TinyFixture global/q/x rows.
                np.testing.assert_allclose(limits[i], max(0.5 * scaled[part].min(), forcing), rtol=1e-6, atol=0)
            return original(*args, **kwargs)

        with patch.object(solver._linear, "solve_pcg", solve):
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
        test.assertTrue(values)
        test.assertEqual(set(values), {atol})


def test_unresolved_q_small_rhs(test, device):
    """Advance a small joint load that passes PCG atol but fails outer q acceptance."""
    for mode in ("diagnostic", "production"):
        fixture, solver = build_contact_free_coupled_fixture(
            device,
            pcg_mode=mode,
            use_cuda_graph=device.is_cuda and mode == "production",
        )
        fixture.state.joint_q.zero_()
        solver.reset_state(fixture.state, dt=0.001)
        fixture.control.joint_f.assign([0, 5e-4])
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
        test.assertGreater(solver.last_stats.linear_iterations, 0)
        test.assertLessEqual(solver.last_stats.convergence_ratio_q, 1)
        # Independent backward Euler response: h^2 * force / 0.5 kg.
        np.testing.assert_allclose(fixture.state_next.joint_q.numpy(), [0, 1e-9], atol=1e-12, rtol=1e-4)


def test_limits_reuse_graph(test, device):
    """Apply changing outer caps on the same buffers, including empty q/x blocks and reset."""
    for case in (benchmark.synthetic_cases()[i] for i in (0, 3, 4)):
        if case.nq == 0:
            # Use identity for the empty-q lifecycle check, avoiding the separate
            # ill-conditioned benchmark's stagnation/iteration-budget behavior.
            case.arrays["K_values"][:] = np.eye(case.n, dtype=np.float32).reshape(-1)
            case.arrays["ax_internal_values"][:] = (
                np.eye(case.n, dtype=np.float32)
                .reshape(case.n // 3, 3, case.n // 3, 3)
                .transpose(0, 2, 1, 3)
                .reshape(-1, 3, 3)
            )
            for name in ("S", "D", "dynamic_diagonal"):
                case.arrays[name][:] = 1
        case.arrays["rhs_hat"] = np.zeros(case.n, dtype=np.float32)
        case.arrays["rhs_hat"][0] = 5e-7
        for mode in (
            ("diagnostic", "production", "production_graph") if device.is_cuda else ("diagnostic", "production")
        ):
            config = benchmark.pcg_config(30)
            runner = benchmark.Runner(case, mode, device, config)
            graph = None
            for limits in (None, (1e-7, 1e-7, 1e-7), (2e-7, 2e-7, 2e-7), None):
                runner.config = replace(config, residual_norm_limits=limits)
                result = runner.solve()
                test.assertEqual(result.status.name, "SUCCESS")
                test.assertEqual(result.iterations == 0, limits is None)
                reference = benchmark.Reference(case, runner.config)
                test.assertTrue(reference.measure(runner.y.numpy())["independent_residual_pass"])
                if limits is not None:
                    test.assertFalse(reference.measure(np.zeros(case.n))["independent_residual_pass"])
                if mode == "production_graph":
                    if graph is None:
                        graph = runner.workspace._masked_pcg_plan.graph
                    else:
                        test.assertFalse(result.graph_rebuilt)
                        test.assertIs(runner.workspace._masked_pcg_plan.graph, graph)


class TestMonolithicPcgTolerances(unittest.TestCase):
    def test_provisional_default(self):
        """Expose the same provisional default at the public solver and PCG boundaries."""
        self.assertEqual(inspect.signature(SolverMonolithic).parameters["linear_absolute_tolerance"].default, 1e-6)
        self.assertEqual(benchmark.pcg_config().linear_absolute_tolerance, 1e-6)
        self.assertEqual(MaskedPcgConfig().linear_absolute_tolerance, 1e-6)

    def test_invalid_absolute_tolerances(self):
        """Reject malformed, negative and unrepresentable tolerances."""
        model = build_tiny_cpu_fixture(device="cpu").model
        for value in (-1.0, float("nan"), float("inf"), True, "1e-6", None, 1e40, 1e-100):
            for config in (benchmark.pcg_config(), MaskedPcgConfig()):
                with self.subTest(value=value, config=type(config).__name__), self.assertRaises(ValueError):
                    replace(config, linear_absolute_tolerance=value)
            if value in (1e40, 1e-100):
                # P3 keeps host-finite values at construction; the FP32 PCG
                # configuration above rejects values it cannot represent.
                solver = SolverMonolithic(
                    model,
                    friction_velocity_scale=0.01,
                    mass_mode="lumped",
                    linear_absolute_tolerance=value,
                )
                self.assertEqual(solver.linear_absolute_tolerance, value)
            else:
                # The option configuration rejects every malformed value before allocation.
                with self.subTest(public=value), self.assertRaises(ValueError):
                    SolverMonolithic(
                        model,
                        friction_velocity_scale=0.01,
                        mass_mode="lumped",
                        linear_absolute_tolerance=value,
                    )

    def test_invalid_outer_limits(self):
        """Reject malformed or unrepresentable per-block residual caps."""
        for limits in (
            (1e-7,),
            [1e-7] * 3,
            (True, 1, 1),
            (0, 1, 1),
            (-1, 1, 1),
            (float("nan"), 1, 1),
            (float("inf"), 1, 1),
            (1e40, 1, 1),
            (1e-100, 1, 1),
        ):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "residual_norm_limits"):
                replace(benchmark.pcg_config(), residual_norm_limits=limits)


for function in (
    test_absolute_acceptance,
    test_block_acceptance,
    test_solver_option,
    test_unresolved_q_small_rhs,
    test_limits_reuse_graph,
):
    add_function_test(TestMonolithicPcgTolerances, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
