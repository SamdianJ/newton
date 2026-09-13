# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check explicit absolute PCG tolerance across native execution modes."""

import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import linear, pcg_production
from newton._src.solvers.monolithic.pcg_production import MaskedPcgConfig
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests import test_solver_monolithic_step as stepping
from newton.tests.test_monolithic_pcg_production import make_plan
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
    packet = wp.zeros(1, dtype=linear._MonolithicPcgPacket, device=device)
    ratios = wp.zeros(3, dtype=float, device=device)
    for norms, rhs, expected in (
        ((5e-7, 3e-7, 4e-7), (5e-7, 3e-7, 4e-7), True),
        ((1e-5, 0, 1e-5), (1, 0, 1), True),
        ((2e-6, 2e-6, 0), (10, 2e-6, 10), False),
        ((2e-6, 0, 2e-6), (10, 10, 2e-6), False),
    ):
        with test.subTest(norms=norms, rhs=rhs):
            plan._launch(pcg_production._initialize_state, 1, [plan.state, plan.error, plan.scalars, 11])
            plan.norms.assign(np.array([*norms, 0, 0, 0, *rhs], dtype=np.float32))
            plan._launch(
                pcg_production._checkpoint_decision,
                1,
                [plan.norms, plan.state, plan.error, plan.params, 0, True, False],
            )
            test.assertEqual(int(plan.state.numpy()[0]["status"]) == 0, expected)
            diagnostic_norms = wp.array([*norms, *rhs], dtype=float, device=device)
            wp.launch(
                linear._pack_monolithic_pcg_residual,
                1,
                [plan.error, ratios, diagnostic_norms, plan.params.floors, 1e-4, 1e-6, packet],
                device=device,
            )
            test.assertEqual(bool(packet.numpy()[0]["check"]), expected)


def test_solver_option(test, device):
    """Pass the public provisional default and explicit overrides into actual linear solves."""
    for atol in (1e-6, 0.0, 2e-6):
        fixture, solver = stepping._free_scene(device, gravity=(0, 0, -9.81), linear_absolute_tolerance=atol)
        test.assertEqual(solver.linear_absolute_tolerance, atol)
        original = solver._linear.solve_pcg
        values = []

        def solve(*args, original=original, values=values, **kwargs):
            values.append(kwargs["config"].linear_absolute_tolerance)
            return original(*args, **kwargs)

        with patch.object(solver._linear, "solve_pcg", solve):
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
        test.assertTrue(values)
        test.assertEqual(set(values), {atol})


class TestMonolithicPcgTolerances(unittest.TestCase):
    def test_provisional_default(self):
        """Expose the same provisional default at the public solver and PCG boundaries."""
        self.assertEqual(inspect.signature(SolverMonolithic).parameters["linear_absolute_tolerance"].default, 1e-6)
        self.assertEqual(benchmark.pcg_config().linear_absolute_tolerance, 1e-6)
        self.assertEqual(MaskedPcgConfig().linear_absolute_tolerance, 1e-6)

    def test_invalid_absolute_tolerances(self):
        """Reject malformed, negative and unrepresentable tolerances."""
        for value in (-1.0, float("nan"), float("inf"), True, "1e-6", None, 1e40, 1e-100):
            for config in (benchmark.pcg_config(), MaskedPcgConfig()):
                with self.subTest(value=value, config=type(config).__name__), self.assertRaises(ValueError):
                    replace(config, linear_absolute_tolerance=value)
            with self.subTest(public=value), self.assertRaisesRegex(ValueError, "linear_absolute_tolerance"):
                SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, linear_absolute_tolerance=value)


for function in (test_absolute_acceptance, test_block_acceptance, test_solver_option):
    add_function_test(TestMonolithicPcgTolerances, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
