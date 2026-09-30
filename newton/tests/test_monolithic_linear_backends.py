# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check external PCG adapters preserve the shared linear problem."""

import unittest

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference.pcg_backends import ExternalRunner


def test_external_solvers(test, device):
    """Verify zero-copy padding and graph reuse with changed RHS values."""
    case = benchmark.synthetic_cases()[0]  # 32 scalars: exercises vec3 padding.
    config = benchmark.pcg_config(maximum_iterations=30)
    reference = benchmark.Reference(case, config)
    for mode in ("style3d", "kamino"):
        for graph in (False, True) if device.is_cuda else (False,):
            with test.subTest(mode=mode, graph=graph):
                runner = ExternalRunner(case, mode + ("_graph" if graph else ""), device, config)
                first = runner.solve()
                accuracy = reference.measure(runner.y.numpy())
                test.assertTrue(accuracy["independent_residual_pass"], accuracy)
                test.assertLess(accuracy["y_relative_error"], 1e-3)
                test.assertEqual(first["graph_rebuilt"], graph)
                runner.refresh_generation()
                runner.factor()
                runner.rhs.assign(case.arrays["rhs_hat"] * 2)
                second = runner.solve()
                test.assertFalse(second["graph_rebuilt"])
                np.testing.assert_allclose(runner.y.numpy(), reference.y * 2, rtol=1e-3, atol=1e-4)
                if mode == "style3d":
                    padding = runner.solution.numpy().reshape(-1)[case.n :]
                    np.testing.assert_array_equal(padding, 0)


class TestMonolithicPcgBackends(unittest.TestCase):
    def test_shared_benchmark(self):
        """Exercise all adapters through the existing measurements and observers."""
        case = benchmark.synthetic_cases()[0]
        result = benchmark.compare_case(
            case,
            "cpu",
            repeats=1,
            warmups=1,
            config=benchmark.pcg_config(maximum_iterations=30),
            include_external=True,
        )
        self.assertEqual(len(set(result["bound_input_sha256"].values())), 1)
        for sample in result["samples"]:
            self.assertTrue(sample["accuracy"]["independent_residual_pass"])
        for mode in ("style3d", "kamino"):
            curve = result["convergence"][mode]
            self.assertTrue(curve["points"][-1]["accuracy"]["independent_residual_pass"])
            self.assertEqual(len(curve["points"]), len(result["vectors"][curve["iterate_key"]]))
        self.assertEqual(result["software_counts"]["style3d"]["style3d_inner_calls"], 60)


add_function_test(TestMonolithicPcgBackends, "test_external_solvers", test_external_solvers, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
