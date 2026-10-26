# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate independent PCG accuracy metrics and fair native-backend comparisons."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import benchmark_pcg as benchmark


def test_native_comparison(test, device):
    """Compare actual backends, observe native updates, and retain generation rebuild costs."""
    case = benchmark.synthetic_cases()[0]
    config = benchmark.pcg_config(maximum_iterations=30)
    result = benchmark.compare_case(case, device, repeats=2, warmups=1, config=config)
    reference = benchmark.Reference(case, config)
    test.assertLess(result["reference"]["relative_residual"], 1e-10)
    test.assertEqual(len(set(result["bound_input_sha256"].values())), 1)
    test.assertEqual(len(result["samples"]), 4 * len(result["modes"]))
    for sample in result["samples"]:
        test.assertEqual(reference.measure(result["vectors"][sample["solution_key"]]), sample["accuracy"])
        test.assertEqual(sample["status"], "SUCCESS")
        test.assertTrue(sample["accuracy"]["independent_residual_pass"])
        test.assertLess(sample["accuracy"]["y_relative_error"], 1e-3)
        test.assertGreaterEqual(sample["total_ms"], sample["solve_ms"])
        if sample["mode"] == "production_graph":
            test.assertFalse(sample["graph_rebuilt"])
    for mode, curve in result["convergence"].items():
        test.assertEqual(len(result["vectors"][curve["iterate_key"]]), len(curve["points"]))
        test.assertEqual(
            reference.measure(result["vectors"][curve["iterate_key"]][-1]), curve["points"][-1]["accuracy"]
        )
        test.assertGreater(len(curve["points"]), 1, mode)
        test.assertEqual(curve["points"][0]["iterations"], 0)
        test.assertAlmostEqual(curve["points"][0]["accuracy"]["y_relative_error"], 1)
        test.assertTrue(curve["points"][-1]["accuracy"]["independent_residual_pass"])
        test.assertEqual(curve["points"][-1]["iterations"], curve["result"]["iterations"])
        plain = next(sample for sample in result["samples"] if sample["mode"] == mode)
        for field in ("status", "iterations", "true_residual_checks", "residual_replacements"):
            test.assertEqual(curve["result"][field], plain[field])


class TestMonolithicPcgBenchmark(unittest.TestCase):
    def test_binding_uses_active_sparse_entries(self):
        """Ignore unused BSR capacity while rejecting differences in the actual operator."""
        case = benchmark.synthetic_cases()[0]
        original_init = benchmark.Runner.__init__
        for change_active in (False, True):
            calls = []

            def initialize(runner, *args, calls=calls, change_active=change_active, **kwargs):
                original_init(runner, *args, **kwargs)
                matrix = runner.workspace.k_global_scalar_bsr
                nnz = int(matrix.offsets.numpy()[-1])
                calls.append(runner.mode)
                values = np.append(matrix.values.numpy()[:nnz], np.float32(len(calls)))
                if change_active:
                    values[0] += len(calls)
                matrix.values = wp.array(values, dtype=float, device="cpu")
                matrix.columns = wp.array(np.append(matrix.columns.numpy()[:nnz], 0), dtype=int, device="cpu")

            with self.subTest(change_active=change_active), patch.object(benchmark.Runner, "__init__", initialize):
                if change_active:
                    with self.assertRaisesRegex(ValueError, "identical matrix"):
                        benchmark.compare_case(case, "cpu", repeats=1, warmups=1, config=benchmark.pcg_config(30))
                else:
                    result = benchmark.compare_case(case, "cpu", repeats=1, warmups=1, config=benchmark.pcg_config(30))
                    self.assertEqual(len(set(result["bound_input_sha256"].values())), 1)

    def test_reference_and_block_metrics(self):
        """Measure error against the quantized input's direct solution, including a hidden q error."""
        case = benchmark.synthetic_cases()[0]
        reference = benchmark.Reference(case, benchmark.pcg_config())
        scale = case.arrays["S"].astype(np.float64)
        dense = scale[:, None] * case.matrix().toarray().astype(np.float64) * scale[None, :]
        np.testing.assert_allclose(reference.y, np.linalg.solve(dense, case.arrays["rhs_hat"]), rtol=1e-12, atol=1e-12)
        exact = reference.measure(reference.y)
        self.assertLess(exact["y_relative_error"], 1e-12)
        self.assertLess(exact["max_true_ratio"], 1e-10)
        wrong = reference.y.copy()
        wrong[0] += 0.1
        error = reference.measure(wrong)
        self.assertGreater(error["q_relative_error"], 0)
        self.assertFalse(error["independent_residual_pass"])
        self.assertEqual(error["x_relative_error"], 0)

    def test_zero_rhs_and_invalid_snapshot(self):
        """Define zero-solution metrics and reject a corrupted snapshot before timing."""
        case = next(case for case in benchmark.synthetic_cases() if case.name == "zero_rhs")
        reference = benchmark.Reference(case, benchmark.pcg_config())
        self.assertEqual(reference.measure(reference.y)["y_relative_error"], 0)
        self.assertEqual(reference.measure(reference.y)["max_true_ratio"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, rhs_hat=[1])
            with self.assertRaises(ValueError):
                benchmark.load_snapshot(path)

    def test_snapshot_roundtrip_and_tamper(self):
        """Restore the exact exported matrix and reject data that no longer matches its hash."""
        case = benchmark.synthetic_cases()[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            np.savez(path, **case.arrays)
            metadata = {**case.metadata, "input_sha256": benchmark.array_fingerprint(case.arrays)}
            path.with_suffix(".json").write_text(json.dumps(metadata))
            restored = benchmark.load_snapshot(path)
            np.testing.assert_array_equal(restored.matrix().toarray(), case.matrix().toarray())
            np.testing.assert_array_equal(restored.arrays["rhs_hat"], case.arrays["rhs_hat"])
            changed = {**case.arrays, "rhs_hat": case.arrays["rhs_hat"] + 1}
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                benchmark.load_snapshot(path)


add_function_test(
    TestMonolithicPcgBenchmark, "test_native_comparison", test_native_comparison, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main()
