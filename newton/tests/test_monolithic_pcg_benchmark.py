# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate independent PCG accuracy metrics and fair native-backend comparisons."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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

    def test_no_diagnostic_fallback(self):
        """Run production directly while the diagnostic solve is forbidden."""
        case = benchmark.synthetic_cases()[0]
        runner = benchmark.Runner(case, "production", "cpu", benchmark.pcg_config(maximum_iterations=30))
        with patch.object(
            benchmark.linear.MonolithicLinearWorkspace, "_solve_pcg_diagnostic", side_effect=AssertionError
        ):
            result = runner.solve()
        self.assertEqual(result.pcg_policy, "masked_smart_restart_v2")


add_function_test(
    TestMonolithicPcgBenchmark, "test_native_comparison", test_native_comparison, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main()
