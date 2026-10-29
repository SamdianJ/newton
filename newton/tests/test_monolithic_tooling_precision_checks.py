# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the two-sided precision measurements on data with known answers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton._src.solvers.monolithic import iteration
from newton.tests.test_monolithic_cable_benchmark import imported_reference_modules
from scripts.monolithic_reference import run_precision_checks as runner
from scripts.monolithic_reference.cable_benchmarks import CableBenchmark
from scripts.monolithic_reference.precision_checks import SolverTrace, load_margin, relative_error, split_force_error

ROOT = Path(__file__).resolve().parents[2]
MATRIX = np.random.default_rng(3).standard_normal((6, 6))
OFFSET = np.arange(6, dtype=float)  # exactly representable in FP32


def linear64(step):
    return MATRIX @ np.asarray(step, dtype=float) + OFFSET


class TestSplitForceError(unittest.TestCase):
    def test_constant_bias_is_not_noise(self):
        """Report a step-constant FP32 offset as bias, with no response error or noise."""
        bias = np.full(6, 1e-3)

        def linear32(step):
            return (MATRIX.astype(np.float32) @ np.asarray(step, dtype=np.float32) + OFFSET).astype(float) + bias

        step = np.random.default_rng(4).standard_normal(6).astype(np.float32)
        split = split_force_error(linear32, linear64, step)
        np.testing.assert_allclose(split["bias"], bias, rtol=1e-3)
        np.testing.assert_allclose(split["total"], bias, rtol=1e-3)
        self.assertLess(split["response"].max(), 1e-5)
        self.assertLess(split["noise"].max(), 1e-5)

    def test_increment_amplified_noise_is_measured(self):
        """Measure a one-ulp increment sensitivity that FP64 does not have."""
        gain = 1e3

        def noisy32(step):
            return linear64(step) + gain * np.asarray(step, dtype=float)

        step = np.full(6, 0.5, dtype=np.float32)
        split = split_force_error(noisy32, linear64, step, rows=np.arange(3))
        expected = gain * float(np.spacing(np.float32(0.5)))
        self.assertEqual(split["noise"].shape, (3,))
        np.testing.assert_allclose(split["noise"], expected, rtol=1e-6)
        self.assertEqual(split["bias"].max(), 0.0)


class TestMargins(unittest.TestCase):
    def test_load_margin_ignores_unloaded_rows(self):
        """Take the largest allowance-to-load ratio over loaded rows and refuse rows without load."""
        load = np.array([0.0, 2.0, -4.0, 0.0])
        allowance = np.array([9.0, 1.0, 1.0, 9.0])
        self.assertEqual(load_margin(load, allowance, np.arange(4)), 0.5)
        with self.assertRaises(ValueError):
            load_margin(load, allowance, np.array([0, 3]))

    def test_relative_error_of_zero_reference(self):
        """Return zero for two zeros and infinity for a nonzero value against a zero reference."""
        self.assertEqual(relative_error(0.0, 0.0), 0.0)
        self.assertEqual(relative_error(1.0, 0.0), float("inf"))
        self.assertAlmostEqual(relative_error(1.1, 1.0), 0.1)

    def test_judge_reports_both_sides(self):
        """Fail resolution on a zero-update step and coverage on an unconverged one."""
        load = np.array([0.0, 1e-6])
        rows = np.array([1])

        def record(updates, response, converged=True, allowance=1e-7):
            return {
                "converged": converged,
                "updates": updates,
                "response": response,
                "allowance": np.array([allowance]),
            }

        reference = [record(2, 1.0), record(2, 2.0)]
        good = runner._judge([record(1, 1.0), record(1, 2.0)], reference, load, rows, 0.02, 0.0, 4)
        self.assertTrue(good["PASS"])
        frozen = runner._judge([record(0, 0.0), record(0, 0.0)], reference, load, rows, 0.02, 0.0, 4)
        self.assertFalse(frozen["resolution"]["updates"])
        self.assertFalse(frozen["resolution"]["response"])
        self.assertTrue(frozen["coverage"]["converged"])
        swallowed = runner._judge([record(1, 1.0, allowance=2e-6)], reference[:1], load, rows, 0.02, 0.0, 4)
        self.assertFalse(swallowed["resolution"]["allowance_below_load"])
        balanced = [record(2, 1.0), record(0, 1.0)]
        steady = runner._judge([record(1, 1.0), record(0, 1.0)], balanced, load, rows, 0.02, 0.0, 4)
        self.assertTrue(steady["resolution"]["updates"])
        stalled = runner._judge([record(1, 1.0, converged=False)], reference, load, rows, 0.02, 0.0, 4)
        self.assertFalse(stalled["coverage"]["converged"])
        self.assertFalse(stalled["PASS"])


class TestSolverTrace(unittest.TestCase):
    def test_records_iterates_and_restores(self):
        """Trace a loaded FP64 step, then leave the iteration module untouched."""
        originals = (iteration.measure, iteration.solve_direction)
        case = CableBenchmark("cantilever", segments=4)
        rows = runner._particle_rows(case.solver, 4)
        force = np.zeros((5, 3), dtype=np.float32)
        force[4, 1] = 1e-6
        with SolverTrace(case.solver, rows=rows) as trace:
            with self.assertRaises(RuntimeError):
                SolverTrace(case.solver).step(lambda: None)

            def advance():
                case.state.particle_f.assign(force)
                case.solver.step(case.state, case.state, case.control, dt=case.dt)

            step = trace.step(advance)
        self.assertEqual((iteration.measure, iteration.solve_direction), originals)
        self.assertTrue(case.solver.last_stats.converged)
        self.assertGreaterEqual(step.updates, 1)
        self.assertEqual(len(step.iterates), step.updates + 1)
        self.assertTrue(step.attempts and step.attempts[-1]["accepted"])
        self.assertFalse(step.regularized)
        self.assertEqual(step.iterates[0]["residual"].shape, (3,))
        self.assertAlmostEqual(step.iterates[0]["residual"][1], -1e-6, delta=1e-12)

    def test_restores_after_failure(self):
        """Restore the iteration module when the traced step raises."""
        originals = (iteration.measure, iteration.solve_direction)
        case = CableBenchmark("cantilever", segments=4)
        with self.assertRaises(ZeroDivisionError):
            with SolverTrace(case.solver) as trace:
                trace.step(lambda: 1 / 0)
        self.assertEqual((iteration.measure, iteration.solve_direction), originals)


class TestRunner(unittest.TestCase):
    def test_cases_cover_every_family(self):
        """Pair every case family in the frozen recipe with exactly one runner family."""
        spec = json.loads(runner.CASES_PATH.read_text())
        self.assertEqual(set(runner.FAMILIES), {k for k in spec if isinstance(spec[k], dict)})

    def test_identity_hashes_every_imported_reference_module(self):
        """Hash every imported reference module and the case recipe in the run receipt."""
        hashes = runner._identity()["source_hashes"]
        modules = imported_reference_modules("scripts.monolithic_reference.run_precision_checks")
        self.assertLessEqual(modules, set(hashes))
        self.assertIn("scripts/monolithic_reference/precision_cases.json", hashes)

    def test_rejects_nonempty_output(self):
        """Refuse to write into an output directory that already holds results."""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "old").write_text("x")
            with self.assertRaises(SystemExit):
                runner.main(["--output", directory, "--family", "tet_noise"])


if __name__ == "__main__":
    unittest.main()
