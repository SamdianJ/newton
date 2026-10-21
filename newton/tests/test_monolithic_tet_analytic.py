# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.solvers.experimental.monolithic import SolverMonolithic
from scripts.monolithic_reference.tet_analytic import (
    fixed_free_frequency,
    homogeneous_traction,
    implicit_euler_oscillation,
    oscillation_period_and_decay,
    timoshenko_tip_deflection,
    young_poisson,
)
from scripts.monolithic_reference.tet_benchmarks import TetBenchmark, expected_cases, qualify_matrix, summarize_matrix


class TestTetAnalytic(unittest.TestCase):
    def test_lame_conversion(self):
        self.assertEqual(young_poisson(1.0, 0.0), (2.0, 0.0))
        young, nu = young_poisson(10000 / 2.6, 10000 * 0.3 / (1.3 * 0.4))
        self.assertAlmostEqual(young, 10000.0, places=8)
        self.assertAlmostEqual(nu, 0.3, places=12)
        with self.assertRaises(ValueError):
            young_poisson(0.0, 1.0)

    def test_rest_is_stress_free_and_small_strain_is_linear(self):
        for smith in (False, True):
            np.testing.assert_allclose(homogeneous_traction(np.eye(3), 2.0, 3.0, smith=smith), 0.0, atol=1e-14)
            eps = 1e-6
            stress = homogeneous_traction(np.diag([1 + eps, 1, 1]), 1.0, 0.0, smith=smith)
            self.assertLess(abs(stress[0, 0] / (2 * eps) - 1), 1e-4)
            self.assertLess(abs(stress[1, 1]), 1e-10)
        with self.assertRaises(ValueError):
            homogeneous_traction(np.diag([1.0, 1.0, -1.0]), 1.0, 0.0, smith=True)

    def test_timoshenko_tip_deflection(self):
        value = timoshenko_tip_deflection(1.0, 1.0, 0.1, 0.1, 1e6, 0.0)
        self.assertAlmostEqual(value, 0.02 + 1.2e-4, delta=1e-12)

    def test_fixed_free_frequency(self):
        self.assertAlmostEqual(fixed_free_frequency(1.0, 1.0, 0.0, 2.0), math.pi / 2, places=14)
        with self.assertRaises(ValueError):
            fixed_free_frequency(1.0, 1.0, 1.0, 2.0)

    def test_implicit_euler_oscillation(self):
        period, ratio = implicit_euler_oscillation(1.0, 1.0)
        self.assertAlmostEqual(period, 8.0, places=12)
        self.assertAlmostEqual(ratio, 1 / 16, places=12)
        period, ratio = implicit_euler_oscillation(100.0, 1e-5)
        self.assertLess(abs(period / (2 * math.pi / 100) - 1), 1e-6)
        self.assertLess(abs(ratio / math.exp(-math.pi * 1e-3) - 1), 1e-3)

    def test_period_and_decay_measurement(self):
        dt, omega, decay = 1e-3, 2 * math.pi, 0.9
        t = np.arange(4000) * dt
        signal = decay ** (t / 1.0) * np.sin(omega * t)
        period, ratio = oscillation_period_and_decay(signal, dt)
        self.assertLess(abs(period - 1.0), 2e-3)
        self.assertLess(abs(ratio - decay), 2e-3)
        with self.assertRaises(ValueError):
            oscillation_period_and_decay(np.ones(10), dt)


class TestTetBenchmarkSmoke(unittest.TestCase):
    """Advance each closed-form tet experiment a few public solver steps on the CPU."""

    def test_nodal_mass_matches_body_mass(self):
        case = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped")
        expected = case.recipe["density"] * np.prod(case.size)
        self.assertAlmostEqual(case.nodal_mass.sum() / expected, 1.0, delta=1e-6)
        dynamic = ~case.fixed
        np.testing.assert_allclose(case.model.particle_mass.numpy()[dynamic], case.nodal_mass[dynamic], rtol=1e-6)

    def test_boundary_and_tip_nodes_survive_float32_storage(self):
        patch = TetBenchmark("patch", case="shear", cells=(2, 2, 2), material_model="kim_stable_no_log")
        self.assertEqual((int(patch.fixed.sum()), int(patch.tip.sum())), (26, 9))
        bar = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped")
        self.assertEqual((int(bar.fixed.sum()), int(bar.tip.sum())), (4, 4))

    def test_patch_returns_toward_homogeneous_state(self):
        case = TetBenchmark("patch", case="tension", cells=(2, 2, 2), material_model="kim_stable_no_log")
        before = case.interior_error()
        for _ in range(3):
            case.step()
        self.assertLess(case.interior_error(), 0.1 * before)

    def test_cantilever_bends_downward(self):
        case = TetBenchmark("cantilever", cells=(10, 1, 1))
        for _ in range(3):
            case.step()
        self.assertLess(case.history[-1][2], 0.0)

    def test_vibration_starts_outward(self):
        case = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped")
        for _ in range(5):
            case.step()
        self.assertGreater(case.history[-1][0], 0.0)

    def test_vibration_velocity_in_float32(self):
        velocities = []
        for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
            case = TetBenchmark("axial_vibration", cells=(10, 1, 1), mass_mode="lumped", precision=precision)
            for _ in range(5):
                case.step()
                # The inertial prediction leaves unbalanced elastic forces; a
                # converged step must have taken at least one Newton update.
                self.assertGreater(case.solver.last_stats.nonlinear_iterations, 0)
            velocities.append(case.state.particle_qd.numpy().astype(float))
        reference, single = velocities
        self.assertLess(np.abs(single - reference).max(), 0.01 * np.abs(reference).max())


class TestTetMatrix(unittest.TestCase):
    """Apply cross-case gates to synthetic single-case receipts."""

    def receipt(self, kind, cells, status="PASS", **metrics):
        return {"kind": kind, "cells": cells, "status": status, "metrics": metrics}

    def test_cantilever_convergence(self):
        rows = [
            self.receipt("cantilever", [10, 1, 1], deflection=0.6, reference=1.0),
            self.receipt("cantilever", [20, 2, 2], deflection=0.85, reference=1.0),
            self.receipt("cantilever", [40, 4, 4], deflection=0.95, reference=1.0),
        ]
        self.assertEqual(summarize_matrix("cantilever", rows), {"status": "PASS", "failures": []})
        rows[-1]["metrics"]["deflection"] = 0.86
        result = summarize_matrix("cantilever", rows)
        self.assertEqual(result["failures"], ["extrapolated_relative_error", "observed_order"])

    def test_vibration_mass_bias(self):
        rows = [
            self.receipt("axial_vibration", [40, 1, 1], mass_mode="consistent", period_error=-0.004),
            self.receipt("axial_vibration", [40, 1, 1], mass_mode="lumped", period_error=0.004),
            self.receipt("axial_vibration", [20, 1, 1], mass_mode="lumped", period_error=0.02),
        ]
        self.assertEqual(summarize_matrix("axial_vibration", rows)["status"], "PASS")
        rows[0]["metrics"]["period_error"] = 0.002
        self.assertEqual(summarize_matrix("axial_vibration", rows)["failures"], ["consistent_period_bias"])

    def test_any_failed_case_fails_the_matrix(self):
        aborted = [
            self.receipt("cantilever", [10, 1, 1], deflection=0.6, reference=1.0),
            {"kind": "cantilever", "cells": [40, 4, 4], "status": "FAIL"},
        ]
        self.assertEqual(summarize_matrix("cantilever", aborted), {"status": "FAIL", "failures": ["case_failed"]})
        rows = [self.receipt("patch", [4, 4, 4], status="FAIL")]
        self.assertEqual(summarize_matrix("patch", rows), {"status": "FAIL", "failures": ["case_failed"]})
        self.assertEqual(summarize_matrix("patch", []), {"status": "FAIL", "failures": ["empty_matrix"]})

    def test_expected_matrix(self):
        self.assertEqual([len(expected_cases(k)) for k in ("patch", "cantilever", "axial_vibration")], [6, 3, 6])
        self.assertIn(
            {"case": "shear", "cells": [4, 4, 4], "material_model": "kim_stable_no_log", "mass_mode": "consistent"},
            expected_cases("patch"),
        )

    def test_matrix_requires_every_case_from_one_source(self):
        rows = [
            {**spec, "kind": "patch", "status": "PASS", "metrics": {}, "sha256": {"a": "1"}}
            for spec in expected_cases("patch")
        ]
        self.assertEqual(qualify_matrix("patch", rows, recipe_status="FROZEN")["status"], "PASS")
        self.assertEqual(qualify_matrix("patch", rows, recipe_status="PROPOSED")["status"], "PARTIAL")
        self.assertEqual(qualify_matrix("patch", rows[1:], recipe_status="FROZEN")["status"], "INCOMPLETE")
        rows[0] = {**rows[0], "sha256": {"a": "2"}}
        self.assertEqual(qualify_matrix("patch", rows, recipe_status="FROZEN")["status"], "FAIL")


class TestTetRunner(unittest.TestCase):
    def test_truncated_case_is_partial_and_matrix_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, "-m", "scripts.monolithic_reference.run_tet_benchmark"]
            arguments = ["case", "patch", "--case", "shear", "--cells", "2", "2", "2", "--max-steps", "1"]
            subprocess.run(
                [*command, *arguments, "--output", str(root / "shear")],
                check=True,
                capture_output=True,
            )
            receipt = json.loads((root / "shear/receipt.json").read_text())
            self.assertEqual((receipt["status"], receipt["cells"]), ("PARTIAL", [2, 2, 2]))
            result = subprocess.run(
                [*command, "summarize", "patch", "--input", str(root)], capture_output=True, check=False
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads((root / "patch-matrix.json").read_text())["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
