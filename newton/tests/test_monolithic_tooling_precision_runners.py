# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Carry the requested solver precision through the material and benchmark scene runners."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.solvers.experimental.monolithic import SolverMonolithic
from scripts.monolithic_reference.cable_benchmarks import CableBenchmark
from scripts.monolithic_reference.cloth_benchmarks import ClothBenchmark
from scripts.monolithic_reference.material_scene import MaterialScene
from scripts.monolithic_reference.tet_benchmarks import TetBenchmark

ROOT = Path(__file__).resolve().parents[2]
FP32 = SolverMonolithic.Precision.FP32


class TestPrecisionRunners(unittest.TestCase):
    def test_scenes_build_and_rebuild_in_precision(self):
        """Construct each scene in FP32 and keep the precision for its checkpoint rebuild."""
        material = MaterialScene("cloth-press", precision="fp32")
        cloth = ClothBenchmark("patch", precision="fp32")
        cable = CableBenchmark("cantilever", segments=8, precision="fp32")
        tet = TetBenchmark("patch", case="shear", cells=(2, 2, 2), precision="fp32")
        for case in (material, cloth, cable, tet):
            with self.subTest(case=type(case).__name__):
                self.assertEqual(case.solver.precision, FP32)
        self.assertEqual((material.options["precision"], cloth.options["precision"]), ("fp32", "fp32"))
        self.assertEqual(cable.solver_options["precision"], FP32)

    def test_receipts_record_precision(self):
        """Record the solver's actual precision in single-case and summary receipts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = {
                "cloth": ["run_cloth_benchmark", "patch"],
                "material": ["run_material_acceptance", "--case", "cloth-press"],
            }
            for name, (module, *arguments) in commands.items():
                options = ["--max-steps", "1", "--precision", "fp32", "--output", str(root / name)]
                subprocess.run(
                    [sys.executable, "-B", "-m", f"scripts.monolithic_reference.{module}", *arguments, *options],
                    cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True,
                    check=True,
                )
            cloth = json.loads((root / "cloth/receipt.json").read_text())
            self.assertEqual((cloth["status"], cloth["precision"]), ("PARTIAL", "fp32"))
            material = json.loads((root / "material/cloth-press/receipt.json").read_text())
            self.assertEqual((material["status"], material["precision"]), ("PARTIAL", "fp32"))
            summary = json.loads((root / "material/summary.json").read_text())
            self.assertEqual((summary["status"], summary["precision"]), ("PARTIAL", "fp32"))


class TestNumericalGates(unittest.TestCase):
    def test_row_excess(self):
        """Admit each row up to the larger of the frozen gate and its rounding floor."""
        from scripts.monolithic_reference.numerical_gates import row_excess  # noqa: PLC0415

        residual = np.array([5e-7, -3e-6, 0.0])
        self.assertAlmostEqual(row_excess(residual, np.zeros(3), 1e-6), 3.0)
        self.assertAlmostEqual(row_excess(residual, np.array([0.0, 4e-6, 0.0]), 1e-6), 0.75)
        self.assertEqual(row_excess([], [], 1e-6), 0.0)

    def test_floor_follows_precision(self):
        """Keep FP64 floors at the rounding term far below the gates; FP32 floors are the solver tolerance."""
        from scripts.monolithic_reference.numerical_gates import (  # noqa: PLC0415
            balance_rounding,
            block_norms,
            gate_floor,
        )

        for precision in ("fp64", "fp32"):
            with self.subTest(precision=precision):
                case = MaterialScene("cable-bend-twist", precision=precision)
                case.step()
                criterion = case.solver._criterion
                floor = gate_floor(case.solver)
                residual = case.solver.last_stats.variational["residual"]
                blocks = case.solver.last_stats.variational["material_blocks"]
                norm, _ = block_norms(case.solver, residual, floor, "cable_0_x")
                expected = blocks["cable_0_x"]["raw_residual_cable_0_x_norm"]
                self.assertAlmostEqual(norm, expected, delta=1e-12 + 1e-6 * norm)
                native = np.float64 if precision == "fp64" else np.float32
                self.assertAlmostEqual(balance_rounding(case.solver), 32 * float(np.finfo(native).eps), delta=1e-20)
                if precision == "fp64":
                    rounding = criterion.noise.numpy() * criterion.magnitude.numpy()
                    np.testing.assert_array_equal(floor, rounding)
                    self.assertLess(floor.max(), 1e-3 * case.common["gates"]["force_residual_inf_N"])
                else:
                    np.testing.assert_array_equal(floor, criterion.tau.numpy().astype(float))


if __name__ == "__main__":
    unittest.main()
