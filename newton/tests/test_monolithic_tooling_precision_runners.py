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


if __name__ == "__main__":
    unittest.main()
