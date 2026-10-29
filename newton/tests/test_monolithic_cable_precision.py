# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold cable responses to the same physics in every orientation and precision."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.cable_benchmarks import CableBenchmark

ROOT = Path(__file__).resolve().parents[2]


def tip_response(case, steps):
    """Return the tip displacement along the rotated load axis after each step."""
    direction = case.rotation[:, case.axis]
    tip = []
    for _ in range(steps):
        case.step()
        tip.append(float((case.state.particle_q.numpy()[-1].astype(float) - case.rest[-1]) @ direction))
    return np.array(tip)


class TestCableOrientation(unittest.TestCase):
    def test_zero_angle_is_the_axis_aligned_recipe(self):
        """Build bit-identical inputs at zero angle and the default."""
        default, zero = CableBenchmark("cantilever", segments=8), CableBenchmark("cantilever", segments=8, angle=0.0)
        np.testing.assert_array_equal(default.rest, zero.rest)
        np.testing.assert_array_equal(default.model.particle_mass.numpy(), zero.model.particle_mass.numpy())
        cable = zero.solver_options["cable_inputs"][0]
        np.testing.assert_array_equal(default.solver_options["cable_inputs"][0].rest_director, cable.rest_director)

    def test_only_the_cantilever_rotates(self):
        """Reject an angle for the helix and a nonfinite angle for the cantilever."""
        with self.assertRaises(ValueError):
            CableBenchmark("localized_helix", segments=8, angle=45.0)
        with self.assertRaises(ValueError):
            CableBenchmark("cantilever", segments=8, angle=float("nan"))

    def test_fp64_response_is_rotation_invariant(self):
        """Rotate the whole FP64 cantilever about z without changing its load response."""
        aligned = tip_response(CableBenchmark("cantilever", segments=16), 3)
        for angle in (30.0, 45.0):
            with self.subTest(angle=angle):
                rotated = CableBenchmark("cantilever", segments=16, angle=angle)
                np.testing.assert_allclose(tip_response(rotated, 3), aligned, rtol=1e-5)
                metrics = rotated.metrics()
                self.assertLess(metrics["maximum_axial_strain"], 1e-9)

    def test_metrics_report_coordinate_resolution(self):
        """Report the strain and velocity that stored coordinates can resolve in each precision."""
        for precision, native in (("fp32", np.float32), ("fp64", np.float64)):
            with self.subTest(precision=precision):
                case = CableBenchmark("cantilever", segments=8, angle=45.0, precision=precision)
                metrics = case.metrics()
                spacing = float(np.spacing(native(np.abs(case.state.particle_q.numpy()).max())))
                length = float(np.linalg.norm(np.diff(case.rest, axis=0), axis=1).min())
                self.assertAlmostEqual(metrics["axial_strain_resolution"], 2 * spacing / length)
                self.assertAlmostEqual(metrics["velocity_resolution"], spacing / case.dt)

    def test_runner_records_the_angle(self):
        """Carry the rotation into the single-case receipt that resume compares."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "case")
            command = [
                sys.executable,
                "-B",
                "-m",
                "scripts.monolithic_reference.run_cable_benchmark",
                "cantilever",
                "--segments",
                "8",
                "--angle",
                "45",
                "--max-steps",
                "1",
                "--output",
                str(output),
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            subprocess.run(command, cwd=ROOT, env=env, capture_output=True, check=True)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual((receipt["status"], receipt["arguments"]["angle"]), ("PARTIAL", 45.0))
            self.assertIn("axial_strain_resolution", receipt["metrics"])


if __name__ == "__main__":
    unittest.main()
