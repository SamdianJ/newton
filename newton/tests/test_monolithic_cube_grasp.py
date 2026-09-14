# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cube entry/asset contracts; these tests cannot certify a grasp fixture."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube
from newton.examples.softbody.monolithic_sharpa_grasp import FIXTURE, GraspCase
from newton.examples.softbody.monolithic_soft_cube import generate_cube
from newton.examples.softbody.sharpa_close import load_fixture
from scripts.monolithic_reference import run_cube_grasp as runner
from scripts.monolithic_reference.run_cube_grasp import offsets


class TestCubeGrasp(unittest.TestCase):
    def test_predeclared_scan(self):
        """Keep the declared 27 positions and middle-grid scan unchanged."""
        grid = offsets()
        self.assertEqual(len(grid), 27)
        self.assertEqual(len(set(grid)), 27)
        self.assertIn((0, 0, 0), grid)
        self.assertEqual({o[2] for o in grid}, {-5, 0, 5})

    def test_reject_failed_calibration_before_asset_load(self):
        """Reject missing physical acceptance before loading the hand."""
        if not wp.is_cuda_available():
            self.skipTest("Grasp requires CUDA")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.json"
            path.write_text(json.dumps({"status": "FAILED_NO_PHYSICAL_ENVELOPE"}))
            args = SimpleNamespace(cube="unused.npz", device="cuda:0", calibration=str(path))
            with patch(
                "newton.examples.softbody.monolithic_sharpa_grasp.load_hand_cube",
                side_effect=AssertionError("Asset loaded before gate"),
            ):
                with self.assertRaisesRegex(ValueError, "Freeze the contact calibration"):
                    GraspCase(args)
            args.experiment = "anchored-close"
            with self.assertRaisesRegex(ValueError, "all dynamic"):
                GraspCase(args)

    def test_real_hand_cube_attachment(self):
        """Check the cube attachment with the unchanged 22-DoF hand and carriage."""
        source = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
        contact = os.environ.get("NEWTON_SHARPA_CONTACT_DIR")
        if not source or not contact:
            self.skipTest("Set real Sharpa assets for attachment checks")
        parameters = load_fixture(Path(__file__).parents[1] / "examples/softbody/sharpa_g1h.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.npz"
            generate_cube(2).save(str(path))
            for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
                model, manifest = load_hand_cube(
                    source,
                    contact,
                    path,
                    device=device,
                    parameters=parameters,
                    position=FIXTURE["ball_position"],
                    _mount={"support_height": 0.1698, "carriage": FIXTURE["carriage"]},
                )
                self.assertEqual(model.joint_dof_count, 23)
                self.assertEqual(model.particle_count, 343)
                self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0))
                self.assertAlmostEqual(float(model.particle_mass.numpy().sum()), 0.064, places=7)
                self.assertEqual(manifest["object_type"], "cube")
                self.assertNotIn("ball_radius_m", manifest)
                np.testing.assert_allclose(model.particle_q.numpy().mean(axis=0), FIXTURE["ball_position"], atol=1e-6)

    def test_uncalibrated_constructor_and_rollback_contract(self):
        """Mock only the calibration gate; this is not a physical acceptance run."""
        source = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
        contact = os.environ.get("NEWTON_SHARPA_CONTACT_DIR")
        trajectory = os.environ.get("NEWTON_SHARPA_TRAJECTORY")
        if not source or not contact or not trajectory or not wp.is_cuda_available():
            self.skipTest("Set real hand/trajectory for the isolated constructor contract")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cube.npz"
            generate_cube(1).save(str(path))
            calibration = root / "NOT_A_CALIBRATION.json"
            calibration.write_text("{}")
            args = SimpleNamespace(
                cube=str(path),
                device="cuda:0",
                calibration=str(calibration),
                asset_dir=source,
                contact_dir=contact,
                trajectory=trajectory,
                disable_aabb=False,
                friction_off=False,
                pcg_mode="production",
            )
            with (
                patch(
                    "newton.examples.softbody.monolithic_sharpa_grasp.load_calibration",
                    return_value={"stiffness_floor_n_m3": 1e7},
                ),
                patch("newton.examples.softbody.monolithic_sharpa_grasp.validate_cube_calibration"),
            ):
                case = GraspCase(args)
            self.assertTrue(case.solver.use_cuda_graph)
            self.assertTrue(case.solver.use_optimized_articulation_mass_matrix)
            self.assertFalse(np.any(case.fixed))
            case.step()
            self.assertTrue(case.records)
            self.assertIn("object_com", case.records[-1])
            self.assertIn("finger_normal_force_on_object", case.records[-1])
            # A non-finite external force must never advance/publish a new state.
            x = case.state.particle_q.numpy().copy()
            q = case.state.joint_q.numpy().copy()
            step = case.step_count
            case.failure = None
            case.state.particle_f.fill_(wp.vec3(float("nan"), 0, 0))
            case.step()
            self.assertIsNotNone(case.failure)
            self.assertEqual(case.step_count, step)
            np.testing.assert_array_equal(case.state.particle_q.numpy(), x)
            np.testing.assert_array_equal(case.state.joint_q.numpy(), q)
            summary = case.save(root / "contract-output")
            self.assertFalse(summary["complete_schedule"])
            self.assertEqual(summary["G6"], "NOT_VALIDATED")
            self.assertNotIn("ball_radius_m", summary)

    def test_batch_retains_failure_and_continues(self):
        """Retain failed and timed-out jobs while executing all later positions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "cube.npz"
            generate_cube(2).save(str(mesh))
            argv = [
                "run_cube_grasp",
                "--scan",
                "--asset-dir",
                "unused",
                "--contact-dir",
                "unused",
                "--trajectory",
                "unused",
                "--calibration",
                "unused",
                "--cube",
                str(mesh),
                "--output",
                str(root / "scan"),
            ]
            processes = []
            for index in range(27):
                process = MagicMock()
                process.__enter__.return_value = process
                process.wait.side_effect = (
                    [subprocess.TimeoutExpired("test", 1), -9] if index == 1 else [1 if index == 0 else 0]
                )
                processes.append(process)
            with (
                patch("sys.argv", argv),
                patch.object(runner, "snapshot_sources"),
                patch.object(runner, "load_calibration", return_value={}),
                patch.object(runner, "validate_cube_calibration"),
                patch.object(runner.subprocess, "Popen", side_effect=processes) as launch,
                patch.object(runner.os, "killpg") as kill_group,
            ):
                self.assertEqual(runner.main(), 1)
            kill_group.assert_called_once_with(processes[1].pid, runner.signal.SIGKILL)
            jobs = json.loads((root / "scan/jobs.json").read_text())
            self.assertEqual(len(jobs), 27)
            self.assertEqual(launch.call_count, 27)
            self.assertEqual(sum(row["exit_code"] != 0 for row in jobs), 2)


if __name__ == "__main__":
    unittest.main()
