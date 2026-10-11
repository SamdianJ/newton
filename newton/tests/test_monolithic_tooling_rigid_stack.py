# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard the large rigid scene's independent geometry and failure classification."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.rigid_stack_scene import stack_scene
from scripts.monolithic_reference.run_rigid_stack import CASE, box_penetration, expected_loads, step_errors


class TestRigidStackMonitoring(unittest.TestCase):
    def test_monitor_failure_keeps_state(self):
        """Keep the native checkpoint when offline monitoring fails after a real step."""
        from scripts.monolithic_reference import run_rigid_stack  # noqa: PLC0415

        model = stack_scene("cpu")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(run_rigid_stack, "stack_scene", return_value=model),
            patch.object(run_rigid_stack, "measure", side_effect=ValueError("injected monitor failure")),
        ):
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "injected monitor failure"):
                run_rigid_stack.record_run(output, max_steps=1, device="cpu")
            self.assertTrue((output / "recording/final.npz").is_file())
            self.assertTrue((output / "steps.jsonl.gz").is_file())

    def test_box_penetration(self):
        """Detect ground/box overlap and reject an AABB-only false collision."""
        rotations = np.repeat(np.eye(3)[None], 2, axis=0)
        ground, pair = box_penetration(np.array([[0, 0, 0.019], [0, 0, 0.058]]), rotations)
        self.assertAlmostEqual(ground, 0.001)
        self.assertAlmostEqual(pair, 0.001)
        angle = np.pi / 4
        rotations[1] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        ground, pair = box_penetration(np.array([[0, 0, 1], [0.045, 0.045, 1]]), rotations)
        self.assertEqual((ground, pair), (0, 0))

    def test_quality_failures(self):
        """Fail capacity, nonfinite, excessive penetration and publication errors independently."""
        import copy  # noqa: PLC0415
        import json  # noqa: PLC0415

        gates = json.loads(CASE.read_text())["gates"]
        row = {
            "finite": True,
            "force_generation_valid": True,
            "layer_order": True,
            "ground_penetration_m": 0,
            "pair_penetration_m": 0,
            "force_balance_relative": 0,
            "torque_balance_relative": 0,
            "pool": {"tainted": False, "dropped": 0},
            "query": {"tainted": False, "capacity_dropped": 0},
        }
        self.assertEqual(step_errors(row, gates), [])
        for field, value in (
            ("finite", False),
            ("force_generation_valid", False),
            ("layer_order", False),
            ("ground_penetration_m", 0.003),
            ("pair_penetration_m", np.nan),
            ("force_balance_relative", 1e-4),
            ("torque_balance_relative", 1e-4),
        ):
            with self.subTest(field=field):
                self.assertIn(field, step_errors({**row, field: value}, gates))
        for owner, count in (("pool", "dropped"), ("query", "capacity_dropped")):
            altered = copy.deepcopy(row)
            altered[owner][count] = 1
            self.assertIn("capacity_overflow", step_errors(altered, gates))
        loads = expected_loads(16, 4)
        self.assertEqual(len(loads), 64)
        self.assertEqual(sum(value for (source, target), value in loads.items() if target == 64), 64 * 0.01 * 9.81)


if __name__ == "__main__":
    unittest.main()
