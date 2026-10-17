# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from scripts.monolithic_reference.physical_gates import evaluate_robot, load_gates

GATES = {
    "hard_failures_max": 0,
    "stopped_fraction_max": 0.01,
    "penetration_max_m": 0.001,
    "det_f_min": 0.2,
    "require_held_cube": True,
}


def summary(**overrides):
    value = {
        "complete": True,
        "counts": {"CONVERGED": 990, "STOPPED_ACCEPTED": 10, "HARD_FAILURE": 0},
        "maximum_penetration_m": 0.0005,
        "minimum_det_f": 0.9,
        "held_cube": True,
    }
    value.update(overrides)
    return value


class TestPhysicalGates(unittest.TestCase):
    def test_complete_physical_run_passes(self):
        self.assertEqual(evaluate_robot(summary(), GATES), {"status": "PASS", "failures": [], "steps": 1000})

    def test_each_violation_is_reported(self):
        result = evaluate_robot(
            summary(
                complete=False,
                counts={"CONVERGED": 980, "STOPPED_ACCEPTED": 19, "HARD_FAILURE": 1},
                maximum_penetration_m=0.002,
                minimum_det_f=0.2,
                held_cube=False,
            ),
            GATES,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["failures"], ["complete", "det_f", "hard_failures", "held_cube", "penetration", "stopped_fraction"]
        )

    def test_sharpa_publication_counts_are_accepted(self):
        value = summary()
        value["publication_counts"] = value.pop("counts")
        del value["held_cube"]
        self.assertEqual(evaluate_robot(value, {**GATES, "require_held_cube": False})["status"], "PASS")

    def test_incomplete_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "publication counts"):
            evaluate_robot(summary(counts={"CONVERGED": 1}), GATES)

    def test_proposed_recipes_load_only_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gates.json"
            path.write_text(json.dumps({"version": 1, "status": "PROPOSED", "robot": {}}))
            with self.assertRaisesRegex(ValueError, "frozen"):
                load_gates(path)
            self.assertEqual(load_gates(path, allow_proposed=True)["status"], "PROPOSED")
            path.write_text(json.dumps({"version": 2, "status": "FROZEN", "robot": {}}))
            with self.assertRaisesRegex(ValueError, "version"):
                load_gates(path, allow_proposed=True)

    def test_repository_recipe_is_frozen_for_both_robot_scenes(self):
        gates = load_gates()
        self.assertEqual(sorted(gates["robot"]), ["franka-cube", "sharpa-recorded-10mm"])
        for scene in gates["robot"].values():
            self.assertEqual(sorted(scene), sorted(GATES))


if __name__ == "__main__":
    unittest.main()
