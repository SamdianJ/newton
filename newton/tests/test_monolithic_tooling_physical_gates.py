# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from scripts.monolithic_reference.physical_gates import evaluate_robot, load_gates

GRASP = {"window_steps": 500, "ground_weight_fraction_max": 0.05, "com_rise_min_edge_fraction": 0.25, "required": True}
GATES = {
    "hard_failures_max": 0,
    "stopped_fraction_max": 0.01,
    "penetration_max_m": 0.001,
    "det_f_min": 0.2,
    "grasp": GRASP,
}


def grasp(**overrides):
    value = {
        "window_first_step": 501,
        "window_last_step": 1000,
        "window_steps": 500,
        "min_com_rise_m": 0.02,
        "max_ground_force_N": 0.0,
        "weight_N": 0.64,
        "object_min_edge_m": 0.04,
        "held": True,
    }
    value.update(overrides)
    return value


def summary(**overrides):
    value = {
        "complete": True,
        "counts": {"CONVERGED": 990, "STOPPED_ACCEPTED": 10, "HARD_FAILURE": 0},
        "maximum_penetration_m": 0.0005,
        "minimum_det_f": 0.9,
        "grasp": grasp(),
    }
    value.update(overrides)
    return value


class TestPhysicalGates(unittest.TestCase):
    def test_complete_physical_run_passes(self):
        self.assertEqual(
            evaluate_robot(summary(), GATES),
            {"status": "PASS", "failures": [], "steps": 1000, "grasp": "PASS", "grasp_required": True},
        )

    def test_each_violation_is_reported(self):
        result = evaluate_robot(
            summary(
                complete=False,
                counts={"CONVERGED": 980, "STOPPED_ACCEPTED": 19, "HARD_FAILURE": 1},
                maximum_penetration_m=0.002,
                minimum_det_f=0.2,
                grasp=grasp(min_com_rise_m=0.002),
            ),
            GATES,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["failures"], ["complete", "det_f", "grasp", "hard_failures", "penetration", "stopped_fraction"]
        )

    def test_grasp_uses_frozen_thresholds(self):
        """Judge the raw window quantities with the recipe; the summary's label is not trusted."""
        self.assertEqual(
            evaluate_robot(summary(grasp=grasp(min_com_rise_m=0.01, max_ground_force_N=0.032)), GATES)["grasp"], "PASS"
        )
        for change in (
            {"min_com_rise_m": 0.0099},
            {"max_ground_force_N": 0.033},
            {"window_steps": 499, "window_last_step": 999},
            {"window_first_step": 502},
        ):
            with self.subTest(**change):
                result = evaluate_robot(summary(grasp=grasp(**change)), GATES)
                self.assertEqual((result["status"], result["failures"], result["grasp"]), ("FAIL", ["grasp"], "FAIL"))
        missing = summary()
        del missing["grasp"]
        self.assertEqual(evaluate_robot(missing, GATES)["failures"], ["grasp"])

    def test_diagnostic_grasp_does_not_fail(self):
        value = summary(grasp=grasp(min_com_rise_m=-0.0004, max_ground_force_N=0.628, held=False))
        value["publication_counts"] = value.pop("counts")
        result = evaluate_robot(value, {**GATES, "grasp": {**GRASP, "required": False}})
        self.assertEqual((result["status"], result["grasp"], result["grasp_required"]), ("PASS", "FAIL", False))

    def test_incomplete_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "publication counts"):
            evaluate_robot(summary(counts={"CONVERGED": 1}), GATES)

    def test_proposed_recipes_load_only_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gates.json"
            path.write_text(json.dumps({"version": 2, "status": "PROPOSED", "robot": {}}))
            with self.assertRaisesRegex(ValueError, "frozen"):
                load_gates(path)
            self.assertEqual(load_gates(path, allow_proposed=True)["status"], "PROPOSED")
            path.write_text(json.dumps({"version": 1, "status": "FROZEN", "robot": {}}))
            with self.assertRaisesRegex(ValueError, "version"):
                load_gates(path, allow_proposed=True)

    def test_repository_recipe_is_frozen_for_both_robot_scenes(self):
        from newton.examples.softbody import monolithic_recording  # noqa: PLC0415

        gates = load_gates()
        self.assertEqual(sorted(gates["robot"]), ["franka-cube", "sharpa-recorded-10mm"])
        for scene in gates["robot"].values():
            self.assertEqual(sorted(scene), sorted(GATES))
            self.assertEqual(sorted(scene["grasp"]), sorted(GRASP))
            # The examples label their summaries with the same frozen definition.
            self.assertEqual(scene["grasp"]["window_steps"], monolithic_recording.GRASP_WINDOW_STEPS)
            self.assertEqual(
                scene["grasp"]["ground_weight_fraction_max"], monolithic_recording.GRASP_GROUND_WEIGHT_FRACTION
            )
            self.assertEqual(
                scene["grasp"]["com_rise_min_edge_fraction"], monolithic_recording.GRASP_RISE_EDGE_FRACTION
            )
        self.assertEqual(
            {k: v["grasp"]["required"] for k, v in gates["robot"].items()},
            {"franka-cube": True, "sharpa-recorded-10mm": False},
        )


if __name__ == "__main__":
    unittest.main()
