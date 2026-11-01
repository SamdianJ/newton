# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quick regression manifest, gates and runner error isolation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import warp as wp

from scripts.monolithic_reference.quick_gates import evaluate
from scripts.monolithic_reference.quick_manifest import load_manifest
from scripts.monolithic_reference.retained_assets import INPUTS

GATES = {"hard_failures_max": 0, "stopped_max": 5, "penetration_max_m": 0.002, "det_f_min": 0.2}
CASE = {
    "id": "sharpa-close",
    "scene": "sharpa",
    "snapshot": {"path": "snapshots/a.npz", "sha256": "a" * 64, "step": 3240},
    "steps": 270,
    "gates": {"fp64": dict(GATES)},
}


def _write(folder, payload):
    path = Path(folder) / "quick_cases_v1.json"
    path.write_text(json.dumps(payload))
    return path


class TestQuickManifest(unittest.TestCase):
    def test_valid_manifest(self):
        """Return cases in file order with the manifest identity."""
        with tempfile.TemporaryDirectory() as folder:
            manifest = load_manifest(_write(folder, {"version": 1, "cases": [CASE]}))
        self.assertEqual(manifest["version"], 1)
        self.assertEqual([case["id"] for case in manifest["cases"]], ["sharpa-close"])
        self.assertEqual(len(manifest["sha256"]), 64)

    def test_rejections(self):
        """Reject unknown scenes and gates, bad identities, missing limits and duplicate ids."""
        grasp = {"last_steps": 300, "ground_weight_fraction_max": 0.05}
        bad = {
            "scene": {**CASE, "scene": "helix"},
            "sha256": {**CASE, "snapshot": {**CASE["snapshot"], "sha256": "xyz"}},
            "unknown gate": {**CASE, "gates": {"fp64": {**GATES, "speed_max": 1}}},
            "missing gate": {**CASE, "gates": {"fp64": {k: v for k, v in GATES.items() if k != "det_f_min"}}},
            "precision": {**CASE, "gates": {"fp16": dict(GATES)}},
            "steps": {**CASE, "steps": 0},
            "grasp window": {**CASE, "gates": {"fp64": {**GATES, "grasp": grasp}}},
        }
        for message, case in bad.items():
            with self.subTest(message), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    load_manifest(_write(folder, {"version": 1, "cases": [case]}))
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "duplicate"):
            load_manifest(_write(folder, {"version": 1, "cases": [CASE, copy.deepcopy(CASE)]}))


def _rows(n, *, stopped=(), hard=(), penetration=0.001, ground=0.0):
    rows = []
    for step in range(n):
        label = "HARD_FAILURE" if step in hard else "STOPPED_ACCEPTED" if step in stopped else "CONVERGED"
        rows.append(
            {
                "step": step,
                "publication": label,
                "penetration": None if label == "HARD_FAILURE" else penetration,
                "min_det_f": 0.6,
                "ground_force_z": ground,
            }
        )
    return rows


class TestQuickGates(unittest.TestCase):
    def test_pass_and_values(self):
        """Report each gate's value and limit; a complete clean window passes."""
        result = evaluate(_rows(10, stopped=(3,)), {**GATES, "stopped_max": 1}, weight=1.0, expected_steps=10)
        self.assertTrue(all(gate["passed"] for gate in result.values()), result)
        self.assertEqual(result["stopped_steps"]["value"], 1)
        self.assertEqual(result["penetration"]["value"], 0.001)

    def test_failures(self):
        """Fail the stopped, hard-failure, penetration, completeness and grasp gates independently."""
        grasp = {**GATES, "grasp": {"last_steps": 5, "ground_weight_fraction_max": 0.05}}
        cases = {
            "stopped_steps": (_rows(10, stopped=(1, 2)), {**GATES, "stopped_max": 1}),
            "hard_failures": (_rows(10, hard=(4,)), GATES),
            "penetration": (_rows(10, penetration=0.003), GATES),
            "complete": (_rows(8), GATES),
            "grasp": (_rows(10, ground=0.5), grasp),
        }
        for gate, (rows, gates) in cases.items():
            with self.subTest(gate):
                result = evaluate(rows, gates, weight=1.0, expected_steps=10)
                self.assertFalse(result[gate]["passed"], result)
                others = [name for name, value in result.items() if name != gate and not value["passed"]]
                self.assertEqual(others, [], result)


@unittest.skipUnless(
    wp.is_cuda_available() and (INPUTS / "sharpa/recorded-control.json").is_file(), "requires CUDA and Sharpa inputs"
)
class TestQuickScenesCUDA(unittest.TestCase):
    def test_sharpa_window_two_steps(self):
        """Import the initial Sharpa state into its own window and publish two observed steps."""
        from newton.examples.softbody.monolithic_recording import state_arrays  # noqa: PLC0415
        from scripts.monolithic_reference.quick_scenes import SharpaWindow  # noqa: PLC0415

        window = SharpaWindow("fp32", "cuda:0")
        model = {"particle_mass": window.solver.model.particle_mass.numpy()}
        window.start(state_arrays(window.state), model, 0)
        rows = [window.advance() for _ in range(2)]
        self.assertEqual([row["step"] for row in rows], [1, 2])
        self.assertIn(rows[-1]["publication"], ("CONVERGED", "STOPPED_ACCEPTED"))
        self.assertGreaterEqual(rows[-1]["penetration"], 0.0)
        self.assertGreater(window.weight, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
