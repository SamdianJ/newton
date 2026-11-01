# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quick regression manifest, gates and runner error isolation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.monolithic_reference.quick_manifest import load_manifest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
