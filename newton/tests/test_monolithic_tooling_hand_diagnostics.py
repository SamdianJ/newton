# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify contact diagnostic attribution independently of candidate storage order."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.monolithic_reference.diagnose_hand_contact import compare_contacts, replay_parameters, restore_control
from scripts.monolithic_reference.hand_task_control import ContactHandControl, HandObservation


def snapshot(ids, mask, energy):
    return {
        "identity": np.asarray(ids),
        "mask": np.asarray(mask),
        "energy": np.asarray(energy),
        "source": np.asarray(ids) + 10,
        "target": np.asarray(ids) + 20,
    }


class TestHandDiagnostics(unittest.TestCase):
    def test_reordered_candidates(self):
        """Attribute an activation jump to its stable identity rather than its array slot."""
        before = snapshot([2, 1], [2, 1], [0.0, 3.0])
        after = snapshot([1, 2], [1, 1], [2.5, 0.75])
        before["point"] = np.array([[2, 0, 0], [1, 0, 0]])
        after["point"] = np.array([[1, 0, 0], [2, 0, 1]])
        result = compare_contacts(before, after)
        self.assertTrue(result["same_candidates"])
        self.assertEqual(result["mask_changes"], 1)
        self.assertAlmostEqual(result["mask_change_energy"], 0.75)
        self.assertAlmostEqual(result["all_contact_energy_change"], 0.25)
        self.assertEqual(result["changed_contacts"][0]["identity"], 2)
        self.assertEqual(result["changed_contacts"][0]["after"]["point"], [2, 0, 1])

    def test_candidate_change_is_not_no_mask_change(self):
        """Leave attribution unresolved when a different candidate set was evaluated."""
        result = compare_contacts(snapshot([1], [1], [1.0]), snapshot([2], [1], [1.0]))
        self.assertEqual(result, {"same_candidates": False})

    def test_energy_change_without_mask_change(self):
        """Separate ordinary contact energy variation from orientation selection changes."""
        result = compare_contacts(snapshot([1], [1], [2.0]), snapshot([1], [1], [1.0]))
        self.assertEqual(result["mask_changes"], 0)
        self.assertEqual(result["mask_change_energy"], 0.0)
        self.assertEqual(result["all_contact_energy_change"], -1.0)

    def test_duplicate_identity(self):
        """Reject ambiguous attribution instead of pairing duplicate contact identifiers."""
        with self.assertRaises(ValueError):
            compare_contacts(snapshot([1, 1], [1, 2], [1.0, 0.0]), snapshot([1, 2], [1, 2], [1.0, 0.0]))

    def test_manifest_parameters(self):
        """Reconstruct explicit nondefault inputs and reject ambiguous old manifests."""
        scene = {"task": "grasp-rigid", "surface_source": "cad", "object_position_m": [0, 0.003, 0.1002]}
        manifest = {
            "task": "all",
            "surface_source": "cad",
            "friction": 0.7,
            "offset": 0.003,
            "height": 0.08,
            "control_mode": "contact",
            "contact_stiffness_N_m3": 2e7,
            "dt": 0.002,
            "precision": "fp64",
        }
        parameters = replay_parameters(scene, manifest)
        self.assertEqual(parameters["friction"], 0.7)
        self.assertEqual(parameters["offset"], 0.003)
        self.assertEqual(parameters["height"], 0.08)
        self.assertEqual(parameters["control_mode"], "contact")
        self.assertEqual(parameters["contact_stiffness"], 2e7)
        with self.assertRaisesRegex(ValueError, "surface_source"):
            replay_parameters({**scene, "surface_source": "qualified"}, manifest)
        with self.assertRaisesRegex(ValueError, "control_mode"):
            replay_parameters(scene, {k: v for k, v in manifest.items() if k != "control_mode"})

    def test_legacy_recipe_requires_known_identity(self):
        """Apply historical constants only to the identified PR09 runner recipe."""
        scene = {"task": "grasp-soft", "surface_source": "qualified", "object_position_m": [0, 0, 0.0202]}
        manifest = {
            "task": "all",
            "surface_source": "qualified",
            "friction": 0.5,
            "offset": 0,
            "dt": 0.002,
            "precision": "fp64",
        }
        with self.assertRaises(ValueError):
            replay_parameters(scene, manifest)
        manifest["source_sha256"] = {
            "scripts/monolithic_reference/run_hand_tasks.py": "f6636403e8fc1e058860f3fd4e10a7a5574d3fc62cb57f896c83fb78db619820"
        }
        parameters = replay_parameters(scene, manifest)
        self.assertEqual(parameters["control_mode"], "open-loop")
        self.assertEqual(parameters["contact_stiffness"], 1e7)
        self.assertEqual(parameters["height"], 0)

    def test_control_restore_and_time_identity(self):
        """Restore the pre-advance controller and reject a different failure time."""
        controller = ContactHandControl("grasp-rigid", mass=0.064, friction=0.5, source_rates=np.ones(5))
        controller.last_time = 0.008
        controller.phase, controller.sources = "close", np.full(5, 0.25)
        observation = HandObservation(4, np.ones(5), np.zeros(5), np.zeros(5), 0, np.zeros(3), np.zeros(3), 0)
        payload = {"time": 0.01, "controller": controller.snapshot(), "feedback": observation.snapshot()}
        scene = SimpleNamespace(
            controller=ContactHandControl("grasp-rigid", mass=0.064, friction=0.5, source_rates=np.ones(5))
        )
        restore_control(scene, payload, 5)
        self.assertEqual(scene.controller.last_time, 0.008)
        self.assertEqual(scene.controller.phase, "close")
        np.testing.assert_array_equal(scene.controller.sources, np.full(5, 0.25))
        self.assertEqual(scene.feedback.generation, 4)
        controller.advance(0.01, observation)
        scene.controller.advance(0.01, scene.feedback)
        self.assertEqual(scene.controller.snapshot(), controller.snapshot())
        with self.assertRaisesRegex(ValueError, "time"):
            restore_control(scene, {**payload, "time": 0.012}, 5)
        with self.assertRaisesRegex(ValueError, "generation"):
            restore_control(scene, {**payload, "feedback": {**payload["feedback"], "generation": 3}}, 5)


if __name__ == "__main__":
    unittest.main()
