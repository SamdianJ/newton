# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent hand-contact audit quadrature, geometry and pressure units."""

import unittest

import numpy as np

from scripts.monolithic_reference.audit_hand_contact import (
    box_geometry,
    control_input_identity,
    normal_response,
    surface_samples,
)
from scripts.monolithic_reference.contact_oracle import penalty


class TestHandContactAudit(unittest.TestCase):
    def test_control_input_identity(self):
        """Reject each changed control input despite equivalent relative and absolute paths."""
        names = ("sharpa_g1h.json", "manifest.json", "recorded-control.json", "closure.csv")
        recorded = {f"../assets/{name}": str(i) for i, name in enumerate(names)}
        current = {f"/workspace/assets/{name}": str(i) for i, name in enumerate(names)}
        self.assertEqual(control_input_identity(recorded, current)["status"], "VERIFIED")
        for path in current:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Control input changed"):
                control_input_identity(recorded, {**current, path: "changed"})

    def test_missing_control_identity_remains_unverified(self):
        """Keep absent or partial historical input hashes distinct from verified identity."""
        current = {"/assets/closure.csv": "curve", "/assets/manifest.json": "mount"}
        unknown = control_input_identity(None, current)
        self.assertEqual(unknown["status"], "UNVERIFIED")
        self.assertEqual(unknown["missing_recorded_inputs"], ["closure.csv", "manifest.json"])
        partial = control_input_identity({"../assets/closure.csv": "curve"}, current)
        self.assertEqual(partial["status"], "UNVERIFIED")
        self.assertEqual(partial["missing_recorded_inputs"], ["manifest.json"])
        with self.assertRaisesRegex(ValueError, "Ambiguous control input"):
            control_input_identity({"/one/closure.csv": "curve", "/two/closure.csv": "curve"}, current)

    def test_triangle_area_and_linear_moment(self):
        """Preserve surface area and linear moments under triangle refinement."""
        vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype=float)
        for level in (0, 1, 2):
            points, normals, weights = surface_samples(vertices, [[0, 1, 2]], refinement=level)
            self.assertAlmostEqual(weights.sum(), 3)
            np.testing.assert_allclose(weights @ points, [2, 3, 0])
            np.testing.assert_allclose(normals, np.tile([0, 0, 1], (len(points), 1)))
        with self.assertRaises(ValueError):
            surface_samples(vertices, [[0, 0, 1]])

    def test_box_distance_and_gradient(self):
        """Match box distances and analytic gradients to centered differences."""
        points = np.array([[0.3, 0.1, 0], [1.4, 1.3, 0.2], [-0.2, -0.7, 0]])
        distance, gradient = box_geometry(points, [1, 1, 1])
        np.testing.assert_allclose(distance, [-0.7, 0.5, -0.3])
        np.testing.assert_allclose(gradient, [[1, 0, 0], [0.8, 0.6, 0], [0, -1, 0]])
        for axis in range(3):
            shift = np.eye(3)[axis] * 1e-6
            plus, _ = box_geometry(points + shift, [1, 1, 1])
            minus, _ = box_geometry(points - shift, [1, 1, 1])
            np.testing.assert_allclose((plus - minus) / 2e-6, gradient[:, axis], atol=1e-9)

    def test_area_and_half_weight_control_force(self):
        """Scale force by effective area and omit aligned surface samples."""
        gradients = np.array([[0.0, 0, 1]])
        full = normal_response(np.array([-0.007]), gradients, -gradients, np.array([1e-5]))
        half = normal_response(np.array([-0.007]), gradients, -gradients, np.array([0.5e-5]))
        self.assertAlmostEqual(full["normal_load_N"], 0.7)
        self.assertAlmostEqual(half["normal_load_N"], 0.35)
        masked = normal_response(np.array([-0.007]), gradients, gradients, np.array([1e-5]))
        self.assertEqual(masked["normal_load_N"], 0)
        self.assertEqual(masked["masked_penetrating_samples"], 1)
        self.assertAlmostEqual(masked["unmasked_normal_load_N"], 0.7)

    def test_smoothed_pressure_matches_scalar_oracle(self):
        """Match the independent pressure law throughout its smoothing interval."""
        distance = np.array([-2e-4, -5e-5, 0, 5e-5, 2e-4])
        gradients = np.tile([0.0, 0, 1], (len(distance), 1))
        weights = np.full(len(distance), 3e-5)
        result = normal_response(distance, gradients, -gradients, weights)
        expected = sum(penalty(d, 1e7 * a, 1e-4)[1] for d, a in zip(distance, weights, strict=True))
        self.assertAlmostEqual(result["normal_load_N"], expected)
        np.testing.assert_allclose(result["source_normal_force_N"], [0, 0, expected])


if __name__ == "__main__":
    unittest.main(verbosity=2)
