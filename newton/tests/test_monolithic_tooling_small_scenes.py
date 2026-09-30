# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check small-scene construction, sampling and result accounting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import build_model
from scripts.monolithic_reference.p3_coupled_fixtures import NORMAL, knots_at, make_tiny
from scripts.monolithic_reference.summarize_p3_coupled import LIMITS, differences, errors, summarize
from scripts.monolithic_reference.verify_p3_coupled import assess, run_case


def test_normal_input(test, device):
    """Match every physical array of the original normal-loading builder."""
    expected = build_model(NORMAL, device=device)
    actual = make_tiny("normal_loading", device)
    for name in (
        "particle_q",
        "particle_qd",
        "particle_mass",
        "particle_inv_mass",
        "tet_indices",
        "tet_poses",
        "tet_materials",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "body_mass",
        "body_inertia",
        "shape_transform",
        "shape_margin",
        "shape_scale",
    ):
        np.testing.assert_array_equal(getattr(actual, name).numpy(), getattr(expected, name).numpy(), err_msg=name)


class TestSmallSceneTools(unittest.TestCase):
    def test_measurement_window(self):
        """Exclude the endpoint immediately before the final 200 physical steps."""
        force = np.array([[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]], dtype=float)
        rows = [
            {
                "time": (i + 1) * 0.001,
                "success": True,
                "penetration": 0.0,
                "min_det_f": 1.0,
                "friction_work": 0.0,
                "body_contact_force": force,
            }
            for i in range(1500)
        ]
        rows[1299]["body_contact_force"] = force * 200
        result = assess(rows, "cube1", 1500, 1.0)
        self.assertEqual(result["physical"], "PASS")
        self.assertEqual(result["force_error"], 0.0)

        rest = np.asarray(NORMAL["soft"]["rest_positions_m"])
        compressed = rest.copy()
        compressed[:3, 2] += 0.006
        rows = [
            {
                "time": (i + 1) * 0.001,
                "success": True,
                "penetration": 0.0,
                "min_det_f": 1.0,
                "friction_work": 0.0,
                "contact_load": 0.0 if i < 200 else 1.0,
                "slip": 0.0,
                "force_balance": 0.0,
                "q": np.array([0.003]),
                "x": compressed,
            }
            for i in range(1000)
        ]
        rows[899]["x"] = rest
        self.assertEqual(assess(rows, "normal_loading", 1000, None)["physical"], "PASS")

    def test_trajectory_metrics(self):
        """Keep metres/radians distinct and reject shifted or missing time grids."""
        a = {
            "time": np.array([0.001, 0.002]),
            "q": np.zeros((2, 2)),
            "qd": np.zeros((2, 2)),
            "com": np.zeros((2, 2, 3)),
            "x": np.zeros((2, 4, 3)),
            "rotation": np.tile(np.eye(3), (2, 2, 1, 1)),
        }
        b = {key: value.copy() for key, value in a.items()}
        b["q"][0] = [1e-4, 1e-4]
        b["x"][0, 0, :2] = [3e-5, 4e-5]
        measured = errors(a, b, "curved_release")
        self.assertGreater(measured["q_m"]["maximum"], LIMITS["q_m"])
        self.assertLess(measured["q_rad"]["maximum"], LIMITS["q_rad"])
        self.assertAlmostEqual(measured["x"]["maximum"], 5e-5)
        self.assertAlmostEqual(measured["x"]["rms"], 5e-5 / np.sqrt(2))
        b["time"] += 0.001
        with self.assertRaisesRegex(ValueError, "grids differ"):
            differences(a, b, "curved_release")

    def test_failure_and_sampling(self):
        """Reject incomplete/failed traces and sample every declared endpoint exactly."""
        self.assertFalse(assess([{"success": False}], "normal_loading", 1000, None)["complete"])
        self.assertFalse(assess([{"success": True}], "normal_loading", 1000, None)["complete"])
        for t, x in ((0.0, 0.0), (0.2, 0.003), (0.7, 0.012)):
            q, v = knots_at(t, [[0.0, 0.0], [0.2, 0.003], [0.7, 0.012]])
            self.assertAlmostEqual(q, x)
            self.assertAlmostEqual(v, 0.0)
        with tempfile.TemporaryDirectory() as folder:
            result = run_case("normal_loading", "C-EE", "cpu", Path(folder) / "partial", steps=2, reference=True)
            self.assertFalse(result["complete"])
            self.assertEqual(result["successful_steps"], 2)
            self.assertEqual(result["expected_steps"], 1000)
            comparison = summarize(Path(folder))
            self.assertFalse(comparison["reference_sensitivity"][0]["complete"])
            self.assertTrue(all(r["status"] == "NOT_CERTIFIED_INCOMPLETE" for r in comparison["refinement"]))


def test_cube_driver(test, device):
    """Build the frozen clamp and advance it without constructing an obsolete solver."""
    with tempfile.TemporaryDirectory() as folder:
        result = run_case("cube1", "C-EE", str(device), Path(folder) / "partial", steps=2)
        test.assertIsNone(result["failure"])
        test.assertEqual(result["successful_steps"], 2)
        test.assertEqual(result["expected_steps"], 1500)
        test.assertFalse(result["complete"])


add_function_test(
    TestSmallSceneTools, "test_cube_driver", test_cube_driver, devices=get_test_devices(), check_output=False
)
add_function_test(TestSmallSceneTools, "test_normal_input", test_normal_input, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
