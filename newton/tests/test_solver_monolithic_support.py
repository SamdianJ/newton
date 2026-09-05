# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the P1Q3 calibration harness against measurable sampling contracts."""

import unittest

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.calibrate_p1q3 import measure_motion, measure_static, refined_tetrahedron


def test_uniform_plane_refinement(test, device):
    """Preserve geometry and uniform-plane total force under boundary refinement."""
    rows = [measure_static(device, level=level, case="plane") for level in range(3)]
    test.assertEqual([row["boundary_face_count"] for row in rows], [4, 16, 64])
    test.assertEqual([row["tet_count"] for row in rows], [1, 8, 64])
    forces = np.array([row["normal_force_n"] for row in rows])
    test.assertLessEqual(float(np.max(np.abs(forces / forces[-1] - 1.0))), 0.05)
    np.testing.assert_allclose(forces, 0.176, rtol=2e-5)
    for row in rows:
        test.assertEqual(row["evaluation_status"], 0)
        test.assertLess(row["contact_sign_error"], 1e-4)
        test.assertGreater(row["active_sample_count"], 0)
    for level in range(3):
        points, tets = refined_tetrahedron(level)
        determinant = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
        test.assertTrue(np.all(determinant > 0.0))
        test.assertAlmostEqual(float(determinant.sum() / 6.0), 0.04**3 / 6.0, delta=1e-11)


def test_sharp_feature_envelope(test, device):
    """Distinguish an inter-sample miss from a broad-feature positive control."""
    sharp = measure_static(device, level=0, case="sharp_box")
    broad = measure_static(device, level=0, case="broad_box")
    test.assertGreater(sharp["witness_penetration_m"], 0.0)
    test.assertEqual(sharp["active_sample_count"], 0)
    test.assertEqual(sharp["normal_force_n"], 0.0)
    test.assertEqual(sharp["sampling_classification"], "UNSUPPORTED_SAMPLING_MISS")
    test.assertGreater(broad["active_sample_count"], 0)
    test.assertGreater(broad["normal_force_n"], 0.0)
    test.assertEqual(broad["evaluation_status"], 0)


def test_broad_curvature_control(test, device):
    """Require real contact and resolved motion in broad edge/vertex controls."""
    for case in ("wide_edge_160", "wide_vertex_160"):
        row = measure_static(device, level=0, case=case)
        test.assertGreater(row["active_sample_count"], 0)
        test.assertGreater(row["normal_force_n"], 0)
        motion = measure_motion(device, level=0, case=case, steps=2, dt=0.001)
        test.assertTrue(motion["all_steps_success"])
        test.assertFalse(any(row["sampling_miss"] for row in motion["trace"]))
        test.assertGreater(
            np.linalg.norm(motion["trace"][-1]["soft_volume_mean_displacement_m"]), float(np.spacing(np.float32(0.04)))
        )


class TestMonolithicSupport(unittest.TestCase):
    """Verify CPU and CUDA measurement behavior without claiming unsupported cases pass."""


for function in (test_uniform_plane_refinement, test_sharp_feature_envelope, test_broad_curvature_control):
    add_function_test(TestMonolithicSupport, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
