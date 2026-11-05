# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check screw-force work, rotation unwrapping, disengagement and hollow cap geometry."""

import unittest

import numpy as np

from scripts.monolithic_reference.bottle_cap import CapThread, cap_geometry
from scripts.monolithic_reference.plastic_bottle import bottle_mesh
from scripts.monolithic_reference.run_bottle_cap import sample_penetration


def pose_at(theta, y):
    """Return a cap pose rotated about the fixed neck axis."""
    return np.array([0, y, 0, 0, np.sin(theta / 2), 0, np.cos(theta / 2)])


class TestBottleCap(unittest.TestCase):
    def test_thread_work_and_axial_load(self):
        """Match the elastic energy gradient and oppose a pull without prescribing rise."""
        thread = CapThread(0.078, theta=0.7)
        h = thread.lead / (2 * np.pi)
        pose = pose_at(thread.theta, thread.seated_y + h * thread.theta + 0.0001)
        wrench = thread.wrench(pose, np.zeros(6))

        def energy(y, theta):
            return 0.5 * thread.stiffness * (y - thread.seated_y - h * theta) ** 2

        eps = 1e-7
        dy = (energy(pose[1] + eps, thread.theta) - energy(pose[1] - eps, thread.theta)) / (2 * eps)
        dq = (energy(pose[1], thread.theta + eps) - energy(pose[1], thread.theta - eps)) / (2 * eps)
        np.testing.assert_allclose(wrench[[1, 4]], -np.array([dy, dq]), rtol=1e-7)
        self.assertLess(wrench[1], 0)
        self.assertGreater(wrench[4], 0)
        velocity = np.array([0, h * 0.5, 0, 0, 0.5, 0])
        self.assertAlmostEqual(float(wrench @ velocity), 0.0, places=14)
        self.assertLess(float((thread.wrench(pose, velocity) - wrench) @ velocity), 0)

    def test_unwrap_and_disengagement(self):
        """Require rotation and axial progress; keep detached forces zero on return."""
        thread = CapThread(0.078)
        for angle in np.linspace(0, 4 * np.pi + 0.02, 150):
            thread.observe(pose_at(angle, thread.seated_y))
        self.assertFalse(thread.released)  # Rotation alone with a blocked cap cannot release.
        self.assertAlmostEqual(thread.theta, 4 * np.pi + 0.02)
        thread.observe(pose_at(thread.theta, thread.seated_y + 0.006))
        self.assertTrue(thread.released)
        for angle in np.linspace(thread.theta, 0, 150):
            thread.observe(pose_at(angle, thread.seated_y))
        np.testing.assert_array_equal(thread.wrench(pose_at(0, 1), np.ones(6)), 0)
        self.assertTrue(thread.released)

    def test_invalid_observation_is_atomic(self):
        """Reject nonfinite or ambiguous observations without changing thread state."""
        thread = CapThread(0.078)
        for pose in (np.zeros(7), np.full(7, np.nan), pose_at(np.pi, 0.078)):
            before = vars(thread).copy()
            with self.assertRaises(ValueError):
                thread.observe(pose)
            self.assertEqual(before, vars(thread))

    def test_hollow_cap_geometry_and_mass(self):
        """Verify manifold orientation, material volume, positive inertia and neck clearance."""
        vertices, faces, mass, inertia, y = cap_geometry()
        triangles = vertices[faces]
        volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        self.assertGreater(volume, 0)
        self.assertAlmostEqual(volume * 950 / mass, np.sin(2 * np.pi / 48) / (2 * np.pi / 48), places=12)
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        self.assertTrue(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2))
        self.assertTrue(np.all(inertia > 0))
        self.assertTrue(np.all(inertia.sum() / 2 - inertia > 0))
        points, faces = bottle_mesh()
        samples = np.concatenate((points, points[faces].mean(1)))
        self.assertEqual(sample_penetration(pose_at(0, y), samples, vertices), 0)
        self.assertGreater(sample_penetration(pose_at(0, y - 0.003), samples, vertices), 0.0005)


if __name__ == "__main__":
    unittest.main()
