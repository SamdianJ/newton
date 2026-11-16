# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline floating-hand criteria against analytic geometry."""

import unittest

import numpy as np

from scripts.monolithic_reference import floating_hand_criteria as fc


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class TestConvexDepth(unittest.TestCase):
    def test_spheres(self):
        a = fc.point(0.02).placed(np.eye(3), (0, 0, 0))
        b = fc.point(0.03).placed(np.eye(3), (0.04, 0, 0))
        self.assertAlmostEqual(fc.signed_distance(a, b), -0.01, places=14)
        c = fc.point(0.03).placed(np.eye(3), (0.06, 0, 0))
        self.assertAlmostEqual(fc.signed_distance(a, c), 0.01, places=14)

    def test_capsules_crossed_and_parallel(self):
        """Segment-segment distance minus both radii, crossed and parallel."""
        a = fc.segment(0.05, 0.01).placed(np.eye(3), (0, 0, 0))
        b = fc.segment(0.05, 0.01).placed(_rotation((1, 0, 0), np.pi / 2), (0.015, 0, 0))
        self.assertAlmostEqual(fc.signed_distance(a, b), 0.015 - 0.02, places=13)
        c = fc.segment(0.05, 0.01).placed(np.eye(3), (0.03, 0, 0.02))
        self.assertAlmostEqual(fc.signed_distance(a, c), 0.01, places=13)
        # Coincident axes: cores intersect, depth is the sum of radii.
        d = fc.segment(0.05, 0.01).placed(np.eye(3), (0, 0, 0.02))
        self.assertAlmostEqual(fc.penetration(a, d), 0.02, places=13)

    def test_sphere_box_face_and_corner(self):
        b = fc.box((0.02, 0.03, 0.04)).placed(np.eye(3), (0, 0, 0))
        face = fc.point(0.01).placed(np.eye(3), (0.025, 0, 0))
        self.assertAlmostEqual(fc.penetration(face, b), 0.005, places=13)
        corner = fc.point(0.01).placed(np.eye(3), (0.025, 0.035, 0.045))
        self.assertAlmostEqual(fc.signed_distance(corner, b), np.sqrt(3) * 0.005 - 0.01, places=13)
        inside = fc.point(0.0).placed(np.eye(3), (0.019, 0, 0))
        self.assertAlmostEqual(fc.penetration(inside, b), 0.001, places=13)

    def test_capsule_box(self):
        """A capsule lying on a box top penetrates by its radius minus its axis height."""
        b = fc.box((0.1, 0.1, 0.02)).placed(np.eye(3), (0, 0, 0))
        cap = fc.segment(0.04, 0.02).placed(_rotation((0, 1, 0), np.pi / 2), (0, 0, 0.039))
        self.assertAlmostEqual(fc.penetration(cap, b), 0.001, places=12)

    def test_box_pairs_match_rigid_stack_sat(self):
        from scripts.monolithic_reference.run_rigid_stack import HALF_EXTENT, box_penetration  # noqa: PLC0415

        rng = np.random.default_rng(3)
        for _ in range(20):
            ra, rb = (
                _rotation(rng.normal(size=3), rng.uniform(0, np.pi)),
                _rotation(rng.normal(size=3), rng.uniform(0, np.pi)),
            )
            ca, cb = np.zeros(3), rng.uniform(-0.03, 0.03, size=3)
            a = fc.box((HALF_EXTENT,) * 3).placed(ra, ca)
            b = fc.box((HALF_EXTENT,) * 3).placed(rb, cb)
            # Lift both boxes clear of the ground term box_penetration also reports.
            lift = np.array([0.0, 0.0, 1.0])
            _, expected = box_penetration(np.stack((ca + lift, cb + lift)), np.stack((ra, rb)))
            self.assertAlmostEqual(fc.penetration(a, b), expected, places=12)

    def test_hull_of_box_equals_box(self):
        corners = fc.box((0.02, 0.03, 0.04)).vertices
        h = fc.hull(corners).placed(_rotation((1, 1, 0), 0.3), (0.01, 0, 0))
        b = fc.box((0.02, 0.03, 0.04)).placed(_rotation((1, 1, 0), 0.3), (0.01, 0, 0))
        probe = fc.segment(0.05, 0.01).placed(_rotation((0, 1, 1), 0.7), (0.03, 0.01, 0.02))
        self.assertAlmostEqual(fc.signed_distance(probe, h), fc.signed_distance(probe, b), places=12)

    def test_random_sphere_box_matches_analytic_sdf(self):
        """GJK and SAT terminate and agree with the exact box SDF for spheres in all regions."""
        rng = np.random.default_rng(11)
        half = np.array([0.0098, 0.0138, 0.027])
        for _ in range(300):
            rotation, center = _rotation(rng.normal(size=3), rng.uniform(0, np.pi)), rng.uniform(-0.05, 0.05, size=3)
            b = fc.box(half).placed(rotation, (0.0, 0.0, 0.0))
            radius = rng.uniform(0.0, 0.012)
            s = fc.point(radius).placed(np.eye(3), center)
            local = np.abs(rotation.T @ center) - half
            sdf = np.linalg.norm(np.maximum(local, 0)) + min(local.max(), 0)
            self.assertAlmostEqual(fc.signed_distance(s, b), sdf - radius, places=11)

    def test_plane(self):
        cap = fc.segment(0.04, 0.02).placed(_rotation((0, 1, 0), np.pi / 2), (0, 0, 0.019))
        self.assertAlmostEqual(fc.plane_signed_distance(cap), -0.001, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
