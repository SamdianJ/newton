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


class TestFeatureInventory(unittest.TestCase):
    def test_classifies_link_capsule_and_free_capsule(self):
        """A link capsule against the static plane is an F1 hit; a FREE capsule only becomes a new collider."""
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415
        from scripts.monolithic_reference.feature_inventory import inventory  # noqa: PLC0415

        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(density=0, has_particle_collision=False)
        link = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)), label="link")
        joint = builder.add_joint_revolute(-1, link, axis=(0, 1, 0), label="hinge")
        builder.add_articulation([joint])
        builder.add_shape_capsule(link, radius=0.01, half_height=0.02, cfg=cfg, label="link_capsule")
        body = builder.add_link(
            xform=wp.transform((0.2, 0, 0.1), wp.quat_identity()), mass=0.1, inertia=wp.diag(wp.vec3(1e-4))
        )
        builder.add_articulation([builder.add_joint_free(body)])
        builder.add_shape_capsule(body, radius=0.01, half_height=0.02, cfg=cfg, label="free_capsule")
        builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg, label="ground")
        result = inventory(builder.finalize(device="cpu"))
        self.assertEqual([hit["link_shape"] for hit in result["f1"]], ["link_capsule"])
        self.assertEqual(sorted(s["shape"] for s in result["capsule_convex"]), ["free_capsule", "link_capsule"])


class TestFloatingHandRecipes(unittest.TestCase):
    def test_wrist_palm_pose_matches_newton_fk(self):
        """The palm target formula used for the floating root equals Newton FK of the wrist mount."""
        import warp as wp  # noqa: PLC0415

        import newton  # noqa: PLC0415
        from scripts.monolithic_reference import floating_hand_scenes as fhs  # noqa: PLC0415

        if not fhs.ALLEGRO.is_dir():
            self.skipTest("Allegro asset not present")
        rng = np.random.default_rng(5)
        mount = np.eye(4)
        mount[:3, :3], mount[:3, 3] = _rotation(rng.normal(size=3), 1.1), (0.1, -0.2, 0.3)
        attach = np.eye(4)
        attach[:3, :3], attach[:3, 3] = _rotation(rng.normal(size=3), 0.4), (0.02, 0.0, 0.05)
        builder = newton.ModelBuilder()
        layout = fhs.add_allegro(builder, mount="wrist", palm_pose=fhs._transform(mount), attach=fhs._transform(attach))
        model = builder.finalize(device="cpu")
        q = model.joint_q.numpy()
        wrist = np.r_[rng.uniform(-0.2, 0.2, 3), rng.uniform(-1.0, 1.0, 3)]
        q[:6] = wrist
        state = model.state()
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        palm = state.body_q.numpy()[layout["bodies"]["palm_link"]].astype(np.float64)
        expected = fhs.wrist_palm_matrix(mount, attach, wrist)
        np.testing.assert_allclose(palm[:3], expected[:3, 3], atol=2e-6)
        np.testing.assert_allclose(fc.quaternion_matrix(palm[3:]), expected[:3, :3], atol=2e-6)
        del wp

    def test_recipe_errors_name_the_field(self):
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from scripts.monolithic_reference.floating_hand_scenes import load_recipe  # noqa: PLC0415

        base = {
            "schema": "p9d-floating-hand/v1",
            "scene": "E3",
            "mount": "floating",
            "dt_s": 0.001,
            "phases_s": {"close": 1.0, "hold": 1.0},
            "solver": {},
            "joint_terms": {},
        }
        cases = {
            "phases_s.close": {"phases_s": {"close": 0.0015, "hold": 1.0}},
            "phases_s must list": {"phases_s": {"hold": 1.0, "close": 1.0}},
            "scene must be": {"scene": "E9"},
            "mount must be": {"mount": "mocap"},
            "dt_s must be": {"dt_s": -1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            for message, change in cases.items():
                path = Path(tmp) / "r.json"
                path.write_text(json.dumps({**base, **change}))
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_recipe(path)
            path.write_text(json.dumps(base))
            self.assertEqual(load_recipe(path)["scene"], "E3")

    def test_frozen_recipes_load(self):
        from pathlib import Path  # noqa: PLC0415

        from scripts.monolithic_reference.floating_hand_scenes import load_recipe  # noqa: PLC0415

        fixtures = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures"
        names = sorted(p.name for p in fixtures.glob("floating_*_v1.json"))
        self.assertEqual(len(names), 5)
        for name in names:
            load_recipe(fixtures / name)


class TestFeatureOffTrace(unittest.TestCase):
    def test_digest_detects_one_bit_and_compare_reports_first_step(self):
        from scripts.monolithic_reference.feature_off_trace import _digest, compare  # noqa: PLC0415

        a = np.linspace(0.0, 1.0, 17)
        b = a.copy()
        b.view(np.uint64)[5] ^= np.uint64(1)
        self.assertNotEqual(_digest([("x", a)]), _digest([("x", b)]))
        self.assertEqual(_digest([("x", a)]), _digest([("x", a.copy())]))
        record = {"state": "s", "forces": "f", "stats": {"status": "SUCCESS"}, "error": None}
        changed = dict(record, forces="g")
        x = {"scenes": {"k": {"status": "TRACED", "records": [record, record]}, "m": {"status": "NOT_RUN"}}}
        y = {"scenes": {"k": {"status": "TRACED", "records": [record, changed]}, "m": {"status": "NOT_RUN"}}}
        result = compare(x, y)
        self.assertEqual(result["k"], {"verdict": "DIFFERENT", "first_step": 1, "fields": ["forces"]})
        self.assertEqual(result["m"]["verdict"], "MISSING")
        self.assertEqual(compare(x, x)["k"]["verdict"], "IDENTICAL")
