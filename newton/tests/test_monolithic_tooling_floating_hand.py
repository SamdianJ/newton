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
        names = (
            "floating_hand_wrist_box_baseline_v1.json",
            "floating_hand_e1a_capsule_v1.json",
            "floating_hand_e1b_convex_v1.json",
            "floating_chain_e2_v1.json",
            "floating_hand_e3_fist_v1.json",
        )
        for name in names:
            self.assertEqual(load_recipe(fixtures / name)["status"], "FROZEN", name)


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


class TestChainBalance(unittest.TestCase):
    def test_projectile_conserves_momentum_and_energy(self):
        """An analytic tumbling projectile has zero momentum drift and no energy gain."""
        dt, n, g = 0.001, 300, np.array([0.0, 0.0, -9.81])
        v0, x0, omega = np.array([0.3, 0.0, 1.0]), np.array([0.0, 0.0, 0.3]), np.array([2.0, 5.0, 1.0])
        t = (np.arange(n) + 1) * dt
        body_q = np.zeros((n, 1, 7))
        body_q[:, 0, :3] = x0 + v0 * t[:, None] + 0.5 * g * t[:, None] ** 2
        angle = np.linalg.norm(omega) * t
        axis = omega / np.linalg.norm(omega)
        body_q[:, 0, 3:6] = np.sin(angle / 2)[:, None] * axis
        body_q[:, 0, 6] = np.cos(angle / 2)
        body_qd = np.zeros((n, 1, 6))
        body_qd[:, 0, :3] = v0 + g * t[:, None]
        body_qd[:, 0, 3:] = omega
        model = {
            "gravity": g.tolist(),
            "body_mass": [2.0],
            "body_com": [[0.0, 0.0, 0.0]],
            "body_inertia": [(np.eye(3) * 0.01).tolist()],
            "joint_labels": [],
            "joint_type": [],
            "joint_q_start": [0],
            "shapes": [],
        }
        frames = {
            "step": np.arange(n),
            "body_q": body_q,
            "body_qd": body_qd,
            "joint_q": np.zeros((n, 7)),
            "contact_body_wrench": np.zeros((n, 1, 6)),
        }
        recipe = {
            "phases_s": {"flight": 0.3, "settle": 0.0},
            "chain": {"spring_ke": 2.0},
            "solver": {},
            "table": {"z": 0.0},
        }
        result = fc.chain_balance(model, frames, recipe, dt)
        self.assertEqual(result["free_flight_steps"], n)
        self.assertLess(result["momentum_drift_ratio_max"], 1e-12)
        self.assertLess(result["energy_gain_ratio"], 1e-12)


def _write_run(directory, *, model, frames, scene_steps=None, dt=0.001, object_pair_shapes=()):
    """Synthetic run_floating_hand output: manifest.json, model.json and one state chunk."""
    import json  # noqa: PLC0415

    n = len(frames["body_q"])
    frames = dict(frames)
    frames.setdefault("step", np.arange(n))
    for name in ("newton_iterations", "linear_iterations", "line_search_iterations", "regularization_retries"):
        frames.setdefault(name, np.zeros(n, dtype=np.int32))
    np.savez(directory / "states-0000.npz", **frames)
    manifest = {
        "source_sha": "synthetic",
        "status": "COMPLETE",
        "counts": {"CONVERGED": n, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0},
        "planned_steps": scene_steps or n,
        "dt_s": dt,
        "chunks": ["states-0000.npz"],
        "object_pair_shapes": list(object_pair_shapes),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    base = {"palm_body": None, "object_body": None, "hand_bodies": [], "same_tree_pairs": [], "gravity": [0, 0, -9.81]}
    (directory / "model.json").write_text(json.dumps({**base, **model}))


def _box(index, body, half, label="b"):
    return {
        "index": index,
        "label": label,
        "body": body,
        "type": "BOX",
        "scale": list(half),
        "transform": [0, 0, 0, 0, 0, 0, 1],
    }


PLANE = {
    "index": 9,
    "label": "table",
    "body": -1,
    "type": "PLANE",
    "scale": [0, 0, 0],
    "transform": [0, 0, 0, 0, 0, 0, 1],
}


class TestRunCriteria(unittest.TestCase):
    def setUp(self):
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_chain_ground_penetration_is_measured(self):
        """E2 has no hand or object, yet a chain box 1 mm into the plane must be reported (review BLOCKER)."""
        body_q = np.zeros((3, 1, 7))
        body_q[:, 0, 6] = 1.0
        body_q[:, 0, 2] = [0.03, 0.02, 0.019]
        model = {
            "shapes": [_box(0, 0, (0.03, 0.02, 0.02)), PLANE],
            "body_labels": ["chain_root"],
            "body_mass": [0.1],
            "body_com": [[0, 0, 0]],
            "body_inertia": [np.eye(3).tolist()],
            "joint_labels": [],
            "joint_type": [],
            "joint_q_start": [0],
        }
        frames = {
            "body_q": body_q,
            "body_qd": np.zeros((3, 1, 6)),
            "joint_q": np.zeros((3, 7)),
            "contact_body_wrench": np.ones((3, 1, 6)),
        }
        _write_run(self.dir, model=model, frames=frames)
        recipe = {
            "scene": "E2",
            "phases_s": {"flight": 0.0, "settle": 0.003},
            "chain": {"spring_ke": 0.0},
            "table": {"z": 0.0},
            "solver": {"contact_stiffness": 1e9, "normal_smoothing": 1e-4},
        }
        out = fc.evaluate(self.dir, recipe)
        self.assertAlmostEqual(out["penetration_m"]["dynamic_table"]["max"], 0.001, places=12)
        for name in ("hand_object", "object_table", "hand_table", "self"):
            self.assertEqual(out["penetration_m"][name], "NOT_APPLICABLE", name)

    def test_root_tracking_known_offset_and_rotation(self):
        n, angle = 120, np.radians(1.5)
        body_q = np.zeros((n, 1, 7))
        body_q[:, 0, :3] = [0.002, 0.0, 0.3]
        body_q[:, 0, 3:] = [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
        target = np.tile(np.eye(4), (n, 1, 1))
        target[:, 2, 3] = 0.3
        model = {"shapes": [], "body_labels": ["palm_link"], "palm_body": 0, "body_mass": [0.4]}
        _write_run(self.dir, model=model, frames={"body_q": body_q, "palm_target": target})
        out = fc.evaluate(self.dir, {"scene": "E1a", "self_collision": None}, settle_s=0.05)
        self.assertAlmostEqual(out["root_tracking"]["position_max_m"], 0.002, places=12)
        self.assertAlmostEqual(out["root_tracking"]["angle_max_rad"], angle, places=9)
        self.assertEqual(out["root_tracking"]["from_step"], 50)

    def _grasp_run(self, rise, support):
        n = len(rise)
        body_q = np.zeros((n, 1, 7))
        body_q[:, 0, 6] = 1.0
        body_q[:, 0, 2] = 0.02 + np.asarray(rise)
        force = np.zeros((n, 1, 3))
        force[:, 0, 2] = support
        model = {
            "shapes": [_box(0, 0, (0.06, 0.02, 0.02)), PLANE],
            "body_labels": ["object"],
            "object_body": 0,
            "body_mass": [0.1],
        }
        _write_run(self.dir, model=model, frames={"body_q": body_q, "object_pair_force": force}, object_pair_shapes=[9])
        recipe = {"scene": "B1", "lift_m": 0.1, "object": {"kind": "box", "half_extents": [0.06, 0.02, 0.02]}}
        return fc.evaluate(self.dir, recipe, grasp_window=10)["grasp"]

    def test_grasp_window_positive_and_negative(self):
        """Lifted and unsupported passes the v2 measures; resting on the table shows the full weight."""
        held = self._grasp_run(np.r_[np.zeros(10), np.full(10, 0.1)], np.r_[np.full(10, 0.981), np.zeros(10)])
        self.assertTrue(held["window_complete"])
        self.assertEqual(held["max_ground_force_N"], 0.0)
        self.assertAlmostEqual(held["lift_ratio_min"], 1.0, places=12)
        self.assertGreaterEqual(held["min_com_rise_m"], 0.25 * held["object_min_edge_m"])
        dropped = self._grasp_run(np.zeros(20), np.full(20, -0.981))
        self.assertAlmostEqual(dropped["max_ground_force_N"], 0.981, places=12)  # sign-independent
        self.assertEqual(dropped["min_com_rise_m"], 0.0)

    def test_self_pairs_respect_the_exclusion_list(self):
        """Overlapping same-tree pairs count unless their bodies are excluded by label."""
        body_q = np.zeros((1, 3, 7))
        body_q[0, :, 6] = 1.0
        body_q[0, 1, 0] = 0.015  # 5 mm overlap with body 0
        body_q[0, 2, 0] = -0.018  # 2 mm overlap with body 0
        shapes = [_box(i, i, (0.01, 0.01, 0.01)) for i in range(3)]
        model = {
            "shapes": shapes,
            "body_labels": ["palm_link", "link_13.0", "link_1.0"],
            "same_tree_pairs": [[0, 1], [0, 2]],
            "body_mass": [1, 1, 1],
        }
        _write_run(self.dir, model=model, frames={"body_q": body_q})
        out = fc.evaluate(self.dir, {"scene": "E1a", "self_collision": {"exclude": [["link_13.0", "palm_link"]]}})
        self.assertEqual(out["self_pairs_checked"], 1)
        self.assertAlmostEqual(out["penetration_m"]["self"]["max"], 0.002, places=12)

    def test_chain_contact_energy_on_samples(self):
        """A box resting 1 mm into the plane stores k * area * p^2 / 2 on its bottom-face samples."""
        p, k, n = 0.001, 1e9, 5
        body_q = np.zeros((n, 1, 7))
        body_q[:, 0, 6] = 1.0
        body_q[:, 0, 2] = 0.02 - p
        model = {
            "shapes": [_box(0, 0, (0.03, 0.02, 0.02)), PLANE],
            "body_labels": ["chain_root"],
            "body_mass": [0.1],
            "body_com": [[0, 0, 0]],
            "body_inertia": [np.eye(3).tolist()],
            "joint_labels": [],
            "joint_type": [],
            "joint_q_start": [0],
            "gravity": [0, 0, -9.81],
        }
        frames = {
            "step": np.arange(n),
            "body_q": body_q,
            "body_qd": np.zeros((n, 1, 6)),
            "joint_q": np.zeros((n, 7)),
            "contact_body_wrench": np.ones((n, 1, 6)),
        }
        recipe = {
            "phases_s": {"flight": 0.0, "settle": 0.005},
            "chain": {"spring_ke": 0.0},
            "table": {"z": 0.0},
            "solver": {"contact_stiffness": k, "normal_smoothing": 1e-4},
        }
        result = fc.chain_balance(model, frames, recipe, 0.001)
        self.assertAlmostEqual(result["energy_initial_J"]["contact"], k * (0.06 * 0.04) * p * p / 2, delta=1e-9)
        self.assertLess(result["energy_gain_max_J"], 1e-12)


class TestRunnerIdentity(unittest.TestCase):
    def test_partial_run_records_identity(self):
        """A --max-steps run is PARTIAL, records SHA and recipe hash, and its frames evaluate."""
        import json  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from scripts.monolithic_reference import floating_hand_scenes as fhs  # noqa: PLC0415
        from scripts.monolithic_reference.run_floating_hand import main  # noqa: PLC0415

        if not fhs.ALLEGRO.is_dir():
            self.skipTest("Allegro asset not present")
        recipe = (
            Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures/floating_hand_e3_fist_v1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            code = main(
                [
                    "E3",
                    "--recipe",
                    str(recipe),
                    "--mount",
                    "wrist",
                    "--self-collision",
                    "off",
                    "--device",
                    "cpu",
                    "--precision",
                    "fp64",
                    "--max-steps",
                    "2",
                    "--output",
                    str(out),
                ]
            )
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "PARTIAL")
            self.assertEqual(manifest["published_steps"], 2)
            self.assertEqual(len(manifest["source_sha"]), 40)
            self.assertEqual(manifest["recipe_sha256"], __import__("hashlib").sha256(recipe.read_bytes()).hexdigest())
            criteria = fc.evaluate(out, fhs.load_recipe(recipe))
            self.assertEqual(criteria["published_steps"], 2)
            self.assertEqual(criteria["penetration_m"]["hand_table"], "NOT_APPLICABLE")


class TestPinnedGeometry(unittest.TestCase):
    def test_convex_object_is_the_pinned_hull(self):
        from scripts.monolithic_reference import floating_hand_scenes as fhs  # noqa: PLC0415

        if not fhs.CONVEX_HULL.is_file():
            self.skipTest("Pinned convex hull not present")
        points, indices = fhs.convex_hull_asset({"radius": 0.02, "half_height": 0.04})
        self.assertEqual(points.shape, (962, 3))
        self.assertEqual(len(indices), 3 * 1920)
        # Inscribed 32-gon rings: radial extent cos(pi/32) * r, axial extent half_height + r.
        np.testing.assert_allclose(
            np.abs(points).max(axis=0), [0.02, 0.02, 0.06], atol=0.02 * (1 - np.cos(np.pi / 32)) + 1e-12
        )
        with self.assertRaisesRegex(ValueError, "pinned hull"):
            fhs.convex_hull_asset({"radius": 0.03, "half_height": 0.04})

    def test_survey_adds_grasp_targets(self):
        from pathlib import Path  # noqa: PLC0415

        from scripts.monolithic_reference import floating_hand_scenes as fhs  # noqa: PLC0415
        from scripts.monolithic_reference.self_collision_survey import build  # noqa: PLC0415

        if not fhs.ALLEGRO.is_dir():
            self.skipTest("Allegro asset not present")
        recipe = (
            Path(__file__).resolve().parents[2]
            / "scripts/monolithic_reference/fixtures/floating_hand_e1a_capsule_v1.json"
        )
        model, poses = build("wrist", [recipe])
        self.assertEqual(sorted(poses), ["E1a_closed", "E1a_open_at_grasp", "fist", "open", "rest"])
        names, _, open_, closed = fhs.hand_keyframes(model, fhs.load_recipe(recipe))
        # Flex joints close past the grasp by squeeze_rad (clipped); joints kept open at the grasp are not squeezed.
        grasp = np.array([fhs.load_recipe(recipe)["grasp"]["grasp"][n] for n in names])
        moved = [i for i, n in enumerate(names) if n.startswith("joint_") and closed[i] != grasp[i]]
        self.assertTrue(moved)
        for i in moved:
            self.assertNotEqual(grasp[i], open_[i], names[i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
