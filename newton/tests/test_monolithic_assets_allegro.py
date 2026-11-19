# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pinned Allegro left-hand asset: identity, collision geometry and mounts."""

import hashlib
import json
import unittest

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.floating_hand_scenes import (
    ALLEGRO,
    HAND_BODIES,
    HAND_COLLISION_SHAPES,
    HAND_DOFS,
    add_allegro,
    verify_allegro,
)


def _require_asset(test):
    if not ALLEGRO.is_dir():
        test.skipTest(f"Allegro asset not present at {ALLEGRO}; it is pinned under asset/monolithic/data")


class TestAllegroAsset(unittest.TestCase):
    def test_locked_hashes(self):
        """Every file the task lock names is pinned with the same bytes."""
        _require_asset(self)
        root = verify_allegro()
        record = json.loads((root / "provenance.json").read_text())
        self.assertEqual(len(record["files"]), 12)
        for relative, digest in record["files"].items():
            data = (root / "wonik_allegro" / relative).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, relative)

    def test_collision_geometry(self):
        """Collision geometry is analytic: 17 boxes and 4 fingertip spheres; visual meshes do not collide."""
        _require_asset(self)
        builder = newton.ModelBuilder()
        add_allegro(builder, mount="wrist", palm_pose=wp.transform_identity())
        model = builder.finalize(device="cpu")
        collide = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        types = model.shape_type.numpy()[collide]
        self.assertEqual(int(collide.sum()), HAND_COLLISION_SHAPES)
        self.assertEqual(int(np.sum(types == newton.GeoType.BOX)), 17)
        self.assertEqual(int(np.sum(types == newton.GeoType.SPHERE)), 4)
        self.assertTrue(np.all(model.shape_type.numpy()[~collide] == newton.GeoType.MESH))

    def test_mounts(self):
        """The wrist mount is one world-rooted tree of 22 DoFs starting at Px; the floating mount has a FREE root."""
        _require_asset(self)
        for mount, joint_type, coords in (
            ("wrist", newton.JointType.PRISMATIC, 22),
            ("floating", newton.JointType.FREE, 23),
        ):
            with self.subTest(mount=mount):
                builder = newton.ModelBuilder()
                layout = add_allegro(builder, mount=mount, palm_pose=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()))
                model = builder.finalize(device="cpu")
                self.assertEqual(model.articulation_count, 1)
                self.assertEqual(model.joint_type.numpy()[layout["root_joint"]], joint_type)
                self.assertEqual(model.joint_dof_count, 22)
                self.assertEqual(model.joint_coord_count, coords)
                self.assertEqual(sum(not d.startswith("wrist_") for d in layout["dofs"]), HAND_DOFS)
                hand = [n for n in layout["bodies"] if not n.startswith("wrist_")]
                self.assertEqual(len(hand), HAND_BODIES)
                modes = model.joint_target_mode.numpy()
                self.assertTrue(np.all(modes == int(newton.JointTargetMode.POSITION_VELOCITY)))

    def test_fp32_initial_sweep(self):
        """The wrist mount stays within the FP32 swept-FK roundoff bound at its initial pose.

        The floating mount needs the PR2 FREE sweep branch and is tested there.
        """
        _require_asset(self)
        from newton._src.solvers.monolithic.articulation import _TreeKinematics  # noqa: PLC0415
        from newton._src.solvers.monolithic.contact_sweep import _ArticulationSweep  # noqa: PLC0415

        builder = newton.ModelBuilder()
        add_allegro(builder, mount="wrist", palm_pose=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
        model = builder.finalize(device="cpu")
        k = _TreeKinematics(model, dtype=wp.float32)
        sweep = _ArticulationSweep(k, np.full(model.body_count, 0.05))
        q = wp.array(model.joint_q.numpy(), dtype=wp.float32, device="cpu")
        sweep.update(q, q)
        self.assertEqual(int(sweep.status.numpy()[0]), 0)

    def test_self_collision_exclusion(self):
        """With self-collision on, the rest overlap palm_link-link_13.0 fails by name unless excluded; with the
        frozen exclusion the hand has the PR0 survey's 190 same-tree pairs minus the excluded one."""
        _require_asset(self)
        from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415
        from scripts.monolithic_reference.floating_hand_scenes import articulation_dof_count  # noqa: PLC0415

        builder = newton.ModelBuilder()
        add_allegro(builder, mount="floating", palm_pose=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
        model = builder.finalize(device="cpu")
        terms = SolverMonolithic.JointTerms(
            implicit_pd=True, limits=True, limit_width=np.full(articulation_dof_count(model), 0.01)
        )

        def solver(option):
            return SolverMonolithic(
                model, friction_velocity_scale=0.01, contact_stiffness=1e9, joint_terms=terms, self_collision=option
            )

        with self.assertRaisesRegex(ValueError, r"rest overlap.*palm_link.*link_13\.0"):
            solver({})
        run = solver({"exclude": [["palm_link", "link_13.0"]]})
        self.assertEqual(len(run._self_collision.pairs), 189)
        # Each direction samples its source: a 12 mm sphere 1984 triangles x 3, a box its face grid of at
        # most 4 mm cells, 2 triangles x 3 per cell (2 x 3 per face pair and axis pair below).
        types, scales = model.shape_type.numpy(), model.shape_scale.numpy()

        def samples(shape):
            if types[shape] == newton.GeoType.SPHERE:
                return 1984 * 3
            n = np.ceil(2 * scales[shape].astype(float) / 4e-3 * (1 - 1e-6)).astype(int)
            return 12 * (n[0] * n[1] + n[1] * n[2] + n[0] * n[2])

        expected = sum(samples(a) + samples(b) for a, b in run._self_collision.pairs)
        self.assertEqual(expected, 1214088)
        self.assertEqual(run._contact.actor_samples.count, expected)

    def test_bad_mount(self):
        with self.assertRaisesRegex(ValueError, "mount must be one of"):
            add_allegro(newton.ModelBuilder(), mount="mocap", palm_pose=wp.transform_identity())


if __name__ == "__main__":
    unittest.main(verbosity=2)
