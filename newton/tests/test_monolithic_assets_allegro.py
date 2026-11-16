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
        """The wrist mount is one world-rooted tree of 22 DoFs; the floating mount has a FREE root."""
        _require_asset(self)
        for mount, joint_type, coords in (
            ("wrist", newton.JointType.FIXED, 22),
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

    def test_bad_mount(self):
        with self.assertRaisesRegex(ValueError, "mount must be one of"):
            add_allegro(newton.ModelBuilder(), mount="mocap", palm_pose=wp.transform_identity())


if __name__ == "__main__":
    unittest.main(verbosity=2)
