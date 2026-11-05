# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in source-URDF FK and mass checks for the external GraspIt human hand."""

import json
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import _TreeKinematics
from newton._src.solvers.monolithic.contact_sweep import _ArticulationSweep
from scripts.monolithic_reference.graspit_hand import build_hand


class TestGraspItAsset(unittest.TestCase):
    def test_lift_axis_preserves_hand(self):
        """Translate the complete hand without changing its geometry or physical inertia."""
        asset = os.environ.get("NEWTON_GRASPIT_ASSET_DIR")
        if not asset:
            self.skipTest("Set NEWTON_GRASPIT_ASSET_DIR for the external hand asset")
        fixed = build_hand(asset)[0].finalize(device="cpu")
        lift = build_hand(asset, lift=True)[0].finalize(device="cpu")
        self.assertEqual(lift.joint_dof_count, 21)
        self.assertEqual(lift.articulation_count, 1)
        self.assertEqual(lift.joint_type.numpy()[0], newton.JointType.PRISMATIC)
        np.testing.assert_array_equal(lift.body_mass.numpy(), fixed.body_mass.numpy())
        np.testing.assert_array_equal(lift.body_inertia.numpy(), fixed.body_inertia.numpy())
        q = np.linspace(0, 0.2, 20)
        states = [fixed.state(), lift.state()]
        states[0].joint_q.assign(q)
        states[1].joint_q.assign(np.r_[0.08, q])
        for model, state in zip((fixed, lift), states, strict=True):
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected = states[0].body_q.numpy()
        expected[:, 1] += 0.08
        np.testing.assert_allclose(states[1].body_q.numpy(), expected, atol=2e-7)

    def test_fp32_initial_sweep(self):
        """Accept the hand's initial sweep without exhausting the native FK roundoff bound."""
        asset = os.environ.get("NEWTON_GRASPIT_ASSET_DIR")
        if not asset:
            self.skipTest("Set NEWTON_GRASPIT_ASSET_DIR for the external hand asset")
        for lift in (False, True):
            with self.subTest(lift=lift):
                builder, _ = build_hand(asset, device="cpu", lift=lift)
                model = builder.finalize(device="cpu")
                k = _TreeKinematics(model, dtype=wp.float32)
                sweep = _ArticulationSweep(k, np.full(model.body_count, 0.2))
                q = wp.array(model.joint_q.numpy(), dtype=wp.float32, device="cpu")
                sweep.update(q, q)
                self.assertEqual(int(sweep.status.numpy()[0]), 0)

    def test_source_fk_and_mass(self):
        """Preserve the source joint tree and derive physical inertia from closed proxies."""
        import trimesh

        asset = os.environ.get("NEWTON_GRASPIT_ASSET_DIR")
        if not asset:
            self.skipTest("Set NEWTON_GRASPIT_ASSET_DIR for the external hand asset")
        asset = Path(asset)
        builder, manifest = build_hand(asset, device="cpu")
        model = builder.finalize(device="cpu")
        self.assertEqual(model.joint_dof_count, 20)
        self.assertEqual(np.count_nonzero(model.joint_parent.numpy() == -1), 1)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.shape_count, 32)
        self.assertEqual(np.count_nonzero(model.body_mass.numpy()), 16)
        self.assertFalse(
            np.any(
                model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
            )
        )
        rows = json.loads((asset / "manifest.json").read_text())["meshes"]
        for row in rows:
            name = "base_link" if row["link"] == "palm" else row["link"] + "_joint"
            body = model.body_label.index("HumanHand20DOF/" + name)
            mass = model.body_mass.numpy()[body]
            mesh = trimesh.load_mesh(asset / "derived/collision" / (row["link"] + ".obj"))
            shapes = [
                i
                for i, b in enumerate(model.shape_body.numpy())
                if b == body and not model.shape_flags.numpy()[i] & int(newton.ShapeFlags.VISIBLE)
            ]
            self.assertEqual(len(shapes), 1)
            pose = model.shape_transform.numpy()[shapes[0]]
            matrix = trimesh.transformations.quaternion_matrix(pose[[6, 3, 4, 5]])
            matrix[:3, 3] = pose[:3]
            mesh.apply_transform(matrix)
            mesh.density = manifest["density_kg_m3"]
            self.assertTrue(mesh.is_volume)
            self.assertAlmostEqual(mass, mesh.volume * manifest["density_kg_m3"], delta=1e-7)
            np.testing.assert_allclose(model.body_com.numpy()[body], mesh.center_mass, atol=1e-7)
            np.testing.assert_allclose(model.body_inertia.numpy()[body], mesh.moment_inertia, atol=1e-10, rtol=1e-5)
            self.assertGreater(np.linalg.eigvalsh(model.body_inertia.numpy()[body]).min(), 0)

        def rotation(axis, angle):
            axis = np.asarray(axis) / np.linalg.norm(axis)
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross

        xml = ET.parse(asset / "upstream/urdf/HumanHand20DOF.urdf").getroot()
        for fraction in (0.0, 0.2, 0.55):
            q = fraction * model.joint_limit_upper.numpy()
            state = model.state()
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            frames = {"base_link": np.eye(4)}
            pending = xml.findall("joint")
            while pending:
                before = len(pending)
                for joint in pending[:]:
                    parent = joint.find("parent").get("link")
                    if parent not in frames:
                        continue
                    origin = joint.find("origin")
                    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    local = np.eye(4)
                    local[:3, 3] = xyz
                    local[:3, :3] = (
                        rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
                    )
                    if joint.get("type") == "revolute":
                        j = model.joint_label.index("HumanHand20DOF/" + joint.get("name"))
                        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                        local[:3, :3] = local[:3, :3] @ rotation(axis, q[model.joint_q_start.numpy()[j]])
                    frames[joint.find("child").get("link")] = frames[parent] @ local
                    pending.remove(joint)
                self.assertLess(len(pending), before)
            for b, label in enumerate(model.body_label):
                expected = frames[label.rsplit("/", 1)[-1]]
                pose = state.body_q.numpy()[b]
                np.testing.assert_allclose(pose[:3], expected[:3, 3], atol=2e-7)
                actual = np.asarray(wp.quat_to_matrix(wp.quat(pose[3:]))).reshape(3, 3)
                np.testing.assert_allclose(actual, expected[:3, :3], atol=2e-6)


if __name__ == "__main__":
    unittest.main()
