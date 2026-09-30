# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-asset G5 checks; CPU SDF remains NOT_REQUIRED."""

import json
import os
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import eval_shape_sdf
from newton.examples.softbody.monolithic_sharpa_assets import build_contact_hand, load_hand_ball
from newton.examples.softbody.sharpa_close import load_fixture
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query_frames(
    points: wp.array[wp.vec3],
    shapes: wp.array[int],
    body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_q: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    types: wp.array[int],
    sdf_indices: wp.array[int],
    table: wp.array[TextureSDFData],
    out: wp.array2d[float],
):
    i = wp.tid()
    s = shapes[i]
    X = wp.transform_multiply(body_q[body[s]], shape_q[s])
    world = wp.transform_point(X, points[i])
    local = wp.transform_point(wp.transform_inverse(X), world)
    _, a, na = eval_shape_sdf(types[s], scales[s], points[i], sdf_indices[s], table)
    _, b, nb = eval_shape_sdf(types[s], scales[s], local, sdf_indices[s], table)
    out[i, 0] = wp.abs(a - b)
    out[i, 1] = wp.length(wp.normalize(na) - wp.normalize(nb))
    out[i, 2] = wp.length(nb)


def assets(test):
    source = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
    contact = os.environ.get("NEWTON_SHARPA_CONTACT_DIR")
    if not source or not contact:
        test.skipTest("Set NEWTON_SHARPA_ASSET_DIR and NEWTON_SHARPA_CONTACT_DIR for real G5 assets")
    return Path(source), Path(contact)


def test_import(test, device):
    """Check every collision reference, imported mass and independent URDF FK."""
    source, contact = assets(test)
    parameters = load_fixture(Path(__file__).parents[1] / "examples/softbody/sharpa_g1h.json")
    ball = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/soft_ball/ball_r1.npz"
    m, manifest = load_hand_ball(
        source, contact, ball, device=device, parameters=parameters, position=(0.025, -0.03, 0.19)
    )
    test.assertEqual((m.body_count, m.joint_dof_count, m.shape_count), (33, 22, 54))
    test.assertEqual(len(manifest["collision_mapping"]), 26)
    flags = m.shape_flags.numpy()
    participating = np.flatnonzero(flags & int(newton.ShapeFlags.COLLIDE_PARTICLES))
    test.assertEqual(len(participating), 26 if wp.get_device(device).is_cuda else 0)
    test.assertFalse(np.any(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))
    xml = ET.parse(source / "left_sharpa_wave.urdf").getroot()
    for i, label in enumerate(m.body_label):
        mass = float(xml.find(f"link[@name='{label.rsplit('/', 1)[-1]}']/inertial/mass").get("value"))
        test.assertAlmostEqual(float(m.body_mass.numpy()[i]), mass, delta=max(1e-9, mass * 1e-6))

    # Independent homogeneous-matrix FK from source origins and joint axes.
    def rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    joints = xml.findall("joint")
    names = manifest["joint_names"]
    for fraction in (0.0, 0.25, 0.5):
        q = fraction * (m.joint_limit_lower.numpy() + m.joint_limit_upper.numpy())
        state = m.state()
        state.joint_q.assign(q)
        newton.eval_fk(m, state.joint_q, state.joint_qd, state)
        frames = {"left_hand_C_MC": np.eye(4)}
        pending = list(joints)
        while pending:
            progress = False
            for joint in pending[:]:
                parent = joint.find("parent").get("link")
                if parent not in frames:
                    continue
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                T = np.eye(4)
                T[:3, 3] = xyz
                T[:3, :3] = rotation([0, 0, 1], rpy[2]) @ rotation([0, 1, 0], rpy[1]) @ rotation([1, 0, 0], rpy[0])
                if joint.get("type") == "revolute":
                    T[:3, :3] = T[:3, :3] @ rotation(
                        np.fromstring(joint.find("axis").get("xyz"), sep=" "), q[names.index(joint.get("name"))]
                    )
                frames[joint.find("child").get("link")] = frames[parent] @ T
                pending.remove(joint)
                progress = True
            test.assertTrue(progress, "Disconnected URDF joint graph")
        for i, label in enumerate(m.body_label):
            X = wp.transform(state.body_q.numpy()[i, :3], state.body_q.numpy()[i, 3:])
            expected = frames[label.rsplit("/", 1)[-1]]
            np.testing.assert_allclose(wp.transform_get_translation(X), expected[:3, 3], atol=2e-7)
            np.testing.assert_allclose(
                np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(X))).reshape(3, 3), expected[:3, :3], atol=2e-6
            )
        if wp.get_device(device).is_cuda:
            pipeline = MonolithicCollisionPipeline(m, soft_contact_gap=0.002)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            test.assertEqual(pipeline.soft_contact_pair_count, 48 * 26)
            test.assertTrue(np.all(m._shape_sdf_index.numpy()[participating] >= 0))
            rows = json.loads((contact / "manifest.json").read_text())["meshes"]
            by_source = {r["source"]: r for r in rows}
            points = []
            shapes = []
            for ref in manifest["collision_mapping"]:
                path = contact / (Path(by_source[ref["source"]]["derived"]).stem + "_probes.npz")
                probes = np.load(path)["points"][::128]
                points.extend(probes)
                shapes.extend([ref["shape"]] * len(probes))
            out = wp.empty((len(points), 3), dtype=float, device=device)
            wp.launch(
                _query_frames,
                len(points),
                [
                    wp.array(points, dtype=wp.vec3, device=device),
                    wp.array(shapes, dtype=int, device=device),
                    m.shape_body,
                    state.body_q,
                    m.shape_transform,
                    m.shape_scale,
                    m.shape_type,
                    m._shape_sdf_index,
                    m._texture_sdf_data,
                    out,
                ],
                device=device,
            )
            values = out.numpy()
            test.assertTrue(np.isfinite(values).all())
            test.assertLess(float(values[:, 0].max()), 1e-7)
            test.assertLess(float(values[:, 1].max()), 0.001)
            test.assertTrue(np.all(values[:, 2] > 0))


def test_unvalidated_resolution(test, device):
    """Reject an SDF resolution that failed the frozen source-probe gate."""
    source, contact = assets(test)
    parameters = load_fixture(Path(__file__).parents[1] / "examples/softbody/sharpa_g1h.json")
    with test.assertRaisesRegex(ValueError, "Unvalidated SDF"):
        build_contact_hand(source, contact, device=device, parameters=parameters, resolution=32)


class TestMonolithicSharpaAssets(unittest.TestCase):
    pass


for device in get_test_devices():
    for name, fn in (("test_import", test_import), ("test_unvalidated_resolution", test_unvalidated_resolution)):
        add_function_test(TestMonolithicSharpaAssets, name, fn, devices=[device])


if __name__ == "__main__":
    unittest.main()
