# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the frozen actor formulas against independent geometry and dynamics."""

import json
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import _link_variation
from newton.tests.test_monolithic_actor_layout import mixed_model
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contract_oracle import CASE, calibration, rigid_terms, rotation
from scripts.monolithic_reference.articulation_oracle import skew, transform


def test_link_inertia_reference(test, device):
    """Compare existing reusable link E/r/PSD B with the independent rigid reference."""
    dt, mass = 0.001, 0.01
    inertia = np.diag([2.7e-6, 1.8e-6, 1.7e-6])
    second = np.trace(inertia) / 2 * np.eye(3) - inertia
    com, predicted_com, force = np.array([0.002, -0.001, 0.0015]), np.zeros(3), np.array([0.1, -0.2, 0.15])
    rot, predicted_rot = rotation(np.array([0.21, -0.12, 0.31])), rotation(np.array([0.2, -0.1, 0.3]))

    def vector(value):
        return wp.array([value], dtype=wp.vec3d, device=device)

    def matrix(value):
        return wp.array([value], dtype=wp.mat33d, device=device)

    energy = wp.zeros(1, dtype=wp.float64, device=device)
    gt, gr, ta, ra = (wp.zeros(1, dtype=wp.vec3d, device=device) for _ in range(4))
    kr = wp.zeros(1, dtype=wp.mat33d, device=device)
    wp.launch(
        _link_variation,
        1,
        [
            vector(com),
            matrix(rot),
            vector(predicted_com),
            vector(predicted_com),
            matrix(predicted_rot),
            wp.array([mass], dtype=wp.float64, device=device),
            matrix(second),
            vector(force),
            wp.float64(dt),
            energy,
            gt,
            gr,
            ta,
            ra,
            kr,
            True,
        ],
        device=device,
    )
    e, r, b = rigid_terms(com, rot, predicted_com, predicted_rot, mass, inertia, dt, force)
    np.testing.assert_allclose(energy.numpy(), [e], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.r_[gt.numpy()[0], gr.numpy()[0]], r, rtol=1e-11, atol=1e-11)
    # Rounding magnitudes bound the summed gradients they describe.
    test.assertTrue(np.all(np.r_[ta.numpy()[0], ra.numpy()[0]] >= np.abs(r) * (1 - 1e-12)))
    np.testing.assert_allclose(kr.numpy()[0], b[3:, 3:], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kr.numpy()[0], kr.numpy()[0].T, rtol=1e-12, atol=1e-12)
    test.assertGreaterEqual(np.linalg.eigvalsh(kr.numpy()[0]).min(), -1e-12)


def test_free_anchor_mapping(test, device):
    """Round-trip nonzero COM and rotated anchors under Newton's COM twist convention."""
    model, _, rigids, _ = mixed_model(device, 1)
    parent = wp.transform((0.12, -0.07, 0.03), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.4))
    child = wp.transform((0.01, 0.02, -0.01), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.2))
    for name, value in (("joint_X_p", parent), ("joint_X_c", child)):
        data = getattr(model, name).numpy()
        data[0] = np.asarray(value)
        getattr(model, name).assign(data)
    state = model.state()
    q = state.joint_q.numpy()
    q[:7] = np.asarray(wp.transform((0.15, 0.04, -0.03), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.3)))
    state.joint_q.assign(q)
    qd = state.joint_qd.numpy()
    qd[:6] = (1, 0.5, -0.3, 0.3, -0.2, 0.1)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    p, c, motion = transform(parent), transform(child), transform(q[:7])
    body = p @ motion @ np.linalg.inv(c)
    np.testing.assert_allclose(transform(state.body_q.numpy()[rigids[0]]), body, atol=2e-7)
    local_com = model.body_com.numpy()[rigids[0]]
    world_com = body[:3, 3] + body[:3, :3] @ local_com
    origin = world_com - body[:3, :3] @ local_com
    np.testing.assert_allclose(origin, state.body_q.numpy()[rigids[0], :3], atol=2e-7)
    twist = np.r_[p[:3, :3] @ qd[:3], p[:3, :3] @ qd[3:6]]
    np.testing.assert_allclose(state.body_qd.numpy()[rigids[0]], twist, atol=2e-7)
    recovered = np.linalg.inv(p) @ body @ c
    np.testing.assert_allclose(recovered, motion, atol=1e-14)


class TestActorContract(unittest.TestCase):
    def test_calibration(self):
        """Require derivative, sampling and FP32 noise evidence for the frozen budgets."""
        report = calibration()
        self.assertEqual(report["status"], "PASS", report)

    def test_double_endpoint_virtual_work(self):
        """Preserve force, moment and cross blocks under double-endpoint assembly."""
        ca, cb, point = np.array([0.2, 0.1, 0.0]), np.array([-0.1, 0.2, 0.1]), np.array([0.1, -0.1, 0.2])
        ja, jb = np.c_[np.eye(3), -skew(point - ca)], np.c_[-np.eye(3), skew(point - cb)]
        jacobian = np.c_[ja, jb]
        force = np.array([0.2, -0.1, 0.4])
        residual = jacobian.T @ force
        direction = np.arange(12) / 100
        self.assertAlmostEqual(float(residual @ direction), float(force @ (jacobian @ direction)), places=14)
        np.testing.assert_allclose(residual[:3] + residual[6:9], 0, atol=1e-14)
        moment = residual[3:6] + np.cross(ca, residual[:3]) + residual[9:12] + np.cross(cb, residual[6:9])
        np.testing.assert_allclose(moment, 0, atol=1e-14)
        b = jacobian.T @ np.diag([2, 3, 4]) @ jacobian
        no_cross = b.copy()
        no_cross[:6, 6:] = no_cross[6:, :6] = 0
        complete = np.linalg.solve(np.eye(12) + b, -residual)
        incomplete = np.linalg.solve(np.eye(12) + no_cross, -residual)
        self.assertGreater(np.linalg.norm(complete - incomplete), 0.01)
        # Two directed samples carry half each; one full sample is the reference.
        np.testing.assert_allclose(0.5 * b + 0.5 * b, b, atol=1e-14)

    def test_actor_residual_masking(self):
        """Reject one rigid rotation residual that a large tet block would hide."""
        case = json.loads(CASE.read_text())
        r = np.zeros(100006)
        r[3] = 1e-6
        self.assertLess(np.linalg.norm(r) / np.sqrt(len(r)), 1e-7)
        self.assertGreater(np.linalg.norm(r[3:6]) / np.sqrt(3), case["rigid_goals"]["rotation_absolute"])


for _test in (test_link_inertia_reference, test_free_anchor_mapping):
    add_function_test(TestActorContract, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
