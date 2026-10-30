# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regress rigid rotation prediction and contact-driven turning."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import solver_monolithic
from newton.tests.test_monolithic_rigid_dynamics import rigid_scene, rigid_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import transform
from scripts.monolithic_reference.run_dexterous_tasks import DT, TaskScene


def test_rotating_predictor(test, device):
    """Predict material-point velocity on rotating bodies, including nontrivial body frames."""
    model = rigid_scene(device, contact=False)
    run = rigid_solver(model)
    h = 0.001
    state = run.create_state(dt=h)
    history = state._monolithic_coupled
    rotations = [transform(np.r_[np.zeros(3), q])[:3, :3] for q in history.rigid_rotation.numpy()]
    omega = history.rigid_omega.numpy()
    iterate = solver_monolithic.iterate

    def check_predictor(owner, dt):
        for i, q in enumerate(owner._accepted._monolithic_coupled.rigid_rotation.numpy()):
            current = transform(np.r_[np.zeros(3), q])[:3, :3]
            rate = (current - rotations[i]) @ rotations[i].T / dt
            angular = np.array([rate[2, 1] - rate[1, 2], rate[0, 2] - rate[2, 0], rate[1, 0] - rate[0, 1]]) / 2
            np.testing.assert_allclose(angular, omega[i], atol=1e-8, rtol=1e-7)
        return iterate(owner, dt)

    with patch.object(solver_monolithic, "iterate", check_predictor):
        run.step(state, state, model.control(), dt=h)
    test.assertTrue(run.last_stats.converged)


def test_fingertip_box_corner(test, device):
    """Converge a real corner contact with outward force and balanced endpoint wrenches."""
    scene = TaskScene("pinch-transfer", device=device, wrist=False)
    q = scene.state.joint_q.numpy()
    positions = np.array([[-0.0078, -0.0244, 0.06145], [0.04, 0.05, 0.06]])
    q[:6] = (positions - scene.initial).ravel()
    q[6:] = [0, 0, 0.06, 0, 0, 0, 1]
    scene.state.joint_q.assign(q)
    scene.solver.reset_state(scene.state, dt=DT)
    targets = np.zeros(scene.control.joint_target_q.shape[0])
    targets[:6] = q[:6]
    scene.control.joint_target_q.assign(targets)
    scene.solver.step(scene.state, scene.state, scene.control, dt=DT)
    test.assertTrue(scene.solver.last_stats.converged, scene.solver.last_stats.failure_reason)
    forces = scene.solver.get_contact_forces(scene.state).body_wrench
    np.testing.assert_allclose(forces[:, :3].sum(axis=0), 0, atol=1e-10)
    test.assertGreater(forces[scene.fingers[0], :3] @ (positions[0] - [0, 0, 0.06]), 0)


def test_wrist_material_velocity(test, device):
    """Match world-space targets through FK, including fingertip material rotation."""
    scene = TaskScene("pinch-transfer", device=device, turn_angle=-np.pi / 2, initial_yaw=np.pi / 6)
    k = scene.solver._kinematics
    local_bodies = [list(k.body_map.numpy()).index(body) for body in scene.fingers]
    for t in (0.0, 0.8, 1.5, 2.5, 4.0):
        with patch.object(scene.solver, "step"):
            expected, _ = scene.step(t)
        q = scene.control.joint_target_q.numpy()[k.coord_map.numpy()]
        v = scene.control.joint_target_qd.numpy()[k.dof_map.numpy()]
        k.evaluate(wp.array(q, dtype=scene.solver._dtype, device=device))
        poses = k.poses.numpy()[local_bodies]
        np.testing.assert_allclose(poses[:, :3], expected, atol=2e-8)
        angular = np.einsum("bji,j->bi", k.jw.numpy()[local_bodies], v)
        np.testing.assert_allclose(angular[:, :2], 0, atol=1e-12)
        np.testing.assert_allclose(angular[:, 2], v[0], atol=1e-12)
        if t == 1.5:
            test.assertGreater(abs(v[0]), 0.1)


class TestRotatingContact(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestRotatingContact, test_rotating_predictor.__name__, test_rotating_predictor, devices=[_device])
    add_function_test(
        TestRotatingContact, test_fingertip_box_corner.__name__, test_fingertip_box_corner, devices=[_device]
    )
    add_function_test(
        TestRotatingContact, test_wrist_material_velocity.__name__, test_wrist_material_velocity, devices=[_device]
    )


if __name__ == "__main__":
    unittest.main()
