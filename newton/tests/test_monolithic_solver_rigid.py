# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise short mixed rigid trajectories without enabling new contact pairs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_monolithic_rigid_dynamics import apply_wrench, independent_step, rigid_scene, rigid_solver
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import transform


def physical_frame(run, state):
    history = state._monolithic_coupled
    arm = run._articulation_actor
    values = {
        "a_q": state.joint_q.numpy()[list(arm.coord_indices)],
        "a_v": state.joint_qd.numpy()[list(arm.dof_indices)],
        "x": state.particle_q.numpy(),
        "v": state.particle_qd.numpy(),
        "force_a": history.contact_body_force.numpy()[list(arm.body_indices)],
        "force_q": history.contact_joint_force.numpy()[list(arm.dof_indices)],
        "force_x": history.contact_particle_force.numpy(),
        "generation": history.force_generation.numpy(),
    }
    for i, actor in enumerate(run._rigid.actors):
        label = run.model.body_label[actor.body_indices[0]]
        for field in ("com", "rotation", "v_com", "omega", "vsym"):
            values[field + "." + label] = getattr(history, "rigid_" + field).numpy()[i]
    return {name: value.copy() for name, value in values.items()}


def assert_physical(test, a, b):
    for name, value in a.items():
        if name == "generation":
            np.testing.assert_array_equal(value, b[name])
        elif name.startswith("rotation."):
            other = b[name]
            sign = 1 if value @ other >= 0 else -1
            angle = 4 * np.arcsin(min(1, np.linalg.norm(value - sign * other) / 2))
            test.assertLessEqual(angle, 1e-7)
        else:
            tolerance = (
                1e-4
                if name.startswith("force")
                else 1e-6
                if name in ("a_v", "v") or name.startswith(("v_com", "omega", "vsym"))
                else 1e-7
            )
            np.testing.assert_allclose(value, b[name], atol=tolerance, rtol=1e-5 if name.startswith("force") else 0)


def check_step(test, run, state, control, step, contact):
    apply_wrench(run, state, control, step)
    history = state._monolithic_coupled
    old = {
        name: getattr(history, "rigid_" + name).numpy().copy() for name in ("com", "rotation", "v_com", "omega", "vsym")
    }
    run.step(state, state, control, dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    test.assertFalse(run.last_stats.rolled_back)
    for record in run.last_stats.variational["linear_postchecks"]:
        if all(v["residual"] <= v["limit"] for v in record["blocks"].values()):
            test.assertEqual(len(record["blocks"]), 7)
    history = state._monolithic_coupled
    for i, actor in enumerate(run._rigid.actors):
        values = old["vsym"][i]
        sym = np.array(
            [[values[0], values[3], values[4]], [values[3], values[1], values[5]], [values[4], values[5], values[2]]]
        )
        expected = independent_step(
            old["com"][i],
            transform(np.r_[np.zeros(3), old["rotation"][i]])[:3, :3],
            old["v_com"][i],
            old["omega"][i],
            sym,
            run._rigid.mass.numpy()[i],
            run.model.body_inertia.numpy()[actor.body_indices[0]],
            run._rigid.force.numpy()[i],
            run._rigid.torque.numpy()[i],
            0.001,
        )
        for name, value, tolerance in (
            ("com", expected[0], 1e-7),
            ("v_com", expected[2], 1e-6),
            ("omega", expected[3], 1e-6),
        ):
            np.testing.assert_allclose(getattr(history, "rigid_" + name).numpy()[i], value, atol=tolerance, rtol=0)
        r = expected[1]
        scalar = np.sqrt(1 + np.trace(r)) / 2
        quat = np.r_[np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (4 * scalar), scalar]
        assert_physical(test, {"rotation.rigid": history.rigid_rotation.numpy()[i]}, {"rotation.rigid": quat})
    test.assertEqual(int(history.force_generation.numpy()[0]), step + 1)
    np.testing.assert_array_equal(history.contact_body_force.numpy()[run._rigid.body_indices], 0)
    matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
    a, t = run._layout.indices("articulation"), run._layout.indices("tet")
    active = bool(np.linalg.norm(matrix[np.ix_(a, t)]) > 0)
    if active:
        test.assertTrue(contact)
        test.assertGreater(np.linalg.norm(matrix[np.ix_(a, t)]), 0)
        test.assertGreater(np.linalg.norm(history.contact_body_force.numpy()), 0)
        test.assertGreater(np.linalg.norm(history.contact_particle_force.numpy()), 0)
        np.testing.assert_allclose(
            history.contact_body_force.numpy()[:, :3].sum(axis=0) + history.contact_particle_force.numpy().sum(axis=0),
            0,
            atol=1e-12,
        )
    return active


def test_mixed_matrix(test, device):
    """Converge ten steps across layouts, creation orders and execution modes with midpoint continuation."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    test.integration_cases = []
    test.integration_frames = {}
    for contact in (False, True):
        reference = None
        for mode, graph in modes:
            for before in (False, True):
                for permuted in (False, True):
                    with test.subTest(contact=contact, mode=mode, graph=graph, before=before, permuted=permuted):
                        model = rigid_scene(device, before=before, contact=contact)
                        options = {
                            "law": "C-IE",
                            "globalization": "ArmijoOrResNorm",
                            "mode": mode,
                            "graph": graph,
                            "allow_nonconservative_energy": True,
                            "permuted": permuted,
                        }
                        run = rigid_solver(model, **options)
                        state = run.create_state(dt=0.001)
                        control = model.control()
                        frames = []
                        active_steps = 0
                        with tempfile.TemporaryDirectory() as folder:
                            checkpoint = Path(folder) / "midpoint.npz"
                            midpoint = None
                            for step in range(10):
                                active_steps += check_step(test, run, state, control, step, contact)
                                frames.append(physical_frame(run, state))
                                if step == 4:
                                    midpoint = snapshot(state)
                                    run.save_checkpoint(state, checkpoint)
                            for _ in range(2):
                                resumed = rigid_solver(model, **options)
                                restored = resumed.create_state(dt=0.001)
                                resumed.load_checkpoint(restored, checkpoint)
                                assert_snapshot(test, restored, midpoint)
                                for step in range(5, 10):
                                    check_step(test, resumed, restored, control, step, contact)
                                    assert_physical(test, physical_frame(resumed, restored), frames[step])
                        test.assertEqual(active_steps, 5 if contact else 0)
                        if reference is None:
                            reference = frames
                            test.integration_frames[str(contact)] = frames
                        else:
                            for current, expected in zip(frames, reference, strict=True):
                                assert_physical(test, current, expected)
                        test.integration_cases.append(
                            {
                                "device": str(device),
                                "contact": contact,
                                "mode": mode,
                                "graph": graph,
                                "before": before,
                                "permuted": permuted,
                                "steps": 10,
                                "restored_steps": 10,
                                "active_steps": active_steps,
                                "converged": True,
                                "layout": run._layout.identity,
                                "physical_digest": run._kinematics.digest.hex(),
                                "rigid_coord_indices": run._rigid.coord_indices.tolist(),
                                "rigid_dof_indices": run._rigid.dof_indices.tolist(),
                                "arm_body_indices": list(run._articulation_actor.body_indices),
                            }
                        )


def test_mixed_acceptance(test, device):
    """Converge active A-T plus torque-driven rigid actors under all laws and acceptance modes."""
    mode, graph = ("production", True) if device.is_cuda else ("diagnostic", False)
    test.integration_cases = []
    for law in ("C-EE", "C-IE", "C-II"):
        for globalization in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            with test.subTest(law=law, globalization=globalization):
                model = rigid_scene(device)
                run = rigid_solver(
                    model,
                    law=law,
                    globalization=globalization,
                    mode=mode,
                    graph=graph,
                    allow_nonconservative_energy=True,
                )
                state, control = run.create_state(dt=0.001), model.control()
                for step in range(3):
                    test.assertTrue(check_step(test, run, state, control, step, True))
                test.integration_cases.append(
                    {"device": str(device), "law": law, "globalization": globalization, "steps": 3, "converged": True}
                )


class TestRigidSolver(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (test_mixed_matrix, test_mixed_acceptance):
        add_function_test(TestRigidSolver, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
