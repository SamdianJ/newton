# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise public P3 state, checkpoint and force ownership contracts."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.solvers.experimental.monolithic import MonolithicContactForces, SolverMonolithic
from newton.tests.test_monolithic_p3_coupled import assert_snapshot, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_state_and_forces(test, device):
    """Keep read-only force snapshots independent across stepping, reset and restore."""
    model = scene(device)
    run = solver(model)
    state = run.create_state(dt=0.001, source=model.state())
    initial = run.get_contact_forces(state)
    test.assertIsInstance(initial, MonolithicContactForces)
    test.assertFalse(initial.force_valid)
    test.assertEqual(initial.force_generation, 0)
    with test.assertRaises(ValueError):
        run.step(model.state(), state, model.control(), dt=0.001)
    run.step(state, state, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged)
    forces = run.get_contact_forces(state)
    test.assertTrue(forces.force_valid)
    test.assertEqual(forces.force_generation, 1)
    frozen = [a.copy() for a in (forces.particle_force, forces.body_wrench, forces.joint_force)]
    for array, shape in zip(
        (forces.particle_force, forces.body_wrench, forces.joint_force),
        ((model.particle_count, 3), (model.body_count, 6), (model.joint_dof_count,)),
        strict=True,
    ):
        test.assertEqual(array.shape, shape)
        test.assertEqual(array.dtype, np.float64)
        with test.assertRaises(ValueError):
            array.flat[0] = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state"
        test.assertIsNone(run.save_checkpoint(state, path))
        run.step(state, state, model.control(), dt=0.001)
        test.assertIsNone(run.reset_state(state, dt=0.001))
        test.assertFalse(run.get_contact_forces(state).force_valid)
        test.assertIsNone(run.load_checkpoint(state, path.with_suffix(".npz")))
        test.assertEqual(run.get_contact_forces(state).force_generation, 1)
    before = snapshot(state)
    with patch.object(run, "_finish_candidate", side_effect=_StepFailure(run.Status.NONFINITE, "injected")):
        run.step(state, state, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, state, before)
    test.assertEqual(run.get_contact_forces(state).force_generation, 1)
    for array, old in zip((forces.particle_force, forces.body_wrench, forces.joint_force), frozen, strict=True):
        np.testing.assert_array_equal(array, old)
    test.assertEqual(np.count_nonzero(initial.particle_force), 0)


def test_checkpoint_atomicity(test, device):
    """Preserve existing files and target states on partial writes and invalid loads."""
    model = scene(device)
    run = solver(model)
    state = run.create_state(dt=0.001)
    before = snapshot(state)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        run.save_checkpoint(state, path)
        original = path.read_bytes()

        def fail(state, temporary):
            Path(temporary).write_bytes(b"partial")
            raise OSError("injected write failure")

        with patch.object(run._states, "save_checkpoint", fail), test.assertRaises(OSError):
            run.save_checkpoint(state, path)
        test.assertEqual(path.read_bytes(), original)
        test.assertEqual(list(Path(directory).iterdir()), [path])
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["history.config_digest"] = np.zeros_like(arrays["history.config_digest"])
        np.savez(path, **arrays)
        with test.assertRaises(ValueError):
            run.load_checkpoint(state, path)
        assert_snapshot(test, state, before)


class TestMonolithicAPI(unittest.TestCase):
    def test_explicit_constructor_and_base_boundary(self):
        parameters = inspect.signature(SolverMonolithic).parameters
        self.assertFalse(any(p.kind == p.VAR_KEYWORD for p in parameters.values()))
        self.assertEqual(parameters["publish_policy"].default, "strict")
        self.assertEqual(parameters["pcg_mode"].default, SolverMonolithic.PCGMode.DIAGNOSTIC)
        self.assertIs(parameters["use_cuda_graph"].default, False)
        model = scene("cpu")
        run = solver(model)
        self.assertFalse(run.supports_collision_pipeline)
        self.assertIsNone(run.contacts)
        run.contacts = None
        with self.assertRaises(ValueError):
            run.contacts = object()
        with self.assertRaises(ValueError):
            run.update_contacts(None)
        for kw in ({"collision_frequency": {}}, {"collision_frequency_type": {}}):
            with self.assertRaises(ValueError):
                run.set_collision_frequency(**kw)
        for kw in ({"collision_pipeline": None}, {"collision_frequency": {}}, {"normal_smoothing_width": 0}):
            with self.assertRaises(TypeError):
                solver(model, **kw)
        with self.assertRaisesRegex(NotImplementedError, "reset_state"):
            run.reset(model.state())
        with self.assertRaises(NotImplementedError):
            run.notify_model_changed(0)
        self.assertFalse(hasattr(run, "_create_coupled_variational"))
        self.assertFalse(hasattr(run, "_create_articulation_only"))


add_function_test(TestMonolithicAPI, "test_state_and_forces", test_state_and_forces, devices=get_test_devices())
add_function_test(TestMonolithicAPI, "test_checkpoint_atomicity", test_checkpoint_atomicity, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
