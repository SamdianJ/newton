# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check device validation of mutable continuation state."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.tests.test_monolithic_rigid_dynamics import rigid_scene, rigid_solver
from newton.tests.test_monolithic_solver_coupled import scene, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_device_validation(test, device):
    """Validate bound states without copying state or history payloads to the host."""
    for precision in ("fp64", "fp32"):
        run = solver(scene(device, fixed=True), precision=precision, mode="production", graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        history = state._monolithic_coupled
        arrays = [value for owner in (state, history) for value in vars(owner).values() if isinstance(value, wp.array)]
        pointers = {a.ptr for a in arrays if a.size}
        numpy = wp.array.numpy

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("Validation read a state/history payload")
            return numpy(array, *args, **kwargs)

        for _ in range(2):
            with patch.object(wp.array, "numpy", read):
                run._states.validate(state)
        # Invalid values must still be rejected after graph replay and recover.
        for array, value in (
            (state.particle_f, float("nan")),
            (history.bound_q, 0.25),
            (history.valid, 0),
            (history.config_digest, 0),
            (history.precision, 16),
            (history.force_generation, -1),
        ):
            old = array.numpy().copy()
            array.fill_(array.dtype(value))
            with patch.object(wp.array, "numpy", read), test.assertRaises(ValueError):
                run._states.validate(state)
            array.assign(old)
            with patch.object(wp.array, "numpy", read):
                run._states.validate(state)
        x = state.particle_q.numpy().copy()
        changed = x.copy()
        changed[0, 0] += 1
        state.particle_q.assign(changed)
        history.bound_x.assign(changed)
        with patch.object(wp.array, "numpy", read), test.assertRaises(ValueError):
            run._states.validate(state)
        state.particle_q.assign(x)
        history.bound_x.assign(x)


def test_device_iteration(test, device):
    """Keep nonlinear vectors on device throughout production iteration."""
    for precision in ("fp64", "fp32"):
        run = solver(scene(device), precision=precision, mode="production", graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        original = run._iterate_articulation
        numpy = wp.array.numpy
        pointers = {
            a.ptr
            for a in (
                run._complete_r,
                run._linear.scale,
                run._delta,
                run._post_error,
                run._accepted_r,
                run._reference_device,
                run._effective_device,
                run._stats_residual,
            )
        }

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("Nonlinear iteration read a complete vector")
            return numpy(array, *args, **kwargs)

        def iterate(dt, original=original, read=read, run=run, pointers=pointers):
            pointers.update(a.ptr for state in (run._accepted, run._trial) for a in (state.joint_q, state.particle_q))
            with patch.object(wp.array, "numpy", read):
                return original(dt)

        with patch.object(run, "_iterate_articulation", iterate):
            run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertIsInstance(run.last_stats.variational["residual"], list)
        test.assertEqual(len(run.last_stats.variational["residual"]), run._complete_r.size)


def test_failed_iteration_stats(test, device):
    """Preserve the last reported residual if a later assembly fails."""
    run = solver(scene(device), mode="production", graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    assemble = run._assemble
    expected = {}

    def fail(iteration, **kwargs):
        if iteration == 1:
            expected["residual"] = run._stats_residual.numpy().copy()
            raise _StepFailure(run.Status.NONFINITE, "injected later assembly failure")
        return assemble(iteration, **kwargs)

    with patch.object(run, "_assemble", fail):
        run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    test.assertIn("residual", expected)
    np.testing.assert_array_equal(run.last_stats.variational["residual"], expected["residual"])


def test_free_coordinate_history(test, device):
    """Check all FREE position/rate history entries with unequal 7/6 lengths."""
    run = rigid_solver(rigid_scene(device, before=True, contact=False), mode="production", graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    test.assertGreater(state.joint_q.size, state.joint_qd.size)
    test.assertFalse(run._device_iteration)
    history = state._monolithic_coupled
    for name in ("bound_q", "bound_qd"):
        array = getattr(history, name)
        saved = array.numpy().copy()
        changed = saved.copy()
        changed[-1] += 0.01
        array.assign(changed)
        with test.assertRaisesRegex(ValueError, "Coordinates/rates changed"):
            run._states.validate(state)
        array.assign(saved)
        run._states.validate(state)


class TestDeviceChecks(unittest.TestCase):
    pass


add_function_test(TestDeviceChecks, "test_device_validation", test_device_validation, devices=get_test_devices())

add_function_test(TestDeviceChecks, "test_device_iteration", test_device_iteration, devices=get_test_devices())

add_function_test(
    TestDeviceChecks, "test_failed_iteration_stats", test_failed_iteration_stats, devices=get_test_devices()
)

add_function_test(
    TestDeviceChecks, "test_free_coordinate_history", test_free_coordinate_history, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main()
