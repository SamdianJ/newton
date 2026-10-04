# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check device validation of mutable continuation state."""

import unittest
from unittest.mock import patch

import warp as wp

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


class TestDeviceChecks(unittest.TestCase):
    pass


add_function_test(TestDeviceChecks, "test_device_validation", test_device_validation, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
