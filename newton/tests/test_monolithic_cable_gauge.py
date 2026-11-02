# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check device qualification and projection of cable common-angle modes."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_gauge import _factor_modes
from newton.tests.test_monolithic_material_state import scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_gauge_device(test, device):
    """Keep raw torques, Jacobians, contact factors and directions on device."""
    for precision in ("fp32", "fp64"):
        _, run = scene(
            cloth=False,
            cables=2,
            fixed_theta=(False, False),
            precision=precision,
            device=device,
            pcg_mode="production",
            use_cuda_graph=device.is_cuda,
        )
        state = run.create_state(dt=0.001)
        run.step(state, state, run.create_control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        direction = np.linspace(-0.5, 0.7, run._delta.size, dtype=run._numpy_dtype)
        expected = direction.copy()
        for cable in run._cables:
            rows = cable.rows_host[3 * cable.ids.size :]
            test.assertTrue(np.all(rows >= 0))
            expected[rows] -= expected[rows].mean(dtype=np.float64)
        run._delta.assign(direction)
        records = run._contact.pool.records
        arrays = [
            run._delta,
            records.factor_count,
            records.factor_offsets,
            records.factor_columns,
            records.factor_values,
            records.factor_weights,
        ]
        arrays.extend(a for cable in run._cables for a in (cable.raw, cable.objective.jacobian))
        pointers = {a.ptr for a in arrays if a.size}
        numpy = wp.array.numpy

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("Cable gauge read a physical payload")
            return numpy(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", read):
            run._settle_cable_gauges()
            # A subsequent trial may remove a null mode; the accepted operator
            # must keep its own pins until another linearization commits them.
            run._gauges.pins.fill_(-1)
            run._project_gauges()
        np.testing.assert_allclose(run._delta.numpy(), expected, rtol=0, atol=4 * np.finfo(run._numpy_dtype).eps)


def test_factor_modes(test, device):
    """Handle unsorted group rows and first, middle, trailing or entirely empty factors."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        row_actor = np.array([-1, 0, 1, 0, 2, -1, 2, 0], dtype=np.int32)
        columns = np.array([1, 2, 3, 7, 4, 6, 5], dtype=np.int32)
        values = np.array([1, 3, -1, 0, 2, -2, 8], dtype=native)
        offsets = np.array([0, 0, 4, 4, 4, 7, 7], dtype=np.int32)
        weights = np.ones(6, dtype=native)
        count = wp.array([6], dtype=int, device=device)
        arrays = [
            wp.array(a, dtype=d, device=device)
            for a, d in ((offsets, int), (columns, int), (values, dtype), (weights, dtype), (row_actor, int))
        ]
        blocked = wp.zeros(3, dtype=int, device=device)
        for kind in ("interleaved", "empty", "zero_count", "positive", "zero_weight"):
            host_offsets = np.zeros_like(offsets) if kind == "empty" else offsets
            host_values = values.copy()
            host_weights = weights.copy()
            if kind in ("positive", "zero_weight"):
                host_values[5] += native(0.5)
            if kind == "zero_weight":
                host_weights[4] = 0
            active = 0 if kind == "zero_count" else 6
            count.fill_(active)
            arrays[0].assign(host_offsets)
            arrays[2].assign(host_values)
            arrays[3].assign(host_weights)
            expected = np.zeros(3, dtype=np.int32)
            for factor in range(active):
                if host_weights[factor] > 0:
                    part = slice(host_offsets[factor], host_offsets[factor + 1])
                    for actor in range(3):
                        if host_values[part][row_actor[columns[part]] == actor].sum() != 0:
                            expected[actor] = 1
            blocked.zero_()
            wp.launch(_factor_modes, 6, [count, *arrays, blocked], device=device)
            np.testing.assert_array_equal(blocked.numpy(), expected)


class TestCableGauge(unittest.TestCase):
    pass


add_function_test(TestCableGauge, "test_gauge_device", test_gauge_device, devices=get_test_devices())
add_function_test(TestCableGauge, "test_factor_modes", test_factor_modes, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
