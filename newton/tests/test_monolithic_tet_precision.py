# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check native tet precision specialization against the independent discrete oracle."""

import unittest
from itertools import product

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.tet_precision import TetObjective
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.tet_fixtures import build_model
from scripts.monolithic_reference.tet_oracle import TetOracle


def test_native_objective(test, device):
    """Evaluate both dtypes, materials, mass modes and fixed-node layouts in one process."""
    for dtype, smith, consistent, fixed in product(
        (wp.float64, wp.float32), (False, True), (False, True), (False, True)
    ):
        with test.subTest(dtype=dtype, smith=smith, consistent=consistent, fixed=fixed):
            model = build_model(device, fixed=fixed, gravity=(1, -2, -9.81))
            obj = TetObjective(
                model,
                material_model="smith_log_stabilized" if smith else "kim_stable_no_log",
                mass_mode="consistent" if consistent else "lumped",
                density=wp.full(model.tet_count, 1000.0, device=device) if consistent else None,
                dtype=dtype,
            )
            vector = wp.vec3d if dtype == wp.float64 else wp.vec3
            initial = model.particle_q.numpy().astype(np.float64)
            positions = initial @ np.array([[1.12, 0.07, 0], [0.01, 0.91, 0.03], [0, 0, 1.06]]).T
            fixed_ids = model.particle_inv_mass.numpy() == 0
            positions[fixed_ids] = initial[fixed_ids]
            x, xn = (wp.array(a, dtype=vector, device=device) for a in (positions, initial))
            vn = wp.zeros_like(xn)
            forces = wp.full(model.particle_count, wp.vec3(0.002), dtype=wp.vec3, device=device)
            obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
            test.assertEqual(obj.residual.dtype, vector)
            test.assertEqual(obj.parts.dtype, dtype)
            test.assertEqual(obj.min_det_f.dtype, dtype)
            test.assertEqual(int(obj.flags.numpy()[0]), 0)
            # Audit the quantized inputs and actual native dt, not unquantized originals.
            native_dt = np.float64(0.001) if dtype == wp.float64 else np.float32(0.001)
            parts, residual, _ = TetOracle(model, obj.physics).evaluate(
                x.numpy().astype(np.float64),
                xn.numpy().astype(np.float64),
                vn.numpy().astype(np.float64),
                model.gravity.numpy()[-1].astype(np.float64),
                forces.numpy(),
                float(native_dt),
            )
            tolerance = 1e-10 if dtype == wp.float64 else 2e-5
            np.testing.assert_allclose(
                obj.parts.numpy(), [parts[0], parts[1], sum(parts[2:])], atol=tolerance, rtol=tolerance
            )
            np.testing.assert_allclose(obj.residual.numpy().ravel(), residual, atol=tolerance, rtol=tolerance)
            obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001, min_det_f_guard=2.0)
            test.assertEqual(int(obj.flags.numpy()[0]), 1)
            corrupt = x.numpy()
            corrupt[np.flatnonzero(~fixed_ids)[0], 0] = np.nan
            x.assign(corrupt)
            obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
            test.assertEqual(int(obj.flags.numpy()[0]), 2)


def test_near_rest_displacement(test, device):
    """Resolve a near-rest FP32 step from its stored displacement, not from rounded positions."""
    model = build_model(device)
    initial = model.particle_q.numpy().astype(np.float32)
    free = model.particle_inv_mass.numpy() > 0
    # At a 1e-5 strain, FP32 positions resolve the step only to ~0.3% and an
    # energy formed from F loses the O(strain^2) elastic part entirely.
    step = (1e-5 * initial @ np.array([[1.0, 0.4, 0], [0.2, -0.7, 0.3], [0, 0.1, 0.5]]).T).astype(np.float32)
    step[~free] = 0.0
    for smith in (False, True):
        with test.subTest(smith=smith):
            obj = TetObjective(
                model,
                material_model="smith_log_stabilized" if smith else "kim_stable_no_log",
                mass_mode="lumped",
                dtype=wp.float32,
            )
            xn = wp.array(initial, dtype=wp.vec3, device=device)
            x = wp.array(initial + step, dtype=wp.vec3, device=device)
            displacement = wp.array(step, dtype=wp.vec3, device=device)
            vn, forces = wp.zeros_like(xn), wp.zeros_like(xn)
            obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.01, displacement=displacement)
            test.assertEqual(int(obj.flags.numpy()[0]), 0)
            exact = initial.astype(np.float64) + step.astype(np.float64)
            parts, residual, _ = TetOracle(model, obj.physics).evaluate(
                exact,
                initial.astype(np.float64),
                np.zeros_like(exact),
                np.zeros(3),
                np.zeros_like(exact),
                float(np.float32(0.01)),
            )
            # Inertia comes from the stored step and is exact to rounding. The elastic
            # part keeps an absolute strain floor of order eps32 (the FP32 product
            # Dx_n pose - I), about 5e-8 here, i.e. 0.5% of this strain.
            energy = obj.parts.numpy()
            np.testing.assert_allclose(energy[0], parts[0], rtol=1e-6)
            np.testing.assert_allclose(energy[1], parts[1], rtol=2e-2)
            test.assertEqual(energy[2], sum(parts[2:]))
            scale = np.abs(residual).max()
            np.testing.assert_allclose(obj.residual.numpy().ravel(), residual, rtol=0, atol=1e-2 * scale)


def test_input_rejection(test, device):
    """Reject mixed coordinates and unrepresentable native timesteps before changing scratch."""
    model = build_model(device)
    obj = TetObjective(model, material_model="kim_stable_no_log", mass_mode="lumped", dtype=wp.float32)
    x = wp.array(model.particle_q.numpy(), dtype=wp.vec3, device=device)
    vn = wp.zeros_like(x)
    forces = wp.zeros_like(x)
    obj.parts.fill_(17.0)
    for dt in (1e-50, 1e40):
        with test.assertRaises(ValueError):
            obj.evaluate(x, x, vn, forces, model.gravity, dt=dt)
        np.testing.assert_array_equal(obj.parts.numpy(), [17.0] * 3)
    with test.assertRaises(ValueError):
        obj.evaluate(wp.array(x.numpy(), dtype=wp.vec3d, device=device), x, vn, forces, model.gravity, dt=0.001)
    np.testing.assert_array_equal(obj.parts.numpy(), [17.0] * 3)
    with test.assertRaises(ValueError):
        TetObjective(model, material_model="kim_stable_no_log", mass_mode="lumped", dtype=wp.float16)


def test_graph_replay(test, device):
    """Capture each native specialization and replay after changing candidate values."""
    model = build_model(device)
    for dtype, vector in ((wp.float64, wp.vec3d), (wp.float32, wp.vec3)):
        obj = TetObjective(model, material_model="smith_log_stabilized", mass_mode="lumped", dtype=dtype)
        xn = wp.array(model.particle_q.numpy(), dtype=vector, device=device)
        x, vn = wp.clone(xn), wp.zeros_like(xn)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
        with wp.ScopedCapture(device=device) as capture:
            obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
        x.assign(x.numpy() * 1.01)
        wp.capture_launch(capture.graph)
        captured_parts, captured_residual = obj.parts.numpy(), obj.residual.numpy()
        test.assertEqual(int(obj.flags.numpy()[0]), 0)
        obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
        np.testing.assert_array_equal(obj.parts.numpy(), captured_parts)
        np.testing.assert_array_equal(obj.residual.numpy(), captured_residual)


class TestTetPrecision(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestTetPrecision, "test_native_objective", test_native_objective, devices=[_device])
    add_function_test(TestTetPrecision, "test_near_rest_displacement", test_near_rest_displacement, devices=[_device])
    add_function_test(TestTetPrecision, "test_input_rejection", test_input_rejection, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestTetPrecision, "test_graph_replay", test_graph_replay, devices=[_device])


if __name__ == "__main__":
    unittest.main()
