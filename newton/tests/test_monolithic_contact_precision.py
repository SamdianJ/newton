# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise native contact branches without changing production noise gates."""

import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests import test_monolithic_contact_components as contact
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import contact_oracle as oracle_math


def native_inputs(model, x, q, dtype):
    """Convert the frozen component poses and Jacobians to the selected native representation."""
    inputs = contact.state(model, x, q)
    if dtype == wp.float64:
        return inputs
    return tuple(
        wp.array(a.numpy(), dtype=wp.transform if i == 1 else wp.vec3, device=model.device)
        for i, a in enumerate(inputs)
    )


def test_native_contact(test, device):
    """Check curved/planar queries, radial slip and hard gates in both dtypes."""
    for dtype in (wp.float64, wp.float32):
        host = np.float64 if dtype == wp.float64 else np.float32
        for kind in ("plane", "sphere", "box"):
            model, xn = contact.fixture(device, kind)
            batch = contact.native(contact.contact(model, dtype=dtype))
            xn = xn.astype(host)
            q = np.zeros(2, dtype=host)
            batch.begin_step(*native_inputs(model, xn, q, dtype))
            frozen = batch.start.y.numpy().copy()
            for ratio in (0, 1e-10, 0.25, 0.9999, 1, 1.0001, 2, -2, 0):
                x = (xn + host(ratio * 1e-5) * np.array([0.6, 0.8, 0], dtype=host)).astype(host)
                batch.evaluate(*native_inputs(model, x, q, dtype), dt=host(0.001))
                test.assertEqual(int(batch.flags.numpy()[0]), 0)
                # Audit native scalar inputs as well as the stored coordinates.
                local = partial(
                    oracle_math.local,
                    width=float(host(1e-4)),
                    a=float(host(0.001) * host(0.01)),
                )
                with patch.object(oracle_math, "local", local):
                    oracle = oracle_math.evaluate(
                        model,
                        q.astype(float),
                        x.astype(float),
                        q.astype(float),
                        xn.astype(float),
                        kinds=[kind],
                        radius=float(host(0.001)),
                    )
                atol, rtol = (1e-12, 3e-10) if dtype == wp.float64 else (2e-5, 1e-3)
                np.testing.assert_allclose(contact.energy(batch), oracle[0].sum(axis=0), atol=atol, rtol=rtol)
                np.testing.assert_allclose(contact.residual(batch), oracle[1], atol=atol, rtol=rtol)
                matrix = contact.batch_matrix(batch)
                test.assertTrue(np.isfinite(matrix).all())
                test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * np.linalg.norm(matrix))
            for bad in ("nan", "degenerate", "backface", "quaternion"):
                inputs = list(native_inputs(model, xn, q, dtype))
                invalid = xn.copy()
                if bad == "nan":
                    invalid[0, 0] = np.nan
                elif bad == "degenerate":
                    invalid[1] = invalid[0]
                elif bad == "backface":
                    invalid[[1, 2]] = invalid[[2, 1]]
                else:
                    pose = inputs[1].numpy()
                    pose[0, 6] = 2
                    inputs[1].assign(pose)
                inputs[0].assign(invalid)
                batch.evaluate(*inputs, dt=host(0.001))
                test.assertNotEqual(int(batch.flags.numpy()[0]), 0, bad)
                np.testing.assert_array_equal(batch.start.y.numpy(), frozen)


def test_quaternion_tolerance(test, device):
    """Allow native rounding while rejecting malformed poses in both query paths."""
    model, x = contact.fixture(device, "plane")
    for dtype, tolerance in ((wp.float64, 1e-10), (wp.float32, 1e-6)):
        batch = contact.native(contact.contact(model, dtype=dtype))
        for error in (0.0, -0.5 * tolerance, 0.5 * tolerance, -2 * tolerance, 2 * tolerance, np.nan):
            with test.subTest(dtype=dtype, error=error):
                inputs = list(native_inputs(model, x, np.zeros(2), dtype))
                pose = inputs[1].numpy()
                pose[0, 3:7] = [0, 0, 0, 1 + error]
                inputs[1].assign(pose)
                valid = np.isfinite(error) and abs(error) < tolerance
                batch.begin_step(*inputs)
                test.assertEqual(int(batch.start_flags.numpy()[0]) == 0, valid)
                batch.evaluate(*inputs, dt=0.001)
                test.assertEqual(int(batch.flags.numpy()[0]) == 0, valid)


class TestContactPrecision(unittest.TestCase):
    pass


add_function_test(TestContactPrecision, "test_native_contact", test_native_contact, devices=get_test_devices())


add_function_test(
    TestContactPrecision, "test_quaternion_tolerance", test_quaternion_tolerance, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
