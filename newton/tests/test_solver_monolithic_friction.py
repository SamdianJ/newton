# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G3 contact mathematics and G4 history transactions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact import _normal_response, _tangent_response
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _math_probe(
    u: wp.array[float],
    xi: wp.array[wp.vec3],
    eps: float,
    k: float,
    cap: float,
    normal: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    energy: wp.array[float],
    hessian: wp.array[wp.mat33],
):
    i = wp.tid()
    normal[i] = _normal_response(u[i], eps, k)
    f, e, h = _tangent_response(xi[i], wp.vec3(0.0, 0.0, 1.0), k, cap)
    force[i] = f
    energy[i] = e
    hessian[i] = h


def _probe(device, u, xi, *, eps=0.2, k=3.0, cap=0.6):
    count = len(u)
    arrays = [wp.zeros(count, dtype=dtype, device=device) for dtype in (wp.vec3, wp.vec3, float, wp.mat33)]
    wp.launch(
        _math_probe,
        count,
        [wp.array(u, dtype=float, device=device), wp.array(xi, dtype=wp.vec3, device=device), eps, k, cap, *arrays],
        device=device,
    )
    return [a.numpy().astype(float) for a in arrays]


def _normal_oracle(u, eps, k):
    if u <= -eps:
        return np.zeros(3)
    if u >= eps:
        return np.array([0.5 * k * u * u, k * u, k])
    t = (u + eps) / (2 * eps)
    p, dp, ddp = eps * (2 * t**3 - t**4), 3 * t * t - 2 * t**3, 3 * t * (1 - t) / eps
    return np.array([0.5 * k * p * p, k * p * dp, k * (dp * dp + p * ddp)])


def test_polyrelu(test, device):
    """Check all pieces, boundary continuity and a five-step FD plateau."""
    u = np.array([-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4])
    actual = _probe(device, u, np.zeros((len(u), 3)))[0]
    np.testing.assert_allclose(actual, [_normal_oracle(v, 0.2, 3) for v in u], rtol=2e-6, atol=1e-7)
    for center in (-0.2, -0.1, 0.0, 0.1, 0.2):
        errors = []
        for h in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
            result = _probe(device, [center - h, center, center + h], np.zeros((3, 3)))[0]
            fd = (result[2, :2] - result[0, :2]) / (2 * h)
            expected = result[1, 1:]
            errors.append(np.linalg.norm(fd - expected) / max(np.linalg.norm(expected), 1e-5))
        test.assertLess(min(errors), 5e-3)
    test.assertGreater(actual[2, 1], 0)  # Positive physical gap inside the shell.
    legacy = _probe(device, [-0.1, 0, 0.1], np.zeros((3, 3)), eps=0)[0]
    np.testing.assert_allclose(legacy, [[0, 0, 0], [0, 0, 0], [0.015, 0.3, 3]], rtol=1e-6)


def test_radial_return(test, device):
    """Verify zero, stick, slip, reversal, cap, potential and local PSD."""
    xi = np.array([[0, 0, 0], [0.01, 0.02, 0], [0.4, 0.3, 0], [-0.4, -0.3, 0], [0.2, 0, 0]])
    _, forces, energies, matrices = _probe(device, np.zeros(5), xi)
    for i, x in enumerate(xi):
        r = np.linalg.norm(x)
        k = 3.0
        cap = 0.6
        expected = -k * x if k * r < cap else -cap * x / r
        np.testing.assert_allclose(forces[i], expected, atol=1e-7)
        test.assertLessEqual(np.linalg.norm(forces[i]), cap + 1e-7)
        test.assertGreaterEqual(np.linalg.eigvalsh(matrices[i]).min(), -1e-6)
        np.testing.assert_allclose(matrices[i], matrices[i].T, atol=1e-8)
        test.assertAlmostEqual(
            energies[i], 0.5 * k * r * r if k * r < cap else cap * r - cap * cap / (2 * k), delta=1e-7
        )
    # Equality chooses the sliding generalized derivative.
    test.assertAlmostEqual(matrices[-1, 0, 0], 0, delta=1e-6)
    for x in xi[1:4]:
        errors = []
        for h in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
            variations = np.array([x + np.eye(3)[j] * sign * h for j in range(2) for sign in (-1, 1)])
            _, f, e, _ = _probe(device, np.zeros(4), variations)
            _, fc, _, hc = _probe(device, [0], [x])
            grad = np.array([(e[2 * j + 1] - e[2 * j]) / (2 * h) for j in range(2)])
            derivative = np.stack([-(f[2 * j + 1] - f[2 * j]) / (2 * h) for j in range(2)], axis=1)
            errors.append(
                max(np.linalg.norm(grad + fc[0, :2]), np.linalg.norm(derivative - hc[0, :, :2]))
                / max(np.linalg.norm(hc), 1e-6)
            )
        test.assertLess(min(errors), 5e-3)
    _, f, e, h = _probe(device, np.zeros(5), xi, cap=0)
    for a in (f, e, h):
        np.testing.assert_array_equal(a, 0)
    # BE work + change in stored elastic energy + return dissipation <= 0.
    old = np.array([0.1, 0, 0])
    increment = np.array([0.3, 0.1, 0])
    trial = old + increment
    _, f, _, _ = _probe(device, [0], [trial])
    pending = -f[0] / 3
    work = f[0] @ increment
    delta = 0.5 * 3 * (pending @ pending - old @ old)
    plastic = 0.6 * np.linalg.norm(trial - pending)
    test.assertLessEqual(work + delta + plastic, 1e-7)


class TestMonolithicFriction(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (test_polyrelu, test_radial_return):
        add_function_test(TestMonolithicFriction, function.__name__, function, devices=[device])

if __name__ == "__main__":
    unittest.main()
