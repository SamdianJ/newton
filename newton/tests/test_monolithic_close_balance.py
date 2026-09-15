# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the independent FP64 energy/transform audit against Warp evaluation."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact import _tangent_response
from newton._src.solvers.monolithic.tet import _evaluate_smith
from scripts.monolithic_reference.probe_close_balance import smith_energy, transform_points


@wp.kernel
def _smith(f: wp.mat33, volume: float, mu: float, lam: float, output: wp.array[float]):
    energy, _stress, _tangent = _evaluate_smith(f, volume, mu, lam)
    output[0] = energy


@wp.kernel
def _huber(radius: wp.array[float], output: wp.array2d[float]):
    i = wp.tid()
    xi = wp.vec3(radius[i], 0.0, 0.0)
    force, energy, _hessian = _tangent_response(xi, wp.vec3(0.0, 0.0, 1.0), 10.0, 2.0)
    pending = -force / 10.0
    output[i, 0] = energy
    output[i, 1] = 5.0 * wp.dot(pending, pending) + 2.0 * wp.length(xi - pending)


class TestCloseBalance(unittest.TestCase):
    def test_local_huber_energy_from_history_diagnostics(self):
        radius = wp.array([0.0, 0.1, 0.2, 0.8], dtype=float, device="cpu")
        output = wp.zeros((4, 2), dtype=float, device="cpu")
        wp.launch(_huber, 4, [radius, output], device="cpu")
        values = output.numpy()
        np.testing.assert_allclose(values[:, 0], values[:, 1], rtol=1e-6, atol=1e-8)

    def test_deformed_smith_energy(self):
        rest = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.03, 0], [0, 0, 0.02]])
        poses = np.array([np.diag([25.0, 1 / 0.03, 50.0])])
        indices = np.array([[0, 1, 2, 3]])
        materials = np.array([[40000.0, 80000.0]])
        f = np.array([[0.8, 0.1, 0], [0, 1.1, 0.03], [0.02, 0, 0.95]])
        output = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(_smith, 1, [wp.mat33(f), 0.04 * 0.03 * 0.02 / 6, 40000.0, 80000.0, output], device="cpu")
        expected = float(output.numpy()[0])
        actual = smith_energy(rest @ f.T, indices, poses, materials)
        self.assertAlmostEqual(actual, expected, delta=abs(expected) * 1e-5)
        self.assertAlmostEqual(smith_energy(rest, indices, poses, materials), 0, delta=1e-14)

    def test_world_contact_point(self):
        rotation = wp.quat_rpy(0.3, -0.7, 1.2)
        pose = wp.transform(wp.vec3(0.1, -0.2, 0.3), rotation)
        point = np.array([0.02, -0.01, 0.03])
        expected = np.array(wp.transform_point(pose, wp.vec3(point)))
        np.testing.assert_allclose(transform_points(np.array(pose), point), expected, atol=4e-8)


if __name__ == "__main__":
    unittest.main()
