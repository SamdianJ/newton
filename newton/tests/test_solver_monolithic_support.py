# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent contact integration without the retired P1 calibration driver."""

import unittest

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p1q3_oracle import integrate_contact_over_mesh


def test_independent_plane_oracle(test, device):
    """Integrate the hinge independently of Warp records and P1Q3 slots."""
    del device
    points = np.array(((0, 0, 0), (0.04, 0, 0), (0, 0.04, 0), (0, 0, 0.04)), dtype=np.float64)
    faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    oracle = integrate_contact_over_mesh(
        points,
        faces,
        shape="plane",
        center=np.array((0.0, 0.0, 0.00025)),
        scale=np.zeros(3),
        particle_radius=0.0001,
        stiffness=2.0e5,
    )
    test.assertGreater(oracle.force_magnitude, 0.056)
    test.assertLess(abs(0.056 / oracle.force_magnitude - 1.0), 0.05)
    test.assertLessEqual(oracle.force_magnitude_absolute_error / oracle.force_magnitude, 1.0e-3)


def test_independent_oracle_resolves_sample_between_box(test, device):
    """Do not infer zero contact when all three production samples miss."""
    del device
    points = np.array(((0, 0, 0), (0.04, 0, 0), (0, 0.04, 0), (0, 0, 0.04)), dtype=np.float64)
    oracle = integrate_contact_over_mesh(
        points,
        np.array(((0, 2, 1),), dtype=np.int32),
        shape="box",
        center=np.array((0.04 / 3, 0.04 / 3, -0.0004)),
        scale=np.array((0.001, 0.001, 0.0006)),
        particle_radius=0.0001,
        stiffness=2.0e5,
    )
    test.assertGreater(oracle.force_magnitude, 0.0)
    test.assertGreater(oracle.active_area, 0.0)
    shallower = integrate_contact_over_mesh(
        points,
        np.array(((0, 2, 1),), dtype=np.int32),
        shape="box",
        center=np.array((0.04 / 3, 0.04 / 3, -0.0004)),
        scale=np.array((0.001, 0.001, 0.0006)),
        particle_radius=0.0001,
        stiffness=2.0e5,
        max_depth=6,
    )
    test.assertLess(abs(shallower.force_magnitude / oracle.force_magnitude - 1.0), 1.0e-3)


class TestMonolithicSupport(unittest.TestCase):
    """Preserve independent integral and missed-sample reference checks."""


for function in (test_independent_plane_oracle, test_independent_oracle_resolves_sample_between_box):
    add_function_test(TestMonolithicSupport, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
