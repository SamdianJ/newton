# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the shared tiny fixture and offline reference contracts."""

import unittest

import numpy as np

from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestMonolithicReference(unittest.TestCase):
    pass


def test_tiny_fixture(test, device):
    """Build the frozen fourteen-scalar fixture with explicit physics inputs."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state, spec = fixture.model, fixture.state, fixture.spec
    test.assertEqual(model.joint_dof_count, 2)
    test.assertEqual(model.particle_count, 4)
    test.assertEqual(spec.scalar_dof_count, 14)
    np.testing.assert_array_equal(model.tet_indices.numpy(), [[0, 1, 2, 3]])
    test.assertEqual(
        {tuple(sorted(f)) for f in model.tri_indices.numpy()}, {tuple(sorted(f)) for f in spec.boundary_faces}
    )
    np.testing.assert_array_equal(state.particle_q.numpy(), np.asarray(spec.rest_positions, dtype=np.float32))
    np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((4, 3)))
    np.testing.assert_array_equal(state.joint_q.numpy(), np.asarray(spec.q0, dtype=np.float32))
    np.testing.assert_array_equal(state.joint_qd.numpy(), np.asarray(spec.qd0, dtype=np.float32))
    np.testing.assert_allclose(model.particle_mass.numpy(), np.full(4, 0.001), rtol=2e-7)
    np.testing.assert_array_equal(model.particle_radius.numpy(), np.full(4, 0.001, dtype=np.float32))
    np.testing.assert_array_equal(model.tet_materials.numpy(), [[3846.15380859375, 5769.23095703125, 0.0]])
    np.testing.assert_array_equal(model.tri_materials.numpy(), np.zeros((4, 5)))
    np.testing.assert_array_equal(model.edge_bending_properties.numpy(), np.zeros((6, 2)))
    np.testing.assert_array_equal(model.body_mass.numpy(), [1.0, 0.5])
    np.testing.assert_allclose(model.body_com.numpy(), [[0.05, 0, 0], [0.02, 0, 0]])
    np.testing.assert_allclose(
        model.body_inertia.numpy(), [np.diag([0.002, 0.003, 0.004]), np.diag([0.001, 0.0015, 0.002])]
    )
    test.assertNotEqual(state.joint_q.ptr, fixture.state_next.joint_q.ptr)
    np.testing.assert_array_equal(state.joint_q.numpy(), fixture.state_next.joint_q.numpy())


for device in get_test_devices():
    add_function_test(TestMonolithicReference, "test_tiny_fixture", test_tiny_fixture, devices=[device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
