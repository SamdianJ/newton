# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve cloth sample identities when grouping a single sphere query."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_actor_pairs import native_inputs
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_contact_materials import query
from newton.tests.unittest_utils import add_function_test, get_test_devices


def fixture(device, precision, *, capacity=None, spheres=1):
    """Build two independent cloth faces with six identities each per sphere."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for i in range(spheres):
        builder.add_shape_sphere(-1, radius=0.5, xform=wp.transform((i * 2, 0, 0), wp.quat_identity()))
    cloth = []
    for i in range(2):
        first = builder.particle_count
        for x, y in ((-0.02, -0.02), (0.02, -0.02), (0, 0.02)):
            builder.add_particle(wp.vec3(x + i * 2, y, 0.501), wp.vec3(0), 0.1)
        face = builder.tri_count
        builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
        points = np.asarray(builder.particle_q[first:], dtype=float)
        mass = np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 6
        builder.particle_mass[first:] = [mass] * 3
        cloth.append(
            SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 1), thickness=0.01, surface_density=1)
        )
    solver = SolverMonolithic(
        builder.finalize(device=device),
        cloth_inputs=cloth,
        precision=precision,
        contact_candidate_capacity=capacity,
        contact_factor_capacity=128,
        mass_mode="lumped",
        friction_velocity_scale=0.1,
        contact_stiffness=100,
        use_cuda_graph=False,
    )
    state = solver.create_state(dt=0.001)
    query(solver, state, begin=True)
    return solver, state, solver._contact._spatial


def test_grouped_candidates(test, device):
    """Avoid repeated BVH queries while preserving swept candidates, response and replay."""
    for precision in ("fp32", "fp64"):
        solver, state, spatial = fixture(device, precision)
        q = solver._kinematics.gather_q(state)
        x = state.particle_q
        original = x.numpy().copy()
        pointers = [a.ptr for a in spatial.buffers()]
        # A single target needs no BVH traversal/refit, even for six face identities.
        with patch.object(spatial.query.bvh, "refit", side_effect=AssertionError("redundant BVH query")):
            spatial.update(q, x)
        spatial.check()
        test.assertEqual(int(spatial.count.numpy()[0]), 6)
        expected = spatial.identities.numpy()[:6].copy()
        samples = solver._contact._batch.all_samples
        np.testing.assert_array_equal(expected, np.flatnonzero(samples.faces.numpy() == 0))
        # Compare the unchanged general query, including its outward FP32 conversion.
        spatial.query.update(spatial.face_lo, spatial.face_hi, spatial.target_lo, spatial.target_hi, sort=False)
        spatial._order(spatial.capacity)
        np.testing.assert_array_equal(spatial.identities.numpy()[: int(spatial.count.numpy()[0])], expected)
        query(solver, state)
        contact = solver._contact
        values = [a.numpy().copy() for a in (contact.energy, contact.residual)]
        matrix = batch_matrix(contact._batch)
        spatial.query.update(spatial.face_lo, spatial.face_hi, spatial.target_lo, spatial.target_hi, sort=False)
        spatial._order(spatial.capacity)
        inputs, options = native_inputs(solver, state)
        contact.evaluate(*inputs, dt=0.001, **options)
        for array, before in zip((contact.energy, contact.residual), values, strict=True):
            np.testing.assert_array_equal(array.numpy(), before)
        np.testing.assert_array_equal(batch_matrix(contact._batch), matrix)
        # Keep the step-start envelope when the current face moves out of contact.
        x.assign(original + np.array((0, 0, 1)))
        spatial.update(q, x)
        spatial.check()
        np.testing.assert_array_equal(spatial.identities.numpy()[:6], expected)
        spatial.begin_step(q, x, solver._kinematics.poses)
        spatial.update(q, x)
        test.assertEqual(int(spatial.count.numpy()[0]), 0)
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                spatial.update(q, x)
        both = original.copy()
        both[3:, 0] -= 2
        later = original + np.array((0, 0, 1))
        later[3:] = both[3:]
        invalid = both.copy()
        invalid[0, 2] = np.nan
        for positions, faces in (
            (later, (1,)),
            (both, (0, 1)),
            (invalid, None),
            (original + np.array((0, 0, 1)), ()),
            (both, (0, 1)),
        ):
            x.assign(positions)
            if device.is_cuda:
                wp.capture_launch(capture.graph)
            else:
                spatial.update(q, x)
            if faces is None:
                with test.assertRaises(ValueError):
                    spatial.check()
                test.assertEqual(int(spatial.count.numpy()[0]), 0)
            else:
                spatial.check()
                expected = np.flatnonzero(np.isin(samples.faces.numpy(), faces))
                test.assertEqual(int(spatial.count.numpy()[0]), len(expected))
                np.testing.assert_array_equal(spatial.identities.numpy()[: len(expected)], expected)
                spatial.query.update(spatial.face_lo, spatial.face_hi, spatial.target_lo, spatial.target_hi, sort=False)
                spatial._order(spatial.capacity)
                np.testing.assert_array_equal(spatial.identities.numpy()[: int(spatial.count.numpy()[0])], expected)
        test.assertEqual(pointers, [a.ptr for a in spatial.buffers()])


def test_grouped_fallback(test, device):
    """Retain generic capacity clipping and multiple-target candidate coverage."""
    for capacity, spheres in ((0, 1), (1, 1), (5, 1), (6, 1), (11, 1), (None, 2)):
        solver, state, spatial = fixture(device, "fp32", capacity=capacity, spheres=spheres)
        with patch.object(spatial.query, "update", wraps=spatial.query.update) as generic:
            spatial.update(solver._kinematics.gather_q(state), state.particle_q)
            test.assertEqual(generic.call_count, 1)
        expected = 6 * spheres
        test.assertEqual(int(spatial.count.numpy()[0]), min(spatial.capacity, expected))
        test.assertEqual(int(spatial.query.dropped.numpy()[0]), max(0, expected - spatial.capacity))
        ids = spatial.identities.numpy()[: int(spatial.count.numpy()[0])]
        test.assertEqual(len(ids), len(set(ids)))


class TestClothSphereQuery(unittest.TestCase):
    pass


for _test in (test_grouped_candidates, test_grouped_fallback):
    add_function_test(TestClothSphereQuery, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
