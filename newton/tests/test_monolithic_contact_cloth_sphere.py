# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve cloth sample identities when grouping a single sphere query."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.contact_sampling import _surface_samples
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf
from newton.tests.test_monolithic_contact_actor_pairs import native_inputs
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_contact_materials import query
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.run_cloth_stack import _sphere_mesh


def fixture(device, precision, *, capacity=None, spheres=1, mesh=False):
    """Build two independent cloth faces with six identities each per sphere."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for i in range(spheres):
        transform = wp.transform((i * 2, 0, 0), wp.quat_identity())
        if mesh:
            source = _sphere_mesh(0.5, 8, 16)
            source.build_sdf(
                device=device, max_resolution=32, margin=0.1, narrow_band_range=(-0.2, 0.2), texture_format="float32"
            )
            builder.add_shape_mesh(-1, mesh=source, xform=transform)
        else:
            builder.add_shape_sphere(-1, radius=0.5, xform=transform)
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
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        cloth_inputs=cloth,
        precision=precision,
        contact_candidate_capacity=capacity,
        contact_factor_capacity=128,
        mass_mode="lumped",
        friction_velocity_scale=0.1,
        contact_stiffness=100,
        use_cuda_graph=False,
        smooth_sdf=build_smooth_sdf(model) if mesh else None,
    )
    state = solver.create_state(dt=0.001)
    query(solver, state, begin=True)
    return solver, state, solver._contact._spatial


def test_grouped_candidates(test, device, *, mesh=False):
    """Avoid repeated BVH queries while preserving swept candidates, response and replay."""
    for precision in ("fp32", "fp64"):
        solver, state, spatial = fixture(device, precision, mesh=mesh)
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
        if mesh:
            # Preserve candidates in the SDF domain outside the geometric sphere.
            outside = original.copy()
            outside[:, 2] = 0.55
            x.assign(outside)
            spatial.begin_step(q, x, solver._kinematics.poses)
            spatial.update(q, x)
            expected = spatial.identities.numpy()[: int(spatial.count.numpy()[0])].copy()
            test.assertEqual(len(expected), 6)
            test.assertGreater(float(spatial.target_local_hi.numpy()[0, 2]), 0.55)
            spatial.query.update(spatial.face_lo, spatial.face_hi, spatial.target_lo, spatial.target_hi, sort=False)
            spatial._order(spatial.capacity)
            np.testing.assert_array_equal(spatial.identities.numpy()[: int(spatial.count.numpy()[0])], expected)


def test_grouped_fallback(test, device, *, mesh=False):
    """Retain generic capacity clipping and multiple-target candidate coverage."""
    for capacity, spheres in ((0, 1), (1, 1), (5, 1), (6, 1), (11, 1), (None, 2)):
        solver, state, spatial = fixture(device, "fp32", capacity=capacity, spheres=spheres, mesh=mesh)
        with patch.object(spatial.query, "update", wraps=spatial.query.update) as generic:
            spatial.update(solver._kinematics.gather_q(state), state.particle_q)
            test.assertEqual(generic.call_count, 1)
        expected = 6 * spheres
        test.assertEqual(int(spatial.count.numpy()[0]), min(spatial.capacity, expected))
        test.assertEqual(int(spatial.query.dropped.numpy()[0]), max(0, expected - spatial.capacity))
        ids = spatial.identities.numpy()[: int(spatial.count.numpy()[0])]
        test.assertEqual(len(ids), len(set(ids)))


def test_grouped_mesh(test, device):
    """Preserve mesh SDF envelopes, contact response, graph replay and generic fallbacks."""
    if not device.is_cuda:
        test.skipTest("Mesh SDF textures require CUDA")
    test_grouped_candidates(test, device, mesh=True)
    test_grouped_fallback(test, device, mesh=True)


class TestClothSphereQuery(unittest.TestCase):
    def test_sphere_mesh_geometry(self):
        """Build outward positive-area faces accepted by the actual mesh target sampler."""
        mesh = _sphere_mesh(0.12, 32, 64)
        _points, _normals, weights = _surface_samples(newton.GeoType.MESH, np.ones(3), mesh)
        self.assertTrue(np.all(weights > 0))
        faces = mesh.vertices[mesh.indices.reshape(-1, 3)].astype(float)
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        self.assertTrue(np.all(np.sum(normals * faces.mean(axis=1), axis=1) > 0))
        self.assertEqual(len(faces), 3968)


for _test in (test_grouped_candidates, test_grouped_fallback, test_grouped_mesh):
    add_function_test(TestClothSphereQuery, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
