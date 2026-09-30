# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actual texture-center FP64 queries, scale and exterior scalar derivatives."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry import GeoType
from newton._src.geometry.sdf_texture import TextureSDFData, create_sparse_sdf_textures
from newton._src.solvers.monolithic.contact_geometry import _shape64
from newton.tests.test_monolithic_contact_components import contact, fixture, state, tangent
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_oracle import evaluate, mesh_reference


@wp.kernel
def query(
    points: wp.array[wp.vec3d], table: wp.array[TextureSDFData], scale: wp.float64, result: wp.array2d[wp.float64]
):
    i = wp.tid()
    value, g = _shape64(int(GeoType.MESH), wp.vec3d(scale), points[i], 0, table)
    result[i, 0] = value
    for a in range(3):
        result[i, a + 1] = g[a]


def synthetic():
    # A exactly representable polynomial with nonunit raw gradient. The first
    # coarse cell has a fine block; the rest use coarse samples of the same field.
    def field(x, y, z):
        return 1.5 * z + 0.25 * x + 0.5 * x * y + x * y * z

    axis = np.linspace(-0.125, 0.125, 3)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    fine_axis = np.linspace(-0.125, 0, 3)
    fz, fy, fx = np.meshgrid(fine_axis, fine_axis, fine_axis, indexing="ij")
    slots = np.full((2, 2, 2), 0xFFFFFFFF, dtype=np.uint32)
    slots[0, 0, 0] = 0
    return {
        "coarse_sdf": field(x, y, z).astype(np.float32),
        "subgrid_data": field(fx, fy, fz).astype(np.float32),
        "subgrid_start_slots": slots,
        "subgrid_size": 2,
        "cell_size": np.full(3, 0.0625),
        "min_extents": np.full(3, -0.125),
        "max_extents": np.full(3, 0.125),
        "subgrids_min_sdf_value": 0.0,
        "subgrids_sdf_value_range": 1.0,
    }


def test_texture_geometry(test, device):
    if wp.get_device(device).is_cpu:
        test.skipTest("Mesh SDF textures require CUDA; analytic CPU coverage is separate")
    data = synthetic()
    rng = np.random.default_rng(971)
    points = rng.uniform(-0.19, 0.19, (128, 3))
    # Fixed-branch derivatives adjacent to, but not crossing, a voxel interface.
    points = np.vstack((points, [-1e-7, -0.07, -0.06], [1e-7, -0.07, -0.06]))
    expected = mesh_reference(data, points)
    for paired in (False, True):
        descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device, paired_samples=paired)
        for baked in (False, True):
            descriptor.scale_baked = baked
            table = wp.array([descriptor], dtype=TextureSDFData, device=device)
            for scale in (0.5, 2.0):
                effective = 1.0 if baked else scale
                p = wp.array(points * effective, dtype=wp.vec3d, device=device)
                output = wp.zeros((len(points), 4), dtype=wp.float64, device=device)
                wp.launch(query, len(points), [p, table, scale, output], device=device)
                reference = expected.copy()
                reference[:, 0] *= effective
                np.testing.assert_allclose(output.numpy(), reference, rtol=0, atol=2e-14)
                direction = np.tile([0.31, -0.29, 0.23], (len(points), 1))
                derivative = np.sum(reference[:, 1:] * direction, axis=1)
                errors = []
                for h in (1e-8, 3e-9, 1e-9):
                    values = []
                    for sign in (1, -1):
                        p.assign(points * effective + sign * h * direction)
                        wp.launch(query, len(points), [p, table, scale, output], device=device)
                        values.append(output.numpy()[:, 0])
                    errors.append(abs((values[0] - values[1]) / (2 * h) - derivative))
                test.assertTrue(np.all(np.sum(np.array(errors) < 3e-6, axis=0) >= 2))
            # Keep texture owners alive until all launches using the descriptor finish.
            test.assertIsNotNone(coarse)
            test.assertIsNotNone(fine)


def test_mesh_contact_and_scale_gates(test, device):
    if wp.get_device(device).is_cpu:
        test.skipTest("Mesh contact requires CUDA")
    data = synthetic()
    descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device)
    model, xn = fixture(device, "plane")
    model.shape_type.assign(np.array([int(GeoType.MESH)], dtype=np.int32))
    model.shape_scale.assign(np.ones((1, 3), dtype=np.float32))
    model._shape_sdf_index.assign(np.array([0], dtype=np.int32))
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    xn[:, 2] -= 0.0003
    q = np.zeros(2)
    for law in ("C-EE", "C-IE", "C-II"):
        obj = contact(model, law)
        obj.begin_step(*state(model, xn, q))
        x = xn + np.array([1e-6, 2e-6, 1e-6])
        obj.evaluate(*state(model, x, q), dt=0.001)
        test.assertEqual(obj.flags.numpy()[0], 0)
        expected = mesh_reference(data, obj.current.y.numpy())
        np.testing.assert_allclose(obj.current.distance.numpy(), expected[:, 0], atol=1e-15)
        np.testing.assert_allclose(obj.current.gradient.numpy(), expected[:, 1:], atol=1e-14)
        oracle = evaluate(model, q, x, q, xn, law, kinds=[data])
        np.testing.assert_allclose(obj.energy.numpy(), oracle[0], atol=1e-15)
        np.testing.assert_allclose(obj.residual.numpy(), oracle[1], atol=1e-12)
        workspace, generation, _ = tangent(obj)
        test.assertEqual(workspace.finalize_assembly(generation=generation).status, 0)
        np.testing.assert_allclose(
            workspace.densify_for_test(generation=generation).raw_matrix, oracle[3], rtol=2e-5, atol=2e-4
        )
        r = obj.residual.numpy().copy()
        rng = np.random.default_rng(977)
        direction = rng.normal(size=14)
        direction /= np.linalg.norm(direction)
        for h in (1e-8, 3e-9):
            values = []
            residuals = []
            for sign in (1, -1):
                obj.evaluate(
                    *state(model, x + sign * h * direction[2:].reshape(4, 3), q + sign * h * direction[:2]), dt=0.001
                )
                test.assertEqual(obj.flags.numpy()[0], 0)
                values.append(obj.energy.numpy().sum())
                residuals.append(obj.residual.numpy().copy())
            dr = (residuals[0] - residuals[1]) / (2 * h)
            test.assertLess(np.linalg.norm(dr - oracle[2] @ direction) / np.linalg.norm(oracle[2] @ direction), 1e-5)
            if law == "C-EE":
                test.assertAlmostEqual((values[0] - values[1]) / (2 * h), r @ direction, delta=1e-7)
    for scale in ([1, 2, 1], [-1, -1, -1], [0, 0, 0]):
        model.shape_scale.assign(np.array([scale], dtype=np.float32))
        with test.assertRaisesRegex(ValueError, "uniform"):
            contact(model, "C-EE")
    test.assertIsNotNone(coarse)
    test.assertIsNotNone(fine)


class TestContactMesh(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestContactMesh, "test_texture_geometry", test_texture_geometry, devices=[_device])
    add_function_test(
        TestContactMesh, "test_mesh_contact_and_scale_gates", test_mesh_contact_and_scale_gates, devices=[_device]
    )

if __name__ == "__main__":
    unittest.main()
