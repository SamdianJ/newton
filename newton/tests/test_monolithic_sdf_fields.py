# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the two smooth scalar fields independently of the contact solver."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry import GeoType
from newton._src.geometry.sdf_texture import TextureSDFData, create_sparse_sdf_textures
from newton._src.solvers.monolithic.contact import ContactObjective, ContactStatus
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, SmoothSDFSet, sample_smooth
from newton.tests.test_monolithic_contact_components import fixture, state, tangent
from newton.tests.test_monolithic_contact_mesh import synthetic
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.build_smooth_sdf import build_coefficients
from scripts.monolithic_reference.contact_oracle import evaluate as contact_oracle
from scripts.monolithic_reference.smooth_sdf_oracle import evaluate
from scripts.monolithic_reference.verify_smooth_sdf import geometry_gate


def grid():
    """Provide a nonseparable field with nonunit gradients on unequal grid spacing."""
    lo, hi = np.array([-0.12, -0.09, -0.07]), np.array([0.13, 0.11, 0.08])
    axes = [np.linspace(a, b, n) for a, b, n in zip(lo, hi, (7, 6, 5), strict=True)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    values = 0.03 + 0.8 * x - 0.6 * y + 1.1 * z + 0.2 * np.sin(8 * x) * np.cos(9 * y) * z
    return lo, hi, values


def sample(field, points):
    """Evaluate value, gradient and diagnostic Hessian on the actual Warp path."""
    points = wp.array(points, dtype=wp.vec3d, device=field.device)
    result = wp.empty((len(points), 13), dtype=wp.float64, device=field.device)
    wp.launch(sample_smooth, len(points), [field.data, points, result], device=field.device)
    return result.numpy()


def test_reference(test, device):
    """Match independent polynomials or SciPy splines, including the legacy extension."""
    lo, hi, values = grid()
    points = np.random.default_rng(972).uniform(lo - 0.03, hi + 0.03, (47, 3))
    nodes = (
        np.array(
            np.meshgrid(
                *[np.linspace(a, b, n) for a, b, n in zip(lo, hi, values.shape[::-1], strict=True)], indexing="ij"
            )
        )
        .reshape(3, -1)
        .T
    )
    for kind in ("c1-hermite", "c2-bspline"):
        with test.subTest(kind=kind):
            field = SmoothSDF(
                build_coefficients(values, (hi - lo) / (np.array(values.shape[::-1]) - 1), kind),
                lo,
                hi,
                values.shape[::-1],
                kind,
                device=device,
            )
            actual = sample(field, points)
            expected = evaluate(values, lo, hi, points, kind)
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
            reconstructed = sample(field, nodes)[:, 0].reshape(values.shape[::-1]).transpose(2, 1, 0)
            np.testing.assert_allclose(reconstructed, values, rtol=0, atol=2e-14)
            direction = np.array([0.31, -0.29, 0.23])
            h = 1e-7
            upper, lower = sample(field, points + h * direction), sample(field, points - h * direction)
            np.testing.assert_allclose(
                (upper[:, 0] - lower[:, 0]) / (2 * h), actual[:, 1:4] @ direction, rtol=2e-7, atol=1e-8
            )
            np.testing.assert_allclose(
                (upper[:, 1:4] - lower[:, 1:4]) / (2 * h),
                actual[:, 4:].reshape(-1, 3, 3) @ direction,
                rtol=2e-6,
                atol=2e-7,
            )


def test_continuity(test, device):
    """Check face, edge and vertex limits without requiring a C1 Hessian to be continuous."""
    lo, hi, values = grid()
    dx = (hi - lo) / (np.array(values.shape[::-1]) - 1)
    for kind in ("c1-hermite", "c2-bspline"):
        field = SmoothSDF(build_coefficients(values, dx, kind), lo, hi, values.shape[::-1], kind, device=device)
        for index in ([2, 2.37, 1.46], [2, 2, 1.46], [2, 2, 2]):
            center = lo + dx * index
            directions = np.array([[1, 0, 0], [1, -1, 0], [1, -1, 1]])
            for direction in directions:
                limits = sample(field, np.array([center - 1e-12 * dx * direction, center + 1e-12 * dx * direction]))
                end = 13 if kind == "c2-bspline" else 4
                np.testing.assert_allclose(limits[0, :end], limits[1, :end], rtol=0, atol=1e-9)


def test_graph(test, device):
    """Replay immutable field buffers through a CUDA graph."""
    if wp.get_device(device).is_cpu:
        test.skipTest("CUDA graph requires CUDA")
    lo, hi, values = grid()
    for kind in ("c1-hermite", "c2-bspline"):
        field = SmoothSDF(
            build_coefficients(values, (hi - lo) / (np.array(values.shape[::-1]) - 1), kind),
            lo,
            hi,
            values.shape[::-1],
            kind,
            device=device,
        )
        points = wp.array([[0.011, 0.022, 0.033]], dtype=wp.vec3d, device=device)
        result = wp.empty((1, 13), dtype=wp.float64, device=device)
        wp.launch(sample_smooth, 1, [field.data, points, result], device=device)
        expected = result.numpy()
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(sample_smooth, 1, [field.data, points, result], device=device)
        result.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(result.numpy(), expected)


def test_manufactured_and_scale(test, device):
    """Reproduce a multilinear field and physical derivatives under uniform scaling."""
    lo, hi, _ = grid()
    z, y, x = np.meshgrid(
        np.linspace(lo[2], hi[2], 5), np.linspace(lo[1], hi[1], 6), np.linspace(lo[0], hi[0], 7), indexing="ij"
    )
    values = 0.03 + x + 2 * y - 3 * z + 4 * x * y - 2 * x * z + y * z + 3 * x * y * z
    points = np.random.default_rng(189).uniform(lo, hi, (29, 3))
    x, y, z = points.T
    expected = np.zeros((len(points), 13))
    expected[:, 0] = 0.03 + x + 2 * y - 3 * z + 4 * x * y - 2 * x * z + y * z + 3 * x * y * z
    expected[:, 1:4] = np.stack(
        (1 + 4 * y - 2 * z + 3 * y * z, 2 + 4 * x + z + 3 * x * z, -3 - 2 * x + y + 3 * x * y), axis=1
    )
    expected[:, 5] = expected[:, 7] = 4 + 3 * z
    expected[:, 6] = expected[:, 10] = -2 + 3 * y
    expected[:, 9] = expected[:, 11] = 1 + 3 * x
    for kind in ("c1-hermite", "c2-bspline"):
        for scale in (0.3, 1.0, 2.7):
            coeff = build_coefficients(scale * values, scale * (hi - lo) / [6, 5, 4], kind)
            field = SmoothSDF(coeff, scale * lo, scale * hi, [7, 6, 5], kind, device=device)
            actual = sample(field, scale * points)
            target = expected.copy()
            target[:, 0] *= scale
            target[:, 4:] /= scale
            np.testing.assert_allclose(actual, target, rtol=0, atol=2e-11)


def test_natural_boundary(test, device):
    """Natural conditions constrain the pure normal second derivative on all six faces."""
    lo, hi, values = grid()
    coeff = build_coefficients(values, (hi - lo) / [6, 5, 4], "c2-bspline")
    field = SmoothSDF(coeff, lo, hi, [7, 6, 5], "c2-bspline", device=device)
    points = np.random.default_rng(301).uniform(lo, hi, (13, 3))
    for axis in range(3):
        for end in (lo, hi):
            face = points.copy()
            face[:, axis] = end[axis]
            np.testing.assert_allclose(sample(field, face)[:, 4 + 4 * axis], 0, rtol=0, atol=2e-11)


def test_binding(test, device):
    """Detect replacement, data edits and descriptor edits before graph replay."""
    lo, hi, values = grid()
    for kind in ("c1-hermite", "c2-bspline"):
        field = SmoothSDF(
            build_coefficients(values, (hi - lo) / [6, 5, 4], kind), lo, hi, [7, 6, 5], kind, device=device
        )
        bundle = SmoothSDFSet([field], source_identity={"asset": "test-grid"})
        bundle.validate_identity()
        original = field.data.coefficients
        field.data.coefficients = wp.clone(original)
        with test.assertRaisesRegex(ValueError, "replaced"):
            bundle.validate_identity()
        field.data.coefficients = original
        saved = original.numpy()
        original.zero_()
        with test.assertRaisesRegex(ValueError, "coefficients changed"):
            bundle.validate_identity()
        original.assign(saved)
        bundle.table.zero_()
        with test.assertRaisesRegex(ValueError, "descriptors changed"):
            bundle.validate_identity()


def test_smooth_contact(test, device):
    """Keep E/r, the declared PSD approximation and coverage guards on the new field."""

    if wp.get_device(device).is_cpu:
        test.skipTest("Full mesh contact keeps the existing CUDA texture prerequisite")
    data = synthetic()
    descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device)
    model, xn = fixture(device, "plane")
    model.shape_type.assign(np.array([int(GeoType.MESH)], dtype=np.int32))
    model.shape_scale.assign(np.ones((1, 3), dtype=np.float32))
    model._shape_sdf_index.assign(np.array([0], dtype=np.int32))
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    xn[:, 2] -= 0.0003
    axis = np.linspace(-0.125, 0.125, 5)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    values = 1.5 * z + 0.25 * x + 0.5 * x * y + x * y * z + 0.003 * np.sin(10 * x) * np.cos(12 * y)
    for kind in ("c1-hermite", "c2-bspline"):
        field = SmoothSDF(
            build_coefficients(values, [0.0625] * 3, kind), [-0.125] * 3, [0.125] * 3, [5] * 3, kind, device=device
        )
        bundle = SmoothSDFSet([field], source_identity={"asset": "synthetic"})
        for law in ("C-EE", "C-IE", "C-II"):
            obj = ContactObjective(
                model,
                np.arange(4, dtype=np.int32),
                q_dof_count=2,
                contact_model=law,
                normal_stiffness=1e7,
                friction_coefficient=0.5,
                friction_velocity_scale=0.01,
                smoothing_width=1e-4,
                particle_radius=0.001,
                smooth_sdf=bundle,
            )
            q = np.array([0.003, 2e-6])
            current = xn + np.array([1e-6, 2e-6, 1e-6])
            obj.begin_step(*state(model, xn, np.zeros(2)))
            obj.evaluate(*state(model, current, q), dt=0.001)
            test.assertEqual(obj.flags.numpy()[0], 0)
            oracle_field = {"smooth_grid": values, "lower": [-0.125] * 3, "upper": [0.125] * 3, "representation": kind}
            oracle = contact_oracle(model, q, current, np.zeros(2), xn, law, kinds=[oracle_field])
            np.testing.assert_allclose(obj.energy.numpy(), oracle[0], atol=1e-15)
            np.testing.assert_allclose(obj.residual.numpy(), oracle[1], atol=1e-12)
            workspace, generation, _ = tangent(obj)
            test.assertEqual(workspace.finalize_assembly(generation=generation).status, 0)
            matrix = workspace.densify_for_test(generation=generation).raw_matrix
            np.testing.assert_allclose(matrix, oracle[3], rtol=2e-5, atol=2e-4)
            test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * np.linalg.norm(matrix))
            direction = np.random.default_rng(187).normal(size=14)
            direction /= np.linalg.norm(direction)
            repeated_e, repeated_r = [], []
            for _ in range(32):
                obj.evaluate(*state(model, current, q), dt=0.001)
                repeated_e.append(obj.energy.numpy().sum())
                repeated_r.append(obj.residual.numpy())
            errors = np.asarray(repeated_r) - oracle[1]
            test.assertLessEqual(8 * np.max(abs(np.asarray(repeated_e) - oracle[0].sum())), 5.820766091346741e-11)
            test.assertLessEqual(8 * np.max(abs(errors)), 2.0935893019213836e-9)
            test.assertLessEqual(
                8 * np.max(abs(errors @ direction)), 2.0935893019213836e-9 * np.linalg.norm(direction, 1)
            )
            h = 1e-8
            energies, residuals = [], []
            for sign in (1, -1):
                obj.evaluate(
                    *state(model, current + sign * h * direction[2:].reshape(4, 3), q + sign * h * direction[:2]),
                    dt=0.001,
                )
                energies.append(obj.energy.numpy().sum())
                residuals.append(obj.residual.numpy())
            np.testing.assert_allclose(
                (residuals[0] - residuals[1]) / (2 * h), oracle[2] @ direction, rtol=1e-5, atol=1e-5
            )
            if law == "C-EE":
                test.assertAlmostEqual((energies[0] - energies[1]) / (2 * h), oracle[1] @ direction, delta=1e-7)
            # An active contact on an unqualified box face must report coverage failure.
            outside = xn.copy()
            outside[:, 2] -= 0.15
            obj.begin_step(*state(model, outside, np.zeros(2)))
            test.assertEqual(obj.start_flags.numpy()[0], ContactStatus.UNCERTIFIED_BOUNDARY)
        test.assertIsNotNone(coarse)
        test.assertIsNotNone(fine)


class TestSmoothSDF(unittest.TestCase):
    """Exercise both approved representations."""

    def test_geometry_gate(self):
        baseline = {
            "max_error_m": 0.0002,
            "rms_error_m": 0.00001,
            "finite": True,
            "minimum_gradient_norm": 0.5,
            "sign_errors_outside_0_4mm": 0,
        }
        self.assertTrue(geometry_gate(baseline, baseline))
        self.assertFalse(geometry_gate({**baseline, "max_error_m": 0.000201, "rms_error_m": 0.000001}, baseline))
        self.assertFalse(geometry_gate({**baseline, "max_error_m": 0.0001, "rms_error_m": 0.000011}, baseline))
        self.assertFalse(geometry_gate({**baseline, "finite": False}, baseline))
        self.assertFalse(geometry_gate({**baseline, "minimum_gradient_norm": 0}, baseline))
        self.assertFalse(geometry_gate({**baseline, "sign_errors_outside_0_4mm": 1}, baseline))


for _device in get_test_devices():
    add_function_test(TestSmoothSDF, "test_binding", test_binding, devices=[_device])
    add_function_test(TestSmoothSDF, "test_smooth_contact", test_smooth_contact, devices=[_device])
    add_function_test(TestSmoothSDF, "test_reference", test_reference, devices=[_device])
    add_function_test(TestSmoothSDF, "test_continuity", test_continuity, devices=[_device])
    add_function_test(TestSmoothSDF, "test_graph", test_graph, devices=[_device])
    add_function_test(TestSmoothSDF, "test_manufactured_and_scale", test_manufactured_and_scale, devices=[_device])
    add_function_test(TestSmoothSDF, "test_natural_boundary", test_natural_boundary, devices=[_device])


if __name__ == "__main__":
    unittest.main()
