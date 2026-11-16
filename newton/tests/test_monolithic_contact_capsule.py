# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capsule contact: analytic field, roundoff magnitude, exact surface samples and resting contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton._src.solvers.monolithic.contact_roundoff import analytic_geometry_magnitude
from newton._src.solvers.monolithic.contact_sampling import _surface_samples
from newton.tests.test_monolithic_contact_unsupported_shapes import _query
from newton.tests.unittest_utils import add_function_test, get_test_devices

RADIUS, HALF = 0.02, 0.04


def _reference(points, radius=RADIUS, half=HALF):
    """Long-double distance to the segment minus the radius, and the unit gradient."""
    p = np.asarray(points, dtype=np.longdouble)
    axial = np.clip(p[:, 2], -half, half)
    offset = p - np.stack((np.zeros(len(p)), np.zeros(len(p)), axial), axis=1).astype(np.longdouble)
    length = np.sqrt((offset * offset).sum(axis=1))
    return length - radius, offset / length[:, None]


def _seam_points():
    """Points near the cap/band seams (z = +-h), off-axis, inside and outside, plus random ones."""
    rng = np.random.default_rng(11)
    z = np.r_[(np.array([-HALF, HALF]) + rng.normal(scale=1e-6, size=(16, 2))).ravel(), rng.uniform(-0.1, 0.1, 32)]
    angle = rng.uniform(0, 2 * np.pi, len(z))
    rho = rng.uniform(0.001, 0.05, len(z))
    return np.stack((rho * np.cos(angle), rho * np.sin(angle), z), axis=1)


def test_field_matches_long_double(test, device):
    points = _seam_points()
    value_ref, gradient_ref = _reference(points)
    for precision, dtype, tolerance in (("fp64", np.float64, 1e-15), ("fp32", np.float32, 2e-7)):
        with test.subTest(precision=precision):
            values, gradients = _query(device, precision, GeoType.CAPSULE, (RADIUS, HALF, 0.0), points.astype(dtype))
            scale = 1.0 + np.abs(points).max()
            np.testing.assert_allclose(values, value_ref.astype(float), atol=8 * tolerance * scale, rtol=0)
            np.testing.assert_allclose(gradients, gradient_ref.astype(float), atol=8 * tolerance * 10, rtol=0)


def test_gradient_is_field_derivative(test, device):
    """Central differences of the FP64 field match its gradient away from the axis and the seams' kinks."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.06, 0.06, size=(40, 3))
    points[:, :2] += np.sign(points[:, :2]) * 0.005
    h = 1e-7
    _values, gradients = _query(device, "fp64", GeoType.CAPSULE, (RADIUS, HALF, 0.0), points)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        plus, _ = _query(device, "fp64", GeoType.CAPSULE, (RADIUS, HALF, 0.0), points + step)
        minus, _ = _query(device, "fp64", GeoType.CAPSULE, (RADIUS, HALF, 0.0), points - step)
        np.testing.assert_allclose((plus - minus) / (2 * h), gradients[:, axis], atol=1e-7)


@wp.kernel
def _magnitude64(
    points: wp.array[wp.vec3d],
    scale: wp.vec3d,
    point_magnitude: wp.array[wp.vec3d],
    values: wp.array[wp.float64],
    gradients: wp.array[wp.vec3d],
):
    i = wp.tid()
    y = points[i]
    axial = wp.clamp(y[2], -scale[1], scale[1])
    offset = wp.vec3d(y[0], y[1], y[2] - axial)
    phi = wp.length(offset) - scale[0]
    g = wp.vec3d(0.0)
    if wp.length(offset) > wp.float64(0.0):
        g = offset / wp.length(offset)
    a, ag = analytic_geometry_magnitude(int(GeoType.CAPSULE), scale, y, point_magnitude[i], phi, g, wp.float64(0.0))
    values[i] = a
    gradients[i] = ag


def test_roundoff_magnitude(test, device):
    """Infinite on the axis (the row cannot pass); elsewhere it bounds the observed FP32 field error."""
    points = _seam_points()
    points = np.vstack((points, [[0.0, 0.0, 0.01], [0.0, 0.0, -0.03]]))
    magnitude = np.abs(points)
    pts = wp.array(points, dtype=wp.vec3d, device=device)
    values = wp.zeros(len(points), dtype=wp.float64, device=device)
    gradients = wp.zeros(len(points), dtype=wp.vec3d, device=device)
    pm = wp.array(magnitude, dtype=wp.vec3d, device=device)
    wp.launch(_magnitude64, len(points), [pts, wp.vec3d(RADIUS, HALF, 0.0), pm, values, gradients], device=device)
    a = values.numpy()
    test.assertTrue(np.all(np.isinf(a[-2:])), a[-2:])
    test.assertTrue(np.all(np.isfinite(a[:-2])))
    value_ref, _ = _reference(points[:-2])
    observed, _ = _query(device, "fp32", GeoType.CAPSULE, (RADIUS, HALF, 0.0), points[:-2].astype(np.float32))
    error = np.abs(observed.astype(np.float64) - value_ref.astype(np.float64))
    # A is an operation magnitude: |error| <= c * eps32 * A with a small constant.
    test.assertTrue(np.all(error <= 32 * np.finfo(np.float32).eps * a[:-2] + 1e-12), np.max(error / a[:-2]))


def test_surface_samples(test, device):
    """Exact area 4 pi r^2 + 4 pi r h, points on the surface, exact outward normals; h = 0 is the sphere area."""
    for half in (HALF, 0.0, 0.001):
        with test.subTest(half_height=half):
            points, normals, weights = _surface_samples(GeoType.CAPSULE, np.array([RADIUS, half, 0.0]))
            area = 4 * np.pi * RADIUS**2 + 4 * np.pi * RADIUS * half
            test.assertAlmostEqual(weights.sum() / area, 1.0, places=13)
            test.assertTrue(np.all(weights > 0))
            value, gradient = _reference(points, half=half)
            np.testing.assert_allclose(value.astype(float), 0.0, atol=1e-17)
            np.testing.assert_allclose(normals, gradient.astype(float), atol=1e-15)
            np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=1e-15)
    with test.assertRaisesRegex(ValueError, "Invalid source capsule scale"):
        _surface_samples(GeoType.CAPSULE, np.array([0.0, HALF, 0.0]))


def _free(builder, shape, pos, quat=None, mass=0.1, label="body"):
    quat = quat or wp.quat_identity()
    body = builder.add_link(xform=wp.transform(pos, quat), label=label)
    builder.add_articulation([builder.add_joint_free(body, parent_xform=wp.transform(pos, quat))])
    shape(builder, body, builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False))
    builder.body_mass[body] = mass
    builder.body_inv_mass[body] = 1.0 / mass
    inertia = wp.mat33(np.eye(3) * mass * 1e-3)
    builder.body_inertia[body] = inertia
    builder.body_inv_inertia[body] = wp.inverse(inertia)
    return body


def test_resting_pairs(test, device):
    """A horizontal capsule rests on a plane, on a box, on a sphere and on another capsule: forces carry the
    weights and nothing rolls back."""
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    s = float(np.sqrt(0.5))
    lying = wp.quat(0.0, s, 0.0, s)

    def capsule(builder, body, cfg):
        builder.add_shape_capsule(body, radius=RADIUS, half_height=HALF, cfg=cfg)

    supports = {
        "plane": (None, RADIUS),
        "box": (lambda b, body, cfg: b.add_shape_box(body, hx=0.06, hy=0.06, hz=0.02, cfg=cfg), 0.02 + 0.02 + RADIUS),
        "sphere": ("static", 0.03 + 0.03 + RADIUS),
        "capsule": (
            lambda b, body, cfg: b.add_shape_capsule(
                body, xform=wp.transform((0, 0, 0), wp.quat(s, 0.0, 0.0, s)), radius=RADIUS, half_height=HALF, cfg=cfg
            ),
            RADIUS + RADIUS + RADIUS,
        ),
    }
    for name, (support, top) in supports.items():
        with test.subTest(support=name):
            builder = newton.ModelBuilder()
            plane_cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
            builder.add_shape_plane(body=-1, width=0.0, length=0.0, cfg=plane_cfg)
            if support == "static":
                # A point contact under a balanced capsule: keep the sphere static so only the capsule moves.
                builder.add_shape_sphere(
                    body=-1, xform=wp.transform((0.0, 0.0, 0.03), wp.quat_identity()), radius=0.03, cfg=plane_cfg
                )
            elif support is not None:
                center = top - RADIUS - (0.02 if name == "box" else RADIUS)
                _free(builder, support, (0.0, 0.0, center), mass=1.0, label="support")
            _free(builder, capsule, (0.0, 0.0, top + 2e-5), lying, mass=0.1, label="capsule")
            model = builder.finalize(device=device)
            solver = SolverMonolithic(
                model,
                friction_velocity_scale=0.01,
                contact_stiffness=1e9,
                friction_coefficient=0.5,
                publish_policy="current_iterate",
            )
            state, control = solver.create_state(dt=0.001), model.control()
            for _ in range(300):
                solver.step(state, state, control, dt=0.001)
                test.assertFalse(solver.last_stats.rolled_back, solver.last_stats.failure_reason)
            forces = solver.get_contact_forces(state).body_wrench
            weights = model.body_mass.numpy() * 9.81
            # The top body (last) carries its own weight; a support carries both.
            test.assertAlmostEqual(forces[-1, 2] / weights[-1], 1.0, delta=0.02)
            if support not in (None, "static"):
                test.assertAlmostEqual(forces[0, 2] / weights[0], 1.0, delta=0.02)
            q = state.body_q.numpy()
            test.assertLess(abs(q[-1, 2] - top), 1e-3)


class TestContactCapsule(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_field_matches_long_double,
        test_gradient_is_field_derivative,
        test_roundoff_magnitude,
        test_surface_samples,
        test_resting_pairs,
    ):
        add_function_test(TestContactCapsule, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
