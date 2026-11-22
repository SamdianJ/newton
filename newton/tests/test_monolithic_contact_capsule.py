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
    # The gradient magnitude bounds the FP32 gradient error too, including within rounding of the cap seams.
    ag = np.linalg.norm(gradients.numpy()[:-2], axis=1)
    _, gradient_ref = _reference(points[:-2])
    _, observed_g = _query(device, "fp32", GeoType.CAPSULE, (RADIUS, HALF, 0.0), points[:-2].astype(np.float32))
    gradient_error = np.linalg.norm(observed_g.astype(np.float64) - gradient_ref.astype(np.float64), axis=1)
    test.assertTrue(np.all(gradient_error <= 32 * np.finfo(np.float32).eps * ag + 1e-12), np.max(gradient_error / ag))


def test_roundoff_magnitude_at_seam(test, device):
    """Per component, the gradient magnitude bounds an FP32 point one ulp across a cap seam.

    The exact points lie on or just inside the axial span, where the exact gradient has no axial part.
    Their FP32 images one ulp beyond float32(h) are within one rounding of the point (eps32 * |z|) and
    switch the axial part on; the magnitude must cover that component (P9d PR1 review round 2).
    """
    rho = 1e-3
    exact, rounded = [], []
    for sign in (1.0, -1.0):
        beyond = sign * np.nextafter(np.float32(HALF), np.float32(np.inf))
        for z in (sign * HALF, sign * (HALF - 1e-9)):
            for angle in (0.3, 2.0):
                exact.append((rho * np.cos(angle), rho * np.sin(angle), z))
                rounded.append((np.float32(exact[-1][0]), np.float32(exact[-1][1]), beyond))
    exact, rounded = np.asarray(exact), np.asarray(rounded, dtype=np.float32)
    test.assertTrue(np.all(np.abs(rounded[:, 2].astype(np.float64) - exact[:, 2]) <= np.finfo(np.float32).eps * HALF))
    pts = wp.array(exact, dtype=wp.vec3d, device=device)
    values = wp.zeros(len(exact), dtype=wp.float64, device=device)
    gradients = wp.zeros(len(exact), dtype=wp.vec3d, device=device)
    pm = wp.array(np.abs(exact), dtype=wp.vec3d, device=device)
    wp.launch(_magnitude64, len(exact), [pts, wp.vec3d(RADIUS, HALF, 0.0), pm, values, gradients], device=device)
    _, gradient_ref = _reference(exact)
    _, observed = _query(device, "fp32", GeoType.CAPSULE, (RADIUS, HALF, 0.0), rounded)
    error = np.abs(observed.astype(np.float64) - gradient_ref.astype(np.float64))
    # The FP32 gradient's axial part is about ulp(h) / rho here: far above eps32, so a zero bound fails.
    test.assertGreater(float(error[:, 2].min()), 1e-6)
    bound = 32 * np.finfo(np.float32).eps * gradients.numpy() + 1e-12
    test.assertTrue(np.all(error <= bound), np.max(error / bound))


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
    for (name, (support, top)), precision in __import__("itertools").product(supports.items(), ("fp64", "fp32")):
        with test.subTest(support=name, precision=precision):
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
                precision=precision,
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


def test_dense_capsule_contact(test, device):
    """A capsule rigid body (axis along y, the fixture box's x extent) matches the independent dense
    oracle's energy, residual and tangent in link, rigid and static pairs (FP64)."""
    from newton.tests.test_monolithic_contact_actor_pairs import native_batch, perturb, physical, query  # noqa: PLC0415
    from newton.tests.test_monolithic_contact_components import batch_matrix  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_actor_contacts import (  # noqa: PLC0415
        contact_scene,
        contact_solver,
        contact_state,
    )
    from scripts.monolithic_reference.actor_contact_oracle import fixture_terms  # noqa: PLC0415

    for kind in ("A-R", "R-R", "R-static"):
        with test.subTest(kind=kind):
            run = contact_solver(contact_scene(device, kind, capsule=True))
            batch = native_batch(run)
            state = contact_state(run, kind)
            start = physical(state)
            query(batch, run, state, start=True)
            direction = np.linspace(-0.2, 0.3, run._layout.scalar_dof_count)
            perturb(run, state, start, direction, 1e-6)
            query(batch, run, state)
            energy, residual, b = fixture_terms(run.model, run._actors, physical(state), start)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            test.assertGreater(abs(energy[0]), 0.0)  # the capsule pair is in contact
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], energy, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(batch.pool.residual.result.numpy(), residual, rtol=1e-8, atol=1e-10)
            matrix = batch_matrix(batch)
            scale = max(1, np.linalg.norm(b))
            test.assertLess(np.linalg.norm(matrix - b) / scale, 1e-5)


def test_curved_target_tangent(test, device):
    """A rigid box pressed into a static capsule, beside the segment and on a cap, frictionless: the
    translation block of the contact tangent is the central difference of the residual, column by
    column. Its transverse columns come only from the field's normal curvature (FP64)."""
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415
    from newton.tests.test_monolithic_contact_actor_pairs import native_batch, query  # noqa: PLC0415
    from newton.tests.test_monolithic_contact_components import batch_matrix  # noqa: PLC0415

    half = 0.004
    for where, center in (
        ("side", (RADIUS + half - 2e-4, 0.001, 0.01)),
        ("cap", (0.002, 0.001, HALF + RADIUS + half - 2e-4)),
    ):
        with test.subTest(where=where):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
            builder.add_shape_capsule(body=-1, radius=RADIUS, half_height=HALF, cfg=cfg)
            _free(builder, lambda b, body, c: b.add_shape_box(body, hx=half, hy=half, hz=half, cfg=c), center)
            model = builder.finalize(device=device)
            run = SolverMonolithic(
                model,
                friction_velocity_scale=0.01,
                friction_coefficient=0.0,
                contact_stiffness=1e7,
                normal_smoothing=1e-4,
                precision="fp64",
            )
            state = run.create_state(dt=1e-3)
            batch = native_batch(run)
            query(batch, run, state, start=True)
            query(batch, run, state)
            rows = run._rigid.actors[0].dof_start + np.arange(3)
            block = batch_matrix(batch)[np.ix_(rows, rows)]
            history = state._monolithic_coupled
            com = history.rigid_com.numpy().astype(np.float64)
            h = 1e-8
            columns = []
            for axis in range(3):
                values = []
                for sign in (1, -1):
                    shifted = com.copy()
                    shifted[0, axis] += sign * h
                    history.rigid_com.assign(shifted)
                    query(batch, run, state)
                    values.append(batch.pool.residual.result.numpy()[rows].copy())
                history.rigid_com.assign(com)
                columns.append((values[0] - values[1]) / (2 * h))
            derivative = np.stack(columns, axis=1)
            np.testing.assert_allclose(block, derivative, rtol=0, atol=1e-5 * np.abs(derivative).max())
            # y is a curved transverse direction both beside the segment and on the cap (contact along +x / +z).
            test.assertGreater(abs(derivative[1, 1]), 1e-4 * np.abs(derivative).max())
            np.testing.assert_allclose(block[:, 1], derivative[:, 1], rtol=0, atol=1e-4 * abs(derivative[1, 1]))


def test_capsule_rejected_as_material_target(test, device):
    """Cloth and cable particles may not contact a capsule (PRD 2.9); the error names the shape type."""
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    for material in ("cloth", "cable"):
        with test.subTest(material=material):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_shape_capsule(body=-1, radius=RADIUS, half_height=HALF, label="static_capsule")
            cloth, cables = [], []
            if material == "cloth":
                first, face = builder.particle_count, builder.tri_count
                # Lumped material mass: area (2e-4 m^2) x surface density / 3 per node.
                for point in ((0.0, 0.0, 0.03), (0.02, 0.0, 0.03), (0.0, 0.02, 0.03)):
                    builder.add_particle(wp.vec3(point), wp.vec3(0.0), 2e-4 / 3)
                builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
                cloth.append(
                    SolverMonolithic.ClothInput(
                        triangle_ids=(face,), bending=(1, 2), thickness=0.001, surface_density=1
                    )
                )
            else:
                # Lumped material mass: linear density x half of each adjacent 0.02 m segment.
                masses = (0.01, 0.02, 0.01)
                ids = tuple(builder.add_particle((0.02 * i, 0.0, 0.03), (0, 0, 0), m) for i, m in enumerate(masses))
                cables.append(
                    SolverMonolithic.CableInput(
                        particle_ids=ids,
                        EA=100,
                        bending=((1, 0), (0, 1)),
                        GJ=1,
                        radius=0.002,
                        linear_density=1,
                        rest_theta=(0, 0),
                        rest_director=((0, 1, 0), (0, 1, 0)),
                        fixed_theta=(False, False),
                    )
                )
            model = builder.finalize(device=device)
            with test.assertRaisesRegex(ValueError, "CAPSULE shape .* particle"):
                SolverMonolithic(
                    model, friction_velocity_scale=0.01, cloth_inputs=cloth, cable_inputs=cables, mass_mode="lumped"
                )


class TestContactCapsule(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_field_matches_long_double,
        test_gradient_is_field_derivative,
        test_roundoff_magnitude,
        test_roundoff_magnitude_at_seam,
        test_surface_samples,
        test_resting_pairs,
        test_dense_capsule_contact,
        test_curved_target_tangent,
        test_capsule_rejected_as_material_target,
    ):
        add_function_test(TestContactCapsule, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
