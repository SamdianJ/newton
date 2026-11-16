# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shape types without a contact implementation fail explicitly instead of acting as another shape."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.types import GeoType
from newton._src.solvers.monolithic.contact_geometry import _shape64
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query32(
    geo: int,
    scale: wp.vec3f,
    points: wp.array[wp.vec3f],
    table: wp.array[TextureSDFData],
    values: wp.array[wp.float32],
    gradients: wp.array[wp.vec3f],
):
    i = wp.tid()
    value, gradient = _shape64(geo, scale, points[i], -1, table)
    values[i] = value
    gradients[i] = gradient


@wp.kernel
def _query64(
    geo: int,
    scale: wp.vec3d,
    points: wp.array[wp.vec3d],
    table: wp.array[TextureSDFData],
    values: wp.array[wp.float64],
    gradients: wp.array[wp.vec3d],
):
    i = wp.tid()
    value, gradient = _shape64(geo, scale, points[i], -1, table)
    values[i] = value
    gradients[i] = gradient


def _query(device, precision, geo, scale, points):
    scalar, vector, kernel = (
        (wp.float32, wp.vec3f, _query32) if precision == "fp32" else (wp.float64, wp.vec3d, _query64)
    )
    points = wp.array(np.asarray(points), dtype=vector, device=device)
    values = wp.zeros(len(points), dtype=scalar, device=device)
    gradients = wp.zeros(len(points), dtype=vector, device=device)
    table = wp.zeros(0, dtype=TextureSDFData, device=device)
    wp.launch(kernel, len(points), [int(geo), vector(*scale), points, table, values, gradients], device=device)
    return values.numpy(), gradients.numpy()


def test_unhandled_type_is_nonfinite(test, device):
    """F3: a type without a field branch returns NaN, not the plane z = 0 (which a CYLINDER used to get)."""
    for precision in ("fp32", "fp64"):
        with test.subTest(precision=precision):
            points = [[0.0, 0.0, 0.5], [0.1, -0.2, -0.3]]
            values, gradients = _query(device, precision, GeoType.CYLINDER, (0.1, 0.2, 0.0), points)
            test.assertTrue(np.all(np.isnan(values)), values)
            test.assertTrue(np.all(np.isnan(gradients)), gradients)


def test_plane_is_unchanged(test, device):
    """The explicit plane branch gives exactly z and the unit normal, including at z = 0 and large coordinates."""
    rng = np.random.default_rng(7)
    points = np.vstack(([[0.0, 0.0, 0.0], [1e6, -1e6, 1e-30], [3.0, 4.0, -1e6]], rng.normal(size=(64, 3))))
    for precision, dtype in (("fp32", np.float32), ("fp64", np.float64)):
        with test.subTest(precision=precision):
            values, gradients = _query(device, precision, GeoType.PLANE, (0.0, 0.0, 0.0), points.astype(dtype))
            np.testing.assert_array_equal(values, points.astype(dtype)[:, 2])
            np.testing.assert_array_equal(gradients, np.tile(np.array([0.0, 0.0, 1.0], dtype=dtype), (len(points), 1)))


def _link_scene(device, link_shape, *, particle_flag=False, plane=True):
    """One revolute link carrying ``link_shape`` (a callable adding it), optionally above a ground plane."""
    import newton  # noqa: PLC0415

    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=particle_flag)
    link = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()), mass=0.1, inertia=wp.diag(wp.vec3(1e-4)), label="arm"
    )
    joint = builder.add_joint_revolute(-1, link, axis=(0.0, 1.0, 0.0), limit_ke=0.0, limit_kd=0.0, label="hinge")
    builder.add_articulation([joint])
    link_shape(builder, link, cfg)
    if plane:
        builder.add_shape_plane(
            body=-1,
            width=0.0,
            length=0.0,
            cfg=builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False),
            label="ground",
        )
    return builder.finalize(device=device)


def _solver(model, **options):
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    return SolverMonolithic(model, friction_velocity_scale=0.01, **options)


def test_link_shape_without_samples_against_static_raises(test, device):
    """F1: a link CYLINDER paired with the ground used to be dropped from contact silently."""

    def cylinder(builder, link, cfg):
        builder.add_shape_cylinder(link, radius=0.02, half_height=0.05, cfg=cfg, label="arm_cylinder")

    model = _link_scene(device, cylinder)
    with test.assertRaises(ValueError) as caught:
        _solver(model)
    message = str(caught.exception)
    for part in ("arm_cylinder", "arm", "CYLINDER", "ground"):
        test.assertIn(part, message)


def test_link_plane_against_static_is_skipped(test, device):
    """A link plane is a target-only shape: pairing it with the ground stays a skipped pair, not an error."""

    def plane(builder, link, cfg):
        builder.add_shape_plane(body=link, width=0.0, length=0.0, cfg=cfg, label="arm_plane")

    _solver(_link_scene(device, plane))


def test_particle_flag_without_particles_constructs(test, device):
    """F2: without particles nothing is a particle target, so a COLLIDE_PARTICLES CYLINDER (the URDF
    import default) must not be rejected. No ground plane, so F1 (A-static pairs) does not apply."""

    def cylinder(builder, link, cfg):
        builder.add_shape_cylinder(link, radius=0.02, half_height=0.05, cfg=cfg, label="arm_cylinder")

    _solver(_link_scene(device, cylinder, particle_flag=True, plane=False))


def test_unsupported_particle_target_still_raises(test, device):
    """With a tet in the scene, COLLIDE_PARTICLES shapes are real particle targets: capsule stays rejected."""
    import newton  # noqa: PLC0415

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=1,
        dim_y=1,
        dim_z=1,
        cell_x=0.02,
        cell_y=0.02,
        cell_z=0.02,
        density=1000.0,
        k_mu=1e4,
        k_lambda=1e4,
        k_damp=0.0,
    )
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()), mass=0.1, inertia=wp.diag(wp.vec3(1e-4))
    )
    builder.add_articulation([builder.add_joint_free(body)])
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=True, has_shape_collision=False)
    builder.add_shape_capsule(body, radius=0.01, half_height=0.02, cfg=cfg, label="probe")
    # Actor sampling (tet-rigid pairs) or the particle-target whitelist rejects it, whichever runs first.
    with test.assertRaisesRegex(ValueError, "Unsupported (contact shape|actor contact target)"):
        _solver(builder.finalize(device=device), mass_mode="lumped")


class TestContactUnsupportedShapes(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_unhandled_type_is_nonfinite,
        test_plane_is_unchanged,
        test_link_shape_without_samples_against_static_raises,
        test_link_plane_against_static_is_skipped,
        test_particle_flag_without_particles_constructs,
        test_unsupported_particle_target_still_raises,
    ):
        add_function_test(TestContactUnsupportedShapes, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
