# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CONVEX_MESH contact uses the MESH path; the pinned E1b hull's C2 field matches its exact distance."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.smooth_sdf import _sdf_type, smooth_query
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices
from scripts.monolithic_reference import floating_hand_criteria as fc
from scripts.monolithic_reference import floating_hand_scenes as fhs

_SDF64 = _sdf_type(wp.float64)
SQRT_HALF = float(np.sqrt(0.5))
# Set before measuring (P9d PR1): 10% of the 2 mm penetration gate, within 5 mm of the surface.
HULL_FIELD_TOLERANCE_M, HULL_FIELD_BAND_M = 2e-4, 5e-3


@wp.kernel
def _field(table: wp.array[_SDF64], points: wp.array[wp.vec3d], values: wp.array[wp.float64]):
    i = wp.tid()
    value, _gradient, _hessian = smooth_query(table[0], points[i])
    values[i] = value


def _hull_mesh(device):
    points, indices = fhs.convex_hull_asset({"radius": 0.02, "half_height": 0.04})
    mesh = newton.Mesh(points, indices)
    if wp.get_device(device).is_cuda:
        mesh.build_sdf(device=device, **fhs.CONVEX_HULL_SDF)
    return mesh, points


def _scene(device, kind, mesh, precision="fp32"):
    """A lying hull object on the ground with a small ball dropped on it; ``kind`` is mesh or convex."""
    from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf  # noqa: PLC0415

    builder = newton.ModelBuilder()
    static = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    builder.add_shape_plane(body=-1, width=0.0, length=0.0, cfg=static)
    pose = wp.transform((0.0, 0.0, 0.0205), wp.quat(0.0, SQRT_HALF, 0.0, SQRT_HALF))
    body = builder.add_link(xform=pose, label="object")
    builder.add_articulation([builder.add_joint_free(body, parent_xform=pose)])
    cfg = builder.ShapeConfig(density=500.0, margin=0.0, has_particle_collision=False)
    add = builder.add_shape_mesh if kind == "mesh" else builder.add_shape_convex_hull
    add(body, mesh=mesh, cfg=cfg, label="object_shape")
    ball_pose = wp.transform((0.0, 0.0, 0.0565), wp.quat_identity())
    ball = builder.add_link(xform=ball_pose, label="ball")
    builder.add_articulation([builder.add_joint_free(ball, parent_xform=ball_pose)])
    builder.add_shape_sphere(ball, radius=0.015, cfg=cfg)
    model = builder.finalize(device=device)
    smooth = build_smooth_sdf(model) if wp.get_device(device).is_cuda else None
    return model, SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        contact_stiffness=1e9,
        friction_coefficient=0.5,
        publish_policy="current_iterate",
        smooth_sdf=smooth,
        precision=precision,
        pcg_mode="production",
        use_cuda_graph=wp.get_device(device).is_cuda,
    )


def test_convex_mesh_equals_mesh(test, device):
    """Same geometry and SDF as MESH and as CONVEX_MESH: identical samples and a bitwise identical run."""
    if not fhs.CONVEX_HULL.is_file():
        test.skipTest("Pinned convex hull not present")
    mesh, _ = _hull_mesh(device)
    runs = {}
    for kind in ("mesh", "convex"):
        model, solver = _scene(device, kind, mesh)
        samples = solver._contact.actor_samples
        state, control = solver.create_state(dt=0.001), model.control()
        trace = []
        for _ in range(60):
            solver.step(state, state, control, dt=0.001)
            test.assertFalse(solver.last_stats.rolled_back, solver.last_stats.failure_reason)
            trace.append((state.body_q.numpy().copy(), solver.get_contact_forces(state).body_wrench.copy()))
        runs[kind] = (samples.points.copy(), samples.weights.copy(), samples.normals.copy(), trace)
    (pa, wa, na, ta), (pb, wb, nb, tb) = runs["mesh"], runs["convex"]
    for x, y in ((pa, pb), (wa, wb), (na, nb)):
        np.testing.assert_array_equal(x, y)
    for (qa, fa), (qb, fb) in zip(ta, tb, strict=True):
        np.testing.assert_array_equal(qa, qb)
        np.testing.assert_array_equal(fa, fb)


def test_convex_mesh_requires_cuda(test, device):
    """On CPU there is no texture SDF, so a CONVEX_MESH target fails at construction with the reason."""
    if not fhs.CONVEX_HULL.is_file():
        test.skipTest("Pinned convex hull not present")
    mesh, _ = _hull_mesh(device)
    with test.assertRaisesRegex(ValueError, "requires CUDA texture SDF"):
        _scene(device, "convex", mesh, precision="fp64")


def test_pinned_hull_field_accuracy(test, device):
    """The C2 field of the pinned hull stays within 0.2 mm of the exact FP64 hull distance near the surface."""
    from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415

    if not fhs.CONVEX_HULL.is_file():
        test.skipTest("Pinned convex hull not present")
    mesh, vertices = _hull_mesh(device)
    builder = newton.ModelBuilder()
    body = builder.add_link(label="object", mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_shape_convex_hull(body, mesh=mesh, cfg=builder.ShapeConfig(density=0.0, margin=0.0))
    model = builder.finalize(device=device)
    fields = build_smooth_sdf(model)
    hull = fc.hull(vertices)
    normals_offsets = []
    for normal in hull.normals:
        for sign in (1.0, -1.0):
            n = sign * normal
            normals_offsets.append((n, float((vertices @ n).max())))
    rng = np.random.default_rng(5)
    # Points along random directions from the axis, at signed distances in the band, over the whole surface.
    surface = vertices[rng.integers(len(vertices), size=400)]
    direction = surface - np.c_[np.zeros((400, 2)), np.clip(surface[:, 2], -0.04, 0.04)]
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    points = surface + direction * rng.uniform(-HULL_FIELD_BAND_M, HULL_FIELD_BAND_M, size=(400, 1))
    exact = []
    for p in points:
        inside = max(float(p @ n) - d for n, d in normals_offsets)
        if inside <= 0:
            exact.append(inside)
        else:
            exact.append(fc.core_distance(fc.point().placed(np.eye(3), p), hull))
    exact = np.asarray(exact)
    keep = np.abs(exact) <= HULL_FIELD_BAND_M
    values = wp.zeros(len(points), dtype=wp.float64, device=device)
    wp.launch(
        _field, len(points), [fields.table, wp.array(points, dtype=wp.vec3d, device=device), values], device=device
    )
    error = np.abs(values.numpy() - exact)[keep]
    test.assertGreater(int(keep.sum()), 300)
    test.assertLessEqual(float(error.max()), HULL_FIELD_TOLERANCE_M, f"max field error {error.max():.3e} m")


class TestContactConvexMesh(unittest.TestCase):
    pass


for _device in get_cuda_test_devices():
    for _test in (test_convex_mesh_equals_mesh, test_pinned_hull_field_accuracy):
        add_function_test(TestContactConvexMesh, _test.__name__, _test, devices=[_device])
for _device in get_test_devices():
    if not wp.get_device(_device).is_cuda:
        add_function_test(
            TestContactConvexMesh,
            test_convex_mesh_requires_cuda.__name__,
            test_convex_mesh_requires_cuda,
            devices=[_device],
        )


if __name__ == "__main__":
    unittest.main()
