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
# Set before measuring (P9d PR1 review round 1): contact normal within 2 degrees of the exact distance
# gradient, 2-5 mm outside face centroids (the distance to a convex set is C1 outside it). Review round 2
# extends the same bound, unchanged, to probes outside edge midpoints and vertices.
HULL_GRADIENT_ANGLE_RAD = np.radians(2.0)


@wp.kernel
def _field(
    table: wp.array[_SDF64], points: wp.array[wp.vec3d], values: wp.array[wp.float64], gradients: wp.array[wp.vec3d]
):
    i = wp.tid()
    value, gradient, _hessian = smooth_query(table[0], points[i])
    values[i] = value
    gradients[i] = gradient


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

    def exact_distance(p):
        inside = max(float(p @ n) - d for n, d in normals_offsets)
        return inside if inside <= 0 else fc.core_distance(fc.point().placed(np.eye(3), p), hull)

    # Radial rays through hull vertices, face centroids along face normals, and edge midpoints along the
    # mean of their two face normals, at signed distances across the band.
    surface = vertices[rng.integers(len(vertices), size=300)]
    radial = surface - np.c_[np.zeros((300, 2)), np.clip(surface[:, 2], -0.04, 0.04)]
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    triangles = vertices[np.asarray(mesh.indices).reshape(-1, 3)]
    face_normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]
    face_normal *= np.sign(np.einsum("fk,fk->f", face_normal, triangles.mean(axis=1)))[:, None]
    faces = rng.choice(len(triangles), size=200, replace=False)
    edge_faces = rng.choice(len(triangles), size=200, replace=False)
    edge_mid = (triangles[edge_faces, 0] + triangles[edge_faces, 1]) / 2
    origins = np.vstack((surface, triangles[faces].mean(axis=1), edge_mid))
    directions = np.vstack((radial, face_normal[faces], face_normal[edge_faces]))
    offsets = rng.uniform(-HULL_FIELD_BAND_M, HULL_FIELD_BAND_M, size=(len(origins), 1))
    points = origins + directions * offsets
    exact = np.array([exact_distance(p) for p in points])
    keep = np.abs(exact) <= HULL_FIELD_BAND_M
    # Gradient probes 2-5 mm outside face centroids, edge midpoints (along the mean of the two face
    # normals) and vertices (along the mean of the incident face normals); the exact gradient by central
    # differences.
    indices = np.asarray(mesh.indices).reshape(-1, 3)
    edge_normals, vertex_normals = {}, np.zeros_like(vertices)
    for face, corners in enumerate(indices):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = tuple(sorted((int(corners[a]), int(corners[b]))))
            edge_normals[key] = edge_normals.get(key, 0.0) + face_normal[face]
        np.add.at(vertex_normals, corners, face_normal[face])
    edges = list(edge_normals)
    edges = [edges[i] for i in rng.choice(len(edges), size=200, replace=False)]
    edge_points = np.array([(vertices[a] + vertices[b]) / 2 for a, b in edges])
    edge_directions = np.array([edge_normals[e] / np.linalg.norm(edge_normals[e]) for e in edges])
    corner_ids = rng.choice(len(vertices), size=200, replace=False)
    corner_directions = vertex_normals[corner_ids] / np.linalg.norm(vertex_normals[corner_ids], axis=1)[:, None]
    probe_kinds = ("face", "edge", "vertex")
    probe_origins = (triangles[faces].mean(axis=1), edge_points, vertices[corner_ids])
    probe_directions = (face_normal[faces], edge_directions, corner_directions)
    outside = np.vstack(
        [
            o + d * rng.uniform(2e-3, 5e-3, size=(len(o), 1))
            for o, d in zip(probe_origins, probe_directions, strict=True)
        ]
    )
    h = 1e-7
    exact_gradient = np.array(
        [[(exact_distance(p + h * e) - exact_distance(p - h * e)) / (2 * h) for e in np.eye(3)] for p in outside]
    )
    queries = np.vstack((points, outside))
    values = wp.zeros(len(queries), dtype=wp.float64, device=device)
    gradients = wp.zeros(len(queries), dtype=wp.vec3d, device=device)
    wp.launch(
        _field,
        len(queries),
        [fields.table, wp.array(queries, dtype=wp.vec3d, device=device), values, gradients],
        device=device,
    )
    error = np.abs(values.numpy()[: len(points)] - exact)[keep]
    test.assertGreater(int(keep.sum()), 500)
    test.assertLessEqual(float(error.max()), HULL_FIELD_TOLERANCE_M, f"max field error {error.max():.3e} m")
    field_gradient = gradients.numpy()[len(points) :]
    cosine = np.einsum("ik,ik->i", field_gradient, exact_gradient) / (
        np.linalg.norm(field_gradient, axis=1) * np.linalg.norm(exact_gradient, axis=1)
    )
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    for kind, part in zip(probe_kinds, np.split(angle, np.cumsum([len(o) for o in probe_origins])[:-1]), strict=True):
        with test.subTest(probe=kind):
            test.assertLessEqual(
                float(part.max()), HULL_GRADIENT_ANGLE_RAD, f"max {kind} normal angle {np.degrees(part.max()):.3f} deg"
            )


def test_convex_mesh_rejected_as_cloth_target(test, device):
    """Cloth particles may not contact a CONVEX_MESH (PRD 2.9); the error names the shape type."""
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    if not fhs.CONVEX_HULL.is_file():
        test.skipTest("Pinned convex hull not present")
    mesh, _ = _hull_mesh(device)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_shape_convex_hull(-1, mesh=mesh, label="static_hull")
    first, face = builder.particle_count, builder.tri_count
    for point in ((0.0, 0.0, 0.05), (0.02, 0.0, 0.05), (0.0, 0.02, 0.05)):
        builder.add_particle(wp.vec3(point), wp.vec3(0.0), 2e-4 / 3)
    builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
    cloth = [SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 2), thickness=0.001, surface_density=1)]
    with test.assertRaisesRegex(ValueError, "CONVEX_MESH shape .* particle"):
        SolverMonolithic(
            builder.finalize(device=device), friction_velocity_scale=0.01, cloth_inputs=cloth, mass_mode="lumped"
        )


class TestContactConvexMesh(unittest.TestCase):
    pass


for _device in get_cuda_test_devices():
    for _test in (
        test_convex_mesh_equals_mesh,
        test_pinned_hull_field_accuracy,
        test_convex_mesh_rejected_as_cloth_target,
    ):
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
