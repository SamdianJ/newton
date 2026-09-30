# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the experimental solver's fixed P1Q3 collision stream."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData, create_texture_sdf_from_primitive
from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _scene(device, *, shape="plane", margin=0.0):
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01))
    joint = builder.add_joint_revolute(-1, body)
    builder.add_articulation([joint])
    cfg = builder.ShapeConfig(margin=margin)
    if shape == "plane":
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=cfg)
    elif shape == "mixed":
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=cfg)
        builder.add_shape_box(
            body=body, xform=wp.transform((10.0, 10.0, 10.0), wp.quat_identity()), hx=0.02, hy=0.03, hz=0.01, cfg=cfg
        )
    elif shape == "sphere":
        builder.add_shape_sphere(body=body, radius=0.03, cfg=cfg)
    elif shape == "box":
        builder.add_shape_box(body=body, hx=0.02, hy=0.03, hz=0.01, cfg=cfg)
    elif shape == "mesh":
        mesh = newton.Mesh.create_box(0.02, 0.03, 0.01)
        mesh.build_sdf(max_resolution=32)
        builder.add_shape_mesh(body=body, mesh=mesh, cfg=cfg)
    elif shape == "ellipsoid":
        builder.add_shape_ellipsoid(body=body, rx=0.02, ry=0.03, rz=0.01, cfg=cfg)
    elif shape == "finite_plane":
        builder.add_shape_plane(body=body, width=1.0, length=1.0, cfg=cfg)
    else:
        getattr(builder, "add_shape_" + shape)(body=body, radius=0.02, half_height=0.03, cfg=cfg)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.002),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.02)],
        indices=[0, 1, 2, 3],
        density=1000.0,
        k_mu=1000.0,
        k_lambda=1000.0,
        k_damp=0.0,
        particle_radius=0.001,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    model = builder.finalize(device=device)
    return model, model.state()


def test_p1q3_records(test, device):
    """Emit three deterministic samples per face with physical plane projections."""
    model, state = _scene(device)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    test.assertEqual(pipeline.soft_contact_pair_count, 4)
    test.assertEqual(pipeline.soft_contact_max, 12)
    test.assertEqual(pipeline.soft_contact_tids_size, 12)
    test.assertEqual(pipeline.rigid_contact_max, 0)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 12)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)
    np.testing.assert_array_equal(contacts.soft_contact_particle.numpy(), -np.ones(12))
    tids = contacts.soft_contact_tids.numpy()
    bary = contacts.soft_contact_barycentric.numpy()
    for tid, record in enumerate(tids):
        expected = np.full(3, 1.0 / 6.0, dtype=np.float32)
        expected[tid % 3] = 2.0 / 3.0
        np.testing.assert_array_equal(bary[record], expected)
    np.testing.assert_allclose(contacts.soft_contact_normal.numpy(), np.tile([0, 0, 1], (12, 1)), atol=1e-6)
    np.testing.assert_allclose(contacts.soft_contact_body_pos.numpy()[:, 2], 0.0, atol=1e-7)
    with patch.object(contacts, "clear", wraps=contacts.clear) as clear:
        pipeline.collide(state, contacts)
        clear.assert_called_once_with(bump_generation=True)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 2)
    positions = state.particle_q.numpy()
    positions[:, 2] += 1.0
    state.particle_q.assign(positions)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(contacts.soft_contact_tids.numpy(), -np.ones(12))


def test_contact_ownership(test, device):
    """Reject foreign or malformed buffers before clearing any state."""
    model, state = _scene(device)
    pipeline = MonolithicCollisionPipeline(model)
    contacts = MonolithicCollisionPipeline(model).contacts()
    with test.assertRaisesRegex(ValueError, "PROVENANCE"):
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 0)
    contacts = pipeline.contacts()
    contacts.soft_contact_normal = wp.zeros(1, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "CAPACITY"):
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 0)
    contacts = pipeline.contacts()
    contacts.soft_contact_normal.shape = (1,)
    with test.assertRaisesRegex(ValueError, "CAPACITY"):
        pipeline.validate_contacts(contacts)
    contacts = pipeline.contacts()
    contacts.requires_grad = True
    with test.assertRaisesRegex(ValueError, "PROVENANCE"):
        pipeline.validate_contacts(contacts)
    with test.assertRaisesRegex(ValueError, "dt"):
        pipeline.collide(state, pipeline.contacts(), dt=0.01)


def test_asset_gates(test, device):
    """Reject invalid radius and unsupported analytic assets explicitly."""
    for shape, code in [("ellipsoid", "UNSUPPORTED_SHAPE_SDF_ACCURACY"), ("finite_plane", "UNSUPPORTED_FINITE_PLANE")]:
        model, _ = _scene(device, shape=shape)
        with test.assertRaisesRegex(ValueError, code):
            MonolithicCollisionPipeline(model)
    for value, reason in [(float("nan"), "nonfinite"), (-0.1, "negative"), (0.002, "nonuniform")]:
        model, _ = _scene(device)
        radius = model.particle_radius.numpy()
        radius[0] = value
        model.particle_radius.assign(radius)
        with test.assertRaisesRegex(ValueError, "UNSUPPORTED_PARTICLE_RADIUS.*" + reason):
            MonolithicCollisionPipeline(model)
    for shape in ["box", "capsule", "cylinder", "cone"]:
        model, state = _scene(device, shape=shape)
        pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
        pipeline.collide(state, pipeline.contacts())


def test_boundary_gates(test, device):
    """Reject missing, extra, duplicate and geometrically degenerate boundaries."""
    for mode in ["missing", "duplicate", "repeated", "extra", "zero_area"]:
        model, _ = _scene(device)
        faces = model.tri_indices.numpy()
        if mode == "missing":
            faces = faces[:-1]
        elif mode == "duplicate":
            faces[0] = faces[1]
        elif mode == "repeated":
            faces[0, 0] = faces[0, 1]
        elif mode == "extra":
            faces = np.concatenate((faces, faces[:1]))
        else:
            pos = model.particle_q.numpy()
            pos[1] = pos[0]
            model.particle_q.assign(pos)
        model.tri_indices = wp.array(faces, dtype=int, device=device)
        with test.assertRaisesRegex(ValueError, "BOUNDARY"):
            MonolithicCollisionPipeline(model)


def test_detection_margin(test, device):
    """Keep the detection shell independent of the physical shape margin."""
    model, state = _scene(device)
    counts = []
    for gap in [0.0, 0.1]:
        pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=gap)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        counts.append(int(contacts.soft_contact_count.numpy()[0]))
    test.assertEqual(counts, [0, 12])
    model, state = _scene(device, margin=0.03)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.0)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 12)
    np.testing.assert_allclose(contacts.soft_contact_body_pos.numpy()[:, 2], 0.0, atol=1e-7)


def test_sphere_frames(test, device):
    """Preserve body-local projections and world normals under both transforms."""
    model, state = _scene(device, shape="sphere")
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.7)
    model.shape_transform.assign([wp.transform((0.01, -0.02, 0.005), rotation)])
    body_transform = wp.transform((0.3, -0.1, 0.4), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.6))
    state.body_q.assign([body_transform])
    state.particle_q.assign([wp.transform_point(body_transform, wp.vec3(p)) for p in state.particle_q.numpy()])
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    positions = state.particle_q.numpy()
    center = np.array(wp.transform_point(body_transform, wp.vec3(0.01, -0.02, 0.005)))
    for indices, bary, point, normal in zip(
        contacts.soft_contact_indices.numpy(),
        contacts.soft_contact_barycentric.numpy(),
        contacts.soft_contact_body_pos.numpy(),
        contacts.soft_contact_normal.numpy(),
        strict=True,
    ):
        sample = bary @ positions[indices]
        expected = (sample - center) / np.linalg.norm(sample - center)
        np.testing.assert_allclose(normal, expected, atol=2e-6)
        world_point = np.array(wp.transform_point(body_transform, wp.vec3(point)))
        np.testing.assert_allclose(world_point, center + 0.03 * expected, atol=2e-6)


def test_query_failure(test, device):
    """Invalidate failed gradient and overflow passes with one generation bump."""
    model, state = _scene(device, shape="sphere")
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    state.particle_q.fill_(wp.vec3(float("nan")))
    with test.assertRaisesRegex(ValueError, "INVALID_SDF_GRADIENT"):
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)
    model, state = _scene(device)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    launch = wp.launch

    def reduced_capacity(kernel, *args, **kwargs):
        if kernel.key == "create_monolithic_p1q3_face_contacts":
            kwargs["inputs"][14] = 1
        return launch(kernel, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=reduced_capacity):
        with test.assertRaisesRegex(ValueError, "CONTACT_CAPACITY_OVERFLOW"):
            pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)


def test_model_contract(test, device):
    """Reject stale model arrays, changed requests and mixed world labels."""
    model, state = _scene(device)
    pipeline = MonolithicCollisionPipeline(model)
    contacts = pipeline.contacts()
    model.tri_indices = wp.clone(model.tri_indices)
    with test.assertRaisesRegex(ValueError, "INVALID_MODEL"):
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 0)
    model, _ = _scene(device)
    pipeline = MonolithicCollisionPipeline(model)
    model.request_contact_attributes("force")
    with test.assertRaisesRegex(ValueError, "PROVENANCE"):
        pipeline.contacts()
    model, _ = _scene(device)
    worlds = model.particle_world.numpy()
    worlds[0] = 0 if worlds[0] == -1 else -1
    model.particle_world.assign(worlds)
    with test.assertRaisesRegex(ValueError, "INVALID_WORLD"):
        MonolithicCollisionPipeline(model)


def test_volume_sdf(test, device):
    """Query real CUDA texture SDFs and enforce baked versus runtime scales."""
    model, state = _scene(device, shape="sphere")
    model.shape_type.assign([int(newton.GeoType.MESH)])
    if device.is_cpu:
        with test.assertRaisesRegex(ValueError, "INVALID_SDF_DESCRIPTOR"):
            MonolithicCollisionPipeline(model)
        return
    descriptor, coarse, fine = create_texture_sdf_from_primitive(
        newton.GeoType.SPHERE,
        (0.03, 0.0, 0.0),
        max_resolution=64,
        scale_baked=False,
        device=device,
    )
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    model._texture_sdf_coarse_textures = [coarse]
    model._texture_sdf_subgrid_textures = [fine]
    model._shape_sdf_index.assign([0])
    model.shape_scale.assign([(1.0, 1.0, 1.0)])
    positions = state.particle_q.numpy()
    positions[:, 2] += 0.05
    state.particle_q.assign(positions)
    projections = []
    for scale in [1.0, 2.0]:
        model.shape_scale.assign([(scale, scale, scale)])
        pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 12)
        normals = contacts.soft_contact_normal.numpy()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-6)
        projections.append(np.linalg.norm(contacts.soft_contact_body_pos.numpy(), axis=1))
        np.testing.assert_allclose(projections[-1], 0.03 * scale, atol=4e-4)
    for scale, reason in [
        ((1, 2, 1), "nonuniform"),
        ((0, 1, 1), "zero"),
        ((-1, 1, 1), "mirrored"),
        ((float("nan"), 1, 1), "nonfinite"),
    ]:
        model.shape_scale.assign([scale])
        with test.assertRaisesRegex(ValueError, "UNSUPPORTED_SDF_RUNTIME_SCALE.*" + reason):
            MonolithicCollisionPipeline(model)
    descriptor.scale_baked = True
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    model.shape_scale.assign([(2.0, 3.0, 4.0)])
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    np.testing.assert_allclose(np.linalg.norm(contacts.soft_contact_body_pos.numpy(), axis=1), 0.03, atol=4e-4)
    # A real constant grid has no valid surface normal, even though its distance is finite.
    zero_textures = [
        wp.Texture3D(
            np.zeros((tex.depth, tex.height, tex.width, tex.num_channels), dtype=np.float32),
            normalized_coords=False,
            device=device,
        )
        for tex in (coarse, fine)
    ]
    descriptor.coarse_texture, descriptor.subgrid_texture = zero_textures
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    model._texture_sdf_coarse_textures, model._texture_sdf_subgrid_textures = [zero_textures[0]], [zero_textures[1]]
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    with test.assertRaisesRegex(ValueError, "INVALID_SDF_GRADIENT"):
        pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)


def test_interior_radius_and_nonmanifold(test, device):
    """Validate interior particle radii and tet faces with three incident cells."""
    model, _ = _scene(device)
    points = model.particle_q.numpy()
    model.particle_q = wp.array(np.vstack((points, points.mean(axis=0))), dtype=wp.vec3, device=device)
    model.tet_indices = wp.array([[0, 1, 2, 4], [0, 3, 1, 4], [0, 2, 3, 4], [1, 3, 2, 4]], dtype=int, device=device)
    model.particle_radius = wp.array([0.001, 0.001, 0.001, 0.001, 0.002], dtype=float, device=device)
    with test.assertRaisesRegex(ValueError, "UNSUPPORTED_PARTICLE_RADIUS.*nonuniform"):
        MonolithicCollisionPipeline(model)
    model, _ = _scene(device)
    points = model.particle_q.numpy()
    model.particle_q = wp.array(np.vstack((points, [0, 0, -0.02], [0.01, 0.01, -0.03])), dtype=wp.vec3, device=device)
    model.tet_indices = wp.array([[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]], dtype=int, device=device)
    with test.assertRaisesRegex(ValueError, "INVALID_BOUNDARY.*nonmanifold_tet_face"):
        MonolithicCollisionPipeline(model)


def test_sdf_query_resources(test, device):
    """Reject missing fine-grid resources while retaining valid coarse-only queries."""
    model, state = _scene(device, shape="sphere")
    descriptor, coarse, fine = create_texture_sdf_from_primitive(
        newton.GeoType.SPHERE,
        (0.03, 0.0, 0.0),
        max_resolution=32,
        device=device,
    )
    test.assertGreater(descriptor.num_subgrids, 0)
    model.shape_type.assign([int(newton.GeoType.MESH)])
    model._shape_sdf_index.assign([0])
    model.shape_scale.assign([(1.0, 1.0, 1.0)])
    model._texture_sdf_coarse_textures = [coarse]
    null_texture = TextureSDFData().subgrid_texture
    indirection = descriptor.subgrid_start_slots
    for missing in ("fine_handle", "fine_owner", "fine_owner_mismatch", "empty_indirection"):
        with test.subTest(missing=missing):
            descriptor.subgrid_texture = null_texture if missing == "fine_handle" else fine
            descriptor.subgrid_start_slots = (
                wp.zeros((0, 0, 0), dtype=wp.uint32, device=device) if missing == "empty_indirection" else indirection
            )
            model._texture_sdf_subgrid_textures = [
                None if missing == "fine_owner" else coarse if missing == "fine_owner_mismatch" else fine
            ]
            model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
            with test.assertRaisesRegex(ValueError, "INVALID_SDF_DESCRIPTOR"):
                MonolithicCollisionPipeline(model)
    # A coarse-only descriptor never dereferences the absent fine texture.
    descriptor.num_subgrids = 0
    descriptor.subgrid_texture = null_texture
    descriptor.subgrid_start_slots = wp.full(indirection.shape, wp.uint32(0xFFFFFFFF), dtype=wp.uint32, device=device)
    model._texture_sdf_subgrid_textures = [None]
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.1)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 12)
    test.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)
    np.testing.assert_allclose(np.linalg.norm(contacts.soft_contact_normal.numpy(), axis=1), 1.0, atol=2e-6)


class TestMonolithicCollision(unittest.TestCase):
    """Exercise fixed quadrature collision on the available devices."""


for _test in [
    test_p1q3_records,
    test_contact_ownership,
    test_asset_gates,
    test_boundary_gates,
    test_detection_margin,
    test_sphere_frames,
    test_query_failure,
    test_model_contract,
    test_volume_sdf,
    test_interior_radius_and_nonmanifold,
]:
    add_function_test(TestMonolithicCollision, _test.__name__, _test, devices=get_test_devices())

add_function_test(
    TestMonolithicCollision,
    "test_sdf_query_resources",
    test_sdf_query_resources,
    devices=[device for device in get_test_devices() if device.is_cuda],
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
