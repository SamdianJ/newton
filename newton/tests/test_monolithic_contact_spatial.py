# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the spatial adapter at geometry, identity and real contact computation boundaries."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry import GeoType, ShapeFlags
from newton._src.geometry.sdf_texture import TextureSDFData, create_sparse_sdf_textures
from newton._src.solvers.monolithic.articulation import _TreeKinematics
from newton._src.solvers.monolithic.contact import ContactObjective
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton._src.solvers.monolithic.contact_sampling import _surface_samples
from newton._src.solvers.monolithic.contact_spatial import _ContactSpatial
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, SmoothSDFSet
from newton._src.solvers.monolithic.smooth_sdf_builder import build_coefficients
from newton.solvers.experimental.monolithic import build_smooth_sdf
from newton.tests.test_monolithic_contact_actor_pairs import perturb, physical, query
from newton.tests.test_monolithic_contact_batch import batch_matrix, native_inputs
from newton.tests.test_monolithic_contact_components import fixture
from newton.tests.test_monolithic_contact_mesh import synthetic
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.unittest_utils import add_function_test, get_test_devices


def spatial_fixture(device, kind, *, law="C-II", capacity=None, margin=0.0, precision="fp64"):
    model = contact_scene(device, kind, before=True)
    if kind == "chain":
        flags = model.shape_flags.numpy()
        flags[model.shape_body.numpy() == model.body_label.index("pusher")] |= int(ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)
    run = contact_solver(model, law=law, permuted=True, precision=precision)
    state = contact_state(run, kind)
    c, k = run._contact, run._kinematics
    inputs, options = native_inputs(run, state)
    batch = _ContactBatch(c, batch_capacity=7, capacity=None, use_cuda_graph=device.is_cuda)
    batch.begin_step(*inputs, **options)
    spatial = _ContactSpatial(
        batch,
        k,
        k.gather_q(state),
        inputs[0],
        inputs[1],
        **options,
        capacity=c.count if capacity is None else capacity,
        law_margin=margin,
    )
    return run, state, batch, spatial


def test_spatial_pairs(test, device):
    """Preserve allowed directed identities, half weights and selected E/r/B/both forces for all supported pairs."""
    for kind in ("A-R", "R-R", "R-T", "R-static", "chain"):
        for law in ("C-EE", "C-IE", "C-II"):
            run, state, batch, spatial = spatial_fixture(device, kind, law=law)
            start = physical(state)
            if kind == "chain":
                query(run, state, start=True)
            perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
            inputs, options = native_inputs(run, state)
            spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
            spatial.check()
            count = int(spatial.count.numpy()[0])
            ids = spatial.identities.numpy()[:count]
            test.assertEqual(len(ids), len(np.unique(ids)))
            test.assertTrue(np.all(ids < run._contact.count))
            # Decode keys independently through the unchanged sampling metadata.
            expected = {}
            samples = batch.all_samples
            fields = [getattr(samples, n).numpy() for n in ("sources", "targets", "faces", "slots")]
            face_index = {tuple(v): i for i, v in enumerate(spatial.query.faces)}
            target_index = {int(v): i for i, v in enumerate(spatial.query.targets)}
            for i, (source, target, face, slot) in enumerate(zip(*fields, strict=True)):
                key = (face_index[source, face] * len(target_index) + target_index[int(target)]) * 3 + slot
                test.assertNotIn(int(key), expected)
                expected[int(key)] = i
            keys = spatial.query.keys.numpy()[:count]
            np.testing.assert_array_equal(ids, sorted(expected[int(key)] for key in keys))
            batch.evaluate(*inputs, dt=0.001, **options)
            full = [p.result.numpy().copy() for p in (batch.pool.energy, batch.pool.residual, batch.pool.force)]
            b = batch_matrix(batch)
            reference = batch.start.y.numpy().copy()
            batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            for actual, wanted in zip((batch.pool.energy, batch.pool.residual, batch.pool.force), full, strict=True):
                np.testing.assert_allclose(actual.result.numpy(), wanted, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(batch_matrix(batch), b, rtol=2e-5, atol=2e-4)
            np.testing.assert_array_equal(batch.start.y.numpy(), reference)
            if kind == "chain":
                # Noncontiguous IDs exercise the legacy/actor boundary and
                # gather old references by original identity, not chunk offset.
                query(run, state)
                keep = ids[::5]
                sparse = spatial.identities.numpy().copy()
                sparse[: len(keep)] = keep
                spatial.identities.assign(sparse)
                spatial.count.fill_(len(keep))
                spatial.check()
                batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
                test.assertEqual(int(batch.flags.numpy()[0]), 0)
                np.testing.assert_allclose(
                    batch.pool.energy.result.numpy()[:2],
                    run._contact.energy.numpy()[keep.astype(int)].sum(axis=0),
                    rtol=1e-9,
                    atol=1e-12,
                )
                np.testing.assert_array_equal(batch.start.y.numpy(), reference)


def test_spatial_sweep(test, device):
    """Cover A motion, R arbitrary rotations/COM offsets and T vertex motion from the physical step start."""
    run, state, batch, spatial = spatial_fixture(device, "chain")
    k = run._kinematics
    start = physical(state)
    direction = np.linspace(-1, 2, run._layout.scalar_dof_count)
    perturb(run, state, start, direction, 0.2)
    inputs, options = native_inputs(run, state)
    spatial.update(k.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()
    lower, upper = spatial.face_lo.numpy(), spatial.face_hi.numpy()
    source = batch.all_samples.sources.numpy()
    face = batch.all_samples.faces.numpy()
    for alpha in np.linspace(0, 1, 17):
        perturb(run, state, start, direction, float(alpha * 0.2))
        inputs, options = native_inputs(run, state)
        rotation = options["rigid_rotation"]
        rotation.assign(rotation.numpy() * (1 + run._contact.quaternion_norm_tolerance / 2))
        # Actual sampled points come from the existing native geometry kernel.
        # Query actual native points through the established chunk stage,
        # independently of the new bounding geometry.
        for mode in batch._chunks():
            batch._stage(batch._state(*inputs, **options), mode, 0.001, begin=False)
            begin, n = int(batch.first.numpy()[0]), int(batch.pool.candidates.count.numpy()[0])
            points = batch.points.numpy()[:n]
            for j, point in enumerate(points):
                index = spatial.face_index[source[begin + j], face[begin + j]]
                test.assertTrue(np.all(point >= lower[index]) and np.all(point <= upper[index]))


def test_spatial_failure_graph(test, device):
    """Keep fixed buffers, repeat/restore candidates and propagate hard errors separately from overflow."""
    run, state, batch, spatial = spatial_fixture(device, "A-R")
    k = run._kinematics
    inputs, options = native_inputs(run, state)
    with patch.object(run._contact, "nq", k.count + 1):
        with test.assertRaises(ValueError):
            _ContactSpatial(
                batch,
                k,
                k.gather_q(state),
                inputs[0],
                inputs[1],
                **options,
                capacity=run._contact.count,
                law_margin=0.0,
            )
    q, x, com = k.gather_q(state), state.particle_q, state._monolithic_coupled.rigid_com
    spatial.update(q, x, rigid_com=com)
    n = int(spatial.count.numpy()[0])
    original = spatial.identities.numpy()[:n].copy()
    pointers = [a.ptr for a in spatial.buffers()]
    test.assertGreater(float(np.linalg.norm(spatial.query.face_lo.numpy())), 0)
    with patch.object(device.get_allocator(), "allocate", side_effect=AssertionError("runtime allocation")):
        spatial.update(q, x, rigid_com=com)
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            spatial.update(q, x, rigid_com=com)
        saved = com.numpy().copy()
        com.assign(saved + 0.1)
        wp.capture_launch(capture.graph)
        com.assign(saved)
        wp.capture_launch(capture.graph)
    test.assertEqual(int(spatial.count.numpy()[0]), n)
    np.testing.assert_array_equal(spatial.identities.numpy()[:n], original)
    test.assertEqual(pointers, [a.ptr for a in spatial.buffers()])
    q.fill_(np.nan)
    spatial.update(q, x, rigid_com=com)
    with test.assertRaises(ValueError):
        spatial.check()
    inputs, options = native_inputs(run, state)
    batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
    test.assertNotEqual(int(batch.flags.numpy()[0]), 0)
    test.assertEqual(int(batch.pool.records.count.numpy()[0]), 0)
    for capacity in (0, 1, 7):
        limited_run, s, b, limited = spatial_fixture(device, "A-R", capacity=capacity)
        limited.update(limited.q0, s.particle_q, rigid_com=s._monolithic_coupled.rigid_com)
        with test.assertWarns(RuntimeWarning):
            limited.check()
        test.assertEqual(int(limited.count.numpy()[0]), capacity)
        # Overflow is permitted; no partial endpoint record may be emitted.
        inputs, options = native_inputs(limited_run, s)
        b.evaluate(*inputs, dt=0.001, selection=limited, **options)
        test.assertEqual(int(b.flags.numpy()[0]), 0)
        test.assertLessEqual(int(b.pool.records.count.numpy()[0]), capacity)
    # A restored valid state must clear the previous hard error.
    spatial.update(k.gather_q(state), state.particle_q, rigid_com=com)
    spatial.check()
    inputs, options = native_inputs(run, state)
    if n:
        bad = spatial.identities.numpy().copy()
        bad[0] = run._contact.count
        spatial.identities.assign(bad)
        batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
        test.assertNotEqual(int(batch.flags.numpy()[0]), 0)
    source = spatial.face_shapes.numpy().copy()
    corrupt = source.copy()
    corrupt[0] = run.model.shape_count
    spatial.face_shapes.assign(corrupt)
    spatial.update(k.gather_q(state), state.particle_q, rigid_com=com)
    with test.assertRaises(ValueError):
        spatial.check()
    spatial.face_shapes.assign(source)
    spatial.update(k.gather_q(state), state.particle_q, rigid_com=com)
    spatial.check()
    shifted = wp.array(ptr=spatial.q0.ptr, shape=spatial.q0.shape, dtype=spatial.q0.dtype, device=device, copy=False)
    with test.assertRaisesRegex(ValueError, "aliases"):
        spatial.update(shifted, state.particle_q, rigid_com=com)
    model = contact_scene(device, "R-R")
    model.shape_contact_pairs = wp.empty((0, 2), dtype=int, device=device)
    model.shape_contact_pair_count = 0
    empty_run = contact_solver(model)
    empty_state = contact_state(empty_run, "R-R")
    inputs, options = native_inputs(empty_run, empty_state)
    empty_batch = _ContactBatch(empty_run._contact, batch_capacity=0, capacity=0)
    empty_batch.begin_step(*inputs, **options)
    empty = _ContactSpatial(
        empty_batch,
        empty_run._kinematics,
        empty_run._kinematics.gather_q(empty_state),
        inputs[0],
        inputs[1],
        **options,
        capacity=0,
        law_margin=0.0,
    )
    empty.update(empty.q0, inputs[0], rigid_com=options["rigid_com"])
    empty.check()
    empty_batch.evaluate(*inputs, dt=0.001, selection=empty, **options)
    test.assertEqual(int(empty.count.numpy()[0]), 0)
    test.assertEqual(int(empty_batch.flags.numpy()[0]), 0)


def test_spatial_projected_surface(test, device):
    """Include native radial sphere samples outside chord triangles and retain static metadata under rotation."""
    model = contact_scene(device, "R-R")
    shape = model.body_label.index("rigid_0")
    shape = int(np.flatnonzero(model.shape_body.numpy() == shape)[0])
    types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
    types[shape], scales[shape] = int(GeoType.SPHERE), (0.01, 0, 0)
    model.shape_type.assign(types)
    model.shape_scale.assign(scales)
    run = contact_solver(model)
    state = contact_state(run, "R-R")
    inputs, options = native_inputs(run, state)
    batch = _ContactBatch(run._contact, batch_capacity=256, capacity=0)
    spatial = _ContactSpatial(
        batch,
        run._kinematics,
        run._kinematics.gather_q(state),
        inputs[0],
        inputs[1],
        **options,
        capacity=run._contact.count,
        law_margin=0.0,
    )
    p, _, _, tri = _surface_samples(GeoType.SPHERE, scales[shape].astype(float), _return_triangles=True)
    points = p.reshape(-1, 3, 3)
    test.assertTrue(np.any(points > tri.max(axis=1)[:, None]) or np.any(points < tri.min(axis=1)[:, None]))
    lo, hi = spatial.face_local_lo.numpy(), spatial.face_local_hi.numpy()
    for face, sample in enumerate(points):
        index = spatial.face_index[shape, face]
        test.assertTrue(np.all(sample >= lo[index]) and np.all(sample <= hi[index]))
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()


def test_spatial_native_precision(test, device):
    """Use native FP32 inputs with a separate arithmetic allowance while preserving selected-state terms."""
    run, state, batch, spatial = spatial_fixture(device, "chain", precision="fp32")
    inputs, options = native_inputs(run, state)
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()
    batch.evaluate(*inputs, dt=0.001, **options)
    expected = batch.pool.residual.result.numpy().copy()
    batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected, rtol=1e-3, atol=2e-5)
    test.assertGreater(float(spatial.sweep.numerical.numpy().max()), 0)


def test_spatial_mesh_domain(test, device):
    """Use raw and C1/C2 SDF finite domains beyond mesh vertices, retaining real selected law evaluation."""
    if not device.is_cuda:
        test.skipTest("Mesh SDF textures require CUDA")
    model, x = fixture(device, "plane")
    data = synthetic()
    descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device)
    model.shape_type.assign(np.array([int(GeoType.MESH)], np.int32))
    model.shape_scale.assign(np.ones((1, 3), np.float32))
    model._shape_sdf_index.assign(np.array([0], np.int32))
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    k = _TreeKinematics(model)
    q = wp.zeros(k.count, dtype=k.dtype, device=device)
    k.evaluate(q)
    inputs = wp.array(x, dtype=wp.vec3d, device=device), k.poses, k.com, k.jt, k.jw
    for representation in (None, "c1-hermite", "c2-bspline"):
        smooth = None
        if representation is not None:
            values = data["coarse_sdf"].astype(float)
            lower, upper = np.asarray(descriptor.sdf_box_lower), np.asarray(descriptor.sdf_box_upper)
            dims = np.array(values.shape[::-1])
            field = SmoothSDF(
                build_coefficients(values, (upper - lower) / (dims - 1), representation),
                lower,
                upper,
                dims,
                representation,
                device=device,
            )
            smooth = SmoothSDFSet([field], source_identity={"fixture": "polynomial"})
        c = ContactObjective(
            model,
            np.arange(model.particle_count, dtype=np.int32),
            q_dof_count=k.count,
            contact_model="C-II",
            normal_stiffness=1e7,
            friction_coefficient=0.5,
            friction_velocity_scale=0.01,
            smoothing_width=1e-4,
            particle_radius=0.001,
            smooth_sdf=smooth,
        )
        batch = _ContactBatch(c, batch_capacity=7, capacity=None, use_cuda_graph=True)
        batch.begin_step(*inputs)
        spatial = _ContactSpatial(batch, k, q, inputs[0], inputs[1], capacity=c.count, law_margin=0.0)
        # The synthetic texture's boundary is negative; exterior support must
        # extend beyond the texture domain, regardless of absent mesh vertices.
        test.assertTrue(np.all(spatial.target_local_lo.numpy()[0] < descriptor.sdf_box_lower))
        test.assertTrue(np.all(spatial.target_local_hi.numpy()[0] > descriptor.sdf_box_upper))
        spatial.update(q, inputs[0])
        spatial.check()
        batch.evaluate(*inputs, dt=0.001)
        expected = batch.pool.residual.result.numpy().copy()
        batch.evaluate(*inputs, dt=0.001, selection=spatial)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected, rtol=1e-8, atol=1e-10)
    test.assertIsNotNone(coarse)
    test.assertIsNotNone(fine)
    # Actor meshes retain their existing C2 requirement; no raw/C1 expansion.
    model = contact_scene(device, "A-R", mesh=True)
    run = contact_solver(model, smooth_sdf=build_smooth_sdf(model))
    state = contact_state(run, "A-R")
    inputs, options = native_inputs(run, state)
    batch = _ContactBatch(run._contact, batch_capacity=7, capacity=None)
    batch.begin_step(*inputs, **options)
    spatial = _ContactSpatial(
        batch,
        run._kinematics,
        run._kinematics.gather_q(state),
        inputs[0],
        inputs[1],
        **options,
        capacity=run._contact.count,
        law_margin=0.0,
    )
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()
    batch.evaluate(*inputs, dt=0.001, **options)
    expected = batch.pool.residual.result.numpy().copy()
    batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected, rtol=1e-8, atol=1e-10)


def test_spatial_known_miss(test, device):
    """Expose a frozen distant list missing a later contact, without querying or resetting the start reference."""
    run = contact_solver(contact_scene(device, "A-R"), law="C-II")
    state = contact_state(run, "A-R")
    original = state._monolithic_coupled.rigid_com.numpy().copy()
    state._monolithic_coupled.rigid_com.assign(original + np.array([0.3, 0, 0]))
    inputs, options = native_inputs(run, state)
    batch = _ContactBatch(run._contact, batch_capacity=7, capacity=None)
    batch.begin_step(*inputs, **options)
    spatial = _ContactSpatial(
        batch,
        run._kinematics,
        run._kinematics.gather_q(state),
        inputs[0],
        inputs[1],
        **options,
        capacity=run._contact.count,
        law_margin=0.0,
    )
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    test.assertEqual(int(spatial.count.numpy()[0]), 0)
    spatial.check()
    frozen = batch.start.y.numpy().copy()
    state._monolithic_coupled.rigid_com.assign(original)
    inputs, options = native_inputs(run, state)
    batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
    test.assertEqual(np.linalg.norm(batch.pool.residual.result.numpy()), 0)
    batch.evaluate(*inputs, dt=0.001, **options)
    test.assertGreater(np.linalg.norm(batch.pool.residual.result.numpy()), 1e-4)
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()
    batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
    test.assertGreater(np.linalg.norm(batch.pool.residual.result.numpy()), 1e-4)
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)


def test_spatial_chunk_counts(test, device):
    """Schedule only occupied chunks across zero, batch and legacy/actor boundaries."""
    run, _state, batch, spatial = spatial_fixture(device, "chain")
    legacy, total = run._contact.legacy_count, run._contact.count
    test.assertGreater(legacy, 8)
    test.assertGreater(total - legacy, 8)
    for left, right in ((0, 0), (1, 0), (7, 1), (8, 8), (legacy, total - legacy), (0, 1), (0, 0)):
        ids = np.r_[np.arange(left), legacy + np.arange(right)].astype(np.uint64)
        if len(ids):
            spatial.identities[: len(ids)].assign(ids)
        spatial.count.fill_(len(ids))
        spatial.check()
        actual = []
        for mode in batch._chunks(spatial):
            actual.append((mode, int(batch.pool.candidates.count.numpy()[0])))
        expected = [(mode, min(7, n - offset)) for mode, n in ((2, left), (3, right)) for offset in range(0, n, 7)]
        test.assertEqual(actual, expected)


class TestContactSpatial(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_spatial_pairs,
        test_spatial_sweep,
        test_spatial_failure_graph,
        test_spatial_known_miss,
        test_spatial_projected_surface,
        test_spatial_native_precision,
        test_spatial_mesh_domain,
        test_spatial_chunk_counts,
    ):
        add_function_test(TestContactSpatial, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
