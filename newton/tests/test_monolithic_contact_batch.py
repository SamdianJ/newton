# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native bounded contact generation against full geometry and independent C-IE oracles."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.geometry import GeoType
from newton._src.geometry.sdf_texture import TextureSDFData, create_sparse_sdf_textures
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton.solvers.experimental.monolithic import build_smooth_sdf
from newton.tests.test_monolithic_contact_actor_pairs import native_inputs, perturb, physical
from newton.tests.test_monolithic_contact_components import batch_matrix, contact, fixture, state
from newton.tests.test_monolithic_contact_mesh import synthetic
from newton.tests.test_monolithic_contact_precision import native_inputs as precision_inputs
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contact_oracle import fixture_terms
from scripts.monolithic_reference.contact_oracle import evaluate


def test_batch_legacy(test, device):
    """Match the independent oracle when real particle-shape geometry is split across native chunks."""
    for shape in ("plane", "sphere", "box"):
        model, xn = fixture(device, shape)
        obj = contact(model, fixed=True)
        qn, q = np.zeros(2), np.array([0.003, 2e-6])
        x = xn + np.array([1e-6, 2e-6, 1e-6])
        x[0] = xn[0]
        oracle = evaluate(model, q, x, qn, xn, kinds=[shape], mapping=obj.mapping.numpy())
        batch = _ContactBatch(obj, batch_capacity=2, capacity=obj.count * 11, use_cuda_graph=device.is_cuda)
        batch.begin_step(*state(model, xn, qn))
        frozen = batch.start.y.numpy().copy()
        batch.evaluate(*state(model, x, q), dt=0.001)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        test.assertEqual(int(batch.pool.records.flags.numpy()[0]), 0)
        np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], oracle[0].sum(axis=0), rtol=3e-10, atol=1e-15)
        np.testing.assert_allclose(batch.pool.residual.result.numpy(), oracle[1], rtol=3e-10, atol=1e-12)
        np.testing.assert_allclose(batch_matrix(batch), oracle[3], rtol=2e-5, atol=2e-4)
        np.testing.assert_array_equal(batch.start.y.numpy(), frozen)
        test.assertEqual(batch.current.jacobian_q.size, 0)
        test.assertEqual(batch.pool.candidates.factors.shape[0], 2)


def test_batch_actor_pairs(test, device):
    """Generate A-R/R-R/R-T/static contacts from compact metadata, including COM and layout shifts."""
    for kind in ("A-R", "R-R", "R-T", "R-static", "chain"):
        run = contact_solver(contact_scene(device, kind, before=True), permuted=True)
        s = contact_state(run, kind)
        start = physical(s)
        batch = _ContactBatch(
            run._contact, batch_capacity=7, capacity=run._contact.count * 18, use_cuda_graph=device.is_cuda
        )
        inputs, options = native_inputs(run, s)
        batch.begin_step(*inputs, **options)
        direction = np.linspace(-0.2, 0.3, run._layout.scalar_dof_count)
        perturb(run, s, start, direction, 1e-6)
        inputs, options = native_inputs(run, s)
        batch.evaluate(*inputs, dt=0.001, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        test.assertEqual(int(batch.pool.records.flags.numpy()[0]), 0)
        expected_energy, expected_r, expected_b = fixture_terms(run.model, run._actors, physical(s), start)
        np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], expected_energy, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected_r, rtol=1e-8, atol=1e-10)
        test.assertLess(np.linalg.norm(batch_matrix(batch) - expected_b) / max(1, np.linalg.norm(expected_b)), 1e-5)
        pairs = batch.pool.records.pairs.numpy()[: int(batch.pool.records.count.numpy()[0])]
        np.testing.assert_allclose(pairs.sum(axis=1), 0, atol=1e-14)
        # Independent total world-origin force/torque balance, including fixed tet nodes.
        forces = batch.pool.force.result.numpy()
        particle = forces[: 3 * run.model.particle_count].reshape(-1, 3)
        body = forces[3 * run.model.particle_count :].reshape(-1, 6)
        qforce = -batch.pool.residual.result.numpy()
        test.assertTrue(np.isfinite(qforce).all())
        world_force = particle.sum(axis=0) + body[:, :3].sum(axis=0)
        centers = np.zeros((run.model.body_count, 3))
        centers[list(run._articulation_actor.body_indices)] = run._kinematics.com.numpy()
        for index, actor in enumerate(run._rigid.actors):
            centers[actor.body_indices[0]] = s._monolithic_coupled.rigid_com.numpy()[index]
        world_torque = np.cross(s.particle_q.numpy(), particle).sum(axis=0)
        world_torque += (body[:, 3:] + np.cross(centers, body[:, :3])).sum(axis=0)
        identities = batch.pool.records.identities.numpy()[: len(pairs)]
        sources, targets = batch.all_samples.sources.numpy(), batch.all_samples.targets.numpy()
        bodies = run.model.shape_body.numpy()
        for i, key in enumerate(identities):
            for end, shape in enumerate((sources[key], targets[key])):
                if shape >= 0 and bodies[shape] < 0:
                    world_force += pairs[i, end, :3]
                    world_torque += pairs[i, end, 3:]
        test.assertLess(np.linalg.norm(world_force), 1e-8)
        test.assertLess(np.linalg.norm(world_torque), 1e-9)


def test_batch_trial_and_failure(test, device):
    """Keep complete step-start references through activation, capacity drops, hard failure and graph reuse."""
    run = contact_solver(contact_scene(device, "R-static"))
    s = contact_state(run, "R-static")
    batch = _ContactBatch(run._contact, batch_capacity=5, capacity=0, use_cuda_graph=device.is_cuda)
    inputs, options = native_inputs(run, s)
    with test.assertRaises(ValueError):
        batch.evaluate(*inputs, dt=0.001, **options)
    batch.begin_step(*inputs, **options)
    frozen = batch.start.y.numpy().copy()
    for _ in range(3):
        batch.evaluate(*inputs, dt=0.001, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        test.assertGreater(int(batch.pool.records.dropped.numpy()[0]), 0)
        test.assertFalse(batch.pool.records.accepted.numpy().any())
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)
    bad = wp.clone(options["rigid_com"])
    bad.fill_(wp.vec3d(float("nan")))
    batch.evaluate(*inputs, dt=0.001, **{**options, "rigid_com": bad})
    test.assertNotEqual(int(batch.flags.numpy()[0]), 0)
    test.assertNotEqual(int(batch.pool.records.flags.numpy()[0]), 0)
    test.assertFalse(batch.pool.records.accepted.numpy().any())
    batch.evaluate(*inputs, dt=0.001, **options)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    if device.is_cuda:
        # Reuse the complete evaluation graph, including all fixed producer chunks.
        test.assertLessEqual(batch.graphs.captures["contact_batch_eval"], 3)
        captures = batch.graphs.captures["contact_batch_eval"]
        run._contact.margins = wp.clone(run._contact.margins)
        batch.evaluate(*inputs, dt=0.001, **options)
        test.assertEqual(batch.graphs.captures["contact_batch_eval"], captures + 1)


def test_batch_precision(test, device):
    """Exercise FP32 geometry, zero/radial slip and finite factors against the FP64 oracle."""
    for shape in ("plane", "sphere", "box"):
        model, xn = fixture(device, shape)
        obj = contact(model, dtype=wp.float32)
        batch = _ContactBatch(obj, batch_capacity=2, capacity=obj.count * 11, use_cuda_graph=device.is_cuda)
        q, xn = np.zeros(2, np.float32), xn.astype(np.float32)
        batch.begin_step(*precision_inputs(model, xn, q, wp.float32))
        for ratio in (0, 1e-10, 0.25, 1, 2, -2, 0):
            x = xn + np.float32(ratio * 1e-5) * np.array([0.6, 0.8, 0], np.float32)
            batch.evaluate(*precision_inputs(model, x, q, wp.float32), dt=0.001)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            # The oracle sees the same FP32-rounded coordinates in FP64 arithmetic.
            oracle = evaluate(model, q.astype(float), x.astype(float), q.astype(float), xn.astype(float), kinds=[shape])
            np.testing.assert_allclose(
                batch.pool.energy.result.numpy()[:2], oracle[0].sum(axis=0), atol=2e-5, rtol=1e-3
            )
            np.testing.assert_allclose(batch.pool.residual.result.numpy(), oracle[1], atol=2e-5, rtol=1e-3)
            matrix = batch_matrix(batch)
            test.assertTrue(np.isfinite(matrix).all())
            test.assertLess(np.linalg.norm(matrix - oracle[3]) / max(1, np.linalg.norm(oracle[3])), 1e-3)


def test_batch_new_activation(test, device):
    """Activate initially zero contacts and reject a trial by recomputing the step-start baseline."""
    run = contact_solver(contact_scene(device, "A-R"))
    s = contact_state(run, "A-R")
    accepted = physical(s)
    separated = accepted["com"] + np.array([0.1, 0, 0])
    s._monolithic_coupled.rigid_com.assign(separated)
    batch = _ContactBatch(
        run._contact, batch_capacity=5, capacity=run._contact.count * 12, use_cuda_graph=device.is_cuda
    )
    inputs, options = native_inputs(run, s)
    batch.begin_step(*inputs, **options)
    start = physical(s)
    batch.evaluate(*inputs, dt=0.001, **options)
    baseline = tuple(o.result.numpy().tobytes() for o in (batch.pool.energy, batch.pool.residual, batch.pool.force))
    frozen = batch.start.y.numpy().copy()
    test.assertEqual(int(batch.pool.records.count.numpy()[0]), 0)
    s._monolithic_coupled.rigid_com.assign(accepted["com"])
    inputs, options = native_inputs(run, s)
    batch.evaluate(*inputs, dt=0.001, **options)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    test.assertGreater(int(batch.pool.records.count.numpy()[0]), 0)
    _, expected_r, _ = fixture_terms(run.model, run._actors, physical(s), start)
    np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected_r, atol=1e-10, rtol=1e-8)
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)
    s._monolithic_coupled.rigid_com.assign(separated)
    inputs, options = native_inputs(run, s)
    batch.evaluate(*inputs, dt=0.001, **options)
    test.assertEqual(
        tuple(o.result.numpy().tobytes() for o in (batch.pool.energy, batch.pool.residual, batch.pool.force)),
        baseline,
    )
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)


def test_batch_mesh(test, device):
    """Keep CUDA texture and C2 actor mesh geometry independent of native chunking."""
    if not device.is_cuda:
        test.skipTest("Mesh SDF textures require CUDA")

    data = synthetic()
    descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device)
    model, xn = fixture(device, "plane")
    model.shape_type.assign(np.array([int(GeoType.MESH)], np.int32))
    model.shape_scale.assign(np.ones((1, 3), np.float32))
    model._shape_sdf_index.assign(np.array([0], np.int32))
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    xn[:, 2] -= 0.0003
    obj = contact(model)
    x = xn + np.array([1e-6, 2e-6, 1e-6])
    oracle = evaluate(model, np.zeros(2), x, np.zeros(2), xn, kinds=[data])
    batch = _ContactBatch(obj, batch_capacity=2, capacity=obj.count * 11, use_cuda_graph=True)
    batch.begin_step(*state(model, xn, np.zeros(2)))
    batch.evaluate(*state(model, x, np.zeros(2)), dt=0.001)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    np.testing.assert_allclose(batch.pool.residual.result.numpy(), oracle[1], atol=1e-12)
    test.assertIsNotNone(coarse)
    test.assertIsNotNone(fine)
    mesh_model = contact_scene(device, "A-R", mesh=True)
    run = contact_solver(mesh_model, smooth_sdf=build_smooth_sdf(mesh_model))
    s = contact_state(run, "A-R")
    inputs, options = native_inputs(run, s)
    results = []
    for capacity in (7, run._contact.count):
        batch = _ContactBatch(run._contact, batch_capacity=capacity, capacity=None, use_cuda_graph=True)
        batch.begin_step(*inputs, **options)
        batch.evaluate(*inputs, dt=0.001, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        results.append(batch.pool.residual.result.numpy().copy())
    test.assertGreater(np.linalg.norm(results[1]), 0)
    np.testing.assert_allclose(results[0], results[1], atol=1e-12, rtol=1e-10)


def test_batch_storage_contract(test, device):
    """Reject bad capacities and state/scratch alias before launch; handle an empty supported set."""
    model, xn = fixture(device, "plane")
    obj = contact(model)
    batch = _ContactBatch(obj, batch_capacity=2, capacity=obj.count * 11)
    inputs = list(state(model, xn, np.zeros(2)))
    backing = batch.pool.candidates.residual
    inputs[0] = wp.array(
        ptr=backing.ptr, dtype=wp.vec3d, shape=(model.particle_count,), capacity=backing.capacity, device=device
    )
    with patch.object(wp, "launch", side_effect=AssertionError("Launch before validating alias")):
        with test.assertRaises(ValueError):
            batch.begin_step(*inputs)
    external = wp.zeros(2 * model.particle_count, dtype=wp.vec3d, device=device)
    inputs[0] = external[::2]
    batch.current.gradient = external[model.particle_count : model.particle_count + 2]
    with patch.object(wp, "launch", side_effect=AssertionError("Launch before validating strided alias")):
        with test.assertRaises(ValueError):
            batch.begin_step(*inputs)
    with patch.object(wp, "array", side_effect=AssertionError("Allocation before validating capacity")):
        for b, c in ((True, 1), (0, 1), (obj.count + 1, 1), (2, -1), (2, True), (2, 2**31)):
            with test.assertRaises(ValueError):
                _ContactBatch(obj, batch_capacity=b, capacity=c)
    model.shape_flags.zero_()
    empty = contact(model)
    batch = _ContactBatch(empty, batch_capacity=0, capacity=0, use_cuda_graph=device.is_cuda)
    inputs = state(model, xn, np.zeros(2))
    batch.begin_step(*inputs)
    batch.evaluate(*inputs, dt=0.001)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    test.assertEqual(int(batch.pool.records.count.numpy()[0]), 0)
    test.assertFalse(batch.pool.energy.result.numpy().any())


def test_batch_storage_rebinding(test, device):
    """Reject aliases introduced by workspace rebinding after successful state validation."""
    model, xn = fixture(device, "plane")
    obj = contact(model)
    batch = _ContactBatch(obj, batch_capacity=2, capacity=obj.count * 11, use_cuda_graph=device.is_cuda)
    inputs = list(state(model, xn, np.zeros(2)))
    external = wp.zeros(model.particle_count + 2, dtype=wp.vec3d, device=device)
    inputs[0] = external[2:]
    inputs[0].assign(xn)
    size = wp.types.type_size_in_bytes(wp.vec3d)
    for changed in (
        external[2:4],
        wp.array(ptr=external.ptr, dtype=wp.vec3d, shape=(3,), capacity=external.capacity, device=device),
        wp.array(
            ptr=external.ptr,
            dtype=wp.vec3d,
            shape=(2,),
            strides=(2 * size,),
            capacity=external.capacity,
            device=device,
        ),
        wp.array(ptr=external.ptr, dtype=wp.spatial_vectord, shape=(2,), capacity=external.capacity, device=device),
    ):
        batch.points = external[:2]
        batch.begin_step(*inputs)
        batch.points = changed
        with patch.object(wp, "launch", side_effect=AssertionError("Launch before validating rebound alias")):
            with test.assertRaisesRegex(ValueError, "aliases producer storage"):
                batch.begin_step(*inputs)
        batch.points = external[:2]
        batch.begin_step(*inputs)

    batch.initialize_reference()
    backing = batch.reference_residual
    inputs[0] = wp.array(
        ptr=backing.ptr, dtype=wp.vec3d, shape=(model.particle_count,), capacity=backing.capacity, device=device
    )
    with patch.object(wp, "launch", side_effect=AssertionError("Launch before validating new reference storage")):
        with test.assertRaisesRegex(ValueError, "aliases producer storage"):
            batch.begin_step(*inputs)


def test_reference_graph_submission(test, device):
    """Replay complete references with one submission and no accumulated prior output."""
    run = contact_solver(contact_scene(device, "R-static"))
    current = contact_state(run, "R-static")
    inputs, options = native_inputs(run, current)
    expected = None
    for graph in (False, True) if device.is_cuda else (False,):
        batch = _ContactBatch(run._contact, batch_capacity=5, capacity=0, use_cuda_graph=graph)
        batch.initialize_reference()
        batch.begin_step(*inputs, **options)
        batch.evaluate_reference(*inputs, dt=0.001, **options)
        if expected is None:
            expected = batch.reference_residual.numpy().copy()
        for _ in range(2):
            batch.reference_residual.fill_(float("nan"))
            with patch.object(wp, "capture_launch", wraps=wp.capture_launch) as launch:
                batch.evaluate_reference(*inputs, dt=0.001, **options)
            if graph:
                test.assertEqual(launch.call_count, 1)
            np.testing.assert_array_equal(batch.reference_residual.numpy(), expected)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)


def test_batch_graph_submission(test, device):
    """Use bounded submissions while preserving whole-contact admission across chunks."""
    if not device.is_cuda:
        test.skipTest("CUDA graph submission check")
    run = contact_solver(contact_scene(device, "R-static"))
    current = contact_state(run, "R-static")
    inputs, options = native_inputs(run, current)
    expected = None
    for graph in (False, True):
        batch = _ContactBatch(run._contact, batch_capacity=5, capacity=run._contact.count * 12, use_cuda_graph=graph)
        batch.begin_step(*inputs, **options)
        batch.evaluate(*inputs, dt=0.001, **options)
        if expected is None:
            expected = batch.pool.residual.result.numpy().copy()
        for _ in range(2):
            with patch.object(wp, "capture_launch", wraps=wp.capture_launch) as launch:
                batch.evaluate(*inputs, dt=0.001, **options)
            if graph:
                test.assertEqual(launch.call_count, 2)
            np.testing.assert_array_equal(batch.pool.residual.result.numpy(), expected)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)


class TestContactBatch(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_batch_graph_submission,
        test_reference_graph_submission,
        test_batch_legacy,
        test_batch_actor_pairs,
        test_batch_trial_and_failure,
        test_batch_precision,
        test_batch_new_activation,
        test_batch_mesh,
        test_batch_storage_contract,
        test_batch_storage_rebinding,
    ):
        add_function_test(TestContactBatch, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
