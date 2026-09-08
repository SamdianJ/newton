# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare candidate AABB gates with the original fixed-table contact oracle."""

import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton.examples.softbody import monolithic_contact_friction as friction_fixture
from newton.tests.test_solver_monolithic_collision import _scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def keyed_contacts(contacts):
    """Extract records in stable key order, ignoring atomic append order."""
    tids = contacts.soft_contact_tids.numpy()
    keys = np.flatnonzero(tids >= 0)
    records = tids[keys]
    return keys, {
        name: getattr(contacts, name).numpy()[records]
        for name in (
            "soft_contact_indices",
            "soft_contact_shape",
            "soft_contact_barycentric",
            "soft_contact_normal",
            "soft_contact_body_pos",
            "soft_contact_body_vel",
        )
    }


def assert_contacts_equal(test, a, b):
    """Require identical sampled geometry for a fixed candidate."""
    ka, va = keyed_contacts(a)
    kb, vb = keyed_contacts(b)
    np.testing.assert_array_equal(ka, kb)
    for name in va:
        np.testing.assert_array_equal(va[name], vb[name], err_msg=name)
    test.assertEqual(int(a.soft_contact_count.numpy()[0]), len(ka))


def test_candidate_equivalence(test, device):
    """Verify all analytic geometries, rotations, translations and A-B-A replay."""
    rng = np.random.default_rng(731)
    for shape in ("sphere", "box", "capsule", "cylinder", "cone", "plane"):
        model, state = _scene(device, shape=shape, margin=0.002)
        transform = wp.transform((0.003, -0.005, 0.001), wp.quat_rpy(0.3, -0.2, 0.7))
        model.shape_transform.assign([transform])
        pipelines = [
            MonolithicCollisionPipeline(model, soft_contact_gap=0.003, _enable_aabb=flag) for flag in (True, False)
        ]
        contacts = [p.contacts() for p in pipelines]
        rest = state.particle_q.numpy()
        samples = rng.uniform(-0.07, 0.07, (30, 3))
        samples = np.concatenate((samples, [[10, 10, 10]], samples[:1]))
        for i, offset in enumerate(samples):
            body = wp.transform((0.3, -0.1, 0.4), wp.quat_rpy(0.4, 0.2, -0.3))
            state.body_q.assign([body])
            state.particle_q.assign([wp.transform_point(body, wp.vec3(v + offset)) for v in rest])
            for pipeline, buffer in zip(pipelines, contacts, strict=True):
                pipeline.collide(state, buffer)
                test.assertEqual(int(buffer.contact_generation.numpy()[0]), i + 1)
            assert_contacts_equal(test, *contacts)
            if i == 30 and shape != "plane":
                np.testing.assert_array_equal(pipelines[0]._query_counts.numpy(), [4, 0, 0, 0])
                np.testing.assert_array_equal(contacts[0].soft_contact_tids.numpy(), -1)


def test_boundary_shell(test, device):
    """Retain tangency, positive-gap records and conservative expanded bounds."""
    model, state = _scene(device, shape="box", margin=0.001)
    pipelines = [MonolithicCollisionPipeline(model, soft_contact_gap=0.002, _enable_aabb=v) for v in (True, False)]
    contacts = [p.contacts() for p in pipelines]
    rest = state.particle_q.numpy()
    for distance in (0, -1e-7, 1e-7, -0.002, 0.002, 1.0):
        x = rest.copy()
        x[:, 2] += 0.01 + 0.001 + 0.001 + 0.002 - rest[:, 2].min() + distance
        state.particle_q.assign(x)
        for p, c in zip(pipelines, contacts, strict=True):
            p.collide(state, c)
        assert_contacts_equal(test, *contacts)


def test_scratch_and_invalid(test, device):
    """Keep candidate storage fixed and invalidate failed streams exactly once."""
    model, state = _scene(device, shape="box")
    pipeline = MonolithicCollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    with (
        patch.object(wp, "empty", side_effect=AssertionError("candidate allocation")),
        patch.object(wp, "zeros", side_effect=AssertionError("candidate allocation")),
        patch.object(wp, "empty_like", side_effect=AssertionError("candidate allocation")),
    ):
        for _ in range(3):
            pipeline.collide(state, contacts)
    x = state.particle_q.numpy()
    x[:] += 10
    x[0, 0] = np.nan
    state.particle_q.assign(x)
    with patch.object(contacts, "clear", wraps=contacts.clear) as clear:
        with test.assertRaisesRegex(ValueError, "INVALID_SDF_GRADIENT"):
            pipeline.collide(state, contacts)
        clear.assert_called_once_with(bump_generation=True)
    np.testing.assert_array_equal(contacts.soft_contact_tids.numpy(), -1)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    pipeline._enable_aabb = False
    with test.assertRaisesRegex(ValueError, "changed_bounds_configuration"):
        pipeline.collide(state, contacts)
    for error in (-1, np.nan, np.inf, 0.02):
        with test.assertRaises(ValueError):
            MonolithicCollisionPipeline(model, _sdf_query_error=error)


def test_texture_bounds(test, device):
    """Compare CUDA mesh records inside, on and outside the texture domain."""
    model, state = _scene(device, shape="mesh")
    model.shape_scale.assign([[1.7, 1.7, 1.7]])
    pipelines = [
        MonolithicCollisionPipeline(model, soft_contact_gap=0.004, _sdf_query_error=0.0004, _enable_aabb=v)
        for v in (True, False)
    ]
    contacts = [p.contacts() for p in pipelines]
    rest = state.particle_q.numpy()
    for offset in np.linspace(-0.1, 0.1, 51):
        state.particle_q.assign(rest + np.array([offset, offset / 2 + 0.00137, offset / 3 + 0.00071]))
        failures = []
        for p, c in zip(pipelines, contacts, strict=True):
            try:
                p.collide(state, c)
                failures.append(None)
            except MonolithicCollisionPipeline.Error as error:
                failures.append(error.status)
        # The box's medial axis can have zero texture gradient; both paths must reject it.
        test.assertEqual(*failures)
        assert_contacts_equal(test, *contacts)


class TestMonolithicAABB(unittest.TestCase):
    pass


def test_plane_union_and_empty(test, device):
    """Never cull planes with the finite union and clear empty eligible tables."""
    model, state = _scene(device, shape="mixed")
    pipelines = [MonolithicCollisionPipeline(model, _enable_aabb=v) for v in (True, False)]
    contacts = [p.contacts() for p in pipelines]
    for p, c in zip(pipelines, contacts, strict=True):
        p.collide(state, c)
    assert_contacts_equal(test, *contacts)
    np.testing.assert_array_equal(pipelines[0]._query_counts.numpy(), [4, 0, 4, 12])
    model.shape_flags.zero_()
    empty = MonolithicCollisionPipeline(model)
    buffer = empty.contacts()
    for i in range(3):
        empty.collide(state, buffer)
        test.assertEqual(int(buffer.contact_generation.numpy()[0]), i + 1)
        test.assertEqual(empty.soft_contact_max, 0)
        np.testing.assert_array_equal(empty._query_counts.numpy(), 0)


def test_physics_parity(test, device):
    """Preserve coupled trajectories, physical forces, history and global assembly."""
    cases = []
    for enabled in (True, False):
        with patch.object(
            friction_fixture, "MonolithicCollisionPipeline", partial(MonolithicCollisionPipeline, _enable_aabb=enabled)
        ):
            cases.append(friction_fixture.FrictionCase(device))
    for _ in range(160):
        for case in cases:
            case.step()
        for name in ("particle_q", "particle_qd", "joint_q", "joint_qd"):
            a_state = getattr(cases[0].state, name).numpy().astype(float)
            b_state = getattr(cases[1].state, name).numpy().astype(float)
            tolerance = 1e-5 if device.is_cpu else 5e-5
            if name == "particle_qd":
                # BE differences two float32 positions. AABB-on/on replay exhibits
                # the same velocity quantization; use its two-position ULP bound.
                position_scale = np.max(np.abs(cases[0].rest)).astype(np.float32)
                velocity_ulp = 2 * float(np.spacing(position_scale)) / cases[0].dt
                np.testing.assert_allclose(a_state, b_state, rtol=tolerance, atol=velocity_ulp)
            else:
                error = np.linalg.norm(a_state - b_state) / max(np.linalg.norm(b_state), 1e-12)
                test.assertLessEqual(error, tolerance, (name, error))
        test.assertEqual(cases[0].solver.last_stats.status, cases[1].solver.last_stats.status)
        test.assertEqual(cases[0].solver._history_epoch, cases[1].solver._history_epoch)
    a, b = (case.solver for case in cases)
    test.assertGreater(a.last_stats.active_sample_count, 0)
    np.testing.assert_array_equal(a._contact.committed.valid.numpy(), b._contact.committed.valid.numpy())
    np.testing.assert_allclose(a._contact.committed.xi_local.numpy(), b._contact.committed.xi_local.numpy(), atol=1e-7)
    matrices = [s._linear.densify_for_test(generation=s._generation).raw_matrix for s in (a, b)]
    # Compare ownership blocks with the PRD Frobenius relative-error convention.
    nq = cases[0].model.joint_dof_count
    for rows in (slice(0, nq), slice(nq, None)):
        for cols in (slice(0, nq), slice(nq, None)):
            left, right = (m[rows, cols] for m in matrices)
            error = np.linalg.norm(left - right) / max(np.linalg.norm(left), np.linalg.norm(right), 1e-12)
            test.assertLessEqual(error, 1e-5 if device.is_cpu else 5e-5)
    for s in (a, b):
        s.update_contacts(s.contacts, cases[0 if s is a else 1].state)
    for field in ("normal_force_sum", "tangent_force_sum", "elastic_energy"):
        if field in a.last_stats.contact_history:
            np.testing.assert_allclose(
                a.last_stats.contact_history[field], b.last_stats.contact_history[field], rtol=2e-5, atol=1e-6
            )


for _device in get_test_devices():
    for _test in (
        test_candidate_equivalence,
        test_boundary_shell,
        test_scratch_and_invalid,
        test_plane_union_and_empty,
    ):
        add_function_test(TestMonolithicAABB, _test.__name__, _test, devices=[_device])
    add_function_test(TestMonolithicAABB, "test_physics_parity", test_physics_parity, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestMonolithicAABB, "test_texture_bounds", test_texture_bounds, devices=[_device])


if __name__ == "__main__":
    unittest.main()
