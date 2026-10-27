# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify articulated surface contact against unowned world-static geometry."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contact_oracle import box_face_samples
from scripts.monolithic_reference.articulation_oracle import Kinematics, transform
from scripts.monolithic_reference.contact_oracle import geometry, local


def static_scene(device, *, kind="plane", static_first=False, filtered=False, root_shape=False):
    """Build a fixed root with revolute/prismatic coordinates and a world target."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    cfg = builder.ShapeConfig(density=0, margin=0.0002, has_particle_collision=False)

    def target():
        if kind == "plane":
            return builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg)
        return builder.add_shape_box(
            -1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform((0, 0, -0.025), wp.quat_identity()), cfg=cfg
        )

    if static_first:
        fixed = target()
    root = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
    joints = [builder.add_joint_fixed(-1, root, parent_xform=wp.transform((0, 0, 0.01), wp.quat_identity()))]
    parent = root
    for method in (builder.add_joint_revolute, builder.add_joint_prismatic):
        body = builder.add_link(mass=0.2, com=(0.002, 0.001, 0.001), inertia=wp.diag(wp.vec3(0.001)))
        joints.append(
            method(
                parent,
                body,
                axis=newton.Axis.Y if parent == root else newton.Axis.Z,
                limit_ke=0,
                limit_kd=0,
                armature=0,
                damping=0,
                friction=0,
                collision_filter_parent=False,
            )
        )
        parent = body
    builder.add_articulation(joints)
    builder.joint_q[:] = [0.03, 0.0001]
    source = builder.add_shape_box(
        body, hx=0.02, hy=0.015, hz=0.012, xform=wp.transform((0.003, -0.002, 0), wp.quat_identity()), cfg=cfg
    )
    if not static_first:
        fixed = target()
    if filtered:
        builder.shape_collision_filter_pairs.append((source, fixed))
    if root_shape:
        builder.add_shape_box(root, hx=0.01, hy=0.01, hz=0.01, cfg=cfg)
    return builder.finalize(device=device), source, fixed


def static_solver(model, *, precision="fp64", **options):
    return SolverMonolithic(
        model,
        precision=precision,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        friction_velocity_scale=0.01,
        normal_smoothing=1e-4,
        alignment_policy="mask",
        **options,
    )


def static_terms(model, source, target, q, qn):
    """Integrate source P1Q3 using independent FK, geometry and C-IE formulas."""
    com, rotation, jt, dr, _, _ = Kinematics(model).evaluate(q)
    old_com, old_rotation, _, _, _, _ = Kinematics(model).evaluate(qn)
    body = int(model.shape_body.numpy()[source])
    body_com = model.body_com.numpy()[body]
    shape = transform(model.shape_transform.numpy()[source])
    world_target = transform(model.shape_transform.numpy()[target])
    rt = world_target[:3, :3]
    origin = com[body] - rotation[body] @ body_com
    old_origin = old_com[body] - old_rotation[body] @ body_com
    kind = "plane" if int(model.shape_type.numpy()[target]) == int(newton.GeoType.PLANE) else "box"
    scales = model.shape_scale.numpy().astype(float)
    offset = model.shape_margin.numpy().astype(float)[[source, target]].sum()
    energy, residual, matrix = np.zeros(2), np.zeros(len(q)), np.zeros((len(q), len(q)))
    wrench = np.zeros(6)
    for point, weight in zip(*box_face_samples(scales[source]), strict=True):
        local_point = shape[:3, :3] @ point + shape[:3, 3]
        world = rotation[body] @ local_point + origin
        old_world = old_rotation[body] @ local_point + old_origin
        y, yn = rt.T @ (world - world_target[:3, 3]), rt.T @ (old_world - world_target[:3, 3])
        normal = np.zeros(3)
        axis = np.argmax(abs(point) / scales[source])
        normal[axis] = np.sign(point[axis])
        if (rotation[body] @ shape[:3, :3] @ normal) @ (rt @ geometry(y, kind, scales[target])[1]) > 0:
            continue
        jacobian = rt.T @ (jt[body].T + np.einsum("nij,j->in", dr[body], local_point - body_com))
        e, r, _, fitted = local(y, yn, kind, scales[target], offset=offset, stiffness=1e7 * weight)
        energy += e
        residual += jacobian.T @ r
        matrix += jacobian.T @ fitted @ jacobian
        force = -rt @ r
        wrench[:3] += force
        wrench[3:] += np.cross(world - com[body], force)
    return energy, residual, matrix, wrench


def test_static_contact_terms(test, device):
    """Match energies, generalized forces, PSD tangent and body wrench in both precisions."""
    for kind, first, precision in itertools.product(("plane", "box"), (False, True), ("fp64", "fp32")):
        with test.subTest(kind=kind, static_first=first, precision=precision):
            model, source, target = static_scene(device, kind=kind, static_first=first)
            run = static_solver(model, precision=precision)
            test.assertEqual(run._contact.actor_samples.count, 36)
            test.assertEqual(set(run._contact.actor_samples.pair_kinds.values()), {"A-static"})
            state = run.create_state(dt=0.001)
            batch = native_batch(run)
            qn = state.joint_q.numpy().astype(float)
            query(batch, run, state, start=True)
            state.joint_q.assign(qn + np.array([2e-5, -1e-6]))
            q = state.joint_q.numpy().astype(float)
            query(batch, run, state)
            e, r, b, wrench = static_terms(model, source, target, q, qn)
            tol = 2e-3 if precision == "fp32" else 2e-8
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            test.assertGreater(np.linalg.norm(r), 1e-3)
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], e, rtol=tol, atol=tol * 1e-5)
            np.testing.assert_allclose(batch.pool.residual.result.numpy(), r, rtol=tol, atol=tol * 1e-2)
            np.testing.assert_allclose(batch_matrix(batch), b, rtol=tol, atol=tol)
            body = int(model.shape_body.numpy()[source])
            actual = batch.pool.force.result.numpy().reshape(-1, 6)[body]
            np.testing.assert_allclose(actual, wrench, rtol=tol, atol=tol * 1e-2)
            np.testing.assert_allclose(batch_matrix(batch), batch_matrix(batch).T, atol=tol)
            test.assertGreaterEqual(np.linalg.eigvalsh(batch_matrix(batch)).min(), -tol)


def test_static_contact_derivatives(test, device):
    """Compare residual differences to the independent law, not to its PSD approximation."""
    for kind in ("plane", "box"):
        model, source, target = static_scene(device, kind=kind)
        run = static_solver(model)
        test.assertEqual(run._contact.actor_samples.count, 36)
        state = run.create_state(dt=0.001)
        batch = native_batch(run)
        qn = state.joint_q.numpy().copy()
        query(batch, run, state, start=True)
        q = qn + np.array([2e-5, -1e-6])
        actual, reference = [], []
        for direction in np.eye(2):
            residuals, expected = [], []
            for sign in (-1, 1):
                state.joint_q.assign(q + sign * 1e-7 * direction)
                query(batch, run, state)
                residuals.append(batch.pool.residual.result.numpy().copy())
                expected.append(static_terms(model, source, target, state.joint_q.numpy(), qn)[1])
            actual.append((residuals[1] - residuals[0]) / 2e-7)
            reference.append((expected[1] - expected[0]) / 2e-7)
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-4)


def test_static_contact_filtering(test, device):
    """Keep explicit exclusions, same-tree filtering and zero-support endpoints intact."""
    model, source, target = static_scene(device, filtered=True, root_shape=True)
    run = static_solver(model)
    test.assertEqual(run._contact.actor_samples.count, 0)
    model, source, target = static_scene(device, root_shape=True)
    run = static_solver(model)
    test.assertEqual(run._contact.actor_samples.count, 36)
    test.assertEqual(set(run._contact.actor_samples.pair_kinds), {tuple(sorted((source, target)))})
    test.assertTrue(all(key[:2] == (source, target) for key in run._contact.actor_samples.keys))


def test_static_contact_step(test, device):
    """Produce a real reaction and common-origin equal/opposite pair wrenches."""
    for kind, first in itertools.product(("plane", "box"), (False, True)):
        model, source, target = static_scene(device, kind=kind, static_first=first)
        run = static_solver(model, pcg_mode="production", use_cuda_graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        qn = state.joint_q.numpy().copy()
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertFalse(run.last_stats.rolled_back)
        test.assertGreater(np.linalg.norm(state.joint_q.numpy() - qn), 1e-7)
        _, residual, _, wrench = static_terms(model, source, target, state.joint_q.numpy(), qn)
        force = run.get_contact_forces(state)
        body = int(model.shape_body.numpy()[source])
        np.testing.assert_allclose(force.body_wrench[body], wrench, rtol=2e-8, atol=1e-8)
        np.testing.assert_allclose(
            state._monolithic_coupled.contact_joint_force.numpy(), -residual, rtol=2e-8, atol=1e-8
        )
        pair = run.last_stats.variational["actor_contact_pairs"]
        test.assertEqual(len(pair), 1)
        test.assertEqual(pair[0]["kind"], "A-static")
        pair_wrenches = np.asarray(pair[0]["world_wrench"])
        test.assertGreater(np.linalg.norm(pair_wrenches[0, :3]), 1e-3)
        np.testing.assert_allclose(pair_wrenches.sum(axis=0), 0, atol=1e-10)
        source_end = int(source > target)
        com = Kinematics(model).evaluate(state.joint_q.numpy())[0][body]
        expected = np.r_[wrench[:3], wrench[3:] + np.cross(com, wrench[:3])]
        np.testing.assert_allclose(pair_wrenches[source_end], expected, rtol=2e-8, atol=1e-8)


def test_sphere_plane_step(test, device):
    """Resolve the single-tree sphere/plane omission with no FREE body or material."""
    for precision in ("fp64", "fp32"):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-5)))
        joint = builder.add_joint_prismatic(
            -1,
            body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform((0, 0, 0.009), wp.quat_identity()),
            limit_ke=0,
            limit_kd=0,
            armature=0,
            damping=0,
            friction=0,
        )
        builder.add_articulation([joint])
        cfg = builder.ShapeConfig(density=0, has_particle_collision=False)
        builder.add_shape_sphere(body, radius=0.01, cfg=cfg)
        builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg)
        model = builder.finalize(device=device)
        run = static_solver(model, precision=precision, pcg_mode="production", use_cuda_graph=device.is_cuda)
        test.assertEqual(run._contact.actor_samples.count, 5952)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertGreater(state.joint_q.numpy()[0], 1e-7)
        force = run.get_contact_forces(state).body_wrench[body]
        test.assertGreater(force[2], 0.01)
        np.testing.assert_allclose(force[[0, 1, 3, 4, 5]], 0, atol=1e-6)


class TestStaticArticulationContact(unittest.TestCase):
    pass


for _test in (
    test_static_contact_terms,
    test_static_contact_derivatives,
    test_static_contact_filtering,
    test_static_contact_step,
    test_sphere_plane_step,
):
    add_function_test(TestStaticArticulationContact, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
