# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify two-ended articulation contact with independent kinematics and contact laws."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.contact_spatial import _ContactSpatial
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, native_inputs, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contact_oracle import box_face_samples
from scripts.monolithic_reference.articulation_oracle import Kinematics, transform
from scripts.monolithic_reference.contact_oracle import geometry, local


def contact_trees(device, *, filtered=False, fixed_target=False, root_shapes=False):
    """Build two fixed roots, unequal chains and offset box/COM frames, without FREE bodies."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    shapes, roots = [], []
    for tree, x in enumerate((-0.018, 0.018)):
        root = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
        roots.append(root)
        joints = [builder.add_joint_fixed(-1, root, parent_xform=wp.transform((x, 0, 0.1), wp.quat_identity()))]
        parent = root
        for i in range(2 if tree == 0 else 1):
            body = builder.add_link(mass=0.2, com=(0.003, -0.002, 0.001), inertia=wp.diag(wp.vec3(0.001)))
            method = builder.add_joint_revolute if i == 0 else builder.add_joint_prismatic
            joints.append(
                method(
                    parent,
                    body,
                    axis=newton.Axis.Z if i == 0 else newton.Axis.Y,
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
        shapes.append(
            builder.add_shape_box(
                root if fixed_target and tree else body,
                hx=0.02,
                hy=0.015,
                hz=0.012,
                xform=wp.transform((0, 0.001, 0), wp.quat_rpy(0.02, -0.03, 0.04)),
                cfg=builder.ShapeConfig(density=0, margin=0.0002, has_particle_collision=False),
            )
        )
    builder.joint_q[:] = [0.11, 0.001, -0.07]
    if filtered:
        builder.shape_collision_filter_pairs.append(tuple(shapes))
    if root_shapes:
        for root in roots:
            builder.add_shape_box(
                root, hx=0.02, hy=0.015, hz=0.012, cfg=builder.ShapeConfig(density=0, has_particle_collision=False)
            )
    return builder.finalize(device=device)


def trees_solver(model, *, reverse=False, precision="fp64", **options):
    """Construct the public solver, optionally permuting independent actor row ranges."""
    actors = ActorTable.from_model(model, actor_order=(1, 0) if reverse else None)
    with patch.object(ActorTable, "from_model", return_value=actors):
        return SolverMonolithic(
            model,
            friction_velocity_scale=0.01,
            contact_stiffness=1e7,
            friction_coefficient=0.5,
            normal_smoothing=1e-4,
            alignment_policy="mask",
            precision=precision,
            **options,
        )


def frames(model, q):
    """Return shape transforms and body derivatives using only the independent matrix oracle."""
    com, rotations, jt, dr, _, _ = Kinematics(model).evaluate(q)
    result = []
    for body, stored in zip(model.shape_body.numpy(), model.shape_transform.numpy(), strict=True):
        body_frame = np.eye(4)
        body_frame[:3, :3] = rotations[body]
        body_frame[:3, 3] = com[body] - rotations[body] @ model.body_com.numpy()[body]
        result.append(body_frame @ transform(stored))
    return result, com, rotations, jt, dr


def contact_terms(model, q, qn, *, fixed_target=False):
    """Integrate both directed box surfaces independently, including both body wrenches."""
    poses, com, rotations, jt, dr = frames(model, q)
    starts = frames(model, qn)[0]
    bodies = model.shape_body.numpy()
    scales = model.shape_scale.numpy().astype(float)
    margins = model.shape_margin.numpy().astype(float)
    energy, residual, matrix = np.zeros(2), np.zeros(len(q)), np.zeros((len(q), len(q)))
    forces = np.zeros((model.body_count, 6))
    for source, target in ((0, 1),) if fixed_target else ((0, 1), (1, 0)):
        points, weights = box_face_samples(scales[source])
        rs, rt = poses[source][:3, :3], poses[target][:3, :3]
        bs, bt = bodies[source], bodies[target]
        for point, weight in zip(points, weights, strict=True):
            world = rs @ point + poses[source][:3, 3]
            y = rt.T @ (world - poses[target][:3, 3])
            worldn = starts[source][:3, :3] @ point + starts[source][:3, 3]
            yn = starts[target][:3, :3].T @ (worldn - starts[target][:3, 3])
            normal = np.zeros(3)
            axis = np.argmax(np.abs(point) / scales[source])
            normal[axis] = np.sign(point[axis])
            if (rs @ normal) @ (rt @ geometry(y, "box", scales[target])[1]) > 0:
                continue
            jacobians = [jt[b].T + np.einsum("nij,j->in", dr[b], rotations[b].T @ (world - com[b])) for b in (bs, bt)]
            relative = rt.T @ (jacobians[0] - jacobians[1])
            e, r, _, fitted = local(
                y,
                yn,
                "box",
                scales[target],
                offset=margins.sum(),
                stiffness=1e7 * weight * (1 if fixed_target else 0.5),
            )
            energy += e
            residual += relative.T @ r
            matrix += relative.T @ fitted @ relative
            f = -rt @ r
            for b, force in ((bs, f), (bt, -f)):
                forces[b, :3] += force
                forces[b, 3:] += np.cross(world - com[b], force)
    return energy, residual, matrix, forces


def set_q(run, state, q):
    """Assign native authoritative coordinates without a float32 state-mirror round trip."""
    state.joint_q.assign(q)


def test_two_tree_contact(test, device):
    """Match E/r/PSD B and equal-opposite wrenches for moving, fixed and reordered endpoints."""
    for reverse, fixed, precision in itertools.product((False, True), (False, True), ("fp64", "fp32")):
        with test.subTest(reverse=reverse, fixed=fixed, precision=precision):
            run = trees_solver(contact_trees(device, fixed_target=fixed), reverse=reverse, precision=precision)
            state = run.create_state(dt=0.001)
            batch = native_batch(run)
            qn = state.joint_q.numpy().astype(float)
            query(batch, run, state, start=True)
            set_q(run, state, qn + np.array([1e-5, -2e-6, -2e-5]))
            q = state.joint_q.numpy().astype(float)
            query(batch, run, state)
            energy, residual, matrix, forces = contact_terms(run.model, q, qn, fixed_target=fixed)
            rows = np.asarray(run._actors.maps["joint_dof_to_global"])
            actual = batch_matrix(batch)[np.ix_(rows, rows)]
            # Micrometre slip is formed from centimetre world positions in FP32.
            tol = 1e-3 if precision == "fp32" else 1e-8
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            test.assertEqual(run._contact.actor_samples.count, 36 if fixed else 72)
            test.assertEqual(run._contact.actor_samples.support_size, 36 * 3 * 2 if fixed else 72 * 3 * 3)
            test.assertEqual(run._contact.actor_samples.triplet_capacity, 36 * 3 * 2**2 if fixed else 72 * 3 * 3**2)
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], energy, rtol=tol, atol=tol * 1e-4)
            np.testing.assert_allclose(batch.pool.residual.result.numpy()[rows], residual, rtol=tol, atol=tol)
            np.testing.assert_allclose(actual, matrix, rtol=max(tol, 1e-5), atol=max(tol, 1e-5))
            scale = max(1, np.linalg.norm(matrix))
            np.testing.assert_allclose(actual, actual.T, rtol=1e-6, atol=1e-6 * scale)
            test.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -1e-6 * scale)
            actual_forces = batch.pool.force.result.numpy()[: 6 * run.model.body_count].reshape(-1, 6)
            np.testing.assert_allclose(actual_forces, forces, rtol=tol, atol=tol)
            com = frames(run.model, q)[1]
            np.testing.assert_allclose(actual_forces[:, :3].sum(axis=0), 0, atol=tol)
            moments = actual_forces[:, 3:] + np.cross(com, actual_forces[:, :3])
            np.testing.assert_allclose(moments.sum(axis=0), 0, atol=tol)
            if fixed:
                np.testing.assert_array_equal(actual[-1], 0)
            else:
                test.assertGreater(np.linalg.norm(matrix[:2, 2]), 1e-3)
                test.assertGreater(abs(residual[2]), 1e-4)
                test.assertGreater(np.linalg.norm(matrix - 2 * actual) / scale, 0.9)


def test_relative_derivatives(test, device):
    """Compare residual differences to an independent C-IE oracle, separately from its PSD tangent."""
    run = trees_solver(contact_trees(device))
    state = run.create_state(dt=0.001)
    batch = native_batch(run)
    qn = state.joint_q.numpy().astype(float)
    query(batch, run, state, start=True)
    q = qn + np.array([1e-5, -2e-6, -2e-5])
    native, reference = [], []
    for direction in np.eye(3):
        values, expected = [], []
        for sign in (-1, 1):
            set_q(run, state, q + sign * 1e-7 * direction)
            actual_q = state.joint_q.numpy().astype(float)
            query(batch, run, state)
            values.append(batch.pool.residual.result.numpy().copy())
            expected.append(contact_terms(run.model, actual_q, qn)[1])
        native.append((values[1] - values[0]) / 2e-7)
        reference.append((expected[1] - expected[0]) / 2e-7)
    np.testing.assert_allclose(native, reference, rtol=1e-6, atol=1e-4)


def test_tree_pair_filtering(test, device):
    """Honor explicit filters and exclude same-owner and zero-support pairs independently."""
    run = trees_solver(contact_trees(device, filtered=True))
    test.assertEqual(run._contact.actor_samples.count, 0)
    run = trees_solver(contact_trees(device, filtered=True, root_shapes=True))
    owners = run._actors.maps["shape_owner"]
    pairs = {tuple(sorted(pair)) for pair in run.model.shape_contact_pairs.numpy()}
    test.assertIn((0, 2), pairs)
    test.assertIn((1, 3), pairs)
    test.assertIn((2, 3), pairs)
    samples = run._contact.actor_samples
    test.assertEqual(samples.count, 72)
    for source, target, _, _ in samples.keys:
        test.assertNotEqual(owners[source], owners[target])
        test.assertNotEqual({source, target}, {0, 1})
        test.assertLess(source, 2)
        test.assertGreaterEqual(target, 2)


def test_two_tree_spatial_sweep(test, device):
    """Enclose both trees' motion and preserve full-contact results inside frozen candidates."""
    run = trees_solver(contact_trees(device))
    state = run.create_state(dt=0.001)
    batch = native_batch(run)
    inputs, options = native_inputs(run, state)
    batch.begin_step(*inputs, **options)
    k = run._kinematics
    spatial = _ContactSpatial(batch, k, k.gather_q(state), inputs[0], inputs[1], **options, capacity=run._contact.count)
    q0 = state.joint_q.numpy().astype(float)
    q1 = q0 + np.array([0.3, -0.01, -0.4])
    set_q(run, state, q1)
    inputs, options = native_inputs(run, state)
    spatial.update(k.gather_q(state), inputs[0], rigid_com=options["rigid_com"])
    spatial.check()
    lower, upper = spatial.face_lo.numpy(), spatial.face_hi.numpy()
    for alpha in np.linspace(0, 1, 7):
        set_q(run, state, (1 - alpha) * q0 + alpha * q1)
        poses = frames(run.model, state.joint_q.numpy().astype(float))[0]
        for shape in range(2):
            points, _ = box_face_samples(run.model.shape_scale.numpy()[shape].astype(float))
            for sample, p in enumerate(points):
                world = poses[shape][:3, :3] @ p + poses[shape][:3, 3]
                index = spatial.face_index[shape, sample // 3]
                test.assertTrue(np.all(world >= lower[index]) and np.all(world <= upper[index]))
        inputs, options = native_inputs(run, state)
        batch.evaluate(*inputs, dt=0.001, **options)
        expected = batch.pool.residual.result.numpy().copy()
        batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
        np.testing.assert_allclose(batch.pool.residual.result.numpy(), expected, rtol=1e-10, atol=1e-10)


class TestArticulationContact(unittest.TestCase):
    pass


def test_coupled_contact_step(test, device):
    """Publish both trees' contact forces and solve the nonzero cross block in production steps."""
    run = trees_solver(contact_trees(device), pcg_mode="production", use_cuda_graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    initial = state.joint_q.numpy().copy()
    for step in range(3):
        qn = state.joint_q.numpy().copy()
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertFalse(run.last_stats.rolled_back)
        _, residual, matrix, forces = contact_terms(run.model, state.joint_q.numpy(), qn)
        history = state._monolithic_coupled
        np.testing.assert_allclose(history.contact_joint_force.numpy(), -residual, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(history.contact_body_force.numpy(), forces, rtol=1e-8, atol=1e-9)
        test.assertEqual(int(history.force_generation.numpy()[0]), step + 1)
        test.assertEqual(len(run.last_stats.variational["articulation_blocks"]), 2)
        actual = run._linear.densify_for_test(generation=run._generation).raw_matrix
        np.testing.assert_allclose(actual[:2, 2], matrix[:2, 2], rtol=2e-5, atol=2e-5)
        test.assertGreater(np.linalg.norm(actual[:2, 2]), 1e-3)
    test.assertGreater(np.linalg.norm(state.joint_q.numpy()[:2] - initial[:2]), 1e-7)
    test.assertGreater(abs(state.joint_q.numpy()[2] - initial[2]), 1e-7)


for _test in (
    test_two_tree_contact,
    test_relative_derivatives,
    test_tree_pair_filtering,
    test_two_tree_spatial_sweep,
    test_coupled_contact_step,
):
    add_function_test(TestArticulationContact, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
