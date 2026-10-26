# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check native actor contact against independent surface and virtual-work formulas."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import GeoType, Mesh
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton._src.solvers.monolithic.contact_sampling import _checked_contact_capacity, _surface_samples
from newton._src.solvers.monolithic.linear_types import MonolithicPcgWarmStart
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contact_oracle import box_face_samples, fixture_terms
from scripts.monolithic_reference.actor_contract_oracle import sphere_samples


def physical(state):
    h = state._monolithic_coupled
    return {
        "q": state.joint_q.numpy().copy(),
        "x": state.particle_q.numpy().copy(),
        "com": h.rigid_com.numpy().copy(),
        "rotation": h.rigid_rotation.numpy().copy(),
    }


def native_inputs(run, state):
    k, h = run._kinematics, state._monolithic_coupled
    k.evaluate(k.gather_q(state))
    return (state.particle_q, k.poses, k.com, k.jt, k.jw), {
        "rigid_com": h.rigid_com,
        "rigid_rotation": h.rigid_rotation,
    }


def native_batch(run):
    """Evaluate the solver's compact contact metadata with every candidate in the production producer."""
    return _ContactBatch(run._contact, batch_capacity=run._contact.count, capacity=None)


def query(batch, run, state, *, start=False):
    inputs, options = native_inputs(run, state)
    if start:
        batch.begin_step(*inputs, **options)
    else:
        batch.evaluate(*inputs, dt=0.001, **options)


def perturb(run, state, values, direction, h):
    q, x, com, quat = (values[n].copy() for n in ("q", "x", "com", "rotation"))
    arm = run._articulation_actor
    q[list(arm.coord_indices)] += h * direction[run._layout.indices("articulation")]
    for particle, row in enumerate(run._actors.maps["particle_to_global"]):
        if row >= 0:
            x[particle] += h * direction[row : row + 3]
    for i, actor in enumerate(run._rigid.actors):
        d = direction[actor.dof_start : actor.dof_start + 6]
        com[i] += h * d[:3]
        w = h * d[3:]
        angle = np.linalg.norm(w)
        dq = np.r_[w * (np.sin(angle / 2) / angle if angle else 0.5), np.cos(angle / 2)]
        a, b = dq[:3], quat[i, :3]
        quat[i] = np.r_[dq[3] * b + quat[i, 3] * a + np.cross(a, b), dq[3] * quat[i, 3] - a @ b]
    state.joint_q.assign(q)
    state.particle_q.assign(x)
    state._monolithic_coupled.rigid_com.assign(com)
    state._monolithic_coupled.rigid_rotation.assign(quat)


def test_surface_area(test, device):
    """Freeze box face keys, spherical projection and original mesh area without law kernels."""
    half = np.array([0.02, 0.03, 0.04])
    p, normals, weights = _surface_samples(GeoType.BOX, half)
    expected, area = box_face_samples(half)
    np.testing.assert_allclose(p, expected, atol=1e-17)
    np.testing.assert_allclose(weights, area, atol=1e-17)
    test.assertAlmostEqual(weights.sum(), 8 * (half[0] * half[1] + half[0] * half[2] + half[1] * half[2]))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    p, normals, weights = _surface_samples(GeoType.SPHERE, np.array([0.02, 0, 0]))
    expected, area = sphere_samples(0.02)
    test.assertEqual(len(p), 5952)
    np.testing.assert_allclose(p, expected, atol=1e-17)
    np.testing.assert_allclose(weights, area, atol=1e-17)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 0.02, atol=1e-17)
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        indices=np.array([0, 1, 2]),
        compute_inertia=False,
    )
    _, _, weights = _surface_samples(GeoType.MESH, np.array([2, 3, 4]), mesh)
    np.testing.assert_allclose(weights, 1)
    mesh.indices[:] = 0
    with test.assertRaisesRegex(ValueError, "positive finite area"):
        _surface_samples(GeoType.MESH, np.ones(3), mesh)


def test_dense_contact(test, device):
    """Match all four paths' E/r/PSD B and full cross blocks against independent virtual work."""
    for kind, permuted in itertools.product(("A-R", "R-R", "R-T", "R-static"), (False, True)):
        with test.subTest(kind=kind, permuted=permuted):
            run = contact_solver(contact_scene(device, kind), permuted=permuted)
            batch = native_batch(run)
            state = contact_state(run, kind)
            start = physical(state)
            query(batch, run, state, start=True)
            direction = np.linspace(-0.2, 0.3, run._layout.scalar_dof_count)
            perturb(run, state, start, direction, 1e-6)
            query(batch, run, state)
            energy, residual, b = fixture_terms(run.model, run._actors, physical(state), start)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], energy, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(batch.pool.residual.result.numpy(), residual, rtol=1e-8, atol=1e-10)
            matrix = batch_matrix(batch)
            scale = max(1, np.linalg.norm(b))
            test.assertLess(np.linalg.norm(matrix - b) / scale, 1e-5)
            test.assertLess(np.linalg.norm(matrix - matrix.T) / scale, 1e-5)
            test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * scale)
            # Each negative example must violate this independent dense comparison.
            missing = matrix.copy()
            rigid = run._rigid.actors[0]
            rows = np.arange(rigid.dof_start, rigid.dof_start + 6)
            if kind != "R-static":
                other = np.setdiff1d(np.arange(len(direction)), rows)
                missing[np.ix_(rows, other)] = 0
                test.assertGreater(np.linalg.norm(missing - b) / scale, 1e-5)
            copied = matrix * 2
            test.assertGreater(np.linalg.norm(copied - b) / scale, 1e-5)
            missing_rotation = matrix.copy()
            missing_rotation[rows[3:], :] = 0
            missing_rotation[:, rows[3:]] = 0
            test.assertGreater(np.linalg.norm(missing_rotation - b) / scale, 1e-5)


def test_sampling_identity_and_capacity(test, device):
    """Preserve rebuilt pair identity and reject int32 capacity overflow."""
    run = contact_solver(contact_scene(device, "R-R"))
    samples = run._contact.actor_samples
    test.assertEqual(samples.count, 72)
    test.assertEqual(samples.support_size, 2592)
    test.assertEqual(samples.triplet_capacity, 31104)
    area = 2 * sum(np.prod(np.delete(run.model.shape_scale.numpy()[0] * 2, i)) for i in range(3))
    for source in set(samples.source_shapes):
        test.assertAlmostEqual(samples.weights[samples.source_shapes == source].sum(), area / 2, places=9)
    for args in ((2**31 // 3 + 1, 0, 0), (0, 2**31, 0), (0, 0, 2**31), (0, 0, 1, 2**31 - 1)):
        with test.assertRaisesRegex(ValueError, "int32"):
            _checked_contact_capacity(*args)
    state = contact_state(run, "R-R")
    before = snapshot(state)
    for name in ("shape_contact_pairs", "shape_flags", "shape_world", "shape_collision_group"):
        with patch.object(run.model, name, wp.clone(getattr(run.model, name))):
            # Static Model storage changes require reconstruction under the P6a contract.
            rebuilt = contact_solver(run.model)
            test.assertEqual(rebuilt._kinematics.digest, run._kinematics.digest)
            test.assertEqual(rebuilt._contact.actor_samples.keys, samples.keys)
            assert_snapshot(test, state, before)


def test_real_trial_activation(test, device):
    """Keep a distant legal pair's stable slots and activate it in a later trial."""
    run = contact_solver(contact_scene(device, "A-R"))
    batch = native_batch(run)
    test.assertEqual(run._contact.actor_samples.count, 72)
    state = contact_state(run, "A-R")
    current = physical(state)
    state._monolithic_coupled.rigid_com.assign(current["com"] + np.array([0.1, 0, 0]))
    query(batch, run, state, start=True)
    query(batch, run, state)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    test.assertEqual(int(batch.pool.records.count.numpy()[0]), 0)
    frozen = batch.start.y.numpy().copy()
    state._monolithic_coupled.rigid_com.assign(current["com"])
    query(batch, run, state)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    test.assertGreater(int(batch.pool.records.count.numpy()[0]), 0)
    test.assertGreater(np.linalg.norm(batch.pool.residual.result.numpy()), 1e-4)
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)


def test_branch_and_fixed_support(test, device):
    """Use only target ancestors and treat fixed roots as static, including same-owner exclusions."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    mass = {"mass": 1, "inertia": wp.diag(wp.vec3(0.01))}
    root, branch, sibling, child = [builder.add_link(**mass) for _ in range(4)]
    options = {"limit_ke": 0, "limit_kd": 0, "armature": 0, "damping": 0, "friction": 0}
    joints = [
        builder.add_joint_fixed(-1, root),
        builder.add_joint_prismatic(root, branch, axis=newton.Axis.X, **options),
        builder.add_joint_prismatic(root, sibling, axis=newton.Axis.Y, **options),
        builder.add_joint_fixed(branch, child),
    ]
    builder.add_articulation(joints)
    rigid = builder.add_body(**mass)
    cfg = builder.ShapeConfig(density=0, has_particle_collision=False)
    shapes = [
        builder.add_shape_box(b, hx=0.01, hy=0.02, hz=0.03, cfg=cfg) for b in (root, child, sibling, rigid, rigid)
    ]
    add_contact_free_tet(builder)
    model = builder.finalize(device=device)
    run = contact_solver(model)
    samples = run._contact.actor_samples
    arm = next(a for a in ActorTable.from_model(model).actors if a.kind == "articulation")
    # Compact metadata stores no per-key support; the producer expands each endpoint body's support.
    support = native_batch(run).support
    offsets, rows, bodies = support.offsets.numpy(), support.columns.numpy(), model.shape_body.numpy()

    def body_columns(shape):
        body = bodies[shape]
        return set(rows[offsets[body] : offsets[body + 1]]) if body >= 0 else set()

    for source, target, _face, _slot in samples.keys:
        test.assertNotEqual(bodies[source], bodies[target])
        endpoint = source if source < 3 else target
        columns = body_columns(source) | body_columns(target)
        if endpoint == shapes[0]:
            test.assertFalse(columns & set(range(arm.dof_start, arm.dof_start + arm.dof_count)))
            test.assertIn(source, shapes[3:])
        elif endpoint == shapes[1]:
            test.assertIn(arm.dof_start, columns)
            test.assertNotIn(arm.dof_start + 1, columns)
        elif endpoint == shapes[2]:
            test.assertNotIn(arm.dof_start, columns)
            test.assertIn(arm.dof_start + 1, columns)
    for target in shapes[:3]:
        indices = samples.targets == target
        area = samples.weights[indices].sum()
        half = model.shape_scale.numpy()[target].astype(float)
        expected = (
            2 * 8 * (half[0] * half[1] + half[0] * half[2] + half[1] * half[2]) * (1 if target == shapes[0] else 0.5)
        )
        test.assertAlmostEqual(area, expected, places=12)


def test_contact_linear_blocks(test, device):
    """Include real contact in actor preconditioning, scaled matvec and PCG dense solution."""
    for mode in ("diagnostic", "production"):
        run = contact_solver(contact_scene(device), mode=mode, graph=bool(device.is_cuda and mode == "production"))
        state = contact_state(run, "chain")
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        linear, generation = run._linear, run._generation
        oracle = linear.densify_for_test(generation=generation)
        matrix = oracle.scaled_matrix
        test.assertEqual(int(linear.factor_actor_preconditioner(generation=generation, pivot_tolerance=0)), 0)
        blocks = np.zeros_like(matrix)
        for rows in (
            run._layout.indices("articulation"),
            *linear.rigid_rows.numpy(),
            *run._layout.indices("tet").reshape(-1, 3),
        ):
            blocks[np.ix_(rows, rows)] = matrix[np.ix_(rows, rows)]
        rhs_host = np.linspace(-0.2, 0.3, len(matrix)).astype(run._numpy_dtype)
        rhs = wp.array(rhs_host, dtype=linear.dtype, device=device)
        out = wp.zeros(len(matrix), dtype=linear.dtype, device=device)
        linear.preconditioner.matvec(rhs, out, out, 1.0, 0.0)
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(blocks, rhs_host), atol=1e-6, rtol=1e-5)
        linear.operator.matvec(rhs, out, out, 1.0, 0.0)
        np.testing.assert_allclose(out.numpy(), matrix @ rhs_host, atol=1e-6, rtol=1e-5)
        result = linear.solve_pcg(
            rhs,
            out,
            generation=generation,
            warm_start=MonolithicPcgWarmStart.ZERO,
            config=_pcg_config(maximum_iterations=200),
        )
        test.assertEqual(int(result.status), 0, result)
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(matrix, rhs_host), atol=1e-5, rtol=1e-5)


def test_local_transform_and_law_branches(test, device):
    """Keep native off-center geometry and the C-IE law correct at zero friction and zero smoothing."""
    for mu, width in itertools.product((0, 0.5), (0, 1e-4)):
        with test.subTest(mu=mu, width=width):
            model = contact_scene(device, "R-static")
            transforms = model.shape_transform.numpy().copy()
            transforms[1] = np.asarray(
                wp.transform((0, 0.001, 0.002), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.005))
            )
            model.shape_transform.assign(transforms)
            run = contact_solver(model, friction_coefficient=mu, normal_smoothing=width)
            batch = native_batch(run)
            state = contact_state(run, "R-static")
            start = physical(state)
            query(batch, run, state, start=True)
            direction = np.linspace(-0.2, 0.3, run._layout.scalar_dof_count)
            perturb(run, state, start, direction, 1e-6)
            query(batch, run, state)
            energy, residual, b = fixture_terms(model, run._actors, physical(state), start, mu=mu, width=width)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], energy, atol=1e-12, rtol=1e-8)
            np.testing.assert_allclose(batch.pool.residual.result.numpy(), residual, atol=1e-10, rtol=1e-8)
            matrix = batch_matrix(batch)
            test.assertLess(np.linalg.norm(matrix - b) / max(1, np.linalg.norm(b)), 1e-5)
            saved = batch.pool.residual.result.numpy().copy()
            state.body_q.fill_(100)
            query(batch, run, state)
            np.testing.assert_array_equal(batch.pool.residual.result.numpy(), saved)


class TestActorContactPairs(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _function in (
        test_surface_area,
        test_dense_contact,
        test_sampling_identity_and_capacity,
        test_real_trial_activation,
        test_branch_and_fixed_support,
        test_contact_linear_blocks,
        test_local_transform_and_law_branches,
    ):
        add_function_test(TestActorContactPairs, _function.__name__, _function, devices=[_device])


if __name__ == "__main__":
    unittest.main()
