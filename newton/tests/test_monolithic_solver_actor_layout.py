# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise actor permutations through actual coupled assembly and publication."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton._src.solvers.monolithic.contact_pipeline import _SpatialContactObjective
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearStatus,
    MonolithicPcgWarmStart,
)
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, independent_objective, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def permuted_solver(model, order, **options):
    table = ActorTable.from_model(model, actor_order=order)
    with patch.object(ActorTable, "from_model", return_value=table):
        return solver(model, **options)


def physical_order(layout):
    return np.r_[layout.indices("articulation"), layout.indices("tet")]


def test_dense_layout(test, device):
    """Match independent E/r/B, block apply, scales and PCG after real permutation."""
    for order in ((0, 1), (1, 0)):
        model = scene(device)
        run = permuted_solver(model, order)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        expected_e, expected_r, expected_b = independent_objective(
            run, state.joint_q.numpy(), state.particle_q.numpy(), 0.001
        )
        energy, residual, _ = run._evaluate(state, 0.001)
        run._assemble(0)
        linear, layout = run._linear, run._linear.layout
        indices = physical_order(layout)
        oracle = linear.densify_for_test(generation=run._generation)
        test.assertLessEqual(abs(energy - expected_e), 5.820766091346741e-11 / 8)
        np.testing.assert_allclose(residual[indices], expected_r, rtol=0, atol=2.0935893019213836e-9 / 8)
        np.testing.assert_allclose(oracle.raw_matrix[np.ix_(indices, indices)], expected_b, rtol=2e-5, atol=2e-3)
        scale = linear.scale.numpy()
        np.testing.assert_allclose(
            scale,
            1
            / np.sqrt(
                np.maximum(oracle.raw_matrix.diagonal(), run._config.epsilon_d * linear.dynamic_diagonal.numpy())
            ),
            rtol=2e-7,
        )
        test.assertEqual(
            linear.factor_actor_preconditioner(generation=run._generation, pivot_tolerance=0),
            MonolithicLinearStatus.SUCCESS,
        )
        block = np.zeros_like(oracle.scaled_matrix)
        arm = layout.indices("articulation")
        block[np.ix_(arm, arm)] = oracle.scaled_matrix[np.ix_(arm, arm)]
        for rows in layout.indices("tet").reshape(-1, 3):
            block[np.ix_(rows, rows)] = oracle.scaled_matrix[np.ix_(rows, rows)]
        rhs = np.linspace(-0.1, 0.2, layout.scalar_dof_count).astype(np.float32)
        run._rhs.assign(rhs)
        linear.preconditioner.matvec(run._rhs, run._y, run._y, 1.0, 0.0)
        np.testing.assert_allclose(run._y.numpy(), np.linalg.solve(block, rhs), rtol=2e-6, atol=2e-7)
        result = linear.solve_pcg(
            run._rhs,
            run._y,
            generation=run._generation,
            warm_start=MonolithicPcgWarmStart.ZERO,
            config=_pcg_config(maximum_iterations=100),
        )
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        np.testing.assert_allclose(run._y.numpy(), np.linalg.solve(oracle.scaled_matrix, rhs), rtol=2e-4, atol=2e-5)
        # Removing the cross block produces an observably different operator.
        missing = oracle.raw_matrix.copy()
        tet = layout.indices("tet")
        missing[np.ix_(arm, tet)] = 0
        missing[np.ix_(tet, arm)] = 0
        test.assertGreater(np.linalg.norm((missing - oracle.raw_matrix) @ np.ones(len(rhs))), 1)
        old = linear.layout
        linear.layout = replace(old, identity="changed")
        for operator in (linear.operator, linear.preconditioner):
            with test.assertRaises(ValueError):
                operator.matvec(run._rhs, run._y, run._y, 1.0, 0.0)
        test.assertEqual(
            linear.recover_delta(run._y, run._delta, generation=run._generation),
            MonolithicLinearStatus.STALE_GENERATION,
        )
        with test.assertRaises(ValueError):
            linear.begin_assembly(replace(run._generation, assembly_sequence=run._generation.assembly_sequence + 1))
        linear.layout = old


def test_layout_integration(test, device):
    """Converge three active-contact steps for every execution route and layout."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for mode, graph in modes:
        reference = None
        for order in ((0, 1), (1, 0)):
            with test.subTest(mode=mode, graph=graph, order=order):
                model = scene(device)
                run = permuted_solver(
                    model,
                    order,
                    mode=mode,
                    graph=graph,
                )
                state = run.create_state(dt=0.001)
                for _ in range(3):
                    run.step(state, state, model.control(), dt=0.001)
                    test.assertTrue(run.last_stats.converged, run.last_stats)
                    matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
                    a, t = run._layout.indices("articulation"), run._layout.indices("tet")
                    test.assertGreater(np.linalg.norm(matrix[np.ix_(a, t)]), 0)
                    history = state._monolithic_coupled
                    test.assertGreater(np.linalg.norm(history.contact_body_force.numpy()), 0)
                    test.assertGreater(np.linalg.norm(history.contact_particle_force.numpy()), 0)
                    np.testing.assert_allclose(
                        history.contact_body_force.numpy()[:, :3].sum(axis=0)
                        + history.contact_particle_force.numpy().sum(axis=0),
                        0,
                        atol=1e-12,
                    )
                current = snapshot(state)
                if reference is None:
                    reference = current
                else:
                    for name in ("joint_q", "particle_q"):
                        np.testing.assert_allclose(current[name], reference[name], rtol=0, atol=1e-7)
                    for name in ("joint_qd", "particle_qd"):
                        np.testing.assert_allclose(current[name], reference[name], rtol=0, atol=1e-6)
                    for name in (
                        "history.contact_body_force",
                        "history.contact_particle_force",
                        "history.contact_joint_force",
                    ):
                        np.testing.assert_allclose(current[name], reference[name], rtol=1e-5, atol=1e-4)


def test_sparse_support(test, device):
    """Exclude sibling joints, fixed vertices and static target rows from CSR support."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    inertia = wp.diag(wp.vec3(0.01))
    bodies = [builder.add_link(mass=1, inertia=inertia) for _ in range(3)]
    joints = [
        builder.add_joint_revolute(-1, bodies[0]),
        builder.add_joint_prismatic(bodies[0], bodies[1]),
        builder.add_joint_revolute(bodies[0], bodies[2]),
    ]
    builder.add_articulation(joints)
    builder.add_shape_plane(body=bodies[1], width=0, length=0, cfg=builder.ShapeConfig(density=0))
    builder.add_ground_plane()
    add_contact_free_tet(builder)
    builder.particle_mass[0] = 0
    model = builder.finalize(device=device)
    for order in ((0, 1), (1, 0)):
        run = permuted_solver(model, order)
        test.assertEqual(run._layout.dynamic_particle_count, 3)
        batch = _ContactBatch(run._contact, batch_capacity=run._contact.count, capacity=None)
        offsets, columns = batch.support.offsets.numpy(), batch.support.columns.numpy()
        a = run._layout.indices("articulation")
        # Each body carries only its ancestor joints; the static ground has no rows.
        for body, expected in zip(bodies, (a[[0]], a[[0, 1]], a[[0, 2]]), strict=True):
            test.assertEqual(set(columns[offsets[body] : offsets[body + 1]]), set(expected))
        particles = batch.support.particles.numpy()
        np.testing.assert_array_equal(particles, run._actors.maps["particle_to_global"])
        test.assertEqual(particles[0], -1)
        # Plane samples: two ancestor q rows and three dynamic vertices of one face.
        test.assertEqual(batch.pool.max_support, 11)

    # Own terms and contact terms must fit together, before allocating CSR storage.
    def overflow(*args, **kwargs):
        kwargs["base_triplet_count"] = 2**31 - 1
        return _SpatialContactObjective(*args, **kwargs)

    with patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", side_effect=overflow):
        with test.assertRaisesRegex(ValueError, "Contact support/triplet capacity"):
            solver(model)


def test_layout_checkpoint(test, device):
    """Restore schema four and reject wrong layout/owner or failed loads atomically."""
    model = scene(device)
    canonical = permuted_solver(model, (0, 1))
    other = permuted_solver(model, (1, 0))
    state = canonical.create_state(dt=0.001)
    canonical.step(state, state, model.control(), dt=0.001)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        canonical.save_checkpoint(state, path)
        restored = canonical.create_state(dt=0.001)
        canonical.load_checkpoint(restored, path)
        assert_snapshot(test, restored, snapshot(state))
        target = other.create_state(dt=0.001)
        before = snapshot(target)
        with test.assertRaises(ValueError):
            other.load_checkpoint(target, path)
        assert_snapshot(test, target, before)
        original = snapshot(state)
        for field in (
            "history.actor_layout",
            "history.layout_digest",
            "history.layout_version",
            "history.rigid_com",
            "history.config_digest",
        ):
            broken = {name: value.copy() for name, value in original.items()}
            if field == "history.rigid_com":
                broken[field] = np.zeros((1, 3))
            else:
                broken[field].flat[0] ^= 1
            np.savez(path, **broken)
            with test.assertRaises(ValueError):
                canonical.load_checkpoint(state, path)
            assert_snapshot(test, state, original)
        del broken["history.actor_layout"]
        np.savez(path, **broken)
        with test.assertRaises(ValueError):
            canonical.load_checkpoint(state, path)
        assert_snapshot(test, state, original)


class TestActorSolverLayout(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _name, _test in (
        ("dense", test_dense_layout),
        ("integration", test_layout_integration),
        ("support", test_sparse_support),
        ("checkpoint", test_layout_checkpoint),
    ):
        add_function_test(TestActorSolverLayout, "test_" + _name, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
