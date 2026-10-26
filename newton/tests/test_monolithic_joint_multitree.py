# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent multi-articulation kinematics and coupled dynamics checks."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import coupled_kernels
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.articulation import ArticulationObjective, _TreeKinematics
from newton._src.solvers.monolithic.linear_types import MonolithicLinearLayout
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import Kinematics, skew


def trees_model(device, *, sizes=(1, 3), coupled=False):
    """Build independent fixed roots with unequal, noncollinear moving chains."""
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    for tree, count in enumerate(sizes):
        root = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.02)))
        joints = [builder.add_joint_fixed(-1, root, parent_xform=wp.transform((tree * 0.5, 0, 0), wp.quat_identity()))]
        parent = root
        for i in range(count):
            body = builder.add_link(mass=0.2, com=(0.03, 0.02, -0.04), inertia=wp.diag(wp.vec3(0.002)))
            method = builder.add_joint_prismatic if i == count - 1 else builder.add_joint_revolute
            joints.append(
                method(
                    parent,
                    body,
                    axis=(1, 1, i % 2),
                    parent_xform=wp.transform((0.02, 0, -0.08), wp.quat_identity()),
                    limit_ke=0,
                    limit_kd=0,
                )
            )
            parent = body
        builder.add_articulation(joints)
    builder.joint_q[:] = np.linspace(0.02, 0.09, sum(sizes)).tolist()
    builder.joint_qd[:] = np.linspace(-0.03, 0.04, sum(sizes)).tolist()
    if coupled:
        add_contact_free_tet(builder)
    return builder.finalize(device=device)


def test_multitree_kinematics(test, device):
    model = trees_model(device)
    table = ActorTable.from_model(model)
    arms = tuple(a for a in table.actors if a.kind == "articulation")
    layout = MonolithicLinearLayout.from_actors(table)
    test.assertEqual(layout.q_dof_count, 4)
    for arm in arms:
        np.testing.assert_array_equal(
            layout.indices("articulation", arm.source_index), np.arange(arm.dof_start, arm.dof_start + arm.dof_count)
        )
    k = _TreeKinematics(model, actors=arms)
    q = model.joint_q.numpy().astype(np.float64)
    k.evaluate(wp.array(q, dtype=wp.float64, device=device))
    com, rotation, jt, jr, _, _ = Kinematics(model).evaluate(q)
    np.testing.assert_allclose(k.com.numpy(), com, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(k.rotations.numpy(), rotation, atol=1e-14, rtol=1e-13)
    packed_t, packed_w = k.jt.numpy().reshape(-1, 3), k.jw.numpy().reshape(-1, 3)
    cursor = 0
    for arm in arms:
        for body in arm.body_indices:
            for dof in arm.dof_indices:
                np.testing.assert_allclose(packed_t[cursor], jt[body, dof], atol=1e-14, rtol=1e-13)
                np.testing.assert_allclose(
                    skew(packed_w[cursor]) @ rotation[body], jr[body, dof], atol=1e-14, rtol=1e-13
                )
                cursor += 1
    test.assertEqual(k.jt.size, sum(len(a.body_indices) * a.dof_count for a in arms))


class TestMultiTree(unittest.TestCase):
    pass


def test_multitree_objective(test, device):
    model = trees_model(device)
    arms = tuple(a for a in ActorTable.from_model(model).actors if a.kind == "articulation")
    k = _TreeKinematics(model, actors=arms)
    q = wp.array(model.joint_q.numpy(), dtype=wp.float64, device=device)
    obj = ArticulationObjective(k, np.zeros((k.count, 10)))
    parameters = obj.params.numpy()
    parameters[:, 7] = parameters[:, 9] = 1
    obj.params.assign(parameters)
    obj.begin(q)
    wp.copy(obj.predicted_r, obj.rn)
    h = 0.002
    obj.evaluate(q, h)
    oracle = Kinematics(model)
    mass, _ = oracle.mass_and_bias(q.numpy(), np.zeros(k.count))
    matrix = np.zeros_like(mass)
    cursor = 0
    packed = obj.tangent.numpy().ravel()
    for arm in arms:
        n = arm.dof_count
        matrix[np.ix_(arm.dof_indices, arm.dof_indices)] = packed[cursor : cursor + n * n].reshape(n, n)
        cursor += n * n
    np.testing.assert_allclose(matrix, mass / h**2, atol=1e-8, rtol=1e-12)
    np.testing.assert_allclose(obj.residual.numpy(), 0, atol=1e-12)
    force = np.arange(k.bodies * 3).reshape(k.bodies, 3) * 0.001
    obj.force.assign(force)
    obj.evaluate(q, h, build_tangent=False)
    expected = -np.einsum("bni,bi->n", oracle.evaluate(q.numpy())[2], force)
    np.testing.assert_allclose(obj.residual.numpy(), expected, atol=1e-14, rtol=1e-12)
    test.assertEqual(obj.tangent.size, sum(a.dof_count**2 for a in arms))
    diagonal = wp.zeros(k.count, dtype=wp.float64, device=device)
    wp.launch(
        coupled_kernels.dynamic_diagonal,
        k.count,
        [
            k.jt,
            k.jw,
            k.rotations,
            k.mass,
            k.second,
            obj.armature,
            wp.zeros(0, dtype=int, device=device),
            wp.zeros(0, dtype=float, device=device),
            wp.float64(h),
            wp.array(np.arange(k.count), dtype=int, device=device),
            k.jac_offsets,
            k.body_dof_starts,
            k.dof_body_starts,
            k.dof_body_ends,
            k.count,
            diagonal,
        ],
        device=device,
    )
    np.testing.assert_allclose(diagonal.numpy(), mass.diagonal() / h**2, rtol=1e-12, atol=1e-9)


def test_reordered_tree_view(test, device):
    model = trees_model(device)
    arms = tuple(reversed(ActorTable.from_model(model).actors))
    k = _TreeKinematics(model, actors=arms)
    state = model.state()
    ids = [d for a in arms for d in a.dof_indices]
    np.testing.assert_array_equal(k.gather_q(state).numpy(), state.joint_q.numpy()[ids])
    np.testing.assert_array_equal(k.gather_qd(state).numpy(), state.joint_qd.numpy()[ids])
    test.assertNotEqual(k.digest, _TreeKinematics(model, actors=tuple(reversed(arms))).digest)


add_function_test(TestMultiTree, "test_multitree_kinematics", test_multitree_kinematics, devices=get_test_devices())
add_function_test(TestMultiTree, "test_multitree_objective", test_multitree_objective, devices=get_test_devices())
add_function_test(TestMultiTree, "test_reordered_tree_view", test_reordered_tree_view, devices=get_test_devices())


def test_multitree_step(test, device):
    for precision, pd in itertools.product(
        (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32), (False, True)
    ):
        with test.subTest(precision=precision, pd=pd):
            model = trees_model(device, coupled=pd)
            models = [model, *(trees_model(device, sizes=(n,)) for n in (1, 3))]
            q, qd = model.joint_q.numpy(), model.joint_qd.numpy()
            for local, start, end in ((models[1], 0, 1), (models[2], 1, 4)):
                local.joint_q.assign(q[start:end])
                local.joint_qd.assign(qd[start:end])
            if pd:
                for local, ids in zip(models, (slice(None), slice(0, 1), slice(1, 4)), strict=True):
                    local.joint_target_ke.assign(np.array([30, 40, 50, 60])[ids])
                    local.joint_target_kd.fill_(0.2)
                    local.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION_VELOCITY))
            solvers = [
                SolverMonolithic(
                    m,
                    friction_velocity_scale=0.01,
                    joint_terms=SolverMonolithic.JointTerms(implicit_pd=pd),
                    precision=precision,
                    newton_max_iterations=20,
                    mass_mode=SolverMonolithic.MassMode.LUMPED,
                    pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
                    use_cuda_graph=device.is_cuda,
                )
                for m in models
            ]
            states = [s.create_state(dt=0.002) for s in solvers]
            outputs = [s._states.copy(state) for s, state in zip(solvers, states, strict=True)]
            controls = [m.control() for m in models]
            # Each tree receives distinct effort or PD targets; compare with independently solved models.
            for control, ids in zip(controls, (slice(None), slice(0, 1), slice(1, 4)), strict=True):
                if pd:
                    control.joint_target_q.assign((q + np.array([0.001, 0.002, -0.003, 0.004]))[ids])
                    control.joint_target_qd.assign(np.array([0.02, -0.01, 0.03, 0.04])[ids])
                else:
                    control.joint_f.assign(np.array([0.01, -0.03, 0.02, 0.04])[ids])
            for _ in range(3):
                for solver, state, output, control in zip(solvers, states, outputs, controls, strict=True):
                    solver.step(state, output, control, dt=0.002)
                    test.assertTrue(solver.last_stats.converged, solver.last_stats)
                states, outputs = outputs, states
            tol = 3e-7 if precision == SolverMonolithic.Precision.FP32 else 1e-10
            np.testing.assert_allclose(
                states[0].joint_q.numpy(), np.concatenate([s.joint_q.numpy() for s in states[1:]]), atol=tol, rtol=tol
            )
            np.testing.assert_allclose(
                states[0].joint_qd.numpy(),
                np.concatenate([s.joint_qd.numpy() for s in states[1:]]),
                atol=tol / 0.002,
                rtol=tol,
            )
            # Fixed roots stay at their original world transforms.
            for body, x in ((0, 0), (2, 0.5)):
                np.testing.assert_allclose(states[0].body_q.numpy()[body], [x, 0, 0, 0, 0, 0, 1], atol=tol)
            test.assertEqual(len(solvers[0].last_stats.variational["articulation_blocks"]), 2)
            test.assertEqual(solvers[0].last_stats.q_block_count, 2)


add_function_test(TestMultiTree, "test_multitree_step", test_multitree_step, devices=get_test_devices())


def test_permuted_multitree_step(test, device):
    model = trees_model(device, coupled=True)
    options = {
        "friction_velocity_scale": 0.01,
        "mass_mode": SolverMonolithic.MassMode.LUMPED,
        "newton_max_iterations": 20,
        "pcg_mode": SolverMonolithic.PCGMode.PRODUCTION,
        "use_cuda_graph": device.is_cuda,
    }
    normal = SolverMonolithic(model, **options)
    table = ActorTable.from_model(model, actor_order=(1, 2, 0))
    with patch.object(ActorTable, "from_model", return_value=table):
        permuted = SolverMonolithic(model, **options)
    states = [s.create_state(dt=0.002) for s in (normal, permuted)]
    outputs = [s._states.copy(state) for s, state in zip((normal, permuted), states, strict=True)]
    control = model.control()
    control.joint_f.assign([0.01, -0.03, 0.02, 0.04])
    for _ in range(3):
        for solver, state, output in zip((normal, permuted), states, outputs, strict=True):
            solver.step(state, output, control, dt=0.002)
            test.assertTrue(solver.last_stats.converged, solver.last_stats)
        states, outputs = outputs, states
    for field in ("joint_q", "joint_qd", "particle_q", "particle_qd", "body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(states[0], field).numpy(), getattr(states[1], field).numpy(), atol=1e-9, rtol=1e-9
        )


add_function_test(
    TestMultiTree, "test_permuted_multitree_step", test_permuted_multitree_step, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
