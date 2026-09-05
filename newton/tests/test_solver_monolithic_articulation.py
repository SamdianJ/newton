# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify reusable articulation dynamics and its residual convention."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    articulation_point_jacobian_column,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
    project_articulation_body_wrenches,
    recover_articulation_candidate_rates,
    scatter_articulation_actor_tangent,
)
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _point_columns(
    J: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    point: wp.vec3,
    result: wp.array[wp.vec3],
):
    d = wp.tid()
    result[d] = articulation_point_jacobian_column(J, 0, 1, d, body_q[1], body_com[1], point)


def test_candidate_parity(test, device):
    """Match public dynamics on twelve alternating candidates without stale scratch."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state = fixture.model, fixture.state
    model.gravity.assign(np.asarray([[1.3, -8.7, -2.0]], dtype=np.float32))
    workspace = MonolithicArticulationWorkspace(model)
    mass, gravity, coriolis = wp.empty_like(workspace.M), wp.empty_like(workspace.g), wp.empty_like(workspace.C)
    a = None
    for i in range(12):
        q = [0.2, 0.01] if i % 2 == 0 else [-0.5 + i * 0.03, 0.07]
        state.joint_q.assign(np.asarray(q, dtype=np.float32))
        state.joint_qd.assign(np.asarray([0.1, -0.02] if i % 2 == 0 else [0.7, 0.4], dtype=np.float32))
        eval_articulation_passive_candidate(model, state, workspace)
        newton.eval_inverse_dynamics_passive(
            model, state, mass_matrix=mass, gravity_force=gravity, coriolis_force=coriolis
        )
        for actual, expected in ((workspace.M, mass), (workspace.g, gravity), (workspace.C, coriolis)):
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=1e-5 if device.is_cuda else 1e-6, atol=1e-8
            )
        values = (workspace.M.numpy().copy(), workspace.g.numpy().copy(), workspace.C.numpy().copy())
        if i == 0:
            a = values
        elif i % 2 == 0:
            for actual, expected in zip(values, a, strict=True):
                np.testing.assert_array_equal(actual, expected)


def test_allocation_and_fk_fallback(test, device):
    """Trace zero allocations after warming both public FK dispatch paths."""
    for fallback in (False, True):
        fixture = build_tiny_cpu_fixture(device=device)
        model, state = fixture.model, fixture.state
        if fallback:
            model._fk_articulation_level_start = None
        workspace = MonolithicArticulationWorkspace(model)
        eval_articulation_passive_candidate(model, state, workspace)
        other = fixture.state_next
        other.joint_q.assign(np.asarray([-0.4, 0.05], dtype=np.float32))
        other.joint_qd.assign(np.asarray([0.6, -0.3], dtype=np.float32))
        allocator = model.device.get_allocator()
        with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
            for i in range(100):
                eval_articulation_passive_candidate(model, state if i % 2 == 0 else other, workspace)
        test.assertEqual(allocate.call_count, 0)


def test_point_jacobian(test, device):
    """Compare a non-COM world point Jacobian with central FK differences."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state = fixture.model, fixture.state
    workspace = MonolithicArticulationWorkspace(model)
    eval_articulation_passive_candidate(model, state, workspace)
    local = wp.vec3(0.07, 0.04, -0.03)
    point = wp.transform_point(wp.transform(*state.body_q.numpy()[1]), local)
    result = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(
        _point_columns, 2, inputs=[workspace.scratch.J, state.body_q, model.body_com, point, result], device=device
    )
    actual = result.numpy()
    q0 = state.joint_q.numpy().copy()
    for h in (0.003, 0.001, 0.0003, 0.0001, 0.00003):
        for d in range(2):
            samples = []
            for sign in (-1, 1):
                q = q0.copy()
                q[d] += sign * h
                state.joint_q.assign(q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                samples.append(np.asarray(wp.transform_point(wp.transform(*state.body_q.numpy()[1]), local)))
            expected = (samples[1] - samples[0]) / (2 * h)
            np.testing.assert_allclose(actual[d], expected, rtol=5e-3, atol=2e-4)


def test_residual_scatter_and_rates(test, device):
    """Check BE rates, world-COM wrench signs, gravity and fixed-slot mass scatter."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state = fixture.model, fixture.state
    model.gravity.assign(np.asarray([[0.0, -9.81, 0.0]], dtype=np.float32))
    workspace = MonolithicArticulationWorkspace(model)
    eval_articulation_passive_candidate(model, state, workspace)
    wrench = np.asarray([[0.0, 2.0, 0.0, 0.0, 0.0, 0.3], [1.0, -0.4, 0.7, 0.0, 0.0, -0.2]], dtype=np.float32)
    state.body_f.assign(wrench)
    wp.launch(
        project_articulation_body_wrenches,
        2,
        inputs=[
            0,
            2,
            model.articulation_start,
            model.articulation_end,
            model.joint_child,
            workspace.scratch.J,
            state.body_f,
            workspace.generalized_body_force,
        ],
        device=device,
    )
    expected_body = workspace.scratch.J.numpy()[0].T @ wrench.flatten()
    np.testing.assert_allclose(workspace.generalized_body_force.numpy(), expected_body, atol=1e-7)
    # Gravity compensation opposes the physical downward body forces.
    gravity_wrench = np.zeros((2, 6), dtype=np.float32)
    gravity_wrench[:, 1] = -9.81 * model.body_mass.numpy()
    np.testing.assert_allclose(
        workspace.g.numpy(), -workspace.scratch.J.numpy()[0].T @ gravity_wrench.flatten(), atol=1e-6
    )
    q0, v0 = wp.clone(state.joint_q), wp.clone(state.joint_qd)
    state.joint_q.assign(np.asarray([0.21, 0.008], dtype=np.float32))
    qdd = wp.empty(2, dtype=float, device=device)
    wp.launch(
        recover_articulation_candidate_rates,
        2,
        inputs=[workspace.dof_to_coord, 0, 2, 100.0, state.joint_q, q0, v0, state.joint_qd, qdd],
        device=device,
    )
    velocity = (state.joint_q.numpy() - q0.numpy()) * 100
    np.testing.assert_allclose(state.joint_qd.numpy(), velocity, atol=1e-6)
    np.testing.assert_allclose(qdd.numpy(), (velocity - v0.numpy()) * 100, atol=1e-5)
    fixture.control.joint_f.assign(np.asarray([0.8, -0.5], dtype=np.float32))
    workspace.validate_candidate(model, state, qdd, fixture.control.joint_f, state.body_f)
    residual = wp.empty(2, dtype=float, device=device)
    wp.launch(
        eval_articulation_actor_residual,
        2,
        inputs=[
            0,
            0,
            2,
            workspace.M,
            qdd,
            workspace.C,
            workspace.g,
            fixture.control.joint_f,
            workspace.generalized_body_force,
            residual,
        ],
        device=device,
    )
    expected = (
        workspace.M.numpy()[0] @ qdd.numpy()
        + workspace.C.numpy()
        + workspace.g.numpy()
        - fixture.control.joint_f.numpy()
        - expected_body
    )
    np.testing.assert_allclose(residual.numpy(), expected, rtol=2e-6, atol=1e-6)
    aq = wp.zeros((2, 2), dtype=float, device=device)
    rows, columns = wp.full(8, -1, dtype=int, device=device), wp.full(8, -1, dtype=int, device=device)
    values = wp.full(8, -1.0, dtype=float, device=device)
    wp.launch(
        scatter_articulation_actor_tangent,
        (2, 2),
        inputs=[0, 2, 3, 2, 10000.0, workspace.M, aq, rows, columns, values],
        device=device,
    )
    np.testing.assert_allclose(aq.numpy(), workspace.M.numpy()[0] * 10000, rtol=1e-7)
    np.testing.assert_array_equal(values.numpy()[2:6].reshape(2, 2), aq.numpy())
    np.testing.assert_array_equal(rows.numpy(), [-1, -1, 3, 3, 4, 4, -1, -1])
    np.testing.assert_array_equal(columns.numpy(), [-1, -1, 3, 4, 3, 4, -1, -1])


def test_scope_and_stale_validation(test, device):
    """Reject unsupported joints, stale topology and malformed candidate inputs before FK."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state = fixture.model, fixture.state
    gains = (
        "joint_armature",
        "joint_damping",
        "joint_friction",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_target_ke",
        "joint_target_kd",
    )
    for name in gains:
        array = getattr(model, name)
        array.fill_(1.0)
        with test.assertRaisesRegex(ValueError, name):
            MonolithicArticulationWorkspace(model)
        array.zero_()
    model.constraint_mimic_count = 1
    with test.assertRaisesRegex(ValueError, "mimic"):
        MonolithicArticulationWorkspace(model)
    model.constraint_mimic_count = 0
    model.actuators.append(object())
    with test.assertRaisesRegex(ValueError, "actuator"):
        MonolithicArticulationWorkspace(model)
    model.actuators.clear()
    joint_types = model.joint_type.numpy().copy()
    model.joint_type.fill_(int(newton.JointType.BALL))
    with test.assertRaisesRegex(ValueError, "joint"):
        MonolithicArticulationWorkspace(model)
    model.joint_type.assign(joint_types)
    workspace = MonolithicArticulationWorkspace(model)
    for name in (
        "joint_parent",
        "body_mass",
        "_fk_articulation_level_start",
        "_fk_level_joint_start",
        "_fk_level_joints",
        "_fk_level_parent_pos",
        "_fk_level_capacity",
    ):
        original = getattr(model, name)
        setattr(model, name, wp.clone(original) if isinstance(original, wp.array) else (original or 0) + 1)
        with test.assertRaisesRegex(ValueError, "stale"):
            eval_articulation_passive_candidate(model, state, workspace)
        setattr(model, name, original)
    qdd = wp.zeros(2, dtype=float, device=device)
    with test.assertRaisesRegex(ValueError, "joint_qdd"):
        workspace.validate_candidate(
            model, state, wp.zeros(1, dtype=float, device=device), fixture.control.joint_f, state.body_f
        )
    original = state.joint_q
    state.joint_q = wp.zeros(2, dtype=wp.float64, device=device)
    with test.assertRaisesRegex(ValueError, "joint_q"):
        workspace.validate_candidate(model, state, qdd, fixture.control.joint_f, state.body_f)
    state.joint_q = original


def test_inertia_validation(test, device):
    """Reject invalid inertia and zero-inertia DOFs without changing model initial state."""
    fixture = build_tiny_cpu_fixture(device=device)
    model = fixture.model
    initial = model.joint_q.numpy().copy()
    mass = model.body_mass.numpy().copy()
    inertia = model.body_inertia.numpy().copy()
    for values in ([-1.0, 0.5], [float("nan"), 0.5]):
        model.body_mass.assign(np.asarray(values, dtype=np.float32))
        with test.assertRaisesRegex(ValueError, "mass"):
            MonolithicArticulationWorkspace(model)
    model.body_mass.assign(mass)
    for bad in (np.full_like(inertia, np.nan), -inertia):
        model.body_inertia.assign(bad)
        with test.assertRaisesRegex(ValueError, "inertia"):
            MonolithicArticulationWorkspace(model)
    model.body_mass.zero_()
    model.body_inertia.zero_()
    with test.assertRaisesRegex(ValueError, "inertia"):
        MonolithicArticulationWorkspace(model)
    model.body_mass.assign(mass)
    model.body_inertia.assign(inertia)
    MonolithicArticulationWorkspace(model)
    np.testing.assert_array_equal(initial, model.joint_q.numpy())


def test_model_storage_validation(test, device):
    """Reject malformed model storage before the constructor launches dynamics."""
    model = build_tiny_cpu_fixture(device=device).model
    for name, bad in (
        ("body_mass", wp.zeros(model.body_count, dtype=wp.float64, device=device)),
        ("joint_q_start", wp.zeros(1, dtype=int, device=device)),
        ("joint_dof_dim", wp.zeros((model.joint_count, 1), dtype=int, device=device)),
    ):
        original = getattr(model, name)
        setattr(model, name, bad)
        with test.assertRaisesRegex(ValueError, name):
            MonolithicArticulationWorkspace(model)
        setattr(model, name, original)


def test_fixed_branch_and_stream(test, device):
    """Retain fixed links and body maps on a branched tree with an owned CUDA stream."""
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    links = [builder.add_link(mass=0.5, com=(0.03, 0.01, 0.0), inertia=wp.diag(wp.vec3(0.01))) for _ in range(3)]
    passive = {
        "armature": 0.0,
        "damping": 0.0,
        "friction": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "target_ke": 0.0,
        "target_kd": 0.0,
    }
    root = builder.add_joint_revolute(-1, links[2], axis=newton.Axis.Z, **passive)
    fixed = builder.add_joint_fixed(links[2], links[0], parent_xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()))
    moving = builder.add_joint_prismatic(links[2], links[1], axis=newton.Axis.X, **passive)
    builder.add_articulation([root, fixed, moving])
    model = builder.finalize(device=device)
    state = model.state()
    stream = wp.Stream(device) if device.is_cuda else None
    workspace = MonolithicArticulationWorkspace(model, stream=stream)
    np.testing.assert_array_equal(workspace.body_to_link_index.numpy(), [1, 2, 0])
    eval_articulation_passive_candidate(model, state, workspace)
    with wp.ScopedStream(stream):
        expected = wp.empty_like(workspace.M)
        newton.eval_inverse_dynamics_passive(model, state, mass_matrix=expected)
        np.testing.assert_allclose(workspace.M.numpy(), expected.numpy(), rtol=1e-6)
    if device.is_cuda:
        wrong = wp.zeros(model.joint_coord_count, dtype=float, device="cpu")
        state.joint_q = wrong
        with test.assertRaisesRegex(ValueError, "device"):
            eval_articulation_passive_candidate(model, state, workspace)


class TestMonolithicArticulation(unittest.TestCase):
    """Exercise the private articulation component on supported devices."""


for _test in (
    test_candidate_parity,
    test_allocation_and_fk_fallback,
    test_point_jacobian,
    test_residual_scatter_and_rates,
    test_scope_and_stale_validation,
    test_inertia_validation,
    test_model_storage_validation,
    test_fixed_branch_and_stream,
):
    add_function_test(TestMonolithicArticulation, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
