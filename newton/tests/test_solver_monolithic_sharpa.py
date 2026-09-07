# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Portable trajectory and q-only lifecycle coverage for Sharpa G1H."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import solver_monolithic as impl
from newton._src.solvers.monolithic.linear import MonolithicPcgConfig, MonolithicPcgWarmStart
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices


def make_model(device):
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    parent = b.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    child = b.add_link(mass=0.5, inertia=wp.diag(wp.vec3(0.005)))
    joints = [
        b.add_joint_revolute(
            -1,
            parent,
            target_ke=10,
            target_kd=1,
            limit_ke=20,
            limit_kd=0.1,
            limit_lower=-2,
            limit_upper=2,
            friction=0.001,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        ),
        b.add_joint_prismatic(
            parent,
            child,
            target_ke=5,
            target_kd=0.5,
            limit_ke=30,
            limit_kd=0.2,
            limit_lower=-0.2,
            limit_upper=0.2,
            friction=0.002,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        ),
    ]
    b.add_articulation(joints)
    return b.finalize(device=device)


def make_solver(device):
    model = make_model(device)
    solver = SolverMonolithic._create_joint_diagnostic(
        model,
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True, limits=True, friction=True, limit_width=(0.01, 0.01), friction_velocity_scale=(0.01, 0.01)
        ),
    )
    return model, solver


def test_q_only_assembly(test, device):
    """Compare the q-only owner, global matrix, SPD scaling and dense solve."""
    model, solver = make_solver(device)
    state, control = model.state(), model.control()
    control.joint_target_q.assign(np.array([0.1, 0.01], dtype=np.float32))
    calls = []
    launch = wp.launch

    def record(kernel, dim, *args, **kwargs):
        calls.append((kernel.key, dim))
        return launch(kernel, dim, *args, **kwargs)

    with patch.object(wp, "launch", side_effect=record):
        solver.step(state, state, control, None, 0.001)
    test.assertTrue(solver.last_stats.converged)
    test.assertIsNone(solver._tet_workspace)
    test.assertIsNone(solver._contact)
    test.assertIsNone(state.particle_q)
    test.assertTrue(all(np.prod(dim) > 0 for _, dim in calls), [(k, d) for k, d in calls if np.prod(d) == 0])
    test.assertFalse(any("particle" in k or "_x_preconditioner" in k for k, _ in calls))
    linear = solver._linear
    owner = linear.aq_actor_dense.numpy().astype(float)
    bsr = linear.k_global_scalar_bsr
    dense = np.zeros((2, 2))
    offsets, columns, values = bsr.offsets.numpy(), bsr.columns.numpy(), bsr.values.numpy()
    for row in range(2):
        for k in range(offsets[row], offsets[row + 1]):
            dense[row, columns[k]] = values[k]
    np.testing.assert_allclose(dense, owner, rtol=1e-6)
    scale = linear.scale.numpy().astype(float)
    scaled = scale[:, None] * dense * scale[None, :]
    test.assertGreater(np.linalg.eigvalsh(scaled).min(), 0)
    # Re-solve the frozen current matrix with an independent RHS.
    rhs = wp.array(np.array([0.2, -0.1], dtype=np.float32), device=device)
    linear.factor_actor_preconditioner(generation=solver._generation, pivot_tolerance=0)
    result = linear.solve_pcg(
        rhs,
        solver._y,
        generation=solver._generation,
        warm_start=MonolithicPcgWarmStart.ZERO,
        config=MonolithicPcgConfig(100, 10, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 1e-12, 0.0, 10, 1e-3),
    )
    test.assertEqual(result.status.name, "SUCCESS")
    np.testing.assert_allclose(solver._y.numpy(), np.linalg.solve(scaled, rhs.numpy()), rtol=1e-4, atol=1e-6)
    test.assertTrue(np.isnan(solver.last_stats.convergence_ratio_x))
    test.assertEqual(solver.last_stats.active_sample_count, 0)


def test_q_only_transactions(test, device):
    """Exercise early convergence, soft stop, hard rollback and exception in place."""
    for inplace in (False, True):
        for mode in ("early", "soft", "hard", "exception"):
            model, solver = make_solver(device)
            state, control = model.state(), model.control()
            out = state if inplace else model.state()
            initial = state.joint_q.numpy().copy()
            if mode == "early":
                solver.step(state, out, control, None, 0.001)
                test.assertTrue(solver.last_stats.converged)
                test.assertEqual(solver.last_stats.nonlinear_iterations, 0)
            else:
                control.joint_target_q.fill_(0.1)
                if mode == "soft":
                    with patch.object(solver, "_iterate", return_value=solver.Status.NONLINEAR_MAX_ITERATIONS):
                        solver.step(state, out, control, None, 0.001)
                    test.assertFalse(solver.last_stats.rolled_back)
                elif mode == "hard":
                    with patch.object(
                        solver, "_iterate", side_effect=impl._StepFailure(solver.Status.NONFINITE, "injected")
                    ):
                        solver.step(state, out, control, None, 0.001)
                    test.assertTrue(solver.last_stats.rolled_back)
                else:
                    with patch.object(solver, "_evaluate_trial", side_effect=RuntimeError("injected")):
                        with test.assertRaises(RuntimeError):
                            solver.step(state, out, control, None, 0.001)
                    test.assertEqual(solver._joint_terms.final_generation, -1)
                    test.assertTrue(solver._transaction._finished)
                np.testing.assert_array_equal(out.joint_q.numpy(), initial)
            if mode != "exception":
                solver._joint_terms.evaluate(out, 0.001)
                np.testing.assert_allclose(solver.last_stats.joint_pd_force, solver._joint_terms.force.numpy()[:, 0])


def test_displacement_precision(test, device):
    """Retain sub-ULP BE motion at nonzero absolute joint angles."""
    model, solver = make_solver(device)
    state, control = model.state(), model.control()
    state.joint_q.assign(np.array([1.0, 0.1], dtype=np.float32))
    state.joint_qd.fill_(1e-7)
    solver._transaction.begin(state, state, control, 0.001)
    accepted = solver._transaction.accepted
    np.testing.assert_allclose(accepted.state.joint_qd.numpy(), 1e-7, rtol=1e-6)
    delta = wp.array(np.array([2e-10, -1e-10], dtype=np.float32), device=device)
    solver._transaction.trial.form_trial(accepted, delta, 1.0, 0.001)
    trial = solver._transaction.trial
    np.testing.assert_allclose(trial.state.joint_qd.numpy(), [3e-7, 0], rtol=1e-6, atol=1e-13)
    # Published coordinates round the same BE position to their declared float32 storage.
    expected = (state.joint_q.numpy().astype(float) + 0.001 * trial.state.joint_qd.numpy()).astype(np.float32)
    np.testing.assert_array_equal(trial.state.joint_q.numpy(), expected)


def test_q_only_scope(test, device):
    """Reject collisions and retain the normal public tet requirement."""
    model, solver = make_solver(device)
    with test.assertRaises(ValueError):
        impl._build_layout(model)
    with test.assertRaises(ValueError):
        solver.step(model.state(), model.state(), model.control(), object(), 0.001)
    with test.assertRaises(ValueError):
        solver.update_contacts(None)
    b = newton.ModelBuilder()
    body = b.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    joint = b.add_joint_revolute(-1, body)
    b.add_articulation([joint])
    b.add_shape_sphere(body, radius=0.01)
    with test.assertRaisesRegex(ValueError, "collision-disabled"):
        SolverMonolithic._create_joint_diagnostic(b.finalize(device=device), joint_terms=SolverMonolithic.JointTerms())


def test_trial_freeze_and_publication(test, device):
    """Keep targets frozen across reversible trials and invalidate failed publication."""
    model, solver = make_solver(device)
    state, control = model.state(), model.control()
    control.joint_target_q.fill_(0.1)
    solver.step(state, state, control, None, 0.001)
    transaction = solver._transaction
    transaction.begin(state, state, control, 0.001)
    solver._joint_terms.snapshot_targets(control)
    solver._evaluate_current(transaction.accepted, 0.001)
    expected = transaction.accepted.residual.numpy().copy()
    control.joint_target_q.fill_(0.8)
    delta = wp.array(np.array([2.1, 0.3], dtype=np.float32), device=device)
    transaction.trial.form_trial(transaction.accepted, delta, 1.0, 0.001)
    solver._evaluate_trial(transaction.trial, 0.001)
    test.assertTrue(np.any(solver._joint_terms.force.numpy()[:, 1] != 0))
    delta.zero_()
    transaction.trial.form_trial(transaction.accepted, delta, 1.0, 0.001)
    solver._evaluate_trial(transaction.trial, 0.001)
    np.testing.assert_array_equal(transaction.trial.residual.numpy(), expected)
    allocator = model.device.get_allocator()
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
        for _ in range(20):
            transaction.trial.form_trial(transaction.accepted, delta, 1.0, 0.001)
            solver._evaluate_trial(transaction.trial, 0.001)
    test.assertEqual(allocate.call_count, 0)
    transaction._finished = True
    original = state.joint_q.numpy().copy()
    publish = solver._publish_final
    calls = []

    def fail_once(returned_state, status):
        calls.append(status)
        if len(calls) == 1:
            raise impl._StepFailure(solver.Status.NONFINITE, "injected publication failure")
        publish(returned_state, status)

    with patch.object(solver, "_publish_final", side_effect=fail_once):
        solver.step(state, state, control, None, 0.001)
    test.assertTrue(solver.last_stats.rolled_back)
    np.testing.assert_array_equal(state.joint_q.numpy(), original)
    test.assertEqual(solver.last_stats.joint_force_generation, solver.last_stats.step_generation)
    with patch.object(solver, "_publish_final", side_effect=impl._StepFailure(solver.Status.NONFINITE, "persistent")):
        with test.assertRaises(RuntimeError):
            solver.step(state, state, control, None, 0.001)
    test.assertEqual(solver._joint_terms.final_generation, -1)
    np.testing.assert_array_equal(state.joint_q.numpy(), original)


class TestMonolithicJointDiagnostic(unittest.TestCase):
    pass


for name, function in (
    ("assembly", test_q_only_assembly),
    ("transactions", test_q_only_transactions),
    ("displacement_precision", test_displacement_precision),
    ("scope", test_q_only_scope),
    ("trial_freeze_and_publication", test_trial_freeze_and_publication),
):
    add_function_test(TestMonolithicJointDiagnostic, "test_" + name, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
