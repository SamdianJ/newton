# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G1 independent scalar oracles for optional monolithic joint physics."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import solver_monolithic as monolithic
from newton._src.solvers.monolithic.articulation import MonolithicArticulationWorkspace, MonolithicJointTermsWorkspace
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


def make_terms(device):
    fixture = build_tiny_cpu_fixture(device=device)
    model = fixture.model
    for name, values in {
        "joint_target_ke": [4.0, 7.0],
        "joint_target_kd": [0.3, 0.5],
        "joint_effort_limit": [2.0, 3.0],
        "joint_velocity_limit": [5.0, 8.0],
        "joint_limit_lower": [-0.4, -0.1],
        "joint_limit_upper": [0.4, 0.1],
        "joint_limit_ke": [12.0, 20.0],
        "joint_limit_kd": [0.4, 0.7],
        "joint_friction": [0.2, 0.3],
    }.items():
        getattr(model, name).assign(np.asarray(values, dtype=np.float32))
    model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION_VELOCITY))
    fixture.control.joint_target_q.assign(np.asarray([0.1, 0.02], dtype=np.float32))
    fixture.control.joint_target_qd.assign(np.asarray([0.2, -0.1], dtype=np.float32))
    terms = MonolithicJointTermsWorkspace(
        model,
        implicit_pd=True,
        limits=True,
        friction=True,
        limit_width=(0.1, 0.05),
        friction_velocity_scale=(0.07, 0.03),
    )
    terms.snapshot_targets(fixture.control)
    return fixture, terms


def oracle(q, v, qt, vt, h, p):
    """Float64 residual and potentials; no production function is called."""
    kp, kd, limit, lo, hi, ke, kl, width, friction, eps = map(float, p)
    raw = kp * (q - qt) + kd * (v - vt)
    pd = np.clip(raw, -limit, limit) if kp + kd else 0.0
    a = kp + kd / h
    energy_pd = (0.5 * raw**2 if abs(raw) <= limit else limit * (abs(raw) - 0.5 * limit)) / a if a else 0.0
    r_limit = energy_limit = power = 0.0
    for sign, bound in ((1, hi), (-1, lo)):
        penetration = max(sign * (q - bound), 0.0)
        t = np.clip(penetration / width, 0.0, 1.0)
        damper = kl * t * t * (3 - 2 * t) * max(sign * v, 0.0)
        r_limit += sign * (ke * penetration + damper)
        energy_limit += 0.5 * ke * penetration**2
        power -= damper * sign * v
    jf = friction * v / np.hypot(v, eps)
    energy_jf = h * friction * (np.hypot(v, eps) - eps)
    return np.array([pd, r_limit, jf]), np.array([energy_pd, energy_limit, energy_jf]), np.array([0.0, power, -jf * v])


def evaluate(test, fixture, terms, q, v, h):
    fixture.state.joint_q.assign(np.asarray(q, dtype=np.float32))
    fixture.state.joint_qd.assign(np.asarray(v, dtype=np.float32))
    terms.evaluate(fixture.state, h)
    test.assertEqual(int(terms.status.numpy()[0]), 0)
    return terms.residual.numpy().astype(float), terms.tangent.numpy().astype(float)


def test_formula_and_derivatives(test, device):
    fixture, terms = make_terms(device)
    params = terms.params.numpy().astype(float)
    qt, vt = terms.target_q.numpy().astype(float), terms.target_qd.numpy().astype(float)
    for h in (0.01, 0.05):
        for q, v in (
            ([0.12, 0.025], [0.22, -0.08]),
            ([0.44, 0.12], [0.3, 0.1]),
            ([-0.44, -0.12], [-0.3, -0.1]),
            ([0.44, 0.12], [-0.3, -0.1]),
            ([-0.44, -0.12], [0.3, 0.1]),
            ([1.0, 0.3], [3.0, 2.0]),
            ([-1.0, -0.3], [-3.0, -2.0]),
            ([0.1, 0.02], [0.0, 0.0]),
            ([0.2, 0.03], [1e-4, -1e-4]),
        ):
            r, k = evaluate(test, fixture, terms, q, v, h)
            stored_q, stored_v = fixture.state.joint_q.numpy().copy(), fixture.state.joint_qd.numpy().copy()
            expected = [oracle(float(stored_q[i]), float(stored_v[i]), qt[i], vt[i], h, params[i]) for i in range(2)]
            np.testing.assert_allclose(r, [x[0] for x in expected], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(terms.force.numpy(), -r, rtol=0, atol=0)
            np.testing.assert_allclose(terms.potential.numpy(), [x[1] for x in expected], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(terms.dissipation_power.numpy(), [x[2] for x in expected], rtol=1e-5, atol=1e-8)
            test.assertTrue(np.all(k >= 0))
            test.assertTrue(np.all(terms.dissipation_power.numpy()[:, 1:] <= 0))
            # Independently differentiate actual float32 device residuals.
            delta = 1e-5
            positive = evaluate(test, fixture, terms, np.asarray(q) + delta, np.asarray(v) + delta / h, h)[0]
            negative = evaluate(test, fixture, terms, np.asarray(q) - delta, np.asarray(v) - delta / h, h)[0]
            np.testing.assert_allclose(k, (positive - negative) / (2 * delta), rtol=5e-3, atol=2e-3)
            # Finite difference q with BE velocity recovery, on each smooth branch.
            for i in range(2):
                dq = 1e-7
                plus = oracle(float(stored_q[i]) + dq, float(stored_v[i]) + dq / h, qt[i], vt[i], h, params[i])
                minus = oracle(float(stored_q[i]) - dq, float(stored_v[i]) - dq / h, qt[i], vt[i], h, params[i])
                fd = (plus[0] - minus[0]) / (2 * dq)
                np.testing.assert_allclose(k[i], fd, rtol=5e-3, atol=1e-5)
                grad = (plus[1] - minus[1]) / (2 * dq)
                np.testing.assert_allclose(r[i, [0, 2]], grad[[0, 2]], rtol=5e-3, atol=1e-6)


def test_saturation_and_kinks(test, device):
    fixture, terms = make_terms(device)
    p = terms.params.numpy()
    p[:, :] = 0
    p[:, 0] = 2
    p[:, 1] = 0.5
    p[:, 2] = 2
    p[:, 7] = 0.1
    p[:, 9] = 0.1
    terms.params.assign(p)
    terms.target_q.zero_()
    terms.target_qd.zero_()
    # Elastic=1, damping=1.5: total must cap at 2, not 2.5.
    r, k = evaluate(test, fixture, terms, [0.5, -0.5], [3, -3], 0.1)
    np.testing.assert_array_equal(r[:, 0], [2, -2])
    np.testing.assert_array_equal(k[:, 0], [0, 0])
    r, k = evaluate(test, fixture, terms, [1, -1], [0, 0], 0.1)
    np.testing.assert_array_equal(k[:, 0], [0, 0])
    test.assertTrue(np.all(terms.saturated.numpy() == 1))
    # One-sided approach to the saturation boundary.
    r, k = evaluate(test, fixture, terms, [1 - 1e-4, 1 + 1e-4], [-1e-3, 1e-3], 0.1)
    test.assertLess(r[0, 0], 2)
    test.assertEqual(r[1, 0], 2)
    test.assertAlmostEqual(k[0, 0], 7)
    test.assertEqual(k[1, 0], 0)
    # Explicit limit activation/speed kinks choose derivative zero at d=0.
    p[:, 0:3] = 0
    p[:, 3] = -0.5
    p[:, 4] = 0.5
    p[:, 5] = 10
    p[:, 6] = 2
    terms.params.assign(p)
    r, k = evaluate(test, fixture, terms, [0.5, -0.5], [1, -1], 0.1)
    np.testing.assert_array_equal(r, 0)
    np.testing.assert_array_equal(k, 0)
    r, k = evaluate(test, fixture, terms, [0.55, -0.55], [0, 0], 0.1)
    np.testing.assert_allclose(r[:, 1], [0.5, -0.5], atol=1e-6)
    np.testing.assert_array_equal(k[:, 1], [10, 10])
    # Zero all terms, and pure P / pure D remain valid.
    p[:, :] = 0
    p[:, 7] = 1
    p[:, 9] = 1
    terms.params.assign(p)
    r, k = evaluate(test, fixture, terms, [1, -1], [1, -1], 0.1)
    np.testing.assert_array_equal(r, 0)
    np.testing.assert_array_equal(k, 0)
    p[0, 0] = 2
    p[1, 1] = 3
    p[:, 2] = 100
    terms.params.assign(p)
    r, k = evaluate(test, fixture, terms, [0.2, 0.2], [0.1, 0.1], 0.1)
    np.testing.assert_allclose(r[:, 0], [0.4, 0.3], rtol=1e-6)
    np.testing.assert_allclose(k[:, 0], [2, 30], rtol=1e-6)


def test_frozen_inputs_and_allocation(test, device):
    fixture, terms = make_terms(device)
    q, v = [0.43, 0.11], [0.3, 0.1]
    original = evaluate(test, fixture, terms, q, v, 0.01)[0]
    fixture.control.joint_target_q.fill_(7)
    fixture.model.joint_target_ke.fill_(900)
    np.testing.assert_array_equal(evaluate(test, fixture, terms, q, v, 0.01)[0], original)
    evaluate(test, fixture, terms, [-0.44, -0.12], [-0.3, -0.1], 0.01)
    np.testing.assert_array_equal(evaluate(test, fixture, terms, q, v, 0.01)[0], original)
    allocator = fixture.model.device.get_allocator()
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as alloc:
        for _ in range(100):
            terms.evaluate(fixture.state, 0.01)
    test.assertEqual(alloc.call_count, 0)
    terms.snapshot_targets(fixture.control)
    test.assertFalse(np.array_equal(evaluate(test, fixture, terms, q, v, 0.01)[0], original))


def test_invalid_config_and_control(test, device):
    for name, values in [
        ("joint_target_ke", [-1, 1]),
        ("joint_target_kd", [0, np.nan]),
        ("joint_effort_limit", [0, 1]),
        ("joint_velocity_limit", [1, -1]),
        ("joint_limit_lower", [1, 1]),
        ("joint_friction", [-1, 1]),
    ]:
        fixture, _ = make_terms(device)
        getattr(fixture.model, name).assign(np.asarray(values, dtype=np.float32))
        with test.assertRaises(ValueError):
            MonolithicJointTermsWorkspace(
                fixture.model,
                implicit_pd=True,
                limits=True,
                friction=True,
                limit_width=(0.1, 0.1),
                friction_velocity_scale=(0.1, 0.1),
            )
    fixture, terms = make_terms(device)
    with test.assertRaises(ValueError):
        MonolithicArticulationWorkspace(fixture.model)
    MonolithicArticulationWorkspace(fixture.model, joint_terms=terms)
    for prop in ("joint_armature", "joint_damping"):
        getattr(fixture.model, prop).fill_(0.1)
        with test.assertRaises(ValueError):
            MonolithicArticulationWorkspace(fixture.model, joint_terms=terms)
        getattr(fixture.model, prop).zero_()
    for args in (
        {"limits": True},
        {"friction": True, "friction_velocity_scale": (0.1,)},
        {"limits": True, "limit_width": (0.1, 0)},
        {"implicit_pd": 1},
    ):
        with test.assertRaises(ValueError):
            MonolithicJointTermsWorkspace(fixture.model, **args)
    fixture.control.joint_f.fill_(1)
    with test.assertRaisesRegex(ValueError, "Duplicate"):
        terms.snapshot_targets(fixture.control)
    fixture.control.joint_f.zero_()
    fixture.control.joint_target_q.fill_(float("nan"))
    with test.assertRaisesRegex(ValueError, "Nonfinite"):
        terms.snapshot_targets(fixture.control)
    fixture.control.joint_target_q.zero_()
    fixture.model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION))
    with test.assertRaisesRegex(ValueError, "POSITION_VELOCITY"):
        MonolithicJointTermsWorkspace(fixture.model, implicit_pd=True)
    fixture.model.joint_target_ke = wp.clone(fixture.model.joint_target_ke)
    with test.assertRaisesRegex(ValueError, "Stale"):
        terms.snapshot_targets(fixture.control)


def make_solver(device, **options):
    fixture, _ = make_terms(device)
    fixture.model.gravity.zero_()
    fixture.state.joint_q.zero_()
    fixture.state.joint_qd.zero_()
    fixture.state.body_qd.zero_()
    fixture.control.joint_target_q.assign(np.array([0.005, 0.001], dtype=np.float32))
    fixture.control.joint_target_qd.zero_()
    newton.eval_fk(fixture.model, fixture.state.joint_q, fixture.state.joint_qd, fixture.state)
    solver = SolverMonolithic(
        fixture.model,
        collision_pipeline=MonolithicCollisionPipeline(fixture.model),
        contact_stiffness=1e5,
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True, limits=True, friction=True, limit_width=(0.1, 0.05), friction_velocity_scale=(0.07, 0.03)
        ),
        **options,
    )
    return fixture, solver


def check_final(test, fixture, solver, state, dt):
    terms = solver._joint_terms
    expected = []
    for i in range(2):
        expected.append(
            oracle(
                float(state.joint_q.numpy()[i]),
                float(state.joint_qd.numpy()[i]),
                float(terms.target_q.numpy()[i]),
                float(terms.target_qd.numpy()[i]),
                dt,
                terms.params.numpy()[i],
            )[0]
        )
    np.testing.assert_allclose(terms.final_force.numpy(), -np.asarray(expected), rtol=1e-5, atol=1e-7)
    stats = solver.last_stats
    test.assertEqual(stats.joint_force_generation, stats.step_generation)
    for i, values in enumerate((stats.joint_pd_force, stats.joint_limit_force, stats.joint_friction_force)):
        np.testing.assert_array_equal(values, terms.final_force.numpy()[:, i])


def test_solver_assembly_and_dense_solve(test, device):
    fixture, solver = make_solver(device)
    dt = 0.005
    current = solver._evaluate_current
    solve = solver._linear.solve_pcg
    calls = [0, 0]

    def inspect_current(candidate, h):
        result = current(candidate, h)
        dense = solver._linear.densify_for_test(generation=solver._generation)
        expected = solver._articulation.M.numpy()[0].astype(float) / h**2 + np.diag(
            solver._joint_terms.tangent.numpy().sum(axis=1)
        )
        tol = 5e-5 if device.is_cuda else 1e-5
        np.testing.assert_allclose(dense.raw_matrix[:2, :2], expected, rtol=tol, atol=1e-6)
        np.testing.assert_allclose(solver._linear.aq_actor_dense.numpy(), expected, rtol=tol, atol=1e-6)
        np.testing.assert_array_equal(dense.raw_matrix[:2, 2:], 0)
        np.testing.assert_array_equal(dense.raw_matrix[2:, :2], 0)
        test.assertGreater(np.linalg.eigvalsh(dense.scaled_matrix).min(), 0)
        np.testing.assert_allclose(dense.scaled_matrix, dense.scaled_matrix.T, rtol=1e-4, atol=1e-6)
        calls[0] += 1
        return result

    def inspect_solve(rhs, y, **kwargs):
        dense = solver._linear.densify_for_test(generation=solver._generation)
        reference = np.linalg.solve(dense.scaled_matrix, rhs.numpy().astype(float))
        result = solve(rhs, y, **kwargs)
        if result.status == monolithic.MonolithicLinearStatus.SUCCESS:
            np.testing.assert_allclose(y.numpy(), reference, rtol=1e-4, atol=1e-7)
            calls[1] += 1
        return result

    with (
        patch.object(solver, "_evaluate_current", side_effect=inspect_current),
        patch.object(solver._linear, "solve_pcg", side_effect=inspect_solve),
    ):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, dt)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
    test.assertGreater(min(calls), 0)
    test.assertGreater(np.linalg.norm(fixture.state_next.joint_q.numpy()), 0)
    check_final(test, fixture, solver, fixture.state_next, dt)


def test_trial_transitions_and_input_freeze(test, device):
    fixture, solver = make_solver(device)
    original = solver._evaluate_current
    observed = []

    def inspect(candidate, h):
        original(candidate, h)
        owner = solver._linear.aq_actor_dense.numpy().copy()
        initial = solver._joint_terms.residual.numpy().copy()
        target = solver._joint_terms.target_q.numpy().copy()
        fixture.control.joint_target_q.fill_(10)
        trial = solver._transaction.trial
        for offset in ([0.44, 0.12], [-0.44, -0.12], [0, 0]):
            delta = np.zeros(solver._layout.scalar_dof_count, dtype=np.float32)
            delta[:2] = offset
            solver._delta.assign(delta)
            trial.form_trial(candidate, solver._delta, 1.0, h)
            solver._evaluate_trial(trial, h)
            np.testing.assert_array_equal(solver._linear.aq_actor_dense.numpy(), owner)
            np.testing.assert_array_equal(solver._joint_terms.target_q.numpy(), target)
            observed.append(solver._joint_terms.residual.numpy().copy())
        np.testing.assert_allclose(observed[-1], initial, rtol=1e-6, atol=1e-7)
        test.assertTrue(np.all(solver._joint_terms.final_force.numpy() == 0))
        raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "discard G1 transition probe")

    with (
        patch.object(solver, "_evaluate_current", side_effect=inspect),
        test.assertLogs(monolithic.__name__, level="WARNING"),
    ):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.005)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertTrue(np.all(observed[0][:, 1] > 0))
    test.assertTrue(np.all(observed[1][:, 1] < 0))
    np.testing.assert_array_equal(np.abs(observed[0][:, 0]), [2, 3])
    check_final(test, fixture, solver, fixture.state_next, 0.005)


def test_joint_final_transactions(test, device):
    for disposition in (
        "success",
        "soft_stop",
        "hard_failure",
        "final_failure",
        "permanent_final_failure",
        "exception",
    ):
        fixture, solver = make_solver(device, newton_max_iterations=1 if disposition == "soft_stop" else 10)
        state = fixture.state
        original = {
            name: getattr(state, name).numpy().copy()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q", "particle_qd")
        }
        dt = 0.005
        with ExitStack() as stack:
            if disposition == "soft_stop":
                stack.enter_context(patch.object(solver, "_converged", return_value=False))
            elif disposition == "hard_failure":
                stack.enter_context(
                    patch.object(
                        solver,
                        "_iterate",
                        side_effect=monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "forced G1 rollback"),
                    )
                )
            elif disposition == "exception":
                stack.enter_context(patch.object(solver, "_iterate", side_effect=ValueError("forced G1 exception")))
            elif disposition in ("final_failure", "permanent_final_failure"):
                publish = solver._joint_terms.publish
                count = [0]

                def fail_once(generation, count=count, disposition=disposition, publish=publish):
                    count[0] += 1
                    if count[0] == 1 or disposition == "permanent_final_failure":
                        raise monolithic._StepFailure(
                            SolverMonolithic.Status.NONFINITE, "forced joint publication failure"
                        )
                    return publish(generation)

                stack.enter_context(patch.object(solver._joint_terms, "publish", side_effect=fail_once))
            if disposition not in ("success", "exception"):
                stack.enter_context(test.assertLogs(monolithic.__name__, level="WARNING"))
            if disposition in ("exception", "permanent_final_failure"):
                with test.assertRaises(ValueError if disposition == "exception" else RuntimeError):
                    solver.step(state, state, fixture.control, None, dt)
                test.assertEqual(solver._joint_terms.final_generation, -1)
            else:
                solver.step(state, state, fixture.control, None, dt)
                check_final(test, fixture, solver, state, dt)
        if disposition == "success":
            test.assertTrue(solver.last_stats.converged)
        elif disposition == "soft_stop":
            test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
            test.assertFalse(solver.last_stats.rolled_back)
        else:
            for name, value in original.items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), value)
        test.assertTrue(solver._transaction._finished)


def test_solver_configuration_and_disabled_parity(test, device):
    fixture, solver = make_solver(device)
    original = fixture.state_next.joint_q.numpy().copy()
    fixture.control.joint_f.fill_(1)
    with test.assertRaisesRegex(ValueError, "Duplicate"):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.005)
    np.testing.assert_array_equal(fixture.state_next.joint_q.numpy(), original)
    test.assertEqual(solver._transaction.step_generation, 0)
    fixture.control.joint_f.zero_()
    fixture.control.joint_target_qd = wp.zeros(1, device=device)
    with test.assertRaisesRegex(ValueError, "array"):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.005)
    # No gain means an undriven DOF still accepts frozen joint_f.
    fixture.model.joint_target_ke.zero_()
    fixture.model.joint_target_kd.zero_()
    terms = MonolithicJointTermsWorkspace(fixture.model, implicit_pd=True)
    control = fixture.model.control()
    control.joint_f.fill_(1)
    terms.snapshot_targets(control)
    base = build_tiny_cpu_fixture(device=device)
    base.model.gravity.zero_()
    pipeline = MonolithicCollisionPipeline(base.model)
    default = SolverMonolithic(base.model, collision_pipeline=pipeline, contact_stiffness=1e5)
    disabled = SolverMonolithic(
        base.model, collision_pipeline=pipeline, contact_stiffness=1e5, joint_terms=SolverMonolithic.JointTerms()
    )
    out = base.model.state()
    default.step(base.state, base.state_next, base.control, None, 0.001)
    disabled.step(base.state, out, base.control, None, 0.001)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd", "particle_q", "particle_qd"):
        np.testing.assert_array_equal(getattr(base.state_next, name).numpy(), getattr(out, name).numpy())


class TestMonolithicJointTerms(unittest.TestCase):
    """G1 scalar joint physics on both devices."""


for _test in (
    test_formula_and_derivatives,
    test_saturation_and_kinks,
    test_frozen_inputs_and_allocation,
    test_invalid_config_and_control,
    test_solver_assembly_and_dense_solve,
    test_trial_transitions_and_input_freeze,
    test_joint_final_transactions,
    test_solver_configuration_and_disabled_parity,
):
    add_function_test(TestMonolithicJointTerms, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
