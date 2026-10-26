# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the private PR2 FP64 kinematics and owned integration state."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import (
    ArticulationKinematics,
    ArticulationObjective,
    _joint_variation,
)
from newton._src.solvers.monolithic.articulation_state import ArticulationState, _predict, _recover_step
from newton._src.solvers.monolithic.coupled_kernels import _coordinate_trial
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import Kinematics, joint_terms, skew


def make_tree(device, *, coupled=False):
    builder = newton.ModelBuilder()
    links = [
        builder.add_link(
            mass=1.0, com=(0.05, 0.02, -0.1), inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
        )
        for _ in range(3)
    ]
    joints = [
        builder.add_joint_revolute(-1, links[0], axis=(0.0, 1.0, 0.0)),
        builder.add_joint_revolute(
            links[0], links[1], axis=(1.0, 1.0, 0.0), parent_xform=wp.transform((0.1, 0.0, -0.4), wp.quat_identity())
        ),
        builder.add_joint_prismatic(links[1], links[2], axis=(0.0, 0.0, 1.0)),
    ]
    builder.add_articulation(joints)
    builder.joint_q[:] = [0.2, -0.15, 0.03]
    builder.joint_qd[:] = [0.1, -0.05, 0.02]
    if coupled:
        add_contact_free_tet(builder)
    return builder.finalize(device=device)


def test_kinematics(test, device):
    """Compare noncollinear FP64 FK and Jacobians with independent matrix products."""
    model = make_tree(device)
    k = ArticulationKinematics(model)
    oracle = Kinematics(model)
    q = wp.array([0.2, -0.15, 0.03], dtype=wp.float64, device=device)
    k.evaluate(q)
    t, r, jt, jr, _, _ = oracle.evaluate(q.numpy())
    np.testing.assert_allclose(k.com.numpy(), t, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(k.rotations.numpy(), r, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(k.jt.numpy(), jt, atol=2e-15, rtol=2e-14)
    jw = k.jw.numpy()
    for b in range(k.bodies):
        for i in range(k.count):
            np.testing.assert_allclose(skew(jw[b, i]) @ r[b], jr[b, i], atol=2e-15, rtol=2e-14)
    q0 = q.numpy()
    h = 1e-5
    for i in range(k.count):
        direction = np.eye(k.count)[i]
        plus, minus = oracle.evaluate(q0 + h * direction), oracle.evaluate(q0 - h * direction)
        np.testing.assert_allclose((plus[0] - minus[0]) / (2 * h), jt[:, i], atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(
            (plus[2] - minus[2]) / (2 * h), oracle.evaluate(q0)[4][:, :, i], atol=1e-10, rtol=1e-8
        )
    q.assign(q0 + 1e-10)
    k.evaluate(q)
    test.assertGreater(np.max(abs(k.com.numpy() - t)), 1e-12)


def test_state_history(test, device):
    """Preserve complete FP64 history through assign, reset and finite differences."""
    k = ArticulationKinematics(make_tree(device))
    states = ArticulationState(k)
    first = states.create(dt=0.001)
    history = states.validate(first)
    copy = states.create(dt=0.001)
    copy.assign(first)
    for name in states.schema:
        np.testing.assert_array_equal(
            getattr(copy._monolithic_articulation, name).numpy(), getattr(history, name).numpy()
        )
    k.evaluate(first.joint_q)
    old_t, old_r = k.com.numpy(), k.rotations.numpy()
    new_q = first.joint_q.numpy() + first.joint_qd.numpy() * 0.001
    copy.joint_q.assign(new_q)
    with test.assertRaisesRegex(ValueError, "explicitly reset"):
        states.validate(copy)
    k.evaluate(copy.joint_q)
    # Recover the rates from the step increments c - c_n and R R_n^T - I.
    steps = [
        wp.array(k.com.numpy() - old_t, dtype=wp.vec3d, device=device),
        wp.array(k.rotations.numpy() @ old_r.transpose(0, 2, 1) - np.eye(3), dtype=wp.mat33d, device=device),
    ]
    wp.launch(
        _recover_step,
        k.bodies,
        [*steps, wp.float64(0.001), history.link_v_com, history.link_omega, history.link_vsym],
        device=device,
    )
    predicted_t, predicted_r = wp.empty_like(k.com), wp.empty_like(k.rotations)
    step_t, step_r = wp.empty_like(k.com), wp.empty_like(k.rotations)
    wp.launch(
        _predict,
        k.bodies,
        [
            k.com,
            k.rotations,
            history.link_v_com,
            history.link_omega,
            history.link_vsym,
            wp.float64(0.001),
            predicted_t,
            predicted_r,
            step_t,
            step_r,
        ],
        device=device,
    )
    np.testing.assert_allclose(predicted_t.numpy(), 2 * k.com.numpy() - old_t, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(predicted_r.numpy(), 2 * k.rotations.numpy() - old_r, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(step_t.numpy(), k.com.numpy() - old_t, atol=2e-15, rtol=2e-14)
    # Initialization is explicitly different from continuation.
    states.reset(copy, dt=0.002)
    states.validate(copy)
    test.assertEqual(copy._monolithic_articulation.last_dt.numpy()[0], 0.002)
    copy.joint_qd.fill_(1000.0)
    with test.assertRaisesRegex(ValueError, "angular velocity"):
        states.reset(copy, dt=0.001)


def test_configuration(test, device):
    """Reject both in-place and array-replacement configuration mutations."""
    model = make_tree(device)
    k = ArticulationKinematics(model)
    model.body_mass.fill_(2.0)
    with test.assertRaisesRegex(ValueError, "configuration changed"):
        k.validate_identity()
    model.body_mass.assign(k.values["body_mass"])
    k.validate_identity()
    model.body_mass = wp.clone(model.body_mass)
    with test.assertRaisesRegex(ValueError, "configuration changed"):
        k.validate_identity()


def test_objective(test, device):
    """Verify complete energy/residual derivatives and distinguish B from exact H."""
    model = make_tree(device)
    model.joint_armature.fill_(0.002)
    model.joint_damping.fill_(0.003)
    k = ArticulationKinematics(model)
    oracle = Kinematics(model)
    params = (
        np.tile([0.5, 0.02, 0.03, -0.25, 0.25, 10.0, 0.05, 0.02, 0.001, 0.02], (3, 1)).astype(np.float32).astype(float)
    )
    objective = ArticulationObjective(k, params)
    h = 0.01
    qn = np.array([0.252, -0.26, 0.02])
    vn = np.array([0.1, -0.2, 0.03])
    objective.begin(wp.array(qn, dtype=wp.float64, device=device))
    objective.vn.assign(vn)
    objective.qt.assign([0.3, -0.3, 0.05])
    objective.vt.assign([0.1, -0.1, 0.02])
    objective.joint_force.assign([0.001, 0, -0.002])
    t, r, jt, jr, _, _ = oracle.evaluate(qn)
    tp = t + h * np.einsum("bni,n->bi", jt, vn)
    rp = r + h * np.einsum("bnij,n->bij", jr, vn)
    objective.predicted_t.assign(tp)
    objective.predicted_r.assign(rp)
    objective.step_t.assign(tp - t)
    objective.step_r.assign((rp - r) @ r.transpose(0, 2, 1))
    force = np.tile([0.2, -0.1, -9.81], (3, 1))
    objective.force.assign(force)
    array = wp.empty(3, dtype=wp.float64, device=device)

    def independent(q):
        e, g, exact = oracle.inertia(q, tp, rp, h, force)
        ej, gj, hj = joint_terms(
            q,
            qn,
            vn,
            h,
            params,
            objective.armature.numpy(),
            objective.damping.numpy(),
            objective.qt.numpy(),
            objective.vt.numpy(),
            objective.joint_force.numpy(),
        )
        return e + np.sum(force * t) + ej.sum(), g + gj, exact + np.diag(hj)

    for q in (
        qn + np.array([0.001, -0.001, 0.0002]),
        qn + np.array([-0.001, 0.001, -0.0002]),
        np.array([0.28, -0.28, 0.03]),
    ):
        array.assign(q)
        objective.evaluate(array, h)
        e, g, exact = independent(q)
        np.testing.assert_allclose(
            objective.link_energy.numpy().sum() + objective.joint_energy.numpy().sum(), e, rtol=1e-10, atol=1e-11
        )
        np.testing.assert_allclose(objective.residual.numpy(), g, rtol=1e-10, atol=1e-10)
        b = objective.tangent.numpy()
        np.testing.assert_allclose(b, b.T, atol=1e-10, rtol=1e-13)
        test.assertGreater(np.linalg.eigvalsh(b).min(), 0)
        test.assertGreater(np.linalg.norm(exact - b), 1e-3)
        for eps in (1e-6, 3e-7):
            for i in range(3):
                d = np.eye(3)[i] * eps
                plus, minus = independent(q + d), independent(q - d)
                np.testing.assert_allclose((plus[0] - minus[0]) / (2 * eps), g[i], rtol=5e-5, atol=1e-6)
                np.testing.assert_allclose((plus[1] - minus[1]) / (2 * eps), exact[:, i], rtol=5e-5, atol=1e-6)
        # Differentiate the production energy through the real mixed-precision
        # trial kernel using its effective stored displacement.
        base = wp.array(q - qn, dtype=wp.float64, device=device)
        trial, velocity, step = wp.empty_like(base), wp.empty_like(base), wp.empty_like(base)
        frozen_b = objective.tangent.numpy().copy()
        for eps in (1e-5, 3e-6):
            for i in range(3):
                direction = wp.array(np.eye(3)[i], dtype=wp.float32, device=device)
                energies, coordinates = [], []
                for sign in (-1, 1):
                    wp.launch(
                        _coordinate_trial,
                        3,
                        [
                            base,
                            direction,
                            k.dof_map,
                            wp.float64(sign * eps),
                            objective.qn,
                            wp.float64(h),
                            k.coord_map,
                            k.dof_map,
                            trial,
                            velocity,
                            step,
                        ],
                        device=device,
                    )
                    objective.evaluate(trial, h, build_tangent=False, step=step)
                    energies.append(float(objective.link_energy.numpy().sum() + objective.joint_energy.numpy().sum()))
                    coordinates.append(float(step.numpy()[i]))
                derivative = (energies[1] - energies[0]) / (coordinates[1] - coordinates[0])
                np.testing.assert_allclose(derivative, g[i], rtol=5e-3, atol=1e-7)
                np.testing.assert_array_equal(objective.tangent.numpy(), frozen_b)


def test_scalar_switches(test, device):
    """Check all scalar switch sides and one-sided energy slopes independently."""
    h = 0.01
    params = np.array([[0.5, 0.02, 0.03, -0.25, 0.25, 10, 0.05, 0.02, 0.001, 0.02]], dtype=np.float32).astype(float)
    saturation = params[0, 2] / (params[0, 0] + params[0, 1] / h)
    q = wp.zeros(1, dtype=wp.float64, device=device)
    qn, vn, qt, vt, force = [wp.zeros_like(q) for _ in range(5)]
    armature, damping = [wp.full(1, x, dtype=wp.float64, device=device) for x in (0.002, 0.003)]
    energy, residual, absolute, tangent = [wp.zeros_like(q) for _ in range(4)]
    parameters = wp.array(params, dtype=wp.float64, device=device)

    def evaluate(value):
        q.fill_(value - qn.numpy()[0])
        wp.launch(
            _joint_variation,
            1,
            [
                q,
                qn,
                vn,
                parameters,
                armature,
                damping,
                qt,
                vt,
                force,
                wp.float64(h),
                energy,
                residual,
                absolute,
                tangent,
            ],
            device=device,
        )
        # The summed term magnitudes bound the joint residual.
        test.assertGreaterEqual(float(absolute.numpy()[0]), abs(float(residual.numpy()[0])))
        return tuple(float(v.numpy()[0]) for v in (energy, residual, tangent))

    for initial, switches in (
        (0, (-saturation, saturation, -0.25, 0.25, -0.27, 0.27, 0)),
        (0.26, (0.26,)),
        (-0.26, (-0.26,)),
    ):
        qn.fill_(initial)
        for switch in switches:
            for side in (-1, 1):
                center = switch + side * 1e-6
                actual = evaluate(center)
                expected = joint_terms(
                    np.array([center]),
                    np.array([initial]),
                    np.zeros(1),
                    h,
                    params,
                    armature.numpy(),
                    damping.numpy(),
                    np.zeros(1),
                    np.zeros(1),
                    np.zeros(1),
                )
                np.testing.assert_allclose(actual, np.asarray(expected)[:, 0], rtol=1e-10, atol=1e-11)
                for eps in (3e-6, 1e-6):
                    endpoint = center + side * eps
                    end = evaluate(endpoint)
                    midpoint = evaluate((center + endpoint) / 2)
                    np.testing.assert_allclose(
                        (end[0] - actual[0]) / (endpoint - center), midpoint[1], rtol=5e-3, atol=1e-7
                    )


def test_force_step(test, device):
    """Keep current residual immutable while trial scratch changes on CPU/CUDA."""
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01))
    joint = builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, limit_ke=0.0, limit_kd=0.0)
    builder.add_articulation([joint])
    add_contact_free_tet(builder)
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        mass_mode="lumped",
        pcg_mode="production" if device.is_cuda else "diagnostic",
        use_cuda_graph=device.is_cuda,
    )
    source = solver.create_state(dt=0.001)
    target = solver.create_state(dt=0.001)
    source.body_f.assign([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with patch.object(solver._states, "create", side_effect=AssertionError("continuation must not reinitialize")):
        solver.step(source, target, model.control(), dt=0.001)
    test.assertTrue(solver.last_stats.converged)
    trial = solver.last_stats.variational["trials"][0]
    test.assertTrue(trial["Armijo"])
    test.assertAlmostEqual(trial["slope"], -1e-6, delta=1e-12)
    np.testing.assert_allclose(target.joint_q.numpy(), [1e-6], rtol=1e-7)
    np.testing.assert_allclose(target.joint_qd.numpy(), [0.001], rtol=1e-7)
    solver._states.validate(target)
    np.testing.assert_array_equal(source.joint_q.numpy(), [0.0])


def test_schema_and_publication(test, device):
    """Reject malformed standard arrays and restore a partially published state."""
    model = make_tree(device, coupled=True)
    for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
        getattr(model, name).zero_()
    model.gravity.zero_()
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        mass_mode="lumped",
    )
    source = solver.create_state(dt=0.001)
    target = solver.create_state(dt=0.001)
    body = target.body_q
    target.body_q = wp.zeros(1, dtype=wp.transform, device=device)
    with test.assertRaisesRegex(ValueError, "body_q"):
        solver.step(source, target, model.control(), dt=0.001)
    target.body_q = body
    target.extra = wp.zeros(1, dtype=float, device=device)
    with test.assertRaisesRegex(ValueError, "Unsupported"):
        solver.step(source, target, model.control(), dt=0.001)
    del target.extra
    original_q = source.joint_q.numpy().copy()
    original_history = {
        name: getattr(source._monolithic_coupled, name).numpy().copy() for name in solver._states.schema
    }

    assign = target.assign

    def broken_publish(other):
        if other is not solver._accepted:
            return assign(other)
        wp.copy(target.joint_q, other.joint_q)
        target._monolithic_coupled.link_vsym.fill_(123.0)
        raise RuntimeError("injected publication failure")

    with patch.object(target, "assign", side_effect=broken_publish):
        with test.assertRaisesRegex(RuntimeError, "injected publication"):
            solver.step(source, target, model.control(), dt=0.001)
    test.assertEqual(solver.last_stats.status, solver.Status.NONFINITE)
    test.assertTrue(solver.last_stats.rolled_back)
    for state in (source, target):
        np.testing.assert_array_equal(state.joint_q.numpy(), original_q)
        for name, values in original_history.items():
            np.testing.assert_array_equal(getattr(state._monolithic_coupled, name).numpy(), values)
        solver._states.validate(state)


def test_fork_alias_and_variable_dt(test, device):
    """Continue complete copied histories across alias steps and changing dt."""
    model = make_tree(device, coupled=True)
    for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
        getattr(model, name).zero_()
    model.gravity.zero_()
    options = {
        "pcg_mode": "production" if device.is_cuda else "diagnostic",
        "use_cuda_graph": device.is_cuda,
    }
    solver = SolverMonolithic(model, friction_velocity_scale=0.01, mass_mode="lumped", **options)
    other = SolverMonolithic(model, friction_velocity_scale=0.01, mass_mode="lumped", **options)
    state = solver.create_state(dt=0.001)
    alias = other.create_state(dt=0.001)
    alias.assign(state)
    out = solver.create_state(dt=0.001)
    control = model.control()
    for dt in (0.001, 0.0005, 0.00025, 0.001):
        solver.step(state, out, control, dt=dt)
        other.step(alias, alias, control, dt=dt)
        test.assertTrue(solver.last_stats.converged)
        test.assertTrue(other.last_stats.converged)
        np.testing.assert_array_equal(out.joint_q.numpy(), alias.joint_q.numpy())
        np.testing.assert_array_equal(out.joint_qd.numpy(), alias.joint_qd.numpy())
        for name in solver._states.schema:
            np.testing.assert_array_equal(
                getattr(out._monolithic_coupled, name).numpy(),
                getattr(alias._monolithic_coupled, name).numpy(),
            )
        state, out = out, state
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        solver.save_checkpoint(state, path)
        restored = solver.create_state(dt=0.001)
        solver.load_checkpoint(restored, path)
        for name in solver._states.schema:
            np.testing.assert_array_equal(
                getattr(restored._monolithic_coupled, name).numpy(),
                getattr(state._monolithic_coupled, name).numpy(),
            )
        np.savez(path, joint_q=state.joint_q.numpy(), joint_qd=state.joint_qd.numpy())
        with test.assertRaisesRegex(ValueError, "Missing checkpoint schema version"):
            solver.load_checkpoint(restored, path)
    # A factory source is initialization; assign is the exact continuation operation.
    reset = solver.create_state(dt=0.001, source=state)
    test.assertGreater(
        np.linalg.norm(reset._monolithic_coupled.link_vsym.numpy() - state._monolithic_coupled.link_vsym.numpy()),
        0.0,
    )
    saved = {name: getattr(alias._monolithic_coupled, name).numpy().copy() for name in solver._states.schema}
    q = alias.joint_q.numpy().copy()
    with patch.object(other, "_iterate_articulation", return_value=other.Status.NONLINEAR_MAX_ITERATIONS):
        other.step(alias, alias, control, dt=0.001)
    test.assertTrue(other.last_stats.rolled_back)
    test.assertFalse(other.last_stats.converged)
    np.testing.assert_array_equal(alias.joint_q.numpy(), q)
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(alias._monolithic_coupled, name).numpy(), value)


def test_continuous_inertia(test, device):
    """Check independent convective inertia against the kinetic-energy derivatives."""
    oracle = Kinematics(make_tree(device))
    q = np.array([0.3, -0.4, 0.02])
    v = np.array([0.2, -0.1, 0.04])
    mass, bias = oracle.mass_and_bias(q, v)
    np.testing.assert_allclose(mass, mass.T, atol=1e-14)
    test.assertGreater(np.linalg.eigvalsh(mass).min(), 0.0)
    for epsilon in (1e-5, 3e-6):
        derivative = np.array(
            [
                (oracle.mass_and_bias(q + epsilon * d, v)[0] - oracle.mass_and_bias(q - epsilon * d, v)[0])
                / (2 * epsilon)
                for d in np.eye(3)
            ]
        )
        expected = np.einsum("kij,j,k->i", derivative, v, v) - 0.5 * np.einsum("ijk,j,k->i", derivative, v, v)
        np.testing.assert_allclose(bias, expected, rtol=1e-7, atol=1e-11)


class TestArticulationVariational(unittest.TestCase):
    pass


for _name in (
    "kinematics",
    "state_history",
    "configuration",
    "objective",
    "force_step",
    "schema_and_publication",
    "fork_alias_and_variable_dt",
    "continuous_inertia",
    "scalar_switches",
):
    add_function_test(
        TestArticulationVariational, "test_" + _name, globals()["test_" + _name], devices=get_test_devices()
    )

if __name__ == "__main__":
    unittest.main()
