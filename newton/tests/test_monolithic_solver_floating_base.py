# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Floating-base trees in the coupled solver: composite-rigid dynamics, continuation and coexistence."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.test_monolithic_joint_floating_base import floating_chain, make_solver, rotation_matrix
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contract_oracle import rigid_terms, rotation
from scripts.monolithic_reference.articulation_oracle import skew

GRAVITY = np.array([0.0, 0.0, -9.81])
H = 1e-3
# One step of the welded floating tree against the independent composite-rigid step from the same
# state, from the second step on. The link-level and composite inertial objectives are equal for a
# rigid composite once the link histories are discrete rates of one rigid motion; reset initializes
# them from instantaneous Jacobian rates instead, so the first step differs
# (measured 5.2e-8 in rotation, FP64). Measured from the second step: FP64 4.2e-13 m / 5e-14,
# FP32 1.3e-7 m / 1.4e-7 (P9d PR2); the bounds leave about two orders of margin.
COMPOSITE_TOLERANCE = {"fp64": (1e-10, 1e-11), "fp32": (1e-5, 1e-5)}


def _composite_step(c, r, v, omega, sym, m, inertia, force, h):
    """Independent composite-rigid step: exact COM update and a Newton solve of the rotational equation.

    Like ``independent_step`` of the rigid-actor tests, with a residual criterion relative to the
    rotational inertia scale (a composite's torque residual has roundoff far above 1e-13 N m).
    """
    cp, rp = c + h * v, r + h * (skew(omega) + sym) @ r
    scale = np.trace(inertia) / h**2
    current = r.copy()
    for _ in range(30):
        residual = rigid_terms(cp, current, cp, rp, m, inertia, h, force)[1][3:]
        if np.linalg.norm(residual) <= 1e-12 * scale:
            break
        eps = 1e-6
        jac = np.column_stack(
            [
                (
                    rigid_terms(cp, rotation(eps * axis) @ current, cp, rp, m, inertia, h, force)[1][3:]
                    - rigid_terms(cp, rotation(-eps * axis) @ current, cp, rp, m, inertia, h, force)[1][3:]
                )
                / (2 * eps)
                for axis in np.eye(3)
            ]
        )
        current = rotation(np.linalg.solve(jac, -residual)) @ current
    else:
        raise AssertionError("Composite rotational reference did not converge")
    return cp + h * h * force / m, current


def _composite(run, state):
    """Composite COM, root rotation, COM velocity, angular velocity, symmetric rate, mass and body inertia."""
    k, history = run._kinematics, state._monolithic_coupled
    k.evaluate_state(state)
    mass = k.mass.numpy().astype(np.float64)
    com = k.com.numpy().astype(np.float64)
    rotations = k.rotations.numpy().astype(np.float64)
    inertia_local = run.model.body_inertia.numpy().astype(np.float64)
    total = mass.sum()
    center = mass @ com / total
    velocity = mass @ history.link_v_com.numpy().astype(np.float64) / total
    root = rotation_matrix(history.floating_rotation.numpy()[0].astype(np.float64))
    world = np.zeros((3, 3))
    for m, c, r, i in zip(mass, com, rotations, inertia_local, strict=True):
        d = c - center
        world += r @ i @ r.T + m * (d @ d * np.eye(3) - np.outer(d, d))
    sym = history.link_vsym.numpy()[0].astype(np.float64)
    symmetric = np.array([[sym[0], sym[3], sym[4]], [sym[3], sym[1], sym[5]], [sym[4], sym[5], sym[2]]])
    omega = history.link_omega.numpy()[0].astype(np.float64)
    return center, root, velocity, omega, symmetric, total, root.T @ world @ root


def test_welded_tree_is_a_composite_rigid_body(test, device):
    """A floating root with only FIXED descendants steps as the independent composite rigid body."""
    for precision, (position_tolerance, rotation_tolerance) in COMPOSITE_TOLERANCE.items():
        with test.subTest(precision=precision):
            model = floating_chain(device, fixed_child=True)
            run = make_solver(model, precision)
            state, control = run.create_state(dt=H), run.create_control()
            worst = np.zeros(2)
            run.step(state, state, control, dt=H)
            for _ in range(30):
                c, r, v, omega, sym, m, inertia = _composite(run, state)
                expected = _composite_step(c, r, v, omega, sym, m, inertia, m * GRAVITY, H)
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                actual = _composite(run, state)
                worst = np.maximum(
                    worst,
                    [np.abs(actual[0] - expected[0]).max(), np.abs(actual[1] - expected[1]).max()],
                )
            test.assertLessEqual(worst[0], position_tolerance, f"COM error {worst[0]:.3e} m")
            test.assertLessEqual(worst[1], rotation_tolerance, f"rotation error {worst[1]:.3e}")


def test_checkpoint_continuation(test, device):
    """Save, reload and continue bitwise; floating and fixed-base checkpoints are not interchangeable."""
    model = floating_chain(device)
    run = make_solver(model)
    state, control = run.create_state(dt=H), run.create_control()
    for _ in range(3):
        run.step(state, state, control, dt=H)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "floating.npz"
        run.save_checkpoint(state, path)
        restored = run.create_state(dt=H)
        run.load_checkpoint(restored, path)
        for _ in range(3):
            run.step(state, state, control, dt=H)
            run.step(restored, restored, control, dt=H)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(state, name).numpy(), getattr(restored, name).numpy())
        history, other = state._monolithic_coupled, restored._monolithic_coupled
        for name in ("floating_com", "floating_rotation", "link_v_com", "link_omega", "link_vsym"):
            np.testing.assert_array_equal(getattr(history, name).numpy(), getattr(other, name).numpy())
        test.assertEqual(int(history.schema_version.numpy()[0]), 104)
        fixed = make_solver(_fixed_base_chain(device))
        fixed_state = fixed.create_state(dt=H)
        with test.assertRaisesRegex(ValueError, "Unsupported checkpoint schema version"):
            fixed.load_checkpoint(fixed_state, path)
        fixed_path = Path(folder) / "fixed.npz"
        fixed.save_checkpoint(fixed_state, fixed_path)
        with test.assertRaisesRegex(ValueError, "Unsupported checkpoint schema version"):
            run.load_checkpoint(restored, fixed_path)


def _fixed_base_chain(device):
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(2e-3)))
    child = builder.add_link(mass=0.5, inertia=wp.diag(wp.vec3(1e-3)))
    builder.add_articulation(
        [
            builder.add_joint_revolute(-1, root, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0),
            builder.add_joint_revolute(root, child, axis=(0, 1, 0), limit_ke=0.0, limit_kd=0.0),
        ]
    )
    return builder.finalize(device=device)


def test_schema_versions(test, device):
    """Floating-base schemas are the fixed-base ones plus 100; fixed-base scenes keep theirs."""

    def two_trees(builder):
        base = builder.add_link(mass=0.4, inertia=wp.diag(wp.vec3(1e-3)))
        builder.add_articulation([builder.add_joint_revolute(-1, base, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0)])

    for edit, version in ((None, 104), (two_trees, 107)):
        with test.subTest(version=version):
            run = make_solver(floating_chain(device, edit=edit))
            test.assertEqual(int(run.create_state(dt=H)._monolithic_coupled.schema_version.numpy()[0]), version)
    fixed = make_solver(_fixed_base_chain(device))
    test.assertEqual(int(fixed.create_state(dt=H)._monolithic_coupled.schema_version.numpy()[0]), 4)
    test.assertNotIn("floating_com", fixed._states.schema)


def test_coexists_with_fixed_trees_and_rigid_bodies(test, device):
    """A floating tree, a fixed-base tree and a FREE rigid body step together; the rigid body falls freely."""
    labels = {}

    def add_others(builder):
        base = builder.add_link(mass=0.4, inertia=wp.diag(wp.vec3(1e-3)))
        builder.add_articulation([builder.add_joint_revolute(-1, base, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0)])
        pose = wp.transform((1.0, 0.0, 2.0), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=0.3, inertia=wp.diag(wp.vec3(1e-3)), label="loose")
        builder.add_articulation([builder.add_joint_free(body, parent_xform=pose)])
        labels["loose"] = body

    model = floating_chain(device, edit=add_others)
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            run = make_solver(model, precision)
            test.assertEqual([a.kind for a in run._actors.actors], ["articulation", "articulation", "rigid"])
            state, control = run.create_state(dt=H), run.create_control()
            z0 = float(state._monolithic_coupled.rigid_com.numpy()[0, 2])
            steps = 20
            for _ in range(steps):
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
            # Implicit Euler free fall from rest: z_n = z_0 - g h^2 n (n + 1) / 2.
            expected = z0 + GRAVITY[2] * H * H * steps * (steps + 1) / 2
            # The solver stops at its residual tolerance, not at the linear law's exact root.
            tolerance = 1e-9 if precision == "fp64" else 1e-5
            test.assertAlmostEqual(float(state._monolithic_coupled.rigid_com.numpy()[0, 2]), expected, delta=tolerance)


def test_predicted_point_outside_domain_restarts_from_step_start(test, device):
    """With a floating root, a predicted start outside the local domain falls back to the whole step
    start (native root pose and increments included) and converges to the same step."""
    from unittest.mock import patch  # noqa: PLC0415

    model = floating_chain(device)
    reference, fallback = make_solver(model), make_solver(model)
    states = []
    for run in (reference, fallback):
        state, control = run.create_state(dt=H), run.create_control()
        run.step(state, state, control, dt=H)
        states.append((state, control))
    reference.step(states[0][0], states[0][0], states[0][1], dt=H)
    original = fallback._domain
    calls = []

    def first_outside(state):
        calls.append(state)
        return False if len(calls) == 1 else original(state)

    iterate, starts = fallback._iterate_articulation, []

    def record_start(dt):
        # The iteration starts from the restored step start: native root pose and zero increments.
        history = fallback._accepted._monolithic_coupled
        steps = fallback._root_steps(fallback._accepted)
        starts.append(
            (
                history.floating_com.numpy().copy(),
                history.floating_rotation.numpy().copy(),
                steps[0].numpy().copy(),
                steps[1].numpy().copy(),
                fallback._original._monolithic_coupled.floating_com.numpy().copy(),
                fallback._original._monolithic_coupled.floating_rotation.numpy().copy(),
            )
        )
        return iterate(dt)

    with (
        patch.object(fallback, "_domain", side_effect=first_outside),
        patch.object(fallback, "_iterate_articulation", side_effect=record_start),
    ):
        fallback.step(states[1][0], states[1][0], states[1][1], dt=H)
    test.assertTrue(fallback.last_stats.converged, fallback.last_stats.failure_reason)
    com, rot, step_com, step_rot, start_com, start_rot = starts[0]
    np.testing.assert_array_equal(com, start_com)
    np.testing.assert_array_equal(rot, start_rot)
    np.testing.assert_array_equal(step_com, 0)
    np.testing.assert_array_equal(step_rot, [[0.0, 0.0, 0.0, 1.0]])
    test.assertGreater(len(calls), 1)
    a, b = states[0][0]._monolithic_coupled, states[1][0]._monolithic_coupled
    np.testing.assert_allclose(b.floating_com.numpy(), a.floating_com.numpy(), atol=1e-10)
    np.testing.assert_allclose(b.floating_rotation.numpy(), a.floating_rotation.numpy(), atol=1e-10)
    np.testing.assert_allclose(states[1][0].joint_q.numpy(), states[0][0].joint_q.numpy(), atol=1e-9)


class TestSolverFloatingBase(unittest.TestCase):
    pass


for _test in (
    test_welded_tree_is_a_composite_rigid_body,
    test_checkpoint_continuation,
    test_schema_versions,
    test_coexists_with_fixed_trees_and_rigid_bodies,
    test_predicted_point_outside_domain_restarts_from_step_start,
):
    add_function_test(TestSolverFloatingBase, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
