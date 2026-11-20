# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Floating-root 6D pose drive and root wrench: energy, gradient and tangent, convergence, rejections."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.floating import _drive_terms
from newton.tests.test_monolithic_joint_floating_base import floating_chain, make_solver, root_offsets, rotation_matrix
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices
from scripts.monolithic_reference.actor_contract_oracle import rotation

H = 1e-3
PRECISIONS = (("fp64", np.float64, wp.float64), ("fp32", np.float32, wp.float32))


def _quat(matrix):
    m = matrix
    w = np.sqrt(max(1.0 + np.trace(m), 1e-30)) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])


class _Drive:
    """One root's drive and wrench terms through the production kernel, in NumPy float64 in and out."""

    def __init__(self, device, dtype, params, error_t, error_r, target_v, target_w, force, torque):
        self.device, self.dtype = device, dtype
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.quaternion = wp.quatd if dtype == wp.float64 else wp.quat

        def vec(value):
            return wp.array([value], dtype=self.vector, device=device)

        self.params = wp.array(np.asarray(params).reshape(1, 6), dtype=dtype, device=device)
        self.inputs = [vec(v) for v in (error_t, error_r, target_v, target_w, force, torque)]
        self.rows = wp.array(np.arange(6, dtype=np.int32).reshape(1, 6), dtype=int, device=device)
        self.entries = wp.array(np.arange(36, dtype=np.int32).reshape(1, 36), dtype=int, device=device)

    def __call__(self, dc, quaternion):
        energy = wp.zeros(1, dtype=self.dtype, device=self.device)
        residual = wp.zeros(6, dtype=self.dtype, device=self.device)
        absolute = wp.zeros(6, dtype=self.dtype, device=self.device)
        tangent = wp.zeros(36, dtype=self.dtype, device=self.device)
        wp.launch(
            _drive_terms,
            1,
            [
                wp.array([dc], dtype=self.vector, device=self.device),
                wp.array([quaternion], dtype=self.quaternion, device=self.device),
                self.params,
                *self.inputs,
                self.rows,
                self.entries,
                self.dtype(H),
                True,
                energy,
                residual,
                absolute,
                tangent,
            ],
            device=self.device,
        )
        return (
            float(energy.numpy()[0]),
            residual.numpy().astype(np.float64),
            tangent.numpy().astype(np.float64).reshape(6, 6),
        )


def _cases():
    """(name, params, error_t, error_r, rates, wrench, dc, phi): inside, saturated and wrench cases."""
    params = (2000.0, 40.0, 1e6, 50.0, 2.0, 1e6)
    return (
        (
            "inside",
            params,
            (0.01, -0.02, 0.005),
            (0.2, -0.1, 0.3),
            True,
            False,
            (1e-4, -2e-4, 3e-4),
            (2e-3, -1e-3, 4e-3),
        ),
        (
            "saturated",
            (2000.0, 40.0, 5.0, 50.0, 2.0, 0.5),
            (0.03, -0.02, 0.01),
            (1.2, 0.8, -0.5),
            True,
            False,
            (2e-4, 1e-4, -1e-4),
            (3e-3, 2e-3, -1e-3),
        ),
        ("wrench", params, (0.01, 0.0, 0.0), (0.1, 0.0, 0.0), False, True, (1e-4, 1e-4, 1e-4), (1e-3, 2e-3, 3e-3)),
    )


def test_energy_gradient_tangent(test, device):
    """The residual is the energy's derivative in (dc, left rotation); the tangent its Gauss-Newton derivative."""
    for name, params, error_t, error_r, rates, wrench, dc, phi in _cases():
        with test.subTest(case=name):
            target = ((0.05, 0.0, -0.02), (0.1, -0.2, 0.05)) if rates else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            loads = ((1.5, -2.0, 0.5), (0.02, -0.01, 0.03)) if wrench else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            drive = _Drive(device, wp.float64, params, error_t, error_r, *target, *loads)
            dc_value, phi_value = np.asarray(dc), np.asarray(phi)
            base = _quat(rotation(phi_value))
            energy, residual, tangent = drive(dc_value, base)
            eps = 1e-6
            gradient, hessian = np.zeros(6), np.zeros((6, 6))
            for k in range(6):
                step = np.eye(3)[k % 3] * eps
                if k < 3:
                    plus, minus = drive(dc_value + step, base), drive(dc_value - step, base)
                else:
                    plus = drive(dc_value, _quat(rotation(step) @ rotation(phi_value)))
                    minus = drive(dc_value, _quat(rotation(-step) @ rotation(phi_value)))
                gradient[k] = (plus[0] - minus[0]) / (2 * eps)
                hessian[:, k] = (plus[1] - minus[1]) / (2 * eps)
            atol = 1e-6 * (1 + abs(residual).max())
            np.testing.assert_allclose(residual[:3], gradient[:3], rtol=1e-6, atol=atol)
            # A root torque is not a potential away from phi = 0 (as for rigid actors): its residual is
            # -tau while the energy's slope differs by O(|tau| |phi|); the solver then treats the energy
            # as heuristic (_conservative_energy).
            torque = np.linalg.norm(loads[1]) * np.linalg.norm(phi_value)
            np.testing.assert_allclose(residual[3:], gradient[3:], rtol=1e-6, atol=atol + 2 * torque)
            np.testing.assert_allclose(tangent[:3, :3], hessian[:3, :3], rtol=1e-5, atol=1e-5 * abs(hessian).max())
            # The derivative of a left-coordinate residual field adds the skew part [g]x / 2; the symmetric
            # part is the Hessian, which the Gauss-Newton block matches up to O(|g| |phi|).
            symmetric = (hessian[3:, 3:] + hessian[3:, 3:].T) / 2
            g = np.linalg.norm(residual[3:]) + 1
            np.testing.assert_allclose(tangent[3:, 3:], symmetric, atol=2 * g * np.linalg.norm(phi_value) + 1e-4)
            test.assertTrue(np.all(np.linalg.eigvalsh((tangent + tangent.T) / 2) >= -1e-9))
            test.assertTrue(np.isfinite(energy))


def test_truncation_is_c1_and_bounded(test, device):
    """Across the force limit the energy is continuous and its one-sided slopes agree (C1); beyond it the
    force norm equals the limit. Steps cross the threshold: x0 sits exactly on |u| = limit."""
    limit = 5.0
    params = (2000.0, 0.0, limit, 0.0, 0.0, 1.0)
    drive = _Drive(
        device, wp.float64, params, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    )
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    direction = np.array([0.6, -0.8, 0.0])
    x0 = limit / 2000.0 * direction
    eps = 1e-6 * np.linalg.norm(x0)
    below, at, above = (drive(x0 + side * eps * direction, identity)[0] for side in (-1.0, 0.0, 1.0))
    left, right = (at - below) / eps, (above - at) / eps
    # A missing -limit/2 offset in the saturated branch would jump by limit^2 / (2 a) = 6.25e-3 here.
    test.assertLess(abs(above - at), 2 * limit * eps)
    test.assertLess(abs(at - below), 2 * limit * eps)
    test.assertAlmostEqual(left, right, delta=1e-3 * limit)
    test.assertAlmostEqual(left, limit, delta=1e-3 * limit)
    for scale in (2.0, 10.0):
        _, residual, _ = drive(scale * x0, identity)
        test.assertAlmostEqual(float(np.linalg.norm(residual[:3])), limit, places=9)


def test_left_jacobian_branches(test, device):
    """The closed-form J_l^-1 branch (|phi| > 1e-2) and its series below match finite differences and each other
    at the switch; FP32 matches FP64 on the same inputs."""
    params = (2000.0, 40.0, 1e6, 50.0, 2.0, 1e6)
    args = ((0.01, -0.02, 0.005), (0.2, -0.1, 0.3), (0.05, 0.0, -0.02), (0.1, -0.2, 0.05), (0, 0, 0), (0, 0, 0))
    drives = {dtype: _Drive(device, dtype, params, *args) for dtype in (wp.float64, wp.float32)}
    axis = np.array([0.48, -0.6, 0.64])
    dc = np.array([1e-4, -2e-4, 3e-4])
    for angle in (0.05, 1.0):
        with test.subTest(angle=angle):
            phi = angle * axis
            base = _quat(rotation(phi))
            _energy, residual, _ = drives[wp.float64](dc, base)
            eps = 1e-6
            for k in range(3):
                step = np.eye(3)[k] * eps
                plus = drives[wp.float64](dc, _quat(rotation(step) @ rotation(phi)))[0]
                minus = drives[wp.float64](dc, _quat(rotation(-step) @ rotation(phi)))[0]
                test.assertAlmostEqual(
                    (plus - minus) / (2 * eps), residual[3 + k], delta=1e-6 * (1 + abs(residual).max())
                )
            single = drives[wp.float32](dc.astype(np.float32), base.astype(np.float32))[1]
            np.testing.assert_allclose(single, residual, rtol=2e-5, atol=2e-5 * abs(residual).max())
    # Continuity at the series/closed-form switch.
    sides = [drives[wp.float64](dc, _quat(rotation((1e-2 + d) * axis)))[1] for d in (-1e-9, 1e-9)]
    np.testing.assert_allclose(sides[0], sides[1], rtol=1e-6, atol=1e-6)


def _drive_terms_options():
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    return {"joint_terms": SolverMonolithic.JointTerms(implicit_pd=True)}


def driven_chain(
    device,
    *,
    kt=400.0,
    dt=40.0,
    kr=2.0,
    dr=0.3,
    force=1e6,
    torque=1e6,
    gravity=(0.0, 0.0, 0.0),
    fixed_child=False,
    prefix=None,
):
    """The floating chain with an isotropic root drive (POSITION_VELOCITY), started at rest."""
    model = floating_chain(device, gravity=gravity, fixed_child=fixed_child, prefix=prefix)
    _, _, d, _ = root_offsets(model)
    rates = model.joint_qd.numpy()
    rates[:] = 0.0
    model.joint_qd.assign(rates)
    for name, values in (
        ("joint_target_ke", (kt, kt, kt, kr, kr, kr)),
        ("joint_target_kd", (dt, dt, dt, dr, dr, dr)),
        ("joint_effort_limit", (force, force, force, torque, torque, torque)),
    ):
        array = getattr(model, name).numpy()
        array[d : d + 6] = values
        getattr(model, name).assign(array)
    modes = model.joint_target_mode.numpy()
    modes[d : d + 6] = int(newton.JointTargetMode.POSITION_VELOCITY)
    model.joint_target_mode.assign(modes)
    return model


def _set_root_target(control, model, rotation_vector, offset=(0.0, 0.0, 0.0)):
    """Target the root at its initial pose turned by a world rotation about its joint frame, then shifted."""
    joint, c, _, t = root_offsets(model)
    q = model.joint_q.numpy().astype(np.float64)
    targets = control.joint_target_q.numpy()
    turn = rotation(np.asarray(rotation_vector, dtype=np.float64))
    # The anchor rotation maps joint-frame rotations to world: R_world = R_p R_joint.
    anchor = model.joint_X_p.numpy()[joint].astype(np.float64)
    parent = rotation_matrix(anchor[3:])
    joint_rotation = parent.T @ turn @ parent @ rotation_matrix(q[c + 3 : c + 7])
    targets[t + 3 : t + 7] = _quat(joint_rotation)
    targets[t : t + 3] = q[c : c + 3] + parent.T @ np.asarray(offset)
    control.joint_target_q.assign(targets)
    return turn


def _rotation_error(run, state, target):
    rotation_now = rotation_matrix(state._monolithic_coupled.floating_rotation.numpy()[0].astype(np.float64))
    relative = rotation_now @ target.T
    return float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)))


def test_converges_from_far_rotations(test, device):
    """From 120 and 179 degrees away the drive pulls a welded tree to its target (R1): every step converges,
    the error decays monotonically (sampled every 100 steps) and ends below 2e-3 of its start.

    Near-critical damping (k_r 2, d_r 0.3 on about 1e-2 kg m^2). Measured in FP64 after 1000 steps:
    1.7e-3 rad from 120 degrees, 2.4e-3 rad from 179 degrees (about 1.5 Newton iterations per step).
    """
    for precision, _numpy_dtype, _dtype in PRECISIONS:
        for degrees in (120.0, 179.0):
            with test.subTest(precision=precision, degrees=degrees):
                model = driven_chain(device, fixed_child=True)
                run = make_solver(model, precision, **_drive_terms_options())
                state, control = run.create_state(dt=H), run.create_control()
                start = rotation_matrix(state._monolithic_coupled.floating_rotation.numpy()[0].astype(np.float64))
                turn = _set_root_target(control, model, np.radians(degrees) * np.array([0.0, 0.6, 0.8]))
                target = turn @ start
                errors = []
                for _ in range(1000):
                    run.step(state, state, control, dt=H)
                    test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                    errors.append(_rotation_error(run, state, target))
                samples = [*errors[::100], errors[-1]]
                test.assertTrue(all(b < a for a, b in itertools.pairwise(samples)), samples)
                test.assertLess(errors[-1], 2e-3 * np.radians(degrees), samples)


def test_problem_graphs_recapture_after_root_storage_replacement(test, device):
    """Same-layout replacements of the root drive's storage recapture the problem graphs: the owners graph
    reads the new arrays (the old ones hold NaN, so a stale replay fails), and graph and eager stay equal."""
    names = ("params", "error_t", "error_r", "target_v", "target_w")
    for precision, _numpy_dtype, _dtype in PRECISIONS:
        results = {}
        for graph in (False, True):
            with test.subTest(precision=precision, graph=graph):
                model = driven_chain(device, fixed_child=True)
                run = make_solver(
                    model, precision, use_cuda_graph=graph, pcg_mode="production", **_drive_terms_options()
                )
                state, control = run.create_state(dt=H), run.create_control()
                _set_root_target(control, model, np.radians(30.0) * np.array([0.0, 0.6, 0.8]))
                for _ in range(3):
                    run.step(state, state, control, dt=H)
                captured = dict(run._graphs.captures)
                roots = run._floating
                stale = [getattr(roots, name) for name in names]
                for name in names:
                    setattr(roots, name, wp.clone(getattr(roots, name)))
                for array in stale:
                    values = array.numpy()
                    values[...] = np.nan
                    array.assign(values)
                for _ in range(3):
                    run.step(state, state, control, dt=H)
                    test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                if graph:
                    test.assertGreater(run._graphs.captures["problem_owners"], captured.get("problem_owners", 0))
                history = state._monolithic_coupled
                results[graph] = (
                    history.floating_com.numpy(),
                    history.floating_rotation.numpy(),
                    state.joint_q.numpy(),
                )
        for eager, graphed in zip(results[False], results[True], strict=True):
            np.testing.assert_array_equal(graphed, eager)


def test_root_wrench_matches_rigid_body(test, device):
    """A world COM wrench on an undriven root moves a welded tree like the same wrench on the composite rigid body."""
    force, torque = np.array([0.3, -0.2, 0.5]), np.array([0.004, -0.002, 0.003])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = np.diag([2e-3, 3e-3, 4e-3])
    pose = wp.transform((0.1, -0.2, 0.5), wp.quat_rpy(0.3, -0.2, 0.4))
    root = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(inertia), label="root")
    joints = [builder.add_joint_free(root, parent_xform=pose)]
    for sign in (1.0, -1.0):
        child = builder.add_link(xform=pose, mass=0.25, inertia=wp.diag(wp.vec3(1e-4)))
        joints.append(
            builder.add_joint_fixed(root, child, parent_xform=wp.transform((0.0, sign * 0.06, 0.0), wp.quat_identity()))
        )
    builder.add_articulation(joints)
    tree = builder.finalize(device=device)
    # The composite: same COM (symmetric children), mass and inertia about the COM.
    composite = inertia + 2 * (
        np.diag([1e-4] * 3) + 0.25 * (0.06**2 * np.eye(3) - np.outer([0, 0.06, 0], [0, 0.06, 0]))
    )
    rigid_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = rigid_builder.add_link(xform=pose, mass=1.5, inertia=wp.mat33(composite))
    rigid_builder.add_articulation([rigid_builder.add_joint_free(body, parent_xform=pose)])
    rigid = rigid_builder.finalize(device=device)
    results = []
    for model in (tree, rigid):
        run = make_solver(model)
        state, control = run.create_state(dt=H), run.create_control()
        efforts = control.joint_f.numpy()
        efforts[:6] = np.r_[force, torque]
        control.joint_f.assign(efforts)
        for _ in range(40):
            run.step(state, state, control, dt=H)
            test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
        test.assertFalse(run._conservative_energy())
        results.append(state.body_q.numpy()[0].astype(np.float64))
    np.testing.assert_allclose(results[0][:3], results[1][:3], atol=1e-9)
    sign = np.sign(results[0][3:] @ results[1][3:])
    np.testing.assert_allclose(results[0][3:], sign * results[1][3:], atol=1e-9)


def test_conservative_energy_truth_table(test, device):
    """Only root or rigid torques make the energy heuristic; a root force keeps it variational."""
    model = floating_chain(device, gravity=(0.0, 0.0, 0.0))
    run = make_solver(model)
    state, control = run.create_state(dt=H), run.create_control()
    for wrench, expected in (((0, 0, 0, 0, 0, 0), True), ((1, 0, 0, 0, 0, 0), True), ((0, 0, 0, 0, 0.01, 0), False)):
        efforts = control.joint_f.numpy()
        efforts[:6] = wrench
        control.joint_f.assign(efforts)
        run.step(state, state, control, dt=H)
        test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
        test.assertEqual(run._conservative_energy(), expected, wrench)
        test.assertEqual(
            run.last_stats.variational["energy_acceptance"], "variational" if expected else "heuristic", wrench
        )


def test_target_conventions(test, device):
    """Targets in Newton's FREE convention with a rotated anchor and offset child frame and COM: a pure damper
    makes the root track target_qd, a stiff drive brings the root to eval_fk(target_q), and a root force
    against the drive settles at f / k_t (joint_f on a driven root is allowed)."""
    for precision, _numpy_dtype, _dtype in PRECISIONS:
        for case, wanted in (
            ("linear rates", (0.2, -0.1, 0.3, 0.0, 0.0, 0.0)),
            ("angular rates", (0, 0, 0, 0.3, 0.1, -0.2)),
        ):
            with test.subTest(precision=precision, case=case):
                model = driven_chain(device, kt=0.0, dt=40.0, kr=0.0, dr=2.0, fixed_child=True)
                run = make_solver(model, precision, **_drive_terms_options())
                state, control = run.create_state(dt=H), run.create_control()
                rates = control.joint_target_qd.numpy()
                rates[:6] = wanted
                control.joint_target_qd.assign(rates)
                # Linear time constant m / d_t = 1.7 / 40 s: 600 steps leave exp(-14) of the start error.
                for _ in range(600):
                    run.step(state, state, control, dt=H)
                    test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                mirror = state.joint_qd.numpy()[:6].astype(np.float64)
                if case == "linear rates":
                    # Measured after 600 steps: 1.3e-6 m/s (FP64), 5.0e-5 m/s (FP32).
                    np.testing.assert_allclose(mirror, rates[:6], atol=2e-5 if precision == "fp64" else 5e-4)
                else:
                    # Turning about the root COM, off the composite COM, needs a centripetal force from the
                    # linear damper (|dv| = m w^2 r / d_t, about 3e-4 m/s) and a gyroscopic torque from the
                    # angular one (w x I w / d_r): the steady rates are within about 1% of the targets.
                    np.testing.assert_allclose(mirror[3:], rates[3:6], rtol=1e-2, atol=1e-2 * abs(rates[3:6]).max())
                    test.assertLess(np.abs(mirror[:3]).max(), 1e-3)
        with test.subTest(precision=precision, case="pose"):
            model = driven_chain(device, fixed_child=True)
            run = make_solver(model, precision, **_drive_terms_options())
            state, control = run.create_state(dt=H), run.create_control()
            _set_root_target(control, model, np.radians(30.0) * np.array([0.6, 0.0, 0.8]), offset=(0.05, -0.03, 0.02))
            for _ in range(1500):
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
            reference = model.state()
            newton.eval_fk(model, control.joint_target_q, wp.zeros_like(control.joint_target_qd), reference)
            np.testing.assert_allclose(state.body_q.numpy()[0, :3], reference.body_q.numpy()[0, :3], atol=2e-4)
            q, r = state.body_q.numpy()[0, 3:].astype(np.float64), reference.body_q.numpy()[0, 3:].astype(np.float64)
            test.assertLess(2 * np.arccos(min(1.0, abs(q @ r))), 2e-3)
            test.assertLess(np.linalg.norm(run._floating.error_t.numpy()[0]), 2e-4)
        with test.subTest(precision=precision, case="force"):
            kt = 400.0
            model = driven_chain(device, kt=kt, dt=40.0, fixed_child=True)
            run = make_solver(model, precision, **_drive_terms_options())
            state, control = run.create_state(dt=H), run.create_control()
            force = np.array([1.0, -0.5, 0.2])
            efforts = control.joint_f.numpy()
            efforts[:3] = force
            control.joint_f.assign(efforts)
            # The force acts at the root COM, off the composite COM: a slowly damped rotation rings first
            # (measured: z within 2.4% after 1500 steps, 1e-4 after 3000).
            for _ in range(3000):
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
            # error_t = c_n - c_t at the last step start; the drive balances f with k_t (c - c_t).
            np.testing.assert_allclose(
                run._floating.error_t.numpy()[0], force / kt, rtol=1e-3, atol=1e-6 if precision == "fp64" else 1e-5
            )


def test_tangent_entries_in_packed_trees(test, device):
    """With a fixed-base tree before it, the floating root's drive block lands in its own packed tree block:
    equal to the single-tree result after one identical step."""

    def first_tree(builder):
        base = builder.add_link(mass=0.4, inertia=wp.diag(wp.vec3(1e-3)))
        builder.add_articulation([builder.add_joint_revolute(-1, base, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0)])

    blocks = []
    for prefix in (None, first_tree):
        model = driven_chain(device, fixed_child=True, prefix=prefix)
        run = make_solver(model, **_drive_terms_options())
        state, control = run.create_state(dt=H), run.create_control()
        _set_root_target(control, model, np.radians(20.0) * np.array([0.0, 0.0, 1.0]))
        run.step(state, state, control, dt=H)
        test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
        test.assertEqual(run._kinematics.tree_count, 1 if prefix is None else 2)
        run._evaluate(run._accepted, H, build_tangent=True, device_result=True)
        tangent = run._objective.tangent.numpy().ravel()
        blocks.append(tangent[run._floating.entries.numpy()[0]].reshape(6, 6))
    test.assertGreater(abs(blocks[0]).max(), 0.0)
    np.testing.assert_allclose(blocks[1], blocks[0], rtol=1e-9, atol=1e-9 * abs(blocks[0]).max())


def test_snapshot_names_root_and_field(test, device):
    """Nonfinite root targets or wrench and a non-unit target name the root joint and the field, with two
    floating roots (the step's own input checks catch nonfinite values earlier; this is the drive's guard)."""

    def second_root(builder):
        body = builder.add_link(mass=0.3, inertia=wp.diag(wp.vec3(1e-3)), label="other_root")
        child = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)), label="other_child")
        builder.add_articulation(
            [builder.add_joint_free(body, label="other_free"), builder.add_joint_fixed(body, child, label="other_weld")]
        )

    model = driven_chain(device, prefix=second_root)
    run = make_solver(model, **_drive_terms_options())
    test.assertEqual(run._floating.count, 2)
    _, _c, d, t = root_offsets(model)
    for name, index, message in (
        ("joint_target_q", t + 1, "nonfinite joint_target_q"),
        ("joint_target_qd", d + 4, "nonfinite joint_target_qd"),
        ("joint_f", d + 2, "nonfinite joint_f"),
        ("joint_target_q", t + 6, "target quaternion must be unit"),
    ):
        with test.subTest(field=name, message=message):
            control = run.create_control()
            values = getattr(control, name).numpy()
            values[index] = np.nan if "nonfinite" in message else 2.0
            getattr(control, name).assign(values)
            with test.assertRaisesRegex(ValueError, f"Floating root joint 'free': .*{message}"):
                run._floating.snapshot(control, run._objective)


def test_drive_rejections(test, device):
    """Anisotropic gains or limits, a non-PD mode, legacy target layouts and non-unit targets fail by name;
    joint_f on a driven scalar joint still fails as a duplicate drive."""
    from unittest.mock import patch  # noqa: PLC0415

    for name, index, value, message in (
        ("joint_target_ke", 1, 401.0, "linear target_ke must be finite and equal"),
        ("joint_target_kd", 5, 0.4, "angular target_kd must be finite and equal"),
        ("joint_effort_limit", 4, 2.0, "angular effort_limit must be finite and equal"),
    ):
        with test.subTest(field=name):
            model = driven_chain(device)
            values = getattr(model, name).numpy()
            values[index] = value
            getattr(model, name).assign(values)
            with test.assertRaisesRegex(ValueError, f"Floating root joint 'free': {message}"):
                make_solver(model, **_drive_terms_options())
    for name, value, message in (
        ("joint_effort_limit", 0.0, "a driven linear triple requires a positive effort limit"),
        ("joint_target_ke", -1.0, "linear drive gains must be nonnegative"),
    ):
        with test.subTest(field=name, value=value):
            model = driven_chain(device)
            values = getattr(model, name).numpy()
            values[:3] = value
            getattr(model, name).assign(values)
            with test.assertRaisesRegex(ValueError, f"Floating root joint 'free': {message}"):
                make_solver(model, **_drive_terms_options())
    model = driven_chain(device)
    modes = model.joint_target_mode.numpy()
    modes[3:6] = int(newton.JointTargetMode.NONE)
    model.joint_target_mode.assign(modes)
    with test.assertRaisesRegex(ValueError, "angular triple requires POSITION_VELOCITY"):
        make_solver(model, **_drive_terms_options())
    with patch.object(newton, "use_coord_layout_targets", False):
        legacy = driven_chain(device)
    with test.assertRaisesRegex(ValueError, "newton.use_coord_layout_targets"):
        make_solver(legacy, **_drive_terms_options())
    model = driven_chain(device)
    run = make_solver(model, **_drive_terms_options())
    state, control = run.create_state(dt=H), run.create_control()
    targets = control.joint_target_q.numpy()
    targets[3:7] = (0.0, 0.0, 0.0, 0.9)
    control.joint_target_q.assign(targets)
    with test.assertRaisesRegex(
        ValueError, "Floating root joint 'free': a driven root's target quaternion must be unit"
    ):
        run.step(state, state, control, dt=H)
    # Translation-only drive: the COM target still reads the quaternion (child frame and COM offsets).
    model = driven_chain(device, kr=0.0, dr=0.0)
    run = make_solver(model, **_drive_terms_options())
    state, control = run.create_state(dt=H), run.create_control()
    targets = control.joint_target_q.numpy()
    targets[3:7] = 0.0
    control.joint_target_q.assign(targets)
    with test.assertRaisesRegex(ValueError, "target quaternion must be unit"):
        run.step(state, state, control, dt=H)
    # A driven hinge with joint_f is still a duplicate drive.
    model = driven_chain(device)
    for name, value in (("joint_target_ke", 5.0), ("joint_effort_limit", 10.0)):
        values = getattr(model, name).numpy()
        values[6] = value
        getattr(model, name).assign(values)
    modes = model.joint_target_mode.numpy()
    modes[6] = int(newton.JointTargetMode.POSITION_VELOCITY)
    model.joint_target_mode.assign(modes)
    run = make_solver(model, **_drive_terms_options())
    state, control = run.create_state(dt=H), run.create_control()
    efforts = control.joint_f.numpy()
    efforts[:7] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    control.joint_f.assign(efforts)
    with test.assertRaisesRegex(ValueError, "Duplicate driven joint_f"):
        run.step(state, state, control, dt=H)


class TestJointFloatingDrive(unittest.TestCase):
    pass


for _test in (
    test_energy_gradient_tangent,
    test_truncation_is_c1_and_bounded,
    test_left_jacobian_branches,
    test_converges_from_far_rotations,
    test_root_wrench_matches_rigid_body,
    test_conservative_energy_truth_table,
    test_target_conventions,
    test_tangent_entries_in_packed_trees,
    test_snapshot_names_root_and_field,
    test_drive_rejections,
):
    add_function_test(TestJointFloatingDrive, _test.__name__, _test, devices=get_test_devices())
add_function_test(
    TestJointFloatingDrive,
    test_problem_graphs_recapture_after_root_storage_replacement.__name__,
    test_problem_graphs_recapture_after_root_storage_replacement,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
