# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise PR2 publication, scope, failure and graph contracts."""

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation_variational import _configuration_digest
from newton._src.solvers.monolithic.tet_variational import acceptance
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices


def model_and_solver(device, **options):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    body = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    joint = builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, limit_ke=0.0, limit_kd=0.0)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverMonolithic._create_articulation_only(model, globalization="ArmijoOrResNorm", **options)
    return model, solver


def snapshot(state):
    arrays = {name: value.numpy().copy() for name, value in vars(state).items() if isinstance(value, wp.array)}
    arrays.update(
        {
            "history." + name: value.numpy().copy()
            for name, value in vars(state._monolithic_articulation).items()
            if isinstance(value, wp.array)
        }
    )
    return arrays


def assert_snapshot(state, expected):
    actual = snapshot(state)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value, err_msg=name)


def test_final_residual_gate(test, device):
    """Reject a finite final state whose recomputed residual no longer converges."""
    model, solver = model_and_solver(device)
    state, out = solver._create_articulation_state(dt=0.001), solver._create_articulation_state(dt=0.001)
    original = snapshot(state)
    iterate = solver._iterate_articulation

    def changed_candidate(dt):
        status = iterate(dt)
        test.assertEqual(status, solver.Status.SUCCESS)
        solver._accepted.joint_q.fill_(0.001)
        return status

    with patch.object(solver, "_iterate_articulation", side_effect=changed_candidate):
        solver.step(state, out, model.control(), dt=0.001)
    test.assertFalse(solver.last_stats.converged)
    test.assertTrue(solver.last_stats.rolled_back)
    assert_snapshot(state, original)
    assert_snapshot(out, original)


def test_muscle_scope(test, device):
    """Reject unsupported muscle forces at construction and after model mutation."""
    model, solver = model_and_solver(device)
    state = solver._create_articulation_state(dt=0.001)
    model.muscle_count = 1
    with test.assertRaisesRegex(ValueError, "muscle"):
        SolverMonolithic._create_articulation_only(model, globalization="Armijo")
    with test.assertRaisesRegex(ValueError, "configuration|muscle"):
        solver.step(state, state, model.control(), dt=0.001)


def test_zero_effective_inertia(test, device):
    """Reject a free coordinate with zero effective inertia despite a PD spring."""
    model, _ = model_and_solver(device)
    model.body_mass.zero_()
    model.body_inertia.zero_()
    model.joint_target_ke.fill_(1.0)
    model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION_VELOCITY))
    terms = SolverMonolithic.JointTerms(implicit_pd=True)
    with test.assertRaisesRegex(ValueError, "inertia"):
        SolverMonolithic._create_articulation_only(model, globalization="Armijo", joint_terms=terms)


def test_body_flags_scope(test, device):
    """Reject movable kinematic bodies while retaining a fixed kinematic root."""
    for fixed in (False, True):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        root = builder.add_link(mass=0 if fixed else 1, inertia=wp.diag(wp.vec3(0.01)), is_kinematic=True)
        if fixed:
            child = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
            joints = [
                builder.add_joint_fixed(-1, root),
                builder.add_joint_prismatic(root, child, axis=newton.Axis.X, limit_ke=0, limit_kd=0),
            ]
        else:
            joints = [builder.add_joint_prismatic(-1, root, axis=newton.Axis.X, limit_ke=0, limit_kd=0)]
        builder.add_articulation(joints)
        model = builder.finalize(device=device)
        if not fixed:
            with test.assertRaisesRegex(ValueError, "[Kk]inematic"):
                SolverMonolithic._create_articulation_only(model, globalization="ArmijoOrResNorm")
            continue
        solver = SolverMonolithic._create_articulation_only(model, globalization="ArmijoOrResNorm")
        state = solver._create_articulation_state(dt=0.001)
        state.body_f.assign([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        solver.step(state, state, model.control(), dt=0.001)
        test.assertTrue(solver.last_stats.converged)
        np.testing.assert_allclose(state.joint_q.numpy(), [1e-6], rtol=1e-6)
        np.testing.assert_array_equal(state.body_q.numpy()[0, :3], [0, 0, 0])
    model, _ = model_and_solver(device)
    for flag in (0, 3, 4, 7, -1):
        model.body_flags.fill_(flag)
        with test.assertRaisesRegex(ValueError, "flags"):
            SolverMonolithic._create_articulation_only(model, globalization="ArmijoOrResNorm")


def pose_dependent_inertia_solver(device):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    root, child = builder.add_link(mass=0), builder.add_link(mass=1)
    joints = [
        builder.add_joint_revolute(-1, root, axis=newton.Axis.Z, target_ke=1, target_kd=0.01, limit_ke=0, limit_kd=0),
        builder.add_joint_prismatic(root, child, axis=newton.Axis.X, limit_ke=0, limit_kd=0),
    ]
    builder.add_articulation(joints)
    builder.joint_q[1] = 1
    model = builder.finalize(device=device)
    # Point mass with PSD second moment, set before configuration is frozen.
    model.body_inertia.zero_()
    model.joint_target_mode.assign([int(newton.JointTargetMode.POSITION_VELOCITY), int(newton.JointTargetMode.NONE)])
    solver = SolverMonolithic._create_articulation_only(
        model, globalization="ArmijoOrResNorm", joint_terms=SolverMonolithic.JointTerms(implicit_pd=True)
    )
    return model, solver


def test_pose_dependent_inertia(test, device):
    """Reject singular actual states through reset, source, continuation and checkpoint."""
    model, solver = pose_dependent_inertia_solver(device)
    for operation in ("reset", "source", "step", "checkpoint"):
        with test.subTest(operation=operation):
            state = solver._create_articulation_state(dt=0.001)
            out = solver._create_articulation_state(dt=0.001)
            output = snapshot(out)
            state.joint_q.zero_()
            state._monolithic_articulation.bound_q.zero_()
            original = snapshot(state)
            with test.assertRaisesRegex(ValueError, "inertia"):
                if operation == "reset":
                    solver._reset_articulation_state(state, dt=0.001)
                elif operation == "source":
                    solver._create_articulation_state(dt=0.001, source=state)
                elif operation == "step":
                    solver.step(state, out, model.control(), dt=0.001)
                else:
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "singular.npz"
                        np.savez(path, **original)
                        solver._states.load_checkpoint(out, path)
            assert_snapshot(state, original)
            assert_snapshot(out, output)


def test_singular_candidate_rollback(test, device):
    """Refuse a converged singular candidate and restore the complete input history."""
    for alias in (False, True):
        model, solver = pose_dependent_inertia_solver(device)
        state = solver._create_articulation_state(dt=0.001)
        state.joint_qd.assign([0, -1000])
        solver._reset_articulation_state(state, dt=0.001)
        out = state if alias else solver._create_articulation_state(dt=0.001)
        original = snapshot(state)
        iterate = solver._iterate_articulation

        def singular_candidate(dt, *, iterate=iterate, solver=solver):
            status = iterate(dt)
            # Make the exact zero root independent of CUDA fused predictor rounding.
            solver._accepted.joint_q.zero_()
            return status

        with patch.object(solver, "_iterate_articulation", side_effect=singular_candidate):
            solver.step(state, out, model.control(), dt=0.001)
        test.assertFalse(solver.last_stats.converged)
        test.assertTrue(solver.last_stats.rolled_back)
        test.assertIn("inertia", solver.last_stats.failure_reason)
        assert_snapshot(state, original)
        assert_snapshot(out, original)


def test_invalid_inputs(test, device):
    """Reject invalid controls, torque and history without mutating either state."""
    model, solver = model_and_solver(device)
    state, out = solver._create_articulation_state(dt=0.001), solver._create_articulation_state(dt=0.001)
    control = model.control()
    for name in ("joint_target_q", "joint_target_qd", "joint_f"):
        old = getattr(control, name)
        for invalid in (
            wp.zeros(2, dtype=float, device=device),
            wp.zeros(1, dtype=wp.float64, device=device),
            wp.full(1, float("nan"), device=device),
        ):
            setattr(control, name, invalid)
            original, output = snapshot(state), snapshot(out)
            with test.assertRaises(ValueError):
                solver.step(state, out, control, dt=0.001)
            assert_snapshot(state, original)
            assert_snapshot(out, output)
        setattr(control, name, old)
    state.body_f.assign([[0, 0, 0, 1, 0, 0]])
    with test.assertRaisesRegex(ValueError, "torque"):
        solver.step(state, out, control, dt=0.001)
    state.body_f.zero_()
    for dt in (0, -1, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            solver.step(state, out, control, dt=dt)
    with test.assertRaises(ValueError):
        solver.step(state, out, control, contacts=object(), dt=0.001)
    state._monolithic_articulation.link_vsym.fill_(float("nan"))
    with test.assertRaisesRegex(ValueError, "Nonfinite"):
        solver.step(state, out, control, dt=0.001)


def test_trial_failure_rollback(test, device):
    """Exhaust all trial retries and restore all arrays for alias and distinct outputs."""
    for alias in (False, True):
        model, solver = model_and_solver(device)
        state = solver._create_articulation_state(dt=0.001)
        out = state if alias else solver._create_articulation_state(dt=0.001)
        state.body_f.assign([[1, 0, 0, 0, 0, 0]])
        original = snapshot(state)
        with patch(
            "newton._src.solvers.monolithic.articulation_solver.acceptance", return_value=(False, False, False, False)
        ):
            solver.step(state, out, model.control(), dt=0.001)
        test.assertEqual(solver.last_stats.status, solver.Status.REGULARIZATION_EXHAUSTED)
        test.assertTrue(solver.last_stats.rolled_back)
        test.assertGreater(solver.last_stats.line_search_iterations, 0)
        assert_snapshot(state, original)
        assert_snapshot(out, original)


def test_linear_postcheck(test, device):
    """Reject corrupted directions despite a successful PCG status."""
    model, solver = model_and_solver(device)
    state = solver._create_articulation_state(dt=0.001)
    state.body_f.assign([[1, 0, 0, 0, 0, 0]])
    original = snapshot(state)
    recover = solver._linear.recover_delta

    def corrupt(*args, **kwargs):
        status = recover(*args, **kwargs)
        solver._delta.fill_(float("nan"))
        return status

    with patch.object(solver._linear, "recover_delta", side_effect=corrupt):
        solver.step(state, state, model.control(), dt=0.001)
    test.assertFalse(solver.last_stats.converged)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertTrue(solver.last_stats.variational["linear_postchecks"])
    test.assertTrue(all(not np.isfinite(row["residual"]) for row in solver.last_stats.variational["linear_postchecks"]))
    assert_snapshot(state, original)


def test_nonfinite_tangent(test, device):
    """Reject FP32 tangent overflow before any candidate publication."""
    model, solver = model_and_solver(device)
    state = solver._create_articulation_state(dt=0.001)
    original = snapshot(state)
    evaluate = solver._objective.evaluate

    def overflow(*args, **kwargs):
        evaluate(*args, **kwargs)
        if kwargs.get("build_tangent", True):
            solver._objective.tangent32.fill_(float("inf"))

    with patch.object(solver._objective, "evaluate", side_effect=overflow):
        solver.step(state, state, model.control(), dt=0.001)
    test.assertEqual(solver.last_stats.status, solver.Status.NONFINITE)
    assert_snapshot(state, original)


def test_graph_rebinding_and_frozen_inputs(test, device):
    """Freeze physical inputs per step and rebind graphs after State and dt changes."""
    options = {"pcg_mode": "production" if device.is_cuda else "diagnostic", "use_cuda_graph": device.is_cuda}
    model, solver = model_and_solver(device, **options)
    state = solver._create_articulation_state(dt=0.001)
    control = model.control()
    control.joint_f.fill_(1.0)
    iterate = solver._iterate_articulation

    def change_input(dt):
        control.joint_f.fill_(2.0)
        return iterate(dt)

    with patch.object(solver, "_iterate_articulation", side_effect=change_input):
        solver.step(state, state, control, dt=0.001)
    np.testing.assert_allclose(state.joint_q.numpy(), [1e-6], rtol=1e-6)
    captures = sum(solver._graphs.captures.values())
    old_buffer = solver._accepted.joint_q
    solver._accepted.joint_q = wp.clone(old_buffer)
    state.joint_q = wp.clone(state.joint_q)
    before_q, before_v = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()
    solver.step(state, state, control, dt=0.0005)
    test.assertTrue(solver.last_stats.converged)
    np.testing.assert_allclose(state.joint_q.numpy(), before_q + 0.0005 * before_v + 2 * 0.0005**2, rtol=1e-6)
    if device.is_cuda:
        test.assertGreater(sum(solver._graphs.captures.values()), captures)


def test_budget_and_legacy(test, device):
    """Distinguish strict rollback from explicitly requested legacy budget publication."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    body = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    builder.add_articulation([builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, limit_ke=0, limit_kd=0)])
    model = builder.finalize(device=device)
    for policy in ("strict", "legacy"):
        solver = SolverMonolithic._create_articulation_only(
            model, globalization="Armijo", publish_policy=policy, newton_max_iterations=1
        )
        state = solver._create_articulation_state(dt=0.1)
        control = model.control()
        control.joint_f.fill_(1.0)
        original = snapshot(state)
        solver.step(state, state, control, dt=0.1)
        test.assertEqual(solver.last_stats.status, solver.Status.NONLINEAR_MAX_ITERATIONS)
        test.assertFalse(solver.last_stats.converged)
        if policy == "strict":
            test.assertTrue(solver.last_stats.rolled_back)
            assert_snapshot(state, original)
        else:
            test.assertFalse(solver.last_stats.rolled_back)
            solver._states.validate(state)
            test.assertGreater(float(state.joint_q.numpy()[0]), 0)


def test_checkpoint_cross_device(test, device):
    """Load a complete checkpoint on CPU without changing its private history."""
    model, solver = model_and_solver(device)
    state = solver._create_articulation_state(dt=0.001)
    state.body_f.assign([[1, 0, 0, 0, 0, 0]])
    solver.step(state, state, model.control(), dt=0.001)
    _, cpu = model_and_solver(wp.get_device("cpu"))
    restored = cpu._create_articulation_state(dt=0.001)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        solver._states.save_checkpoint(state, path)
        cpu._states.load_checkpoint(restored, path)
    assert_snapshot(restored, snapshot(state))


class TestArticulationContracts(unittest.TestCase):
    def test_acceptance_branches(self):
        """Preserve non-descent residual acceptance and the A/R/OR truth table."""
        for mode, wanted in (("Armijo", False), ("ResNorm", True), ("ArmijoOrResNorm", True)):
            passed, a, r, _ = acceptance(
                mode, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, energy_noise=0, residual_noise=0, slope_noise=0
            )
            self.assertEqual((passed, a, r), (wanted, False, True))

    def test_digest_encoding(self):
        """Match a fixed little-endian, length-prefixed configuration digest."""
        metadata = b'{"dtype":"<i4","field_name":"a","schema_version":1,"shape":[2]}'
        data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
        expected = hashlib.sha256(
            struct.pack("<Q", len(metadata)) + metadata + struct.pack("<Q", len(data)) + data
        ).digest()
        self.assertEqual(_configuration_digest({"a": np.array([1, 2], dtype=">i4")}), expected)


for _name in (
    "final_residual_gate",
    "muscle_scope",
    "zero_effective_inertia",
    "body_flags_scope",
    "pose_dependent_inertia",
    "singular_candidate_rollback",
    "invalid_inputs",
    "trial_failure_rollback",
    "linear_postcheck",
    "nonfinite_tangent",
    "graph_rebinding_and_frozen_inputs",
    "budget_and_legacy",
    "checkpoint_cross_device",
):
    add_function_test(
        TestArticulationContracts, "test_" + _name, globals()["test_" + _name], devices=get_test_devices()
    )


if __name__ == "__main__":
    unittest.main()
