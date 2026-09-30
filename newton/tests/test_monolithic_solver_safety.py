# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise native trial, publication and complete-state failure boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.tests import test_monolithic_solver_coupled as coupled
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, independent_objective, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_failure_matrix(test, device):
    """Repeat the existing physical failure and two-buffer rollback gates in both precisions."""
    for precision in ("fp64", "fp32"):

        def native(*args, precision=precision, **kwargs):
            return solver(*args, precision=precision, **kwargs)

        with test.subTest(precision=precision), patch.object(coupled, "solver", native):
            coupled.test_failure_gates(test, device)
            coupled.test_state_and_rollback(test, device)


def test_publication(test, device):
    """Publish only the allowed initialized or accepted iterate, never a rejected trial."""
    for precision in ("fp64", "fp32"):
        for policy in ("strict", "last_accepted", "current_iterate"):
            for accepted in (False, True):
                model = scene(device)
                run = solver(model, precision=precision, publish_policy=policy)
                state = run.create_state(dt=0.001)
                out = run._states.copy(state)
                out.particle_f.fill_(wp.vec3(2))
                before, oldout = snapshot(state), snapshot(out)
                expected = {}

                def stop(dt, run=run, accepted=accepted, expected=expected):
                    run._metrics["nonlinear_iterations"] = 0
                    run._reference_residual = run._initial_reference(dt)
                    run._evaluate(run._accepted, dt)
                    if accepted:
                        run._variational["trials"].append({"accepted": True})
                    expected.update(snapshot(run._accepted))
                    run._trial.particle_q.fill_(run._trial.particle_q.dtype(float("nan")))
                    raise _StepFailure(run.Status.NONLINEAR_MAX_ITERATIONS, "publication fixture")

                with patch(
                    "newton._src.solvers.monolithic.solver_monolithic.iterate",
                    side_effect=lambda current, dt, stop=stop: stop(dt),
                ):
                    run.step(state, out, model.control(), dt=0.001)
                publish = policy == "current_iterate" or (policy == "last_accepted" and accepted)
                test.assertEqual(run.last_stats.rolled_back, not publish)
                test.assertFalse(run.last_stats.converged)
                assert_snapshot(test, state, before)
                if publish:
                    for name in ("joint_q", "joint_qd", "particle_q", "particle_qd"):
                        np.testing.assert_array_equal(getattr(out, name).numpy(), expected[name])
                    test.assertTrue(run.get_contact_forces(out).force_valid)
                    run._states.validate(out)
                else:
                    assert_snapshot(test, out, oldout)


def test_native_trials(test, device):
    """Check written trial displacements, independent E/r and actual B32 posterior."""
    for precision in ("fp64", "fp32"):
        for law in ("C-EE", "C-IE", "C-II"):
            model = scene(device)
            run = solver(model, law, precision=precision, publish_policy="current_iterate")
            state = run.create_state(dt=0.001)
            run.step(state, state, model.control(), dt=0.001)
            test.assertFalse(run.last_stats.rolled_back)
            original = snapshot(run._original)
            direction = np.linspace(-0.3, 0.7, 14, dtype=np.float32) * np.float32(1e-5)
            run._delta.assign(direction)
            q, x = run._accepted.joint_q.numpy(), run._accepted.particle_q.numpy()
            for alpha in (1.0, 0.5, 0.125):
                run._take_trial(alpha, run._dt)
                effective = run._effective_displacement()
                actual = np.r_[run._trial.joint_q.numpy() - q, (run._trial.particle_q.numpy() - x).ravel()]
                np.testing.assert_array_equal(effective, actual)
                test.assertEqual(effective.dtype, run._numpy_dtype)
                qe, xe = run._trial.joint_q.numpy().astype(float), run._trial.particle_q.numpy().astype(float)
                e, r, _ = run._evaluate(run._trial, run._dt)
                with patch.object(coupled, "joint_terms", wraps=coupled.joint_terms) as reference:
                    oe, orr, _ = independent_objective(
                        run, run._trial.joint_q.numpy(), run._trial.particle_q.numpy(), run._dt
                    )
                for value in reference.call_args.args:
                    if isinstance(value, np.ndarray):
                        test.assertEqual(value.dtype, np.float64)
                tolerance = 2e-9 if precision == "fp64" else 8e-5
                np.testing.assert_allclose(r, orr, rtol=tolerance, atol=tolerance)
                test.assertAlmostEqual(float(e), float(oe), delta=tolerance)
                _, scale = run._assemble(0)
                matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
                post = run._physical_linear_error(None, direction, r, scale)
                expected = scale.astype(float) * (matrix @ direction.astype(float) + r.astype(float))
                np.testing.assert_allclose(post, expected, rtol=tolerance, atol=tolerance)
                if law == "C-EE":
                    # Differentiate the independent E at the actual native candidate.
                    z = np.r_[qe, xe.ravel()]
                    h = 1e-7
                    d = np.linspace(-0.2, 0.3, 14)
                    d[:2] *= 0.01
                    values = []
                    for sign in (-1, 1):
                        trial = z + sign * h * d
                        values.append(
                            independent_objective(run, trial[:2], trial[2:].reshape(-1, 3), float(run._dt))[0]
                        )
                    test.assertAlmostEqual((values[1] - values[0]) / (2 * h), orr @ d, delta=2e-7)
            assert_snapshot(test, run._original, original)


def test_checkpoint_corruption(test, device):
    """Reject corrupt schema, precision, history, force generation and dtype atomically."""
    model = scene(device)
    for precision in ("fp64", "fp32"):
        run = solver(model, precision=precision)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        expected = snapshot(state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            for field, value in [
                ("history.precision", 16),
                ("history.schema_version", 99),
                ("history.config_digest", 0),
                ("history.force_valid", 2),
                ("history.force_generation", -1),
                ("history.last_dt", 0),
                ("history.bound_q", 1),
                ("history.link_vsym", np.nan),
            ]:
                bad = {k: v.copy() for k, v in expected.items()}
                bad[field].fill(value)
                np.savez(path, **bad)
                with test.assertRaises(ValueError):
                    run.load_checkpoint(state, path)
                assert_snapshot(test, state, expected)
            bad = {k: v.copy() for k, v in expected.items()}
            bad["joint_q"] = bad["joint_q"].astype(np.float32 if precision == "fp64" else np.float64)
            np.savez(path, **bad)
            with test.assertRaises(ValueError):
                run.load_checkpoint(state, path)
            assert_snapshot(test, state, expected)
            run.reset_state(state, dt=0.001)
            test.assertFalse(run.get_contact_forces(state).force_valid)
            test.assertEqual(run.get_contact_forces(state).force_generation, 0)
            state.joint_q.assign(state.joint_q.numpy() + run._numpy_dtype(0.001))
            run.reset_state(state, dt=0.001)
            run._states.validate(state)


def test_modes_and_identity(test, device):
    """Exercise each globalization in native modes, then reject the other precision atomically."""
    model = scene(device)
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for globalization in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
        for law in ("C-EE", "C-IE", "C-II"):
            for mode, graph in modes:
                runs = [
                    solver(
                        model,
                        law,
                        precision=precision,
                        globalization=globalization,
                        allow_nonconservative_energy=True,
                        mode=mode,
                        graph=graph,
                        publish_policy="current_iterate",
                    )
                    for precision in ("fp64", "fp32")
                ]
                states = [run.create_state(dt=0.001) for run in runs]
                for _ in range(2):
                    for run, state in zip(runs, states, strict=True):
                        run.step(state, state, model.control(), dt=0.001)
                        test.assertFalse(run.last_stats.rolled_back, run.last_stats)
                        run._states.validate(state)
                        test.assertTrue(run.get_contact_forces(state).force_valid)
                for i, run in enumerate(runs):
                    before = snapshot(states[i])
                    with test.assertRaises(ValueError):
                        run.step(states[1 - i], states[i], model.control(), dt=0.001)
                    assert_snapshot(test, states[i], before)


class TestPrecisionSafety(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_failure_matrix,
        test_publication,
        test_native_trials,
        test_checkpoint_corruption,
        test_modes_and_identity,
    ):
        add_function_test(TestPrecisionSafety, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
