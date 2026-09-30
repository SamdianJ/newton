# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check coupled derivatives, graph replay and atomic geometry failure."""

import unittest
from unittest.mock import patch

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p3_coupled_fixtures import make_solver, make_tiny, set_control
from scripts.monolithic_reference.p3_coupled_oracle import CoupledOracle
from scripts.monolithic_reference.verify_p3_coupled import state_arrays


def test_reference_jacobian(test, device):
    """Compare independent full Dr to actual q/x differences and production E/r."""
    for case in ("normal_loading", "tangent_reversal", "curved_release"):
        for law in ("C-EE", "C-IE", "C-II"):
            model = make_tiny(case, device)
            run = make_solver(model, case, law)
            state = run.create_state(dt=0.001)
            run.step(state, state, model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged)
            oracle = CoupledOracle(
                model,
                run._tet.physics,
                run._objective.params.numpy(),
                law=law,
                mu=0.0 if case == "normal_loading" else 0.5,
                width=0.0,
                goal_scale=0.1,
            )
            oracle.freeze(0.001, np.zeros(model.joint_dof_count))
            z = np.r_[oracle.q, oracle.x[oracle.tet.free].ravel()]
            z[0] = 0.0105 if case == "curved_release" else 0.0042
            if len(oracle.q) > 1:
                z[1] = 0.002 if case == "curved_release" else 2e-6
            z[oracle.nq :] += np.tile([1e-6, -2e-6, 0.0], len(oracle.tet.free))
            e, r, h, _, _ = oracle.evaluate(z)
            candidate = run._states.copy(state)
            candidate.joint_q.assign(z[: oracle.nq])
            x = oracle.x.copy()
            x[oracle.tet.free] = z[oracle.nq :].reshape(-1, 3)
            candidate.particle_q.assign(x)
            pe, pr, _ = run._evaluate(candidate, 0.001)
            np.testing.assert_allclose(pr, r, atol=2e-10, rtol=3e-11)
            test.assertAlmostEqual(pe, e, delta=2e-11)
            p = np.random.default_rng(916).normal(size=len(z))
            p /= np.linalg.norm(p)
            for delta in (1e-8, 3e-9):
                plus = oracle.evaluate(z + delta * p)
                minus = oracle.evaluate(z - delta * p)
                np.testing.assert_allclose((plus[1] - minus[1]) / (2 * delta), h @ p, atol=2e-5, rtol=3e-6)
                if law == "C-EE":
                    test.assertAlmostEqual((plus[0] - minus[0]) / (2 * delta), r @ p, delta=2e-6)
            with patch.object(run, "_evaluate", side_effect=AssertionError("production called by reference")):
                oracle.step(0.001, np.full(oracle.nq, 0.001))


def test_graph_after_early_convergence(test, device):
    """Replay a new step after an initial converged candidate with no trial capture."""
    results = []
    for graph in (False, True):
        model = make_tiny("normal_loading", device)
        run = make_solver(model, "normal_loading", "C-EE", mode="production", graph=graph)
        state = run.create_state(dt=0.001)
        control = model.control()
        for step in range(3):
            set_control(control, "normal_loading", (step + 1) * 0.001, state.joint_q.numpy(), state.joint_qd.numpy())
            run.step(state, state, control, dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
        results.append(np.r_[state.joint_q.numpy(), state.particle_q.numpy().ravel()])
        if graph:
            test.assertTrue(run._contact._evaluated)
            test.assertGreater(run._graphs.summary()["segments"]["coupled_objective"]["replays"], 3)
    np.testing.assert_allclose(results[0], results[1], atol=1e-14, rtol=0)


def test_invalid_geometry_rollback(test, device):
    """Reject a sphere entering the bottom backface and preserve all state/history."""
    for law in ("C-EE", "C-IE", "C-II"):
        model = make_tiny("curved_release", device)
        run = make_solver(model, "curved_release", law)
        state = run.create_state(dt=0.001)
        state.joint_q.assign([0.03, 0.0])
        run.reset_state(state, dt=0.001)
        before = state_arrays(state)
        run.step(state, state, model.control(), dt=0.001)
        test.assertFalse(run.last_stats.converged)
        test.assertTrue(run.last_stats.rolled_back)
        test.assertIn("contact geometry", run.last_stats.failure_reason)
        for name, value in state_arrays(state).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)


class TestSolverOracle(unittest.TestCase):
    pass


for function in (test_reference_jacobian, test_invalid_geometry_rollback):
    add_function_test(TestSolverOracle, function.__name__, function, devices=get_test_devices())
add_function_test(
    TestSolverOracle,
    "test_graph_after_early_convergence",
    test_graph_after_early_convergence,
    devices=[device for device in get_test_devices() if device.is_cuda],
)

if __name__ == "__main__":
    unittest.main()
