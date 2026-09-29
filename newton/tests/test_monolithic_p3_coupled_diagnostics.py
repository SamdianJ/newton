# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify T4 reference derivatives, frozen input recipes and failure accounting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import build_model
from scripts.monolithic_reference.p3_coupled_fixtures import NORMAL, knots_at, make_solver, make_tiny, set_control
from scripts.monolithic_reference.p3_coupled_oracle import CoupledOracle
from scripts.monolithic_reference.summarize_p3_coupled import LIMITS, differences, errors, summarize
from scripts.monolithic_reference.verify_p3_coupled import assess, run_case, state_arrays


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


def test_normal_input(test, device):
    """Match every physical array of the original normal-loading builder."""
    expected = build_model(NORMAL, device=device)
    actual = make_tiny("normal_loading", device)
    for name in (
        "particle_q",
        "particle_qd",
        "particle_mass",
        "particle_inv_mass",
        "tet_indices",
        "tet_poses",
        "tet_materials",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "body_mass",
        "body_inertia",
        "shape_transform",
        "shape_margin",
        "shape_scale",
    ):
        np.testing.assert_array_equal(getattr(actual, name).numpy(), getattr(expected, name).numpy(), err_msg=name)


class TestCoupledDiagnostics(unittest.TestCase):
    def test_measurement_window(self):
        """Exclude the endpoint immediately before the final 200 physical steps."""
        force = np.array([[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]], dtype=float)
        rows = [
            {
                "time": (i + 1) * 0.001,
                "success": True,
                "penetration": 0.0,
                "min_det_f": 1.0,
                "friction_work": 0.0,
                "body_contact_force": force,
            }
            for i in range(1500)
        ]
        rows[1299]["body_contact_force"] = force * 200
        result = assess(rows, "cube1", 1500, 1.0)
        self.assertEqual(result["physical"], "PASS")
        self.assertEqual(result["force_error"], 0.0)

        rest = np.asarray(NORMAL["soft"]["rest_positions_m"])
        compressed = rest.copy()
        compressed[:3, 2] += 0.006
        rows = [
            {
                "time": (i + 1) * 0.001,
                "success": True,
                "penetration": 0.0,
                "min_det_f": 1.0,
                "friction_work": 0.0,
                "contact_load": 0.0 if i < 200 else 1.0,
                "slip": 0.0,
                "force_balance": 0.0,
                "q": np.array([0.003]),
                "x": compressed,
            }
            for i in range(1000)
        ]
        rows[899]["x"] = rest
        self.assertEqual(assess(rows, "normal_loading", 1000, None)["physical"], "PASS")

    def test_trajectory_metrics(self):
        """Keep metres/radians distinct and reject shifted or missing time grids."""
        a = {
            "time": np.array([0.001, 0.002]),
            "q": np.zeros((2, 2)),
            "qd": np.zeros((2, 2)),
            "com": np.zeros((2, 2, 3)),
            "x": np.zeros((2, 4, 3)),
            "rotation": np.tile(np.eye(3), (2, 2, 1, 1)),
        }
        b = {key: value.copy() for key, value in a.items()}
        b["q"][0] = [1e-4, 1e-4]
        b["x"][0, 0, :2] = [3e-5, 4e-5]
        measured = errors(a, b, "curved_release")
        self.assertGreater(measured["q_m"]["maximum"], LIMITS["q_m"])
        self.assertLess(measured["q_rad"]["maximum"], LIMITS["q_rad"])
        self.assertAlmostEqual(measured["x"]["maximum"], 5e-5)
        self.assertAlmostEqual(measured["x"]["rms"], 5e-5 / np.sqrt(2))
        b["time"] += 0.001
        with self.assertRaisesRegex(ValueError, "grids differ"):
            differences(a, b, "curved_release")

    def test_failure_and_sampling(self):
        """Reject incomplete/failed traces and sample every declared endpoint exactly."""
        self.assertFalse(assess([{"success": False}], "normal_loading", 1000, None)["complete"])
        self.assertFalse(assess([{"success": True}], "normal_loading", 1000, None)["complete"])
        for t, x in ((0.0, 0.0), (0.2, 0.003), (0.7, 0.012)):
            q, v = knots_at(t, [[0.0, 0.0], [0.2, 0.003], [0.7, 0.012]])
            self.assertAlmostEqual(q, x)
            self.assertAlmostEqual(v, 0.0)
        with tempfile.TemporaryDirectory() as folder:
            result = run_case("normal_loading", "C-EE", "cpu", Path(folder) / "partial", steps=2, reference=True)
            self.assertFalse(result["complete"])
            self.assertEqual(result["successful_steps"], 2)
            self.assertEqual(result["expected_steps"], 1000)
            comparison = summarize(Path(folder))
            self.assertFalse(comparison["reference_sensitivity"][0]["complete"])
            self.assertTrue(all(r["status"] == "NOT_CERTIFIED_INCOMPLETE" for r in comparison["refinement"]))


def test_cube_driver(test, device):
    """Build the frozen clamp and advance it without constructing an obsolete solver."""
    with tempfile.TemporaryDirectory() as folder:
        result = run_case("cube1", "C-EE", str(device), Path(folder) / "partial", steps=2)
        test.assertIsNone(result["failure"])
        test.assertEqual(result["successful_steps"], 2)
        test.assertEqual(result["expected_steps"], 1500)
        test.assertFalse(result["complete"])


for _device in get_test_devices():
    add_function_test(
        TestCoupledDiagnostics, "test_cube_driver", test_cube_driver, devices=[_device], check_output=False
    )
    add_function_test(TestCoupledDiagnostics, "test_reference_jacobian", test_reference_jacobian, devices=[_device])
    add_function_test(TestCoupledDiagnostics, "test_normal_input", test_normal_input, devices=[_device])
    add_function_test(
        TestCoupledDiagnostics, "test_invalid_geometry_rollback", test_invalid_geometry_rollback, devices=[_device]
    )
    if _device.is_cuda:
        add_function_test(
            TestCoupledDiagnostics,
            "test_graph_after_early_convergence",
            test_graph_after_early_convergence,
            devices=[_device],
        )

if __name__ == "__main__":
    unittest.main()
