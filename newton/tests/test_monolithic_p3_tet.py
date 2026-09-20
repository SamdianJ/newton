# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the internal P3 x-only variational lifecycle."""

import unittest
from dataclasses import replace
from itertools import pairwise, product
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicLinearStatus
from newton._src.solvers.monolithic.tet import evaluate_tet_residual
from newton._src.solvers.monolithic.tet_variational import acceptance
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle


def make_solver(model, *, mode="ArmijoOrResNorm", mass="lumped", **options):
    """Construct the explicit-mode private entry with a consistent density."""
    return SolverMonolithic._create_tet_only(
        model,
        globalization=mode,
        mass_mode=mass,
        tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device)
        if mass == "consistent"
        else None,
        **options,
    )


def test_x_only_lifecycle(test, device):
    """Advance actual zero-body states without an articulation or collision owner."""
    for mass in ("lumped", "consistent"):
        model = build_model(device, gravity=(0, 0, -9.81))
        solver = make_solver(model, mass=mass)
        state, out = solver._create_tet_state(), solver._create_tet_state()
        initial = state.particle_q.numpy().copy()
        test.assertIsNone(state.joint_q)
        test.assertIsNone(solver._articulation)
        test.assertIsNone(solver._contact)
        solver.step(state, out, None, None, 0.001)
        test.assertTrue(solver.last_stats.converged)
        test.assertTrue(np.isnan(solver.last_stats.convergence_ratio_q))
        np.testing.assert_allclose(out.particle_q.numpy(), initial + np.array([0, 0, -9.81e-6]), atol=1e-7)
        np.testing.assert_array_equal(state.particle_q.numpy(), initial)


def probe(model, solver, *, dt=0.001):
    """Start a physical context for evaluation without publishing a step."""
    state = solver._create_tet_state()
    solver._transaction.begin(state, solver._create_tet_state(), None, dt)
    solver._metrics = {"nonlinear_iterations": 0, "matrix_assembly_count": 0}
    return state, solver._transaction.accepted


def test_objective_derivatives(test, device):
    """Check complete E/r/H through representable coordinate trials with fixed-node elimination."""
    for smith, consistent, fixed in product((False, True), repeat=3):
        model = build_model(device, fixed=fixed, gravity=(1, -2, -9.81))
        solver = make_solver(
            model,
            mass="consistent" if consistent else "lumped",
            material_model="smith_log_stabilized" if smith else "kim_stable_no_log",
        )
        _state, accepted = probe(model, solver)
        oracle = TetOracle(model, solver._tet_physics)
        # A nonzero frozen force exercises the external work term.
        forces = np.full((model.particle_count, 3), 0.002, dtype=np.float32)
        solver._transaction.inputs.particle_f.assign(forces)
        origin = solver._transaction._original
        initial = origin.particle_q.numpy()
        f = np.array([[1.12, 0.07, 0], [0.01, 0.91, 0.03], [0, 0, 1.06]])
        deformed = initial @ f.T
        accepted.z.assign(deformed[oracle.free].astype(np.float32).ravel())
        accepted._recover(0.001)
        current = solver._evaluate_current(accepted, 0.001)
        x = accepted.state.particle_q.numpy()
        expected, gradient, hessian = oracle.evaluate(
            x, initial, origin.particle_qd.numpy(), [1, -2, -9.81], forces, 0.001
        )
        np.testing.assert_allclose(current.objective, expected, rtol=3e-6, atol=1e-12)
        np.testing.assert_allclose(accepted.residual.numpy(), gradient, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(hessian, hessian.T, rtol=1e-12, atol=1e-10)
        n = accepted.z.size
        delta = wp.zeros(n, dtype=float, device=device)
        r0 = accepted.residual.numpy().astype(float)
        z = accepted.z.numpy().astype(float)
        errors = []
        for eps in (2e-4, 6e-5, 2e-5, 6e-6):
            ge, he = np.zeros(n), np.zeros((n, n))
            for j in range(n):
                vals, steps = [], []
                for sign in (1, -1):
                    d = np.zeros(n, dtype=np.float32)
                    d[j] = sign * eps
                    delta.assign(d)
                    trial = solver._transaction.trial
                    trial.form_trial(accepted, delta, 1, 0.001)
                    result = solver._evaluate_trial(trial, 0.001)
                    vals.append((sum(result.objective), trial.residual.numpy().astype(float)))
                    steps.append(abs(float(trial.z.numpy()[j]) - z[j]))
                hp, hm = steps
                ge[j] = (
                    hm * (vals[0][0] - sum(current.objective)) / hp + hp * (sum(current.objective) - vals[1][0]) / hm
                ) / (hp + hm)
                he[:, j] = (hm * (vals[0][1] - r0) / hp + hp * (r0 - vals[1][1]) / hm) / (hp + hm)
            errors.append(
                (
                    np.linalg.norm(ge - gradient) / np.linalg.norm(gradient),
                    np.linalg.norm(he - hessian) / np.linalg.norm(hessian),
                )
            )
        gate = 5e-3 if device.is_cpu else 1e-2
        test.assertTrue(any(max(*a, *b) <= gate for a, b in pairwise(errors)), errors)
        # Trial requests must preserve the current production matrix.
        test.assertEqual(solver._metrics["matrix_assembly_count"], 1)


def test_modes_and_fixed_nodes(test, device):
    """Advance each material/mass/mode without moving fixed nodes or inventing a q block."""
    for mode, mass, material in product(
        ("Armijo", "ResNorm", "ArmijoOrResNorm"),
        ("lumped", "consistent"),
        ("kim_stable_no_log", "smith_log_stabilized"),
    ):
        model = build_model(device, fixed=True, gravity=(0, 0, -9.81))
        solver = make_solver(model, mode=mode, mass=mass, material_model=material)
        state = solver._create_tet_state()
        initial = state.particle_q.numpy().copy()
        for _ in range(3):
            solver.step(state, state, None, None, 0.001)
            test.assertTrue(solver.last_stats.converged, (mode, mass, material, solver.last_stats))
            np.testing.assert_array_equal(state.particle_q.numpy()[0], initial[0])
            np.testing.assert_array_equal(state.particle_qd.numpy()[0], 0)
            test.assertTrue(np.isnan(solver.last_stats.merit_q_final))
            test.assertEqual(solver.last_stats.variational["returned_state"], "accepted")


def test_freezing_and_stale_trials(test, device):
    """Keep E and r bound to frozen gravity/forces and reject old candidate generations."""
    model = build_model(device, gravity=(0, 0, -9.81))
    solver = make_solver(model)
    state, candidate = probe(model, solver)
    first = solver._evaluate_current(candidate, 0.001)
    residual = candidate.residual.numpy().copy()
    model.gravity.fill_(wp.vec3(30, 40, 50))
    state.particle_f.fill_(wp.vec3(20))
    second = solver._evaluate_trial(candidate, 0.001)
    np.testing.assert_array_equal(first.objective, second.objective)
    np.testing.assert_array_equal(candidate.residual.numpy(), residual)
    delta = wp.zeros(candidate.z.size, dtype=float, device=device)
    trial = solver._transaction.trial
    trial.form_trial(candidate, delta, 1, 0.001)
    candidate.generation += 1
    with test.assertRaisesRegex(ValueError, "stale"):
        solver._evaluate_objective(trial, 0.001)
    with test.assertRaises(ValueError):
        solver._evaluate_objective(candidate, 0.002)


def test_publish_policies(test, device):
    """Separate safe budget stops from convergence and restore every state array on strict failure."""
    for policy in ("strict", "legacy"):
        model = build_model(device, fixed=True, gravity=(0, 0, -9.81))
        solver = make_solver(model, newton_max_iterations=1, publish_policy=policy)
        state = solver._create_tet_state()
        before = {name: getattr(state, name).numpy().copy() for name in ("particle_q", "particle_qd", "particle_f")}
        with patch.object(solver, "_converged", return_value=False), test.assertLogs(level="WARNING"):
            solver.step(state, state, None, None, 0.001)
        test.assertFalse(solver.last_stats.converged)
        test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
        test.assertEqual(solver.last_stats.rolled_back, policy == "strict")
        if policy == "strict":
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(state, name).numpy(), value)
        # Every policy rolls back a hard failure and remains reusable afterward.
        before = state.particle_q.numpy().copy()
        solver._config = replace(solver._config, det_f_guard=2.0)
        with test.assertLogs(level="WARNING"):
            solver.step(state, state, None, None, 0.001)
        test.assertTrue(solver.last_stats.rolled_back)
        np.testing.assert_array_equal(state.particle_q.numpy(), before)
        solver._config = replace(solver._config, det_f_guard=0.2)
        solver.newton_max_iterations = 10
        solver.step(state, state, None, None, 0.001)
        test.assertTrue(solver.last_stats.converged)


def test_failure_classification(test, device):
    """Classify finite rejected trials, linear failures and unresolved small steps without publishing."""
    for failure in ("globalization", "linear", "small_step"):
        model = build_model(device, fixed=True, gravity=(0, 0, -9.81))
        solver = make_solver(model)
        state = solver._create_tet_state()
        initial = state.particle_q.numpy().copy()
        if failure == "linear":
            injection = patch.object(
                solver._linear, "factor_actor_preconditioner", return_value=MonolithicLinearStatus.NON_POSITIVE_PIVOT
            )
            terminal = "LINEAR_FAILED"
        elif failure == "globalization":
            injection = patch("newton._src.solvers.monolithic.solver_monolithic.acceptance", return_value=(False,) * 4)
            terminal = "GLOBALIZATION_FAILED"
        else:
            solver._config = replace(
                solver._config, step_tolerance_global=1e8, step_tolerance_q=1e8, step_tolerance_x=1e8
            )
            injection = patch.object(solver, "_converged", return_value=False)
            terminal = "STOPPED_SMALL_STEP"
        with injection, test.assertLogs(level="WARNING"):
            solver.step(state, state, None, None, 0.001)
        test.assertEqual(solver.last_stats.variational["termination"], terminal)
        test.assertFalse(solver.last_stats.variational["published"])
        test.assertFalse(solver.last_stats.converged)
        np.testing.assert_array_equal(state.particle_q.numpy(), initial)
        if failure != "small_step":
            test.assertEqual(solver.last_stats.matrix_assembly_count, 1)


def test_execution_paths(test, device):
    """Match changing physical inputs across eager diagnostic/production and CUDA graphs."""
    configurations = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        configurations.append(("production", True))
    scenes = []
    for mode, graph in configurations:
        model = build_model(device, fixed=True)
        solver = make_solver(
            model, mass="consistent", material_model="smith_log_stabilized", pcg_mode=mode, use_cuda_graph=graph
        )
        scenes.append((model, solver, solver._create_tet_state()))
    for i in range(8):
        for model, solver, state in scenes:
            model.set_gravity((0.1 * i, 0, -9.81))
            state.particle_f.fill_(wp.vec3(0.001 * i, 0, 0))
            solver.step(state, state, None, None, 0.001 if i < 4 else 0.0005)
            test.assertTrue(solver.last_stats.converged)
        for _, _, state in scenes[1:]:
            np.testing.assert_allclose(state.particle_q.numpy(), scenes[0][2].particle_q.numpy(), rtol=2e-5, atol=2e-6)
    if device.is_cuda:
        cache = scenes[-1][1]._segment_graphs
        test.assertNotIn("articulation", cache.captures)
        test.assertGreater(cache.replays["tet_objective"], cache.captures["tet_objective"])


def test_objective_only_and_invalid_inputs(test, device):
    """Reject unsupported scope and nonfinite inputs without touching the tangent or output state."""
    model = build_model(device)
    with test.assertRaises(TypeError):
        SolverMonolithic._create_tet_only(model)
    with test.assertRaises(ValueError):
        make_solver(model, mode="unknown")
    with test.assertRaises(ValueError):
        make_solver(model, publish_policy="unknown")
    model.particle_inv_mass.zero_()
    with test.assertRaises(ValueError):
        make_solver(model)
    model = build_model(device)
    solver = make_solver(model)
    state, candidate = probe(model, solver)
    before = solver._linear.k_global_scalar_bsr.values.numpy().copy()
    parts, _ = solver._evaluate_objective(candidate, 0.001)
    test.assertTrue(np.isfinite(parts).all())
    np.testing.assert_array_equal(solver._linear.k_global_scalar_bsr.values.numpy(), before)
    test.assertEqual(solver._metrics["matrix_assembly_count"], 0)
    solver._transaction._finished = True
    state.particle_f.fill_(wp.vec3(float("nan"), 0, 0))
    initial = state.particle_q.numpy().copy()
    with test.assertLogs(level="WARNING"):
        solver.step(state, state, None, None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONFINITE)
    test.assertEqual(solver.last_stats.variational["termination"], "DIVERGED")
    np.testing.assert_array_equal(state.particle_q.numpy(), initial)


def test_graph_rebinding_and_stale_objective(test, device):
    """Rebind objective scratch while preserving values and reject completed contexts."""
    if not device.is_cuda:
        return
    model = build_model(device, fixed=True)
    solver = make_solver(model, pcg_mode="production", use_cuda_graph=True)
    _state, candidate = probe(model, solver)
    first, _ = solver._evaluate_objective(candidate, 0.001)
    cache = solver._segment_graphs
    captures = cache.captures["tet_objective"]
    solver._tet_objective.parts = wp.clone(solver._tet_objective.parts)
    second, _ = solver._evaluate_objective(candidate, 0.001)
    test.assertEqual(cache.captures["tet_objective"], captures + 1)
    np.testing.assert_array_equal(first, second)
    solver._transaction._finished = True
    with test.assertRaises(ValueError):
        solver._evaluate_objective(candidate, 0.001)


def test_precision_publication(test, device):
    """Resolve the released-beam sub-ULP correction and certify the actual published State."""
    model = build_model(device, cantilever=True)
    solver = make_solver(model, mass="consistent", material_model="smith_log_stabilized")
    state = solver._create_tet_state()
    test.assertEqual(state.particle_q.dtype, wp.vec3d)
    with test.assertRaisesRegex(ValueError, "particle_q"):
        solver.step(model.state(), state, None, None, 0.001)
    with test.assertRaisesRegex(ValueError, "particle_q"):
        solver.step(state, model.state(), None, None, 0.001)
    initial = state.particle_q.numpy().copy()
    initial[:, 0] += 0.04 * np.sin(np.pi * initial[:, 0] / 2.4)
    state.particle_q.assign(initial)
    solver.step(state, state, None, None, 0.001)
    test.assertTrue(solver.last_stats.converged)
    np.testing.assert_array_equal(state.particle_q.numpy(), solver._transaction.accepted.state.particle_q.numpy())
    args = solver._tet_arguments(solver._transaction.accepted, 0.001)
    args["candidate_particle_q"] = state.particle_q
    evaluate_tet_residual(model, **args)
    residual = solver._particle_residual.numpy().ravel().astype(float)
    actual = np.linalg.norm(residual * solver._linear.scale.numpy()) / np.sqrt(len(residual))
    test.assertLessEqual(
        actual,
        solver._config.merit_absolute_global + solver._config.merit_relative_global * solver.last_stats.merit_reference,
    )
    test.assertLessEqual(
        actual, solver._config.merit_absolute_x + solver._config.merit_relative_x * solver.last_stats.merit_x_reference
    )
    np.testing.assert_allclose(
        state.particle_qd.numpy(), (state.particle_q.numpy() - initial) / 0.001, rtol=1e-12, atol=1e-14
    )


def test_stopping_precision(test, device):
    """Prevent an early PCG stop from accumulating excess error in a converged free fall."""
    model = build_model(device, gravity=(0, 0, -9.81))
    solver = make_solver(model, mass="consistent", material_model="kim_stable_no_log")
    state = solver._create_tet_state()
    initial = state.particle_q.numpy().copy()
    for _ in range(100):
        solver.step(state, state, None, None, 0.001)
        test.assertTrue(solver.last_stats.converged)
    expected = initial + np.array([0, 0, -9.81]) * 0.001**2 * 100 * 101 / 2
    test.assertLessEqual(float(np.max(np.abs(state.particle_q.numpy() - expected))), 5e-5)


def test_configuration_snapshot(test, device):
    """Keep E/r/B and fixed-node checks bound to owned construction data after in-place source edits."""
    model = build_model(device, fixed=True, gravity=(0, 0, -9.81))
    solver = make_solver(
        model,
        mass="consistent",
        pcg_mode="production" if device.is_cuda else "diagnostic",
        use_cuda_graph=device.is_cuda,
    )
    state, candidate = probe(model, solver)
    initial = solver._evaluate_current(candidate, 0.001)
    residual = candidate.residual.numpy().copy()
    matrix = solver._linear.k_global_scalar_bsr.values.numpy().copy()
    model.tet_indices.zero_()
    model.particle_world.fill_(-1)
    model.tet_materials.zero_()
    model.tet_poses.zero_()
    actual = solver._evaluate_current(candidate, 0.001)
    np.testing.assert_array_equal(initial.objective, actual.objective)
    np.testing.assert_array_equal(residual, candidate.residual.numpy())
    np.testing.assert_array_equal(matrix, solver._linear.k_global_scalar_bsr.values.numpy())
    trial = solver._evaluate_trial(candidate, 0.001)
    np.testing.assert_array_equal(initial.objective, trial.objective)
    solver._transaction._finished = True
    solver.step(state, state, None, None, 0.001)
    test.assertTrue(solver.last_stats.converged)
    model.tet_indices = wp.clone(model.tet_indices)
    with test.assertRaisesRegex(ValueError, "replaced"):
        solver.step(state, state, None, None, 0.001)


class TestP3Tet(unittest.TestCase):
    """Run real tet-only checks on the available CPU and CUDA devices."""

    def test_predicates(self):
        """Exercise A/R truth tables, nonnegative slopes, equality and nonfinite rejection."""
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        for mode in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            for a, r in product((False, True), repeat=2):
                value = acceptance(mode, 1, 0.5 if a else 2, -1, 1, 1, 0.5 if r else 2, **kw)
                self.assertEqual(value[:3], (a if mode == "Armijo" else r if mode == "ResNorm" else a or r, a, r))
        self.assertEqual(acceptance("ArmijoOrResNorm", 1, 2, 1, 1, 1, 0.5, **kw), (True, False, True, True))
        self.assertEqual(acceptance("ResNorm", 1, 1, 0, 1, 1, 1, **kw), (True, False, True, False))
        self.assertFalse(acceptance("ResNorm", 1, 0.5, -1, 1, 1, 1, **kw)[3])
        self.assertFalse(
            acceptance("Armijo", 1, 1, -1, 1, 1, 0.5, energy_noise=0.001, residual_noise=0, slope_noise=0)[3]
        )
        with self.assertRaises(ValueError):
            acceptance("ArmijoOrResNorm", 1, float("nan"), -1, 1, 1, 0, **kw)


for function in (
    test_x_only_lifecycle,
    test_objective_derivatives,
    test_modes_and_fixed_nodes,
    test_freezing_and_stale_trials,
    test_publish_policies,
    test_execution_paths,
    test_objective_only_and_invalid_inputs,
    test_graph_rebinding_and_stale_objective,
    test_precision_publication,
    test_failure_classification,
    test_stopping_precision,
    test_configuration_snapshot,
):
    add_function_test(TestP3Tet, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
