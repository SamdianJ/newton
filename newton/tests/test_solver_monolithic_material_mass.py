# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G2 exact constitutive, element mass and solver integration oracles."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import solver_monolithic as impl
from newton._src.solvers.monolithic.linear import MonolithicLinearStatus, MonolithicPcgWarmStart
from newton._src.solvers.monolithic.tet import (
    TetPhysicsWorkspace,
    _evaluate_smith,
    _nodal_hessian,
    _project_element_psd,
    assemble_tet_residual_tangent,
    evaluate_tet_residual,
    mat99,
    mat1212,
)
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.test_solver_monolithic_linear import _pcg_config
from newton.tests.test_solver_monolithic_tet import _setup
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _smith_probe(
    fs: wp.array[wp.mat33],
    pose: wp.array[wp.mat33],
    energy: wp.array[float],
    stress: wp.array[wp.mat33],
    raw: wp.array[mat99],
    projected: wp.array[mat1212],
):
    i = wp.tid()
    e, p, h = _evaluate_smith(fs[i], 1.0, 3846.1538, 5769.231)
    energy[i] = e
    stress[i] = p
    raw[i] = h
    k, success = _project_element_psd(_nodal_hessian(h, pose[0]))
    projected[i] = k
    wp.expect_eq(success, True)


def smith_energy(f):
    """Evaluate the independently rest-shifted float64 PRD energy density."""
    mu, lam = float(np.float32(3846.1538)), float(np.float32(5769.231))
    mh, lh = 4 * mu / 3, lam + 5 * mu / 6
    alpha = 1 + 3 * mh / (4 * lh)
    ic, j = np.sum(f * f), np.linalg.det(f)
    return mh / 2 * (ic - 3) + lh / 2 * ((j - alpha) ** 2 - (1 - alpha) ** 2) - mh / 2 * np.log((ic + 1) / 4)


def smith_stress(f):
    """Differentiate the float64 scalar energy with central differences."""
    p = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            d = np.zeros((3, 3))
            d[a, b] = 1e-5
            p[a, b] = (smith_energy(f + d) - smith_energy(f - d)) / 2e-5
    return p


def test_smith_oracle(test, device):
    """Check energy, stress, raw FD and full nodal spectral projection including inversion."""
    fs = np.array(
        [
            np.eye(3),
            np.diag([0.1, 0.4, 0.8]),
            np.diag([-0.3, 0.9, 1.2]),
            [[1.2, 0.3, 0.1], [0.0, 0.8, -0.2], [0.1, 0.2, 1.1]],
        ],
        dtype=np.float32,
    )
    pose = np.array([[2, 0.1, 0], [0, 3, 0.2], [0, 0, 4]], dtype=np.float32)
    arrays = [
        wp.array(fs, dtype=wp.mat33, device=device),
        wp.array([pose], dtype=wp.mat33, device=device),
        wp.empty(4, dtype=float, device=device),
        wp.empty(4, dtype=wp.mat33, device=device),
        wp.empty(4, dtype=mat99, device=device),
        wp.empty(4, dtype=mat1212, device=device),
    ]
    wp.launch(_smith_probe, 4, arrays, device=device)
    e, p, h, k = [v.numpy() for v in arrays[2:]]
    gradients = np.vstack((-pose.sum(axis=0), pose)).astype(float)
    b = np.zeros((9, 12))
    for a in range(4):
        for row in range(3):
            for col in range(3):
                b[3 * col + row, 3 * a + row] = gradients[a, col]
    for i, f32 in enumerate(fs):
        f = f32.astype(float)
        test.assertAlmostEqual(float(e[i]), smith_energy(f), delta=0.002)
        np.testing.assert_allclose(p[i], smith_stress(f), rtol=3e-5, atol=0.003)
        fd = np.empty((9, 9))
        for j in range(9):
            d = np.zeros((3, 3))
            d[j % 3, j // 3] = 1e-4
            fd[:, j] = ((smith_stress(f + d) - smith_stress(f - d)) / 2e-4).ravel(order="F")
        np.testing.assert_allclose(h[i], fd, rtol=3e-4, atol=0.02)
        nodal = b.T @ fd @ b
        eig, vec = np.linalg.eigh((nodal + nodal.T) / 2)
        expected = (vec * np.maximum(eig, 0)) @ vec.T
        test.assertLess(np.linalg.norm(k[i] - expected) / np.linalg.norm(expected), 2e-5)
        test.assertGreaterEqual(np.linalg.eigvalsh(k[i].astype(float)).min(), -2e-6 * np.linalg.norm(k[i], 2))
        for direction in np.random.default_rng(17).normal(size=(100, 12)):
            test.assertGreaterEqual(direction @ k[i].astype(float) @ direction, -1e-6 * np.dot(direction, direction))
    # The small-strain tangent is the input Lame law for both material modes.
    linear = np.zeros((9, 9))
    mu = float(np.float32(3846.1538))
    lam = float(np.float32(5769.231))
    for j in range(9):
        delta = np.zeros((3, 3))
        delta[j % 3, j // 3] = 1
        linear[:, j] = (mu * (delta + delta.T) + lam * np.trace(delta) * np.eye(3)).ravel(order="F")
    test.assertLess(np.linalg.norm(h[0] - linear) / np.linalg.norm(linear), 1e-6 if device.is_cpu else 1e-5)


def test_consistent_mass(test, device):
    """Check complete mass blocks and full gravity rows before Dirichlet elimination."""
    for fixed, multi in product((False, True), repeat=2):
        source = None
        if multi:
            builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
            builder.add_soft_mesh(
                pos=(0, 0, 0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0, 0, 0),
                vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02), (0.04, 0.03, 0.02)],
                indices=[0, 1, 2, 3, 4, 1, 3, 2],
                density=1000,
                k_mu=1,
                k_lambda=1,
                k_damp=0,
                tri_ke=0,
                tri_ka=0,
                tri_kd=0,
                tri_drag=0,
                tri_lift=0,
                edge_ke=0,
                edge_kd=0,
                particle_radius=0.001,
            )
            model = builder.finalize(device=device)
            source = SimpleNamespace(model=model, state=model.state())
        fixture, pattern, scatter, args = _setup(device, fixed=fixed, fixture=source)
        model = fixture.model
        model.tet_materials.assign(np.tile(np.array([0, 1e-10, 0], dtype=np.float32), (model.tet_count, 1)))
        physics = TetPhysicsWorkspace(
            model,
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            density=wp.full(model.tet_count, 1000, dtype=float, device=device),
        )
        args["workspace"].physics = physics
        n = model.particle_count
        ids = args["dynamic_particle_ids"].numpy()
        mass = np.zeros((n, n))
        for tet, coefficient in zip(model.tet_indices.numpy(), physics.coefficients.numpy(), strict=True):
            mass[np.ix_(tet, tet)] += coefficient * (np.ones((4, 4)) + np.eye(4))
        v = np.arange(n * 3).reshape(n, 3) * 0.003
        if fixed:
            v[0] = 0
        args["particle_qd_n"] = wp.array(v, dtype=wp.vec3, device=device)
        assemble_tet_residual_tangent(model, **args, scatter=scatter)
        gravity = model.gravity.numpy()[-1]
        expected = -mass[np.ix_(ids, ids)] @ v[ids] / args["dt"] - mass.sum(axis=1)[ids, None] * gravity
        np.testing.assert_allclose(args["residual_x"].numpy(), expected, rtol=3e-6, atol=1e-7)
        dense = np.zeros((len(ids) * 3, len(ids) * 3))
        np.add.at(dense, (pattern.global_rows - 2, pattern.global_columns - 2), scatter.global_values.numpy())
        expected_matrix = np.kron(mass[np.ix_(ids, ids)], np.eye(3)) / args["dt"] ** 2
        test.assertLess(
            np.linalg.norm(dense - expected_matrix) / np.linalg.norm(expected_matrix), 1e-6 if device.is_cpu else 1e-5
        )
        np.testing.assert_array_equal(scatter.global_values.numpy().reshape(-1, 3, 3), scatter.ax_values.numpy())
        before = scatter.global_values.numpy().copy()
        evaluate_tet_residual(model, **args)
        np.testing.assert_array_equal(scatter.global_values.numpy(), before)


def test_modes_integration(test, device):
    """Run all four modes through shared current/trial/scaling and frozen identities."""
    for material in ("kim_stable_no_log", "smith_log_stabilized"):
        for mass in ("lumped", "consistent"):
            fixture = build_tiny_cpu_fixture(device=device)
            model = fixture.model
            density = wp.full(model.tet_count, 1000, dtype=float, device=device) if mass == "consistent" else None
            solver = SolverMonolithic(
                model,
                collision_pipeline=MonolithicCollisionPipeline(model),
                contact_stiffness=1e5,
                material_model=material,
                mass_mode=mass,
                tet_rest_density=density,
            )
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
            test.assertTrue(solver.last_stats.converged)
            test.assertEqual(solver.last_stats.material_model, material)
            test.assertEqual(solver.last_stats.mass_mode, mass)
            test.assertEqual(len(solver.last_stats.tet_config_identity), 64)
            expected = solver._tet_workspace.physics.dynamic_mass_diagonal.numpy()
            np.testing.assert_allclose(
                solver._dynamic_diagonal.numpy()[2:].reshape(-1, 3),
                np.repeat((expected / 0.01**2)[:, None], 3, axis=1),
                rtol=2e-6,
            )
            workspace, generation = solver._linear, solver._generation
            oracle = workspace.densify_for_test(generation=generation)
            np.testing.assert_allclose(oracle.raw_matrix, oracle.raw_matrix.T, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(oracle.raw_matrix[:2, 2:], 0)
            test.assertGreater(np.linalg.eigvalsh(oracle.scaled_matrix).min(), 0)
            test.assertEqual(
                workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
                MonolithicLinearStatus.SUCCESS,
            )
            rhs = wp.array(np.linspace(0.1, 1, len(oracle.raw_matrix)).astype(np.float32), device=device)
            solution = wp.zeros_like(rhs)
            result = workspace.solve_pcg(
                rhs,
                solution,
                generation=generation,
                warm_start=MonolithicPcgWarmStart.ZERO,
                config=_pcg_config(linear_tolerance=solver.linear_tolerance, curvature_relative_tolerance=1e-12),
            )
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
            expected_solution = np.linalg.solve(oracle.scaled_matrix, rhs.numpy())
            # PRD tiny dense direct oracle: 1e-4 for PCG forward error.
            test.assertLess(
                np.linalg.norm(solution.numpy() - expected_solution) / np.linalg.norm(expected_solution), 1e-4
            )
            recovered = oracle.scale * solution.numpy()
            expected_delta = np.linalg.solve(oracle.regularized_matrix, rhs.numpy() / oracle.scale)
            test.assertLess(np.linalg.norm(recovered - expected_delta) / np.linalg.norm(expected_delta), 1e-4)
            old = model.tet_materials
            model.tet_materials = wp.clone(old)
            with test.assertRaises(ValueError):
                solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
            model.tet_materials = old


def test_density_and_lifecycle(test, device):
    """Reject invalid density, preserve snapshots, freeze trial assembly and roll back in place."""
    fixture = build_tiny_cpu_fixture(device=device)
    model = fixture.model
    pipeline = MonolithicCollisionPipeline(model)

    def create(density):
        return SolverMonolithic(
            model,
            collision_pipeline=pipeline,
            contact_stiffness=1e5,
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            tet_rest_density=density,
        )

    for density in (
        None,
        wp.full(2, 1000, dtype=float, device=device),
        wp.full(1, 1000, dtype=wp.float64, device=device),
        wp.full(1, -1, dtype=float, device=device),
        wp.full(1, float("nan"), dtype=float, device=device),
        wp.full(1, 2000, dtype=float, device=device),
    ):
        with test.assertRaises(ValueError):
            create(density)
    rho = wp.full(1, 1000, dtype=float, device=device)
    solver = create(rho)
    identity = solver.tet_config_identity
    rho.fill_(2000)
    np.testing.assert_array_equal(solver._tet_physics.density.numpy(), 1000)
    test.assertEqual(solver.tet_config_identity, identity)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    candidate = solver._transaction.accepted
    matrix = solver._linear.k_global_scalar_bsr.values.numpy().copy()
    solver._evaluate_trial(candidate, 0.01)
    baseline = candidate.residual.numpy().copy()
    allocator = model.device.get_allocator()
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as alloc:
        for _ in range(100):
            solver._evaluate_trial(candidate, 0.01)
    test.assertEqual(alloc.call_count, 0)
    np.testing.assert_array_equal(candidate.residual.numpy(), baseline)
    np.testing.assert_array_equal(solver._linear.k_global_scalar_bsr.values.numpy(), matrix)
    before = fixture.state.particle_q.numpy().copy()
    with patch.object(
        solver, "_iterate", side_effect=impl._StepFailure(SolverMonolithic.Status.NONFINITE, "G2 injected")
    ):
        solver.step(fixture.state, fixture.state, fixture.control, None, 0.01)
    test.assertTrue(solver.last_stats.rolled_back)
    np.testing.assert_array_equal(fixture.state.particle_q.numpy(), before)


class TestMaterialMass(unittest.TestCase):
    pass


for name, fn in [
    ("smith_oracle", test_smith_oracle),
    ("consistent_mass", test_consistent_mass),
    ("modes_integration", test_modes_integration),
    ("density_and_lifecycle", test_density_and_lifecycle),
]:
    add_function_test(TestMaterialMass, "test_" + name, fn, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
