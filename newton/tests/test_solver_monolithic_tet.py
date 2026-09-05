# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check exact tet derivatives separately from the production GN approximation."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.tet import (
    TetEvaluationStatus,
    TetScatterBuffers,
    _compute_cofactor_derivative,
    _evaluate_stable_neo_hookean,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
    evaluate_tet_residual,
    mat99,
    validate_tet_scope,
)
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _constitutive_oracle(
    deformation: wp.array[wp.mat33],
    energy: wp.array[float],
    stress: wp.array[wp.mat33],
    projected: wp.array[mat99],
    cofactor_derivative: wp.array[mat99],
):
    f = deformation[0]
    e, p, h = _evaluate_stable_neo_hookean(f, 1.0, 3846.15380859375, 5769.23095703125)
    energy[0] = e
    stress[0] = p
    projected[0] = h
    cofactor_derivative[0] = _compute_cofactor_derivative(f)


def _oracle(f, device):
    arrays = [
        wp.array([f], dtype=wp.mat33, device=device),
        wp.empty(1, dtype=float, device=device),
        wp.empty(1, dtype=wp.mat33, device=device),
        wp.empty(1, dtype=mat99, device=device),
        wp.empty(1, dtype=mat99, device=device),
    ]
    wp.launch(_constitutive_oracle, 1, arrays, device=device)
    return tuple(a.numpy()[0] for a in arrays[1:])


def _setup(device, fixed=False, fixture=None):
    if fixture is None:
        fixture = build_tiny_cpu_fixture(device=device)
    model = fixture.model
    if fixed:
        inv_mass = model.particle_inv_mass.numpy()
        inv_mass[0] = 0.0
        model.particle_inv_mass.assign(inv_mass)
    ids = np.arange(1 if fixed else 0, model.particle_count, dtype=np.int32)
    mapping = np.full(model.particle_count, -1, dtype=np.int32)
    mapping[ids] = np.arange(len(ids))
    validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=mapping)
    pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), mapping, x_dof_start=2)
    workspace = create_tet_assembly_workspace(pattern, tet_count=model.tet_count, device=device)
    scatter = TetScatterBuffers(
        wp.empty(len(pattern.ax_rows), dtype=wp.mat33, device=device),
        wp.empty(len(pattern.global_rows), dtype=float, device=device),
    )
    arguments = {
        "candidate_particle_q": wp.clone(fixture.state.particle_q),
        "particle_q_n": fixture.state.particle_q,
        "particle_qd_n": fixture.state.particle_qd,
        "frozen_particle_f": wp.clone(fixture.state.particle_f),
        "dynamic_particle_ids": wp.array(ids, dtype=int, device=device),
        "particle_to_dynamic": wp.array(mapping, dtype=int, device=device),
        "residual_x": wp.empty(len(ids), dtype=wp.vec3, device=device),
        "dt": 0.01,
        "min_det_f_guard": 0.05,
        "workspace": workspace,
    }
    return fixture, pattern, scatter, arguments


def _dense(pattern, scatter, count):
    owner = np.zeros((count, count), dtype=np.float64)
    scalar = np.zeros_like(owner)
    for row, col, block in zip(pattern.ax_rows, pattern.ax_columns, scatter.ax_values.numpy(), strict=True):
        owner[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += block
    np.add.at(scalar, (pattern.global_rows - 2, pattern.global_columns - 2), scatter.global_values.numpy())
    return owner, scalar


def test_constitutive_derivatives(test, device):
    """Calibrate central differences and distinguish raw Hessian from GN."""
    f = np.array([[1.12, 0.07, -0.03], [0.02, 0.91, 0.08], [0.04, -0.02, 1.06]], dtype=np.float32)
    energy, stress, projected, dc = _oracle(f, device)
    mu, lam = 3846.15380859375, 5769.23095703125
    j = np.linalg.det(f.astype(np.float64))
    expected_energy = 0.5 * mu * (np.sum(f.astype(np.float64) ** 2) - 3) - mu * (j - 1)
    expected_energy += 0.5 * (lam + mu) * (j - 1) ** 2
    test.assertAlmostEqual(float(energy), expected_energy, delta=0.003)
    raw = projected + ((lam + mu) * (j - 1) - mu) * dc
    errors = []
    for h in (0.02, 0.006, 0.002, 0.0006, 0.0002):
        grad = np.zeros(9)
        hessian = np.zeros((9, 9))
        for index in range(9):
            perturbation = np.zeros((3, 3), dtype=np.float32)
            perturbation[index % 3, index // 3] = h
            plus = _oracle(f + perturbation, device)
            minus = _oracle(f - perturbation, device)
            grad[index] = (plus[0] - minus[0]) / (2 * h)
            hessian[:, index] = ((plus[1] - minus[1]) / (2 * h)).flatten(order="F")
        errors.append(
            (
                np.linalg.norm(grad - stress.flatten(order="F")) / np.linalg.norm(stress),
                np.linalg.norm(hessian - raw) / np.linalg.norm(raw),
            )
        )
    test.assertLess(min(e[0] for e in errors), 5.0e-3)
    test.assertLess(min(e[1] for e in errors), 5.0e-3)
    direction = np.array([[0.8, 0.2, -0.4], [0.1, -0.7, 0.3], [0.5, -0.1, 0.6]], dtype=np.float32)
    directional_errors = []
    for h in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003):
        plus, minus = _oracle(f + h * direction, device), _oracle(f - h * direction, device)
        derivative = ((plus[1] - minus[1]) / (2 * h)).flatten(order="F")
        expected = raw @ direction.flatten(order="F")
        directional_errors.append(np.linalg.norm(derivative - expected) / np.linalg.norm(expected))
    test.assertLess(directional_errors[2], directional_errors[0] * 0.1)
    test.assertLess(min(directional_errors), 5.0e-3)
    np.testing.assert_allclose(raw, raw.T, atol=0.002)
    test.assertGreater(np.linalg.norm(raw - projected) / np.linalg.norm(raw), 0.1)
    test.assertGreaterEqual(np.linalg.eigvalsh(projected.astype(np.float64)).min(), -1.0e-6)
    _, rest_stress, _, _ = _oracle(np.eye(3, dtype=np.float32), device)
    np.testing.assert_allclose(rest_stress, 0.0, atol=0.001)


def test_assembly_sign_psd_and_owner_parity(test, device):
    """Keep all sixteen elastic blocks and expand exactly the same owner values."""
    fixture, pattern, scatter, args = _setup(device)
    model = fixture.model
    np.testing.assert_array_equal(model.tet_materials.numpy(), [[3846.15380859375, 5769.23095703125, 0]])
    test.assertEqual(len(pattern.ax_rows), 20)
    test.assertEqual(len(set(pattern.elastic_block_slots[0])), 16)
    positions = args["candidate_particle_q"].numpy().copy()
    positions[1] += (0.003, 0.001, -0.0005)
    args["candidate_particle_q"].assign(positions)
    args["frozen_particle_f"].assign(np.full((4, 3), 0.01, dtype=np.float32))
    assemble_tet_residual_tangent(model, **args, scatter=scatter)
    test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.SUCCESS)
    residual = args["residual_x"].numpy().copy()
    owner, scalar = _dense(pattern, scatter, 12)
    np.testing.assert_array_equal(owner, scalar)
    np.testing.assert_allclose(owner, owner.T, rtol=1.0e-6, atol=2.0e-5)
    inertia = np.repeat(model.particle_mass.numpy() / args["dt"] ** 2, 3)
    elastic = owner - np.diag(inertia)
    test.assertGreaterEqual(np.linalg.eigvalsh(elastic).min(), -1.0e-5)
    test.assertGreater(np.linalg.norm(elastic[:3, 3:6]), 1.0)
    mass = model.particle_mass.numpy()[:, None]
    gravity = model.gravity.numpy()[model.particle_world.numpy()]
    inertial_residual = mass * ((positions - fixture.state.particle_q.numpy()) / args["dt"] ** 2 - gravity) - 0.01
    elastic_residual = residual - inertial_residual
    np.testing.assert_allclose(elastic_residual.sum(axis=0), 0, atol=2.0e-6)
    physical_internal_force = -elastic_residual
    np.testing.assert_allclose(residual, inertial_residual - physical_internal_force, atol=1.0e-7)
    gradient = np.empty((4, 3))
    h = 2.0e-5
    for node in range(4):
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[node, axis] += h
            minus[node, axis] -= h
            args["candidate_particle_q"].assign(plus)
            evaluate_tet_residual(model, **args)
            e_plus = args["workspace"].tet_energy.numpy()[0]
            args["candidate_particle_q"].assign(minus)
            evaluate_tet_residual(model, **args)
            gradient[node, axis] = (e_plus - args["workspace"].tet_energy.numpy()[0]) / (2 * h)
    test.assertLess(np.linalg.norm(gradient - elastic_residual) / np.linalg.norm(gradient), 5.0e-3)
    args["candidate_particle_q"].assign(positions)
    evaluate_tet_residual(model, **args)
    np.testing.assert_array_equal(args["residual_x"].numpy(), residual)
    np.testing.assert_array_equal(_dense(pattern, scatter, 12)[0], owner)


def test_fixed_and_failures(test, device):
    """Reject invalid candidates without mutating fixed nodes or hiding failure."""
    fixture, pattern, scatter, args = _setup(device, fixed=True)
    test.assertEqual(len(pattern.ax_rows), 12)
    test.assertEqual(np.sum(pattern.elastic_block_slots < 0), 7)
    assemble_tet_residual_tangent(fixture.model, **args, scatter=scatter)
    test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.SUCCESS)
    before = args["candidate_particle_q"].numpy().copy()
    velocity = args["particle_qd_n"].numpy()
    velocity[0, 0] = 1.0e-30
    args["particle_qd_n"].assign(velocity)
    evaluate_tet_residual(fixture.model, **args)
    test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.INVALID_FIXED_PARTICLE)
    np.testing.assert_array_equal(args["candidate_particle_q"].numpy(), before)
    args["particle_qd_n"].zero_()
    for value, expected in ((-0.01, TetEvaluationStatus.DET_F_GUARD), (np.nan, TetEvaluationStatus.NONFINITE)):
        positions = before.copy()
        positions[3, 2] = value
        args["candidate_particle_q"].assign(positions)
        assemble_tet_residual_tangent(fixture.model, **args, scatter=scatter)
        test.assertEqual(args["workspace"].failure_flags.numpy()[0], expected)
    args["candidate_particle_q"].assign(before)
    evaluate_tet_residual(fixture.model, **args)
    test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.SUCCESS)
    test.assertAlmostEqual(args["workspace"].min_det_f.numpy()[0], 1.0)
    for invalid in (0.0, -1.0, np.nan, np.inf, 1.0e-200, 1.0e200):
        with test.assertRaises(ValueError):
            evaluate_tet_residual(fixture.model, **dict(args, dt=invalid))
    short_scatter = TetScatterBuffers(
        wp.empty(3, dtype=wp.mat33, device=device), wp.empty(27, dtype=float, device=device)
    )
    assemble_tet_residual_tangent(fixture.model, **args, scatter=short_scatter)
    test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.INVALID_SCATTER)


def test_full_element_raw_derivative(test, device):
    """Compare all twelve residual derivatives against the raw element oracle."""
    fixture, pattern, scatter, args = _setup(device)
    model = fixture.model
    f = np.array([[1.12, 0.07, -0.03], [0.02, 0.91, 0.08], [0.04, -0.02, 1.06]], dtype=np.float32)
    positions = fixture.state.particle_q.numpy().copy() @ f.T
    args["candidate_particle_q"].assign(positions)
    _, _, projected, dc = _oracle(f, device)
    inverse_rest = model.tet_poses.numpy()[0].astype(np.float64)
    volume = 1.0 / (6.0 * np.linalg.det(inverse_rest))
    mu, lam = model.tet_materials.numpy()[0, :2].astype(np.float64)
    material_raw = projected + ((mu + lam) * (np.linalg.det(f) - 1) - mu) * dc
    gradients = np.vstack((-inverse_rest.sum(axis=0), inverse_rest))
    b = np.zeros((9, 12))
    for node in range(4):
        for axis in range(3):
            for column in range(3):
                b[3 * column + axis, 3 * node + axis] = gradients[node, column]
    raw = volume * b.T @ material_raw @ b
    inertia = np.diag(np.repeat(model.particle_mass.numpy() / args["dt"] ** 2, 3))
    hessian = np.zeros((12, 12))
    for index in range(12):
        delta = np.zeros((4, 3), dtype=np.float32)
        delta[index // 3, index % 3] = 2.0e-5
        args["candidate_particle_q"].assign(positions + delta)
        evaluate_tet_residual(model, **args)
        plus = args["residual_x"].numpy().copy()
        args["candidate_particle_q"].assign(positions - delta)
        evaluate_tet_residual(model, **args)
        minus = args["residual_x"].numpy().copy()
        hessian[:, index] = ((plus - minus) / 4.0e-5).flatten()
    test.assertLess(np.linalg.norm(hessian - inertia - raw) / np.linalg.norm(raw), 5.0e-3)
    args["candidate_particle_q"].assign(positions)
    assemble_tet_residual_tangent(model, **args, scatter=scatter)
    production, _ = _dense(pattern, scatter, 12)
    test.assertGreater(np.linalg.norm(production - inertia - raw) / np.linalg.norm(raw), 0.1)


def test_guard_runtime_precision(test, device):
    """Reject a positive host guard that becomes zero in the runtime precision."""
    fixture, _, scatter, args = _setup(device)
    positions = args["candidate_particle_q"].numpy().copy()
    positions[3] = positions[0]
    args["candidate_particle_q"].assign(positions)
    for evaluate in (evaluate_tet_residual, assemble_tet_residual_tangent):
        with test.subTest(evaluate=evaluate.__name__), test.assertRaisesRegex(ValueError, "min_det_f_guard"):
            extra = {"scatter": scatter} if evaluate is assemble_tet_residual_tangent else {}
            evaluate(fixture.model, **dict(args, min_det_f_guard=1.0e-100), **extra)


def test_rest_volume_runtime_precision(test, device):
    """Refuse finite rest data whose runtime determinant overflows to infinity."""
    fixture, _, scatter, args = _setup(device)
    model = fixture.model
    model.tet_poses.assign(np.array([np.eye(3) * 1.0e13], dtype=np.float32))
    model.tet_materials.assign(np.array([[1.0e30, 1.0e30, 0]], dtype=np.float32))
    rest_positions = np.array([[0, 0, 0], [1.0e-13, 0, 0], [0, 1.0e-13, 0], [0, 0, 1.0e-13]], dtype=np.float32)
    model.particle_q.assign(rest_positions)
    args["particle_q_n"].assign(rest_positions)
    positions = rest_positions.copy()
    positions[1, 0] *= 1.2
    args["candidate_particle_q"].assign(positions)
    with test.subTest(stage="construction"), test.assertRaisesRegex(ValueError, "rest volume"):
        validate_tet_scope(
            model, dynamic_particle_ids=np.arange(4, dtype=np.int32), particle_to_dynamic=np.arange(4, dtype=np.int32)
        )
    for evaluate in (evaluate_tet_residual, assemble_tet_residual_tangent):
        extra = {"scatter": scatter} if evaluate is assemble_tet_residual_tangent else {}
        evaluate(model, **args, **extra)
        with test.subTest(stage=evaluate.__name__):
            test.assertEqual(args["workspace"].failure_flags.numpy()[0], TetEvaluationStatus.NONFINITE)


def test_shared_nodes_dense_oracle(test, device):
    """Accumulate two tetrahedra at shared nodes and solve the same dense matrix."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02), (0.04, 0.03, 0.02)],
        indices=[0, 1, 2, 3, 4, 1, 3, 2],
        density=1000.0,
        k_mu=3846.15380859375,
        k_lambda=5769.23095703125,
        k_damp=0.0,
        particle_radius=0.001,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    model = builder.finalize(device=device)
    fixture = SimpleNamespace(model=model, state=model.state())
    fixture, pattern, scatter, args = _setup(device, fixture=fixture)
    np.testing.assert_array_equal(model.tet_materials.numpy(), [[3846.15380859375, 5769.23095703125, 0]] * 2)
    test.assertEqual(len(pattern.ax_rows), 37)
    positions = fixture.state.particle_q.numpy().copy()
    positions[1, 0] += 0.002
    positions[4, 1] -= 0.001
    args["candidate_particle_q"].assign(positions)
    assemble_tet_residual_tangent(model, **args, scatter=scatter)
    owner, scalar = _dense(pattern, scatter, 15)
    np.testing.assert_array_equal(owner, scalar)
    test.assertGreater(np.linalg.eigvalsh(owner).min(), 0.0)
    residual = args["residual_x"].numpy().copy().flatten()
    correction = np.linalg.solve(scalar, -residual)
    np.testing.assert_allclose(owner @ correction, -residual, atol=1.0e-12)
    saved_values = scatter.global_values.numpy().copy()
    args["candidate_particle_q"].assign(fixture.state.particle_q)
    assemble_tet_residual_tangent(model, **args, scatter=scatter)
    args["candidate_particle_q"].assign(positions)
    assemble_tet_residual_tangent(model, **args, scatter=scatter)
    np.testing.assert_array_equal(scatter.global_values.numpy(), saved_values)
    np.testing.assert_allclose(args["residual_x"].numpy().flatten(), residual, atol=1.0e-7)


def test_scope_and_pattern_validation(test, device):
    """Refuse unsupported materials, inconsistent maps and singular rest data."""
    fixture, _, _, _ = _setup(device)
    model = fixture.model
    ids = np.arange(4, dtype=np.int32)

    def validate():
        validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=ids)

    original = model.tet_materials.numpy().copy()
    for row in ((-1, 5000, 0), (1000, -1000, 0), (1000, 5000, 0.1), (np.nan, 5000, 0)):
        model.tet_materials.assign(np.array([row], dtype=np.float32))
        with test.assertRaises(ValueError):
            validate()
    model.tet_materials.assign(original)
    poses = model.tet_poses.numpy().copy()
    model.tet_poses.zero_()
    with test.assertRaises(ValueError):
        validate()
    model.tet_poses.assign(poses)
    with test.assertRaises(ValueError):
        validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=ids[::-1].copy())
    with test.assertRaises(ValueError):
        build_tet_triplet_pattern(np.array([[0, 1, 2, 4]], dtype=np.int32), ids, x_dof_start=2)
    with test.assertRaises(ValueError):
        build_tet_triplet_pattern(model.tet_indices.numpy(), ids, x_dof_start=-1)


class TestMonolithicTet(unittest.TestCase):
    """Exercise the same component contracts on every available test device."""


for device in get_test_devices():
    for test_function in (
        test_constitutive_derivatives,
        test_assembly_sign_psd_and_owner_parity,
        test_fixed_and_failures,
        test_full_element_raw_derivative,
        test_guard_runtime_precision,
        test_rest_volume_runtime_precision,
        test_shared_nodes_dense_oracle,
        test_scope_and_pattern_validation,
    ):
        add_function_test(TestMonolithicTet, test_function.__name__, test_function, devices=[device])


if __name__ == "__main__":
    unittest.main()
