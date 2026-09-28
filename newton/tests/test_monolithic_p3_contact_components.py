# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production FP64 contact/tet components against independent coordinate oracles."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact_variational import ContactObjective, ContactStatus
from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearWorkspace,
)
from newton._src.solvers.monolithic.tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from newton._src.solvers.monolithic.tet_precision import TetObjective64
from newton.tests.test_solver_monolithic_collision import _scene
from newton.tests.test_solver_monolithic_tet import _dense
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p3_contact_oracle import evaluate, penalty, rotation
from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle


def contact(model, law, *, fixed=False, mu=0.5):
    mapping = np.arange(model.particle_count, dtype=np.int32)
    if fixed:
        mapping[0], mapping[1:] = -1, np.arange(model.particle_count - 1)
        mass = model.particle_inv_mass.numpy()
        mass[0] = 0
        model.particle_inv_mass.assign(mass)
    return ContactObjective(
        model,
        mapping,
        q_dof_count=2,
        contact_model=law,
        normal_stiffness=1e7,
        friction_coefficient=mu,
        friction_velocity_scale=0.01,
        smoothing_width=1e-4,
        particle_radius=0.001,
    )


def state(model, x, q):
    device = model.device
    pose = np.array([[q[1], 0, 0, 0, np.sin(q[0] / 2), 0, np.cos(q[0] / 2)]])
    jt, jw = np.zeros((1, 2, 3)), np.zeros((1, 2, 3))
    jt[0, 1, 0], jw[0, 0, 1] = 1, 1
    return (
        wp.array(x, dtype=wp.vec3d, device=device),
        wp.array(pose, dtype=wp.transformd, device=device),
        wp.array([[q[1], 0, 0]], dtype=wp.vec3d, device=device),
        wp.array(jt, dtype=wp.vec3d, device=device),
        wp.array(jw, dtype=wp.vec3d, device=device),
    )


def fixture(device, kind):
    model, _ = _scene(device, shape=kind)
    transforms = model.shape_transform.numpy()
    if kind == "sphere":
        transforms[0, 2] = -0.03
    elif kind == "box":
        transforms[0, 2] = -0.01
    model.shape_transform.assign(transforms)
    x = model.particle_q.numpy().astype(float)
    # Place a small face near the top of the curved collider, with outward -Z normal.
    x[:, :2] *= 0.01
    x[:, 2] -= 0.0012
    return model, x


def tangent(objective, *, capacity=None, triplets=None):
    layout = MonolithicLinearLayout(2, (objective.residual.size - 2) // 3)
    count = 3 * objective.count if capacity is None else capacity
    workspace = MonolithicLinearWorkspace(
        layout,
        MonolithicLinearCapacities(4 + count * 121 if triplets is None else triplets, 0, count, 200, 202),
        objective.mapping,
        device=objective.model.device,
    )
    generation = MonolithicLinearGeneration(1, 1, 1, 1)
    assembly = workspace.begin_assembly(generation)
    objective.assemble_tangent(assembly)
    return workspace, generation, assembly


def test_oracle_and_actual_trials(test, device):
    rng = np.random.default_rng(893)
    for kind in ("plane", "sphere", "box"):
        for law in ("C-EE", "C-IE", "C-II"):
            for fixed in (False, True):
                with test.subTest(kind=kind, law=law, fixed=fixed):
                    model, xn = fixture(device, kind)
                    obj = contact(model, law, fixed=fixed)
                    qn, q = np.zeros(2), np.array([0.003, 2e-6])
                    x = xn + np.array([1e-6, 2e-6, 1e-6])
                    if fixed:
                        x[0] = xn[0]
                    obj.begin_step(*state(model, xn, qn))
                    obj.evaluate(*state(model, x, q), dt=0.001)
                    test.assertEqual(obj.flags.numpy()[0], ContactStatus.SUCCESS)
                    oracle = evaluate(model, q, x, qn, xn, law, kinds=[kind], mapping=obj.mapping.numpy())
                    np.testing.assert_allclose(obj.energy.numpy(), oracle[0], rtol=3e-10, atol=1e-15)
                    np.testing.assert_allclose(obj.residual.numpy(), oracle[1], rtol=3e-10, atol=1e-12)
                    workspace, generation, assembly = tangent(obj)
                    test.assertEqual(int(workspace.finalize_assembly(generation=generation).status), 0)
                    matrix = workspace.densify_for_test(generation=generation).raw_matrix
                    np.testing.assert_allclose(matrix, oracle[3], rtol=2e-5, atol=2e-4)
                    test.assertLessEqual(
                        np.linalg.norm(matrix - matrix.T), np.finfo(np.float32).eps * np.linalg.norm(matrix)
                    )
                    test.assertGreaterEqual(
                        np.linalg.eigvalsh(matrix.astype(float)).min(), -1e-6 * np.linalg.norm(matrix)
                    )
                    factors = assembly.contact_factors
                    test.assertTrue(np.all(factors.candidate_tid.numpy()[: factors.count.numpy()[0]] >= 0))
                    direction = rng.normal(size=obj.residual.size)
                    direction /= np.linalg.norm(direction)
                    free = np.flatnonzero(obj.mapping.numpy() >= 0)
                    errors_r, errors_e = [], []
                    for h in (2e-8, 5e-9, 1e-9):
                        values = []
                        for sign in (1, -1):
                            xt = x.copy()
                            xt[free] += sign * h * direction[2:].reshape(-1, 3)
                            obj.evaluate(*state(model, xt, q + sign * h * direction[:2]), dt=0.001)
                            test.assertEqual(obj.flags.numpy()[0], 0)
                            values.append((obj.energy.numpy().sum(), obj.residual.numpy().copy()))
                        fd = (values[0][1] - values[1][1]) / (2 * h)
                        errors_r.append(
                            np.linalg.norm(fd - oracle[2] @ direction) / max(1, np.linalg.norm(oracle[2] @ direction))
                        )
                        errors_e.append(abs((values[0][0] - values[1][0]) / (2 * h) - oracle[1] @ direction))
                    test.assertGreaterEqual(sum(e < 1e-5 for e in errors_r), 2, errors_r)
                    if law == "C-EE":
                        test.assertGreaterEqual(sum(e < 1e-7 for e in errors_e), 2, errors_e)
                    test.assertEqual(obj.conservative_energy, law == "C-EE")


def test_friction_radial_tangent(test, device):
    """Match actual slip derivatives through zero, smoothing boundary and sliding."""
    along = np.array([0.6, 0.8, 0.0])
    across = np.array([-0.8, 0.6, 0.0])
    for law in ("C-EE", "C-IE", "C-II"):
        model, xn = fixture(device, "plane")
        obj = contact(model, law)
        q = np.zeros(2)
        obj.begin_step(*state(model, xn, q))
        # Revisit zero and reverse slip to detect stale weights/directions.
        for ratio in (0, 1e-10, 0.25, 0.9999, 1, 1.0001, 2, -2, 0):
            with test.subTest(law=law, ratio=ratio):
                x = xn + ratio * 1e-5 * along
                obj.evaluate(*state(model, x, q), dt=0.001)
                test.assertEqual(obj.flags.numpy()[0], ContactStatus.SUCCESS)
                expected = evaluate(model, q, x, q, xn, law, kinds=["plane"])
                np.testing.assert_allclose(obj.energy.numpy(), expected[0], atol=1e-15)
                np.testing.assert_allclose(obj.residual.numpy(), expected[1], atol=1e-12)
                workspace, generation, assembly = tangent(obj)
                test.assertEqual(int(workspace.finalize_assembly(generation=generation).status), 0)
                matrix = workspace.densify_for_test(generation=generation).raw_matrix.astype(float)
                test.assertTrue(np.isfinite(matrix).all())
                test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * np.linalg.norm(matrix))
                factors = assembly.contact_factors
                count = int(factors.count.numpy()[0])
                test.assertLessEqual(count, 3 * obj.count)
                test.assertTrue(np.all(factors.weights.numpy()[:count] > 0))
                for axis in (along, across):
                    direction = np.concatenate((np.zeros(2), np.tile(axis, len(x))))
                    values = []
                    h = 1e-10
                    for sign in (1, -1):
                        obj.evaluate(*state(model, x + sign * h * axis, q), dt=0.001)
                        test.assertEqual(obj.flags.numpy()[0], ContactStatus.SUCCESS)
                        values.append(obj.residual.numpy().copy())
                    derivative = (values[0] - values[1]) / (2 * h)
                    # Only the x block: the q rows also contain omitted FK geometry.
                    error = np.linalg.norm((matrix @ direction - derivative)[2:])
                    scale = max(1, np.linalg.norm(matrix[2:, 2:]) * np.linalg.norm(direction))
                    test.assertLess(error / scale, 3e-5)
                    if abs(ratio) >= 1.0001 and axis is along:
                        test.assertLess(np.linalg.norm((matrix @ direction)[2:]) / scale, 1e-6)


def test_friction_normal_motion_limit(test, device):
    """Keep the zero-slip tangent isotropic despite normal projection roundoff."""
    for law in ("C-EE", "C-IE", "C-II"):
        model, xn = fixture(device, "sphere")
        obj = contact(model, law)
        q = np.zeros(2)
        obj.begin_step(*state(model, xn, q))
        center = model.shape_transform.numpy()[0, :3].astype(float)
        for scale in (1.001, 1.0000001, 1.0):
            with test.subTest(law=law, scale=scale):
                x = center + scale * (xn - center)
                obj.evaluate(*state(model, x, q), dt=0.001)
                test.assertEqual(obj.flags.numpy()[0], ContactStatus.SUCCESS)
                u = obj.current.y.numpy() - obj.start.y.numpy()
                n = obj.normal.numpy()
                slip = u - np.sum(n * u, axis=1)[:, None] * n
                test.assertLess(np.linalg.norm(slip), 1e-13)
                test.assertGreater(obj.weights.numpy()[:, 1].max(), 0)
                workspace, generation, _ = tangent(obj)
                test.assertEqual(int(workspace.finalize_assembly(generation=generation).status), 0)
                matrix = workspace.densify_for_test(generation=generation).raw_matrix
                expected = evaluate(model, q, x, q, xn, law, kinds=["sphere"])[3]
                # FP32 factor sums can cancel in individual off-diagonal entries.
                test.assertLess(
                    np.linalg.norm(matrix - expected), 5 * np.finfo(np.float32).eps * np.linalg.norm(expected)
                )


def test_tet64_oracle(test, device):
    for material in ("kim_stable_no_log", "smith_log_stabilized"):
        for mass in ("lumped", "consistent"):
            for fixed in (False, True):
                with test.subTest(material=material, mass=mass, fixed=fixed):
                    model = build_model(device, fixed=fixed, gravity=(0, 0, -9.81))
                    obj = TetObjective64(
                        model,
                        material_model=material,
                        mass_mode=mass,
                        density=wp.full(model.tet_count, 1000, dtype=float, device=device)
                        if mass == "consistent"
                        else None,
                    )
                    xn = model.particle_q.numpy().astype(float)
                    x = xn @ np.array([[1.02, 0.001, 0], [0, 0.99, 0.01], [0, 0, 1.01]])
                    vn = np.full_like(x, 0.02)
                    if fixed:
                        x[0], vn[0] = xn[0], 0
                    forces = np.full_like(x, 0.003, dtype=np.float32)
                    inputs = [
                        wp.array(a, dtype=dtype, device=device)
                        for a, dtype in ((x, wp.vec3d), (xn, wp.vec3d), (vn, wp.vec3d), (forces, wp.vec3))
                    ]
                    obj.evaluate(*inputs, model.gravity, dt=0.001)
                    test.assertEqual(obj.flags.numpy()[0], 0)
                    oracle = TetOracle(model, obj.physics)
                    parts, r, _ = oracle.evaluate(x, xn, vn, model.gravity.numpy()[0], forces, 0.001)
                    np.testing.assert_allclose(
                        obj.parts.numpy(), [parts[0], parts[1], parts[2] + parts[3]], rtol=1e-11, atol=1e-15
                    )
                    np.testing.assert_allclose(obj.residual.numpy().ravel(), r, rtol=1e-11, atol=1e-12)
                    # Sub-FP32 coordinate corrections remain visible to the direct residual.
                    dx = np.zeros_like(x)
                    dx[-1, 2] = 1e-11
                    inputs[0].assign(x + dx)
                    obj.evaluate(*inputs, model.gravity, dt=0.001)
                    test.assertGreater(np.linalg.norm(obj.residual.numpy().ravel() - r), 1e-9)


def test_law_branches_and_failures(test, device):
    for law in ("C-EE", "C-IE", "C-II"):
        model, xn = fixture(device, "mixed")
        obj = contact(model, law)
        q = np.zeros(2)
        obj.begin_step(*state(model, xn, q))
        frozen = obj.start.y.numpy().copy()
        test.assertEqual(obj.count, 24)
        for ratio in (0, 0.25, 1, 2):
            x = xn + np.array([ratio * 1e-5, 0, 0])
            obj.evaluate(*state(model, x, q), dt=0.001)
            oracle = evaluate(model, q, x, q, xn, law, kinds=["plane", "box"])
            np.testing.assert_allclose(obj.residual.numpy(), oracle[1], atol=1e-12)
            test.assertEqual(obj.flags.numpy()[0], 0)
            # Far box is still queried; zero force does not remove a key.
            test.assertTrue(np.all(obj.current.distance.numpy()[1::2] > 10))
        np.testing.assert_array_equal(obj.start.y.numpy(), frozen)
        for capacity, triplets in ((0, None), (None, 4)):
            workspace, generation, _ = tangent(obj, capacity=capacity, triplets=triplets)
            test.assertEqual(obj.flags.numpy()[0], ContactStatus.CAPACITY)
            test.assertNotEqual(workspace.finalize_assembly(generation=generation).status, 0)
            obj.evaluate(*state(model, x, q), dt=0.001)
        for bad in ("nan", "degenerate", "backface", "quaternion"):
            inputs = list(state(model, xn, q))
            invalid = xn.copy()
            if bad == "nan":
                invalid[0, 0] = np.nan
            elif bad == "degenerate":
                invalid[1] = invalid[0]
            elif bad == "backface":
                invalid[[1, 2]] = invalid[[2, 1]]
            else:
                pose = inputs[1].numpy()
                pose[0, 6] = 2
                inputs[1].assign(pose)
            inputs[0].assign(invalid)
            obj.evaluate(*inputs, dt=0.001)
            test.assertNotEqual(obj.flags.numpy()[0], 0, bad)
            np.testing.assert_array_equal(obj.start.y.numpy(), frozen)
        obj.evaluate(*state(model, xn, q), dt=0.001)
        test.assertEqual(obj.flags.numpy()[0], 0)
        # C-EE alone carries start load through release and delays new friction.
        if law == "C-EE":
            released = xn + np.array([2e-5, 0, 0.001])
            obj.evaluate(*state(model, released, q), dt=0.001)
            test.assertEqual(obj.weights.numpy()[:, 0].max(), 0)
            test.assertGreater(obj.weights.numpy()[:, 1].max(), 0)
            obj.begin_step(*state(model, released, q))
            obj.evaluate(*state(model, xn, q), dt=0.001)
            test.assertGreater(obj.weights.numpy()[:, 0].max(), 0)
            test.assertEqual(obj.weights.numpy()[:, 1].max(), 0)
        scale = model.shape_scale.numpy()
        scale[-1] *= 2
        model.shape_scale.assign(scale)
        with test.assertRaisesRegex(ValueError, "changed"):
            obj.begin_step(*state(model, xn, q))
    model, xn = fixture(device, "plane")
    obj = contact(model, "C-II", mu=0)
    obj.begin_step(*state(model, xn, q))
    obj.evaluate(*state(model, xn + np.array([1e-4, 0, 0]), q), dt=0.001)
    np.testing.assert_array_equal(obj.weights.numpy()[:, 1:], 0)
    for kind in ("finite_plane", "ellipsoid"):
        invalid, _ = _scene(device, shape=kind)
        with test.assertRaises(ValueError):
            contact(invalid, "C-EE")


def test_start_domain_and_boundary_branches(test, device):
    model, xn = fixture(device, "plane")
    obj = contact(model, "C-EE")
    q = np.zeros(2)
    bad = xn.copy()
    bad[[1, 2]] = bad[[2, 1]]
    obj.begin_step(*state(model, bad, q))
    obj.evaluate(*state(model, xn, q), dt=0.001)
    test.assertEqual(obj.flags.numpy()[0], ContactStatus.INVALID_GEOMETRY)
    # Verify actual scalar branches on both sides of normal activation/smoothing.
    for gap in (-1.001e-4, -0.999e-4, 0, 0.999e-4, 1.001e-4):
        x = xn.copy()
        x[:, 2] += 0.001 + gap - xn[0, 2]
        obj.begin_step(*state(model, x, q))
        obj.evaluate(*state(model, x, q), dt=0.001)
        expected = evaluate(model, q, x, q, x, "C-EE", kinds=["plane"])
        test.assertEqual(obj.flags.numpy()[0], 0)
        np.testing.assert_allclose(obj.residual.numpy(), expected[1], atol=1e-12)
    # Common rigid motion gives zero relative slip even with rotation.
    obj.begin_step(*state(model, xn, q))
    moved_q = np.array([0.23, 0.12])
    moved_x = xn @ rotation(moved_q[0]).T + [moved_q[1], 0, 0]
    obj.evaluate(*state(model, moved_x, moved_q), dt=0.001)
    np.testing.assert_allclose(obj.current.y.numpy(), obj.start.y.numpy(), atol=3e-17)
    test.assertLess(obj.energy.numpy()[:, 1].sum(), 1e-25)
    # No allowed shapes is a valid zero-contribution component.
    model.shape_flags.zero_()
    empty = contact(model, "C-EE")
    empty.begin_step(*state(model, xn, q))
    empty.evaluate(*state(model, xn, q), dt=0.001)
    test.assertEqual(empty.count, 0)
    test.assertEqual(np.linalg.norm(empty.residual.numpy()), 0)


def test_tet_tangent_and_domain(test, device):

    for material in ("kim_stable_no_log", "smith_log_stabilized"):
        model = build_model(device, fixed=True)
        obj = TetObjective64(
            model,
            material_model=material,
            mass_mode="consistent",
            density=wp.full(model.tet_count, 1000, dtype=float, device=device),
        )
        xn = model.particle_q.numpy().astype(float)
        x = xn * [1.02, 0.97, 1.01]
        inputs = [wp.array(a, dtype=wp.vec3d, device=device) for a in (x, xn, np.zeros_like(x))]
        force = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        obj.evaluate(*inputs, force, model.gravity, dt=0.001)
        expected_residual = obj.residual.numpy().copy()
        pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), obj.mapping.numpy(), x_dof_start=2)
        workspace = create_tet_assembly_workspace(pattern, tet_count=model.tet_count, device=device)
        workspace.physics = obj.physics
        scatter = TetScatterBuffers(
            wp.empty(len(pattern.ax_rows), dtype=wp.mat33, device=device),
            wp.empty(len(pattern.global_rows), dtype=float, device=device),
        )
        assemble_tet_residual_tangent(
            model,
            candidate_particle_q=inputs[0],
            particle_q_n=inputs[1],
            particle_qd_n=inputs[2],
            frozen_particle_f=force,
            dynamic_particle_ids=obj.ids,
            particle_to_dynamic=obj.mapping,
            residual_x=wp.empty(obj.ids.size, dtype=wp.vec3, device=device),
            dt=0.001,
            min_det_f_guard=0.2,
            workspace=workspace,
            scatter=scatter,
            frozen_gravity=model.gravity,
        )
        test.assertEqual(workspace.failure_flags.numpy()[0], 0)
        owner, global_b = _dense(pattern, scatter, 3 * obj.ids.size)
        np.testing.assert_array_equal(owner, global_b)
        np.testing.assert_allclose(owner, owner.T, atol=1e-3)
        test.assertGreater(np.linalg.eigvalsh(owner).min(), 0)
        np.testing.assert_array_equal(obj.residual.numpy(), expected_residual)
        # Actual E/r finite differences with unchanged fixed node.
        direction = np.random.default_rng(811).normal(size=x.shape)
        direction[0] = 0
        for h in (1e-7, 3e-8):
            energies = []
            for sign in (1, -1):
                inputs[0].assign(x + sign * h * direction)
                obj.evaluate(*inputs, force, model.gravity, dt=0.001)
                energies.append(obj.parts.numpy().sum())
            test.assertAlmostEqual(
                (energies[0] - energies[1]) / (2 * h), np.sum(expected_residual * direction[1:]), delta=1e-7
            )
        for failure in ("fixed", "det", "nonfinite"):
            invalid = x.copy()
            if failure == "fixed":
                invalid[0, 0] += 1e-10
            elif failure == "det":
                invalid[3] = invalid[0]
            else:
                invalid[3, 0] = np.nan
            inputs[0].assign(invalid)
            obj.evaluate(*inputs, force, model.gravity, dt=0.001)
            test.assertNotEqual(obj.flags.numpy()[0], 0)
        inputs[0].assign(x)
        obj.evaluate(*inputs, force, model.gravity, dt=0.001)
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                obj.evaluate(*inputs, force, model.gravity, dt=0.001)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(obj.residual.numpy(), expected_residual, atol=1e-12)


def test_component_graph(test, device):
    if wp.get_device(device).is_cpu:
        test.skipTest("CUDA graph requires CUDA")
    model, xn = fixture(device, "sphere")
    obj = contact(model, "C-EE")
    inputs = state(model, xn, np.zeros(2))
    obj.begin_step(*inputs)
    obj.evaluate(*inputs, dt=0.001)
    with wp.ScopedCapture(device=device) as capture:
        obj.evaluate(*inputs, dt=0.001)
    for shift in (2e-6, 0.001, -0.1):
        inputs[0].assign(xn + np.array([2e-6, 0, shift]))
        obj.evaluate(*inputs, dt=0.001)
        expected = (
            obj.energy.numpy(),
            obj.residual.numpy(),
            obj.flags.numpy(),
            obj.weights.numpy(),
            obj.tangent.numpy(),
        )
        wp.capture_launch(capture.graph)
        actual = (obj.energy.numpy(), obj.residual.numpy(), obj.flags.numpy(), obj.weights.numpy(), obj.tangent.numpy())
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want, atol=1e-13)


def test_zero_smoothing(test, device):
    """Preserve the frozen quadratic normal law without introducing smoothing."""
    model, xn = fixture(device, "plane")
    kwargs = {
        "q_dof_count": 2,
        "contact_model": "C-EE",
        "normal_stiffness": 1e7,
        "friction_coefficient": 0.0,
        "friction_velocity_scale": 0.01,
        "particle_radius": 0.001,
    }
    with test.assertRaises(ValueError):
        ContactObjective(model, np.arange(4, dtype=np.int32), smoothing_width=-1e-4, **kwargs)
    obj = ContactObjective(model, np.arange(4, dtype=np.int32), smoothing_width=0.0, **kwargs)
    for height in (0.0008, 0.001, 0.0012):
        x = xn.copy()
        x[:, 2] += height - xn[0, 2]
        inputs = state(model, x, np.zeros(2))
        obj.begin_step(*inputs)
        obj.evaluate(*inputs, dt=0.001)
        test.assertEqual(obj.flags.numpy()[0], 0)
        gaps = obj.current.distance.numpy() - 0.001
        expected = [penalty(g, 1e7 * obj.area.numpy()[i // 3] / 3, 0.0) for i, g in enumerate(gaps)]
        np.testing.assert_allclose(obj.energy.numpy()[:, 0], np.array(expected)[:, 0], atol=1e-15)
        np.testing.assert_allclose(obj.weights.numpy()[:, 0], np.array(expected)[:, 2], atol=1e-12)


class TestContactComponents(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestContactComponents, "test_friction_normal_motion_limit", test_friction_normal_motion_limit, devices=[_device]
    )
    add_function_test(
        TestContactComponents, "test_friction_radial_tangent", test_friction_radial_tangent, devices=[_device]
    )
    add_function_test(TestContactComponents, "test_zero_smoothing", test_zero_smoothing, devices=[_device])
    add_function_test(
        TestContactComponents, "test_oracle_and_actual_trials", test_oracle_and_actual_trials, devices=[_device]
    )
    add_function_test(TestContactComponents, "test_tet64_oracle", test_tet64_oracle, devices=[_device])
    add_function_test(
        TestContactComponents, "test_law_branches_and_failures", test_law_branches_and_failures, devices=[_device]
    )
    add_function_test(TestContactComponents, "test_component_graph", test_component_graph, devices=[_device])
    add_function_test(
        TestContactComponents,
        "test_start_domain_and_boundary_branches",
        test_start_domain_and_boundary_branches,
        devices=[_device],
    )
    add_function_test(
        TestContactComponents, "test_tet_tangent_and_domain", test_tet_tangent_and_domain, devices=[_device]
    )

if __name__ == "__main__":
    unittest.main()
