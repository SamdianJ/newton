# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent local response checks for weighted shell point contacts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.cloth_contact_point_cloud import _evaluate_local, _PointCloudContact
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_material_state import snapshot
from scripts.monolithic_reference.cloth_contact_method import evaluate_static


def _scene(
    *,
    enabled=True,
    precision=SolverMonolithic.Precision.FP64,
    capacity=64,
    fixed=False,
    self_owner=False,
    self_enabled=True,
    friction=0,
    device="cpu",
    graph=False,
    contact_options=None,
):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    shells = []
    if self_owner:
        # The material's natural reference is flat; folding belongs to the state.
        for x, z in ((0, 0), (0.015, 0), (0.03, 0), (0.035, 0), (0.05, 0), (0.065, 0)):
            for y in (0.0, 0.02):
                builder.add_particle(wp.vec3(x, y, z), wp.vec3(0), 0.0)
        for i in range(5):
            for tri in ((2 * i, 2 * i + 2, 2 * i + 1), (2 * i + 1, 2 * i + 2, 2 * i + 3)):
                builder.add_triangle(*tri, tri_ke=2, tri_ka=3, tri_kd=0)
                a, b, c = np.asarray(builder.particle_q)[list(tri)]
                area = np.linalg.norm(np.cross(b - a, c - a)) / 2
                for p in tri:
                    builder.particle_mass[p] += area / 3
        shells = [
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(range(10)), bending=(0.01, 0.01), thickness=0.01, surface_density=1
            )
        ]
        capacity = max(capacity, 512)
    else:
        for sheet, z in enumerate((0.0, 0.005)):
            first, face = builder.particle_count, builder.tri_count
            for x, y in ((0, 0), (0.03, 0), (0, 0.03)):
                builder.add_particle(wp.vec3(x, y, z), wp.vec3(0), 0 if fixed and sheet == 0 else 0.00015)
            builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0)
            shells.append(
                SolverMonolithic.ClothInput(
                    triangle_ids=(face,),
                    bending=(0.01, 0.01),
                    thickness=0.01,
                    surface_density=1,
                )
            )
    model = builder.finalize(device=device)
    config = (
        {"radius": 0.02, "normal_stiffness": 1e3, "candidate_capacity": capacity, "self": self_enabled}
        if enabled
        else None
    )
    if config is not None:
        config.update(contact_options or {})
    solver = SolverMonolithic(
        model,
        cloth_inputs=shells,
        cloth_contact=config,
        friction_velocity_scale=0.1,
        friction_coefficient=friction,
        precision=precision,
        newton_max_iterations=40,
        pcg_mode=SolverMonolithic.PCGMode.PRODUCTION if graph else SolverMonolithic.PCGMode.DIAGNOSTIC,
        use_cuda_graph=graph,
    )
    return model, solver


def _folded_strip():
    return np.array(
        [
            (x, y, z)
            for x, z in ((0, 0), (0.015, 0), (0.03, 0), (0.03, 0.005), (0.015, 0.005), (0, 0.005))
            for y in (0.0, 0.02)
        ]
    )


def _self_pair_scene(reference_gap, *, thickness, smoothing, padding, precision, rotation=None, offset=(0, 0, 0)):
    rotation = np.eye(3) if rotation is None else rotation
    reference = np.array(
        [
            (0, 0, 0),
            (0, 3 / 64, 0),
            (0, 0, 3 / 64),
            (reference_gap, 1 / 128, 1 / 128),
            (reference_gap, 7 / 128, 1 / 128),
            (reference_gap, 1 / 128, 7 / 128),
            (3 / 64, 1 / 16, 1 / 64),
            (1 / 16, 1 / 32, 3 / 64),
            (5 / 64, 5 / 64, 5 / 64),
        ]
    )
    current = reference.copy()
    current[3:6, 0] = 1 / 128
    # A bridge keeps the cloth connected while preserving the selected reference gap.
    faces = ((0, 1, 2), (3, 5, 4), (2, 1, 6), (2, 6, 7), (7, 6, 8), (7, 8, 4), (4, 8, 3))
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for point in reference @ rotation.T + offset:
        builder.add_particle(wp.vec3(point), wp.vec3(0), 0.0)
    for tri in faces:
        builder.add_triangle(*tri, tri_ke=2, tri_ka=3, tri_kd=0)
        a, b, c = np.asarray(builder.particle_q)[list(tri)]
        for p in tri:
            builder.particle_mass[p] += np.linalg.norm(np.cross(b - a, c - a)) / 6
    solver = SolverMonolithic(
        builder.finalize(device="cpu"),
        cloth_inputs=[
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(range(len(faces))), bending=(0.01, 0.01), thickness=thickness, surface_density=1
            )
        ],
        cloth_contact={"inter": False, "radius": 1 / 64, "candidate_padding": padding},
        friction_velocity_scale=0.1,
        friction_coefficient=0.0,
        normal_smoothing=smoothing,
        precision=precision,
        use_cuda_graph=False,
    )
    state = solver.create_state(dt=0.001)
    state.particle_q.assign(current @ rotation.T + offset)
    return solver, state


def _candidate_oracle(solver, start, current):
    """Enumerate ordered material samples independently of the native spatial hash."""
    contact = solver._cloth_contact
    objects = solver._materials.cloth
    nodes = [(owner, int(p)) for owner, obj in enumerate(objects) for p in obj["particle_ids"]]
    expected, face = set(), 0
    for owner, obj in enumerate(objects):
        reference = dict(zip(obj["particle_ids"], obj["positions"], strict=True))
        for tri in obj["triangles"]:
            for slot in range(3):
                bary = np.array([2 / 3 if i == slot else 1 / 6 for i in range(3)])
                for node, (other, particle) in enumerate(nodes):
                    if owner == other:
                        if not contact.self_enabled:
                            continue
                        q = bary @ np.array([reference[p] for p in tri])
                        radius = contact.self_exclusion_ratio * contact.length + obj["thickness"] + contact.smoothing
                        if np.sum((q - reference[particle]) ** 2) < radius**2:
                            continue
                    elif not contact.inter_enabled:
                        continue
                    cutoff = (
                        contact.length
                        + (obj["thickness"] + objects[other]["thickness"]) / 2
                        + contact.smoothing
                        + contact.padding
                    )
                    if any(np.linalg.norm(bary @ x[tri] - x[particle]) <= cutoff for x in (start, current)):
                        expected.add((3 * face + slot, node))
            face += 1
    return expected


def _pairs(contact):
    count = int(contact.pool.candidates.count.numpy()[0])
    pairs = contact.pairs.numpy()[:count]
    assert len(set(map(tuple, pairs))) == count
    return set(map(tuple, pairs))


def _penalty(gap, k, smoothing):
    u = -gap
    if u >= smoothing:
        p, derivative = u, 1.0
    elif u <= -smoothing:
        p, derivative = 0.0, 0.0
    else:
        t = (u + smoothing) / (2 * smoothing)
        p = smoothing * (2 * t**3 - t**4)
        derivative = 3 * t**2 - 2 * t**3
    return 0.5 * k * p**2, k * p * derivative


def _curvature(gap, k, smoothing):
    u = -gap
    if u >= smoothing:
        return k
    if u <= -smoothing:
        return 0.0
    t = (u + smoothing) / (2 * smoothing)
    p, dp, ddp = smoothing * (2 * t**3 - t**4), 3 * t**2 - 2 * t**3, 3 * t * (1 - t) / smoothing
    return k * (dp**2 + p * ddp)


def _full_jacobian(y, slip):
    length, radius, k, smoothing, mu, delta = np.linalg.norm(y), 0.023, 140.0, 0.001, 0.3, 0.003
    n = y / length
    projection = np.eye(3) - np.outer(n, n)
    normal_derivative = projection / length
    gap = length - radius
    _, load = _penalty(gap, k, smoothing)
    ie_gap = gap - n @ slip
    _, ie_load = _penalty(ie_gap, k, smoothing)
    v = projection @ slip
    rho = np.linalg.norm(v)
    dv = projection - (n @ slip) * normal_derivative - np.outer(n, normal_derivative @ slip)
    factor = 2 / delta - rho / delta**2 if rho < delta else 1 / rho
    dfactor = -1 / delta**2 if rho < delta else -1 / rho**2
    dbeta = mu * factor * _curvature(ie_gap, k, smoothing) * (normal_derivative @ slip)
    if rho:
        dbeta += mu * ie_load * dfactor * (dv.T @ v) / rho
    return (
        _curvature(gap, k, smoothing) * np.outer(n, n)
        - load * normal_derivative
        + mu * ie_load * factor * dv
        + np.outer(v, dbeta)
    )


def _oracle(
    y,
    slip,
    *,
    thickness=(0.002, 0.004),
    areas=(0.0004, 0.0007),
    length=0.02,
    stiffness=2e5,
    smoothing=0.001,
    mu=0.3,
    falloff=0.003,
):
    n = y / np.linalg.norm(y)
    gap = np.linalg.norm(y) - length - sum(thickness) / 2
    k = stiffness * np.prod(areas) / length**2
    energy, load = _penalty(gap, k, smoothing)
    _, friction_load = _penalty(gap - n @ slip, k, smoothing)
    v = slip - (n @ slip) * n
    rho = np.linalg.norm(v)
    if rho < falloff:
        psi = rho**2 / falloff - rho**3 / (3 * falloff**2)
        factor = 2 / falloff - rho / falloff**2
    else:
        psi = rho - falloff / 3
        factor = 1 / rho
    return np.array([energy, mu * friction_load * psi]), -load * n + mu * friction_load * factor * v


def _native(
    y,
    slip,
    *,
    dtype=wp.float64,
    thickness=(0.002, 0.004),
    areas=(0.0004, 0.0007),
    length=0.02,
    stiffness=2e5,
    smoothing=0.001,
    mu=0.3,
    falloff=0.003,
):
    vector, matrix = (wp.vec3d, wp.mat33d) if dtype == wp.float64 else (wp.vec3, wp.mat33)

    def array(values, kind=dtype):
        return wp.array(values, dtype=kind, device="cpu")

    energy = wp.zeros((1, 2), dtype=dtype, device="cpu")
    residual, tangent = wp.zeros(1, dtype=vector, device="cpu"), wp.zeros(1, dtype=matrix, device="cpu")
    loads, flags = wp.zeros((1, 2), dtype=dtype, device="cpu"), wp.zeros(1, dtype=int, device="cpu")
    wp.launch(
        _evaluate_local,
        1,
        [
            array([y], vector),
            array([slip], vector),
            array([thickness]),
            array([areas]),
            dtype(length),
            dtype(stiffness),
            dtype(smoothing),
            dtype(mu),
            dtype(falloff),
            energy,
            residual,
            tangent,
            loads,
            flags,
        ],
        device="cpu",
    )
    return (
        energy.numpy()[0].copy(),
        residual.numpy()[0].copy(),
        tangent.numpy()[0].copy(),
        loads.numpy()[0].copy(),
        int(flags.numpy()[0]),
    )


class TestPointCloudResponse(unittest.TestCase):
    def test_full_cie_jacobian_and_four_node_blocks(self):
        """Differentiate the nonconservative C-IE residual and pull back all four-node blocks."""
        y, slip = np.array([0.002, -0.001, 0.018]), np.array([0.0006, 0.0004, -0.0002])
        direction = np.array([0.3, -0.5, 0.8])
        full = _full_jacobian(y, slip)
        for h in (2e-7, 2e-8):
            plus = _native(y + h * direction, slip + h * direction)[1]
            minus = _native(y - h * direction, slip - h * direction)[1]
            np.testing.assert_allclose((plus - minus) / (2 * h), full @ direction, rtol=2e-7, atol=2e-6)
        _, r, psd, _, _ = _native(y, slip)
        jacobian = np.concatenate([b * np.eye(3) for b in (2 / 3, 1 / 6, 1 / 6, -1)], axis=1)
        block = jacobian.T @ psd @ jacobian
        self.assertGreater(np.linalg.norm(block[:9, 9:]), 0)
        np.testing.assert_allclose(block[:9, 9:], block[9:, :9].T, atol=1e-12)
        force = (-jacobian.T @ r).reshape(4, 3)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-15)
        self.assertGreater(np.linalg.norm(np.cross(y, -r)), 0)

    def test_native_cie_and_psd(self):
        """Match independent C-IE values and retain a nonnegative symmetric approximation."""
        y = np.array([0.002, -0.001, 0.018])
        for dtype, tolerance in ((wp.float64, 2e-12), (wp.float32, 3e-5)):
            for slip in (np.zeros(3), np.array([0.0005, 0.0003, -0.0002]), np.array([0.006, -0.003, 0.001])):
                with self.subTest(dtype=dtype, slip=slip):
                    en, r, h, _, status = _native(y, slip, dtype=dtype)
                    expected_e, expected_r = _oracle(y, slip)
                    self.assertEqual(status, 0)
                    np.testing.assert_allclose(en, expected_e, rtol=tolerance, atol=tolerance * 1e-4)
                    np.testing.assert_allclose(r, expected_r, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(h, h.T, rtol=tolerance, atol=tolerance)
                    self.assertGreaterEqual(np.linalg.eigvalsh(h).min(), -tolerance * np.linalg.norm(h))

    def test_energy_and_full_normal_derivative(self):
        """Differentiate the normal energy and the complete geometric residual independently."""
        y = np.array([0.004, -0.002, 0.017])
        direction = np.array([0.2, -0.3, 0.7])
        en, r, psd, loads, _ = _native(y, np.zeros(3), mu=0)
        n = y / np.linalg.norm(y)
        exact = psd - loads[0] / np.linalg.norm(y) * (np.eye(3) - np.outer(n, n))
        for step in (2e-6, 2e-7, 2e-8):
            plus = _native(y + step * direction, np.zeros(3), mu=0)
            minus = _native(y - step * direction, np.zeros(3), mu=0)
            self.assertAlmostEqual((plus[0][0] - minus[0][0]) / (2 * step), r @ direction, delta=1e-7)
            np.testing.assert_allclose((plus[1] - minus[1]) / (2 * step), exact @ direction, rtol=1e-7, atol=1e-7)
        self.assertGreater(np.linalg.norm(exact - psd), 1.0)
        self.assertGreater(en[0], 0)

    def test_thickness_weights_exchange_and_rotation(self):
        """Use each physical thickness and both area weights while preserving endpoint symmetry."""
        y, slip = np.array([0.001, -0.002, 0.019]), np.array([0.0003, 0.0002, 0.0])
        baseline = _native(y, slip)
        for thickness in ((0.004, 0.004), (0.002, 0.006)):
            result = _native(y, slip, thickness=thickness)
            self.assertGreater(result[3][0], baseline[3][0])
        for areas in ((0.0008, 0.0007), (0.0004, 0.0014)):
            result = _native(y, slip, areas=areas)
            np.testing.assert_allclose(result[0], 2 * baseline[0], rtol=1e-13)
            np.testing.assert_allclose(result[1], 2 * baseline[1], rtol=1e-13)
        reverse = _native(-y, -slip, thickness=(0.004, 0.002), areas=(0.0007, 0.0004))
        np.testing.assert_allclose(reverse[0], baseline[0], rtol=1e-13)
        np.testing.assert_allclose(reverse[1], -baseline[1], rtol=1e-13)
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        rotated = _native(rotation @ y, rotation @ slip)
        np.testing.assert_allclose(rotated[1], rotation @ baseline[1], rtol=1e-13)
        np.testing.assert_allclose(rotated[2], rotation @ baseline[2] @ rotation.T, rtol=1e-13)

    def test_zero_distance_is_invalid(self):
        """Reject a loaded coincident pair instead of inventing its normal."""
        self.assertEqual(_native(np.zeros(3), np.zeros(3))[-1], 2)


class TestPointCloudSolver(unittest.TestCase):
    def test_flat_reference_excludes_local_two_hop_loads(self):
        """Suppress reference-local self preload on a flat two-hop neighborhood."""
        _, solver = _scene(self_owner=True)
        state = solver.create_state(dt=0.001)
        contact = solver._cloth_contact
        contact.begin_step(state)
        contact.evaluate(wp.zeros_like(state.particle_q), 0.001)
        self.assertEqual(float(contact.summary.numpy()[5]), 0.0)
        np.testing.assert_array_equal(contact.pool.force.result.numpy(), 0.0)

    def test_diagnostic_summary_ignores_candidate_slot_order(self):
        """Publish the same diagnostic counts and loads for any candidate slot order."""
        for self_owner in (False, True):
            summaries = []
            for reverse in (False, True):
                _, solver = _scene(self_owner=self_owner)
                state = solver.create_state(dt=0.001)
                contact = solver._cloth_contact
                if self_owner:
                    state.particle_q.assign(wp.array(_folded_strip(), dtype=contact.vector, device="cpu"))
                contact.begin_step(state)
                if reverse:
                    # GPU queries reserve slots atomically, so slot order is not reproducible.
                    count = int(contact.pool.candidates.count.numpy()[0])
                    pairs = contact.pairs.numpy()
                    pairs[:count] = pairs[:count][::-1].copy()
                    contact.pairs.assign(pairs)
                contact.evaluate(wp.zeros_like(state.particle_q), 0.001)
                summaries.append(contact.summary.numpy())
            with self.subTest(self_owner=self_owner):
                self.assertGreater(summaries[0][5 if self_owner else 2], 0)
                np.testing.assert_array_equal(summaries[1], summaries[0])

    def test_invalid_configuration_and_grid_domain(self):
        """Reject invalid native lengths and unrepresentable hash coordinates."""
        _, solver = _scene()
        for config in (
            {"radius": 0},
            {"normal_stiffness": float("nan")},
            {"radius": True},
            {"self": 1},
            {"candidate_capacity": -1},
            {"self_exclusion_ratio": True},
            {"self_exclusion_ratio": 1.0},
            {"self_exclusion_ratio": float("nan")},
            {"self_exclusion_ratio": float("inf")},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                _PointCloudContact(solver, config, mu=0, smoothing=1e-4, velocity_scale=0.1, use_cuda_graph=False)
        _PointCloudContact(
            solver,
            {"self": False, "self_exclusion_ratio": 1.0},
            mu=0,
            smoothing=1e-4,
            velocity_scale=0.1,
            use_cuda_graph=False,
        )
        _, native32 = _scene(precision=SolverMonolithic.Precision.FP32)
        with self.assertRaises(ValueError):
            _PointCloudContact(
                native32,
                {"self_exclusion_ratio": 1 + 1e-9},
                mu=0,
                smoothing=1e-4,
                velocity_scale=0.1,
                use_cuda_graph=False,
            )
        contact = solver._cloth_contact
        contact.start.fill_(wp.vec3d(1e20))
        contact.refresh(contact.start)
        contact.evaluate(wp.zeros_like(contact.start), 0.001)
        self.assertEqual(int(contact.flags.numpy()[0]), 2)

    def test_start_current_union_and_static_adapter(self):
        """Keep start-near friction candidates and evaluate static forces without changing the model."""
        model, solver = _scene(fixed=True)
        state = solver.create_state(dt=0.001)
        contact = solver._cloth_contact
        contact.begin_step(state)
        original = model.particle_q.numpy().copy()
        far = state.particle_q.numpy().copy()
        far[3:, 2] += 0.1
        contact.refresh(wp.array(far, dtype=contact.vector, device="cpu"))
        count = int(contact.pool.candidates.count.numpy()[0])
        pairs = contact.pairs.numpy()[:count]
        self.assertEqual(count, 18)
        self.assertEqual(len(set(map(tuple, pairs))), count)
        inputs = [
            SolverMonolithic.ClothInput(triangle_ids=(i,), bending=(0.01, 0.01), thickness=0.01, surface_density=1)
            for i in range(2)
        ]
        options = {"radius": 0.02, "normal_stiffness": 1000, "candidate_capacity": 64}
        result = evaluate_static(
            model, inputs, options, original, precision="fp64", device="cpu", solver_kwargs={"normal_smoothing": 1e-4}
        )
        self.assertGreater(result["inter"]["compressive_load_N"], 0)
        np.testing.assert_array_equal(model.particle_q.numpy(), original)

    def test_friction_classification_without_shapes(self):
        """Classify new C-IE friction as nonconservative even without shape contacts."""
        _, solver = _scene(friction=0.3)
        self.assertFalse(solver._conservative_energy())

    def test_pool_matches_independent_all_pairs(self):
        """Match both ordered quadratures and their global cross blocks to an exhaustive oracle."""
        model, solver = _scene()
        state = solver.create_state(dt=0.001)
        contact = solver._cloth_contact
        contact.begin_step(state)
        contact.evaluate(wp.zeros_like(state.particle_q), 0.001)
        positions = state.particle_q.numpy()
        faces = model.tri_indices.numpy()
        expected_force, expected_matrix, expected_energy = np.zeros((6, 3)), np.zeros((18, 18)), 0.0
        for source in range(2):
            tri, other = faces[source], faces[1 - source]
            area_a = (
                np.linalg.norm(np.cross(positions[tri[1]] - positions[tri[0]], positions[tri[2]] - positions[tri[0]]))
                / 2
            )
            area_b = (
                np.linalg.norm(
                    np.cross(positions[other[1]] - positions[other[0]], positions[other[2]] - positions[other[0]])
                )
                / 2
            )
            for slot in range(3):
                bary = np.full(3, 1 / 6)
                bary[slot] = 2 / 3
                for p in other:
                    y = bary @ positions[tri] - positions[p]
                    radius = np.linalg.norm(y)
                    k = 1000 * area_a * area_b / (9 * 0.02**2)
                    energy, load = _penalty(radius - 0.03, k, 1e-4)
                    n = y / radius
                    expected_energy += energy
                    jacobian = np.zeros((3, 18))
                    for particle, coefficient in zip((*tri, p), (*bary, -1), strict=True):
                        jacobian[:, 3 * particle : 3 * particle + 3] += coefficient * np.eye(3)
                    expected_force += (jacobian.T @ (load * n)).reshape(6, 3)
                    normal_factor = jacobian.T @ n
                    expected_matrix += _curvature(radius - 0.03, k, 1e-4) * np.outer(normal_factor, normal_factor)
        np.testing.assert_allclose(
            contact.pool.force.result.numpy().reshape(6, 3), expected_force, rtol=2e-13, atol=1e-17
        )
        np.testing.assert_allclose(
            contact.pool.residual.result.numpy(), -expected_force.reshape(-1), rtol=2e-13, atol=1e-17
        )
        self.assertAlmostEqual(float(contact.energy.numpy()[0]), expected_energy, delta=1e-16)
        records = contact.pool.records
        count = int(records.triplet_count.numpy()[0])
        matrix = np.zeros((18, 18))
        np.add.at(
            matrix,
            (records.triplet_rows.numpy()[:count], records.triplet_columns.numpy()[:count]),
            records.triplet_values.numpy()[:count],
        )
        np.testing.assert_allclose(matrix, expected_matrix, rtol=2e-12, atol=1e-14)
        self.assertGreater(np.linalg.norm(matrix[:9, 9:]), 0)

    def test_self_filter_and_native_precision(self):
        """Keep folded nonlocal contact using the flat material reference in both precisions."""
        for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
            for enabled in (True, False):
                _, solver = _scene(self_owner=True, self_enabled=enabled, precision=precision)
                state = solver.create_state(dt=0.001)
                state.particle_q.assign(_folded_strip())
                solver.reset_state(state, dt=0.001)
                solver.step(state, state, solver.create_control(), dt=0.001)
                self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
                diagnostics = solver.get_cloth_contact_diagnostics(state)
                self.assertEqual(diagnostics["inter"]["active"], 0)
                self.assertEqual(diagnostics["self"]["active"] > 0, enabled)

    def test_reference_filter_boundary_and_threshold_mapping(self):
        """Use a strict squared threshold with additive thickness and smoothing, independent of padding."""
        for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
            for thickness, smoothing in ((1 / 128, 1 / 256), (1 / 64, 1 / 128)):
                radius = 1.5 / 64 + thickness + smoothing
                for padding in (0.0, 0.125):
                    for delta in (-1 / 2**20, 0.0, 1 / 2**20):
                        with self.subTest(precision=precision, thickness=thickness, padding=padding, delta=delta):
                            solver, state = _self_pair_scene(
                                radius + delta,
                                thickness=thickness,
                                smoothing=smoothing,
                                padding=padding,
                                precision=precision,
                            )
                            contact = solver._cloth_contact
                            contact.begin_step(state)
                            self.assertEqual((0, 3) in _pairs(contact), delta >= 0)

    def test_reference_candidates_match_oracle_and_start_current_union(self):
        """Keep both ordered nonlocal passes through start/current refreshes without changing reference."""
        for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
            model, solver = _scene(self_owner=True, precision=precision, contact_options={"candidate_padding": 0.0})
            state = solver.create_state(dt=0.001)
            contact = solver._cloth_contact
            reference = model.particle_q.numpy().copy()
            state.particle_q.assign(_folded_strip())
            contact.begin_step(state)
            start = state.particle_q.numpy().copy()
            current = start.copy()
            current[:2, 2] -= 0.1
            current[2:4, 2] += 0.015
            contact.refresh(wp.array(current, dtype=contact.vector, device="cpu"))
            expected = _candidate_oracle(solver, start, current)
            self.assertEqual(_pairs(contact), expected)
            start_only = _candidate_oracle(solver, start, start)
            current_only = _candidate_oracle(solver, current, current)
            self.assertTrue(start_only - current_only)
            self.assertTrue(current_only - start_only)
            self.assertTrue(any(sample < 6 and node >= 8 for sample, node in expected))
            self.assertTrue(any(sample >= 24 and node < 4 for sample, node in expected))
            # Changing Model storage or the next step's anchor must not replace material reference.
            model.particle_q.assign(current)
            state.particle_q.assign(current)
            contact.begin_step(state)
            self.assertEqual(_pairs(contact), current_only)
            np.testing.assert_array_equal(contact.reference.numpy(), reference)

    def test_reference_filter_rigid_transform_and_near_reference_limit(self):
        """Preserve ordered forces under a rigid transform and permanently exclude reference-near sheets."""
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        pairs, forces = [], []
        for transform in (np.eye(3), rotation):
            solver, state = _self_pair_scene(
                0.125,
                thickness=1 / 128,
                smoothing=1 / 256,
                padding=0.125,
                precision=SolverMonolithic.Precision.FP64,
                rotation=transform,
                offset=(0.25, -0.125, 0.5),
            )
            contact = solver._cloth_contact
            contact.begin_step(state)
            contact.evaluate(wp.zeros_like(state.particle_q), 0.001)
            pairs.append(_pairs(contact))
            forces.append(contact.pool.force.result.numpy().reshape(-1, 3))
            self.assertEqual(pairs[-1], _candidate_oracle(solver, state.particle_q.numpy(), state.particle_q.numpy()))
            self.assertTrue(any(sample < 3 and node >= 3 for sample, node in pairs[-1]))
            self.assertTrue(any(sample >= 3 and node < 3 for sample, node in pairs[-1]))
            self.assertGreater(np.linalg.norm(forces[-1]), 0)
            np.testing.assert_allclose(forces[-1].sum(axis=0), 0, atol=1e-15)
        self.assertEqual(pairs[0], pairs[1])
        np.testing.assert_allclose(forces[1], forces[0] @ rotation.T, atol=1e-15)
        solver, state = _self_pair_scene(
            0.0, thickness=1 / 128, smoothing=1 / 256, padding=0.125, precision=SolverMonolithic.Precision.FP64
        )
        contact = solver._cloth_contact
        contact.begin_step(state)
        self.assertNotIn((0, 3), _pairs(contact))
        current = state.particle_q.numpy()
        far = current.copy()
        far[3:6, 0] += 0.25
        state.particle_q.assign(far)
        contact.begin_step(state)
        state.particle_q.assign(current)
        contact.refresh(state.particle_q)
        self.assertNotIn((0, 3), _pairs(contact))

    def test_reference_filter_checkpoint_identity(self):
        """Reject changed reference-filter configuration before changing checkpoint state."""
        _, solver = _scene(self_owner=True)
        state = solver.create_state(dt=0.001)
        self.assertEqual(solver._cloth_contact.config["self_exclusion_ratio"], 1.5)
        self.assertEqual(solver._cloth_contact.config["self_exclusion"], "material-reference-distance")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            solver.save_checkpoint(state, path)
            for ratio in (1.5, 1.6):
                _, other = _scene(self_owner=True, contact_options={"self_exclusion_ratio": ratio})
                target = other.create_state(dt=0.001)
                before = snapshot(target)
                if ratio == 1.5:
                    other.load_checkpoint(target, path)
                else:
                    with self.assertRaises(ValueError):
                        other.load_checkpoint(target, path)
                for name, value in before.items():
                    np.testing.assert_array_equal(snapshot(target)[name], value, err_msg=name)

    def test_capacity_rollback_and_checkpoint(self):
        """Roll back every published array on overflow and continue from a fresh checkpoint owner."""
        _, solver = _scene(capacity=1)
        state = solver.create_state(dt=0.001)
        before = snapshot(state)
        solver.step(state, state, solver.create_control(), dt=0.001)
        self.assertEqual(solver.last_stats.status, solver.Status.CONTACT_OVERFLOW)
        self.assertTrue(solver.last_stats.rolled_back)
        for name, value in before.items():
            np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)
        _, solver = _scene(fixed=True)
        state = solver.create_state(dt=0.001)
        solver.step(state, state, solver.create_control(), dt=0.001)
        self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        self.assertGreater(np.linalg.norm(solver.get_contact_forces(state).particle_force[:3]), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloth.npz"
            solver.save_checkpoint(state, path)
            _, restored_solver = _scene(fixed=True)
            restored = restored_solver.create_state(dt=0.001)
            restored_solver.load_checkpoint(restored, path)
        for owner, current in ((solver, state), (restored_solver, restored)):
            owner.step(current, current, owner.create_control(), dt=0.001)
            self.assertFalse(owner.last_stats.rolled_back, owner.last_stats)
        for name, value in snapshot(state).items():
            np.testing.assert_array_equal(snapshot(restored)[name], value, err_msg=name)

    def test_compressed_free_shells_and_off_control(self):
        """Exchange equal opposite forces through the public step and preserve the disabled baseline."""
        for enabled in (False, True):
            model, solver = _scene(enabled=enabled)
            state = solver.create_state(dt=0.001)
            solver.step(state, state, solver.create_control(), dt=0.001)
            self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
            force = solver.get_contact_forces(state).particle_force
            if enabled:
                self.assertGreater(np.linalg.norm(force), 0)
                np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-14)
                diagnostics = solver.get_cloth_contact_diagnostics(state)
                self.assertGreater(diagnostics["inter"]["compressive_load_N"], 0)
                self.assertEqual(diagnostics["self"]["active"], 0)
            else:
                np.testing.assert_array_equal(force, np.zeros((model.particle_count, 3)))


if __name__ == "__main__":
    unittest.main()
