# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Virtual work and lifetime of step-frozen material contact weights."""

import unittest

import numpy as np

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_contact_materials import contact_solver, query
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _values(state):
    history = state._monolithic_coupled
    return tuple(
        array.numpy().copy()
        for array in (state.particle_q, history.cable_theta, history.rigid_com, history.rigid_rotation)
    )


def _perturb(solver, state, values, row, step):
    x, theta, com, rotation = (value.copy() for value in values)
    for particle, first in enumerate(solver._actors.maps["particle_to_global"]):
        if first >= 0 and first <= row < first + 3:
            x[particle, row - first] += step
    for angle, index in enumerate(solver._actors.maps["theta_to_global"]):
        if row == index:
            theta[angle] += step
    for index, actor in enumerate(solver._rigid.actors):
        local = row - actor.dof_start
        if 0 <= local < 3:
            com[index, local] += step
        elif 3 <= local < 6:
            axis = np.eye(3)[local - 3]
            v, scalar = axis * np.sin(step / 2), np.cos(step / 2)
            old = rotation[index].copy()
            rotation[index, :3] = scalar * old[:3] + old[3] * v + np.cross(v, old[:3])
            rotation[index, 3] = scalar * old[3] - v @ old[:3]
    state.particle_q.assign(x)
    history = state._monolithic_coupled
    history.cable_theta.assign(theta)
    history.rigid_com.assign(com)
    history.rigid_rotation.assign(rotation)


def _check_virtual_work(test, solver, state):
    """Differentiate actual contact energy with all other producers excluded."""
    query(solver, state, begin=True)
    query(solver, state)
    contact = solver._contact
    values = _values(state)
    residual = contact.residual.numpy().copy()
    test.assertGreater(np.linalg.norm(residual), 1e-8)
    matrix = batch_matrix(contact._batch)
    scale = max(1.0, np.linalg.norm(matrix))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12 * scale)
    test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-12 * scale)
    rigid_rows = solver._rigid.rows_numpy.ravel()
    material_rows = np.setdiff1d(np.arange(len(residual)), rigid_rows)
    test.assertGreater(np.linalg.norm(matrix[np.ix_(rigid_rows, material_rows)]), 1e-8)

    forces = contact.pool.force.result.numpy().copy()
    count = solver.model.particle_count
    particle = forces[: 3 * count].reshape(-1, 3)
    body = forces[3 * count : 3 * count + 6]
    np.testing.assert_allclose(particle.sum(axis=0) + body[:3], 0, atol=1e-11)
    moment = np.cross(values[0], particle).sum(axis=0)
    moment += body[3:] + np.cross(values[2][0], body[:3])
    if len(values[1]):
        # For a straight anchor, a world spin has dtheta = dot(omega, tangent).
        tangent = values[0][1] - values[0][0]
        tangent /= np.linalg.norm(tangent)
        moment += tangent * forces[3 * count + 6 :].sum()
    np.testing.assert_allclose(moment, 0, atol=2e-11)

    h = 2e-8
    for row in range(len(residual)):
        energies = []
        for sign in (-1, 1):
            _perturb(solver, state, values, row, sign * h)
            query(solver, state)
            energies.append(contact.energy.numpy().sum())
        derivative = (energies[1] - energies[0]) / (2 * h)
        np.testing.assert_allclose(residual[row], derivative, rtol=2e-5, atol=2e-7, err_msg=f"row {row}")
    return residual, matrix


def test_cloth_orientation_virtual_work(test, device):
    """Cloth virtual work holds with coefficients frozen at either endpoint or inside the transition."""
    solver = contact_solver(
        body=True, device=device, alignment_policy="frozen", box_smoothing=25e-6, alignment_smoothing=0.1
    )
    state = solver.create_state(dt=0.001)
    for alignment in (0.0, 0.05, 0.1):
        with test.subTest(alignment=alignment):
            state._monolithic_coupled.rigid_com.zero_()
            state._monolithic_coupled.rigid_rotation.assign([[0, 0, 0, 1]])
            state.particle_q.assign(
                [[0, 0, -0.02], [0, 0.01, -0.02], [-0.01 * alignment, 0, -0.02 + 0.01 * np.sqrt(1 - alignment**2)]]
            )
            _check_virtual_work(test, solver, state)


def test_cable_orientation_virtual_work(test, device):
    """Source spin, endpoint translations and target rotation share the frozen potential."""
    solver = contact_solver(
        cable=True, body=True, device=device, alignment_policy="frozen", box_smoothing=25e-6, alignment_smoothing=0.1
    )
    state = solver.create_state(dt=0.001)
    x = state.particle_q.numpy().copy()
    # Tilt both end-cap normals into the angular transition. Compressing the
    # sampling geometry keeps every section near the same broad target face.
    x[:, 0] = np.arange(len(x)) * 0.1 * np.sqrt(1 - 0.05**2)
    x[:, 2] = -0.013 + np.arange(len(x)) * 0.005
    state.particle_q.assign(x)
    state._monolithic_coupled.cable_theta.fill_(0.05)
    _, matrix = _check_virtual_work(test, solver, state)
    rows = np.asarray(solver._actors.maps["theta_to_global"])
    test.assertGreater(np.linalg.norm(matrix[np.ix_(solver._rigid.rows_numpy.ravel(), rows)]), 1e-8)


def test_cable_surface_jacobian(test, device):
    """Differentiate material sample positions off the transport anchor, including end caps."""
    solver = contact_solver(
        cable=True, device=device, alignment_policy="frozen", box_smoothing=25e-6, alignment_smoothing=0.1
    )
    state = solver.create_state(dt=0.001)
    surface = solver._contact._batch.material_surface
    query(solver, state, begin=True)
    x = state.particle_q.numpy().copy()
    x[1] += (0.02, 0.01, -0.002)
    theta = np.array([0.13, -0.17])
    state.particle_q.assign(x)
    state._monolithic_coupled.cable_theta.assign(theta)
    surface.update(state)
    jacobian = surface.jacobian.numpy().copy()
    edges = surface.edges.numpy()
    h = 1e-5
    for edge in (0, 1):
        selected = np.flatnonzero(edges[:, 2] == edge)
        for coordinate in range(7):
            positions = []
            for sign in (-1, 1):
                candidate_x, candidate_theta = x.copy(), theta.copy()
                if coordinate < 6:
                    candidate_x[edge + coordinate // 3, coordinate % 3] += sign * h
                else:
                    candidate_theta[edge] += sign * h
                state.particle_q.assign(candidate_x)
                state._monolithic_coupled.cable_theta.assign(candidate_theta)
                surface.update(state)
                positions.append(surface.points.numpy()[selected])
            np.testing.assert_allclose(
                (positions[1] - positions[0]) / (2 * h), jacobian[selected, coordinate], rtol=2e-7, atol=3e-10
            )


def cloth_energy_ratio(solver, state, alignment, *, begin=False):
    contact = solver._contact
    bary = np.full((3, 3), 1 / 6)
    np.fill_diagonal(bary, 2 / 3)
    areas = contact._batch.all_samples.area.numpy()[:3]
    offsets = contact._batch.all_samples.gaps.numpy()[:3]
    x = np.array(
        [
            [0, 0, -0.02],
            [0, 0.01, -0.02],
            [-0.01 * alignment, 0, -0.02 + 0.01 * np.sqrt(1 - alignment**2)],
        ]
    )
    state.particle_q.assign(x)
    if begin:
        query(solver, state, begin=True)
    query(solver, state)
    # The wide BOX top is z=0; all other LSE planes are outside
    # the floating-point support. The independent penalty here
    # is in its exact quadratic branch on both cloth sides.
    gap = bary @ x[:, 2] - offsets
    one_side_energy = 50 * (areas @ (gap * gap))
    return contact.energy.numpy().sum() / one_side_energy


def test_cloth_weight_lifetime(test, device):
    """Cross both angular thresholds inside a step; only begin_step updates the coefficient."""
    for width in (0.0, 0.1):
        with test.subTest(alignment_smoothing=width):
            solver = contact_solver(
                body=True, device=device, alignment_policy="frozen", box_smoothing=25e-6, alignment_smoothing=width
            )
            state = solver.create_state(dt=0.001)
            contact = solver._contact
            keys = contact.actor_samples.keys.copy()
            coefficient = 1.5 if width else 1.0
            test.assertAlmostEqual(cloth_energy_ratio(solver, state, 0.05, begin=True), coefficient, delta=2e-12)
            count = int(contact.pool.records.count.numpy()[0])
            active = contact.pool.records.identities.numpy()[:count].copy()
            for alignment in (0.1, 0.2, -0.2, 0.0):
                test.assertAlmostEqual(cloth_energy_ratio(solver, state, alignment), coefficient, delta=2e-12)
                test.assertEqual(contact.actor_samples.keys, keys)
                test.assertEqual(int(contact.pool.records.count.numpy()[0]), count)
                np.testing.assert_array_equal(contact.pool.records.identities.numpy()[:count], active)
            test.assertAlmostEqual(cloth_energy_ratio(solver, state, 0.2, begin=True), 1.0, delta=2e-12)
            test.assertAlmostEqual(cloth_energy_ratio(solver, state, 0.0, begin=True), 2.0, delta=2e-12)
            # Reconstructing the original start recomputes the same weights,
            # rather than retaining the previous step's selection.
            test.assertAlmostEqual(cloth_energy_ratio(solver, state, 0.05, begin=True), coefficient, delta=2e-12)
            test.assertEqual(contact.actor_samples.keys, keys)


def test_material_same_input_precision(test, device):
    """Qualify broad-face evaluation separately from input and model approximation errors.

    Both precisions consume the same FP32-representable state, parameters and
    quadrature. The force/torque budget is 1e-4 of each unit group's peak load,
    with absolute floors of 1e-8 N and 1e-10 Nm. This is a component budget,
    not a claimed floating-point error bound or a nonlinear stopping rule.
    """
    for cable in (False, True):
        with test.subTest(cable=cable):
            solvers, states = [], []
            for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
                solver = contact_solver(
                    cable=cable,
                    body=True,
                    device=device,
                    precision=precision,
                    alignment_policy="frozen",
                    smoothing=float(np.float32(1e-5)),
                    box_smoothing=float(np.float32(25e-6)),
                    alignment_smoothing=float(np.float32(0.1)),
                )
                state = solver.create_state(dt=0.001)
                if cable:
                    x = np.zeros((3, 3))
                    x[:, 0] = np.arange(3) * 0.1 * np.sqrt(1 - 0.05**2)
                    x[:, 2] = -0.013 + np.arange(3) * 0.005
                    state._monolithic_coupled.cable_theta.fill_(float(np.float32(0.05)))
                else:
                    x = np.array([[0, 0, -0.02], [0, 0.01, -0.02], [-0.0005, 0, -0.02 + 0.01 * np.sqrt(1 - 0.05**2)]])
                state.particle_q.assign(x.astype(np.float32))
                # Quadrature values are generated on the host in double. Round
                # these inputs once for both paths, before any native queries.
                contact, batch = solver._contact, solver._contact._batch
                for owner, names in (
                    (contact, ("scales", "transforms", "margins")),
                    (batch.all_samples, ("area", "gaps", "points", "normals")),
                    (batch.support, ("com",)),
                    (batch.material_surface, ("coordinates", "normals")),
                ):
                    for name in names:
                        array = getattr(owner, name)
                        array.assign(array.numpy().astype(np.float32))
                solvers.append(solver)
                states.append(state)
            # Verify actual stored inputs, including the source frame anchor.
            for first, second in zip(_values(states[0]), _values(states[1]), strict=True):
                np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(
                states[0]._monolithic_coupled.cable_ref_director.numpy(),
                states[1]._monolithic_coupled.cable_ref_director.numpy(),
            )
            reference, _ = _check_virtual_work(test, solvers[0], states[0])
            # Restore the unperturbed state after the reference's FD sweep.
            states[0].particle_q.assign(states[1].particle_q.numpy())
            states[0]._monolithic_coupled.rigid_com.assign(states[1]._monolithic_coupled.rigid_com.numpy())
            states[0]._monolithic_coupled.rigid_rotation.assign(states[1]._monolithic_coupled.rigid_rotation.numpy())
            states[0]._monolithic_coupled.cable_theta.assign(states[1]._monolithic_coupled.cable_theta.numpy())
            query(solvers[0], states[0])
            query(solvers[1], states[1], begin=True)
            query(solvers[1], states[1])
            actual = solvers[1]._contact.residual.numpy()
            torque_rows = list(solvers[0]._rigid.rows_numpy.ravel()[3:])
            torque_rows.extend(row for row in solvers[0]._actors.maps["theta_to_global"] if row >= 0)
            force_rows = np.setdiff1d(np.arange(len(reference)), torque_rows)
            for rows, floor, unit in ((force_rows, 1e-8, "N"), (torque_rows, 1e-10, "Nm")):
                budget = max(floor, 1e-4 * np.max(np.abs(reference[rows])))
                error = np.max(np.abs(actual[rows] - reference[rows]))
                test.assertLessEqual(error, budget, f"cable={cable}, same-input {unit} error={error}, budget={budget}")
            pools = [solver._contact.pool.records for solver in solvers]
            counts = [int(pool.count.numpy()[0]) for pool in pools]
            test.assertEqual(counts[0], counts[1])
            np.testing.assert_array_equal(
                pools[0].identities.numpy()[: counts[0]], pools[1].identities.numpy()[: counts[1]]
            )


class TestContactSmoothingMaterial(unittest.TestCase):
    pass


for _test in (
    test_cloth_orientation_virtual_work,
    test_cable_orientation_virtual_work,
    test_cable_surface_jacobian,
    test_cloth_weight_lifetime,
    test_material_same_input_precision,
):
    add_function_test(TestContactSmoothingMaterial, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
