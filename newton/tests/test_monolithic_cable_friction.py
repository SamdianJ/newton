# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable friction through the production contact pool, with an independent planar oracle."""

import unittest

import numpy as np
import warp as wp

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_cable_contact import L, R, cable_contact_solver, hairpin


def sliding_fixture(mu, device="cpu", gap=0.0, **options):
    # The lower strand overhangs both ends. Each upper quadrature sample has one planar contact patch.
    x = L * np.arange(8)
    y = L * np.arange(-5, 13)
    upper = np.column_stack([x, 0 * x, 0 * x + 2 * R - 0.0004 + gap])
    lower = np.column_stack([y, 0 * y, 0 * y])
    ids = range(len(x))
    fixed = range(len(x), len(x) + len(y))
    solver, state = cable_contact_solver(
        np.vstack([upper, lower]),
        [ids, fixed],
        fixed=fixed,
        config={
            "self": False,
            "inter": True,
            "friction_coefficient": mu,
            "friction_velocity_scale": 0.01,
        },
        device=device,
        **options,
    )
    return solver, state, ids


class TestMonolithicCableFriction(unittest.TestCase):
    def test_lagged_planar_sliding(self):
        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            solver, state, ids = sliding_fixture(
                0.35, device, use_cuda_graph=device.startswith("cuda"), pcg_mode=SolverMonolithic.PCGMode.PRODUCTION
            )
            contact = solver._cable_contact
            contact.begin_step(state)
            dx = np.zeros_like(state.particle_q.numpy())
            steps = wp.zeros_like(state.particle_q)
            contact.evaluate(steps, 0.001)
            load = contact.summary.numpy()[2]
            self.assertGreater(load, 0.01)
            for dt, slip in [(0.001, 0.0), (0.001, 1e-6), (0.002, -1e-6), (0.001, 0.0001)]:
                dx[:] = [0.0003, 0.0007, -0.0002]  # A common translation produces no slip.
                dx[ids, 0] += slip
                steps.assign(dx)
                contact.evaluate(steps, dt)
                force = contact.pool.force.result.numpy().reshape(-1, 3)
                ratio = min(abs(slip) / (dt * 0.01), 1.0)
                expected = -np.sign(slip) * 0.35 * load * ratio * (2 - ratio)
                # Normal contact sums to zero along this overhanging planar track.
                self.assertAlmostEqual(force[ids, 0].sum(), expected, delta=1e-8)
                np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-11)
                r = abs(slip)
                e = dt * 0.01
                psi = r * r / e - r**3 / (3 * e**2) if r < e else r - e / 3
                self.assertAlmostEqual(contact.energy.numpy()[1], 0.35 * load * psi, delta=1e-12)

    def test_unloaded_pair_has_no_shear(self):
        solver, state, ids = sliding_fixture(0.7, gap=0.003)
        contact = solver._cable_contact
        contact.begin_step(state)
        dx = np.zeros_like(state.particle_q.numpy())
        dx[ids, 0] = 0.0001
        steps = wp.zeros_like(state.particle_q)
        steps.assign(dx)
        contact.evaluate(steps, 0.001)
        self.assertEqual(contact.energy.numpy()[1], 0.0)
        np.testing.assert_array_equal(contact.pool.force.result.numpy(), 0.0)

    def test_self_contact_and_tangent(self):
        points, chains = hairpin()
        solver, state = cable_contact_solver(
            points, chains, config={"self": True, "inter": False, "friction_coefficient": 0.35}
        )
        contact = solver._cable_contact
        contact.begin_step(state)
        steps = wp.zeros_like(state.particle_q)
        dx = np.zeros_like(state.particle_q.numpy())
        # Slide the back leg within the smooth near-stick region.
        dx[len(points) // 2 + 1 :, 0] = 2e-6
        steps.assign(dx)
        contact.evaluate(steps, 0.001)
        self.assertGreater(contact.energy.numpy()[1], 0.0)
        candidates = contact.pool.candidates
        count = int(candidates.count.numpy()[0])
        k = candidates.support_counts.numpy()[:count]
        factors = candidates.factors.numpy()[:count, 1:]
        weights = candidates.weights.numpy()[:count, 1:]
        columns = candidates.columns.numpy()[:count]
        self.assertTrue((weights >= 0).all())
        direction = np.random.default_rng(28).normal(size=dx.shape)
        dofs = contact.rows.numpy()
        rowdir = np.zeros(solver._layout.scalar_dof_count)
        for d, row in zip(direction, dofs, strict=True):
            if row >= 0:
                rowdir[row : row + 3] = d
        curvature = sum(
            np.dot(w, (f[:, :n] @ rowdir[col[:n]]) ** 2)
            for n, f, w, col in zip(k, factors, weights, columns, strict=True)
        )

        def energy(h):
            steps.assign(dx + h * direction)
            contact.evaluate(steps, 0.001)
            return float(contact.energy.numpy()[1])

        h = 1e-9
        fd = (energy(h) - 2 * energy(0) + energy(-h)) / h**2
        self.assertGreater(curvature, 0.0)
        self.assertAlmostEqual(curvature, fd, delta=1e-4 * curvature)

    def test_published_sliding_force_and_zero_identity(self):
        for precision in [SolverMonolithic.Precision.FP32, SolverMonolithic.Precision.FP64]:
            solver, state, ids = sliding_fixture(0.35, precision=precision)
            dt = 0.001
            v = np.zeros_like(state.particle_q.numpy())
            v[ids, 0] = 0.1
            state.particle_qd.assign(v)
            solver.reset_state(state, dt=dt)
            solver.step(state, state, solver.create_control(), dt=dt)
            self.assertFalse(solver.last_stats.rolled_back, solver.last_stats.status)
            diag = solver.get_cable_contact_diagnostics(state)
            self.assertTrue(diag["force_valid"])
            self.assertLess(diag["particle_force_N"][ids, 0].sum(), -0.01)
            self.assertLess(state.particle_qd.numpy()[ids, 0].mean(), 0.09)
        points, chains = hairpin()
        old, _ = cable_contact_solver(points, chains, config={"self": True})
        zero, _ = cable_contact_solver(points, chains, config={"self": True, "friction_coefficient": 0.0})
        np.testing.assert_array_equal(old._cable_contact.identity, zero._cable_contact.identity)
