# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify offline PR-6D volume assets without an external tetrahedralizer."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_soft_ball import generate_ball, load_ball, validate_ball
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestMonolithicSoftBall(unittest.TestCase):
    def test_resolution(self):
        """Check conforming closed spheres and decreasing geometric volume error."""
        previous = 1.0
        for refinement, counts in ((1, (27, 48, 48)), (2, (125, 384, 192)), (3, (343, 1296, 432))):
            mesh = generate_ball(refinement)
            report = validate_ball(mesh.vertices, mesh.tet_indices.reshape(-1, 4))
            self.assertEqual(tuple(report[k] for k in ("node_count", "tet_count", "boundary_count")), counts)
            self.assertLess(report["relative_volume_error"], previous)
            self.assertGreater(report["interior_node_count"], 0)
            previous = report["relative_volume_error"]
            np.testing.assert_array_equal(mesh.vertices, generate_ball(refinement).vertices)
        self.assertLess(previous, 0.03)

    def test_invalid_topology(self):
        """Reject inverted, duplicate, disconnected and unreferenced volume data."""
        mesh = generate_ball(1)
        x, t = mesh.vertices, mesh.tet_indices.reshape(-1, 4)
        bad = t.copy()
        bad[0, :2] = bad[0, 1::-1]
        for vertices, indices in (
            (x, bad),
            (x, np.vstack((t, t[:1]))),
            (np.vstack((x, x[:1])), t),
            (x, t.astype(float)),
            (x * np.nan, t),
            (x, t[:-1]),
            (np.vstack((x, x + np.array([0.1, 0, 0]))), np.vstack((t, t + len(x)))),
        ):
            with self.assertRaises(ValueError):
                validate_ball(vertices, indices)
        for refinement in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                generate_ball(refinement)

    def test_frozen_loading(self):
        """Round-trip through Newton npz and reject a changed material snapshot."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ball.npz"
            mesh = generate_ball(1)
            mesh.save(str(path))
            loaded = load_ball(path)
            np.testing.assert_array_equal(loaded.tet_indices, mesh.tet_indices)
            with np.load(path) as data:
                arrays = dict(data)
            arrays["k_damp"] = np.array([1.0])
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "material mismatch"):
                load_ball(path)


def test_free_ball(test, device):
    """Verify all-dynamic sphere BE free fall for both material/mass modes."""
    for refinement in (1, 2, 3):
        for material, mass in (("kim_stable_no_log", "lumped"), ("smith_log_stabilized", "consistent")):
            b = newton.ModelBuilder(gravity=(0, 0, -9.81))
            body = b.add_link(mass=1.0, inertia=wp.mat33(np.eye(3).flatten()))
            joint = b.add_joint_prismatic(-1, body, limit_ke=0.0, limit_kd=0.0)
            b.add_articulation([joint])
            b.add_soft_mesh(
                pos=(0, 0, 0.05),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0, 0, 0),
                mesh=generate_ball(refinement),
                particle_radius=0.0002,
                add_surface_mesh_edges=False,
            )
            m = b.finalize(device=device)
            test.assertTrue(np.all(m.particle_inv_mass.numpy() > 0))
            kwargs = (
                {"tet_rest_density": wp.full(m.tet_count, 1000.0, dtype=float, device=device)}
                if mass == "consistent"
                else {}
            )
            solver = SolverMonolithic(
                m,
                collision_pipeline=MonolithicCollisionPipeline(m, soft_contact_gap=0.002),
                contact_stiffness=1e7,
                material_model=material,
                mass_mode=mass,
                **kwargs,
            )
            state, next_state = m.state(), m.state()
            rest = state.particle_q.numpy().copy()
            dt = 0.001
            for _ in range(5):
                solver.step(state, next_state, m.control(), None, dt)
                state, next_state = next_state, state
            expected = -9.81 * dt * dt * 15
            np.testing.assert_allclose(
                state.particle_q.numpy() - rest, np.tile([0, 0, expected], (len(rest), 1)), atol=2e-7
            )
            np.testing.assert_allclose(
                state.particle_qd.numpy(), np.tile([0, 0, -9.81 * dt * 5], (len(rest), 1)), atol=2e-5
            )
            test.assertTrue(solver.last_stats.converged)


for device in get_test_devices():
    add_function_test(TestMonolithicSoftBall, "test_free_ball", test_free_ball, devices=[device])


if __name__ == "__main__":
    unittest.main()
