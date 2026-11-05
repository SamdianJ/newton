# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check open-bottle topology, mass and its stress-free curved reference."""

import os
import unittest

import numpy as np

import newton
from scripts.monolithic_reference.cloth_oracle import shell_residual
from scripts.monolithic_reference.graspit_hand import build_hand
from scripts.monolithic_reference.hand_admittance import HandAdmittance
from scripts.monolithic_reference.material_recipes import shell_coefficients, shell_stencils
from scripts.monolithic_reference.plastic_bottle import add_bottle
from scripts.monolithic_reference.run_graspit_hand import shape_vertices


class TestPlasticBottle(unittest.TestCase):
    def test_initial_hand_clears_bottle_and_support(self):
        """Reject initial hand collisions with the bottle or its support plane."""
        from scipy.spatial import ConvexHull

        asset = os.environ.get("NEWTON_GRASPIT_ASSET_DIR")
        if not asset:
            self.skipTest("Set NEWTON_GRASPIT_ASSET_DIR for the external hand asset")
        builder, _ = build_hand(asset)
        add_bottle(builder)
        model = builder.finalize(device="cpu")
        controller = HandAdmittance(model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy())
        state = model.state()
        state.joint_q.assign(controller.opening)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        x = model.particle_q.numpy().astype(float)
        samples = np.concatenate((x, x[model.tri_indices.numpy()].mean(axis=1)))
        for s, flags in enumerate(model.shape_flags.numpy()):
            if flags & int(newton.ShapeFlags.VISIBLE):
                continue
            points = shape_vertices(model, state.body_q.numpy(), s)
            self.assertGreater(float(points[:, 1].min()), -0.080125 + 0.0005)
            planes = ConvexHull(points).equations
            self.assertGreater(float((samples @ planes[:, :3].T + planes[:, 3]).max(1).min()), 0.000125)

    def test_free_oriented_shell_mass_and_rest_state(self):
        """Keep one open neck, welded sides and bottom, and zero reference stress."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        material = add_bottle(builder, center=(0, 0, 0))
        model = builder.finalize(device="cpu")
        x, faces = model.particle_q.numpy().astype(float), model.tri_indices.numpy()
        self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0))
        edges, counts = np.unique(
            np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        self.assertTrue(np.all(counts <= 2))
        self.assertEqual(np.sum(counts == 1), 24)
        np.testing.assert_allclose(x[edges[counts == 1], 1], 0.08, atol=1e-8)
        p = x[faces]
        normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 1e-8))
        area = np.linalg.norm(normals, axis=1).sum() / 2
        self.assertAlmostEqual(model.particle_mass.numpy().sum(), area * material["surface_density_kg_m2"], places=8)
        stencils = shell_stencils(x, faces)
        for rest, missing in stencils[::37]:
            np.testing.assert_allclose(
                shell_residual(rest, rest, shell_coefficients(material), boundary=missing), 0, atol=1e-8
            )
        centers = p.mean(1)
        side = np.abs(centers[:, 1] + 0.08) > 1e-7
        self.assertTrue(np.all(np.sum(normals[side][:, [0, 2]] * centers[side][:, [0, 2]], axis=1) > 0))
        self.assertTrue(np.all(normals[~side, 1] < 0))


if __name__ == "__main__":
    unittest.main()
