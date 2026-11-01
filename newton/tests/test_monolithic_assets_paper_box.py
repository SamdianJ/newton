# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify a free open paper shell and its geometry diagnostics."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_paper_box import PaperBoxMetrics, add_paper_box
from newton.solvers.experimental.monolithic import SolverMonolithic
from scripts.monolithic_reference.cloth_oracle import shell_residual
from scripts.monolithic_reference.material_recipes import shell_stencils


class TestPaperBox(unittest.TestCase):
    def test_open_connected_free_shell_at_rest(self):
        """Construct five welded faces with outward normals, physical mass and stress-free folds."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        material = add_paper_box(builder, (0.0, 0.0, 0.02))
        model = builder.finalize(device="cpu")
        x, triangles = model.particle_q.numpy(), model.tri_indices.numpy()
        self.assertEqual((len(x), len(triangles), model.tet_count), (193, 360, 0))
        np.testing.assert_allclose(np.ptp(x, axis=0), 0.04, rtol=1e-6)
        self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0))
        self.assertAlmostEqual(float(model.particle_mass.numpy().sum()), 0.0028, places=9)
        faces = x[triangles].astype(float)
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        self.assertTrue(np.all(np.sum(normal * (faces.mean(axis=1) - [0, 0, 0.02]), axis=1) > 0))
        edges, counts = np.unique(
            np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        self.assertTrue(np.all(counts <= 2))
        boundary = edges[counts == 1]
        self.assertEqual(len(boundary), 24)
        np.testing.assert_allclose(x[boundary, 2], 0.04, rtol=1e-6)
        coeff = (material["lambda_2d_N_m"], material["mu_2d_N_m"], *material["bending_Nm"])
        for rest, missing in shell_stencils(x.astype(float), triangles):
            np.testing.assert_allclose(shell_residual(rest, rest, coeff, boundary=missing), 0, atol=1e-8)
        solver = SolverMonolithic(
            model,
            cloth_inputs=(
                SolverMonolithic.ClothInput(
                    triangle_ids=tuple(range(model.tri_count)),
                    bending=material["bending_Nm"],
                    thickness=material["thickness_m"],
                    surface_density=material["surface_density_kg_m2"],
                ),
            ),
            mass_mode="lumped",
            friction_velocity_scale=0.01,
        )
        self.assertEqual(len(solver._cloths), 1)
        self.assertEqual(
            PaperBoxMetrics(model).measure(solver.create_state(dt=0.001).particle_q)["self_intersections"], 0
        )

    def test_self_intersection_and_area_measurements(self):
        """Detect crossing and coplanar overlap, but exclude separated and adjacent faces."""
        builder = newton.ModelBuilder()
        for p in [
            (0, 0, 0),
            (0.01, 0, 0),
            (0, 0.01, 0),
            (0.002, 0.002, -0.005),
            (0.002, 0.002, 0.005),
            (0.008, 0.002, 0),
        ]:
            builder.add_particle(wp.vec3(p), wp.vec3(0), 1)
        builder.add_triangle(0, 1, 2)
        builder.add_triangle(3, 4, 5)
        model = builder.finalize(device="cpu")
        metrics = PaperBoxMetrics(model)
        x = model.particle_q.numpy().astype(float)
        for dtype in (wp.vec3, wp.vec3d):
            positions = wp.array(x, dtype=dtype, device="cpu")
            self.assertEqual(metrics.measure(positions)["self_intersections"], 1)
            separated = x.copy()
            separated[3:] += [0, 0, 0.02]
            positions.assign(separated)
            result = metrics.measure(positions)
            self.assertEqual(result["self_intersections"], 0)
            self.assertAlmostEqual(result["area_ratio_min"], 1, places=5)
            overlap = x.copy()
            overlap[3:] = [[0.002, 0.002, 0], [0.008, 0.002, 0], [0.002, 0.008, 0]]
            positions.assign(overlap)
            self.assertEqual(metrics.measure(positions)["self_intersections"], 1)


if __name__ == "__main__":
    unittest.main()
