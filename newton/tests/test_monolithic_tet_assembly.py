# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the shared tet components without a tet-only solver entry."""

import subprocess
import sys
import unittest
from itertools import product

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from newton._src.solvers.monolithic.tet_precision import TetObjective
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle, material


def test_objective_and_assembly(test, device):
    """Compare both materials and mass modes with an independent E/r and scatter oracle."""
    for smith, consistent, fixed in product((False, True), repeat=3):
        model = build_model(device, fixed=fixed, gravity=(1, -2, -9.81))
        objective = TetObjective(
            model,
            material_model="smith_log_stabilized" if smith else "kim_stable_no_log",
            mass_mode="consistent" if consistent else "lumped",
            density=wp.full(model.tet_count, 1000.0, device=device) if consistent else None,
        )
        initial = model.particle_q.numpy().astype(float)
        positions = initial @ np.array([[1.12, 0.07, 0], [0.01, 0.91, 0.03], [0, 0, 1.06]]).T
        positions[model.particle_inv_mass.numpy() == 0] = initial[model.particle_inv_mass.numpy() == 0]
        x = wp.array(positions, dtype=wp.vec3d, device=device)
        xn = wp.array(initial, dtype=wp.vec3d, device=device)
        vn = wp.zeros_like(xn)
        forces = wp.full(model.particle_count, wp.vec3(0.002), dtype=wp.vec3, device=device)
        objective.evaluate(x, xn, vn, forces, model.gravity, dt=0.001)
        test.assertEqual(int(objective.flags.numpy()[0]), 0)
        oracle = TetOracle(model, objective.physics)
        parts, residual, _ = oracle.evaluate(
            positions, initial, vn.numpy(), model.gravity.numpy()[-1].astype(float), forces.numpy(), 0.001
        )
        np.testing.assert_allclose(
            objective.parts.numpy(), [parts[0], parts[1], sum(parts[2:])], atol=1e-12, rtol=1e-10
        )
        np.testing.assert_allclose(objective.residual.numpy().ravel(), residual, atol=1e-10, rtol=1e-10)

        pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), objective.mapping.numpy(), x_dof_start=0)
        workspace = create_tet_assembly_workspace(
            pattern, tet_count=model.tet_count, device=device, physics=objective.physics
        )
        scatter = TetScatterBuffers(
            wp.empty(len(pattern.ax_rows), dtype=wp.mat33, device=device),
            wp.empty(len(pattern.global_rows), dtype=float, device=device),
        )
        assembly_residual = wp.empty(objective.ids.size, dtype=wp.vec3, device=device)
        assemble_tet_residual_tangent(
            model,
            candidate_particle_q=x,
            particle_q_n=xn,
            particle_qd_n=vn,
            frozen_particle_f=forces,
            dynamic_particle_ids=objective.ids,
            particle_to_dynamic=objective.mapping,
            residual_x=assembly_residual,
            dt=0.001,
            min_det_f_guard=0.2,
            workspace=workspace,
            scatter=scatter,
        )
        test.assertEqual(int(workspace.failure_flags.numpy()[0]), 0)
        np.testing.assert_allclose(assembly_residual.numpy().ravel(), residual, atol=2e-5, rtol=2e-5)
        matrix = np.zeros((len(residual), len(residual)))
        np.add.at(matrix, (pattern.global_rows, pattern.global_columns), scatter.global_values.numpy())
        owner = np.zeros_like(matrix)
        for row, col, block in zip(pattern.ax_rows, pattern.ax_columns, scatter.ax_values.numpy(), strict=True):
            owner[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += block
        np.testing.assert_array_equal(owner, matrix)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-4, rtol=1e-6)
        test.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)
        expected_b = oracle.mass.copy() / 0.001**2
        for ids, pose, params in zip(oracle.indices, oracle.poses, oracle.materials, strict=True):
            deformation = (positions[ids[1:]] - positions[ids[0]]).T @ pose
            gradients = np.vstack((-pose.sum(axis=0), pose))
            mapping = np.column_stack([np.kron(g[:, None], np.eye(3)) for g in gradients])
            if smith:
                _, _, tangent = material(deformation, params[0], params[1], True)
            else:
                a, b, c = deformation.T
                cof = np.column_stack((np.cross(b, c), np.cross(c, a), np.cross(a, b))).ravel(order="F")
                tangent = params[0] * np.eye(9) + (params[0] + params[1]) * np.outer(cof, cof)
            nodal = mapping.T @ tangent @ mapping / (6 * np.linalg.det(pose))
            if smith:
                eigenvalues, eigenvectors = np.linalg.eigh(nodal)
                nodal = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
            dofs = (3 * ids[:, None] + np.arange(3)).ravel()
            expected_b[np.ix_(dofs, dofs)] += nodal
        np.testing.assert_allclose(matrix, expected_b[np.ix_(oracle.dofs, oracle.dofs)], atol=2e-3, rtol=2e-5)


class TestTetComponents(unittest.TestCase):
    def test_import_boundary(self):
        """Load production tet and iteration components without either historical tet module."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic import tet_common, tet_material, tet_assembly, tet_precision; "
                "from newton._src.solvers.monolithic.variational_iteration import acceptance; "
                "assert not any('newton._src.solvers.monolithic.' + name in sys.modules "
                "for name in ('tet', 'tet_variational', 'solver_monolithic', 'articulation_solver'))",
            ],
            check=True,
        )


for _device in get_test_devices():
    add_function_test(TestTetComponents, "test_objective_and_assembly", test_objective_and_assembly, devices=[_device])


if __name__ == "__main__":
    unittest.main()
