# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare complete cloth assembly and material preconditioning with dense references."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicPcgWarmStart,
)
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from scripts.monolithic_reference.cloth_benchmarks import ClothBenchmark
from scripts.monolithic_reference.cloth_oracle import shell_bending_tangent, shell_membrane_tangent


class TestMonolithicClothLinear(unittest.TestCase):
    def test_global_shell_matrix(self):
        """Keep every oracle cross-node block after fixed-vertex elimination and CSR reduction."""
        case = ClothBenchmark("prefold")
        case.step()
        block = case.solver._cloths[0]
        x = case.state.particle_q.numpy()
        stencils = block.objective.stencils.numpy()
        expected = np.diag(np.repeat(case.mass, 3) / case.dt**2)
        for stencil in stencils:
            boundary = tuple(stencil[3:] < 0)
            ids = np.maximum(stencil, 0)
            material = block.objective.materials.numpy()[0]
            tangent = shell_bending_tangent(case.rest[ids], x[ids], material, boundary=boundary)
            tangent[:9, :9] += shell_membrane_tangent(case.rest[ids], x[ids], material)
            for a, p in enumerate(stencil):
                for b, q in enumerate(stencil):
                    if p >= 0 and q >= 0:
                        expected[3 * p : 3 * p + 3, 3 * q : 3 * q + 3] += tangent[3 * a : 3 * a + 3, 3 * b : 3 * b + 3]
        free = np.repeat(~case.fixed, 3)
        expected = expected[np.ix_(free, free)]
        matrix = case.solver._linear.k_global_scalar_bsr
        actual = np.zeros_like(expected)
        offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
        for row in range(len(actual)):
            actual[row, columns[offsets[row] : offsets[row + 1]]] = values[offsets[row] : offsets[row + 1]]
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), 2e-5)
        self.assertGreater(np.linalg.norm(actual[:3, 3:]), 1)

    def test_coupled_dense_reference_and_storage(self):
        """Solve cloth/cable cross-object terms and reject graph factor buffer replacement."""
        layout = MonolithicLinearLayout(
            0,
            0,
            actor_ranges=(("cloth", 0, 6), ("cable", 6, 4), ("cloth", 10, 3)),
            material_ranges=(("cloth", 0, 0, 6, 0), ("cable", 0, 6, 3, 1), ("cloth", 1, 10, 3, 0)),
        )
        rng = np.random.default_rng(17)
        a = rng.normal(size=(13, 13))
        matrix = a @ a.T + 3 * np.eye(13)
        rows, columns = np.indices(matrix.shape).reshape(2, -1)
        for device in ["cpu"] + ([wp.get_cuda_device()] if wp.is_cuda_available() else []):
            for mode in ("diagnostic", "production"):
                with self.subTest(device=str(device), mode=mode):
                    workspace = MonolithicLinearWorkspace(
                        layout,
                        MonolithicLinearCapacities(169, 0, 0, 100, 102),
                        wp.array([], dtype=int, device=device),
                        device=device,
                        pcg_mode=mode,
                        use_cuda_graph=(mode == "production" and str(device) != "cpu"),
                        cable_row_order=np.arange(6, 10),
                    )
                    workspace._set_fixed_triplet_pattern(
                        global_rows=rows,
                        global_columns=columns,
                        internal_rows=np.array([], dtype=int),
                        internal_columns=np.array([], dtype=int),
                    )
                    generation = MonolithicLinearGeneration(1, 0, 1, 1)
                    assembly = workspace.begin_assembly(generation)
                    assembly.global_scalar_triplets.values.assign(matrix.ravel().astype(np.float32))
                    self.assertEqual(
                        workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS
                    )
                    self.assertEqual(
                        workspace.build_coordinate_scaling(np.linspace(0.5, 2, 13), generation=generation),
                        MonolithicLinearStatus.SUCCESS,
                    )
                    self.assertEqual(
                        workspace.set_regularization(0.1, generation=generation), MonolithicLinearStatus.SUCCESS
                    )
                    self.assertEqual(
                        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
                        MonolithicLinearStatus.SUCCESS,
                    )
                    scale = workspace.scale.numpy()
                    rhs = np.linspace(-1, 1, 13)
                    rhs_hat = wp.array(rhs * scale, dtype=float, device=device)
                    output, delta = wp.zeros(13, dtype=float, device=device), wp.zeros(13, dtype=float, device=device)
                    config = _pcg_config(
                        linear_tolerance=1e-6,
                        true_residual_interval=1,
                        group_norm_limits=(1e-5,) * len(layout.groups),
                        group_residual_floors=(1e-12,) * len(layout.groups),
                    )
                    result = workspace.solve_pcg(
                        rhs_hat,
                        output,
                        generation=generation,
                        warm_start=MonolithicPcgWarmStart.ZERO,
                        config=config,
                        delta=delta,
                    )
                    self.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
                    if mode == "diagnostic":
                        workspace.recover_delta(output, delta, generation=generation)
                    actual_matrix = matrix + np.diag(0.1 / scale**2)
                    np.testing.assert_allclose(delta.numpy(), np.linalg.solve(actual_matrix, rhs), atol=2e-6, rtol=2e-5)
                    self.assertLess(np.linalg.norm(actual_matrix @ delta.numpy() - rhs), 1e-5)
                    if mode == "production":
                        workspace._cloth_preconditioner.factors = wp.zeros_like(workspace._cloth_preconditioner.factors)
                        with self.assertRaises(ValueError):
                            workspace._masked_pcg_plan._validate_storage()


if __name__ == "__main__":
    unittest.main()
