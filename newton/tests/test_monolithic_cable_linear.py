# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify cable band factors inside the existing global diagnostic/production PCG."""

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


class TestMonolithicCableLinear(unittest.TestCase):
    device = "cpu"

    def test_coupled_permuted_band_solve(self):
        """Keep cross-object terms in the global solve while factoring each cable band."""
        layout = MonolithicLinearLayout(
            0,
            0,
            actor_ranges=(("cable", 0, 24), ("cable", 24, 4)),
            material_ranges=(("cable", 0, 0, 18, 6), ("cable", 1, 24, 3, 1)),
        )
        order = np.r_[np.array([[3 * i, 3 * i + 1, 3 * i + 2, 18 + i] for i in range(6)]).ravel(), [24, 25, 26, 27]]
        lower = np.eye(28) * 2
        for start, end in ((0, 24), (24, 28)):
            for i in range(start, end):
                for j in range(max(start, i - 10), i):
                    lower[i, j] = 0.02 * np.cos(i + j)
        matrix = np.zeros((28, 28))
        matrix[np.ix_(order, order)] = lower @ lower.T
        matrix[0, 24] = matrix[24, 0] = 0.1
        rows, columns = np.nonzero(matrix)
        for mode in ("diagnostic", "production"):
            with self.subTest(mode=mode):
                workspace = MonolithicLinearWorkspace(
                    layout,
                    MonolithicLinearCapacities(len(rows), 0, 0, 100, 102),
                    wp.array([], dtype=int, device=self.device),
                    device=self.device,
                    pcg_mode=mode,
                    cable_row_order=order,
                    use_cuda_graph=(mode == "production" and self.device != "cpu"),
                )
                workspace._set_fixed_triplet_pattern(
                    global_rows=rows,
                    global_columns=columns,
                    internal_rows=np.array([], dtype=int),
                    internal_columns=np.array([], dtype=int),
                )
                generation = MonolithicLinearGeneration(1, 0, 1, 1)
                assembly = workspace.begin_assembly(generation)
                assembly.global_scalar_triplets.values.assign(matrix[rows, columns].astype(np.float32))
                self.assertEqual(
                    workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS
                )
                self.assertEqual(
                    workspace.build_scaling(
                        wp.ones(28, dtype=float, device=self.device), epsilon_d=1e-12, generation=generation
                    ),
                    MonolithicLinearStatus.SUCCESS,
                )
                self.assertEqual(
                    workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
                    MonolithicLinearStatus.SUCCESS,
                )
                scale = workspace.scale.numpy()
                rhs = np.linspace(-1, 1, 28)
                rhs_hat = wp.array(rhs * scale, dtype=float, device=self.device)
                output, delta = (
                    wp.zeros(28, dtype=float, device=self.device),
                    wp.zeros(28, dtype=float, device=self.device),
                )
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
                np.testing.assert_allclose(delta.numpy(), np.linalg.solve(matrix, rhs), atol=2e-6, rtol=2e-5)
                self.assertLess(np.linalg.norm(matrix @ delta.numpy() - rhs), 1e-5)
                if mode == "production":
                    cable = workspace._cable_preconditioner
                    cable.factor_values = wp.zeros_like(cable.factor_values)
                    with self.assertRaises(ValueError):
                        workspace._masked_pcg_plan._validate_storage()

                self.assertEqual(
                    workspace.build_coordinate_scaling(np.ones(28), generation=generation),
                    MonolithicLinearStatus.SUCCESS,
                )
                self.assertEqual(
                    workspace.build_coordinate_scaling(np.ones(27), generation=generation),
                    MonolithicLinearStatus.INVALID_ARGUMENT,
                )
                self.assertEqual(
                    workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
                    MonolithicLinearStatus.STALE_GENERATION,
                )


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMonolithicCableLinearCUDA(TestMonolithicCableLinear):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
