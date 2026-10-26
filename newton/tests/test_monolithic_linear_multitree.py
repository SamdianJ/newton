# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Packed tree factors against independent dense algebra, including cross-tree blocks."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.convergence import ConvergencePolicy, Unit, row_groups, tolerances
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def layout():
    # Exercise row mappings independently of the production q-first ordering.
    return MonolithicLinearLayout(
        4,
        1,
        tet_scalar_offset=1,
        actor_ranges=(("articulation", 0, 1), ("tet", 1, 3), ("articulation", 4, 3)),
        articulation_ranges=((7, 0, 1), (2, 4, 3)),
    )


class TestMultiTreeLinear(unittest.TestCase):
    def test_physical_groups(self):
        rows = row_groups(layout(), [Unit.FORCE, Unit.TORQUE, Unit.TORQUE, Unit.FORCE])
        reference = np.array([1e3, 0, 0, 0, 1e-3, 2e-3, 1e-4])
        tau, _ = tolerances(reference, np.zeros(7), rows, ConvergencePolicy(), np.float64)
        np.testing.assert_allclose(tau[[0, 4, 5, 6]], [1e-3, 2e-9, 2e-9, 1e-10])
        self.assertEqual(len(set(rows.names)), len(rows.names))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            layout().indices("articulation")
        np.testing.assert_array_equal(layout().indices("articulation", 2), [4, 5, 6])


def test_packed_factors(test, device):
    for dtype in (wp.float32, wp.float64):
        for mode in ("diagnostic", "production"):
            with test.subTest(dtype=dtype, mode=mode):
                native = np.float32 if dtype == wp.float32 else np.float64
                ws = MonolithicLinearWorkspace(
                    layout(),
                    MonolithicLinearCapacities(49, 1, 0, 80, 82),
                    wp.array([0], dtype=int, device=device),
                    device=device,
                    dtype=dtype,
                    pcg_mode=mode,
                    use_cuda_graph=device.is_cuda and mode == "production",
                )
                test.assertEqual(ws.aq_actor_dense.size, 1**2 + 3**2)
                direction = np.linspace(-1, 1, 7)
                matrix = np.diag(np.arange(2, 9)) + 2 * np.outer(direction, direction)
                ws._set_fixed_triplet_pattern(
                    global_rows=np.repeat(np.arange(7), 7),
                    global_columns=np.tile(np.arange(7), 7),
                    internal_rows=np.array([0]),
                    internal_columns=np.array([0]),
                )
                generation = MonolithicLinearGeneration(1, 0, 1, 1)
                assembly = ws.begin_assembly(generation)
                assembly.global_scalar_triplets.values.assign(matrix.ravel().astype(native))
                assembly.aq_actor_dense.assign(np.r_[matrix[:1, :1].ravel(), matrix[4:, 4:].ravel()].reshape(10, 1))
                assembly.ax_internal_triplets.values.assign(matrix[None, 1:4, 1:4])
                test.assertEqual(ws.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
                ws.build_scaling(
                    wp.array(matrix.diagonal(), dtype=dtype, device=device), epsilon_d=0.01, generation=generation
                )
                ws.set_regularization(0.03, generation=generation)
                test.assertEqual(
                    ws.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
                    MonolithicLinearStatus.SUCCESS,
                )
                scale = ws.scale.numpy()
                scaled = scale[:, None] * matrix * scale[None, :] + 0.03 * np.eye(7)
                block = np.zeros((7, 7))
                for start, end in ((0, 1), (1, 4), (4, 7)):
                    block[start:end, start:end] = scaled[start:end, start:end]
                rhs = wp.array(np.arange(1, 8), dtype=dtype, device=device)
                y = wp.zeros_like(rhs)
                ws.preconditioner.matvec(rhs, y, y, 1.0, 0.0)
                tol = 2e-6 if dtype == wp.float32 else 1e-12
                np.testing.assert_allclose(y.numpy(), np.linalg.solve(block, rhs.numpy()), rtol=tol, atol=tol)
                config = MonolithicPcgConfig(
                    80, 1, 1e-7 if dtype == wp.float32 else 1e-12, 1e-12, 1e-12, 1e-12, linear_absolute_tolerance=0
                )
                result = ws.solve_pcg(
                    rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config
                )
                test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
                np.testing.assert_allclose(
                    y.numpy(), np.linalg.solve(scaled, rhs.numpy()), rtol=10 * tol, atol=10 * tol
                )


add_function_test(TestMultiTreeLinear, "test_packed_factors", test_packed_factors, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
