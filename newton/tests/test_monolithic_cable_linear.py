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
from newton.tests.test_monolithic_cable_contact import L, R, cable_contact_solver, hairpin
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from newton.tests.test_monolithic_material_state import snapshot


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
                    cable.block_factor = wp.zeros_like(cable.block_factor)
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

    def test_far_band_entry_is_rejected_without_cable_contact(self):
        """Without cable contact a nonzero entry beyond the band of one cable is still an invalid argument."""
        layout = MonolithicLinearLayout(
            0,
            0,
            actor_ranges=(("cable", 0, 24),),
            material_ranges=(("cable", 0, 0, 18, 6),),
        )
        order = np.array([[3 * i, 3 * i + 1, 3 * i + 2, 18 + i] for i in range(6)]).ravel()
        matrix = np.eye(24) * 2
        # Permuted positions 0 and 20 of the same cable lie 20 > 10 apart.
        matrix[order[20], order[0]] = matrix[order[0], order[20]] = 0.05
        rows, columns = np.nonzero(matrix)
        workspace = MonolithicLinearWorkspace(
            layout,
            MonolithicLinearCapacities(len(rows), 0, 0, 100, 102),
            wp.array([], dtype=int, device=self.device),
            device=self.device,
            cable_row_order=order,
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
        self.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
        self.assertEqual(
            workspace.build_scaling(
                wp.ones(24, dtype=float, device=self.device), epsilon_d=1e-12, generation=generation
            ),
            MonolithicLinearStatus.SUCCESS,
        )
        self.assertEqual(
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
            MonolithicLinearStatus.INVALID_ARGUMENT,
        )
        # The contact template needs the cable element operator that defines it.
        with self.assertRaisesRegex(ValueError, "element operator"):
            MonolithicLinearWorkspace(
                layout,
                MonolithicLinearCapacities(len(rows), 0, 0, 100, 102),
                wp.array([], dtype=int, device=self.device),
                device=self.device,
                cable_row_order=order,
                cable_contact=True,
            )

    def test_inactive_cable_contact_changes_nothing(self):
        """Cable contact enabled with no pair in reach leaves steps and band factors equal bit for bit.

        A single-edge cable and a cable with a fixed vertex and fixed twists lie on a ground plane, so
        shape contact writes position and twist entries the contact template must keep.
        """
        single = np.array([[0.0, 0.0, R - 0.0002], [L, 0.0, R - 0.0002]])
        longer = np.column_stack([L * np.arange(6), np.full(6, 0.05), np.full(6, R - 0.0002)])
        points = np.vstack([single, longer])
        chains = [range(2), range(2, 8)]
        results = {}
        for config in (None, {}):
            solver, state = cable_contact_solver(
                points,
                chains,
                config=config,
                fixed=(2,),
                fixed_theta=[None, [True, False, False, False, True]],
                ground=0.0,
                gravity=(0.0, 0.0, -9.81),
                device=self.device,
            )
            control = solver.create_control()
            for _ in range(3):
                solver.step(state, state, control, dt=0.001)
                self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
            if config is not None:
                self.assertEqual(solver._cable_contact._candidate_count, 0)
                self.assertEqual(solver._linear._cable_preconditioner.masked, 1)
                # Shape contact couples the single-edge cable's twist to its positions; only the edge stencil
                # of the template covers these entries (a single edge has no bend element).
                matrix = solver._linear.k_global_scalar_bsr
                offsets, columns, values = (a.numpy() for a in (matrix.offsets, matrix.columns, matrix.values))
                theta = int(solver._actors.maps["theta_to_global"][solver._cables[0].theta_offset])
                rows = solver._actors.maps["particle_to_global"]
                xs = {int(rows[p]) + axis for p in (0, 1) for axis in range(3)}
                coupling = [abs(values[k]) for k in range(offsets[theta], offsets[theta + 1]) if columns[k] in xs]
                self.assertGreaterEqual(theta, 0)
                self.assertGreater(max(coupling, default=0.0), 0.0)
            self.assertGreater(np.abs(solver.get_contact_forces(state).particle_force).max(), 0.0)
            cable = solver._linear._cable_preconditioner
            factors = [a.numpy() for a in (cable.block_factor, cable.block_left, cable.block_right)]
            results[config is None] = (snapshot(state), factors)
        enabled, disabled = results[False], results[True]
        # The configuration digest differs by design: cable contact enters the checkpoint identity.
        self.assertFalse(np.array_equal(enabled[0]["history.config_digest"], disabled[0]["history.config_digest"]))
        for name, value in disabled[0].items():
            if name != "history.config_digest":
                np.testing.assert_array_equal(enabled[0][name], value, err_msg=name)
        for a, b in zip(enabled[1], disabled[1], strict=True):
            np.testing.assert_array_equal(a, b)

    def test_far_self_contact_still_factors(self):
        """A hairpin pressed onto itself couples rows far apart in one cable; the band preconditioner still factors."""
        solver, state = cable_contact_solver(*hairpin(), config={}, device=self.device)
        solver.step(state, state, solver.create_control(), dt=0.001)
        stats = solver.last_stats
        self.assertFalse(stats.rolled_back, stats)
        self.assertGreater(solver._cable_contact._candidate_count, 0)
        self.assertGreater(np.abs(solver.get_contact_forces(state).particle_force).max(), 0.0)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMonolithicCableLinearCUDA(TestMonolithicCableLinear):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
