# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent projection and native dispatch checks for the private PR8F prototype."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicLinearStatus
from newton._src.solvers.monolithic.pcg_production import MaskedPcgPlan
from newton._src.solvers.monolithic.soft_preconditioner import _BlockPreconditioner
from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference import preconditioner_study as study


def case_with_geometry(index=0):
    case = benchmark.synthetic_cases()[index]
    nx = (case.n - case.nq) // 3
    case.arrays.update(
        {
            k: v
            for k, v in study.synthetic_case(nx).arrays.items()
            if k in ("rest_x", "dynamic_particle_ids", "tet_indices")
        }
    )
    return case


class TestSoftPreconditioner(unittest.TestCase):
    def test_native_projection_and_apply(self):
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            for candidate in study.CANDIDATES[1:]:
                case = case_with_geometry()
                reference = benchmark.Reference(case, benchmark.pcg_config())
                hierarchy = study.build_hierarchy(case.arrays, candidate)
                oracle = study.BlockOracle(reference, hierarchy)
                runner = benchmark.Runner(case, "diagnostic", device, reference.config)
                w = runner.workspace
                w._study_soft = block = _BlockPreconditioner(w, hierarchy)
                runner.factor()
                for i, expected in enumerate(oracle.blocks):
                    actual = block.matrices.numpy()[i * block.width : i * block.width + len(expected), : len(expected)]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                out = wp.zeros_like(runner.rhs)
                w.preconditioner.matvec(runner.rhs, out, out, 1.0, 0.0)
                np.testing.assert_allclose(out.numpy(), oracle.apply(case.arrays["rhs_hat"]), rtol=2e-5, atol=2e-5)
                for mode in ("diagnostic", "production_graph") if device != "cpu" else ("diagnostic", "production"):
                    runner = benchmark.Runner(case, mode, device, reference.config)
                    runner.workspace._study_soft = _BlockPreconditioner(runner.workspace, hierarchy)
                    runner.factor()
                    result = runner.solve()
                    self.assertEqual(result.status.name, "SUCCESS")
                    self.assertTrue(reference.measure(runner.y.numpy())["independent_residual_pass"])

    def test_coarse_updates_alias_and_linearity(self):
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            case = case_with_geometry(1)
            hierarchy = study.build_hierarchy(case.arrays, "mas_16")
            self.assertGreater(len(hierarchy.level_counts), 1)
            runner = benchmark.Runner(case, "diagnostic", device, benchmark.pcg_config(), study_hierarchy=hierarchy)
            w = runner.workspace
            signature = w._study_soft.signature()
            rng = np.random.default_rng(85)
            a, b = rng.normal(size=(2, case.n)).astype(np.float32)

            def apply(value, device=device, w=w):
                vector = wp.array(value, dtype=float, device=device)
                w.preconditioner.matvec(vector, vector, vector, 1.0, 0.0)
                return vector.numpy()

            for update in range(2):
                if update:
                    runner.refresh_generation()
                    case.arrays["S"] *= np.float32(1.3)
                    case.metadata["lambda"] = 0.07
                    w.scale.assign(case.arrays["S"])
                    self.assertEqual(
                        w.set_regularization(0.07, generation=w._generation), MonolithicLinearStatus.SUCCESS
                    )
                    runner.factor()
                reference = benchmark.Reference(case, runner.config)
                oracle = study.BlockOracle(reference, hierarchy)
                for actual in (apply(a), apply(a)):
                    np.testing.assert_allclose(actual, oracle.apply(a), rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(apply(2 * a - b), 2 * apply(a) - apply(b), rtol=2e-5, atol=2e-5)
                self.assertAlmostEqual(float(a @ apply(b)), float(b @ apply(a)), delta=2e-3)
                self.assertGreater(float(a @ apply(a)), 0)
                for i, expected in enumerate(oracle.blocks):
                    block = w._study_soft
                    actual = block.matrices.numpy()[i * block.width : i * block.width + len(expected), : len(expected)]
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                self.assertEqual(w._study_soft.signature(), signature)

    def test_failures_and_empty_blocks(self):
        for index in (3, 5):
            case = case_with_geometry(index)
            hierarchy = study.build_hierarchy(case.arrays, "mas_16")
            runner = benchmark.Runner(case, "production", "cpu", benchmark.pcg_config(), study_hierarchy=hierarchy)
            self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
            if index == 5:
                np.testing.assert_array_equal(runner.y.numpy(), 0)
        case = case_with_geometry()
        hierarchy = study.build_hierarchy(case.arrays, "mas_16")
        for bad in (-1.0, float("nan")):
            runner = benchmark.Runner(case, "diagnostic", "cpu", benchmark.pcg_config(), study_hierarchy=hierarchy)
            w = runner.workspace
            matrix = w.k_global_scalar_bsr
            values = matrix.values.numpy()
            values[0:] = bad
            matrix.values.assign(values)
            w._factor_valid = False
            status = w.factor_actor_preconditioner(generation=w._generation, pivot_tolerance=0)
            self.assertIn(
                status, (MonolithicLinearStatus.NON_POSITIVE_PIVOT, MonolithicLinearStatus.NONFINITE_ITERATION)
            )
            self.assertFalse(w._factor_valid)

    def test_mask_and_nonfinite_rhs(self):
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            case = case_with_geometry()
            hierarchy = study.build_hierarchy(case.arrays, "mas_16")
            runner = benchmark.Runner(case, "production", device, benchmark.pcg_config(), study_hierarchy=hierarchy)
            block = runner.workspace._study_soft
            mask = wp.zeros(1, dtype=int, device=device)
            out = wp.full(case.n, 7.0, dtype=float, device=device)
            runner.rhs.fill_(float("nan"))
            block.apply(runner.rhs, out, active=mask)
            np.testing.assert_array_equal(out.numpy(), 7.0)
            self.assertEqual(runner.solve().status, MonolithicLinearStatus.NONFINITE_ITERATION)
            runner.rhs.zero_()
            self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
            np.testing.assert_array_equal(runner.y.numpy(), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph required")
    def test_graph_reuse_rebinding_and_stale_generation(self):
        case = case_with_geometry()
        hierarchy = study.build_hierarchy(case.arrays, "mas_16")
        runner = benchmark.Runner(case, "production_graph", "cuda:0", benchmark.pcg_config(), study_hierarchy=hierarchy)
        w = runner.workspace
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
        plan = w._masked_pcg_plan
        graph = plan.graph
        old_generation = w._generation
        runner.refresh_generation()
        runner.factor()
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
        self.assertIs(w._masked_pcg_plan, plan)
        self.assertIs(plan.graph, graph)
        w.set_regularization(0.2, generation=w._generation)
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
        runner.factor()
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
        self.assertIs(w._masked_pcg_plan.graph, graph)
        with wp.ScopedCapture(device=w.device):
            runner.y.zero_()
            with patch.object(wp, "zeros", side_effect=AssertionError("Allocation before capture guard")):
                with self.assertRaisesRegex(RuntimeError, "Construct/bind"):
                    MaskedPcgPlan.from_workspace(w, runner.rhs, runner.y, config=plan.config, delta=runner.delta)
        self.assertEqual(
            w.factor_actor_preconditioner(generation=old_generation, pivot_tolerance=0),
            MonolithicLinearStatus.STALE_GENERATION,
        )
        w._study_soft = _BlockPreconditioner(w, hierarchy)
        runner.factor()
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.SUCCESS)
        self.assertIsNot(w._masked_pcg_plan, plan)
        block = w._study_soft
        block.blocks = wp.clone(block.blocks)
        self.assertEqual(runner.solve().status, MonolithicLinearStatus.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main()
