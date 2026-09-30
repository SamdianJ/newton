# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run with the isolated pinned PyAMG research dependency; normal runtime has no dependency."""

import importlib.util
import unittest
from itertools import pairwise

import numpy as np

from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference import preconditioner_study as study
from scripts.monolithic_reference.preconditioner_amg import AmgOracle, elasticity_candidates


class TestAmgCandidates(unittest.TestCase):
    def test_rank_and_physical_scaling(self):
        case = study.synthetic_case(8)
        candidates = elasticity_candidates(case.arrays, case.nq)
        self.assertEqual(candidates.shape, (24, 6))
        np.testing.assert_allclose(candidates.T @ candidates, np.eye(6), atol=1e-12)
        case.arrays["rest_x"][:] = 0
        candidates = elasticity_candidates(case.arrays, case.nq)
        self.assertEqual(candidates.shape, (24, 3))
        physical = candidates * case.arrays["S"][case.nq :, None]
        np.testing.assert_allclose(physical.reshape(8, 3, 3), np.broadcast_to(physical[:3], (8, 3, 3)), atol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec("pyamg"), "Requires isolated pyamg==5.3.0")
    def test_symmetric_fixed_cycle_and_numeric_refresh(self):
        case = study.synthetic_case(30)
        reference = benchmark.Reference(case, benchmark.pcg_config())
        oracle = AmgOracle(reference, case.arrays)
        rng = np.random.default_rng(86)
        a, b = rng.normal(size=(2, case.n))
        for update in range(2):
            if update:
                case.arrays["S"] *= np.float32(1.4)
                case.metadata["lambda"] = 0.13
                reference = benchmark.Reference(case, reference.config)
                oracle.update(reference)
            np.testing.assert_allclose(oracle.apply(a), oracle.apply(a), rtol=0, atol=0)
            np.testing.assert_allclose(oracle.apply(2 * a - b), 2 * oracle.apply(a) - oracle.apply(b), atol=1e-11)
            self.assertAlmostEqual(float(a @ oracle.apply(b)), float(b @ oracle.apply(a)), delta=1e-10)
            self.assertGreater(float(a @ oracle.apply(a)), 0)
            levels = oracle.hierarchy.levels
            for fine, coarse in pairwise(levels):
                np.testing.assert_allclose(coarse.A.toarray(), (fine.R @ fine.A @ fine.P).toarray(), atol=1e-12)
            result = study.solve_reference_pcg(reference, oracle.apply)
            self.assertTrue(result["accuracy"]["independent_residual_pass"])


if __name__ == "__main__":
    unittest.main()
