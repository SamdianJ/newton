# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check fixed soft-block hierarchies and independently projected inverse applications."""

import unittest

import numpy as np

from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference import preconditioner_study as study


class TestMonolithicPreconditionerStudy(unittest.TestCase):
    def test_projection_and_spd(self):
        """Preserve all coupled-system terms, physical coarse modes and regularization in the projection."""
        case = study.synthetic_case()
        reference = benchmark.Reference(case, benchmark.pcg_config())
        for candidate in study.CANDIDATES:
            hierarchy = study.build_hierarchy(case.arrays, candidate)
            oracle = study.BlockOracle(reference, hierarchy)
            inverse = np.column_stack([oracle.apply(v) for v in np.eye(case.n)])
            np.testing.assert_allclose(inverse, inverse.T, rtol=1e-12, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(inverse).min(), 0)
            for restriction, block in zip(oracle.restrictions, oracle.blocks, strict=True):
                np.testing.assert_allclose(
                    block, (restriction @ reference.matrix[case.nq :, case.nq :] @ restriction.T).toarray()
                )
            result = study.solve_reference_pcg(reference, oracle.apply)
            self.assertTrue(result["accuracy"]["independent_residual_pass"])
            self.assertLess(result["accuracy"]["y_relative_error"], 1e-3)

    def test_disconnected_fixed_tail_and_determinism(self):
        """Exclude fixed nodes, preserve isolated/tail nodes and never merge disconnected coarse components."""
        arrays = {
            "rest_x": np.arange(36, dtype=float).reshape(12, 3),
            "dynamic_particle_ids": np.array([0, 1, 2, 3, 5, 6, 7, 8, 10]),
            "tet_indices": np.array([[0, 1, 2, 3], [5, 6, 7, 8]]),
        }
        for candidate in study.CANDIDATES:
            a = study.build_hierarchy(arrays, candidate)
            b = study.build_hierarchy(arrays, candidate)
            np.testing.assert_array_equal(a.block_ids, b.block_ids)
            self.assertEqual(a.fingerprint, b.fingerprint)
            self.assertTrue(np.all(a.block_ids >= 0))
            self.assertTrue(np.all(a.local_ids >= 0))
            if candidate.startswith("mas"):
                for groups in a.domains[a.level_counts[0] :]:
                    for group in groups:
                        self.assertTrue(
                            set(group) <= set(range(4)) or set(group) <= set(range(4, 8)) or set(group) == {8}
                        )

    def test_q_only_and_invalid_geometry(self):
        """Handle no soft unknowns and reject malformed geometry before constructing mappings."""
        arrays = {
            "rest_x": np.zeros((0, 3)),
            "dynamic_particle_ids": np.zeros(0, dtype=int),
            "tet_indices": np.zeros((0, 4), dtype=int),
        }
        hierarchy = study.build_hierarchy(arrays, "mas_16")
        self.assertEqual(len(hierarchy.domains), 0)
        self.assertEqual(hierarchy.block_ids.shape, (0, 0))
        arrays["dynamic_particle_ids"] = np.array([0])
        with self.assertRaises(ValueError):
            study.build_hierarchy(arrays, "mas_16")

    def test_mas_has_coarse_correction(self):
        """Construct additional layers for a connected mesh larger than one domain."""
        case = study.synthetic_case(40)
        hierarchy = study.build_hierarchy(case.arrays, "mas_16")
        self.assertGreater(len(hierarchy.level_counts), 1)
        self.assertLessEqual(len(hierarchy.level_counts), 3)
        self.assertEqual(hierarchy.block_ids.shape[1], 40)


if __name__ == "__main__":
    unittest.main()
