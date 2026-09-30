# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Close the shared-support SPD counterexample before any production patch apply."""

import unittest

import numpy as np

from scripts.monolithic_reference.normal_patch_spd import patch_oracle


class TestMonolithicSchurSpd(unittest.TestCase):
    def test_shared_scalar_counterexamples(self):
        """Show successful local factors can produce a singular or indefinite full apply."""
        for count in (2, 3, 8):
            result = patch_oracle([[1]], np.ones((count, 1)), np.ones(count), [1], [[i] for i in range(count)])
            self.assertAlmostEqual(result["naive_inverse"][0, 0], 1 - count / 2)
            self.assertAlmostEqual(result["inverse"][0, 0], 1 / (1 + count))
            np.testing.assert_allclose(result["inverse"], result["exact_inverse"])

    def test_single_patch_and_disjoint_support(self):
        """Recover exact Woodbury for one patch or disjoint base-block supports."""
        base = np.diag([2.0, 3.0, 4.0, 5.0])
        rows = np.array([[1, 2, 0, 0], [2, -1, 0, 0], [0, 0, 1, 3]], dtype=float)
        for patches in ([[0, 1, 2]], [[0, 1], [2]]):
            result = patch_oracle(base, rows, [2, 3, 4], [2, 2], patches)
            np.testing.assert_allclose(result["inverse"], result["exact_inverse"], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(result["schur_bound"], result["schur"], rtol=1e-12, atol=1e-14)

    def test_shared_q_and_nodes(self):
        """Verify Loewner bounds, symmetry and SPD with shared dense q and node blocks."""
        rng = np.random.default_rng(20260914)
        for regularization in (0, 1e-4, 1):
            for _ in range(12):
                base = np.zeros((8, 8))
                for lo, hi in ((0, 2), (2, 5), (5, 8)):
                    a = rng.normal(size=(hi - lo, hi - lo))
                    base[lo:hi, lo:hi] = a @ a.T + (0.5 + regularization) * np.eye(hi - lo)
                rows = rng.normal(size=(6, 8))
                rows[:2, 5:] = 0
                rows[4:, 2:5] = 0
                result = patch_oracle(base, rows, np.geomspace(0.1, 100, 6), [2, 3, 3], [[0, 1], [2, 3], [4, 5]])
                np.testing.assert_array_equal(result["degrees"], [3, 2, 2])
                np.testing.assert_allclose(result["inverse"], result["inverse"].T, atol=1e-13, rtol=1e-13)
                np.linalg.cholesky(result["inverse"])
                for matrix in (
                    result["schur_bound"] - result["schur"],
                    result["inverse"] - result["exact_inverse"],
                    result["base_inverse"] - result["inverse"],
                ):
                    self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-10)

    def test_dense_q_coordinate_coupling(self):
        """Count the whole dense q block even when patches use different q coordinates."""
        result = patch_oracle([[2, 1], [1, 2]], np.eye(2), [1, 1], [2], [[0], [1]])
        np.testing.assert_array_equal(result["degrees"], [2])
        self.assertGreaterEqual(np.linalg.eigvalsh(result["schur_bound"] - result["schur"]).min(), 0)
        # Treating dense q coordinates as independent drops a nonzero Gram coupling.
        invalid_bound = np.diag(np.diag(result["schur"]))
        self.assertLess(np.linalg.eigvalsh(invalid_bound - result["schur"]).min(), 0)

    def test_zero_weights_and_empty_contacts(self):
        """Keep the base inverse with no active normal rows."""
        base = np.diag([2.0, 3.0, 4.0])
        for rows, weights, patches in ((np.ones((2, 3)), [0, 0], [[0], [1]]), (np.empty((0, 3)), [], [])):
            result = patch_oracle(base, rows, weights, [3], patches)
            np.testing.assert_array_equal(result["inverse"], result["base_inverse"])

    def test_invalid_input(self):
        """Reject illegal weights, pivots, partitions and undeclared base coupling."""
        for base, rows, weights, sizes, patches in (
            (np.eye(2), [[1, 1]], [-1], [2], [[0]]),
            (np.eye(2), [[1, 1]], [np.nan], [2], [[0]]),
            (np.eye(2), [[1, 1]], [1], [2], [[0], [0]]),
            (np.eye(2), [[1, 1]], [1], [2], []),
            (np.eye(2), [[1, 1]], [1], [1], [[0]]),
            ([[1, 0.1], [0.1, 1]], [[1, 1]], [1], [1, 1], [[0]]),
            (np.diag([1, -1]), [[1, 1]], [1], [2], [[0]]),
        ):
            with self.subTest(base=base, weights=weights), self.assertRaises((ValueError, np.linalg.LinAlgError)):
                patch_oracle(base, rows, weights, sizes, patches)


if __name__ == "__main__":
    unittest.main()
