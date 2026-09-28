# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared acceptance independently of retired tet-only solvers."""

import unittest
from itertools import product

from newton._src.solvers.monolithic.variational_iteration import acceptance


class TestP3Tet(unittest.TestCase):
    """Guard the shared Armijo/ResNorm acceptance predicate."""

    def test_predicates(self):
        """Exercise A/R truth tables, nonnegative slopes, equality and nonfinite rejection."""
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        for mode in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            for a, r in product((False, True), repeat=2):
                value = acceptance(mode, 1, 0.5 if a else 2, -1, 1, 1, 0.5 if r else 2, **kw)
                self.assertEqual(value[:3], (a if mode == "Armijo" else r if mode == "ResNorm" else a or r, a, r))
        self.assertEqual(acceptance("ArmijoOrResNorm", 1, 2, 1, 1, 1, 0.5, **kw), (True, False, True, True))
        self.assertEqual(acceptance("ResNorm", 1, 1, 0, 1, 1, 1, **kw), (True, False, True, False))
        self.assertFalse(acceptance("ResNorm", 1, 0.5, -1, 1, 1, 1, **kw)[3])
        self.assertFalse(
            acceptance("Armijo", 1, 1, -1, 1, 1, 0.5, energy_noise=0.001, residual_noise=0, slope_noise=0)[3]
        )
        with self.assertRaises(ValueError):
            acceptance("ArmijoOrResNorm", 1, float("nan"), -1, 1, 1, 0, **kw)


if __name__ == "__main__":
    unittest.main()
