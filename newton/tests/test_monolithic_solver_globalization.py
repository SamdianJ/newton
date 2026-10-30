# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the fixed ArmijoOrResNorm acceptance and native convergence goals."""

import unittest
from itertools import product

import numpy as np

from newton._src.solvers.monolithic import convergence
from newton._src.solvers.monolithic.convergence import ConvergencePolicy
from newton._src.solvers.monolithic.iteration import acceptance


class TestGlobalization(unittest.TestCase):
    """Guard the fixed ArmijoOrResNorm acceptance predicate."""

    def test_predicates(self):
        """Exercise the A/R/OR truth table, nonnegative slopes, equality, noise and nonfinite rejection."""
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        for a, r in product((False, True), repeat=2):
            value = acceptance(1, 0.5 if a else 2, -1, 1, 1, 0.5 if r else 2, **kw)
            self.assertEqual(value[:3], (a or r, a, r))
        # Non-descent directions are still accepted by a strict residual decrease.
        self.assertEqual(acceptance(1, 2, 1, 1, 1, 0.5, **kw), (True, False, True, True))
        # Equal residuals are accepted but are not an improvement.
        self.assertEqual(acceptance(1, 1, 0, 1, 1, 1, **kw), (True, False, True, False))
        # An energy decrease within its noise is not an improvement either.
        self.assertEqual(
            acceptance(1, 1, -1, 1, 1, 1, energy_noise=0.001, residual_noise=0, slope_noise=0),
            (True, True, True, False),
        )
        with self.assertRaises(ValueError):
            acceptance(1, float("nan"), -1, 1, 1, 0, **kw)

    def test_native_host_targets(self):
        """Compute row tolerances and the line-search merit in the selected precision."""
        rows = convergence.RowGroups(("x",), np.zeros(5, np.int32), np.zeros(5, np.int32))
        for dtype in (np.float32, np.float64):
            residual = np.array([0.1, 0.02, -0.5, 3.0, 0.9], dtype=dtype)
            tau, physical = convergence.tolerances(
                residual * dtype(7), np.abs(residual), rows, ConvergencePolicy(), dtype
            )
            self.assertEqual((tau.dtype, physical.dtype), (np.dtype(dtype), np.dtype(dtype)))
            self.assertEqual(type(convergence.merit(residual, tau)), dtype)

    def test_reject_alternating_regressions(self):
        """Reject a cycle between metrics while permitting residual progress with rising energy."""
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        self.assertTrue(acceptance(1, 0.9, -1, 0.1, 1, 1.1, **kw)[0])
        bounds = {"best_energy": 0.9, "best_merit": 1.0}
        self.assertFalse(acceptance(0.9, 1.05, -1, 0.1, 1.1, 1.02, **bounds, **kw)[0])
        self.assertTrue(acceptance(0.9, 1.05, -1, 0.1, 1.1, 0.8, **bounds, **kw)[0])

    def test_merit_resolves_each_unit(self):
        """Accept a torque improvement that outweighs sub-tolerance position noise, and not the reverse."""
        tau = np.array([1e-10, 1e-12])
        current = convergence.merit(np.array([2e-10, 4e-12]), tau)
        trial = convergence.merit(np.array([3e-10, 1e-12]), tau)
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        self.assertTrue(acceptance(1, 1, 0, 1, current, trial, **kw)[0])
        self.assertFalse(acceptance(1, 1, 0, 1, trial, current, **kw)[0])


if __name__ == "__main__":
    unittest.main()
