# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check globalization acceptance and native convergence goals."""

import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np

from newton._src.solvers.monolithic.iteration import acceptance, measure
from newton._src.solvers.monolithic.solver_types import _SolverMonolithicInternalConfig


class TestGlobalization(unittest.TestCase):
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

    def test_native_host_targets(self):
        for dtype in (np.float32, np.float64):
            solver = SimpleNamespace(
                _numpy_dtype=dtype,
                _metrics={},
                _config=_SolverMonolithicInternalConfig(),
                _linear=SimpleNamespace(layout=SimpleNamespace(scalar_dof_count=5, q_dof_count=2)),
            )
            residual = np.array([0.1, 0.02, -0.5, 3.0, 0.9], dtype=dtype)
            reference = residual * dtype(7)
            scale = np.linspace(0.3, 2, 5, dtype=dtype)
            values = measure(solver, residual, reference, scale)
            for merit, ref, goal, _ in values.values():
                self.assertEqual(type(merit), dtype)
                self.assertEqual(type(ref), dtype)
                self.assertEqual(type(goal), dtype)
            np.testing.assert_equal(values["global"][0], np.linalg.norm(scale * residual) / np.sqrt(dtype(5)))


if __name__ == "__main__":
    unittest.main()
