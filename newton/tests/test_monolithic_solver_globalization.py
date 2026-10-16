# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check globalization acceptance and native convergence goals."""

import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np

from newton._src.solvers.monolithic.iteration import acceptance, material_merit, measure
from newton._src.solvers.monolithic.linear_types import MonolithicLinearLayout
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
                _linear=SimpleNamespace(layout=MonolithicLinearLayout(2, 1)),
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

    def test_material_merit_resolves_each_physical_goal(self):
        """Resolve torque improvements despite sub-goal position noise, retaining gauge equations."""
        solver = SimpleNamespace(
            _numpy_dtype=np.float64,
            _linear=SimpleNamespace(
                layout=SimpleNamespace(
                    groups=(
                        ("global", 0, 2),
                        ("cable_0_x", 0, 1),
                        ("cable_0_theta", 1, 1),
                    )
                )
            ),
            _cable_raw_theta_norms={},
        )
        values = {"global": (0, 0, 1e-7, 2), "cable_0_x": (0, 0, 1e-9, 1), "cable_0_theta": (0, 0, 1e-12, 1)}
        scale = np.array([1.0, 10.0])
        current = material_merit(solver, np.array([2e-10, 4e-13]), values, scale)
        trial = material_merit(solver, np.array([4e-10, 2e-13]), values, scale)
        kw = {"energy_noise": 0, "residual_noise": 0, "slope_noise": 0}
        with self.subTest(case="torque_improvement"):
            self.assertTrue(acceptance("ResNorm", 1, 1, 0, 1, current, trial, **kw)[0])
            self.assertFalse(acceptance("ResNorm", 1, 1, 0, 1, trial, current, **kw)[0])
        solver._cable_raw_theta_norms["cable_0_theta"] = 4e-13
        current = material_merit(solver, np.array([4e-10, 0.0]), values, scale)
        solver._cable_raw_theta_norms["cable_0_theta"] = 8e-13
        trial = material_merit(solver, np.array([2e-10, 0.0]), values, scale)
        with self.subTest(case="omitted_gauge_equation"):
            self.assertFalse(acceptance("ResNorm", 1, 1, 0, 1, current, trial, **kw)[0])


if __name__ == "__main__":
    unittest.main()
