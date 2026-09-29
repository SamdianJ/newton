# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen precision matrix and host decision arithmetic contracts."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton._src.solvers.monolithic.solver_types import _SolverMonolithicInternalConfig
from newton._src.solvers.monolithic.variational_iteration import measure
from scripts.monolithic_reference.run_p5_precision import cases


class TestPrecisionRunner(unittest.TestCase):
    def test_matrix(self):
        rows = cases()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({r["id"] for r in rows}), 16)
        for precision in ("fp32", "fp64"):
            selected = [r for r in rows if r["precision"] == precision]
            self.assertEqual(sum(r["steps"] for r in selected), 28917)
            self.assertEqual(sum(r["scene"].startswith("close-") for r in selected), 6)
            self.assertTrue(all(r["graph"] and r["pcg_mode"] == "production" for r in selected))
            self.assertTrue(all(r["publish_policy"] == "current_iterate" for r in selected))

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
