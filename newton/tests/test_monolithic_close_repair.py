# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Residual-compatible direction checks for the Close research repair."""

import ast
import inspect
import textwrap
import unittest

import numpy as np

from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic
from scripts.monolithic_reference.run_close_repair import least_squares_direction, repair_source, residual_goal_weights


class TestCloseRepair(unittest.TestCase):
    def test_goal_weights_and_empty_blocks(self):
        """Match the sum of squared block acceptance ratios including q-only systems."""
        for nq in (0, 2, 5):
            residual = np.arange(1, 6, dtype=float)
            targets = np.array([2.0, 0.1, 3.0])
            weights = residual_goal_weights(5, nq, targets)
            raw = np.full(5, 1 / (5 * targets[0] ** 2))
            expected = np.sum(residual**2) / (5 * targets[0] ** 2)
            for block, goal in ((slice(0, nq), targets[1]), (slice(nq, 5), targets[2])):
                if residual[block].size:
                    raw[block] += 1 / (residual[block].size * goal**2)
                    expected += np.mean(residual[block] ** 2) / goal**2
            self.assertAlmostEqual(np.sum((weights * residual) ** 2) * np.max(raw), expected)

    def test_preserve_original_rules(self):
        """Keep the original nonlinear loop intact outside the research dispatch."""
        original = ast.parse(textwrap.dedent(inspect.getsource(SolverMonolithic._iterate)))
        modified = ast.parse(repair_source(SolverMonolithic._iterate))

        class RemoveResearch(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == "research_active" for t in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                if (
                    isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.operand, ast.Name)
                    and node.test.operand.id == "research_active"
                ):
                    return node.body
                if any(isinstance(item, ast.Name) and item.id == "research_active" for item in ast.walk(node.test)):
                    return None
                return self.generic_visit(node)

        self.assertEqual(ast.dump(original), ast.dump(RemoveResearch().visit(modified)))

    def test_repair_gn_ascent(self):
        """Recover residual descent when the positive normal approximation ascends."""
        jacobian = np.diag([11.0, -9.0])
        residual = np.array([0.0, 1.0])
        gn_direction = -np.linalg.solve(np.diag([11.0, 1.0]), residual)
        self.assertGreater(residual @ jacobian @ gn_direction, 0)
        direction, _ = least_squares_direction(jacobian, residual)
        self.assertLess(residual @ jacobian @ direction, 0)
        np.testing.assert_allclose(residual + jacobian @ direction, 0, atol=1e-14)

    def test_nonsymmetric_rank_deficient(self):
        """Reduce the reachable residual without requiring an SPD Jacobian."""
        jacobian = np.array([[1.0, 2.0], [0.0, 0.0]])
        residual = np.array([3.0, 1.0])
        direction, rank = least_squares_direction(jacobian, residual)
        self.assertEqual(rank, 1)
        np.testing.assert_allclose(residual + jacobian @ direction, [0, 1], atol=1e-14)

    def test_damping_limits_correction(self):
        """Keep residual descent while limiting a nearly singular correction."""
        jacobian = np.diag([1.0, 0.001])
        residual = np.ones(2)
        undamped, _ = least_squares_direction(jacobian, residual)
        damped, _ = least_squares_direction(jacobian, residual, 0.1)
        self.assertLess(np.linalg.norm(damped), np.linalg.norm(undamped))
        self.assertLess(residual @ jacobian @ damped, 0)
        self.assertLess(np.linalg.norm(residual + jacobian @ damped), np.linalg.norm(residual))
        for damping in (-1, np.nan):
            with self.assertRaises(ValueError):
                least_squares_direction(jacobian, residual, damping)

    def test_reject_invalid_or_stationary(self):
        """Reject nonfinite inputs and directions without predicted descent."""
        for jacobian, residual in (
            (np.array([[np.nan]]), np.ones(1)),
            (np.eye(1), np.array([np.inf])),
            (np.zeros((1, 1)), np.ones(1)),
        ):
            with self.subTest(jacobian=jacobian, residual=residual), self.assertRaises(ValueError):
                least_squares_direction(jacobian, residual)


if __name__ == "__main__":
    unittest.main()
