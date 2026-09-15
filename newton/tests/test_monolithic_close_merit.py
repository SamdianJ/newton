# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The counterfactual source must preserve every original nonlinear rule."""

import ast
import inspect
import textwrap
import unittest

from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic
from scripts.monolithic_reference.run_close_merit import research_source


class TestCloseMerit(unittest.TestCase):
    def test_only_one_bypass_block_added(self):
        """Preserve the original iteration rules outside the single research bypass."""
        original = ast.parse(textwrap.dedent(inspect.getsource(SolverMonolithic._iterate)))
        modified = ast.parse(research_source(SolverMonolithic._iterate))
        bypasses = []

        class RemoveResearch(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id == "research_used" for target in node.targets):
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                if any(isinstance(item, ast.Name) and item.id == "research_used" for item in ast.walk(node.test)):
                    bypasses.append(node)
                    return None
                return self.generic_visit(node)

        cleaned = RemoveResearch().visit(modified)
        self.assertEqual(len(bypasses), 1)
        self.assertEqual(ast.dump(original), ast.dump(cleaned))
        compile(research_source(SolverMonolithic._iterate), "research", "exec")


if __name__ == "__main__":
    unittest.main()
