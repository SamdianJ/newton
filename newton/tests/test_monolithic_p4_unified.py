# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard the complete q+x solver's allocation and dependency boundaries."""

import subprocess
import sys
import unittest
from unittest.mock import patch

from newton._src.solvers.monolithic import solver_monolithic
from newton.tests.test_monolithic_p3_coupled import scene, solver


class TestUnifiedSolver(unittest.TestCase):
    def test_import_boundary(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic; "
                "from newton._src.solvers.solver import SolverBase; "
                "assert SolverMonolithic.__bases__ == (SolverBase,); "
                "assert not any('newton._src.solvers.monolithic.' + name in sys.modules "
                "for name in ('solver_monolithic_legacy', 'coupled_solver', 'articulation_solver', 'articulation', 'tet', "
                "'tet_variational', 'contact'))",
            ],
            check=True,
        )

    def test_single_linear_workspace(self):
        model = scene("cpu")
        with patch.object(
            solver_monolithic, "MonolithicLinearWorkspace", wraps=solver_monolithic.MonolithicLinearWorkspace
        ) as workspace:
            run = solver(model)
        self.assertEqual(workspace.call_count, 1)
        self.assertEqual(run._delta.size, model.joint_dof_count + 3 * run._tet.ids.size)


if __name__ == "__main__":
    unittest.main()
