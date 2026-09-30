# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check lazy imports, shared type identities and immutable solver statistics."""

import subprocess
import sys
import unittest


class TestMonolithicTypes(unittest.TestCase):
    def test_definitions_do_not_load_solver(self):
        """Import neutral definitions without loading numerical implementation modules."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic.solver_types import Status, Stats, _StepFailure; "
                "assert Stats(Status.NOT_RUN).status is Status.NOT_RUN; "
                "assert _StepFailure(Status.NONFINITE, 'probe').status is Status.NONFINITE; "
                "assert not any('newton._src.solvers.monolithic.' + name in sys.modules "
                "for name in ('solver_monolithic', 'articulation', "
                "'tet_assembly', 'contact', 'collision'))",
            ],
            check=True,
        )

    def test_collision_import_is_independent(self):
        """Resolve the public collision pipeline without importing the coupled solver."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline; "
                "import newton.solvers.experimental.monolithic as module; "
                "assert module.MonolithicCollisionPipeline is MonolithicCollisionPipeline; "
                "assert 'SolverMonolithic' in dir(module); "
                "assert 'newton._src.solvers.monolithic.solver_monolithic' not in sys.modules; "
                "assert not hasattr(module, 'SolverNonexistent')",
            ],
            check=True,
        )

    def test_shared_identity_and_immutable_stats(self):
        """Keep public type access and a single exception identity across current solver components."""
        from dataclasses import FrozenInstanceError, replace  # noqa: PLC0415

        from newton._src.solvers.monolithic import (  # noqa: PLC0415
            iteration,
            solver_monolithic,
            solver_types,
        )
        from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

        for name in ("Status", "Stats", "JointTerms", "PCGMode", "MassMode", "MaterialModel"):
            self.assertIs(getattr(SolverMonolithic, name), getattr(solver_types, name))
        for module in (solver_monolithic, iteration):
            self.assertIs(module._StepFailure, solver_types._StepFailure)
        timings = {"solve": 1.0}
        stats = SolverMonolithic.Stats(SolverMonolithic.Status.NOT_RUN, timings=timings)
        timings["solve"] = 2.0
        self.assertEqual(stats.timings["solve"], 1.0)
        with self.assertRaises(TypeError):
            stats.timings["solve"] = 3.0
        with self.assertRaises(FrozenInstanceError):
            stats.converged = True
        self.assertTrue(replace(stats, converged=True).converged)
        self.assertFalse(stats.converged)


if __name__ == "__main__":
    unittest.main()
