# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise the temporary 8C2-0 rejection boundary before a new backend exists."""

import unittest
from unittest.mock import patch

from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton.solvers.experimental.monolithic import SolverMonolithic
from scripts.monolithic_reference import run_p2_8bc


class TestProductionCleanup(unittest.TestCase):
    def test_reject_all_entries(self):
        """Reject production before allocating or invoking a historical solver."""
        with self.assertRaisesRegex(ValueError, "unavailable"):
            SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, pcg_mode="production")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            SolverMonolithic._create_joint_diagnostic(
                None, joint_terms=SolverMonolithic.JointTerms(), pcg_mode="production"
            )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            MonolithicLinearWorkspace(None, None, None, device="cpu", pcg_mode="production")
        with patch.object(run_p2_8bc.run_p2_8a, "main") as run:
            with self.assertRaises(SystemExit):
                run_p2_8bc.main(["--kind", "tet", "--pcg-mode", "production", "--output", "/tmp/not-run-p28c2"])
            run.assert_not_called()
