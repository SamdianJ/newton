# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard timing summaries against incomplete or capture-contaminated claims."""

import unittest

from scripts.monolithic_reference.run_p5_performance import summarize


class TestPerformanceSummary(unittest.TestCase):
    def test_completed_stopped_is_not_qualified(self):
        """Keep completed Stopped runs distinct from qualified acceleration."""
        result = summarize([1.0, 2.0, 3.0], [False, True, False], {"STOPPED": 3}, 3)
        self.assertEqual(result["status"], "COMPLETE_NOT_AUDITED")
        self.assertEqual(result["solve_seconds"], 6.0)
        self.assertEqual(result["capture_free_seconds"], 4.0)
        self.assertEqual(result["capture_free_steps"], 2)
        self.assertEqual(result["qualified_speedup"], None)

    def test_partial_and_failure(self):
        """Reject incomplete advancement as a full performance result."""
        self.assertEqual(summarize([1.0], [False], {"STOPPED": 1}, 3)["status"], "PARTIAL")
        self.assertEqual(summarize([1.0], [False], {"HARD_FAILURE": 1}, 3)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
