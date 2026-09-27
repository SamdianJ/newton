# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the Close diagnostic's clock, controller and failure accounting."""

import unittest
from unittest.mock import Mock

import numpy as np

from scripts.monolithic_reference.verify_p3_close import complete_publication, complete_trace, targets


class TestCloseDiagnostics(unittest.TestCase):
    def test_targets(self):
        """Preserve endpoint sampling, independent offsets and the carriage target."""
        curve = Mock(names=["finger"])
        curve.sample.return_value = (np.array([0.123456789]), np.array([0.456789123]))
        mapping = {"finger": (0, 1, 2), "monolithic_lift": (2, 0, 1)}
        stage, q, v = targets(curve, mapping, 0.501, 3)
        self.assertEqual(stage, "close")
        self.assertAlmostEqual(curve.sample.call_args.args[0], 0.001)
        self.assertEqual(q.dtype, np.float32)
        self.assertEqual(q[2], np.float32(0.123456789))
        self.assertEqual(v[1], np.float32(0.456789123))
        self.assertEqual(q[1], 0)
        self.assertEqual(v[0], 0)
        stage, _, v = targets(curve, mapping, 0.5, 3)
        self.assertEqual(stage, "prepare")
        np.testing.assert_array_equal(v, 0)

    def test_complete_trace(self):
        """Reject a partial or failed trajectory and any shifted clock."""
        rows = [{"success": True, "step": i + 1, "time": (i + 1) * 0.001} for i in range(3)]
        self.assertTrue(complete_trace(rows, 3))
        self.assertFalse(complete_trace(rows, 2500))
        rows[-1]["success"] = False
        self.assertFalse(complete_trace(rows, 3))
        rows[-1]["success"] = True
        rows[-1]["time"] += 0.001
        self.assertFalse(complete_trace(rows, 3))

    def test_complete_publication(self):
        """Allow labeled Stopped publication without treating it as convergence."""
        rows = [
            {"success": True, "published": True, "publication": "CONVERGED", "step": 1, "time": 0.001},
            {"success": False, "published": True, "publication": "STOPPED_ACCEPTED", "step": 2, "time": 0.002},
        ]
        self.assertTrue(complete_publication(rows, 2))
        self.assertFalse(complete_trace(rows, 2))
        self.assertFalse(complete_publication(rows, 3))
        rows[-1]["publication"] = "HARD_FAILURE"
        self.assertFalse(complete_publication(rows, 2))
        rows[-1]["publication"] = "STOPPED_ACCEPTED"
        rows[-1]["published"] = False
        self.assertFalse(complete_publication(rows, 2))
        rows[-1]["published"] = True
        rows[-1]["time"] += 0.001
        self.assertFalse(complete_publication(rows, 2))


if __name__ == "__main__":
    unittest.main()
