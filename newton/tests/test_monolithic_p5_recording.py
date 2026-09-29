# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit legacy FP64 recording migration without relaxing physical comparison."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.check_p3_recording import compare


class TestPrecisionRecording(unittest.TestCase):
    def test_legacy_schema(self):
        """Accept only validated FP64 metadata migration and compare every physical field."""
        before = {
            "frame_steps": np.arange(2),
            "joint_q": np.zeros((2, 1), np.float64),
            "history.schema_version": np.full((2, 1), 2, np.int32),
        }
        after = {
            **before,
            "history.schema_version": np.full((2, 1), 3, np.int32),
            "history.precision": np.full((2, 1), 64, np.int32),
        }
        with patch(
            "scripts.monolithic_reference.check_p3_recording.frames",
            side_effect=lambda p: iter([before if p == "old" else after]),
        ):
            with self.assertRaises(ValueError):
                compare("old", "new")
            result = compare("old", "new", legacy_fp64=True)
            self.assertEqual(result["status"], "PASS_EXACT")
            self.assertEqual(result["schema_migration"], "2 -> 3 (FP64)")
            after["joint_q"] = np.ones((2, 1), np.float64)
            self.assertEqual(compare("old", "new", legacy_fp64=True)["status"], "NOT_READY_TOLERANCE")
            for field, value in [("history.precision", 32), ("history.schema_version", 4)]:
                saved = after[field].copy()
                after[field].fill(value)
                with self.assertRaises(ValueError):
                    compare("old", "new", legacy_fp64=True)
                after[field] = saved
            after["unexpected"] = np.zeros((2, 1))
            with self.assertRaises(ValueError):
                compare("old", "new", legacy_fp64=True)


if __name__ == "__main__":
    unittest.main()
