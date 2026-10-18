# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the producer step contract shared by every monolithic physical term."""

import unittest

import numpy as np

from newton._src.solvers.monolithic.producers import Producer, validate_row_partition


class _Rows:
    def __init__(self, name, rows):
        self.name, self.rows = name, np.asarray(rows, dtype=np.int32)


class TestProducerRows(unittest.TestCase):
    def test_rows_partition_the_layout(self):
        validate_row_partition([_Rows("a", [0, 1]), _Rows("b", [2])], 3)

    def test_overlap_and_gap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_row_partition([_Rows("a", [0, 1]), _Rows("b", [1, 2])], 3)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_row_partition([_Rows("a", [0, 0, 1])], 2)
        with self.assertRaisesRegex(ValueError, "uncovered"):
            validate_row_partition([_Rows("a", [0])], 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_row_partition([_Rows("a", [0, 2])], 2)

    def test_protocol_lists_the_step_contract(self):
        for name in ("begin_step", "candidate", "evaluate", "assemble", "scale", "displacement", "domain", "publish"):
            self.assertTrue(hasattr(Producer, name), name)


if __name__ == "__main__":
    unittest.main()
