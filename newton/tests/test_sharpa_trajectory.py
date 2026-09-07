# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the Sharpa CSV target contract without external assets."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.softbody.sharpa_close import JOINT_MAPPING, ClosureTrajectory


class TestSharpaTrajectory(unittest.TestCase):
    def test_csv_contract(self):
        """Validate timestamp interpolation, name ordering, mapping and malformed input."""
        mapping = list(JOINT_MAPPING)
        names = [r[1] for r in mapping][::-1]
        header = ["Time(sec)"] + [
            f"Curve{i}:SubSystem_Left_Hand_Flex/FN_{r[0]}_STEP/Expression/Value(Dimensionless)"
            for i, r in enumerate(mapping)
        ]
        data = [[i * 0.02] + [-min(i * 0.02, 2) * 0.1] * 22 for i in range(201)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"

            def write(h=header, d=data):
                with path.open("w", encoding="utf-8-sig", newline="") as stream:
                    csv.writer(stream).writerows([h, *d])

            def load(**kwargs):
                return ClosureTrajectory(path, names, np.full(22, -2), np.full(22, 2), np.full(22, 5), **kwargs)

            write()
            trajectory = load()
            np.testing.assert_allclose(trajectory.sample(0.013)[0][0], 0.0013)
            np.testing.assert_allclose(trajectory.sample(0.013)[1][0], 0.1)
            np.testing.assert_array_equal(trajectory.sample(0)[1], np.zeros(22))
            np.testing.assert_array_equal(trajectory.sample(5)[1], np.zeros(22))
            changed = [(a, b, -s, 0.03) for a, b, s, _ in mapping]
            shifted = load(mapping=changed)
            np.testing.assert_allclose(shifted.sample(1)[0], -trajectory.sample(1)[0] + 0.03)
            for bad_header in (header[:-1], [*header[:-1], header[1]]):
                write(bad_header)
                with self.assertRaises(ValueError):
                    load()
            for value in (0, float("nan"), -1):
                bad = [r.copy() for r in data]
                bad[1][0] = value
                write(d=bad)
                with self.assertRaises(ValueError):
                    load()
            write(d=data[:-1])
            with self.assertRaises(ValueError):
                load()
            write()
            with self.assertRaisesRegex(ValueError, "joint limits"):
                ClosureTrajectory(path, names, np.zeros(22), np.ones(22), np.ones(22))
            with self.assertRaisesRegex(ValueError, "velocity"):
                ClosureTrajectory(path, names, -np.ones(22), np.ones(22), np.full(22, 0.001))


if __name__ == "__main__":
    unittest.main()
