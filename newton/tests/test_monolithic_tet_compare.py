# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G2H fixture authority and anti-false-positive checks."""

import copy
import unittest

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import FIXTURE, Case, load_fraction
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_fixture(test, device):
    """Verify physical mass, distributed end traction, fixed nodes and incomplete-run failure."""
    case = Case(device, material="smith_log_stabilized", mass="consistent", refinement=1)
    test.assertAlmostEqual(case.weights.sum(), 1)
    np.testing.assert_allclose(case.force.sum(axis=0), [0, 0, FIXTURE["transverse_force"]])
    total = FIXTURE["length"] * FIXTURE["width"] * FIXTURE["height"] * FIXTURE["density"]
    test.assertAlmostEqual(case.mass[::3, ::3].sum(), total, delta=1e-5 * total)
    dynamic = case.dynamic
    test.assertGreater(np.linalg.eigvalsh(case.mass[np.ix_(dynamic, dynamic)]).min(), 0)
    test.assertGreater(np.linalg.eigvalsh(case.linear_k[np.ix_(dynamic, dynamic)]).min(), 0)
    case.step()
    test.assertFalse(case.summary()["passed"])
    # A full-length trace with convergence but no response must still fail G2H.
    entry = copy.deepcopy(case.records[0])
    entry["tip"] = [0, 0, 0]
    entry["linear_tip"] = [0, 0, 0.01]
    case.records = [entry] * round(FIXTURE["duration"] / case.dt)
    case.steps = len(case.records)
    test.assertFalse(case.summary()["passed"])


class TestTetCompare(unittest.TestCase):
    def test_load_schedule(self):
        """Keep the hold and free-response phases explicit and loading C1 at transitions."""
        for t, value in ((0, 0), (1, 1), (2, 1), (3, 0), (4, 0)):
            self.assertEqual(load_fraction(t), value)
        for t in (1, 2, 3):
            self.assertLess(abs(load_fraction(t + 1e-5) - load_fraction(t - 1e-5)), 1e-8)


add_function_test(TestTetCompare, "test_fixture", test_fixture, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
