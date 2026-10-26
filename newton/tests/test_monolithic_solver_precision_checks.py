# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold the FP32 solver to both sides of the precision check set.

Each required case must cover its real rounding noise (no stall) and still
resolve its small load (a Newton update and the FP64 or physical response).
Cases join this list when a change makes them pass; none is ever removed to
obtain a pass. The full set, including cases not yet required, is run by
``scripts.monolithic_reference.run_precision_checks``.
"""

import unittest

import warp as wp

from scripts.monolithic_reference.run_precision_checks import iter_cases

# Short CPU cases; the n=128 cantilevers and stiff-bulk tet run only in the full set.
REQUIRED = (
    "cable_cantilever/0deg/n16/ramp_first",
    "cable_cantilever/0deg/n16/full",
    # PR-02: split-form curvature and its operand bound (failed at 3b3c7417).
    "cable_cantilever/30deg/n16/ramp_first",
    "cable_cantilever/45deg/n16/ramp_first",
    "cable_twist/0deg/n16",
    "cable_twist/45deg/n16",
    "shell_patch/0deg/small",
    "shell_patch/30deg/small",
    "moving_tet/g0.05/vx0",
    "moving_tet/g9.81/vx0",
    "moving_tet/g9.81/vx1",
    "tet_noise/nu0.3/stretch1",
    "tet_noise/nu0.3/stretch1.2",
    "tet_noise/nu0.49/stretch1",
    "tet_noise/nu0.499/stretch1",
)


class TestPrecisionChecks(unittest.TestCase):
    device = "cpu"

    def test_required_cases_pass_both_sides(self):
        cases = {name: run for name, _, run in iter_cases(device=self.device)}
        self.assertLessEqual(set(REQUIRED), set(cases))
        for name in REQUIRED:
            with self.subTest(case=name):
                judgement = cases[name]()["judgement"]
                self.assertTrue(judgement["PASS"], judgement)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestPrecisionChecksCUDA(TestPrecisionChecks):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
