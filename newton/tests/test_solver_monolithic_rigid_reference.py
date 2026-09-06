# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent rigid dynamics and isolated force sign acceptance."""

import copy
import unittest

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.check_rigid_reference import assess, measure


def test_rigid_reference(test, device):
    """Compare actual rigid-only steps and reject wrong signs or loss of convergence."""
    result = measure(str(device))
    test.assertTrue(assess(result), result)
    for term in range(3):
        bad = copy.deepcopy(result)
        bad["isolated_forces"][term]["residual_contribution"][0] *= -1
        test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    bad["isolated_forces"] = [copy.deepcopy(result["isolated_forces"][0])] * 3
    test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    for row in bad["isolated_forces"]:
        for key in ("residual_contribution", "generalized_physical_force", "independent_generalized_force"):
            row[key] = [0.0]
    test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    bad["isolated_forces"][2]["physical_world_force_or_wrench"][0][1] *= -1
    test.assertFalse(assess(bad))
    for solver in ("monolithic", "featherstone"):
        for key in ("body_q", "body_qd"):
            bad = copy.deepcopy(result)
            bad["steps"][0][f"{key}_{solver}"][0][0] = 1000.0
            test.assertFalse(assess(bad))
    for key in ("residual_contribution", "generalized_physical_force", "independent_generalized_force"):
        for value in ([float("nan")], [], [[1.0]]):
            bad = copy.deepcopy(result)
            bad["isolated_forces"][0][key] = value
            test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    bad["steps"][-1]["converged"] = False
    test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    bad["steps"][-1]["joint_q_monolithic"][0] += 0.1
    test.assertFalse(assess(bad))
    bad = copy.deepcopy(result)
    bad["steps"][-1]["joint_qd_monolithic"][0] = float("nan")
    test.assertFalse(assess(bad))


class TestMonolithicRigidReference(unittest.TestCase):
    """Run the release rigid reference gates on each supported device."""


add_function_test(
    TestMonolithicRigidReference, "test_rigid_reference", test_rigid_reference, devices=get_test_devices()
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
