# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit selected spatial states independently of query and capacity decisions."""

import unittest
import warnings

from newton.tests.test_monolithic_contact_actor_pairs import physical
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_state
from newton.tests.test_monolithic_solver_contact_pool import pooled_solver
from newton.tests.test_monolithic_solver_contact_spatial import spatial_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.compare_contact_dataflows import audit_spatial_contact_state, verify_contract


def test_spatial_state_audit(test, device):
    """Check selected native E/r/raw B and both forces against independent full enumeration."""
    for law in ("C-EE", "C-IE", "C-II"):
        options = {"law": law, "mode": "production", "graph": device.is_cuda}
        full = pooled_solver(contact_scene(device, "chain"), **options)
        run = spatial_solver(contact_scene(device, "chain"), **options)
        contract = verify_contract(full, run)
        test.assertEqual(contract["comparison"], "pooled/spatial")
        state = contact_state(run, "chain")
        start = physical(state)
        frame = {k: v.copy() for k, v in start.items()}
        frame["com"][:, 1] += 0.0002
        result = audit_spatial_contact_state(run, frame, start, 0.001, frozen=start)
        test.assertEqual(result["status"], "MEASURED_NOT_QUALIFIED")
        test.assertFalse(result["tainted"])
        test.assertEqual(result["full_count"], run._contact.count)
        test.assertLess(result["native_selected_errors"]["tangent_relative"], 1e-5)
        test.assertEqual(result["capacity_omission"]["contributing_count"], 0)
        run.linear_tolerance *= 2
        with test.assertRaises(ValueError):
            verify_contract(full, run)
    full = pooled_solver(contact_scene(device, "chain"))
    run = spatial_solver(contact_scene(device, "chain"), contact_factor_capacity=3072)
    test.assertEqual(verify_contract(full, run)["comparison"], "pooled/spatial")
    run._coupled_config["guard_capacity"][0] *= 2
    with test.assertRaises(ValueError):
        verify_contract(full, run)


def test_spatial_known_miss_audit(test, device):
    """Detect nonzero contacts outside a frozen query without refreshing at the trial state."""
    run = spatial_solver(contact_scene(device, "A-R"), law="C-II")
    touching = physical(contact_state(run, "A-R"))
    far = {k: v.copy() for k, v in touching.items()}
    far["com"][:] = 10
    result = audit_spatial_contact_state(run, touching, far, 0.001, frozen=far)
    test.assertFalse(result["tainted"])
    test.assertEqual(result["candidate_count"], 0)
    test.assertGreater(result["spatial_omission"]["contributing_count"], 0)
    test.assertGreater(result["spatial_omission"]["errors"]["residual"]["l2"], 0)
    test.assertEqual(run._contact._refreshes, 1)


def test_spatial_capacity_audit(test, device):
    """Keep query overflow, pool overflow and saved zero work distinct from spatial qualification."""
    for options, query_drop in (({"capacity": 0}, True), ({"contact_factor_capacity": 0}, False)):
        run = spatial_solver(contact_scene(device, "chain"), **options)
        start = physical(contact_state(run, "chain"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = audit_spatial_contact_state(run, start, start, 0.001)
        test.assertTrue(result["tainted"])
        test.assertEqual(result["spatial_omission"] is None, query_drop)
        test.assertGreater(result["full_contributing_count"], 0)
        if not query_drop:
            test.assertGreater(result["capacity_omission"]["contributing_count"], 0)
            test.assertEqual(result["pool"]["admitted"], 0)
        with test.assertRaises(ValueError):
            verify_contract(run, run)


class TestContactSpatialAudit(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _name in ("test_spatial_state_audit", "test_spatial_known_miss_audit", "test_spatial_capacity_audit"):
        add_function_test(TestContactSpatialAudit, _name, globals()[_name], devices=[_device])


if __name__ == "__main__":
    unittest.main()
