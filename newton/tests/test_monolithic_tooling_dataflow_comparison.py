# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check strict old/new contact comparisons without relaxing checkpoint identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.workload_io import digest, write_json
from scripts.monolithic_reference.workload_scene import build_scene


def test_dataflow_comparison(test, device):
    """Advance independent native trajectories and audit both paths at identical contact states."""
    from scripts.monolithic_reference.compare_contact_dataflows import compare  # noqa: PLC0415

    for scene in ("actor-coupling-before", "actor-coupling-permuted"):
        for graph in (False, True) if device.is_cuda else (False,):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bundle"
                record(root, scene=scene, device=device, graph=graph, steps=2, windows=[(1, 2)], boundaries=())
                result = compare(root, device=device)
                test.assertEqual(result["status"], "PASS_LOCAL_COMPLETE_CONTACT")
                test.assertEqual(result["steps"], 2)
                test.assertEqual(result["same_state_checks"], 8)
                test.assertEqual(result["algorithm_differences"].keys(), {"contact_dataflow", "contact_resources"})
                test.assertNotEqual(*result["config_digests"])
                test.assertTrue(all(s["converged"] for row in result["work"] for s in row["algorithms"]))
                test.assertTrue(all(s["step_seconds"] > 0 for row in result["work"] for s in row["algorithms"]))
                test.assertTrue(all(v["within_tolerance"] for v in result["trajectory_fields"].values()))
                manifest = json.loads((root / "manifest.json").read_text())
                for key, value in (
                    ("source", {}),
                    ("physical_digest", "00" * 32),
                    ("dt", 0.002),
                    ("comparison", {"rtol": 1, "atol": 1}),
                    ("windows", [[2, 2]]),
                    ("contact_dataflow", "pooled"),
                ):
                    damaged = copy.deepcopy(manifest)
                    damaged[key] = value
                    write_json(root / "manifest.json", damaged)
                    with test.assertRaises(ValueError):
                        compare(root, device=device)
                write_json(root / "manifest.json", manifest)
                # An edited payload with a recomputed file hash still must pass physics/trace checks.
                path = root / "state-1.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = dict(archive)
                arrays["history.config_digest"] = np.zeros(32, dtype=np.uint8)
                np.savez_compressed(path, **arrays)
                manifest["files"][path.name] = digest(path)
                write_json(root / "manifest.json", manifest)
                with test.assertRaises(AssertionError):
                    compare(root, device=device)


def test_contract_rejection(test, device):
    """Reject changed law, Q1, layout, sampling or Model input before permitting algorithm differences."""
    from scripts.monolithic_reference.compare_contact_dataflows import verify_contract  # noqa: PLC0415

    _, left, _, _ = build_scene("actor-coupling", device=device, graph=False)
    _, right, _, _ = build_scene("actor-coupling", device=device, graph=False, contact_dataflow="pooled")
    verify_contract(left, right)
    with patch.object(right, "linear_tolerance", left.linear_tolerance * 2):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    with patch.object(right._contact, "contact_model", "C-EE"):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    for key in ("contact_model", "direction_policy", "iteration_policy", "guard_capacity"):
        original = right._coupled_config[key]
        right._coupled_config[key] = original.copy()
        right._coupled_config[key].flat[0] += 1
        with test.assertRaises(ValueError):
            verify_contract(left, right)
        right._coupled_config[key] = original
    key = "actor_contact.weights"
    original = right._contact.actor_samples.fields[key]
    right._contact.actor_samples.fields[key] = original * 2
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    right._contact.actor_samples.fields[key] = original
    original = right.model.body_mass.numpy().copy()
    right.model.body_mass.assign(original * 2)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    right.model.body_mass.assign(original)
    with patch.object(right._contact, "dataflow_identity", np.frombuffer(b"unknown", dtype=np.uint8)):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    # Original checkpoint guards remain strict across the algorithms.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        left.save_checkpoint(left.create_state(dt=0.001), path)
        with test.assertRaises(ValueError):
            right.load_checkpoint(right.create_state(dt=0.001), path)


def test_hotspot_and_failure(test, device):
    """Audit authentic Newton/trial inputs and reject overflow or a corrupted complete oracle."""
    from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_state  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_contact_pool import pooled_solver  # noqa: PLC0415
    from scripts.monolithic_reference.compare_contact_dataflows import _contact_at, _physical, compare  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "bundle"
        record(
            root,
            scene="actor-coupling",
            device=device,
            graph=device.is_cuda,
            steps=1,
            windows=[(1, 1)],
            boundaries=("objective",),
            call_ordinals={"objective": list(range(10))},
        )
        result = compare(root, device=device)
        test.assertGreater(result["same_state_checks"], 4)
        test.assertTrue(any(row.get("state_owner") == "_trial" for row in result["same_state"]))
        from scripts.monolithic_reference.contact_full_oracle import evaluate_stream  # noqa: PLC0415

        def wrong_oracle(*args, **kwargs):
            full = evaluate_stream(*args, **kwargs)
            full["full"]["energy"] += 1
            return full

        with patch("scripts.monolithic_reference.compare_contact_dataflows.evaluate_stream", wrong_oracle):
            with test.assertRaises(AssertionError):
                compare(root, device=device)
    run = pooled_solver(contact_scene(device, "chain"), contact_factor_capacity=0)
    state = _physical(contact_state(run, "chain"))
    with test.assertWarns(RuntimeWarning), test.assertRaisesRegex(ValueError, "overflow"):
        _contact_at(run, state, state, 0.001)


class TestContactDataflowComparison(unittest.TestCase):
    pass


for _test in (test_dataflow_comparison, test_contract_rejection, test_hotspot_and_failure):
    add_function_test(TestContactDataflowComparison, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
