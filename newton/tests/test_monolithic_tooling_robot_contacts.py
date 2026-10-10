# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authenticate frozen robot policies and record pooled contact diagnostics."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.articulation import _configuration_digest
from newton.tests import test_monolithic_tooling_recording_compatibility as compatibility
from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.check_recording import compare
from scripts.monolithic_reference.robot_contact_acceptance import (
    FrozenRobotFactory,
    certify_configuration,
    check_robot_reports,
)


def test_robot_policy_certificate(test, device):
    """Authenticate only the declared Q1/contact changes and every recorded frame digest."""
    factory = FrozenRobotFactory("franka-cube", dataflow="legacy")
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    factory.JointTerms = SolverMonolithic.JointTerms
    with patch("newton.tests.test_monolithic_solver_coupled.SolverMonolithic", factory):
        run = solver(scene(device))
    fields = factory.fields
    old = {name: value.copy() for name, value in fields.items() if name != "direction_policy"}
    digest = _configuration_digest(old).hex()
    proof = certify_configuration(run, fields, digest)
    current = snapshot(run.create_state(dt=0.001))
    extra = {
        name
        for name in current
        if name.startswith("history.rigid_")
        or name in {"history.layout_version", "history.layout_digest", "history.actor_layout"}
    }
    legacy = {name: value.copy() for name, value in current.items() if name not in extra}
    legacy["history.schema_version"][:] = 3
    legacy["history.config_digest"] = np.frombuffer(bytes.fromhex(digest), np.uint8).copy()
    import warp as wp  # noqa: PLC0415

    arrays = {name: value.numpy() for name, value in vars(run.model).items() if isinstance(value, wp.array)}
    with tempfile.TemporaryDirectory() as folder:
        left, right = Path(folder) / "old", Path(folder) / "new"
        left.mkdir()
        right.mkdir()
        for path in (left, right):
            np.savez_compressed(path / "model.npz", **arrays)
        compatibility.TestActorRecording.write(left, legacy)
        compatibility.TestActorRecording.write(right, current)
        with test.assertRaisesRegex(ValueError, "digest"):
            compare(left, right, actor_layout_migration=True)
        test.assertEqual(
            compare(left, right, actor_layout_migration=True, configuration_certificate=proof)["status"], "PASS_EXACT"
        )
        for path, state in ((left, legacy), (right, current)):
            bad = {name: value.copy() for name, value in state.items()}
            bad["history.config_digest"][0] ^= 1
            compatibility.TestActorRecording.write(path, bad)
            with test.assertRaisesRegex(ValueError, "digest"):
                compare(left, right, actor_layout_migration=True, configuration_certificate=proof)
            compatibility.TestActorRecording.write(path, state)
    for name in ("direction_policy", "iteration_policy", "contact_parameters"):
        bad = {k: v.copy() for k, v in fields.items()}
        bad[name].flat[0] += 1
        with test.assertRaises(ValueError):
            certify_configuration(run, bad, digest)
    # A changed physical field plus a matching actual hash still cannot match the frozen hash.
    bad = {k: v.copy() for k, v in fields.items()}
    bad["contact_parameters"][0] *= 2
    saved = run._kinematics.digest
    run._kinematics.digest = _configuration_digest(bad)
    try:
        with test.assertRaisesRegex(ValueError, "frozen"):
            certify_configuration(run, bad, digest)
    finally:
        run._kinematics.digest = saved


class TestRobotContacts(unittest.TestCase):
    pass


def test_robot_diagnostics(test, device):
    """Match nonzero native diagnostics and committed forces without touching the spatial run."""
    import warp as wp  # noqa: PLC0415

    from newton.examples.softbody.monolithic_recording import measure, serial  # noqa: PLC0415
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    for law in ("C-EE", "C-IE", "C-II"):
        factory = FrozenRobotFactory("franka-cube", dataflow="spatial")
        factory.JointTerms = SolverMonolithic.JointTerms
        with patch("newton.tests.test_monolithic_solver_coupled.SolverMonolithic", factory):
            run = solver(scene(device), law=law, mode="production", graph=device.is_cuda)
        legacy = solver(scene(device), law=law, mode="production", graph=device.is_cuda)
        states = [owner.create_state(dt=0.001) for owner in (run, legacy)]
        for owner, state in zip((run, legacy), states, strict=True):
            owner.step(state, state, owner.model.control(), dt=0.001)
            test.assertTrue(owner.last_stats.converged, owner.last_stats)
        before = snapshot(states[0])
        candidate = run._contact._spatial.identities.numpy().copy()
        generation = run._contact.evaluation_generation
        stats = serial(run.last_stats)
        fields = factory.measure(run, states[0])
        expected = measure(legacy, states[1])
        test.assertEqual(fields["full_system"]["status"], "PASS_FULL_RESIDUAL")
        test.assertTrue(run._device_iteration)
        test.assertIsInstance(run._reference_residual, wp.array)
        test.assertGreater(fields["contact_load"], 0)
        for name in ("penetration", "contact_load", "friction_work", "slip", "force_balance"):
            np.testing.assert_allclose(fields[name], expected[name], rtol=1e-8, atol=1e-10, err_msg=name)
        np.testing.assert_allclose(
            fields["ground_force"],
            states[0]._monolithic_coupled.contact_particle_force.numpy().sum(axis=0),
            rtol=1e-8,
            atol=1e-10,
        )
        for name, value in snapshot(states[0]).items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)
        np.testing.assert_array_equal(run._contact._spatial.identities.numpy(), candidate)
        test.assertEqual(run._contact.evaluation_generation, generation)
        test.assertEqual(serial(run.last_stats), stats)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            row = serial({"step": 1, "publication": "CONVERGED", **fields})

            def write(value, path=path):
                with gzip.open(path / "steps.jsonl.gz", "wt") as stream:
                    stream.write(json.dumps(value) + "\n")

            write(row)
            test.assertEqual(check_robot_reports(path, run, 1)["full_converged"], 1)
            row["contact_pool"]["tainted"] = True
            write(row)
            with test.assertRaisesRegex(ValueError, "tainted"):
                check_robot_reports(path, run, 1)
            row["contact_pool"]["tainted"] = False
            row["full_system"]["blocks"]["q"]["goal"] *= 2
            write(row)
            with test.assertRaisesRegex(ValueError, "original physical gate"):
                check_robot_reports(path, run, 1)
            row["full_system"] = serial(fields["full_system"])
            block = row["full_system"]["blocks"]["q"]
            block["full_merit"], block["full_pass"] = block["goal"] * 2, False
            row["full_system"]["status"] = "FAIL_FULL_RESIDUAL"
            write(row)
            with test.assertRaisesRegex(ValueError, "complete system gate"):
                check_robot_reports(path, run, 1)
            row["publication"] = "STOPPED_ACCEPTED"
            write(row)
            summary = check_robot_reports(path, run, 1)
            test.assertEqual(summary["full_converged"], 0)
            test.assertEqual(summary["stopped"], 1)


for device in get_test_devices():
    add_function_test(
        TestRobotContacts, "test_robot_policy_certificate", test_robot_policy_certificate, devices=[device]
    )
    add_function_test(TestRobotContacts, "test_robot_diagnostics", test_robot_diagnostics, devices=[device])


if __name__ == "__main__":
    unittest.main()
