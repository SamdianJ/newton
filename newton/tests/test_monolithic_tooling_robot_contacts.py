# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authenticate frozen robot policies and record pooled contact diagnostics."""

import gzip
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.articulation import _configuration_digest
from newton.tests import test_monolithic_tooling_recording_compatibility as compatibility
from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.check_recording import compare
from scripts.monolithic_reference.contact_routes import pooled_audit_route
from scripts.monolithic_reference.robot_contact_acceptance import (
    FrozenRobotFactory,
    certify_configuration,
    check_robot_reports,
)


def test_robot_policy_certificate(test, device):
    """Bind the actual identity; report frozen-reference matches and still reject corrupted frames."""
    factory = FrozenRobotFactory("franka-cube", dataflow="default")
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    factory.JointTerms = SolverMonolithic.JointTerms
    with patch("newton.tests.test_monolithic_solver_coupled.SolverMonolithic", factory):
        run = solver(scene(device))
    recipe = factory.recipe()
    policy = recipe["numerical_policy"]
    test.assertEqual(policy["effective_internal_config"], asdict(run._config))
    test.assertEqual(
        policy["effective_internal_config"]["regularization_values"], (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
    )
    test.assertEqual(policy["constructor_digest"], run._kinematics.digest.hex())
    test.assertEqual(
        recipe["numerical_policy_sha256"],
        hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    original = run._config
    run._config = replace(original, regularization_values=(0.0, 0.01))
    test.assertNotEqual(factory.recipe()["numerical_policy_sha256"], recipe["numerical_policy_sha256"])
    run._config = original
    test.assertEqual(factory.recipe(), recipe)
    fields = factory.fields
    # A frozen reference recorded before identity version 3 cannot match the current digest.
    old = {name: value.copy() for name, value in fields.items() if name != "config_version"}
    digest = _configuration_digest(old).hex()
    proof = certify_configuration(run, fields, digest)
    test.assertFalse(proof.receipt()["authenticated"])
    test.assertTrue(certify_configuration(run, fields, run._kinematics.digest.hex()).receipt()["authenticated"])
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
        result = compare(left, right, actor_layout_migration=True, configuration_certificate=proof)
        test.assertEqual(result["status"], "PASS_EXACT")
        test.assertFalse(result["configuration_certificate"]["authenticated"])
        for path, state in ((left, legacy), (right, current)):
            bad = {name: value.copy() for name, value in state.items()}
            bad["history.config_digest"][0] ^= 1
            compatibility.TestActorRecording.write(path, bad)
            with test.assertRaisesRegex(ValueError, "digest"):
                compare(left, right, actor_layout_migration=True, configuration_certificate=proof)
            compatibility.TestActorRecording.write(path, state)
    # The actual digest must reproduce from the recorded fields.
    for name in ("config_version", "iteration_policy", "contact_parameters"):
        bad = {k: v.copy() for k, v in fields.items()}
        bad[name].flat[0] += 1
        with test.assertRaisesRegex(ValueError, "Actual constructor configuration"):
            certify_configuration(run, bad, digest)


def test_robot_precision(test, device):
    """Build, certify and audit an FP32 robot recipe; a scene of another precision is rejected."""
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    def build(**options):
        factory = FrozenRobotFactory("franka-cube", precision="fp32")
        factory.JointTerms = SolverMonolithic.JointTerms
        with patch("newton.tests.test_monolithic_solver_coupled.SolverMonolithic", factory):
            return factory, solver(scene(device), mode="production", graph=device.is_cuda, **options)

    with test.assertRaisesRegex(ValueError, "precision"):
        build()
    factory, run = build(precision="fp32")
    test.assertEqual(run.precision, SolverMonolithic.Precision.FP32)
    # FP32 keeps the complete candidate bounds; the reduced Franka bounds are FP64 calibrations.
    test.assertEqual(run._contact.candidate_capacity, run._contact.count)
    test.assertEqual(factory.recipe()["precision"], "fp32")
    certificate = factory.certify(run._kinematics.digest.hex())
    test.assertTrue(certificate.receipt()["authenticated"])
    with test.assertRaisesRegex(ValueError, "Actual constructor configuration"):
        certify_configuration(run, factory.fields, run._kinematics.digest.hex())
    state = run.create_state(dt=0.001)
    run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    test.assertEqual(factory.measure(run, state)["full_system"]["status"], "PASS_FULL_RESIDUAL")


class TestRobotContacts(unittest.TestCase):
    pass


def test_robot_diagnostics(test, device):
    """Match nonzero native diagnostics and committed forces without touching the spatial run."""
    import warp as wp  # noqa: PLC0415

    from newton.examples.softbody.monolithic_recording import measure, serial  # noqa: PLC0415
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    for dataflow in ("spatial", "default"):
        factory = FrozenRobotFactory("franka-cube", dataflow=dataflow)
        factory.JointTerms = SolverMonolithic.JointTerms
        with patch("newton.tests.test_monolithic_solver_coupled.SolverMonolithic", factory):
            run = solver(scene(device), mode="production", graph=device.is_cuda)
        test.assertEqual(run._config.regularization_values, (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0))
        test.assertEqual(factory.recipe()["production_default"], dataflow == "default")
        if dataflow == "default":
            test.assertEqual(run._contact.candidate_capacity, run._contact.count)
        test.assertEqual(factory.recorder.audit.complete._config, run._config)
        with pooled_audit_route():
            complete = solver(scene(device), mode="production", graph=device.is_cuda)
        states = [owner.create_state(dt=0.001) for owner in (run, complete)]
        for owner, state in zip((run, complete), states, strict=True):
            owner.step(state, state, owner.model.control(), dt=0.001)
            test.assertTrue(owner.last_stats.converged, owner.last_stats)
        before = snapshot(states[0])
        candidate = run._contact._spatial.identities.numpy().copy()
        generation = run._contact.evaluation_generation
        stats = serial(run.last_stats)
        fields = factory.measure(run, states[0])
        expected = measure(complete, states[1])
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
            row["full_system"]["blocks"]["q"]["full_ratio"] = 2.0
            write(row)
            with test.assertRaisesRegex(ValueError, "original physical gate"):
                check_robot_reports(path, run, 1)
            row["full_system"] = serial(fields["full_system"])
            block = row["full_system"]["blocks"]["q"]
            block["full_ratio"], block["full_pass"] = 2.0, False
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
    add_function_test(TestRobotContacts, "test_robot_precision", test_robot_precision, devices=[device])


if __name__ == "__main__":
    unittest.main()
