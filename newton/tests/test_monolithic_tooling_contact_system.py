# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep complete-system residual checks independent of spatial convergence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.examples.softbody.monolithic_recording import serial
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_state, frame
from newton.tests.test_monolithic_solver_contact_pool import pooled_solver
from newton.tests.test_monolithic_solver_contact_spatial import spatial_solver
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.test_monolithic_solver_rigid import assert_physical
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import run_actor_coupling as coupling
from scripts.monolithic_reference.actor_recording import check_actor, compare_actor
from scripts.monolithic_reference.compare_contact_dataflows import compare_actor_contact_recordings
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit


def test_full_system_residual(test, device):
    """Check every active block and preserve production state, diagnostics and candidate generation."""
    for mode, graph in [("diagnostic", False), ("production", device.is_cuda)]:
        for law in ("C-EE", "C-IE", "C-II"):
            options = {"mode": mode, "graph": graph, "law": law, "permuted": True}
            run = spatial_solver(contact_scene(device), **options)
            full = pooled_solver(contact_scene(device), **options)
            # The native full stream does not consume the auditor's admission budget.
            audit_run = pooled_solver(contact_scene(device), contact_factor_capacity=0, **options)
            audit = ContactSystemAudit(run, audit_run)
            states = [contact_state(owner, "chain") for owner in (run, full)]
            for _ in range(2):
                for owner, state in zip((run, full), states, strict=True):
                    owner.step(state, state, owner.model.control(), dt=0.001)
                    test.assertTrue(owner.last_stats.converged, owner.last_stats)
                before = snapshot(states[0])
                metrics, variational = serial(run._metrics), serial(run._variational)
                candidates = run._contact._spatial.identities.numpy().copy()
                count = run._contact._spatial.count.numpy().copy()
                generation = run._contact.candidate_generation
                result = audit.evaluate()
                test.assertEqual(result["status"], "PASS_FULL_RESIDUAL", result)
                test.assertEqual(len(result["blocks"]), 7)
                np.testing.assert_allclose(audit.residual.numpy(), full._complete_r.numpy(), rtol=1e-5, atol=1e-8)
                assert_physical(test, frame(full, states[1]), frame(run, states[0]))
                assert_snapshot(test, states[0], before)
                test.assertEqual(serial(run._metrics), metrics)
                test.assertEqual(serial(run._variational), variational)
                np.testing.assert_array_equal(run._contact._spatial.identities.numpy(), candidates)
                np.testing.assert_array_equal(run._contact._spatial.count.numpy(), count)
                test.assertEqual(run._contact.candidate_generation, generation)


def test_full_system_detects_miss(test, device):
    """Reject clipped convergence when a deliberately omitted contact violates the full residual gate."""
    options = {"mode": "production", "graph": device.is_cuda, "law": "C-II"}
    run = spatial_solver(contact_scene(device), **options)
    audit = ContactSystemAudit(run, pooled_solver(contact_scene(device), contact_factor_capacity=0, **options))
    state = contact_state(run, "chain")
    refresh = run._contact.refresh

    def omit(*args, **kwargs):
        refresh(*args, **kwargs)
        run._contact._spatial.count.zero_()

    with patch.object(run._contact, "refresh", omit):
        run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    result = audit.evaluate()
    test.assertEqual(result["candidate_count"], 0)
    test.assertEqual(result["status"], "FAIL_FULL_RESIDUAL", result)
    test.assertGreater(result["residual_difference_l2"], 0)
    test.assertTrue(all(b["clipped_pass"] for b in result["blocks"].values()))
    with patch.object(run._contact, "_tainted", True):
        with test.assertRaisesRegex(ValueError, "untainted"):
            audit.evaluate()


def test_actor_system_report(test, device):
    """Bind complete-system reports to a short actor recording without changing its trajectory schema."""
    options = {"mode": "production", "graph": device.is_cuda}
    run = coupling.make_solver(
        coupling.build_model(device=device),
        contact_dataflow="spatial",
        spatial={"candidate_capacity": 4096, "law_margin": 0.0},
        **options,
    )
    complete = coupling.make_solver(
        coupling.build_model(device=device), contact_dataflow="pooled", contact_factor_capacity=0, **options
    )
    audit = ContactSystemAudit(run, complete)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "recording"
        result = coupling.record_run(
            run, run.create_state(dt=0.001), path, end_step=3, checkpoint_step=2, system_audit=audit
        )
        test.assertEqual(result["status"], "PASS_PREFIX", result)
        report = result["full_system"]
        test.assertEqual(report["checked_steps"], 3)
        test.assertEqual(report["sha256"], coupling.sha(path / report["file"]))
        rows = [json.loads(line) for line in (path / report["file"]).read_text().splitlines()]
        test.assertEqual([row["step"] for row in rows], [1, 2, 3])
        test.assertTrue(all(row["status"] == "PASS_FULL_RESIDUAL" for row in rows))
        check_actor(path, expected_run=run, require_complete=False)
        restored = coupling.make_solver(
            coupling.build_model(device=device),
            contact_dataflow="spatial",
            spatial={"candidate_capacity": 4096, "law_margin": 0.0},
            **options,
        )
        target = restored.create_state(dt=0.001)
        coupling.restore_midpoint(restored, target, path / "midpoint.npz", 2)
        complete = coupling.make_solver(
            coupling.build_model(device=device), contact_dataflow="pooled", contact_factor_capacity=0, **options
        )
        resumed_path = Path(folder) / "resumed"
        resumed = coupling.record_run(
            restored,
            target,
            resumed_path,
            start_step=2,
            end_step=3,
            system_audit=ContactSystemAudit(restored, complete),
        )
        test.assertEqual(resumed["status"], "PASS_PREFIX", resumed)
        test.assertEqual(resumed["full_system"]["checked_steps"], 1)
        test.assertEqual(
            compare_actor(path, resumed_path, reference_run=run, actual_run=restored, require_complete=False)["status"],
            "PASS_FROZEN_ENVELOPE",
        )
        bad = json.loads((path / "summary.json").read_text())
        bad["full_system"]["sha256"] = "00" * 32
        (path / "summary.json").write_text(json.dumps(bad))
        with test.assertRaisesRegex(ValueError, "report identity"):
            check_actor(path, expected_run=run, require_complete=False)
        (path / "summary.json").write_text(json.dumps(result))
        damaged = [dict(row) for row in rows]
        damaged[0] = {
            **rows[0],
            "blocks": {
                **rows[0]["blocks"],
                "global": {**rows[0]["blocks"]["global"], "goal": rows[0]["blocks"]["global"]["goal"] * 2},
            },
        }
        (path / report["file"]).write_text("".join(json.dumps(row) + "\n" for row in damaged))
        result["full_system"]["sha256"] = coupling.sha(path / report["file"])
        (path / "summary.json").write_text(json.dumps(result))
        with test.assertRaisesRegex(ValueError, "physical gate"):
            check_actor(path, expected_run=run, require_complete=False)


def test_actor_contact_trajectories(test, device):
    """Compare independent B2/B3 histories and reject source or physical input substitutions."""
    options = {"mode": "production", "graph": device.is_cuda}
    full = coupling.make_solver(coupling.build_model(device=device), contact_dataflow="pooled", **options)
    spatial = coupling.make_solver(
        coupling.build_model(device=device),
        contact_dataflow="spatial",
        spatial={"candidate_capacity": 4096, "law_margin": 0.0},
        **options,
    )
    audit_run = coupling.make_solver(
        coupling.build_model(device=device), contact_dataflow="pooled", contact_factor_capacity=0, **options
    )
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / name for name in ("complete", "spatial")]
        for run, path in zip((full, spatial), paths, strict=True):
            summary = coupling.record_run(
                run,
                run.create_state(dt=0.001),
                path,
                end_step=3,
                system_audit=ContactSystemAudit(run, audit_run) if run._contact_spatial else None,
            )
            test.assertEqual(summary["status"], "PASS_PREFIX", summary)
        with test.assertRaises(ValueError):
            compare_actor(*paths, reference_run=full, actual_run=spatial, require_complete=False)
        options = {"reference_run": full, "actual_run": spatial, "require_complete": False}
        result = compare_actor_contact_recordings(*paths, **options)
        test.assertEqual(result["status"], "PASS_FROZEN_ENVELOPE")
        test.assertEqual(result["frames"], 4)
        source = paths[1] / "source-hashes.json"
        hashes = json.loads(source.read_text())
        source.write_text(json.dumps({**hashes, "changed": "00" * 32}))
        with test.assertRaisesRegex(ValueError, "Mixed source"):
            compare_actor_contact_recordings(*paths, **options)
        source.write_text(json.dumps(hashes))
        altered = {**hashes, "scripts/monolithic_reference/run_actor_coupling.py": "00" * 32}
        for path in paths:
            (path / "source-hashes.json").write_text(json.dumps(altered))
        with test.assertRaisesRegex(ValueError, "current source"):
            compare_actor_contact_recordings(*paths, **options)
        for path in paths:
            (path / "source-hashes.json").write_text(json.dumps(hashes))
        with patch.object(spatial, "linear_tolerance", full.linear_tolerance * 2):
            with test.assertRaises(ValueError):
                compare_actor_contact_recordings(*paths, **options)


class TestContactSystemAudit(unittest.TestCase):
    pass


for device in get_test_devices():
    for name in (
        "test_full_system_residual",
        "test_full_system_detects_miss",
        "test_actor_system_report",
        "test_actor_contact_trajectories",
    ):
        add_function_test(TestContactSystemAudit, name, globals()[name], devices=[device])


if __name__ == "__main__":
    unittest.main()
