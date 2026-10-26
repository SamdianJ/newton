# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify frozen mixed-actor controls, recording and continuation."""

import copy
import gzip
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import run_actor_coupling as coupling
from scripts.monolithic_reference.check_recording import check_actor, compare_actor, sha


def rewrite_chunk(folder, change):
    with np.load(next((folder / "trajectory").glob("*.npz"))) as archive:
        arrays = {n: archive[n].copy() for n in archive.files}
    change(arrays)
    index = json.loads((folder / "trajectory-index.json").read_text())
    path = folder / index["chunks"][0]["path"]
    np.savez_compressed(path, **arrays)
    index["chunks"][0]["sha256"] = sha(path)
    (folder / "trajectory-index.json").write_text(json.dumps(index))


def frozen_case(device, *, before=False, mode="diagnostic", graph=False):
    run = coupling.make_solver(coupling.build_model(device=device, before=before), mode=mode, graph=graph)
    return run, run.create_state(dt=0.001)


def test_prefix_matrix(test, device):
    """Advance the real frozen model, restore its midpoint and compare physical order."""
    routes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        routes.append(("production", True))
    test.integration_cases = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = None
        for (mode, graph), before in itertools.product(routes, (False, True)):
            run, state = frozen_case(device, before=before, mode=mode, graph=graph)
            folder = root / f"{mode}-{graph}-{before}"
            summary = coupling.record_run(run, state, folder, before=before, end_step=20, checkpoint_step=10)
            test.assertEqual(summary["status"], "PASS_PREFIX", summary)
            audit = check_actor(folder, require_complete=False)
            with test.assertRaisesRegex(ValueError, "short prefix|production/graph"):
                check_actor(folder)
            restored, target = frozen_case(device, before=before, mode=mode, graph=graph)
            coupling.restore_midpoint(restored, target, folder / "midpoint.npz", 10)
            resumed = root / (folder.name + "-restored")
            summary = coupling.record_run(restored, target, resumed, before=before, start_step=10, end_step=20)
            test.assertEqual(summary["status"], "PASS_PREFIX", summary)
            comparison = compare_actor(folder, resumed, require_complete=False)
            test.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE", comparison)
            if reference is None:
                reference = folder
            else:
                comparison = compare_actor(reference, folder, creation_order=before, require_complete=False)
                test.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE", comparison)
            test.integration_cases.append(
                {
                    "device": str(device),
                    "mode": mode,
                    "graph": graph,
                    "before": before,
                    "main_steps": 20,
                    "restored_steps": 10,
                    "audit": audit,
                    "comparison": comparison,
                }
            )


def test_active_recording(test, device):
    """Audit nonzero three-interface contact with native history and a real layout permutation."""
    test.integration_cases = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference, reference_run = None, None
        for permuted in (False, True):
            model = contact_scene(device)
            model.body_label = [
                "rigid_1" if name == "rigid_0" else "rigid_2" if name == "rigid_1" else name
                for name in model.body_label
            ]
            options = {
                "mode": "production" if device.is_cuda else "diagnostic",
                "graph": device.is_cuda,
                "permuted": permuted,
            }
            run = contact_solver(model, **options)
            state = contact_state(run, "chain")
            folder = root / str(permuted)
            summary = coupling.record_run(run, state, folder, end_step=10, checkpoint_step=5, component=True)
            test.assertEqual(summary["status"], "PASS_PREFIX", summary)
            audit = check_actor(folder, expected_run=run, require_complete=False)
            with test.assertRaisesRegex(ValueError, "explicit Model"):
                check_actor(folder, require_complete=False)
            restored = contact_solver(model, **options)
            target = contact_state(restored, "chain")
            coupling.restore_midpoint(restored, target, folder / "midpoint.npz", 5)
            resumed = root / (str(permuted) + "-restored")
            summary = coupling.record_run(restored, target, resumed, start_step=5, end_step=10, component=True)
            test.assertEqual(summary["status"], "PASS_PREFIX", summary)
            comparison = compare_actor(folder, resumed, reference_run=run, actual_run=restored, require_complete=False)
            test.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE")
            if reference is not None:
                comparison = compare_actor(
                    reference,
                    folder,
                    creation_order=True,
                    reference_run=reference_run,
                    actual_run=run,
                    require_complete=False,
                )
                test.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE")
            else:
                reference, reference_run = folder, run
            with gzip.open(folder / "steps.jsonl.gz", "rt") as stream:
                rows = [json.loads(line) for line in stream]
            test.assertTrue(
                all(np.linalg.norm(np.asarray(p["world_wrench"])[0, :3]) > 1e-4 for row in rows for p in row["pairs"])
            )
            test.integration_cases.append(
                {
                    "device": str(device),
                    "permuted": permuted,
                    "main_steps": 10,
                    "restored_steps": 5,
                    "audit": audit,
                    "comparison": comparison,
                }
            )


class TestActorCoupling(unittest.TestCase):
    def test_control_clock(self):
        """Sample quintic targets at absolute step endpoints and phase boundaries."""
        case = coupling.load_case()
        for step, position, velocity, phase in (
            (0, 0, 0, "load"),
            (500, 0.004, 0.015, "load"),
            (1000, 0.008, 0, "hold"),
            (1500, 0.008, 0, "unload"),
            (1750, 0.004, -0.03, "unload"),
            (2000, 0, 0, "unload"),
        ):
            q, qd, actual = coupling.targets(case, step)
            self.assertAlmostEqual(q, position)
            self.assertAlmostEqual(qd, velocity)
            self.assertEqual(actual, phase)
        q, v, _ = coupling.targets(case, 347)
        delta = 1e-3
        numerical = (coupling.targets(case, 347 + delta)[0] - coupling.targets(case, 347 - delta)[0]) / (
            2 * delta * case["dt"]
        )
        self.assertAlmostEqual(v, numerical, places=10)
        self.assertGreater(q, 0)
        for step in (-1, 2001, float("nan")):
            with self.assertRaises(ValueError):
                coupling.targets(case, step)

    def test_model_roles_and_filtering(self):
        """Keep authored masses, fixed nodes and only the intended three interfaces."""
        for before in (False, True):
            model = coupling.build_model(device="cpu", before=before)
            table = ActorTable.from_model(model)
            role_map = coupling.roles(model, table)
            self.assertEqual(set(role_map), {"pusher", "rigid_1", "rigid_2", "tet"})
            self.assertEqual((model.particle_count, model.tet_count, table.scalar_dof_count), (125, 384, 313))
            for name, mass in (("pusher", 1), ("rigid_1", 0.01), ("rigid_2", 0.01)):
                self.assertAlmostEqual(float(model.body_mass.numpy()[role_map[name]["body"]]), mass)
            fixed = model.particle_inv_mass.numpy() == 0
            self.assertEqual(fixed.sum(), 25)
            self.assertTrue(np.all(model.particle_q.numpy()[fixed, 0] == model.particle_q.numpy()[:, 0].max()))
            bodies = model.shape_body.numpy()
            pairs = {frozenset(model.body_label[bodies[i]] for i in pair) for pair in model.shape_contact_pairs.numpy()}
            self.assertEqual(pairs, {frozenset(("pusher", "rigid_1")), frozenset(("rigid_1", "rigid_2"))})
            flags = model.shape_flags.numpy()
            for name in ("pusher", "rigid_1", "rigid_2"):
                self.assertEqual(
                    bool(flags[role_map[name]["shape"][0]] & int(newton.ShapeFlags.COLLIDE_PARTICLES)),
                    name == "rigid_2",
                )

    def test_recording_corruption(self):
        """Reject unknown fields, bad identities, histories, publication and final snapshots."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, state = frozen_case("cpu")
            base = root / "valid"
            self.assertEqual(coupling.record_run(run, state, base, end_step=3)["status"], "PASS_PREFIX")
            self.assertEqual(check_actor(base, require_complete=False)["frames"], 4)
            mutations = {
                "unknown": lambda a: a.update(unexpected=np.zeros(4)),
                "missing": lambda a: a.pop("history.rigid_vsym"),
                "dtype": lambda a: a.update({"history.rigid_com": a["history.rigid_com"].astype(np.float32)}),
                "nonfinite": lambda a: a["history.rigid_vsym"].__setitem__((1, 0, 0), float("nan")),
                "generation": lambda a: a["history.force_generation"].__setitem__((1, 0), 3),
                "owner": lambda a: a["history.actor_layout"].__setitem__((1, 10), 255),
                "bound": lambda a: a["history.bound_q"].__setitem__((1, 0), 1),
                "rotation": lambda a: a["history.rigid_rotation"].__setitem__((1, 0, 3), 0.5),
                "rate": lambda a: a["history.rigid_v_com"].__setitem__((1, 0, 0), 0.1),
                "clock": lambda a: a["frame_times"].__setitem__(1, 0.002),
                "target": lambda a: a["frame_target_qd"].__setitem__(1, 0.1),
                "actual_target": lambda a: a["frame_control_qd"].__setitem__(1, 0.1),
                "stopped": lambda a: a["frame_publication"].__setitem__(1, "STOPPED_ACCEPTED"),
                "fixed": lambda a: a["particle_q"].__setitem__((1, -1, 0), 0.04),
                "force": lambda a: a["history.contact_body_force"].__setitem__((1, 1, 0), 1),
                "fixed_force": lambda a: a["history.contact_particle_force"].__setitem__((1, -1, 0), 1),
            }
            for name, change in mutations.items():
                with self.subTest(name=name):
                    folder = root / name
                    shutil.copytree(base, folder)
                    rewrite_chunk(folder, change)
                    with self.assertRaises(ValueError):
                        check_actor(folder, require_complete=False)
            for name in ("hash", "model", "final", "metadata", "execution", "summary", "log"):
                with self.subTest(name=name):
                    folder = root / name
                    shutil.copytree(base, folder)
                    if name == "hash":
                        with next((folder / "trajectory").glob("*.npz")).open("ab") as stream:
                            stream.write(b"bad")
                    elif name in ("model", "final"):
                        path = folder / (name + ".npz")
                        with np.load(path) as archive:
                            data = dict(archive)
                        if name == "model":
                            data["body_mass"] = data["body_mass"] + 1
                        else:
                            data.pop("history.rigid_com")
                        np.savez_compressed(path, **data)
                    elif name == "summary":
                        path = folder / "summary.json"
                        data = json.loads(path.read_text())
                        data["checked_steps"] = 1
                        path.write_text(json.dumps(data))
                    elif name in ("metadata", "execution"):
                        path = folder / "configuration.json"
                        data = json.loads(path.read_text())
                        if name == "execution":
                            data["execution"]["graph"] = True
                        else:
                            data["roles"]["rigid_1"]["body"] = 0
                        path.write_text(json.dumps(data))
                    else:
                        with gzip.open(folder / "steps.jsonl.gz", "rt") as stream:
                            rows = [json.loads(line) for line in stream]
                        rows[0]["stats"]["convergence_ratio"] = None
                        with gzip.open(folder / "steps.jsonl.gz", "wt") as stream:
                            for row in rows:
                                stream.write(json.dumps(row) + "\n")
                    with self.assertRaises(ValueError):
                        check_actor(folder, require_complete=False)

    def test_hold_and_cross_block_negatives(self):
        """Detect missing interface force, cross blocks and masked rigid residuals."""
        run, state = frozen_case("cpu")
        run.step(state, state, run.model.control(), dt=0.001)
        row = coupling.measure(run, state, 1)
        coupling.validate_row(row, coupling.load_case())
        data = run._linear.densify_for_test(generation=run._generation).raw_matrix.copy()
        arm = run._layout.indices("articulation")
        data[arm, :] = 0
        with patch.object(run._linear, "densify_for_test", return_value=SimpleNamespace(raw_matrix=data)):
            with self.assertRaisesRegex(ValueError, "cross block"):
                coupling.validate_row(coupling.measure(run, state, 1), coupling.load_case())
        bad = copy.deepcopy(row)
        bad["pairs"].pop()
        with self.assertRaises(ValueError):
            coupling.validate_row(bad, coupling.load_case())
        bad = copy.deepcopy(row)
        name, block = next(iter(bad["stats"]["variational"]["rigid_blocks"].items()))
        block["convergence_ratio_" + name] = 1.1
        with self.assertRaises(ValueError):
            coupling.validate_row(bad, coupling.load_case())
        bad = copy.deepcopy(row)
        bad["pairs"][0]["world_wrench"][0][3] += 0.01
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            coupling.validate_row(bad, coupling.load_case())
        rows = []
        for step in range(1401, 1501):
            value = copy.deepcopy(row)
            value.update(step=step, force_generation=step, tet_displacement=0.003)
            value["stats"].update(step_generation=step, joint_force_generation=step)
            value["stats"]["variational"]["contact_force_generation"] = step
            for pair in value["pairs"]:
                pair["force_generation"] = step
                pair["world_wrench"] = [[1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0]]
            rows.append(value)
        self.assertEqual(coupling.hold_gates(rows, coupling.load_case())["steps"], 100)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            coupling.hold_gates(rows[:-1], coupling.load_case())
        for field, value in (("tet_displacement", 0),):
            bad = copy.deepcopy(rows)
            bad[0][field] = value
            with self.assertRaisesRegex(ValueError, "insufficient"):
                coupling.hold_gates(bad, coupling.load_case())
        for low in (0, 0.01):
            bad = copy.deepcopy(rows)
            for pair in bad[0]["pairs"]:
                pair["world_wrench"] = [[low, 0, 0, 0, 0, 0], [-low, 0, 0, 0, 0, 0]]
            with self.assertRaisesRegex(ValueError, "insufficient"):
                coupling.hold_gates(bad, coupling.load_case())
        bad = copy.deepcopy(rows)
        for pair in bad[0]["pairs"]:
            pair["world_wrench"] = [[10, 0, 0, 0, 0, 0], [-10, 0, 0, 0, 0, 0]]
        with self.assertRaisesRegex(ValueError, "noisy"):
            coupling.hold_gates(bad, coupling.load_case())

    def test_checkpoint_and_failure_atomicity(self):
        """Reject wrong clocks/layouts and preserve complete state, history and force."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, state = frozen_case("cpu")
            base = root / "base"
            self.assertEqual(
                coupling.record_run(run, state, base, end_step=3, checkpoint_step=2)["status"], "PASS_PREFIX"
            )
            other, target = frozen_case("cpu")
            before = snapshot(target)
            with self.assertRaisesRegex(ValueError, "clock"):
                coupling.restore_midpoint(other, target, base / "midpoint.npz", 1)
            assert_snapshot(self, target, before)
            different, target2 = frozen_case("cpu", before=True)
            before2 = snapshot(target2)
            with self.assertRaisesRegex(ValueError, "identity"):
                coupling.restore_midpoint(different, target2, base / "midpoint.npz", 2)
            assert_snapshot(self, target2, before2)
            corrupted = root / "corrupted.npz"
            with np.load(base / "midpoint.npz") as archive:
                arrays = dict(archive)
            arrays["history.force_generation"] = np.array([1], np.int64)
            np.savez_compressed(corrupted, **arrays)
            sidecar = json.loads((base / "midpoint.json").read_text())
            sidecar["checkpoint_sha256"] = sha(corrupted)
            corrupted.with_suffix(".json").write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "generation"):
                coupling.restore_midpoint(other, target, corrupted, 2)
            assert_snapshot(self, target, before)
            arrays["history.force_generation"] = np.array([2], np.int64)
            arrays["history.last_dt"] = np.array([0.002], np.float64)
            np.savez_compressed(corrupted, **arrays)
            sidecar["checkpoint_sha256"] = sha(corrupted)
            corrupted.with_suffix(".json").write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "timestep"):
                coupling.restore_midpoint(other, target, corrupted, 2)
            assert_snapshot(self, target, before)
            with patch.object(other, "_prepare_step", side_effect=ValueError("injected assembly failure")):
                summary = coupling.record_run(other, target, root / "failed", end_step=2)
            self.assertEqual(summary["status"], "FAIL")
            self.assertTrue((root / "failed/failure-input.npz").exists())
            assert_snapshot(self, target, before)
            with self.assertRaises(ValueError):
                check_actor(root / "failed", require_complete=False)
            comparison = compare_actor(base, base, require_complete=False)
            self.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE")

    def test_device_equivalence(self):
        """Compare complete native CPU/CUDA histories under the frozen physical envelope."""
        if not wp.is_cuda_available():
            self.skipTest("CPU/CUDA equivalence requires CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for device in ("cpu", "cuda:0"):
                run, state = frozen_case(device, mode="production", graph=device != "cpu")
                summary = coupling.record_run(run, state, root / device, end_step=20)
                self.assertEqual(summary["status"], "PASS_PREFIX", summary)
            # Pool batch resources are device-specific, so configuration digests differ;
            # compare the physical roles under the frozen envelope instead.
            comparison = compare_actor(root / "cpu", root / "cuda:0", creation_order=True, require_complete=False)
            self.assertEqual(comparison["status"], "PASS_FROZEN_ENVELOPE", comparison)
            self.integration_cases = [comparison]

    def test_command_collection_and_retention(self):
        """Collect dependency failures and retain receipts before deleting successful recordings."""
        from scripts.monolithic_reference import check_recording, contact_audit, contact_system_audit  # noqa: PLC0415

        for failure, retain in ((None, "all"), (None, "failures"), ("step", "failures"), ("compare", "failures")):
            with self.subTest(failure=failure, retain=retain), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "batch"

                def record(run, state, folder, failure=failure, **kwargs):
                    folder.mkdir(parents=True)
                    for name in ("command.json", "source-hashes.json", "configuration.json"):
                        (folder / name).write_text("{}")
                    failed = failure == "step" and folder.parent.name == "restored"
                    return {"status": "FAIL" if failed else "PASS", "error": "injected" if failed else None}

                run = SimpleNamespace(create_state=lambda **kw: object())
                with (
                    patch("sys.argv", ["actor-coupling", "--output", str(output), "--retain-recordings", retain]),
                    patch.object(coupling, "build_model", return_value=object()),
                    patch.object(coupling, "make_solver", return_value=run),
                    patch.object(coupling, "record_run", side_effect=record) as recording,
                    patch.object(coupling, "restore_midpoint") as restore,
                    patch.object(check_recording, "check_actor", return_value={"status": "PASS"}),
                    patch.object(
                        check_recording,
                        "compare_actor",
                        return_value={"status": "FAIL" if failure == "compare" else "PASS_FROZEN_ENVELOPE"},
                    ),
                    patch.object(contact_system_audit, "ContactSystemAudit"),
                    patch.object(
                        contact_audit,
                        "compare_actor_contact_recordings",
                        return_value={"status": "PASS_FROZEN_ENVELOPE"},
                    ),
                ):
                    self.assertEqual(coupling.main(), int(failure is not None))
                result = json.loads((output / "summary.json").read_text())
                self.assertEqual(result["nightly"], "NOT_RUN")
                self.assertEqual(result["p5_status"], "PARTIAL")
                # The complete counter-trajectory follows every passing primary run.
                self.assertEqual(recording.call_count, 4 if failure is None else 3)
                self.assertEqual(restore.call_args.args[-1], 1000)
                for name in ("primary", "restored", "reordered"):
                    status = result["cases"][name]["status"]
                    if name == "reordered" and failure is not None:
                        self.assertEqual(status, "NOT_RUN")
                        continue
                    self.assertEqual(status, "FAIL" if name == "restored" and failure is not None else "PASS")
                    self.assertTrue((output / name / "receipt.json").exists())
                    self.assertEqual((output / name / "recording").exists(), failure is not None or retain == "all")


for device in get_test_devices():
    add_function_test(TestActorCoupling, "test_prefix_matrix", test_prefix_matrix, devices=[device])
    add_function_test(TestActorCoupling, "test_active_recording", test_active_recording, devices=[device])


if __name__ == "__main__":
    unittest.main()
