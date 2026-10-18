# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard exact profile expansion and qualification labels."""

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference import retained_assets
from scripts.monolithic_reference import run_baseline as baseline
from scripts.monolithic_reference.native_reference import frames as native_frames
from scripts.monolithic_reference.run_regression import (
    ROOT,
    aggregate,
    finalize_case,
    qualify_robot,
    reference_binding,
    select_cases,
    worker,
    write_report,
)


class TestSceneRunner(unittest.TestCase):
    def test_rigid_recipe_mismatch_fails(self):
        """Reject an otherwise passed rigid run whose recipe changed before qualification."""
        case = select_cases("nightly")[-1]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("scripts.monolithic_reference.run_regression.validate_inputs", return_value="frozen"),
            patch(
                "scripts.monolithic_reference.run_rigid_stack.record_run",
                return_value={"status": "PASS", "scene_input_sha256": "changed"},
            ),
        ):
            output = Path(directory) / "case"
            self.assertEqual(worker(case, output), 1)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "FAIL")
            self.assertIn("fixture changed", receipt["exception"])

    def test_recording_retention(self):
        """Drop only passed output recordings; preserve failures, summaries and opt-in full data."""
        for status, exit_code, retention, keep in (
            ("PASS", 0, "failures", False),
            ("FAIL", 0, "failures", True),
            ("PASS", 1, "failures", True),
            ("PASS", 0, "all", True),
        ):
            with (
                self.subTest(status=status, exit_code=exit_code, retention=retention),
                tempfile.TemporaryDirectory() as folder,
            ):
                root = Path(folder)
                recording = root / "recording"
                recording.mkdir()
                (recording / "frame.npz").write_bytes(b"new output")
                reference = root / "reference.npz"
                reference.write_bytes(b"retained reference")
                receipt = {"status": status, "scene_summary": {"published_steps": 3}}
                result = finalize_case(root, receipt, exit_code, retention)
                self.assertEqual(recording.exists(), keep)
                self.assertEqual(result, "PASS" if status == "PASS" and exit_code == 0 else "FAIL")
                self.assertEqual(json.loads((root / "receipt.json").read_text())["recording_retained"], keep)
                self.assertEqual(receipt["scene_summary"]["published_steps"], 3)
                self.assertEqual(reference.read_bytes(), b"retained reference")

    def test_nightly_report_keeps_stopped_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = {
                "profile": "nightly",
                "status": "FAIL",
                "complete": True,
                "cases": [
                    {
                        "case": "case1",
                        "status": "FAIL",
                        "recording_retained": True,
                        "counts": {"CONVERGED": 1, "STOPPED_ACCEPTED": 2},
                    }
                ],
            }
            write_report(output, summary)
            report = (output / "RESULTS.md").read_text()
            self.assertIn("NOT_RUN_BY_SCENE_RUNNER", report)
            self.assertIn("case1 | FAIL | 1 | 2 | True", report)

    def test_relocated_input_identity(self):
        """Read relocated bytes and reject mutation without replacing the frozen digest."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "deleted/asset.bin"
            asset = root / "data/asset.bin"
            asset.parent.mkdir()
            asset.write_bytes(b"qualified input")
            manifest = root / "inputs.json"
            manifest.write_text(json.dumps({"input_hashes": {str(old): retained_assets.sha(asset)}}))
            (root / "relocations.json").write_text(json.dumps([{"old_path": str(old), "asset_path": "data/asset.bin"}]))
            with patch.object(retained_assets, "ASSET_ROOT", root):
                self.assertEqual(retained_assets.validate_inputs(manifest), retained_assets.sha(manifest))
                asset.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    retained_assets.validate_inputs(manifest)

    def test_incomplete_reference_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "references.json").write_text("[]")
            with patch("scripts.monolithic_reference.run_regression.ASSET_ROOT", root):
                with self.assertRaisesRegex(ValueError, "exact ten"):
                    reference_binding(select_cases("full")[0])

    def test_native_reference_alignment_and_corruption(self):
        """Bridge the initial-frame offset and reject corrupt or shifted native chunks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = {
                "joint_qd": np.zeros(1, np.float64),
                "history.force_generation": np.zeros(1, np.int64),
                "history.force_valid": np.zeros(1, np.int32),
            }
            np.savez(root / "initial.npz", **initial)
            controls = root / "controls.npy"
            np.save(controls, np.zeros((102, 2, 1), np.float32))
            chunks = []
            for start, stop in ((1, 101), (101, 102)):
                path = root / f"states-{start}.npz"
                block = {name: np.repeat(value[None], stop - start, axis=0) for name, value in initial.items()}
                block["history.force_generation"][:, 0] = np.arange(start, stop)
                block["history.force_valid"][:] = 1
                np.savez(path, **block)
                chunks.append({"file": path.name, "frames": stop - start, "sha256": retained_assets.sha(path)})
            receipt = {
                "case": {"steps": 101, "dt": 0.001, "precision": "fp64"},
                "status": "COMPLETE_NOT_AUDITED",
                "published_steps": 101,
                "counts": {"CONVERGED": 100, "STOPPED": 1, "HARD_FAILURE": 0},
                "controls_sha256": retained_assets.sha(controls),
                "chunks": chunks,
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            with gzip.open(root / "steps.jsonl.gz", "wt") as log:
                for step in range(1, 102):
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "publication": "STOPPED" if step == 101 else "CONVERGED",
                                "stats": {"rolled_back": False, "min_det_f": 1.0},
                            }
                        )
                        + "\n"
                    )
            blocks = list(native_frames(root, controls))
            self.assertEqual([len(b["frame_steps"]) for b in blocks], [100, 2])
            np.testing.assert_array_equal(np.concatenate([b["frame_steps"] for b in blocks]), np.arange(102))
            self.assertEqual(blocks[0]["frame_publication"][0], "INITIAL")
            self.assertEqual(blocks[-1]["frame_publication"][-1], "STOPPED_ACCEPTED")
            path = root / chunks[-1]["file"]
            block["history.force_generation"][:] = 100
            np.savez(path, **block)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                list(native_frames(root, controls))
            chunks[-1]["sha256"] = retained_assets.sha(path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "publication generation"):
                list(native_frames(root, controls))

    def test_profiles(self):
        """Run only the retained production robot scenes, adding the rigid stack nightly."""
        full, dev = select_cases("full"), select_cases("dev")
        self.assertEqual(
            [case["id"] for case in full],
            [
                "franka-cube--ArmijoOrResNorm--C-IE--production--graph1",
                "sharpa-recorded-10mm--ArmijoOrResNorm--C-IE--production--graph1",
            ],
        )
        nightly = select_cases("nightly")
        self.assertEqual(nightly[:2], full)
        self.assertEqual(len(nightly), 3)
        self.assertEqual(nightly[-1]["id"], "rigid-stack-64--C-IE--ArmijoOrResNorm")
        self.assertEqual(select_cases("nightly", [nightly[-1]["id"]]), [nightly[-1]])
        self.assertEqual(dev, full)
        self.assertEqual(select_cases("full", [full[1]["id"]]), [full[1]])
        self.assertEqual(aggregate([{"status": "PASS"}], True), "PARTIAL")
        self.assertEqual(aggregate([{"status": "PASS"}], False), "PASS")
        self.assertEqual(aggregate([{"status": "FAIL"}], True), "FAIL")
        self.assertEqual(
            aggregate([{"status": "PASS", "commit": "a"}, {"status": "PASS", "commit": "b"}], False), "FAIL"
        )
        with self.assertRaises(ValueError):
            select_cases("dev", ["franka-cube--Armijo--C-EE--production--graph1"])

    def test_physical_gates_qualify_robot_runs(self):
        """Gate on physical plausibility; trajectory comparison is only reported."""
        scene_gates = {
            "hard_failures_max": 0,
            "stopped_fraction_max": 0.01,
            "penetration_max_m": 0.001,
            "det_f_min": 0.2,
            "require_held_cube": True,
        }
        summary = {
            "complete": True,
            "counts": {"CONVERGED": 100, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0},
            "maximum_penetration_m": 0.0,
            "minimum_det_f": 1.0,
            "held_cube": True,
        }
        frozen = {"version": 1, "status": "FROZEN", "robot": {"franka-cube": scene_gates}}
        result = qualify_robot("franka-cube", summary, {"status": "NOT_READY_TOLERANCE"}, gates=frozen)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["diagnostic_comparison"], "NOT_READY_TOLERANCE")
        proposed = {**frozen, "status": "PROPOSED"}
        self.assertEqual(qualify_robot("franka-cube", summary, {}, gates=proposed)["status"], "PARTIAL")
        failed = qualify_robot("franka-cube", {**summary, "held_cube": False}, {}, gates=frozen)
        self.assertEqual((failed["status"], failed["failures"]), ("FAIL", ["held_cube"]))

    def test_list(self):
        """List the frozen matrix without initializing a GPU or loading scenes."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.monolithic_reference.run_regression", "--profile", "full", "--list"],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), select_cases("full"))

    def test_scene_imports(self):
        """Load supported scenes without importing historical diagnostic drivers."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import newton.examples.softbody.monolithic_franka; "
                "import newton.examples.softbody.monolithic_sharpa_recorded; import sys; "
                "assert not any(n.startswith('scripts.monolithic_reference.') for n in sys.modules)",
            ],
            check=True,
        )

    def test_matrix(self):
        """List the production Franka case and the reuse-only Sharpa case."""
        cases = baseline.cases()
        self.assertEqual([(c["law"], c["globalization"]) for c in cases], [("C-IE", "ArmijoOrResNorm")] * 2)
        self.assertEqual([bool(c.get("reuse_only")) for c in cases], [False, True])
        self.assertEqual(sum(c["steps"] for c in cases), 13917)

    def test_wrong_import_origin_rejected(self):
        """Reject an imported solver from a different checkout."""
        module = types.ModuleType("newton")
        module.__file__ = "/unexpected-checkout/newton/__init__.py"
        with patch.dict(baseline.sys.modules, {"newton": module}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unexpected module origin"):
                baseline.module_origins(ROOT, {})


class TestManualAcceptance(unittest.TestCase):
    def setUp(self):
        path = ROOT.parent.parent / "dev/monolithic/test/run_manual_acceptance.py"
        if not path.is_file():
            self.skipTest("Manual launcher belongs to the external Monolithic workspace")
        spec = importlib.util.spec_from_file_location("manual_acceptance", path)
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)

    def test_dry_run(self):
        """Print the full fixed plan without creating output or starting subprocesses."""
        with tempfile.TemporaryDirectory() as directory, patch.object(self.launcher.subprocess, "run") as run:
            output = Path(directory) / "not-created"
            self.assertEqual(self.launcher.main(["--worktree", str(ROOT), "--output", str(output), "--dry-run"]), 0)
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_collection_and_failures(self):
        """Preserve failure receipts and launch nightly only after complete actor qualification."""
        identity = {"commit": "tested", "dirty": False}
        nightly = [case["id"] for case in select_cases("nightly")]
        for failure in (
            None,
            "actor_exit",
            "actor_summary",
            "nightly_filtered",
            "nightly_incomplete",
            "missing",
            "interrupt",
            "source",
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "batch"
                calls = []

                def execute(command, failure=failure, calls=calls, **kwargs):
                    calls.append(command)
                    self.assertEqual(kwargs["cwd"], ROOT)
                    self.assertEqual(kwargs["env"]["PYTHONPATH"], str(ROOT))
                    self.assertNotIn("VIRTUAL_ENV", kwargs["env"])
                    self.assertEqual(
                        Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]), ROOT.parent / "monolithic-integration/.venv"
                    )
                    destination = Path(command[command.index("--output") + 1])
                    destination.mkdir()
                    if failure == "interrupt":
                        raise KeyboardInterrupt()
                    if destination.name == "actor_coupling":
                        summary = {
                            "status": "FAIL" if failure == "actor_summary" else "PASS",
                            "cases": {n: {"status": "PASS"} for n in ("primary", "restored", "reordered", "complete")},
                        }
                    else:
                        self.assertIn("nightly", command)
                        self.assertNotIn("--case", command)
                        summary = {
                            "status": "PASS",
                            "profile": "nightly",
                            "filtered": failure == "nightly_filtered",
                            "complete": True,
                            "cases": [
                                {"case": name, "status": "PASS"}
                                for name in (nightly[:-1] if failure == "nightly_incomplete" else nightly)
                            ],
                        }
                    if failure != "missing":
                        (destination / "summary.json").write_text(json.dumps(summary))
                    return types.SimpleNamespace(returncode=int(failure == "actor_exit"))

                identities = [identity] * 5
                if failure == "source":
                    identities[2] = {**identity, "commit": "changed"}
                with (
                    patch.object(self.launcher, "source_identity", side_effect=identities),
                    patch.object(self.launcher.subprocess, "run", side_effect=execute),
                    patch.dict(
                        "os.environ",
                        {"VIRTUAL_ENV": "/wrong", "UV_PROJECT_ENVIRONMENT": "../monolithic-integration/.venv"},
                    ),
                ):
                    code = self.launcher.main(["--worktree", str(ROOT), "--output", str(output)])
                self.assertEqual(code, 0 if failure is None else 130 if failure == "interrupt" else 1)
                report = json.loads((output / "summary.json").read_text())
                self.assertEqual(report["status"], "PASS" if failure is None else "FAIL")
                self.assertEqual(report["stage_acceptance"], "REVIEW_REQUIRED")
                expected_calls = 2 if failure in (None, "nightly_filtered", "nightly_incomplete") else 1
                self.assertEqual(len(calls), expected_calls)
                if expected_calls == 1:
                    self.assertEqual(report["phases"]["nightly"]["status"], "NOT_RUN")
                self.assertTrue((output / "actor_coupling").is_dir())
                self.assertTrue((output / "RESULTS.md").is_file())

    def test_identity_and_output_refusals(self):
        """Refuse dirty checkouts and existing output without starting a long test."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            with (
                patch.object(self.launcher, "source_identity", return_value={"dirty": True}),
                self.assertRaises(SystemExit),
            ):
                self.launcher.main(["--worktree", str(ROOT), "--output", str(output)])
            with (
                patch.object(self.launcher, "source_identity", return_value={"dirty": False}),
                self.assertRaises(FileExistsError),
            ):
                self.launcher.main(["--worktree", str(ROOT), "--output", str(output)])

    def test_relative_environment_from_workspace(self):
        """Resolve an inherited environment relative to the caller before changing checkout."""
        previous = Path.cwd()
        shared = ROOT.parent / "monolithic-integration/.venv"
        try:
            os.chdir(ROOT.parent.parent)
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.dict("os.environ", {"UV_PROJECT_ENVIRONMENT": os.path.relpath(shared)}),
                patch.object(self.launcher, "source_identity", return_value={"dirty": False}),
                patch.object(self.launcher.subprocess, "run", side_effect=ValueError("environment probe")) as run,
            ):
                self.assertEqual(self.launcher.main(["--worktree", str(ROOT), "--output", directory + "/batch"]), 1)
                self.assertEqual(run.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(shared))
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
