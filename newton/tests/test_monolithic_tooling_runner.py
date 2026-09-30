# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard exact profile expansion and qualification labels."""

import gzip
import json
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
    reference_binding,
    select_cases,
    write_report,
)


class TestSceneRunner(unittest.TestCase):
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
        """Expand ten full and two dev cases without granting filtered qualification."""
        full, dev = select_cases("full"), select_cases("dev")
        self.assertEqual(len(full), 10)
        self.assertEqual(select_cases("nightly"), full)
        self.assertEqual(dev, [full[0], full[-1]])
        self.assertEqual(select_cases("full", [full[2]["id"]]), [full[2]])
        self.assertEqual(aggregate([{"status": "PASS"}], True), "PARTIAL")
        self.assertEqual(aggregate([{"status": "PASS"}], False), "PASS")
        self.assertEqual(aggregate([{"status": "FAIL"}], True), "FAIL")
        self.assertEqual(
            aggregate([{"status": "PASS", "commit": "a"}, {"status": "PASS", "commit": "b"}], False), "FAIL"
        )
        with self.assertRaises(ValueError):
            select_cases("dev", [full[2]["id"]])

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
        """Keep exactly nine Franka variants and one reuse-only Sharpa case."""
        cases = baseline.cases()
        self.assertEqual(len({c["id"] for c in cases}), 10)
        franka = [c for c in cases if not c.get("reuse_only")]
        self.assertEqual(
            {(c["law"], c["globalization"]) for c in franka},
            {(law, g) for law in baseline.LAWS for g in baseline.GLOBALIZATIONS},
        )
        self.assertEqual(sum(c["steps"] for c in cases), 55917)

    def test_wrong_import_origin_rejected(self):
        """Reject an imported solver from a different checkout."""
        module = types.ModuleType("newton")
        module.__file__ = "/unexpected-checkout/newton/__init__.py"
        with patch.dict(baseline.sys.modules, {"newton": module}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unexpected module origin"):
                baseline.module_origins(ROOT, {})


if __name__ == "__main__":
    unittest.main()
