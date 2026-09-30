# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen precision matrix and host decision arithmetic contracts."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.monolithic_reference.run_p3_baseline import cases as baseline_cases
from scripts.monolithic_reference.run_p5_precision import cases, main, sha, validate_inputs


class TestPrecisionRunner(unittest.TestCase):
    def test_matrix(self):
        """Match the ten existing P4 configurations in each precision, excluding Close."""
        rows = cases()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r["id"] for r in rows}), 20)
        for precision in ("fp32", "fp64"):
            selected = [r for r in rows if r["precision"] == precision]
            self.assertEqual(sum(r["steps"] for r in selected), 55917)
            self.assertEqual(sum(r["scene"].startswith("close-") for r in selected), 0)
            self.assertEqual(
                {(r["reference_case"], r["law"], r["globalization"]) for r in selected},
                {(r["id"], r["law"], r["globalization"]) for r in baseline_cases()},
            )
            self.assertTrue(all(r["graph"] and r["pcg_mode"] == "production" for r in selected))
            self.assertTrue(all(r["publish_policy"] == "current_iterate" for r in selected))

    def test_default_fp32_list(self):
        """List only ten FP32 cases by default so baseline trajectories are not rerun."""
        output = io.StringIO()
        with patch("sys.argv", ["run_p5_precision", "--list"]), redirect_stdout(output):
            self.assertEqual(main(), 0)
        rows = json.loads(output.getvalue())
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["precision"] == "fp32" for r in rows))

    def test_explicit_input_manifest(self):
        """A new asset generation must have its own checked manifest."""
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset.bin"
            manifest = Path(directory) / "inputs.json"
            asset.write_bytes(b"new qualified asset")
            manifest.write_text(json.dumps({"input_hashes": {str(asset): sha(asset)}}))
            self.assertEqual(validate_inputs(manifest), sha(manifest))
            asset.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_inputs(manifest)
            asset.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_inputs(manifest)

    def test_worker_manifest_forwarding(self):
        case = next(c for c in cases() if c["precision"] == "fp32")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = [
                "runner",
                "--worker",
                "--case",
                case["id"],
                "--controls",
                str(root),
                "--output",
                str(root / "run"),
                "--input-manifest",
                str(root / "inputs.json"),
            ]
            with (
                patch("sys.argv", args),
                patch("scripts.monolithic_reference.run_p5_precision.worker", return_value=0) as worker,
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(worker.call_args.kwargs["input_manifest"], root / "inputs.json")

    def test_fp32_tolerance_profile(self):
        output = io.StringIO()
        with patch("sys.argv", ["runner", "--list", "--fp32-absolute-tolerance", "1e-6"]), redirect_stdout(output):
            self.assertEqual(main(), 0)
        rows = json.loads(output.getvalue())
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["fp32_absolute_tolerance"] == 1e-6 for r in rows))
        with patch("sys.argv", ["runner", "--list", "--precision", "fp64", "--fp32-absolute-tolerance", "1e-6"]):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
