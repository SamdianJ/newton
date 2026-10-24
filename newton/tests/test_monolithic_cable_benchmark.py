# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify auditable restart of real cable benchmark runs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestMonolithicCableBenchmark(unittest.TestCase):
    def test_progress_resume_and_identity_rejection(self):
        """Resume actual steps exactly and reject mismatched, corrupt or failed evidence."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run(name, *options, success=True):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "scripts.monolithic_reference.run_cable_benchmark",
                        "cantilever",
                        "--segments",
                        "32",
                        "--output",
                        str(root / name),
                        *options,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode == 0, success, result.stderr[-2000:])
                return result.stderr

            run("resumed", "--max-steps", "100")
            run("resumed", "--resume", "--max-steps", "101")
            run("continuous", "--max-steps", "101")
            receipt = root / "resumed/receipt.json"
            original = receipt.read_bytes()
            record = json.loads(original)
            self.assertEqual(record["status"], "PARTIAL")
            self.assertEqual(record["metrics"]["steps"], 101)
            self.assertEqual(record["resumed_from_step"], 100)
            self.assertEqual(record["precision"], "fp64")
            with np.load(root / "resumed/final.npz") as resumed, np.load(root / "continuous/final.npz") as continuous:
                for key in resumed.files:
                    np.testing.assert_array_equal(resumed[key], continuous[key])
            for options in (("--ea-multiplier", "10"), ("--precision", "fp32")):
                error = run("resumed", "--resume", *options, success=False)
                self.assertIn("identical source, recipe and execution arguments", error)
            self.assertEqual(receipt.read_bytes(), original)
            record["status"] = "FAIL"
            receipt.write_text(json.dumps(record))
            error = run("resumed", "--resume", success=False)
            self.assertIn("failed experiment cannot be resumed", error)
            receipt.write_bytes(original)
            checkpoint = root / "resumed/progress.npz"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupted")
            error = run("resumed", "--resume", success=False)
            self.assertIn("checkpoint and metadata do not match", error)
            self.assertEqual(receipt.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
