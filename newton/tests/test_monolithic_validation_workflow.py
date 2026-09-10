# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check validation scope, resumability and retained acceptance failures."""

import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.monolithic_reference import validate_p2 as workflow


class TestMonolithicValidationWorkflow(unittest.TestCase):
    def test_development_scope(self):
        """Keep full trajectories out of the development gate."""
        jobs = workflow.build_jobs("dev", Path("/candidate"), Path("/baseline"))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["kind"], "tests")
        self.assertIn("newton.tests.test_monolithic_execution_options", jobs[0]["arguments"])
        self.assertNotIn("scripts.monolithic_reference.run_p2_8bc", jobs[0]["arguments"])

    def test_night_scope(self):
        """Retain the fifty full trajectories and independent variant comparisons."""
        jobs = workflow.build_jobs("night", Path(__file__).resolve().parents[2], Path("/baseline"))
        trajectories = [j for j in jobs if j["kind"] != "tests"]
        self.assertEqual(len(trajectories), 50)
        self.assertEqual(len({j["name"] for j in jobs}), len(jobs))
        for repeat in range(1, 6):
            for variant in ("baseline", "diagnostic", "owned", "production"):
                self.assertIn(f"sharpa-r3-{variant}-rep{repeat:02d}", {j["name"] for j in trajectories})
        self.assertTrue(all("--smoke" not in j["arguments"] for j in trajectories))
        old = next(j for j in trajectories if j["name"] == "sharpa-r3-baseline-rep01")
        self.assertEqual(old["cwd"], "/baseline")
        self.assertNotIn("--pcg-mode", old["arguments"])

    def test_resume_source_and_log_integrity(self):
        """Reject stale source and modified successful logs before skipping work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "test.log"
            log.write_text("passed\n")
            state = {
                "signature": {"commit": "old"},
                "completed": {"tests": {"log": str(log), "log_sha256": workflow.digest(log), "output": None}},
            }
            with self.assertRaises(ValueError):
                workflow.validate_resume(state, {"commit": "new"})
            workflow.validate_resume(state, state["signature"])
            log.write_text("changed\n")
            with self.assertRaises(ValueError):
                workflow.validate_resume(state, state["signature"])

    def test_interrupted_output_is_retained(self):
        """Archive an incomplete attempt before retrying the same logical job."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "runs" / "case"
            output.mkdir(parents=True)
            (output / "run.json").write_text(json.dumps({"status": "RUNNING"}))
            archive = workflow.archive_attempt(root, "case", output)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads((archive / "run.json").read_text())["status"], "RUNNING")

    def test_resume_evidence_integrity(self):
        """Reject a changed raw artifact even when its index remains untouched."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "test.log"
            log.write_text("passed\n")
            raw = root / "trace.jsonl"
            raw.write_text("original\n")
            index = root / "evidence-index.json"
            index.write_text(json.dumps({raw.name: workflow.digest(raw)}))
            state = {
                "signature": {},
                "completed": {
                    "case": {
                        "log": str(log),
                        "log_sha256": workflow.digest(log),
                        "output": str(root),
                        "index_sha256": workflow.digest(index),
                    }
                },
            }
            workflow.validate_resume(state, {})
            raw.write_text("changed\n")
            with self.assertRaises(ValueError):
                workflow.validate_resume(state, {})

    def test_failure_and_retry_logs(self):
        """Propagate child failure and retain its log when a retry succeeds."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            job = {
                "name": "regression",
                "kind": "tests",
                "cwd": str(root),
                "arguments": [
                    "-c",
                    "from pathlib import Path; print('attempt'); raise SystemExit(0 if Path('ready').exists() else 1)",
                ],
            }
            state = {"signature": {"sources": {str(root): {}}}, "completed": {}}
            with patch.object(workflow, "source_signature", return_value={}):
                self.assertEqual(workflow.run_jobs(root, [job], state, clean=False), 1)
                old_log = Path(state["failures"][0]["log"])
                old_hash = workflow.digest(old_log)
                (root / "ready").touch()
                self.assertEqual(workflow.run_jobs(root, [job], state, clean=False), 0)
            self.assertEqual(workflow.digest(old_log), old_hash)
            self.assertNotEqual(state["completed"]["regression"]["log"], str(old_log))

    def test_actual_pcg_call_budget(self):
        """Retain a failed p95 budget even when all nonlinear steps succeeded."""
        rows = [{"pcg_calls": [{"iterations": 32}, {"iterations": 200}], "linear_iterations": 232}]
        budget = workflow.pcg_budget(rows)
        self.assertEqual(budget["actual_calls"], 2)
        self.assertAlmostEqual(budget["p95"], 191.6)
        self.assertFalse(budget["p95_passed"])
        self.assertTrue(budget["hard_cap_passed"])

    def test_interrupt_stops_child_group(self):
        """Stop the owned child group and persist resumable progress on interruption."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            job = {"name": "regression", "kind": "tests", "cwd": str(root), "arguments": []}
            state = {"signature": {"sources": {str(root): {}}}, "completed": {}}
            with (
                patch.object(workflow, "source_signature", return_value={}),
                patch.object(workflow.subprocess, "Popen") as start,
                patch.object(workflow.os, "killpg") as stop,
            ):
                start.return_value.pid = 12345
                start.return_value.wait.side_effect = [KeyboardInterrupt(), 0]
                with self.assertRaises(KeyboardInterrupt):
                    workflow.run_jobs(root, [job], state, clean=False)
                stop.assert_called_once_with(12345, signal.SIGTERM)
            self.assertEqual(json.loads((root / "progress.json").read_text())["status"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
