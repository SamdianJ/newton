# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check validation scope, resumability and retained acceptance failures."""

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.monolithic_reference import validate_p2 as workflow


class TestMonolithicValidationWorkflow(unittest.TestCase):
    def test_development_scope(self):
        """Keep full trajectories out of the development gate."""
        jobs = workflow.build_jobs("dev", Path("/candidate"))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["kind"], "tests")
        self.assertIn("newton.tests.test_monolithic_execution_options", jobs[0]["arguments"])
        self.assertNotIn("scripts.monolithic_reference.run_p2_8bc", jobs[0]["arguments"])

    def test_night_scope(self):
        """Schedule only current policies with explicit graph variants and balanced repeats."""
        repo = Path(__file__).resolve().parents[2]
        jobs = workflow.build_jobs("night", repo)
        self.assertEqual(len({j["name"] for j in jobs}), len(jobs))
        self.assertTrue(all(j["cwd"] == str(repo) for j in jobs))
        for job in jobs[1:]:
            self.assertNotIn("baseline", job["name"])
            self.assertNotIn("--smoke", job["arguments"])

        names = {job["name"] for job in jobs}
        for repeat in range(1, 6):
            for mode in ("diagnostic", "production", "production-graph"):
                self.assertIn(f"tet-r5-{mode}-rep{repeat:02d}", names)
        for job in jobs[1:]:
            self.assertEqual("--use-cuda-graph" in job["arguments"], "-graph-" in job["name"])

    def test_reject_old_policy(self):
        """Reject an old saved plan even when its evidence is intact."""
        with self.assertRaises(ValueError):
            workflow.validate_resume({"signature": {"policy": workflow.WORKFLOW_POLICY}, "completed": {}}, {})

    def test_old_resume_starts_no_subprocess(self):
        """Reject a historical plan before running git, probes or child jobs."""

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "progress.json").write_text(
                json.dumps({"signature": {"policy": "p2-8c2-masked-smart-v1"}, "completed": {}})
            )
            with (
                patch.object(sys, "argv", ["validate", "--resume", "--output", tmp]),
                patch.object(workflow.subprocess, "check_output") as read,
                patch.object(workflow.subprocess, "run") as run,
                patch.object(workflow.subprocess, "Popen") as start,
            ):
                with self.assertRaises(SystemExit):
                    workflow.main()
                read.assert_not_called()
                run.assert_not_called()
                start.assert_not_called()

    def test_resume_source_and_log_integrity(self):
        """Reject stale source and modified successful logs before skipping work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "test.log"
            log.write_text("passed\n")
            state = {
                "signature": {"policy": workflow.WORKFLOW_POLICY, "commit": "old"},
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
                "signature": {"policy": workflow.WORKFLOW_POLICY},
                "completed": {
                    "case": {
                        "log": str(log),
                        "log_sha256": workflow.digest(log),
                        "output": str(root),
                        "index_sha256": workflow.digest(index),
                    }
                },
            }
            workflow.validate_resume(state, {"policy": workflow.WORKFLOW_POLICY})
            raw.write_text("changed\n")
            with self.assertRaises(ValueError):
                workflow.validate_resume(state, {"policy": workflow.WORKFLOW_POLICY})

    def test_failure_and_retry_logs(self):
        """Finish independent jobs, report failures, then resume only failed jobs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            jobs = [
                {
                    "name": name,
                    "kind": "tests",
                    "cwd": str(root),
                    "arguments": [
                        "-c",
                        "from pathlib import Path; print('attempt'); "
                        f"raise SystemExit(0 if Path('ready').exists() else {code})",
                    ],
                }
                for name, code in (("first", 1), ("middle", 0), ("last", 2))
            ]
            state = {"signature": {"policy": workflow.WORKFLOW_POLICY, "sources": {str(root): {}}}, "completed": {}}
            with patch.object(workflow, "source_signature", return_value={}):
                self.assertEqual(workflow.run_jobs(root, jobs, state, clean=False), 1)
                self.assertEqual(set(state["completed"]), {"middle"})
                self.assertEqual(state["status"], "COMPLETED_WITH_FAILURES")
                self.assertEqual(state["failed_jobs"], ["first", "last"])
                old_logs = {Path(row["log"]): row["log_sha256"] for row in state["failures"]}
                self.assertEqual(len(old_logs), 2)
                middle_result = state["completed"]["middle"].copy()
                summary = (root / "summary.md").read_text()
                for name in ("first", "middle", "last"):
                    self.assertIn(name, summary)
                self.assertIn("logs/first.001.log", summary)
                self.assertIn("logs/last.001.log", summary)
                self.assertEqual(json.loads((root / "progress.json").read_text())["failed_jobs"], ["first", "last"])
                workflow.validate_resume(state, state["signature"])
                (root / "ready").touch()
                self.assertEqual(workflow.run_jobs(root, jobs, state, clean=False), 0)
            self.assertEqual(state["status"], "COMPLETED")
            self.assertEqual(state["failed_jobs"], [])
            self.assertEqual(len(state["failures"]), 2)  # Keep resolved failure history.
            self.assertEqual(state["completed"]["middle"], middle_result)
            self.assertEqual(len(list((root / "logs").iterdir())), 5)
            for log, expected in old_logs.items():
                self.assertEqual(workflow.digest(log), expected)
            self.assertNotIn("COMPLETED_WITH_FAILURES", (root / "summary.md").read_text())

    def test_invalid_job_output_continues(self):
        """Treat missing or malformed per-job results as failures and keep running."""
        for contents in (None, "invalid json"):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "logs").mkdir()
                script = (
                    "import sys; from pathlib import Path; "
                    "out=Path(sys.argv[-1]); out.mkdir(parents=True); "
                    "(out/'evidence-index.json').write_text('{}'); "
                )
                if contents is not None:
                    script += f"(out/'compact.json').write_text({contents!r})"
                jobs = [
                    {"name": "bad", "kind": "tet", "cwd": str(root), "arguments": ["-c", script]},
                    {"name": "next", "kind": "tests", "cwd": str(root), "arguments": ["-c", "print('next')"]},
                ]
                state = {"signature": {"sources": {str(root): {}}}, "completed": {}}
                with patch.object(workflow, "source_signature", return_value={}):
                    self.assertEqual(workflow.run_jobs(root, jobs, state, clean=False), 1)
                self.assertEqual(set(state["completed"]), {"next"})
                self.assertEqual(state["failed_jobs"], ["bad"])
                self.assertEqual(state["failures"][0]["returncode"], 0)
                self.assertIn("error", state["failures"][0])
                self.assertTrue((root / "runs/bad/evidence-index.json").exists())

    def test_failure_does_not_hide_source_or_launch_errors(self):
        """Stop on frozen-source changes and host launch errors, even after a failed job."""
        for problem in ("source", "launch"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "logs").mkdir()
                jobs = [
                    {"name": name, "kind": "tests", "cwd": str(root), "arguments": []}
                    for name in ("first", "second", "third")
                ]
                state = {"signature": {"sources": {str(root): {}}}, "completed": {}}
                with (
                    patch.object(
                        workflow,
                        "source_signature",
                        side_effect=[{}, {"changed": True}] if problem == "source" else None,
                        return_value={},
                    ),
                    patch.object(workflow.subprocess, "Popen") as start,
                ):
                    child = start.return_value
                    child.pid = 12345
                    child.wait.return_value = 1
                    if problem == "launch":
                        start.side_effect = [child, OSError("cannot launch")]
                    with self.assertRaises(ValueError if problem == "source" else OSError):
                        workflow.run_jobs(root, jobs, state, clean=False)
                    self.assertEqual(start.call_count, 1 if problem == "source" else 2)
                self.assertEqual(len(state["failures"]), 1)

    def test_failed_jobs_and_open_gates_remain_distinct(self):
        """Keep unmet budgets after a failed job recovers; successful execution is not admission."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            screen = {
                "screen_passed": True,
                "pcg": workflow.pcg_budget([{"pcg_calls": [{"iterations": 200}]}]),
                "solver_seconds": 1.0,
            }
            state = {"signature": {"sources": {str(root): {}}}, "completed": {"budget": {"screen": screen}}}
            job = {
                "name": "retry",
                "kind": "tests",
                "cwd": str(root),
                "arguments": ["-c", "from pathlib import Path; raise SystemExit(0 if Path('ready').exists() else 1)"],
            }
            with patch.object(workflow, "source_signature", return_value={}):
                self.assertEqual(workflow.run_jobs(root, [job], state, clean=False), 1)
                self.assertEqual(state["status"], "COMPLETED_WITH_FAILURES")
                self.assertEqual(state["open_gates"], ["budget"])
                (root / "ready").touch()
                self.assertEqual(workflow.run_jobs(root, [job], state, clean=False), 2)
            self.assertEqual(state["status"], "COMPLETED_WITH_OPEN_GATES")
            self.assertEqual(state["failed_jobs"], [])
            self.assertEqual(state["open_gates"], ["budget"])

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
                    workflow.run_jobs(root, [job, {**job, "name": "later"}], state, clean=False)
                start.assert_called_once()
                stop.assert_called_once_with(12345, signal.SIGTERM)
            self.assertEqual(json.loads((root / "progress.json").read_text())["status"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
