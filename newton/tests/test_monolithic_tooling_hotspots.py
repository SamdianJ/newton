# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check frozen objective/assembly dependencies and independent recomputation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic import iteration
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.record_workload import CallRecorder, main, record
from scripts.monolithic_reference.replay_workload import acceptance_trace, replay
from scripts.monolithic_reference.summarize_workload import summarize
from scripts.monolithic_reference.workload_io import complete_bundle, digest, load_arrays, save_arrays, write_json
from scripts.monolithic_reference.workload_objective import compare_assembly, replay_objective


def test_hotspot_dependencies(test, device):
    """Replay initial, tangent and rejected-trial objectives plus sparse assembly."""
    for graph in (False, True) if device.is_cuda else (False,):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            acceptance = iteration.acceptance
            decisions = []

            def reject_first(*args, acceptance=acceptance, decisions=decisions, **kwargs):
                result = acceptance(*args, **kwargs)
                decisions.append(result)
                return (False, False, False, False) if len(decisions) == 1 else result

            with patch.object(iteration, "acceptance", reject_first):
                manifest = record(
                    root,
                    device=device,
                    graph=graph,
                    steps=3,
                    windows=[(1, 3)],
                    boundaries=("objective", "assembly"),
                    call_ordinals={"objective": list(range(16)), "assembly": [0]},
                )
            test.assertGreater(len(decisions), 1)
            report = replay(root, mode="hotspot", device=device, repeat=5)
            test.assertEqual(report["status"], "PASS")
            first_step = json.loads((root / "trace.jsonl").read_text().splitlines()[0])
            last_objective = max(c["ordinal"] for c in first_step["calls"] if c["boundary"] == "objective")
            final_name = f"snapshots/000001-objective-{last_objective}"
            test.assertIn(final_name, manifest["snapshots"])
            final = json.loads((root / final_name / "call.json").read_text())
            test.assertEqual(final["state"], "_accepted")
            test.assertTrue(final["build_tangent"])
            flags, states = set(), set()
            for name in manifest["snapshots"]:
                path = root / name
                metadata = json.loads((path / "call.json").read_text())
                arrays = load_arrays(path / "input.npz", metadata["input_schema"])
                output = load_arrays(path / "output.npz", metadata["output_schema"])
                if metadata["boundary"] == "objective":
                    flags.add(metadata["build_tangent"])
                    states.add(metadata["state"])
                else:
                    bad = {k: v.copy() for k, v in output.items()}
                    # Removing true A-T cross terms must be observable.
                    offsets, columns = bad["global.offsets"], bad["global.columns"]
                    nq = output["actor_dense"].shape[0]
                    active = False
                    for row in range(nq):
                        for i in range(offsets[row], offsets[row + 1]):
                            if columns[i] >= nq and bad["global.values"][i] != 0:
                                bad["global.values"][i] = 0
                                active = True
                    test.assertTrue(active)
                    with test.assertRaises(AssertionError):
                        replay_objective(
                            arrays, metadata, bad, scene="tiny-contact", device=device, graph=graph, repeat=1
                        )
                    damaged = {k: v.copy() for k, v in arrays.items()}
                    damaged["contact.pool.records.triplet_values"].fill(0)
                    with test.assertRaises(AssertionError):
                        replay_objective(
                            damaged, metadata, output, scene="tiny-contact", device=device, graph=graph, repeat=1
                        )
                    damaged = {k: v.copy() for k, v in arrays.items()}
                    damaged["contact.pool.records.triplet_values"].fill(np.nan)
                    with test.assertRaises(ValueError):
                        replay_objective(
                            damaged, metadata, output, scene="tiny-contact", device=device, graph=graph, repeat=1
                        )
                    damaged = {k: v.copy() for k, v in arrays.items()}
                    damaged["_contact.triangles"].fill(2**30)
                    with test.assertRaises(ValueError):
                        replay_objective(
                            damaged, metadata, output, scene="tiny-contact", device=device, graph=graph, repeat=1
                        )
            test.assertEqual(flags, {False, True})
            test.assertTrue({"_original", "_accepted", "_trial"} <= states)


class TestHotspots(unittest.TestCase):
    def test_reduction_roundoff(self):
        """Allow summation-order cancellation but reject errors beyond its bound."""

        def array(values):
            return SimpleNamespace(numpy=lambda: np.asarray(values))

        triplets = SimpleNamespace(
            count=array([3]),
            rows=array([0, 0, 0]),
            columns=array([0, 0, 0]),
            values=array(np.asarray([1e8, 1, -1e8], dtype=np.float32)),
            column_count=1,
        )
        run = SimpleNamespace(_linear=SimpleNamespace(_global=triplets, _internal=SimpleNamespace(count=array([0]))))
        actual = {
            "global.values": np.asarray([0], dtype=np.float32),
            "global.offsets": np.asarray([0, 1], dtype=np.int32),
            "global.columns": np.asarray([0], dtype=np.int32),
            "internal.values": np.empty((0, 3, 3), dtype=np.float32),
        }
        expected = {**actual, "global.values": np.asarray([1], dtype=np.float32)}
        self.assertEqual(compare_assembly(run, actual, expected)["global"]["raw_elementwise_mismatches"], 1)
        expected["global.values"][0] = 100
        with self.assertRaises(AssertionError):
            compare_assembly(run, actual, expected)

    def test_failed_call_trace(self):
        """A rejected objective remains observable and its failure survives comparison."""

        def invalid():
            raise ValueError("invalid candidate")

        owner = SimpleNamespace(evaluate=invalid)
        recorder = CallRecorder(None, None)
        try:
            recorder._observe(owner, "evaluate", "objective")
            with self.assertRaisesRegex(ValueError, "invalid candidate"):
                owner.evaluate()
        finally:
            recorder.patches.close()
        self.assertEqual(recorder.calls[0]["failure"], "ValueError")
        self.assertGreaterEqual(recorder.calls[0]["host_seconds_with_recording"], 0)
        self.assertIs(owner.evaluate, invalid)
        failed = acceptance_trace([{"accepted": False, "domain_failure": "invalid candidate"}])
        self.assertNotEqual(failed, acceptance_trace([{"accepted": False}]))

    def test_selection_validation(self):
        """Reject a mismatched scene selection before building the workload."""
        for ordinals in ({"objective": [-1]}, {"objective": [1.5]}, {"unknown": [0]}):
            with self.assertRaises(ValueError):
                CallRecorder(None, None, call_ordinals=ordinals)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            path.write_text(json.dumps({"scene": "sharpa-default", "windows": [[1, 5]]}))
            with (
                patch("sys.argv", ["record_workload", "--selection", str(path), "--output", temporary]),
                patch("scripts.monolithic_reference.record_workload.record") as capture,
                self.assertRaises(SystemExit),
            ):
                main()
            capture.assert_not_called()

    def test_summary_identity_and_scope(self):
        """Use observed host costs as denominator and reject a foreign bundle receipt."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stats = {
                "rolled_back": False,
                "converged": True,
                "active_sample_count": 2,
                "linear_iterations": 10,
                "nonlinear_iterations": 1,
            }
            row = {
                "step": 1,
                "stats": stats,
                "calls": [
                    {"boundary": "objective", "host_seconds_with_recording": 1.0},
                    {"boundary": "assembly", "host_seconds_with_recording": 3.0},
                ],
            }
            (root / "trace.jsonl").write_text(json.dumps(row) + "\n")
            schema = save_arrays(
                root / "state-1.npz",
                {"history.contact_particle_force": np.ones((1, 3)), "history.contact_body_force": np.zeros((1, 6))},
            )
            complete_bundle(
                root,
                {"scene": "tiny-contact", "steps": 1, "windows": [[1, 1]], "array_schemas": {"state-1.npz": schema}},
            )
            receipt = root / "replay.json"
            value = {"status": "PASS", "bundle_manifest_sha256": digest(root / "manifest.json"), "results": []}
            write_json(receipt, value)
            result = summarize(root, [receipt])
            self.assertEqual(result["host_modules"]["objective"]["share_of_observed"], 0.25)
            value["bundle_manifest_sha256"] = "foreign"
            write_json(receipt, value)
            with self.assertRaises(ValueError):
                summarize(root, [receipt])


add_function_test(TestHotspots, "test_hotspot_dependencies", test_hotspot_dependencies, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
