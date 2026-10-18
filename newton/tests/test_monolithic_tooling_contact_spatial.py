# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Restore frozen spatial candidates at the existing workload boundaries."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.benchmark_workload import benchmark
from scripts.monolithic_reference.profiling import ROOT, capture_count
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.workload_io import load_arrays
from scripts.monolithic_reference.workload_objective import replay_objective
from scripts.monolithic_reference.workload_scene import build_scene


def test_spatial_workload(test, device):
    """Replay one small recorded window and frozen reference/current/assembly calls in fresh processes."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "bundle"
        manifest = record(
            root,
            scene="actor-coupling-permuted",
            device=device,
            graph=device.is_cuda,
            contact_dataflow="spatial",
            spatial={"candidate_capacity": 4096},
            steps=2,
            windows=[(1, 1)],
            boundaries=("objective", "assembly"),
            call_ordinals={"objective": [0, 1, 2], "assembly": [0]},
        )
        for mode in ("window", "hotspot"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "scripts.monolithic_reference.replay_workload",
                    "--bundle",
                    str(root),
                    "--mode",
                    mode,
                    "--device",
                    str(device),
                    "--repeat",
                    "1",
                    "--output",
                    str(Path(directory) / mode),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        test.assertEqual(benchmark(root, device=device, warmup=1, repeat=1)["status"], "PASS")
        for name in manifest["snapshots"]:
            path = root / name
            metadata = json.loads((path / "call.json").read_text())
            arrays = load_arrays(path / "input.npz", metadata["input_schema"])
            expected = load_arrays(path / "output.npz", metadata["output_schema"])
            test.assertIn("contact.spatial.identities", arrays)
            test.assertIn("contact.spatial.q0", arrays)
            missing = {k: v for k, v in arrays.items() if k != "contact.spatial.count"}
            with test.assertRaises(ValueError):
                replay_objective(
                    missing, metadata, expected, scene=manifest["scene"], device=device, graph=device.is_cuda, repeat=1
                )
            damaged = copy.deepcopy(metadata)
            damaged["spatial"]["selection_ready"] = 1
            with test.assertRaises(ValueError):
                replay_objective(
                    arrays, damaged, expected, scene=manifest["scene"], device=device, graph=device.is_cuda, repeat=1
                )
        _, run, _, _ = build_scene(
            "tiny-contact",
            device=device,
            graph=device.is_cuda,
            contact_dataflow="spatial",
            spatial={"candidate_capacity": 256},
        )
        before = capture_count(run)
        run._contact.query_graphs.captures["probe"] = 3
        run._contact._batch.reference_rows.graphs.captures["probe"] = 2
        test.assertEqual(capture_count(run), before + 5)
    with test.assertRaises(ValueError):
        build_scene("tiny-contact", device=device, contact_dataflow="spatial")


class TestContactSpatialWorkload(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestContactSpatialWorkload, "test_spatial_workload", test_spatial_workload, devices=[_device])


if __name__ == "__main__":
    unittest.main()
