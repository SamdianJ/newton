# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise computational replay, not visual trajectory playback."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_recording import state_arrays
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.profiling import ROOT
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.replay_workload import replay
from scripts.monolithic_reference.workload_io import load_arrays, open_bundle, save_arrays
from scripts.monolithic_reference.workload_pcg import replay_pcg, validate_pcg
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays


def test_window_and_pcg(test, device):
    """Restore step five in a fresh process and replay PCG five times on CPU/CUDA."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "bundle"
        manifest = record(root, device=device, graph=device.is_cuda, windows=[(1, 1), (6, 10)])
        # Recording must not perturb the physical computation or force history.
        model, run, state, _ = build_scene("tiny-contact", device=device, graph=device.is_cuda)
        control = model.control()
        for _ in range(10):
            run.step(state, state, control, dt=0.001)
            test.assertTrue(run.last_stats.converged)
        compare_arrays(
            state_arrays(state), load_arrays(root / "state-10.npz", manifest["array_schemas"]["state-10.npz"])
        )
        first = load_arrays(root / "state-1.npz", manifest["array_schemas"]["state-1.npz"])
        test.assertGreater(np.linalg.norm(first["history.contact_particle_force"]), 0)
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
                    "5",
                    "--output",
                    str(Path(temporary) / mode),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = root / manifest["snapshots"][0]
        metadata = json.loads((path / "call.json").read_text())
        arrays = load_arrays(path / "input.npz", metadata["input_schema"])
        output = load_arrays(path / "output.npz", metadata["output_schema"])
        damaged = {k: v.copy() for k, v in arrays.items()}
        damaged["columns"][0] = len(damaged["rhs_hat"])
        with test.assertRaises(ValueError):
            validate_pcg(damaged)
        damaged = {k: v.copy() for k, v in arrays.items()}
        damaged["rhs_hat"] += 1.0
        with test.assertRaises((AssertionError, ValueError)):
            replay_pcg(damaged, metadata, output, metadata["result"], device=device, repeat=1)
        # A zero warm start must overwrite an arbitrary initial guess.
        arrays["y"].fill(7.0)
        test.assertTrue(metadata["zero_initial_guess"])
        test.assertEqual(
            replay_pcg(arrays, metadata, output, metadata["result"], device=device, repeat=5)["status"], "PASS"
        )
        (root / "inputs-6.npz").write_bytes(b"corrupt")
        with test.assertRaises(ValueError):
            replay(root, device=device)


class TestWorkload(unittest.TestCase):
    def test_schema_rejection(self):
        """Reject incomplete bundles, object data and shape/dtype drift."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text('{"schema":"monolithic-workload/1","status":"WRITING"}')
            with self.assertRaises(ValueError):
                open_bundle(root)
            with self.assertRaises(ValueError):
                save_arrays(root / "objects.npz", {"bad": np.array([object()], dtype=object)})
            schema = save_arrays(root / "array.npz", {"x": np.ones(2, dtype=np.float32)})
            np.savez(root / "array.npz", x=np.ones(2, dtype=np.float64))
            with self.assertRaises(ValueError):
                load_arrays(root / "array.npz", schema)

    def test_window_validation(self):
        """Reject invalid window ranges before constructing a scene."""
        for windows in ([], [(0, 1)], [(2, 1)], [(1, 11)], [(1, 3), (3, 4)]):
            with self.assertRaises(ValueError):
                record("must-not-be-created", windows=windows)


add_function_test(TestWorkload, "test_window_and_pcg", test_window_and_pcg, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
