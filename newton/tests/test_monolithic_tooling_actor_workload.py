# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify complete mixed-actor computational snapshots and independent replay."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.profiling import ROOT
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.replay_workload import replay
from scripts.monolithic_reference.run_actor_coupling import build_model, make_solver
from scripts.monolithic_reference.workload_io import load_arrays
from scripts.monolithic_reference.workload_objective import arrays_for, replay_objective
from scripts.monolithic_reference.workload_pcg import replay_pcg


def test_actor_snapshots(test, device):
    """Recompute all mixed boundaries and reject missing native or mapping inputs."""
    for scene in ("actor-coupling", "actor-coupling-before", "actor-coupling-permuted"):
        for graph in (False, True) if device.is_cuda else (False,):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "bundle"
                manifest = record(
                    root,
                    scene=scene,
                    device=device,
                    graph=graph,
                    steps=10,
                    windows=[(6, 10)],
                    boundaries=("objective", "assembly", "pcg"),
                    call_ordinals={"objective": [0, 1, 2, 3], "assembly": [0], "pcg": [0]},
                )
                boundaries = set()
                for name in manifest["snapshots"]:
                    path = root / name
                    metadata = json.loads((path / "call.json").read_text())
                    arrays = load_arrays(path / "input.npz", metadata["input_schema"])
                    expected = load_arrays(path / "output.npz", metadata["output_schema"])
                    boundaries.add(metadata["boundary"])
                    if metadata["boundary"] == "production_pcg_core":
                        test.assertEqual(arrays["rigid_cholesky"].shape, (2, 6, 6))
                        test.assertEqual(len(arrays["limits"]), 7)
                        damaged = {k: v.copy() for k, v in arrays.items()}
                        damaged["rigid_rows"][0, 0] = len(arrays["rhs_hat"])
                        with test.assertRaises(ValueError):
                            replay_pcg(damaged, metadata, expected, metadata["result"], device=device, repeat=1)
                        damaged_meta = copy.deepcopy(metadata)
                        damaged_meta["source_hashes"] = {}
                        with test.assertRaises(ValueError):
                            replay_pcg(arrays, damaged_meta, expected, metadata["result"], device=device, repeat=1)
                        damaged = {k: v.copy() for k, v in arrays.items() if k != "rigid_cholesky"}
                        with test.assertRaises(ValueError):
                            replay_pcg(damaged, metadata, expected, metadata["result"], device=device, repeat=1)
                        continue
                    test.assertIn("_rigid.tn", arrays)
                    test.assertIn("actor_contact.start.y", arrays)
                    test.assertEqual(arrays["_original.history.rigid_com"].dtype, np.float64)
                    test.assertEqual(arrays["_accepted.history.rigid_rotation"].shape, (2, 4))
                    for key in ("_rigid.tn", "actor_contact.start.y", "_accepted.history.rigid_rotation"):
                        damaged = {k: v.copy() for k, v in arrays.items() if k != key}
                        with test.assertRaises(ValueError):
                            replay_objective(
                                damaged, metadata, expected, scene=scene, device=device, graph=graph, repeat=1
                            )
                    damaged = {k: v.copy() for k, v in arrays.items()}
                    damaged["actor_contact.selectors"][0, 0] = 99
                    with test.assertRaises(ValueError):
                        replay_objective(damaged, metadata, expected, scene=scene, device=device, graph=graph, repeat=1)
                    damaged_meta = copy.deepcopy(metadata)
                    damaged_meta["layout"]["identity"] = "foreign"
                    with test.assertRaises(ValueError):
                        replay_objective(
                            arrays, damaged_meta, expected, scene=scene, device=device, graph=graph, repeat=1
                        )
                    damaged = {k: v.copy() for k, v in arrays.items()}
                    damaged["_accepted.history.rigid_rotation"] = np.zeros((2, 7), dtype=np.float64)
                    with test.assertRaises(ValueError):
                        replay_objective(damaged, metadata, expected, scene=scene, device=device, graph=graph, repeat=1)
                test.assertEqual(boundaries, {"objective", "assembly", "production_pcg_core"})
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
                manifest["source"]["source_hashes"] = {}
                (root / "manifest.json").write_text(json.dumps(manifest))
                with test.assertRaises(ValueError):
                    replay(root, device=device)


class TestActorWorkload(unittest.TestCase):
    def test_native_owners(self):
        """Include rigid predictors and both contact frames without serializing handles."""
        run = make_solver(build_model(device="cpu"), mode="production", graph=False)
        state = run.create_state(dt=0.001)
        for name in ("_original", "_accepted", "_trial"):
            setattr(run, name, run._states.copy(state))
        arrays = arrays_for(run)
        self.assertIn("_rigid.predicted_r", arrays)
        self.assertIn("actor_contact.current.rotation", arrays)
        self.assertIn("_trial.history.rigid_vsym", arrays)
        self.assertFalse(any(name.endswith(".table") for name in arrays))


add_function_test(TestActorWorkload, "test_actor_snapshots", test_actor_snapshots, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
