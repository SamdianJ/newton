# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-frame state snapshots for quick regression windows."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver
from scripts.monolithic_reference.state_snapshots import (
    check_snapshot_scene,
    import_state,
    load_snapshot,
    read_source,
    write_snapshot,
)


def _recording(folder, run, states):
    """Write a minimal example-format recording: model, initial state and one chunk of two frames."""
    folder = Path(folder)
    (folder / "trajectory").mkdir(parents=True)
    arrays = {name: value.numpy() for name, value in vars(run.model).items() if isinstance(value, wp.array)}
    np.savez_compressed(folder / "model.npz", **arrays)
    np.savez_compressed(folder / "initial.npz", **states[0])
    path = folder / "trajectory/frames-000001-000002.npz"
    frames = {name: np.stack([state[name] for state in states[1:]]) for name in states[0]}
    frames.update(
        frame_steps=np.array([1, 2]),
        frame_publication=np.array(["CONVERGED", "STOPPED_ACCEPTED"], dtype="U16"),
    )
    np.savez_compressed(path, **frames)
    chunk = {
        "path": "trajectory/frames-000001-000002.npz",
        "count": 2,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (folder / "trajectory-index.json").write_text(json.dumps({"chunks": [chunk]}))
    return folder


class TestSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = scene("cpu")
        cls.solver_run = solver(cls.model)
        state = cls.solver_run.create_state(dt=0.001)
        cls.states = [snapshot(state)]
        for _ in range(2):
            cls.solver_run.step(state, state, cls.model.control(), dt=0.001)
            cls.states.append(snapshot(state))

    def test_round_trip_and_import(self):
        """A snapshot of step 2 imports into a fresh solver exactly like the recorded frame."""
        with tempfile.TemporaryDirectory() as folder:
            recording = _recording(Path(folder) / "rec", self.solver_run, self.states)
            meta = write_snapshot(recording, 2, Path(folder) / "snap", source_commit="abc")
            self.assertEqual((meta["step"], meta["publication"]), (2, "STOPPED_ACCEPTED"))
            self.assertEqual(json.loads((Path(folder) / "snap.json").read_text()), meta)
            values, model_values = load_snapshot(Path(folder) / "snap.npz", meta["sha256"])
            self.assertEqual(sorted(model_values), meta["model_arrays"])
            other = solver(self.model)
            state = other.create_state(dt=0.001)
            self.assertEqual(check_snapshot_scene(other, model_values), meta["model_arrays"])
            import_state(other, state, values)
            frame, _ = read_source(recording, 2)
            for name in ("particle_q", "joint_q", "history.bound_x"):
                np.testing.assert_array_equal(snapshot(state)[name], frame[name], err_msg=name)

    def test_rejections(self):
        """Reject a changed file, a missing step and a different scene."""
        with tempfile.TemporaryDirectory() as folder:
            recording = _recording(Path(folder) / "rec", self.solver_run, self.states)
            meta = write_snapshot(recording, 1, Path(folder) / "snap", source_commit="abc")
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_snapshot(Path(folder) / "snap.npz", "0" * 64)
            with self.assertRaisesRegex(ValueError, "lacks step"):
                write_snapshot(recording, 7, Path(folder) / "other", source_commit="abc")
            _, model_values = load_snapshot(Path(folder) / "snap.npz", meta["sha256"])
            changed = {**model_values, "particle_mass": model_values["particle_mass"] * 2}
            with self.assertRaisesRegex(ValueError, "particle_mass"):
                check_snapshot_scene(solver(self.model), changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
