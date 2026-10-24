# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import saved monolithic states by field name for diagnostic continuation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton._src.solvers.monolithic.material_inputs import _cable_geometry
from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver
from scripts.monolithic_reference.cable_benchmarks import CableBenchmark
from scripts.monolithic_reference.import_failure_state import (
    _project_cable_frames,
    check_scene,
    import_state,
    read_source,
)


def saved(run, state):
    """Return a complete saved state in the format the runners write, and a checkpoint restore."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.npz"
        run.save_checkpoint(state, path)
        restored = solver(run.model)
        restored_state = restored.create_state(dt=0.001)
        restored.load_checkpoint(restored_state, path)
        return read_source(path)[0], (restored, restored_state)


class TestFailureImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = scene("cpu")
        cls.original = solver(cls.model)
        cls.state = cls.original.create_state(dt=0.001)
        for _ in range(3):
            cls.original.step(cls.state, cls.state, cls.model.control(), dt=0.001)
        cls.source, cls.restored = saved(cls.original, cls.state)

    def test_import_continues_bitwise(self):
        """Import every physical field into a fresh solver and continue exactly like a checkpoint restore."""
        other = solver(self.model)
        state = other.create_state(dt=0.001)
        initial = other._states.copy(state)
        self.assertEqual(check_scene(other, initial, self.source, None), ["actor_layout"])
        report = import_state(other, state, self.source)
        self.assertEqual(report["converted"], [])
        expected = snapshot(self.state)
        actual = snapshot(state)
        for name in ("particle_q", "joint_q", "history.bound_x", "history.joint_v", "history.force_generation"):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        restored, reference = self.restored
        restored.step(reference, reference, self.model.control(), dt=0.001)
        other.step(state, state, self.model.control(), dt=0.001)
        for name, value in snapshot(reference).items():
            np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)

    def test_import_converts_precision(self):
        """Convert an FP64 state to an FP32 solver and keep its own identities."""
        other = solver(self.model, precision="fp32")
        state = other.create_state(dt=0.001)
        identity = snapshot(state)["history.config_digest"]
        report = import_state(other, state, self.source)
        self.assertIn("particle_q", report["converted"])
        np.testing.assert_array_equal(
            state.particle_q.numpy(), self.source["particle_q"].astype(np.float32), err_msg="particle_q"
        )
        np.testing.assert_array_equal(snapshot(state)["history.config_digest"], identity)
        self.assertEqual(int(snapshot(state)["history.precision"][0]), 32)
        other.step(state, state, self.model.control(), dt=0.001)
        self.assertFalse(other.last_stats.rolled_back, other.last_stats)

    def test_converted_cable_frames(self):
        """Re-project FP64 cable frames onto FP32-rounded short edges so the converted state stays valid."""
        case = CableBenchmark("cantilever", segments=256, precision="fp32")
        states = case.solver._states
        scratch = states.copy(case.state)
        s = np.linspace(0.0, 1.0, 257)
        x = np.c_[s, 0.3 * np.cos(4 * s), 0.3 * np.sin(4 * s)]
        tangent = np.diff(x, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        director = np.cross(tangent, [1.0, 0.0, 0.0])
        director /= np.linalg.norm(director, axis=1)[:, None]
        twist = _cable_geometry(x, director)[3]
        history = scratch._monolithic_coupled
        scratch.particle_q.assign(x.astype(np.float32))
        history.bound_x.assign(x.astype(np.float32))
        for prefix in ("cable_", "cable_bound_"):
            getattr(history, prefix + "ref_director").assign(director.astype(np.float32))
            getattr(history, prefix + "ref_twist").assign(twist.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "directors"):
            states.validate(scratch)
        self.assertEqual(_project_cable_frames(states, scratch), 256)
        states.validate(scratch)
        self.assertLess(np.abs(history.cable_ref_director.numpy() - director).max(), 1e-4)

    def test_rejections(self):
        """Reject changed shapes, unknown or missing fields and states of another scene."""
        other = solver(self.model)
        state = other.create_state(dt=0.001)
        before = snapshot(state)
        changes = {
            "shape": {**self.source, "particle_q": self.source["particle_q"][:-1]},
            "unknown": {**self.source, "history.retired": np.zeros(1)},
            "missing": {k: v for k, v in self.source.items() if k != "history.bound_x"},
        }
        for name, source in changes.items():
            with self.subTest(name), self.assertRaises(ValueError):
                import_state(other, state, source)
        for name, value in before.items():
            np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)
        contact_free = {k: np.zeros_like(v) if k.startswith("history.contact_") else v for k, v in self.source.items()}
        del contact_free["history.contact_body_force"]
        self.assertEqual(import_state(other, state, contact_free)["zero_filled"], ["history.contact_body_force"])
        with self.assertRaisesRegex(ValueError, "lacks history.contact_body_force"):
            import_state(other, state, {k: v for k, v in self.source.items() if k != "history.contact_body_force"})
        initial = other._states.copy(state)
        layout = bytearray(self.source["history.actor_layout"].tobytes())
        layout[-2] ^= 1
        with self.assertRaisesRegex(ValueError, "actor layout"):
            check_scene(other, initial, {**self.source, "history.actor_layout": np.frombuffer(layout, np.uint8)}, None)
        bare = {k: v for k, v in self.source.items() if k != "history.actor_layout"}
        with self.assertRaisesRegex(ValueError, "no layout"):
            check_scene(other, initial, bare, None)

    def test_recordings(self):
        """Read one step from both recording formats and validate their sidecars."""
        frames = [self.source, {**self.source, "particle_q": self.source["particle_q"] + 1}]
        stacked = {k: np.stack([f[k] for f in frames]) for k in self.source}
        other = solver(self.model)
        initial = other._states.copy(other.create_state(dt=0.001))
        with tempfile.TemporaryDirectory() as folder:
            native, example = Path(folder) / "native", Path(folder) / "example"
            for path in (native, example):
                path.mkdir()
                np.savez(path / "initial.npz", **snapshot(initial))
            np.savez(native / "states-0000.npz", **stacked)
            digest = hashlib.sha256((native / "states-0000.npz").read_bytes()).hexdigest()
            (native / "receipt.json").write_text(
                json.dumps({"chunks": [{"file": "states-0000.npz", "frames": 2, "sha256": digest}]})
            )
            np.savez(example / "frames.npz", **stacked, frame_steps=np.array([7, 8]))
            digest = hashlib.sha256((example / "frames.npz").read_bytes()).hexdigest()
            (example / "trajectory-index.json").write_text(
                json.dumps({"chunks": [{"path": "frames.npz", "count": 2, "sha256": digest}]})
            )
            for path, step in ((native, 2), (example, 8)):
                with self.subTest(path.name):
                    values, directory = read_source(path, step)
                    np.testing.assert_array_equal(values["particle_q"], frames[1]["particle_q"])
                    self.assertNotIn("frame_steps", values)
                    self.assertEqual(check_scene(other, initial, values, directory), ["actor_layout", "initial_state"])
                    with self.assertRaisesRegex(ValueError, "lacks step"):
                        read_source(path, 3 if path is native else 9)
            np.testing.assert_array_equal(read_source(native, 0)[0]["particle_q"], initial.particle_q.numpy())
            moved = snapshot(initial)
            moved["particle_q"] = moved["particle_q"] + 1
            np.savez(native / "initial.npz", **moved)
            with self.assertRaisesRegex(ValueError, "initial state"):
                check_scene(other, initial, read_source(native, 1)[0], native)


if __name__ == "__main__":
    unittest.main()
