# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Authenticate local direction replay and preserve independent equation audits."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from newton.examples.softbody.monolithic_recording import state_arrays
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, solver
from scripts.monolithic_reference.audit_precision import posterior_error
from scripts.monolithic_reference.check_recording import sha
from scripts.monolithic_reference.replay_direction import recorded_frame, restore


class TestDirectionTooling(unittest.TestCase):
    """Keep replay identity checks and offline mathematics independent of PCG."""

    def test_frame_hash_and_step(self):
        """Reject changed chunks, escaped paths and missing or duplicate frame indices."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "frames.npz"
            np.savez(payload, frame_steps=np.array([0, 1]), value=np.array([3.0, 4.0]))
            chunk = {"path": payload.name, "count": 2, "sha256": sha(payload)}
            (root / "trajectory-index.json").write_text(json.dumps({"chunks": [chunk]}))
            frame, _ = recorded_frame(root, 1)
            self.assertEqual(frame["value"], 4.0)
            with self.assertRaises(ValueError):
                recorded_frame(root, 2)
            np.savez(payload, frame_steps=np.array([0, 1]), value=np.array([3.0, 5.0]))
            with self.assertRaises(ValueError):
                recorded_frame(root, 1)
            np.savez(payload, frame_steps=np.array([1, 1]), value=np.array([3.0, 4.0]))
            chunk["sha256"] = sha(payload)
            (root / "trajectory-index.json").write_text(json.dumps({"chunks": [chunk]}))
            with self.assertRaises(ValueError):
                recorded_frame(root, 1)
            chunk["path"] = "../frames.npz"
            (root / "trajectory-index.json").write_text(json.dumps({"chunks": [chunk]}))
            with self.assertRaises(ValueError):
                recorded_frame(root, 1)

    def test_policy_rejection_is_atomic(self):
        """Reject mismatched policy and incomplete native state before any copy."""
        run = solver(scene("cpu"))
        state = run.create_state(dt=0.001)
        before = state_arrays(state)
        case = SimpleNamespace(solver=run, state=state)
        restore(case, before)
        for field in ("history.config_digest", "history.layout_digest"):
            bad = {name: value.copy() for name, value in before.items()}
            bad[field][0] ^= 1
            bad["joint_q"].fill(100)
            with self.assertRaises(AssertionError):
                restore(case, bad)
            assert_snapshot(self, state, before)
        bad = dict(before)
        del bad["history.link_vsym"]
        with self.assertRaises(ValueError):
            restore(case, bad)
        assert_snapshot(self, state, before)
        bad = {name: value.copy() for name, value in before.items()}
        bad["particle_q"][0, 0] += 0.01
        with self.assertRaises(ValueError):
            restore(case, bad)
        assert_snapshot(self, state, before)

    def test_regularized_and_legacy_audit(self):
        """Audit rounded lambda and recovered delta while retaining legacy e0 semantics."""
        scale = np.array([0.1, 10.0], dtype=np.float32)
        delta = np.array([0.2, -0.3], dtype=np.float32)
        matrix = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        residual = np.array([-0.7, 0.1])
        data = {
            "offsets": np.array([0, 2, 4]),
            "columns": np.array([0, 1, 0, 1]),
            "values": matrix.ravel(),
            "scale": scale,
            "direction": delta,
        }
        expected = scale.astype(float) * (matrix.astype(float) @ delta.astype(float) + residual)
        result, original, term = posterior_error(data, residual)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(term, 0)
        data["actual_regularization"] = np.array([1.234567891], dtype=np.float32)
        result, original, term = posterior_error(data, residual)
        np.testing.assert_allclose(original, expected)
        np.testing.assert_allclose(term, float(np.float32(1.234567891)) * delta.astype(float) / scale.astype(float))
        np.testing.assert_allclose(result, expected + term)


if __name__ == "__main__":
    unittest.main()
