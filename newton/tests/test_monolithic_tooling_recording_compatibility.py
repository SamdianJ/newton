# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit legacy FP64 recording migration without relaxing physical comparison."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.check_recording import compare, sha


class TestPrecisionRecording(unittest.TestCase):
    def test_legacy_schema(self):
        """Accept only validated FP64 metadata migration and compare every physical field."""
        before = {
            "frame_steps": np.arange(2),
            "joint_q": np.zeros((2, 1), np.float64),
            "history.schema_version": np.full((2, 1), 2, np.int32),
        }
        after = {
            **before,
            "history.schema_version": np.full((2, 1), 3, np.int32),
            "history.precision": np.full((2, 1), 64, np.int32),
        }
        with patch(
            "scripts.monolithic_reference.check_recording.frames",
            side_effect=lambda p: iter([before if p == "old" else after]),
        ):
            with self.assertRaises(ValueError):
                compare("old", "new")
            result = compare("old", "new", legacy_fp64=True)
            self.assertEqual(result["status"], "PASS_EXACT")
            self.assertEqual(result["schema_migration"], "2 -> 3 (FP64)")
            after["joint_q"] = np.ones((2, 1), np.float64)
            self.assertEqual(compare("old", "new", legacy_fp64=True)["status"], "NOT_READY_TOLERANCE")
            for field, value in [("history.precision", 32), ("history.schema_version", 4)]:
                saved = after[field].copy()
                after[field].fill(value)
                with self.assertRaises(ValueError):
                    compare("old", "new", legacy_fp64=True)
                after[field] = saved
            after["unexpected"] = np.zeros((2, 1))
            with self.assertRaises(ValueError):
                compare("old", "new", legacy_fp64=True)


class TestActorRecording(unittest.TestCase):
    @staticmethod
    def write(directory, arrays):
        np.savez_compressed(directory / "initial.npz", **arrays)
        data = {name: value[None] for name, value in arrays.items()}
        data.update(
            frame_steps=np.array([0]), frame_times=np.array([0.0]), frame_publication=np.array(["INITIAL"], dtype="U16")
        )
        path = directory / "frames.npz"
        np.savez_compressed(path, **data)
        (directory / "trajectory-index.json").write_text(
            json.dumps(
                {
                    "chunks": [{"path": "frames.npz", "count": 1, "sha256": sha(path)}],
                    "total_frames": 1,
                    "includes_initial": True,
                }
            )
        )

    def test_actor_bridge(self):
        """Audit complete Model/layout metadata and preserve physical comparison gates."""
        import warp as wp  # noqa: PLC0415

        from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver  # noqa: PLC0415

        model = scene("cpu")
        run = solver(model)
        current = snapshot(run.create_state(dt=0.001))
        extra = {
            name
            for name in current
            if name.startswith("history.rigid_")
            or name in {"history.layout_version", "history.layout_digest", "history.actor_layout"}
        }
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "old", Path(folder) / "new"
            left.mkdir()
            right.mkdir()
            model_arrays = {name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)}
            for path in (left, right):
                np.savez_compressed(path / "model.npz", **model_arrays)
            for version in (2, 3):
                old = {name: value.copy() for name, value in current.items() if name not in extra}
                old["history.schema_version"][:] = version
                if version == 2:
                    del old["history.precision"]
                self.write(left, old)
                self.write(right, current)
                with self.assertRaises(ValueError):
                    compare(left, right)
                result = compare(left, right, actor_layout_migration=True)
                self.assertEqual(result["status"], "PASS_EXACT")
                self.assertEqual(result["schema_migration"], f"{version} -> 4 (canonical A+T, FP64)")
                relocated = {name: value.copy() for name, value in model_arrays.items()}
                relocated["shape_source_ptr"][:] = 123
                np.savez_compressed(right / "model.npz", **relocated)
                self.assertEqual(compare(left, right, actor_layout_migration=True)["status"], "PASS_EXACT")
                np.savez_compressed(right / "model.npz", **model_arrays)

                for field in ("particle_colors", "bvh_shape_enabled"):
                    auxiliary = {name: value.copy() for name, value in model_arrays.items()}
                    # These caches are not consumed by Monolithic physics.
                    auxiliary[field] = np.arange(4, dtype=np.int32)
                    np.savez_compressed(left / "model.npz", **auxiliary)
                    auxiliary[field] = auxiliary[field][::-1].copy()
                    np.savez_compressed(right / "model.npz", **auxiliary)
                    self.assertEqual(compare(left, right, actor_layout_migration=True)["status"], "PASS_EXACT")
                    for invalid in (np.zeros(3, np.int32), np.zeros(4, np.float32)):
                        auxiliary[field] = invalid
                        np.savez_compressed(right / "model.npz", **auxiliary)
                        with self.assertRaisesRegex(ValueError, "Model layouts differ"):
                            compare(left, right, actor_layout_migration=True)
                    del auxiliary[field]
                    np.savez_compressed(right / "model.npz", **auxiliary)
                    with self.assertRaisesRegex(ValueError, "Model schemas differ"):
                        compare(left, right, actor_layout_migration=True)
                    for path in (left, right):
                        np.savez_compressed(path / "model.npz", **model_arrays)

                changed = {name: value.copy() for name, value in current.items()}
                changed["particle_q"][0, 0] += 1e-5
                self.write(right, changed)
                self.assertEqual(compare(left, right, actor_layout_migration=True)["status"], "NOT_READY_TOLERANCE")
                for field in (
                    "history.actor_layout",
                    "history.layout_digest",
                    "history.layout_version",
                    "history.precision",
                    "history.config_digest",
                ):
                    changed = {name: value.copy() for name, value in current.items()}
                    changed[field].flat[0] ^= 1
                    self.write(right, changed)
                    with self.assertRaises(ValueError):
                        compare(left, right, actor_layout_migration=True)
                for field in ("history.rigid_com", "unexpected"):
                    changed = {name: value.copy() for name, value in current.items()}
                    changed[field] = np.zeros((1, 3))
                    self.write(right, changed)
                    with self.assertRaises(ValueError):
                        compare(left, right, actor_layout_migration=True)
                self.write(right, current)
                changed_model = {name: value.copy() for name, value in model_arrays.items()}
                changed_model["body_mass"][0] += 1
                np.savez_compressed(right / "model.npz", **changed_model)
                with self.assertRaises(ValueError):
                    compare(left, right, actor_layout_migration=True)
                np.savez_compressed(right / "model.npz", **model_arrays)

    def test_native_actor_bridge(self):
        """Authenticate native references without Model archives using their original digest."""
        import warp as wp  # noqa: PLC0415

        from newton.tests.test_monolithic_solver_coupled import scene, snapshot, solver  # noqa: PLC0415

        model = scene("cpu")
        current = snapshot(solver(model).create_state(dt=0.001))
        old = {
            name: value.copy()
            for name, value in current.items()
            if not name.startswith("history.rigid_")
            and name not in {"history.layout_version", "history.layout_digest", "history.actor_layout"}
        }
        old["history.schema_version"][:] = 3
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "native", Path(folder) / "new"
            left.mkdir()
            right.mkdir()
            controls = Path(folder) / "controls.npy"
            np.save(controls, np.zeros((1, 2, model.joint_dof_count), dtype=np.float32))
            np.savez_compressed(left / "initial.npz", **old)
            with gzip.open(left / "steps.jsonl.gz", "wt"):
                pass
            receipt = {
                "status": "COMPLETE_NOT_AUDITED",
                "published_steps": 0,
                "case": {"precision": "fp64", "dt": 0.001, "steps": 0},
                "counts": {"CONVERGED": 0, "STOPPED": 0, "HARD_FAILURE": 0},
                "chunks": [],
                "controls_sha256": sha(controls),
                "config_digest": old["history.config_digest"].tobytes().hex(),
            }
            (left / "receipt.json").write_text(json.dumps(receipt))
            arrays = {name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)}
            np.savez_compressed(right / "model.npz", **arrays)
            current.update(
                frame_target_q=np.zeros(model.joint_dof_count, dtype=np.float32),
                frame_target_qd=np.zeros(model.joint_dof_count, dtype=np.float32),
            )
            self.write(right, current)
            self.assertEqual(
                compare(left, right, native_controls=controls, actor_layout_migration=True)["status"], "PASS_EXACT"
            )
            receipt["config_digest"] = "0" * 64
            (left / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Native reference physical identity"):
                compare(left, right, native_controls=controls, actor_layout_migration=True)


if __name__ == "__main__":
    unittest.main()
