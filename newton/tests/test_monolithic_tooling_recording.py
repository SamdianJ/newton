# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate recording comparison, complete state data and runtime provenance."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference import check_recording as recording


class TestContactFields(unittest.TestCase):
    def test_forces_match_publication_in_both_precisions(self):
        """Split committed contact forces by target in FP64 and FP32 without the FP64 audit."""
        from newton.examples.softbody.monolithic_recording import contact_fields, measure  # noqa: PLC0415
        from newton.tests.test_monolithic_solver_coupled import scene, solver  # noqa: PLC0415

        model = scene("cpu")
        for precision in ("fp64", "fp32"):
            with self.subTest(precision=precision):
                run = solver(model, precision=precision)
                state = run.create_state(dt=0.001)
                for _ in range(3):
                    run.step(state, state, model.control(), dt=0.001)
                fields = contact_fields(run, state)
                published = state._monolithic_coupled.contact_particle_force.numpy().astype(float).sum(axis=0)
                total = fields["ground_force"] + fields["robot_force"]
                self.assertGreater(np.linalg.norm(published), 0)
                tolerance = 1e-10 if precision == "fp64" else 1e-5
                np.testing.assert_allclose(total, published, rtol=tolerance, atol=tolerance * np.linalg.norm(published))
                self.assertTrue(np.isfinite(fields["penetration"]) and fields["penetration"] >= 0)
                mask = measure(run, state)["mask"]
                self.assertEqual(mask["max_all_penetration"], fields["penetration"])
                self.assertLessEqual(mask["max_masked_penetration"], mask["max_all_penetration"])
                self.assertEqual(mask["masked_active_count"], len(mask["masked_active_keys"]))


class TestRecordingValidation(unittest.TestCase):
    def test_comparison_envelope(self):
        """Bound floating differences without relaxing metadata or layout equality."""
        reference = {
            "frame_steps": np.array([0, 1]),
            "frame_publication": np.array(["INITIAL", "CONVERGED"], dtype="U32"),
            "particle_q": np.array([[0.0], [1.0]]),
        }

        def compare(actual, limits=None):
            with patch.object(recording, "frames", side_effect=[[reference], [actual]]):
                return recording.compare(Path("reference"), Path("actual"), limits)["status"]

        self.assertEqual(compare(reference), "PASS_EXACT")
        changed = dict(reference, particle_q=np.array([[0.0], [1.125]]))
        self.assertEqual(compare(changed), "NOT_READY_TOLERANCE")
        self.assertEqual(compare(changed, {"particle_q": 0.25}), "PASS_FROZEN_ENVELOPE")
        self.assertEqual(compare(changed, {"particle_q": 0.0625}), "NOT_READY_TOLERANCE")
        changed = dict(reference, frame_publication=np.array(["INITIAL", "STOPPED_ACCEPTED"], dtype="U32"))
        self.assertEqual(compare(changed, {"frame_publication": 100}), "NOT_READY_TOLERANCE")
        changed = dict(reference, frame_steps=np.array([0, 2]))
        self.assertEqual(compare(changed, {"frame_steps": 100}), "NOT_READY_TOLERANCE")
        changed = dict(reference, particle_q=reference["particle_q"].astype(np.float32))
        with self.assertRaisesRegex(ValueError, "layouts differ"):
            compare(changed)

    def test_recording_integrity_gates(self):
        """Reject tampered chunks, missing history and wrong force generations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                name: np.zeros((2, 1))
                for name in ("joint_q", "joint_qd", "particle_q", "particle_qd", "history.link_vsym")
            }
            values.update(
                {"history.force_generation": np.array([[0], [1]]), "history.force_valid": np.array([[0], [1]])}
            )
            np.savez(root / "initial.npz", **{name: a[0] for name, a in values.items()})
            np.savez(root / "final.npz", **{name: a[-1] for name, a in values.items()})
            values.update(
                frame_steps=np.array([0, 1]),
                frame_times=np.array([0, 0.001]),
                frame_publication=np.array(["INITIAL", "CONVERGED"]),
            )
            path = root / "frames.npz"

            def write(arrays):
                np.savez(path, **arrays)
                (root / "trajectory-index.json").write_text(
                    json.dumps(
                        {
                            "chunks": [{"path": path.name, "count": 2, "sha256": recording.sha(path)}],
                            "includes_initial": True,
                            "total_frames": 2,
                        }
                    )
                )

            write(values)
            with gzip.open(root / "steps.jsonl.gz", "wt") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": 1,
                            "time": 0.001,
                            "publication": "CONVERGED",
                            "force_valid": 1,
                            "force_generation": 1,
                            "stats": {"rolled_back": False, "min_det_f": 1},
                        }
                    )
                    + "\n"
                )
            (root / "summary.json").write_text(json.dumps({"complete": True, "published_steps": 1}))
            self.assertEqual(recording.check(root, 1)["status"], "PASS")
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                recording.check(root, 1)
            write({key: value for key, value in values.items() if key != "history.link_vsym"})
            with self.assertRaisesRegex(ValueError, "state schema"):
                recording.check(root, 1)
            invalid = dict(values)
            invalid["history.force_generation"] = np.array([[0], [2]])
            write(invalid)
            with self.assertRaisesRegex(ValueError, "generation mismatch"):
                recording.check(root, 1)


if __name__ == "__main__":
    unittest.main()
