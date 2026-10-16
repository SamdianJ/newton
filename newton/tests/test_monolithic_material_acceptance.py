# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check production acceptance scenes without substituting short runs for stage gates."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.compare_material_acceptance import compare
from scripts.monolithic_reference.material_acceptance import MaterialAcceptance, segment_distance, triangle_distance
from scripts.monolithic_reference.material_recipes import RECIPE_PATH, build_case
from scripts.monolithic_reference.material_scene import MaterialScene, load_fraction


class TestMaterialAcceptance(unittest.TestCase):
    def test_physical_cap_inputs_and_checkpoint(self):
        """Reject invalid raw caps and checkpoint continuation under different solve goals."""
        recipe = json.loads(RECIPE_PATH.with_name("material_acceptance_v3.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            for key in ("joint_residual_tolerance", "cloth_position_tolerance"):
                for invalid in (True, 0, -1, float("nan"), "1e-9"):
                    changed = dict(recipe, common={**recipe["common"], key: invalid})
                    path.write_text(json.dumps(changed))
                    with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                        MaterialScene("cloth-press", recipe=path)
            path.write_text(json.dumps(recipe))
            first = MaterialScene("cloth-press", recipe=path)
            first.step()
            checkpoint = Path(folder) / "checkpoint.npz"
            first.solver.save_checkpoint(first.state, checkpoint)
            recipe["common"]["joint_residual_tolerance"] *= 2
            path.write_text(json.dumps(recipe))
            second = MaterialScene("cloth-press", recipe=path)
            with self.assertRaises(ValueError):
                second.solver.load_checkpoint(second.state, checkpoint)

    def test_cross_mode_evidence(self):
        """Reject an intermediate-frame drift and partial or mixed-source matrices."""
        recipe = json.loads(RECIPE_PATH.read_text())
        steps = recipe["common"]["steps"]
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("cpu", "cuda", "graph")]
            for index, path in enumerate(paths):
                path.mkdir()
                (path / "summary.json").write_text(
                    json.dumps({"status": "PASS", "cases": dict.fromkeys(recipe["cases"], "PASS")})
                )
                for name in recipe["cases"]:
                    case = path / name
                    case.mkdir()
                    (case / "checkpoint.npz").write_bytes(b"checkpoint fixture")
                    receipt = {
                        "status": "PASS",
                        "steps": steps,
                        "source_unchanged": True,
                        "source": {"source_hashes": {"solver": "same"}},
                        "recipe_sha256": hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest(),
                        "checkpoint_sha256": hashlib.sha256(b"checkpoint fixture").hexdigest(),
                        "arguments": {
                            "device": "cpu" if index == 0 else "cuda:0",
                            "pcg_mode": "diagnostic" if index == 0 else "production",
                            "use_cuda_graph": int(index == 2),
                        },
                    }
                    (case / "receipt.json").write_text(json.dumps(receipt))
                    np.savez_compressed(
                        case / "frames.npz",
                        position=np.zeros((steps, 2, 3)),
                        theta=np.zeros((steps, 1)),
                        q=np.zeros((steps, 1)),
                    )
            self.assertEqual(compare(paths, RECIPE_PATH)["status"], "PASS")
            target = paths[1] / "cable-bend-twist"
            for value in (1e-5, float("nan")):
                positions = np.zeros((steps, 2, 3))
                positions[steps // 2, 0, 0] = value
                np.savez_compressed(
                    target / "frames.npz", position=positions, theta=np.zeros((steps, 1)), q=np.zeros((steps, 1))
                )
                with self.assertRaises(AssertionError):
                    compare(paths, RECIPE_PATH)
            np.savez_compressed(
                target / "frames.npz",
                position=np.zeros((steps, 2, 3)),
                theta=np.zeros((steps, 1)),
                q=np.zeros((steps, 1)),
            )
            receipt = json.loads((target / "receipt.json").read_text())
            receipt["source"]["source_hashes"]["solver"] = "other"
            (target / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(AssertionError, "Mixed source"):
                compare(paths, RECIPE_PATH)
            (paths[0] / "summary.json").write_text(json.dumps({"status": "PARTIAL"}))
            with self.assertRaisesRegex(AssertionError, "Incomplete"):
                compare(paths, RECIPE_PATH)

    def test_continuous_geometry(self):
        """Detect interior and boundary penetration independently of sparse quadrature points."""
        faces = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]])
        self.assertAlmostEqual(triangle_distance([0.2, 0.2, 1], faces)[0], 1)
        self.assertAlmostEqual(triangle_distance([1.0, 1, 0], faces)[0], np.sqrt(0.5))
        self.assertAlmostEqual(triangle_distance([-1.0, -1, 0], faces)[0], np.sqrt(2))
        np.testing.assert_allclose(segment_distance([0.5, 1, 0], faces[:, 0], faces[:, 1]), [1])

    def test_checkpoint_continuation(self):
        """Compare several real continuation steps after rebuilding the complete solver."""
        for name in ("cable-bend-twist", "cloth-press", "precreased-box"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                case = MaterialAcceptance(name)
                case.step()
                case.restore_checkpoint(Path(folder) / "state.npz")
                for _ in range(3):
                    case.step()
                self.assertEqual(case.restart.steps, 4)
                self.assertEqual(case.resume_position_error, 0)
                self.assertEqual(case.resume_theta_error, 0)
                with self.assertRaises(AssertionError):
                    case.check_final()

    def test_loading_protocol(self):
        """Preserve rest, smooth load, hold and complete unloading at the frozen step boundaries."""
        c = build_case("cloth-press")["common"]
        for step, expected in ((0, 0), (500, 0), (1000, 0.5), (1500, 1), (2500, 1), (3000, 0.5), (3500, 0), (5999, 0)):
            self.assertEqual(load_fraction(step, c), expected)
        with self.assertRaises(ValueError):
            load_fraction(6000, c)

    def test_rest_scenes(self):
        """Construct and advance every frozen coupled scene at its natural rest shape."""
        for name in ("cable-bend-twist", "cloth-press", "precreased-box"):
            with self.subTest(name=name):
                scene = MaterialScene(name)
                for _ in range(2):
                    scene.step()
                np.testing.assert_allclose(scene.state.particle_q.numpy(), scene.rest, atol=1e-10)
                self.assertEqual(scene.model.joint_dof_count, 1)
                self.assertEqual(scene.model.tet_count, 0)
                self.assertTrue(scene.solver.last_stats.converged)


if __name__ == "__main__":
    unittest.main()
