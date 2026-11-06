# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent geometric and publication checks for cloth experiments."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.solver_types import Stats, Status
from scripts.monolithic_reference.cloth_contact_geometry import inspect_geometry
from scripts.monolithic_reference.cloth_contact_run import execute, publication, write_json


class TestClothContactGeometry(unittest.TestCase):
    def test_parallel_layers(self):
        """Measure known layer spacing and thickness compression independently."""
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        x = np.vstack((a, a + np.array([0, 0, 0.0003])))
        result = inspect_geometry(x, [[0, 1, 2], [3, 4, 5]], [0, 1], [0.0005, 0.0005])
        self.assertAlmostEqual(result["minimum_distance_m"], 0.0003)
        self.assertAlmostEqual(result["maximum_thickness_compression_m"], 0.0002)
        self.assertEqual(result["intersections"], [])

    def test_face_interior_crossing(self):
        """Detect an edge piercing a face away from every vertex and edge."""
        x = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [0.3, 0.3, 1]]
        result = inspect_geometry(x, [[0, 1, 2], [3, 4, 5]], [0, 1], [0.001, 0.001])
        self.assertEqual(result["intersections"], [[0, 1]])
        self.assertEqual(result["minimum_distance_m"], 0.0)

    def test_shared_edge_and_foldover(self):
        """Exclude a legal shared edge while retaining overlap beyond that edge."""
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0]], dtype=float)
        faces = [[0, 1, 2], [1, 0, 3]]
        self.assertEqual(inspect_geometry(x, faces, [0, 0], [0.001] * 2)["intersections"], [])
        x[3] = [0.2, 0.2, 0]
        self.assertEqual(inspect_geometry(x, faces, [0, 0], [0.001] * 2)["intersections"], [[0, 1]])

    def test_invalid_geometry(self):
        """Reject nonfinite geometry and invalid mesh indices explicitly."""
        for x, faces in (([[np.nan, 0, 0]] * 3, [[0, 1, 2]]), ([[0, 0, 0]] * 3, [[0, 1, 4]])):
            with self.assertRaises(ValueError):
                inspect_geometry(x, faces, [0], [0.001])

    def test_uncertain_geometry_is_not_clear(self):
        """Expose skinny and degenerate overlapping pairs as unresolved geometry."""
        a = [[-10, 0, 0], [0, 1e-8, 0], [10, 0, 0]]
        b = [[-9, -5e-9, 0], [0, 1.5e-8, 0], [9, -5e-9, 0]]
        for middle in (1e-8, 0.0):
            a[1][1] = middle
            result = inspect_geometry(a + b, [[0, 1, 2], [3, 4, 5]], [0, 1], [0.001] * 2)
            self.assertFalse(result["coverage_complete"])
            self.assertEqual(result["uncertain_pairs"], [[0, 1]])
            self.assertIsNone(result["maximum_thickness_compression_m"])


class TestClothPublication(unittest.TestCase):
    def test_frozen_stats_serialization(self):
        """Retain failure diagnostics containing immutable mapping snapshots."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.json"
            write_json(path, Stats(status=Status.SUCCESS, variational=MappingProxyType({"probe": 1})))
            self.assertEqual(json.loads(path.read_text())["variational"]["probe"], 1)

    def test_stopped_publication(self):
        """Accept valid stopped feedback without labeling it converged."""
        stats = SimpleNamespace(
            step_generation=3,
            rolled_back=False,
            converged=False,
            variational={"publication_status": "STOPPED_ACCEPTED"},
        )
        forces = SimpleNamespace(force_valid=True, force_generation=3)
        self.assertEqual(publication(stats, forces, previous_generation=2), "STOPPED_ACCEPTED")

    def test_stale_or_invalid_publication(self):
        """Reject stale stats and mismatched force generations as hard failures."""
        stats = SimpleNamespace(
            step_generation=3,
            rolled_back=False,
            converged=True,
            variational={"publication_status": "CONVERGED"},
        )
        for previous, valid, generation in ((3, True, 3), (2, False, 3), (2, True, 2)):
            forces = SimpleNamespace(force_valid=valid, force_generation=generation)
            self.assertEqual(publication(stats, forces, previous_generation=previous), "HARD_FAILURE")

    def _record(self, directory, *, fail_read=False, fail_observer=False):
        def array(value):
            return SimpleNamespace(numpy=lambda: np.asarray(value))

        x = [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]
        stats = SimpleNamespace(
            step_generation=0,
            rolled_back=False,
            converged=True,
            status="SUCCESS",
            failure_reason=None,
            variational={"publication_status": "CONVERGED"},
        )
        forces = SimpleNamespace(
            force_valid=True, force_generation=1, particle_force=np.zeros((3, 3)), joint_force=np.zeros(0)
        )
        scene = SimpleNamespace(
            metadata={},
            dt=0.1,
            total_steps=1,
            steps=0,
            phase="test",
            model=SimpleNamespace(device="cpu", particle_q=array(x)),
            triangles=[[0, 1, 2]],
            owners=[0],
            thicknesses=[0.001],
        )
        scene.state = SimpleNamespace(
            particle_q=array(x),
            particle_qd=array(np.zeros((3, 3))),
            joint_q=array([]),
            joint_qd=array([]),
            particle_f=array(np.zeros((3, 3))),
        )
        if fail_read:
            scene.state.joint_q.numpy = lambda: (_ for _ in ()).throw(RuntimeError("readback failed"))
        scene.solver = SimpleNamespace(
            last_stats=stats, get_contact_forces=lambda state: forces, save_checkpoint=lambda state, path: None
        )

        def step():
            stats.step_generation += 1
            scene.steps += 1

        scene.step = step
        recipe = Path(directory) / "recipe.json"
        recipe.write_text("{}")
        args = SimpleNamespace(
            geometry_stride=1,
            max_steps=None,
            output=Path(directory) / "run",
            recipe=recipe,
            method_config=None,
            method="off",
        )
        with (
            patch("scripts.monolithic_reference.cloth_contact_run.wp.synchronize_device"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with patch("scripts.monolithic_reference.cloth_contact_run.inspect_geometry") as observer:
                observer.side_effect = [{}, RuntimeError("observer failed")] if fail_observer else None
                observer.return_value = {}
                execute(lambda: scene, args)
        return json.loads((args.output / "summary.json").read_text()), json.loads(
            (args.output / "steps.jsonl").read_text()
        )

    def test_readback_failure_preserves_publication(self):
        """Keep valid advancement when later evidence collection fails."""
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, fail_read=True)
            self.assertEqual(summary["published_steps"], 1)
            self.assertEqual(summary["publication_counts"]["HARD_FAILURE"], 0)
            self.assertEqual(row["publication_status"], "CONVERGED")
            self.assertEqual(row["time_after_s"], 0.1)
            self.assertIn("recording_error_after_publication", row)

    def test_observer_failure_retains_summary(self):
        """Report an offline observer gap without erasing the simulation receipt."""
        with tempfile.TemporaryDirectory() as folder:
            summary, _ = self._record(folder, fail_observer=True)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["geometry_unchecked_frames"], 1)
            self.assertEqual(len(summary["observer_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
