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
from scripts.monolithic_reference.cloth_contact_geometry import (
    _segment_points,
    _segment_points_batch,
    _triangle_pair,
    _triangle_pairs,
    inspect_geometry,
)
from scripts.monolithic_reference.cloth_contact_run import execute, publication, write_json


class TestClothContactGeometry(unittest.TestCase):
    def test_batched_witnesses_match_scalar_reference(self):
        """Preserve scalar distances, event classification and witness order in a batch."""
        rng = np.random.default_rng(2718)
        a, b = rng.normal(size=(2, 48, 3, 3)) * 0.01
        b[:8] = a[:8] + np.array([0, 0, 0.0003])
        b[8:16, :, 2] = a[8:16, :, 2] = 0
        distances, hits, mask, coplanar, _, _ = _triangle_pairs(a, b, 1e-9)
        for index in range(len(a)):
            expected_distance, expected_hits, expected_coplanar = _triangle_pair(a[index], b[index], 1e-9)
            self.assertAlmostEqual(distances[index], expected_distance, places=14)
            self.assertEqual(bool(coplanar[index]), expected_coplanar)
            np.testing.assert_allclose(
                hits[index, mask[index]], np.asarray(expected_hits).reshape(-1, 3), atol=2e-16, rtol=1e-12
            )

    def test_nearly_parallel_crossing_segments(self):
        """Keep the cross-product denominator for edges whose dot determinant cancels."""
        a = np.tile([-1.0, 0, 0], (3, 1))
        b = -a
        c = np.array([[-1.0, -angle, 0] for angle in (1e-8, 1e-10, 1e-12)])
        d = -c
        p, q = _segment_points_batch(a, b, c, d)
        np.testing.assert_allclose(p, 0, atol=1e-15)
        np.testing.assert_allclose(q, 0, atol=1e-15)
        for index in range(3):
            expected = _segment_points(a[index], b[index], c[index], d[index])
            np.testing.assert_array_equal(p[index], expected[0])
            np.testing.assert_array_equal(q[index], expected[1])

    def test_tolerance_boundary_uses_scalar_witnesses(self):
        """Retain the scalar decision immediately on either side of the event tolerance."""
        a = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0]])
        for height in (np.nextafter(1e-9, 0), 1e-9, np.nextafter(1e-9, np.inf)):
            b = a + np.array([0, 0, height])
            result = inspect_geometry(np.vstack((a, b)), [[0, 1, 2], [3, 4, 5]], [0, 1], [0.0005] * 2)
            distance, hits, coplanar = _triangle_pair(a, b, 1e-9)
            self.assertEqual(result["minimum_distance_m"], distance)
            self.assertEqual(result["intersections"], [[0, 1]] if hits else [])
            self.assertEqual(
                result["intersection_kinds"],
                ["coplanar_overlap_or_touch" if coplanar else "surface_intersection_or_touch"] if hits else [],
            )

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

    def _record(
        self,
        directory,
        *,
        fail_read=False,
        fail_observer=False,
        fail_step=False,
        fail_body_read=False,
        method_force=None,
        joint_control=None,
        mutate_frames=False,
    ):
        def array(value):
            value = np.asarray(value)
            return SimpleNamespace(numpy=lambda: value)

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
            force_valid=True,
            force_generation=1,
            particle_force=np.zeros((3, 3)),
            joint_force=np.zeros(0),
            body_wrench=np.zeros((0, 6)),
        )
        scene = SimpleNamespace(
            metadata={},
            dt=0.1,
            total_steps=2 if fail_body_read or mutate_frames else 1,
            steps=0,
            phase="test",
            model=SimpleNamespace(device="cpu", particle_q=array(x), body_label=[], joint_label=[]),
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
        if joint_control is not None:
            scene.joint_reference_q = joint_control["joint_reference_q"]
            scene.joint_gravity_force = joint_control["joint_gravity_force"]
            scene.control = SimpleNamespace(
                joint_target_q=array(joint_control["joint_target_q"]),
                joint_target_qd=array(joint_control["joint_target_qd"]),
            )
            scene.state.joint_q, scene.state.joint_qd = array([0.29]), array([-0.01])
            scene.model.joint_label = ["hinge"]
            forces.joint_force = np.zeros(1)
        if fail_read:
            scene.state.joint_q.numpy = lambda: (_ for _ in ()).throw(RuntimeError("readback failed"))
        if fail_body_read:

            def body_positions():
                if scene.steps == 2:
                    raise RuntimeError("body readback failed")
                return np.zeros((0, 7))

            scene.state.body_q = SimpleNamespace(numpy=body_positions)
        scene.solver = SimpleNamespace(
            last_stats=stats, get_contact_forces=lambda state: forces, save_checkpoint=lambda state, path: None
        )

        def step():
            stats.step_generation += 1
            forces.force_generation = stats.step_generation
            if fail_step:
                stats.linear_iterations = 7
                stats.matrix_assembly_count = 3
                stats.rolled_back = True
                stats.converged = False
                stats.status = "LINEAR_MAX_ITERATIONS"
                stats.failure_reason = "injected linear failure"
                stats.variational = {"publication_status": "HARD_FAILURE"}
                forces.force_valid = False
                raise RuntimeError("step failed")
            scene.steps += 1
            if mutate_frames:
                scene.state.particle_q.numpy()[:, 2] = scene.steps
                scene.control.joint_target_q.numpy()[0] = scene.steps / 4

        scene.step = step
        recipe = Path(directory) / "recipe.json"
        recipe.write_text("{}")
        args = SimpleNamespace(
            geometry_stride=1,
            max_steps=None,
            output=Path(directory) / "run",
            recipe=recipe,
            method_config=None,
            method="off" if method_force is None else "point-cloud",
        )
        with (
            patch("scripts.monolithic_reference.cloth_contact_run.wp.synchronize_device"),
            patch("scripts.monolithic_reference.cloth_contact_run.inspect_geometry") as observer,
            patch(
                "scripts.monolithic_reference.cloth_contact_run.importlib.import_module",
                return_value=SimpleNamespace(
                    diagnostics=lambda solver, state: {"method": "point-cloud", "particle_force_N": method_force}
                ),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            observer.side_effect = [{}, RuntimeError("observer failed")] if fail_observer else None
            observer.return_value = {}
            execute(lambda: scene, args)
        return json.loads((args.output / "summary.json").read_text()), json.loads(
            (args.output / "steps.jsonl").read_text().splitlines()[-1]
        )

    def test_method_force_is_retained_in_state_chunk(self):
        """Retain separate method force once without duplicating the full array in JSON."""
        expected = np.full((3, 3), 0.125)
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, method_force=expected)
            self.assertEqual(summary["published_steps"], 1)
            self.assertNotIn("particle_force_N", row["method_diagnostics"])
            with np.load(Path(folder) / "run/states-0000.npz") as states:
                np.testing.assert_array_equal(states["method_particle_force"][0], expected)

    def test_nominal_target_is_recorded_separately_from_gravity_drive(self):
        """Retain nominal pose, applied PD target and gravity torque without conflating them."""
        expected = {
            name: np.array([value], dtype=np.float32)
            for name, value in (
                ("joint_reference_q", 0.3),
                ("joint_target_q", 0.25),
                ("joint_target_qd", 0.1),
                ("joint_gravity_force", -5.0),
            )
        }
        with tempfile.TemporaryDirectory() as folder:
            summary, _ = self._record(folder, joint_control=expected)
            self.assertEqual(summary["published_steps"], 1)
            manifest = json.loads((Path(folder) / "run/manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 3)
            with np.load(Path(folder) / "run/states-0000.npz") as states:
                for name, value in expected.items():
                    np.testing.assert_array_equal(states[name][0], value)
                    self.assertEqual(states[name].dtype, np.float32)
                np.testing.assert_array_equal(states["q"][0], [0.29])

    def test_recorded_frames_do_not_alias_reused_cpu_arrays(self):
        """Keep earlier geometry and applied controls when a later step mutates CPU storage."""
        control = {
            name: np.zeros(1, dtype=np.float32)
            for name in ("joint_reference_q", "joint_target_q", "joint_target_qd", "joint_gravity_force")
        }
        with tempfile.TemporaryDirectory() as folder:
            summary, _ = self._record(folder, joint_control=control, mutate_frames=True)
            self.assertEqual(summary["published_steps"], 2)
            with np.load(Path(folder) / "run/states-0000.npz") as states:
                np.testing.assert_array_equal(states["step"], [1, 2])
                np.testing.assert_array_equal(states["x"][:, :, 2], [[1.0] * 3, [2.0] * 3])
                np.testing.assert_array_equal(states["joint_target_q"][:, 0], [0.25, 0.5])

    def test_partial_frame_does_not_discard_earlier_states(self):
        """Retain complete earlier frames when optional state readback fails."""
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, fail_body_read=True)
            self.assertEqual(summary["published_steps"], 2)
            self.assertIn("body readback failed", row["recording_error_after_publication"])
            with np.load(Path(folder) / "run/states-0000.npz") as states:
                np.testing.assert_array_equal(states["step"], [1])
                self.assertEqual(states["body_q"].shape, (1, 0, 7))

    def test_invalid_method_force_preserves_publication(self):
        """Reject malformed feedback without reclassifying an already published step."""
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, method_force=np.zeros((2, 3)))
            self.assertEqual(summary["published_steps"], 1)
            self.assertEqual(row["publication_status"], "CONVERGED")
            self.assertIn("Invalid method particle force", row["recording_error_after_publication"])
            self.assertFalse(list((Path(folder) / "run").glob("states-*.npz")))

    def test_readback_failure_preserves_publication(self):
        """Keep valid advancement when later evidence collection fails."""
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, fail_read=True)
            self.assertEqual(summary["published_steps"], 1)
            self.assertEqual(summary["publication_counts"]["HARD_FAILURE"], 0)
            self.assertEqual(row["publication_status"], "CONVERGED")
            self.assertEqual(row["time_after_s"], 0.1)
            self.assertIn("recording_error_after_publication", row)

    def test_failed_step_preserves_fresh_solver_cost(self):
        """Retain work spent by a fresh failed attempt without counting a publication."""
        with tempfile.TemporaryDirectory() as folder:
            summary, row = self._record(folder, fail_step=True)
            self.assertEqual(summary["published_steps"], 0)
            self.assertEqual(row["publication_status"], "HARD_FAILURE")
            self.assertTrue(row["stats_fresh"])
            self.assertEqual(row.get("solver_stats", {}).get("linear_iterations"), 7)
            self.assertEqual(row.get("solver_stats", {}).get("matrix_assembly_count"), 3)

    def test_observer_failure_retains_summary(self):
        """Report an offline observer gap without erasing the simulation receipt."""
        with tempfile.TemporaryDirectory() as folder:
            summary, _ = self._record(folder, fail_observer=True)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["geometry_unchecked_frames"], 1)
            self.assertEqual(len(summary["observer_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
