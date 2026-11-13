# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent offline criteria and receipts for cable-contact experiments."""

import contextlib
import copy
import hashlib
import io
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.cable_contact_criteria import (
    _frame,
    closed_writhe,
    closure_clearance,
    contact_pairs,
    crossing_events,
    driver_work,
    elastic_energy,
    elastic_gradient,
    evaluate,
    frame_penetration,
    ground_contact,
    load_run,
    max_rise,
    penalty,
    segment_distance,
    segment_table,
    total_twist,
    writhe,
)
from scripts.monolithic_reference.cable_contact_criteria import main as criteria_main
from scripts.monolithic_reference.cable_contact_scenes import CableContactScene, cable_section, trapezoid
from scripts.monolithic_reference.cable_oracle import CableFrame, CableReference, cable_derivatives
from scripts.monolithic_reference.run_cable_contact import main as run_main

FIXTURES = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures"
GATES = json.loads((FIXTURES / "cable_contact_gates_frozen_v1.json").read_text())


def _topology(curves, radius=0.002):
    ranges, start = [], 0
    for curve in curves:
        ranges.append((start, start + len(curve)))
        start += len(curve)
    return {
        "vertex_ranges": np.asarray(ranges),
        "rest_positions": np.vstack(curves),
        "radius": np.full(len(curves), radius),
    }


def _brute_distance(p1, q1, p2, q2, samples=2001):
    s = np.linspace(0, 1, samples)
    a = p1 + np.outer(s, q1 - p1)
    b = p2 + np.outer(s, q2 - p2)
    return np.min(np.linalg.norm(a[:, None] - b[None], axis=2))


class TestCableContactGeometry(unittest.TestCase):
    def test_segment_distance_matches_dense_sampling(self):
        """Exact clamped distances agree with dense sampling, including caps and parallel pairs."""
        rng = np.random.default_rng(7)
        cases = [rng.normal(size=(4, 3)) for _ in range(40)]
        cases.append(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5], [1, 1, 0.5]], dtype=float))  # parallel
        cases.append(np.array([[0, 0, 0], [1, 0, 0], [2, 0.3, 0], [3, 0.3, 0]], dtype=float))  # collinear caps
        cases.append(np.array([[0, 0, 0], [1, 0, 0], [0.5, -1, 0.004], [0.5, 1, 0.004]], dtype=float))  # crossing
        points = np.asarray(cases)
        d, _, _ = segment_distance(points[:, 0], points[:, 1], points[:, 2], points[:, 3])
        for case, value in zip(cases, d, strict=True):
            self.assertAlmostEqual(value, _brute_distance(*case), delta=2e-3 * np.linalg.norm(case[1] - case[0]))
        self.assertAlmostEqual(d[-1], 0.004, places=12)
        self.assertAlmostEqual(d[-3], math.hypot(1.0, 0.5), places=12)
        self.assertAlmostEqual(d[-2], math.hypot(1.0, 0.3), places=12)

    def test_degenerate_segment_is_rejected(self):
        p = np.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            segment_distance(p, p, p, p + 1)

    def test_perpendicular_crossing_penetration(self):
        """Two crossing capsules at axis distance 3 mm with r = 2 mm overlap by 1 mm."""
        a = np.column_stack([np.linspace(-0.05, 0.05, 11), np.zeros(11), np.zeros(11)])
        b = np.column_stack([np.zeros(11), np.linspace(-0.05, 0.05, 11), np.full(11, 0.003)])
        topology = _topology([a, b])
        segments = segment_table(topology)
        i, j, same = contact_pairs(segments, index_span=3, arc_factor=np.pi / 2)
        self.assertFalse(np.any(same[(segments["cable"][i] != segments["cable"][j])]))
        near, depth = frame_penetration(np.vstack([a, b]), segments, i, j)
        self.assertAlmostEqual(depth.max(), 0.001, places=12)
        inter = segments["cable"][i[near]] != segments["cable"][j[near]]
        self.assertTrue(np.all(inter[depth > 0]))

    def test_exclusion_rule(self):
        """Same-cable neighbours within three edges or the arc reach are excluded; other cables never."""
        straight = np.column_stack([0.007 * np.arange(30), np.zeros(30), np.zeros(30)])
        segments = segment_table(_topology([straight, straight + np.array([0, 0.1, 0])]))
        i, j, same = contact_pairs(segments, index_span=3, arc_factor=np.pi / 2)
        kept = {(int(a), int(b)) for a, b in zip(i, j, strict=True)}
        self.assertNotIn((0, 3), kept)
        # Arc reach pi/2 * 4 mm = 6.28 mm < gap of 4 edges = 21 mm: index rule decides here.
        self.assertIn((0, 4), kept)
        self.assertIn((0, 29), kept)  # first segment of the second cable
        self.assertTrue(np.all(~same[segments["cable"][i] != segments["cable"][j]]))
        wide = contact_pairs(segments, index_span=3, arc_factor=10.0)
        kept_wide = {(int(a), int(b)) for a, b in zip(wide[0], wide[1], strict=True)}
        # Reach 10 * 4 mm = 40 mm: the rest-arc gap of (0, 6) is 35 mm, of (0, 7) 42 mm.
        self.assertNotIn((0, 6), kept_wide)
        self.assertIn((0, 7), kept_wide)

    def test_helix_neighbouring_turns_are_kept(self):
        """A coiled rest shape keeps pairs from adjacent turns; only arc-near pairs are excluded."""
        u = np.linspace(0, 6 * np.pi, 120)
        helix = np.column_stack([0.02 * np.cos(u), 0.02 * np.sin(u), 0.0045 * u / (2 * np.pi)])
        segments = segment_table(_topology([helix]))
        i, j, _ = contact_pairs(segments, index_span=3, arc_factor=np.pi / 2)
        turn = int(np.argmin(np.abs(u - 2 * np.pi)))
        kept = {(int(a), int(b)) for a, b in zip(i, j, strict=True)}
        self.assertIn((0, turn), kept)
        _, depth = frame_penetration(helix, segments, i, j)
        # Adjacent turns are 4.5 mm apart: no overlap of 2 mm capsules, but 0.5 mm clearance.
        self.assertAlmostEqual(depth.max(), 0.004 - 0.0045, delta=1e-4)

    def test_crossing_detection(self):
        """A strand pushed through another is one event; a miss or a coplanar slide is none."""
        a = np.column_stack([np.linspace(-0.03, 0.03, 6), np.zeros(6), np.zeros(6)])
        b0 = np.column_stack([np.zeros(6), np.linspace(-0.03, 0.03, 6), np.full(6, 0.002)])
        segments = segment_table(_topology([a, b0]))
        i, j, _ = contact_pairs(segments, index_span=3, arc_factor=np.pi / 2)
        threshold = GATES["crossing"]["min_sine"]
        through = crossing_events(
            np.vstack([a, b0]), np.vstack([a, b0 - np.array([0, 0, 0.004])]), segments, i, j, min_sine=threshold
        )
        self.assertEqual(len(through[0]), 1)
        miss = crossing_events(
            np.vstack([a, b0 + np.array([0, 0, 0.02])]),
            np.vstack([a, b0 + np.array([0, 0, 0.01])]),
            segments,
            i,
            j,
            min_sine=threshold,
        )
        self.assertEqual(len(miss[0]), 0)
        flat = np.column_stack([np.linspace(-0.03, 0.03, 6), np.full(6, 0.003), np.zeros(6)])
        slide = crossing_events(
            np.vstack([a, flat]), np.vstack([a, flat + np.array([0.005, 0, 0])]), segments, i, j, min_sine=threshold
        )
        self.assertEqual(len(slide[0]), 0)


class TestCableContactGates(unittest.TestCase):
    def test_frozen_recipes_are_unchanged(self):
        """The frozen gates name the exact recipes they judge; a recipe edit must not pass silently."""
        self.assertEqual(GATES["status"], "frozen")
        for name, digest in GATES["frozen"]["recipes_sha256"].items():
            with self.subTest(recipe=name):
                self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), digest)


class TestCableContactLoading(unittest.TestCase):
    @staticmethod
    def _write_run(directory, *, initial_target):
        """A two-step synthetic record whose clamped angle target jumps to 1 rad at step 1."""
        x = np.column_stack([0.01 * np.arange(4), np.zeros(4), np.zeros(4)])
        state = {"x": x, "v": np.zeros_like(x), "theta": np.zeros(3), "director": np.tile([0.0, 0.0, 1.0], (3, 1))}
        state["twist"] = np.zeros(2)
        initial = dict(state, target_theta=np.zeros(3)) if initial_target else state
        np.savez(directory / "initial.npz", **initial)
        chunk = {key: np.stack([value, value]) for key, value in state.items()}
        chunk.update(step=np.array([1, 2]), target_theta=np.array([[1.0, 0, 0], [1.0, 0, 0]]))
        np.savez(directory / "states-0000.npz", **chunk)
        np.savez(directory / "topology.npz", dt=np.array([0.002]))
        (directory / "manifest.json").write_text("{}")
        (directory / "summary.json").write_text(json.dumps({"chunks": ["states-0000.npz"]}))

    def test_initial_target_marks_a_step_one_jump_as_driven(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self._write_run(directory, initial_target=True)
            targets = load_run(directory).frames["target_theta"]
            np.testing.assert_array_equal(targets[:, 0], [0.0, 1.0, 1.0])
            self.assertGreater(np.ptp(targets[:, 0]), 0)

    def test_records_without_initial_target_repeat_the_first_step(self):
        """Documented fallback for runs before c4c1e330: the step-1 target stands in for the initial one."""
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self._write_run(directory, initial_target=False)
            np.testing.assert_array_equal(load_run(directory).frames["target_theta"][:, 0], [1.0, 1.0, 1.0])


class TestCableContactLink(unittest.TestCase):
    def test_planar_closed_curve_has_zero_writhe(self):
        u = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        self.assertAlmostEqual(writhe(np.column_stack([np.cos(u), 2 * np.sin(u), np.zeros(80)])), 0.0, places=12)

    def test_strand_passage_changes_writhe_by_two(self):
        """Pushing one strand of a closed curve through another jumps the Gauss writhe by 2."""

        def curve(h):
            # Figure-eight-like planar loop with its crossing lifted by +/- h.
            u = np.linspace(0, 2 * np.pi, 400, endpoint=False)
            x, y = np.sin(u), np.sin(u) * np.cos(u)
            z = h * np.cos(u)
            return np.column_stack([x, y, z])

        jump = writhe(curve(0.02)) - writhe(curve(-0.02))
        self.assertAlmostEqual(abs(jump), 2.0, delta=0.1)

    def test_helix_writhe_follows_fuller(self):
        """A right-handed helix closed far away has Wr = N (1 - cos psi) > 0; its mirror image the opposite."""
        turns, radius, psi = 3, 0.05, np.radians(60.0)
        pitch = 2 * np.pi * radius / np.tan(psi)
        phi = np.linspace(0, 2 * np.pi * turns, 1201)
        helix = np.column_stack([pitch * phi / (2 * np.pi), radius * np.cos(phi), radius * np.sin(phi)])
        expected = turns * (1 - np.cos(psi))
        self.assertAlmostEqual(closed_writhe(helix, 10.0), expected, delta=0.01)
        helix[:, 1] *= -1
        self.assertAlmostEqual(closed_writhe(helix, 10.0), -expected, delta=0.01)

    def test_closure_clearance_of_a_straight_cable(self):
        """Interior vertices x[2:-2] of a straight cable are two edges from the closure lines at its ends."""
        x = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)])
        self.assertAlmostEqual(closure_clearance(x, 10.0), 0.2, places=12)

    def test_straight_clamped_cable_link(self):
        """A straight cable twisted by N turns has Tw = N and closed writhe 0."""
        x = np.column_stack([np.linspace(-0.4, 0.4, 50), np.zeros(50), np.ones(50)])
        theta = np.linspace(0, 2 * np.pi * 3, 49)
        self.assertAlmostEqual(total_twist(theta, np.zeros(48)), 3.0, places=12)
        self.assertAlmostEqual(closed_writhe(x, 10.0), 0.0, places=10)


class TestCableContactEnergy(unittest.TestCase):
    def test_penalty_is_continuous_at_the_smoothing_band(self):
        s, k = 1e-4, 1e7
        for edge in (-s, s):
            left, right = penalty(edge - 1e-12, k, s), penalty(edge + 1e-12, k, s)
            self.assertAlmostEqual(float(left), float(right), delta=1e-8)  # slope k s = 1e3
        self.assertEqual(float(penalty(2 * s, k, s)), 0.0)
        self.assertAlmostEqual(float(penalty(-3 * s, k, s)), k * (3 * s) ** 2 / 2, places=6)

    def test_ground_contact_only_below_the_surface(self):
        x = np.column_stack([0.007 * np.arange(5), np.zeros(5), np.full(5, 0.005)])
        director = np.tile([0.0, 0.0, 1.0], (4, 1))
        lengths = np.full(4, 0.007)
        energy, depth = ground_contact(x, np.zeros(4), director, 0.002, lengths, 0.0, 1e7, 1e-4)
        self.assertEqual(energy, 0.0)
        self.assertAlmostEqual(depth, -0.003, places=12)
        energy, depth = ground_contact(
            x - np.array([0, 0, 0.0035]), np.zeros(4), director, 0.002, lengths, 0.0, 1e7, 1e-4
        )
        self.assertGreater(energy, 0.0)
        self.assertAlmostEqual(depth, 0.0005, places=12)

    @staticmethod
    def _straight(count=6, length=0.01):
        x = np.column_stack([length * np.arange(count), np.zeros(count), np.zeros(count)])
        directors = np.tile([0.0, 0.0, 1.0], (count - 1, 1))
        rest = CableReference(x, np.zeros(count - 1), directors, (10.0, 1e-3, 2e-3))
        return rest, x, directors

    def test_driver_work_of_a_quasi_static_pull_equals_the_stretch_energy(self):
        """Pulling one kinematic end with the cable in uniform equilibrium stores exactly the work done."""
        rest, x0, directors = self._straight()
        frames = 51
        stretch = np.linspace(0.0, 1e-3, frames)
        xs = np.stack([x0 * (1 + s / x0[-1, 0]) for s in stretch])
        thetas, twists = np.zeros((frames, 5)), np.zeros((frames, 4))
        ds = np.repeat(directors[None], frames, axis=0)
        work = driver_work(rest, xs, thetas, ds, twists, [0, 5], [])
        energy = [elastic_energy(rest, xs[k], thetas[k], _frame(xs[k], ds[k], twists[k])) for k in (0, -1)]
        self.assertAlmostEqual(work.sum(), energy[1] - energy[0], delta=1e-12)
        self.assertAlmostEqual(work.sum(), 0.5 * 10.0 / 0.05 * 1e-6, delta=1e-12)
        self.assertGreater(work.sum(), 0.0)

    def test_driver_work_of_a_clamped_twist_equals_the_twist_energy(self):
        """Rotating a clamped end angle with linear interior angles stores GJ Theta^2 / (2 L_dual)."""
        rest, x0, directors = self._straight()
        frames = 41
        total = np.linspace(0.0, 1.5, frames)
        thetas = np.stack([np.linspace(0.0, angle, 5) for angle in total])
        xs = np.repeat(x0[None], frames, axis=0)
        ds = np.repeat(directors[None], frames, axis=0)
        work = driver_work(rest, xs, thetas, ds, np.zeros((frames, 4)), [], [0, 4])
        self.assertAlmostEqual(work.sum(), 0.5 * 2e-3 * 1.5**2 / (4 * 0.01), delta=1e-12)

    def test_elastic_gradient_matches_the_oracle_derivatives(self):
        rest, x0, directors = self._straight()
        rng = np.random.default_rng(3)
        x = x0 + rng.normal(scale=5e-4, size=x0.shape)
        theta = rng.normal(scale=0.2, size=5)
        frame = CableFrame(x0, directors)
        residual, _ = cable_derivatives(rest, x, theta, frame)
        gx, gt = elastic_gradient(rest, x, theta, frame, [0, 3], [1, 4])
        np.testing.assert_allclose(gx, residual[:18].reshape(6, 3)[[0, 3]], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(gt, residual[18:][[1, 4]], rtol=1e-12, atol=1e-15)

    def test_max_rise(self):
        self.assertEqual(max_rise([3.0, 2.0, 1.0]), 0.0)
        self.assertAlmostEqual(max_rise([3.0, 1.0, 1.5, 0.5, 1.25]), 0.75)


class TestCableContactScenes(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((FIXTURES / "cable_contact_e4_cross_v1.json").read_text())

    def test_section_from_material(self):
        section = cable_section(
            {"radius_m": 0.002, "young_modulus_Pa": 2e6, "poisson_ratio": 0.3, "density_kg_m3": 1200}
        )
        self.assertAlmostEqual(section["EA_N"], 2e6 * np.pi * 4e-6)
        self.assertAlmostEqual(section["EI_Nm2"], 2e6 * np.pi * 16e-12 / 4)
        self.assertAlmostEqual(section["GJ_Nm2"] / section["EI_Nm2"], 1 / 1.3)

    def test_trapezoid_profile_reaches_the_distance(self):
        self.assertEqual(trapezoid(0.0, 1.0, 0.5, 0.1), 0.0)
        self.assertAlmostEqual(trapezoid(10.0, 1.0, 0.5, 0.1), 1.0)
        self.assertAlmostEqual(trapezoid(1.0, 1.0, 0.5, 0.1), 0.5 * 0.5 * 0.1 + 0.5 * 0.9)

    def test_recipe_errors_name_the_field(self):
        for path, pattern in (
            (("cross",), "lacks field cross"),
            (("cross", "wavelength_m"), "cross.wavelength_m"),
            (("material", "radius_m"), "material.radius_m"),
        ):
            recipe = copy.deepcopy(self.recipe)
            target = recipe
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, pattern):
                CableContactScene(recipe)
        recipe = copy.deepcopy(self.recipe)
        recipe["duration_s"] = 0.0031
        with self.assertRaisesRegex(ValueError, "integral"):
            CableContactScene(recipe)

    def test_spiral_stack_keeps_the_requested_gap(self):
        """The fixed timing state: near-equal edges, bottom layer on the table, closest surfaces exactly g apart."""
        for name, target in (("cable_contact_fixed1024_v1.json", 2e-4), ("cable_contact_fixed1024_g0_v1.json", 0.0)):
            recipe = json.loads((FIXTURES / name).read_text())
            scene = CableContactScene.__new__(CableContactScene)
            scene.section, scene.ground_height = {"radius_m": 0.002}, 0.0
            (points,), *_ = scene._build_spiral_stack(recipe)
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.assertLess(np.ptp(lengths) / lengths.max(), 0.05)  # equal arc, chords shorten on the ramps
            self.assertAlmostEqual(points[:, 2].min(), 0.002, places=12)
            topology = {
                "vertex_ranges": np.array([[0, len(points)]]),
                "rest_positions": points,
                "radius": np.array([0.002]),
            }
            segments = segment_table(topology)
            i, j, _ = contact_pairs(segments, index_span=3, arc_factor=np.pi / 2)
            _, depth = frame_penetration(points, segments, i, j)
            with self.subTest(name=name):
                self.assertAlmostEqual(-depth.max(), target, delta=1e-6)

    def test_contact_method_is_not_available_yet(self):
        with self.assertRaisesRegex(ValueError, "only method 'off'"):
            CableContactScene(self.recipe, method="cable-contact")


class TestCableContactReceipt(unittest.TestCase):
    def test_short_run_binds_source_and_recipe(self):
        """Two CPU steps record the tested commit, recipe hash and contiguous frames; criteria read them."""
        recipe = json.loads((FIXTURES / "cable_contact_e4_cross_v1.json").read_text())
        recipe["cables"] = [{"vertices": 8, "segment_length_m": 0.007}] * 2
        recipe["cross"] = dict(recipe["cross"], angles_deg=[0.0, 90.0], heights_m=[0.02, 0.03])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.json"
            path.write_text(json.dumps(recipe))
            output = Path(directory) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_main(["E4", "--recipe", str(path), "--max-steps", "2", "--output", str(output)])
            self.assertEqual(code, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
            self.assertEqual(manifest["source_sha"], head)
            self.assertEqual(len(manifest["recipe_sha256"]), 64)
            run = load_run(output)
            np.testing.assert_array_equal(run.frames["step"], [0, 1, 2])
            self.assertEqual(run.summary["status"], "PARTIAL")
            criteria, _ = evaluate(run, GATES)
            self.assertEqual(criteria["verdict"], "PARTIAL")
            self.assertEqual(criteria["gates_status"], "frozen")
            # A complete run of a recipe the frozen gates do not name must not be PASS.
            recipe["duration_s"] = 0.004
            path.write_text(json.dumps(recipe))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_main(["E4", "--recipe", str(path), "--output", str(output / "complete")]), 0)
            run = load_run(output / "complete")
            run.manifest["git_status"], run.manifest["source_files_changed_during_run"] = "", []
            self.assertEqual(evaluate(run, GATES)[0]["verdict"], "NOT_FROZEN_RECIPE")
            frozen = copy.deepcopy(GATES)
            frozen["frozen"]["recipes_sha256"]["test.json"] = run.manifest["recipe_sha256"]
            self.assertEqual(evaluate(run, frozen)[0]["verdict"], "PASS")
            run.manifest["git_status"] = " M scripts/monolithic_reference/cable_contact_scenes.py"
            self.assertEqual(evaluate(run, frozen)[0]["verdict"], "DIRTY_SOURCE")
            del run.manifest["git_status"]  # a receipt without the field is not clean either
            self.assertEqual(evaluate(run, frozen)[0]["verdict"], "DIRTY_SOURCE")
            run.manifest["git_status"] = ""
            run.manifest["scene"]["recipe"]["scene"] = "E1-1024"  # performance scenes have no E2E gates
            self.assertEqual(evaluate(run, frozen)[0]["verdict"], "NOT_GATED")
            # The CLI records the gates file and the judge's own identity.
            with contextlib.redirect_stdout(io.StringIO()):
                criteria_main([str(output / "complete"), "--output", str(output / "criteria.json")])
            written = json.loads((output / "criteria.json").read_text())
            gates_file = FIXTURES / "cable_contact_gates_frozen_v1.json"
            self.assertEqual(written["gates_sha256"], hashlib.sha256(gates_file.read_bytes()).hexdigest())
            judge = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/cable_contact_criteria.py"
            self.assertEqual(written["evaluator"]["sha256"], hashlib.sha256(judge.read_bytes()).hexdigest())
            self.assertEqual(written["evaluator"]["source_sha"], head)
            with self.assertRaisesRegex(ValueError, "differs from the recipe scene"):
                run_main(["E1", "--recipe", str(path), "--max-steps", "1", "--output", str(output / "again")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
