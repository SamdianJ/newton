# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent controls and physical acceptance checks for full-hand task probes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.hand_task_control import ContactHandControl, HandObservation, contact_loads
from scripts.monolithic_reference.run_hand_tasks import TASKS, HandTask, physical_gates, run_case, schedule


class TestHandTaskControls(unittest.TestCase):
    def test_schedule_derivatives(self):
        for task in TASKS:
            for t in np.linspace(0.01, 6.99, 231):
                h = 1e-6
                actual = schedule(task, t)
                left, right = schedule(task, t - h), schedule(task, t + h)
                np.testing.assert_allclose(
                    [actual[1], actual[3]], [(right[0] - left[0]) / (2 * h), (right[2] - left[2]) / (2 * h)], atol=1e-8
                )

    def test_release_and_carriage_return(self):
        for task in TASKS:
            final = schedule(task, 7)
            self.assertEqual(final[0], 0)
            self.assertEqual(final[1], 0)
            self.assertEqual(final[3], 0)
            self.assertEqual(final[2], 0.05 if task == "pull-slider" else 0)
            self.assertEqual(schedule(task, 4.25)[2], 0.05)

    @staticmethod
    def good_rows(task):
        base = {
            "time": 7,
            "contact_force_N": 0.5,
            "hand_object_force_N": 0.5,
            "joint_limit_violation_rad": 0,
            "speed_ratio": 0.5,
            "effort_ratio": 0.5,
            "object_active_force_N": 0,
            "min_det_f": 0.8,
            "sample_penetration_m": 0.001,
            "object_position": [0.05 if task == "pull-slider" else 0, 0, 0.02],
        }
        return [{**base, "time": 4, "object_position": [0, 0, 0.07]}, base]

    def test_static_object_rejected(self):
        for task in TASKS:
            rows = [{**r, "object_position": [0, 0, 0.0202]} for r in self.good_rows(task)]
            self.assertFalse(physical_gates(task, rows)[1]["motion"])

    def test_unphysical_success_rejected(self):
        for task in TASKS:
            rows = self.good_rows(task)
            self.assertTrue(all(physical_gates(task, rows)[1].values()))
            for key, value, gate in (
                ("sample_penetration_m", 0.002, "penetration"),
                ("joint_limit_violation_rad", 0.03, "joint_limits"),
                ("effort_ratio", 1.3, "effort"),
                ("speed_ratio", 1.3, "speed"),
                ("object_active_force_N", 0.01, "passive_object"),
                ("min_det_f", -0.1, "material"),
            ):
                bad = [{**r, key: value} for r in rows]
                self.assertFalse(physical_gates(task, bad)[1][gate])

    def test_missing_observation_or_geometry_is_not_success(self):
        rows = self.good_rows("grasp-rigid")
        self.assertFalse(physical_gates("grasp-rigid", rows[:1])[1]["released_stable"])
        for row in rows:
            del row["sample_penetration_m"]
        self.assertFalse(physical_gates("grasp-rigid", rows)[1]["penetration"])

    def test_contact_gate_requires_observed_object_force(self):
        for task in TASKS:
            for load, expected in ((0.0, False), (0.01, False), (0.02, True), (None, False)):
                rows = [{**r, "contact_force_N": 3.5, "hand_object_force_N": load} for r in self.good_rows(task)]
                metrics, gates = physical_gates(task, rows)
                self.assertEqual(metrics["max_contact_force_N"], 3.5)
                self.assertEqual(metrics["max_hand_object_force_N"], load)
                self.assertEqual(gates["contact"], expected)
            rows = self.good_rows(task)
            del rows[-1]["hand_object_force_N"]
            metrics, gates = physical_gates(task, rows)
            self.assertIsNone(metrics["max_hand_object_force_N"])
            self.assertFalse(gates["contact"])

    def test_slider_gate_follows_elevated_target(self):
        rows = self.good_rows("pull-slider")
        raised = [{**r, "object_position": np.add(r["object_position"], (0, 0, 0.08)).tolist()} for r in rows]
        self.assertTrue(all(physical_gates("pull-slider", raised, height=0.08)[1].values()))
        self.assertFalse(physical_gates("pull-slider", rows, height=0.08)[1]["placement"])


class TestContactHandControls(unittest.TestCase):
    @staticmethod
    def controller(task="grasp-rigid"):
        return ContactHandControl(task, mass=0.064, friction=0.5, source_rates=np.ones(5))

    @staticmethod
    def observation(generation, *, load=0.0, depth=0.0, support=0.0, z=0.0202, velocity=0.0):
        return HandObservation(
            generation,
            np.array([load, load, 0, 0, 0]),
            np.zeros(5),
            np.full(5, depth),
            support,
            np.array([0.0, 0.0, z]),
            np.array([0.0, 0.0, velocity]),
            z - 0.02,
        )

    def test_ground_load_cannot_establish_grip(self):
        pair = {
            "source": 1,
            "target": 2,
            "force_generation": 5,
            "world_wrench": [[0, 0, 10, 0, 0, 0], [0, 0, -10, 0, 0, 0]],
        }
        fingers, ground, support, hand_object = contact_loads([pair], {1: 0}, {3}, {2}, material=False, generation=5)
        np.testing.assert_array_equal(fingers, np.zeros(5))
        self.assertEqual(ground[0], 10)
        self.assertEqual(support, 0)
        self.assertEqual(hand_object, 0)
        control = self.controller()
        for step in range(1, 1401):
            observation = self.observation(step)
            observation.finger_ground_force[:] = 10
            control.advance(step * 0.002, observation)
        self.assertEqual(control.carriage, 0)
        self.assertEqual(control.failure, "grip-not-established")
        self.assertFalse(control.completed)

    def test_object_and_material_pair_loads(self):
        pair = {
            "source": 1,
            "target": 3,
            "force_generation": 5,
            "world_wrench": [[0.2, 0, 0, 0, 0, 0], [-0.2, 0, 0, 0, 0, 0]],
        }
        self.assertAlmostEqual(contact_loads([pair], {1: 0}, {3}, {2}, material=False, generation=5)[0][0], 0.2)
        material = {**pair, "source": -1, "target": 1}
        self.assertAlmostEqual(contact_loads([material], {1: 0}, set(), {2}, material=True, generation=5)[0][0], 0.2)
        with self.assertRaises(ValueError):
            contact_loads([pair], {1: 0}, {3}, {2}, material=False, generation=6)

    def test_grip_requires_opposing_sustained_load(self):
        control = self.controller()
        for step in range(1, 181):
            observation = self.observation(step, load=2)
            observation.finger_force[0] = 0
            control.advance(step * 0.002, observation)
        self.assertEqual(control.phase, "close")
        for step in range(181, 210):
            control.advance(step * 0.002, self.observation(step, load=1))
        self.assertEqual(control.phase, "close")
        for step in range(210, 240):
            control.advance(step * 0.002, self.observation(step, load=1))
        self.assertEqual(control.phase, "move")

    def test_penetration_guard_opens_and_prevents_move(self):
        control = self.controller()
        control.sources[:] = 1.5
        for step in range(1, 250):
            control.advance(step * 0.002, self.observation(step, load=1, depth=0.0011))
        self.assertEqual(control.phase, "close")
        self.assertLess(np.max(control.sources), 1.5)
        self.assertEqual(control.carriage, 0)

    def test_stale_or_nonfinite_feedback_holds_targets(self):
        for broken in ("stale", "nonfinite"):
            control = self.controller()
            control.sources[:] = 1.5
            control.advance(0.002, self.observation(1))
            before = control.sources.copy()
            observation = self.observation(1 if broken == "stale" else 2)
            if broken == "nonfinite":
                observation.finger_force[0] = np.nan
            control.advance(0.004, observation)
            np.testing.assert_array_equal(control.sources, before)
            np.testing.assert_array_equal(control.rates, np.zeros(5))
            self.assertEqual(control.phase, "feedback-invalid")

    def test_snapshot_replays_next_command(self):
        control = self.controller()
        for step in range(1, 200):
            control.advance(step * 0.002, self.observation(step, load=0.2))
        restored = ContactHandControl.from_snapshot(control.snapshot())
        observation = self.observation(200, load=1)
        restored.advance(0.4, HandObservation.from_snapshot(observation.snapshot()))
        control.advance(0.4, observation)
        self.assertEqual(restored.snapshot(), control.snapshot())

    def test_snapshot_does_not_share_transition_history(self):
        control = self.controller()
        for step in range(1, 125):
            control.advance(step * 0.002, self.observation(step))
        snapshot = control.snapshot()
        frozen = copy.deepcopy(snapshot)
        restored = ContactHandControl.from_snapshot(snapshot)
        control.advance(0.25, self.observation(125))
        self.assertEqual(snapshot, frozen)
        self.assertEqual(restored.snapshot(), frozen)
        restored.advance(0.25, self.observation(125))
        self.assertEqual(restored.snapshot(), control.snapshot())
        self.assertEqual(snapshot, frozen)

    def test_lost_grip_aborts_motion(self):
        control = self.controller()
        control.phase, control.phase_start = "move", 0.0
        control.initial_position = np.array([0, 0, 0.0202])
        for step in range(1, 111):
            control.advance(step * 0.002, self.observation(step))
        self.assertEqual(control.failure, "grip-lost")
        self.assertEqual(control.carriage_velocity, 0)
        self.assertLess(control.carriage, 0.05)
        self.assertFalse(control.completed)

    def test_grasp_release_waits_for_support(self):
        control = self.controller()
        control.sources[:] = 1.0
        control.phase, control.phase_start = "lower", 0.0
        control.move_start = 0.05
        control.initial_position = np.array([0, 0, 0.0202])
        for step in range(1, 600):
            control.advance(step * 0.002, self.observation(step, load=1, z=0.0202))
        self.assertEqual(control.phase, "lower")
        control.advance(1.2, self.observation(600, load=1, support=0.5))
        self.assertEqual(control.phase, "unload")
        for step in range(601, 710):
            control.advance(step * 0.002, self.observation(step, load=1, support=0.5))
        self.assertEqual(control.phase, "release")

    def test_feedback_motion_and_closure_rates_are_bounded(self):
        control = self.controller()
        previous = control.sources.copy()
        phases = set()
        for step in range(1, 3501):
            bottom = control.carriage
            support = 0.62784 if control.phase in ("lower", "unload", "release", "observe") and bottom < 0.0005 else 0.0
            load = 1 if step > 150 and control.phase not in ("release", "observe") else 0
            observation = self.observation(
                step, load=load, support=support, z=0.0202 + control.carriage, velocity=control.carriage_velocity
            )
            control.advance(step * 0.002, observation)
            phases.add(control.phase)
            np.testing.assert_allclose((control.sources - previous) / 0.002, control.rates, atol=1e-11)
            self.assertLessEqual(np.max(np.abs(control.rates)), 1.0 + 1e-10)
            self.assertLessEqual(abs(control.carriage_velocity), 0.09375 + 1e-10)
            self.assertTrue(np.all(control.sources >= 0) and np.all(control.sources <= 3))
            previous = control.sources.copy()
        self.assertTrue({"close", "move", "hold", "lower", "unload", "release", "observe"}.issubset(phases))
        self.assertTrue(control.completed)
        np.testing.assert_array_equal(control.sources, np.zeros(5))

    def test_open_targets_do_not_certify_actual_release(self):
        for held in ("force", "geometry", "joint", "palm-force", "palm-geometry"):
            control = self.controller()
            control.phase = "release"
            for step in range(1, 200):
                observation = self.observation(
                    step, load=0.1 if held == "force" else 0, depth=0.0005 if held == "geometry" else 0
                )
                observation.open_error_rad = 0.03 if held == "joint" else 0
                observation.hand_object_force = 0.2 if held == "palm-force" else 0
                observation.hand_object_penetration = 0.0005 if held == "palm-geometry" else 0
                control.advance(step * 0.002, observation)
            self.assertEqual(control.phase, "release")
            self.assertFalse(control.completed)

    def test_palm_object_load_is_observed_without_finger_load(self):
        pair = {
            "source": 1,
            "target": 3,
            "force_generation": 5,
            "world_wrench": [[0.2, 0, 0, 0, 0, 0], [-0.2, 0, 0, 0, 0, 0]],
        }
        fingers, floor, support, hand_object = contact_loads([pair], {1: None}, {3}, {2}, material=False, generation=5)
        self.assertEqual(fingers.sum(), 0)
        self.assertEqual(floor.sum(), 0)
        self.assertEqual(support, 0)
        self.assertEqual(hand_object, 0.2)

    def test_measure_separates_object_force_in_both_control_modes(self):
        def array(values):
            return SimpleNamespace(numpy=lambda: np.asarray(values))

        scene = HandTask.__new__(HandTask)
        scene.hand_dofs, scene.hand_coords, scene.hand_bodies = [0], [0], [0]
        scene.hand_shape_fingers, scene.object_shapes, scene.ground_shapes = {1: 0}, {3}, {2}
        scene.identity = {"object_body": 1, "object_dof": None}
        scene.model = SimpleNamespace(
            joint_limit_lower=array([-1]),
            joint_limit_upper=array([1]),
            joint_velocity_limit=array([1]),
            joint_effort_limit=array([1]),
        )
        scene.state = SimpleNamespace(
            joint_q=array([0]),
            joint_qd=array([0]),
            body_q=array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0.02, 0, 0, 0, 1]]),
        )
        stats = SimpleNamespace(
            converged=True,
            rolled_back=False,
            nonlinear_iterations=1,
            linear_iterations=1,
            joint_pd_force=[0],
            failure_reason=None,
            variational={},
        )
        forces = SimpleNamespace(body_wrench=np.array([[0, 0, 2, 0, 0, 0]]), force_valid=True, force_generation=5)
        scene.solver = SimpleNamespace(last_stats=stats, get_contact_forces=lambda state: forces)
        scene.feedback = None
        pair = {"source": 1, "force_generation": 5, "world_wrench": [[0, 0, 2, 0, 0, 0], [0, 0, -2, 0, 0, 0]]}
        for controller in (None, self.controller()):
            scene.controller = controller
            for target, expected in ((2, 0.0), (3, 2.0)):
                stats.variational = {"actor_contact_pairs": [{**pair, "target": target}]}
                row = scene.measure(0.002, "close")
                self.assertEqual(row["contact_force_N"], 2.0)
                self.assertEqual(row["hand_object_force_N"], expected)
            for invalid in ("missing", "stale", "nonfinite", "invalid-force", "failed"):
                with self.subTest(control_mode="open-loop" if controller is None else "contact", invalid=invalid):
                    bad = {**pair, "target": 3}
                    if invalid == "stale":
                        bad["force_generation"] = 4
                    elif invalid == "nonfinite":
                        bad["world_wrench"] = [[float("nan")] * 6] * 2
                    stats.variational = {} if invalid == "missing" else {"actor_contact_pairs": [bad]}
                    forces.force_valid = invalid != "invalid-force"
                    stats.converged = invalid != "failed"
                    self.assertIsNone(scene.measure(0.002, "close")["hand_object_force_N"])
            forces.force_valid, stats.converged = True, True

    def test_failed_slider_saves_physics_and_control_without_pd_stats(self):
        def array(values):
            return SimpleNamespace(numpy=lambda: np.asarray(values))

        scene = HandTask.__new__(HandTask)
        scene.hand_dofs, scene.hand_coords, scene.hand_bodies = [0], [0], [0]
        scene.identity = {"object_body": 1, "object_dof": 1, "surface_sdf_error_m": {"test": 0}}
        scene.smooth = SimpleNamespace(identity={})
        scene.model = SimpleNamespace(
            joint_limit_lower=array([-1, -1]),
            joint_limit_upper=array([1, 1]),
            joint_velocity_limit=array([1, 1]),
            joint_effort_limit=array([1, 1]),
        )
        scene.state = SimpleNamespace(
            joint_q=array([0, 0]),
            joint_qd=array([0, 0]),
            body_q=array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0.02, 0, 0, 0, 1]]),
            particle_q=array(np.empty((0, 3))),
        )
        scene.control = SimpleNamespace(joint_f=array([0, 0]))
        scene.controller = self.controller("pull-slider")
        scene.feedback = scene.feedback_geometry = None
        scene.control_before_step = {"time": 0.002, "controller": scene.controller.snapshot(), "feedback": None}
        stats = SimpleNamespace(
            converged=False,
            rolled_back=True,
            nonlinear_iterations=1,
            linear_iterations=7,
            joint_pd_force=None,
            status="REGULARIZATION_EXHAUSTED",
            failure_reason="rejected",
        )
        scene.solver = SimpleNamespace(
            last_stats=stats,
            _kinematics=SimpleNamespace(digest=b"test"),
            get_contact_forces=lambda state: SimpleNamespace(body_wrench=np.zeros((2, 6))),
            save_checkpoint=lambda state, path: Path(path).write_bytes(b"stub checkpoint"),
        )
        scene.step = lambda t: "close"
        scene.penetration = lambda: {"hand_object_m": 0.0}
        row = scene.measure(0.002, "close")
        self.assertIsNone(row["object_active_force_N"])
        self.assertIsNone(row["effort_ratio"])
        self.assertIsNone(row["hand_object_force_N"])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory),
                assets=Path(directory),
                friction=0.5,
                offset=0,
                surface_source="cad",
                height=0,
                control_mode="contact",
                contact_stiffness=1e8,
                max_steps=1,
            )
            with (
                patch("scripts.monolithic_reference.run_hand_tasks.HandTask", return_value=scene),
                patch("builtins.print"),
            ):
                result = run_case("pull-slider", args)
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["passive_object"])
            self.assertFalse(result["gates"]["effort"])
            self.assertFalse(result["gates"]["contact"])
            self.assertTrue((Path(directory) / "pull-slider/last-published-state.npz").is_file())
            saved = json.loads((Path(directory) / "pull-slider/control-before-failed-step.json").read_text())
            self.assertEqual(saved, scene.control_before_step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
