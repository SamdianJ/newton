# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check folding targets and the single-step control transaction without a rollout."""

import copy
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.cloth_franka_scene import (
    DEFAULT_RECIPE,
    FrankaClothScene,
    build_model,
    fold_targets,
    load_recipe,
    plan_joint_targets,
    tcp_targets,
)


class Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.asarray(value).copy()


class TestClothFoldControls(unittest.TestCase):
    def test_low_grasp_targets_clear_actual_finger_boxes(self):
        """Keep the real fingertip boxes above the table while preserving low grasp reach."""
        recipe = load_recipe(DEFAULT_RECIPE)
        recipe["keyframes"] = recipe["keyframes"][:3]
        model, _, _ = build_model(recipe, device="cpu", cache="/tmp/cloth-fold-test", prepare_sdf=False)
        times, targets, phases, plan = plan_joint_targets(model, recipe)
        bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        ids = [
            i
            for i, b in enumerate(bodies)
            if b >= 0 and "finger" in model.body_label[b] and flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
        ]
        state = model.state()
        signs = np.array(list(product((-1, 1), repeat=3)))
        minima = []
        for i, phase in enumerate(phases):
            if phase not in ("left_sleeve_approach", "left_sleeve_close"):
                continue
            for fraction in (0.0, 0.5, 1.0):
                q, _, _ = fold_targets(
                    times,
                    targets,
                    times[i] + fraction * (times[i + 1] - times[i]),
                    phase_times=np.asarray(plan["phase_times_s"]),
                )
                state.joint_q.assign(q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                poses = state.body_q.numpy()
                minimum = min(
                    float(
                        wp.transform_point(
                            wp.transform(*poses[bodies[s]]) * wp.transform(*model.shape_transform.numpy()[s]),
                            wp.vec3(corner * model.shape_scale.numpy()[s]),
                        )[2]
                    )
                    for s in ids
                    for corner in signs
                )
                self.assertGreaterEqual(minimum, 0.2)
                if phase == "left_sleeve_close":
                    minima.append(minimum)
        self.assertTrue(minima)
        self.assertLess(max(minima), 0.201)

    def test_only_finger_table_rigid_collision_is_enabled(self):
        """Preserve particle colliders and activate exactly the supported eight static pairs."""
        recipe = load_recipe(DEFAULT_RECIPE)
        model, _, _ = build_model(recipe, device="cpu", cache="/tmp/cloth-fold-test", prepare_sdf=False)
        body, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        enabled = set(np.flatnonzero(flags & int(newton.ShapeFlags.COLLIDE_SHAPES)))
        fingers = {
            i
            for i, b in enumerate(body)
            if b >= 0 and "finger" in model.body_label[b] and flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
        }
        table = {i for i, b in enumerate(body) if b < 0 and model.shape_type.numpy()[i] == newton.GeoType.BOX}
        self.assertEqual(len(fingers), 8)
        self.assertEqual(enabled, fingers | table)
        pairs = {tuple(sorted(p)) for p in model.shape_contact_pairs.numpy() if all(s in enabled for s in p)}
        self.assertEqual(pairs, {tuple(sorted((s, t))) for s in fingers for t in table})

    def test_cartesian_nodes_do_not_add_stops(self):
        """Keep target velocity continuous through IK nodes and stop only at phase boundaries."""
        times = np.linspace(0, 2, 9)
        poses = np.sin(times)[:, None]
        boundaries = np.array([0.0, 2.0])
        for time in (0.0, 2.0):
            _, velocity, _ = fold_targets(times, poses, time, phase_times=boundaries)
            np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
        left = fold_targets(times, poses, 1 - 1e-7, phase_times=boundaries)
        right = fold_targets(times, poses, 1 + 1e-7, phase_times=boundaries)
        self.assertGreater(left[1][0], 0.1)
        np.testing.assert_allclose(left[0], right[0], atol=3e-7)
        np.testing.assert_allclose(left[1], right[1], atol=1e-6)
        for time in (0.37, 0.81, 1.43):
            h = 1e-6
            numeric = (
                fold_targets(times, poses, time + h, phase_times=boundaries)[0]
                - fold_targets(times, poses, time - h, phase_times=boundaries)[0]
            ) / (2 * h)
            analytic = fold_targets(times, poses, time, phase_times=boundaries)[1]
            np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)

    def test_ik_interpolation_does_not_overshoot_joint_range(self):
        """Prevent valid IK nodes near a joint limit from producing out-of-range commands."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        poses = np.array([[0.0], [0.99], [1.0], [0.99]])
        for time in np.linspace(0, 3, 121):
            value, _, _ = fold_targets(times, poses, time, phase_times=np.array([0.0, 3.0]))
            self.assertGreaterEqual(value[0], 0.0)
            self.assertLessEqual(value[0], 1.0)

    def test_targets_use_absolute_physical_time(self):
        times = np.array([0.0, 2.0, 5.0])
        q = np.array([[0.0, 0.04], [1.0, 0.001], [-0.5, 0.04]])
        for i, time in enumerate(times):
            actual, velocity, _ = fold_targets(times, q, time)
            np.testing.assert_allclose(actual, q[i])
            np.testing.assert_allclose(velocity, 0.0)
        for time in (0.7, 2.6, 4.0):
            actual, velocity, index = fold_targets(times, q, time)
            h = 1e-5
            numeric = (fold_targets(times, q, time + h)[0] - fold_targets(times, q, time - h)[0]) / (2 * h)
            np.testing.assert_allclose(velocity, numeric, rtol=1e-7, atol=1e-9)
            self.assertEqual(index, int(time >= 2))
            self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(fold_targets(times, q, 20)[0], q[-1])

    def test_tcp_frame_mapping_preserves_link_pose(self):
        recipe = load_recipe(DEFAULT_RECIPE)
        positions, rotations = tcp_targets(recipe)
        row = recipe["keyframes"][0]
        q = np.asarray(row["quaternion_xyzw"], dtype=float)
        q /= np.linalg.norm(q)
        v = np.asarray(recipe["robot"]["tcp_offset_m"]) - recipe["robot"]["source_tcp_offset_m"]
        rotated = v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)
        np.testing.assert_allclose(positions[0], np.asarray(row["position_m"]) + rotated)
        self.assertGreater(np.linalg.norm(positions[0] - row["position_m"]), 0.009)
        np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1.0)
        self.assertEqual(len(positions), 33)

    def test_invalid_recipe_rejected_before_assets(self):
        recipe = load_recipe(DEFAULT_RECIPE)
        for path, value in (
            (["dt"], 0),
            (["material", "thickness_m"], -1),
            (["keyframes", 0, "duration_s"], float("nan")),
        ):
            candidate = copy.deepcopy(recipe)
            current = candidate
            for key in path[:-1]:
                current = current[key]
            current[path[-1]] = value
            with self.assertRaises(ValueError):
                load_recipe(candidate)
        for time in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fold_targets(np.array([0, 1]), np.zeros((2, 9)), time)

    @staticmethod
    def scene(publication="CONVERGED", *, rolled_back=False, error=None):
        scene = FrankaClothScene.__new__(FrankaClothScene)
        scene.dt, scene.total_steps, scene.steps, scene.failed = 0.1, 4, 0, False
        scene.last_forces, scene._published_generation = None, 0
        scene.times = np.array([0.0, 0.4])
        scene.phase_times = scene.times
        scene.joint_targets = np.array([np.zeros(9), np.ones(9)])
        scene.phases, scene.phase = ["grasp"], "initial"
        scene.control = SimpleNamespace(joint_target_q=Array(np.zeros(9)), joint_target_qd=Array(np.zeros(9)))
        scene.state = object()
        stats = SimpleNamespace(
            rolled_back=rolled_back,
            variational={"publication_status": publication},
            step_generation=1,
            converged=publication == "CONVERGED",
        )
        scene.solver = SimpleNamespace(
            last_stats=stats,
            step=Mock(side_effect=error),
            get_contact_forces=Mock(return_value=SimpleNamespace(force_valid=True, force_generation=1)),
        )
        return scene

    def test_one_solver_step_and_clock_advance_for_valid_publication(self):
        for publication in ("CONVERGED", "STOPPED_ACCEPTED"):
            scene = self.scene(publication)
            scene.step()
            self.assertEqual(scene.steps, 1)
            self.assertEqual(scene.phase, "grasp")
            scene.solver.step.assert_called_once_with(scene.state, scene.state, scene.control, dt=0.1)
            expected = fold_targets(scene.times, scene.joint_targets, 0.1)
            np.testing.assert_allclose(scene.control.joint_target_q.numpy(), expected[0])
            np.testing.assert_allclose(scene.control.joint_target_qd.numpy(), expected[1])
            scene.solver.get_contact_forces.assert_called_once_with(scene.state)
            self.assertIs(scene.last_forces, scene.solver.get_contact_forces.return_value)

    def test_invalid_force_generation_is_not_published(self):
        scene = self.scene()
        scene.solver.get_contact_forces.return_value.force_generation = 0
        with self.assertRaises(RuntimeError):
            scene.step()
        self.assertEqual(scene.steps, 0)
        self.assertIsNone(scene.last_forces)

    def test_failed_publication_restores_control_and_clock(self):
        for scene in (
            self.scene("HARD_FAILURE", rolled_back=True),
            self.scene("UNKNOWN"),
            self.scene(error=RuntimeError("failure")),
        ):
            with self.assertRaises(RuntimeError):
                scene.step()
            self.assertEqual(scene.steps, 0)
            self.assertTrue(scene.failed)
            self.assertEqual(scene.phase, "initial")
            np.testing.assert_array_equal(scene.control.joint_target_q.numpy(), np.zeros(9))
            np.testing.assert_array_equal(scene.control.joint_target_qd.numpy(), np.zeros(9))
            scene.step()
            scene.solver.step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
