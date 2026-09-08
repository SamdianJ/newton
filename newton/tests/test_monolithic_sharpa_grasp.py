# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the exploratory schedule and single-tree support/carriage assembly."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_ball
from newton.examples.softbody.monolithic_sharpa_grasp import (
    ANCHORED_FIXTURE,
    FIXTURE,
    anchor_ball,
    joint_mapping,
    load_calibration,
    stage_target,
)
from newton.examples.softbody.monolithic_soft_ball import load_ball
from newton.examples.softbody.sharpa_close import build_hand, finalize_hand, load_fixture
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices


class _Trajectory:
    def sample(self, t):
        return np.array([min(t, 2.0)]), np.array([1.0 if t < 2 else 0.0])


class TestGraspSchedule(unittest.TestCase):
    def test_calibration(self):
        """Reject incomplete, stale and altered frozen calibration inputs."""
        source = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json"
        frozen = load_calibration(source)
        self.assertEqual(frozen["stiffness_floor_n_m3"], 2e6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            for field, value in (
                ("status", "DRAFT"),
                ("pcg_iteration_p95_budget", 65),
                ("results", []),
                ("protocol_sha256", "stale"),
                ("stiffness_floor_n_m3", 1e6),
            ):
                changed = {**frozen, field: value}
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_calibration(path)

    def test_phases(self):
        """Preserve closure, holds, release reversal and a smooth bounded lift."""
        trajectory = _Trajectory()
        times = [0.25, 1.5, 3.5, 5.0, 6.0, 7.5, 8.75]
        expected = ["prepare", "close", "hold", "lift", "hold_high", "release", "observe"]
        for t, phase in zip(times, expected, strict=True):
            self.assertEqual(stage_target(trajectory, t)[0], phase)
        self.assertEqual(stage_target(trajectory, 1.5)[2][0], 1)
        self.assertEqual(stage_target(trajectory, 7.5)[2][0], -1)
        for t in (0.0, 0.5, 2.5, 4.5, 5.5, 6.5, 8.5, 9.0):
            _, _q, _, lift, lv = stage_target(trajectory, t)
            self.assertGreaterEqual(lift, 0)
            self.assertLessEqual(lift, 0.05)
            self.assertEqual(lv, 0)
            if 0 < t < 9:
                before = stage_target(trajectory, t - 1e-7)
                after = stage_target(trajectory, t + 1e-7)
                np.testing.assert_allclose(before[1], after[1], atol=3e-7)
                self.assertAlmostEqual(before[3], after[3], delta=1e-7)
        for t in np.linspace(4.51, 5.49, 19):
            h = 1e-5
            fd = (stage_target(trajectory, t + h)[3] - stage_target(trajectory, t - h)[3]) / (2 * h)
            self.assertAlmostEqual(fd, stage_target(trajectory, t)[4], delta=1e-8)
            self.assertLessEqual(stage_target(trajectory, t)[4], FIXTURE["carriage"]["velocity"])
        for t in (-1, 10, np.nan, np.inf):
            with self.assertRaises(ValueError):
                stage_target(trajectory, t)


def test_mount(test, device):
    """Keep hand-relative kinematics intact while only the carriage translates it."""
    asset = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
    if not asset:
        test.skipTest("Set NEWTON_SHARPA_ASSET_DIR for real URDF import")
    parameters = load_fixture(Path(newton.__file__).parent / "examples/softbody/sharpa_g1h.json")
    original, manifest = build_hand(asset, device=device, parameters=parameters)
    baseline, _ = finalize_hand(original, manifest, device=device)
    builder, manifest = build_hand(
        asset, device=device, parameters=parameters, _mount={"support_height": 0.1698, "carriage": FIXTURE["carriage"]}
    )
    model, manifest = finalize_hand(builder, manifest, device=device)
    test.assertEqual(
        (model.body_count, model.joint_dof_count, model.shape_count, model.articulation_count), (34, 23, 55, 1)
    )
    mapping = joint_mapping(model, manifest["joint_names"])
    np.testing.assert_array_equal(model.body_mass.numpy()[1:], baseline.body_mass.numpy())
    np.testing.assert_array_equal(model.body_inertia.numpy()[1:], baseline.body_inertia.numpy())
    state, reference = model.state(), baseline.state()
    q = reference.joint_q.numpy()
    q[:] = np.linspace(-0.1, 0.1, len(q))
    reference.joint_q.assign(q)
    newton.eval_fk(baseline, reference.joint_q, reference.joint_qd, reference)
    targets = state.joint_q.numpy()
    for i, name in enumerate(manifest["joint_names"]):
        targets[mapping[name][0]] = q[i]
    targets[mapping["monolithic_lift"][0]] = 0.05
    state.joint_q.assign(targets)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    actual, expected = state.body_q.numpy(), reference.body_q.numpy()
    np.testing.assert_allclose(actual[1:, :3], expected[:, :3] + [0, 0, 0.05], atol=1e-7)
    np.testing.assert_allclose(actual[1:, 3:], expected[:, 3:], atol=1e-7)
    np.testing.assert_array_equal(actual[0], [0, 0, 0, 0, 0, 0, 1])
    # Exercise the new drive in the shared diagnostic stepper on both devices.
    model.shape_flags.zero_()
    widths, velocities = np.empty(23), np.empty(23)
    for name, (_, dof, _) in mapping.items():
        cfg = FIXTURE["carriage"] if name == "monolithic_lift" else parameters["joints"][name]
        widths[dof], velocities[dof] = cfg["limit_width"], cfg["friction_velocity_scale"]
    solver = SolverMonolithic._create_joint_diagnostic(
        model,
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True,
            limits=True,
            friction=True,
            limit_width=tuple(widths),
            friction_velocity_scale=tuple(velocities),
        ),
    )
    initial, control = model.state(), model.control()
    control.joint_target_q.assign(initial.joint_q)
    for _ in range(3):
        solver.step(initial, initial, control, None, 0.001)
        test.assertTrue(solver.last_stats.converged)
        test.assertFalse(solver.last_stats.rolled_back)


def test_anchor_cap(test, device):
    """Pin the same geometric cap across resolutions, retaining physical mass for diagnostics."""
    for refinement, count in ((1, 5), (2, 13), (3, 29)):
        builder = newton.ModelBuilder()
        ball = Path(__file__).parents[2] / f"scripts/monolithic_reference/fixtures/soft_ball/ball_r{refinement}.npz"
        builder.add_soft_mesh(
            pos=ANCHORED_FIXTURE["ball_position"],
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            mesh=load_ball(ball),
            add_surface_mesh_edges=False,
        )
        model = builder.finalize(device=device)
        mass = model.particle_mass.numpy().copy()
        fixed, rest, weights = anchor_ball(model, ANCHORED_FIXTURE["ball_position"])
        test.assertEqual(int(fixed.sum()), count)
        np.testing.assert_array_equal(weights, mass)
        np.testing.assert_array_equal(model.particle_q.numpy(), rest)
        np.testing.assert_array_equal(model.particle_mass.numpy()[~fixed], mass[~fixed])
        test.assertTrue(np.all(model.particle_inv_mass.numpy()[fixed] == 0))
        test.assertTrue(np.all(model.particle_inv_mass.numpy()[~fixed] > 0))
        with test.assertRaises(ValueError):
            anchor_ball(model, np.array(ANCHORED_FIXTURE["ball_position"]) + np.array([100, 0, 0]))


def test_anchored_step(test, device):
    """Use the coupled stepper with a fixed cap, including its CPU empty-contact path."""
    asset, contact = os.environ.get("NEWTON_SHARPA_ASSET_DIR"), os.environ.get("NEWTON_SHARPA_CONTACT_DIR")
    if not asset or not contact:
        test.skipTest("Set Sharpa asset/contact paths for coupled anchored checks")
    parameters = load_fixture(Path(newton.__file__).parent / "examples/softbody/sharpa_g1h.json")
    ball = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/soft_ball/ball_r1.npz"
    model, manifest = load_hand_ball(
        asset, contact, ball, device=device, parameters=parameters, position=ANCHORED_FIXTURE["ball_position"]
    )
    fixed, rest, _ = anchor_ball(model, ANCHORED_FIXTURE["ball_position"])
    mapping = joint_mapping(model, manifest["joint_names"], carriage=False)
    test.assertEqual(len(mapping), 22)
    test.assertEqual(model.body_count, 33)
    solver = SolverMonolithic(
        model,
        contact_stiffness=1e7,
        collision_pipeline=MonolithicCollisionPipeline(
            model, soft_contact_gap=0.002, _sdf_query_error=0.0004 if wp.get_device(device).is_cuda else 0.0
        ),
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, device=device),
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True,
            limits=True,
            friction=True,
            limit_width=tuple(parameters["joints"][n]["limit_width"] for n in mapping),
            friction_velocity_scale=tuple(parameters["joints"][n]["friction_velocity_scale"] for n in mapping),
        ),
    )
    state, control = model.state(), model.control()
    control.joint_target_q.assign(state.joint_q)
    for _ in range(3):
        solver.step(state, state, control, None, 0.001)
        test.assertTrue(solver.last_stats.converged)
        np.testing.assert_array_equal(state.particle_q.numpy()[fixed], rest[fixed])
        np.testing.assert_array_equal(state.particle_qd.numpy()[fixed], 0)


class TestGraspMount(unittest.TestCase):
    pass


add_function_test(TestGraspMount, "test_mount", test_mount, devices=get_test_devices())
add_function_test(TestGraspMount, "test_anchor_cap", test_anchor_cap, devices=get_test_devices())
add_function_test(TestGraspMount, "test_anchored_step", test_anchored_step, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
