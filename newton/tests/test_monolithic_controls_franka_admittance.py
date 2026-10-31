# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check Franka opening coordinates, published feedback and controller transactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scripts.monolithic_reference.franka_admittance import FrankaFingerAdmittance
from scripts.monolithic_reference.run_franka_admittance import FrankaAdmittance, evaluate


class TestFrankaAdmittance(unittest.TestCase):
    @staticmethod
    def forces(loads=(0.0, 0.0), *, generation=1, valid=True):
        """Create published generalized forces in opening coordinates [N]."""
        return SimpleNamespace(force_valid=valid, force_generation=generation, joint_force=np.r_[np.zeros(7), loads])

    def test_opposing_force_reverses_closure(self):
        """Close below the force target and open above it with the correct coordinate sign."""
        controller = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        for generation in range(1, 201):
            q, v = controller.command("close", self.forces(generation=generation), 0.001)
        self.assertTrue(np.all(q < 0.04))
        self.assertTrue(np.all(v < 0))
        before = q.copy()
        for generation in range(201, 301):
            q, v = controller.command("hold", self.forces((2.0, 2.0), generation=generation), 0.001)
        self.assertTrue(np.all(q > before))
        self.assertTrue(np.all(v > 0))

    def test_each_finger_uses_its_own_signed_force(self):
        """Respond independently to unequal loads instead of canceling opposing world forces."""
        controller = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        q, v = controller.command("close", self.forces((0.0, 1.0)), 0.001)
        self.assertLess(q[0], 0.04)
        self.assertEqual(q[1], 0.04)
        self.assertLess(v[0], 0)
        self.assertEqual(v[1], 0)

    def test_invalid_feedback_is_atomic(self):
        """Reject stale, invalid or nonfinite observations before changing either finger."""
        controller = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        controller.command("close", self.forces(), 0.001)
        before = controller.snapshot()
        for observation in (
            self.forces(),
            self.forces(generation=2, valid=False),
            self.forces((0.0, float("nan")), generation=2),
        ):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                controller.command("hold", observation, 0.001)
            self.assertEqual(controller.snapshot(), before)

    def test_release_reaches_open_travel_without_windup(self):
        """Open under residual contact load and remain at the physical travel boundary."""
        controller = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        for generation in range(1, 501):
            controller.command("close", self.forces(generation=generation), 0.001)
        for generation in range(501, 1501):
            q, v = controller.command("release", self.forces((2.0, 2.0), generation=generation), 0.001)
        np.testing.assert_array_equal(q, [0.04, 0.04])
        np.testing.assert_array_equal(v, [0, 0])
        q, _ = controller.command("close", self.forces(generation=1501), 0.001)
        self.assertTrue(np.all(q < 0.04))

    def test_restore_replays_the_same_command(self):
        """Restore virtual dynamics and consumed generation after a failed physical step."""
        controller = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        before = controller.snapshot()
        expected = controller.command("close", self.forces(), 0.001)
        controller.restore(before)
        self.assertEqual(controller.snapshot(), before)
        actual = controller.command("close", self.forces(), 0.001)
        np.testing.assert_array_equal(actual, expected)

    def test_sensor_storage_precision_does_not_change_virtual_dynamics(self):
        """Produce identical commands from identical forces stored in FP32 or FP64 arrays."""
        controllers = [FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04]) for _ in range(2)]
        for generation in range(1, 101):
            outputs = []
            for controller, dtype in zip(controllers, (np.float32, np.float64), strict=True):
                observation = self.forces(generation=generation)
                observation.joint_force = observation.joint_force.astype(dtype)
                outputs.append(controller.command("close", observation, 0.001))
            np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_failed_physics_restores_adapter_and_pd_command(self):
        """Restore both the virtual state and actual control arrays on a solver failure."""
        scene = FrankaAdmittance.__new__(FrankaAdmittance)
        scene.failed, scene.steps = False, 1500
        scene.high = scene.low = np.r_[np.zeros(7), [0.04, 0.04]]
        scene.admittance = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        before = scene.admittance.snapshot()
        scene.control = SimpleNamespace(joint_target_q=Mock(), joint_target_qd=Mock())
        scene.control.joint_target_q.numpy.return_value = scene.high.copy()
        scene.control.joint_target_qd.numpy.return_value = np.zeros(9)
        scene.state, scene.solver = object(), Mock()
        scene.solver.get_contact_forces.return_value = self.forces()
        scene.solver.step.side_effect = RuntimeError("intentional failure")
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            scene.step()
        self.assertEqual(scene.admittance.snapshot(), before)
        self.assertEqual(scene.steps, 1500)
        self.assertTrue(scene.failed)
        np.testing.assert_array_equal(scene.control.joint_target_q.assign.call_args.args[0], scene.high)
        np.testing.assert_array_equal(scene.control.joint_target_qd.assign.call_args.args[0], np.zeros(9))

    def test_complete_run_gates_cannot_hide_task_failure(self):
        """Fail missing windows, unreleased objects, excessive penetration and force errors."""
        rows = []
        for step in range(1, 5251):
            hold = 3000 <= step < 3500
            rows.append(
                {
                    "step": step,
                    "phase": "hold" if hold else "release",
                    "force_N": [1.0, 1.0] if hold else [0.0, 0.0],
                    "finger_velocity_m_s": [0.0, 0.0],
                    "finger_position_m": [0.04, 0.04],
                    "cube_com_m": [0.45, 0, 0.1 if hold else 0.02],
                    "ground_force_N": [0, 0, 0 if hold else 0.63],
                    "penetration_m": 0.001,
                    "min_det_f": 0.9,
                    "force_projection_error_N": 0.0,
                    "converged": True,
                    "publication": "CONVERGED",
                }
            )
        kwargs = {"initial_com_z": 0.02, "weight": 0.63, "edge": 0.04}
        self.assertTrue(all(evaluate(rows, **kwargs)[1].values()))
        for row in rows:
            row.update(converged=False, publication="STOPPED_ACCEPTED")
        metrics, gates = evaluate(rows, **kwargs)
        self.assertTrue(all(gates.values()))
        self.assertEqual(metrics["counts"], {"CONVERGED": 0, "STOPPED_ACCEPTED": 5250, "HARD_FAILURE": 0})
        rows[0]["publication"] = "HARD_FAILURE"
        self.assertFalse(evaluate(rows, **kwargs)[1]["complete_published"])
        rows[0]["publication"] = "STOPPED_ACCEPTED"
        self.assertFalse(evaluate(rows[:-1], **kwargs)[1]["complete_published"])
        self.assertFalse(evaluate(rows[:3200], **kwargs)[1]["force_tracking"])
        rows[3300]["force_N"] = [1.3, 1.0]
        self.assertFalse(evaluate(rows, **kwargs)[1]["force_tracking"])
        rows[3300]["ground_force_N"] = [0, 0, 0.63]
        self.assertFalse(evaluate(rows, **kwargs)[1]["grasp"])
        rows[-1]["penetration_m"] = 0.003
        self.assertFalse(evaluate(rows, **kwargs)[1]["penetration"])
        rows[-1]["force_N"] = [0.1, 0.0]
        self.assertFalse(evaluate(rows, **kwargs)[1]["release"])

    def test_stopped_publication_keeps_controller_progress(self):
        """Advance the reference and consume a fresh generation after an accepted STOPPED step."""
        scene = FrankaAdmittance.__new__(FrankaAdmittance)
        scene.failed, scene.steps = False, 1500
        scene.high = scene.low = np.r_[np.zeros(7), [0.04, 0.04]]
        scene.admittance = FrankaFingerAdmittance([0.0, 0.0], [0.04, 0.04])
        scene.control = SimpleNamespace(joint_target_q=Mock(), joint_target_qd=Mock())
        scene.control.joint_target_q.numpy.return_value = scene.high.copy()
        scene.control.joint_target_qd.numpy.return_value = np.zeros(9)
        scene.state, scene.solver = object(), Mock()
        scene.solver.last_stats = SimpleNamespace(
            converged=False,
            rolled_back=False,
            status="nonlinear_max_iterations",
            failure_reason="budget",
            variational={"publication_status": "STOPPED_ACCEPTED"},
        )
        for generation in (1, 2):
            scene.solver.get_contact_forces.return_value = self.forces(generation=generation)
            self.assertFalse(scene.step().converged)
            self.assertEqual(scene.admittance.last_generation, generation)
            self.assertEqual(scene.steps, 1500 + generation)
            self.assertFalse(scene.failed)
        before = scene.admittance.snapshot()
        scene.solver.get_contact_forces.return_value = self.forces(generation=3)
        scene.solver.last_stats.rolled_back = True
        scene.solver.last_stats.variational["publication_status"] = "HARD_FAILURE"
        with self.assertRaises(RuntimeError):
            scene.step()
        self.assertEqual(scene.admittance.snapshot(), before)
        self.assertEqual(scene.steps, 1502)


if __name__ == "__main__":
    unittest.main()
