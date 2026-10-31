# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check physical-axis admittance against independent dynamics and safety invariants."""

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.monolithic_reference.admittance_control import AxisAdmittance
from scripts.monolithic_reference.run_admittance_gripper import AdmittanceGripper, evaluate


class TestAxisAdmittance(unittest.TestCase):
    @staticmethod
    def controller(**overrides):
        """Create a translational controller with explicit SI parameters."""
        parameters = {"mass": 0.2, "damping": 2.0, "position_limits": (-1.0, 1.0), "max_speed": 1.0}
        return AxisAdmittance(**(parameters | overrides))

    def test_constant_force_matches_continuous_solution(self):
        """Converge toward the analytic mass-damper response under timestep refinement."""
        errors = []
        duration, force = 0.2, 0.5
        expected_v = force / 2 * (1 - math.exp(-2 / 0.2 * duration))
        expected_x = force / 2 * (duration - 0.2 / 2 * (1 - math.exp(-2 / 0.2 * duration)))
        for dt in (0.002, 0.001):
            controller = self.controller()
            for _ in range(round(duration / dt)):
                controller.advance(force, dt)
            errors.append(abs(controller.position - expected_x) + abs(controller.velocity - expected_v))
        self.assertLess(errors[1], 0.55 * errors[0])
        self.assertLess(errors[1], 0.0005)

    def test_spring_equilibrium_and_energy_decay(self):
        """Recover the spring equilibrium and dissipate unforced virtual energy."""
        controller = self.controller(stiffness=20.0, reference=0.01)
        for _ in range(2000):
            controller.advance(0.2, 0.002)
        self.assertAlmostEqual(controller.position, 0.02, delta=1e-8)
        previous = float("inf")
        for _ in range(1000):
            controller.advance(0.0, 0.002)
            energy = 0.5 * 0.2 * controller.velocity**2 + 0.5 * 20 * (controller.position - 0.01) ** 2
            self.assertLessEqual(energy, previous + 1e-15)
            previous = energy
        self.assertAlmostEqual(controller.position, 0.01, delta=1e-6)

    def test_position_limit_has_no_hidden_windup(self):
        """Reverse promptly after a long saturated command without accumulated displacement."""
        controller = self.controller(position_limits=(0.0, 0.01))
        for _ in range(1000):
            controller.advance(10.0, 0.002)
        self.assertEqual(controller.position, 0.01)
        self.assertEqual(controller.velocity, 0.0)
        controller.advance(-0.5, 0.002)
        self.assertLess(controller.position, 0.01)
        self.assertLess(controller.velocity, 0.0)

    def test_protection_overrides_inward_velocity(self):
        """Honor a retreat command immediately and keep position and velocity consistent."""
        controller = self.controller()
        for _ in range(20):
            controller.advance(1.0, 0.002)
        before = controller.position
        position, velocity = controller.advance(1.0, 0.002, velocity_bounds=(-0.1, -0.05))
        self.assertLessEqual(velocity, -0.05 + 1e-12)
        self.assertAlmostEqual(position - before, 0.002 * velocity)
        controller.reset(0.0)
        self.assertEqual(controller.velocity, 0.0)

    def test_velocity_limit_and_zero_error_damping(self):
        """Bound motion and decay residual velocity when force error disappears."""
        controller = self.controller(max_speed=0.01)
        for _ in range(100):
            before = controller.position
            position, velocity = controller.advance(100.0, 0.002)
            self.assertLessEqual(abs(velocity), 0.01 + 1e-12)
            self.assertAlmostEqual(position - before, 0.002 * velocity)
        initial = controller.velocity
        for _ in range(1000):
            controller.advance(0.0, 0.002)
        self.assertLess(abs(controller.velocity), initial * 1e-8)

    def test_saturated_state_can_be_restored(self):
        """Restore every applied speed-limited state despite subtraction roundoff."""
        controller = self.controller(max_speed=0.01)
        for _ in range(100):
            controller.advance(100.0, 0.002)
            before = (controller.position, controller.velocity)
            controller.reset(*before)
            self.assertEqual((controller.position, controller.velocity), before)

    def test_invalid_inputs_do_not_change_state(self):
        """Reject nonfinite inputs and invalid limits before advancing state."""
        for kwargs in (
            {"mass": 0},
            {"damping": -1},
            {"stiffness": -1},
            {"max_speed": 0},
            {"position_limits": (1, -1)},
            {"reference": math.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.controller(**kwargs)
        controller = self.controller()
        controller.advance(0.1, 0.002)
        before = vars(controller).copy()
        for force, dt, bounds in (
            (math.nan, 0.002, None),
            (0, 0, None),
            (0, math.inf, None),
            (0, 0.002, (1, -1)),
            (0, 0.002, (2, 3)),
        ):
            with self.subTest(force=force, dt=dt, bounds=bounds), self.assertRaises(ValueError):
                controller.advance(force, dt, velocity_bounds=bounds)
            self.assertEqual(vars(controller), before)
        with self.assertRaises(ValueError):
            controller.reset(2.0)
        self.assertEqual(vars(controller), before)


class TestAdmittanceFeedback(unittest.TestCase):
    @staticmethod
    def scene():
        """Isolate publication and rollback contracts without a physical simulation."""
        scene = AdmittanceGripper.__new__(AdmittanceGripper)
        scene.steps, scene.last_generation = 1, 0
        scene.controllers = [TestAxisAdmittance.controller(), TestAxisAdmittance.controller()]
        scene.feedback = {"valid": True, "generation": 1, "depth_m": [0, 0], "load_N": [0, 0]}
        scene.control = SimpleNamespace(joint_target_q=Mock(), joint_target_qd=Mock())
        scene.state, scene.solver = object(), Mock()
        return scene

    def test_stale_feedback_does_not_move_targets(self):
        """Refuse reused measurements before changing reference state or calling the solver."""
        scene = self.scene()
        scene.last_generation = 1
        with self.assertRaises(ValueError):
            scene.step()
        self.assertEqual(scene.controllers[0].position, 0)
        scene.solver.step.assert_not_called()

    def test_failed_step_restores_controller(self):
        """Restore virtual state and observation identity when physical stepping fails."""
        scene = self.scene()
        scene.solver.step.side_effect = RuntimeError("intentional failure")
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            scene.step()
        self.assertEqual(scene.controllers[0].position, 0)
        self.assertEqual(scene.controllers[0].velocity, 0)
        self.assertEqual(scene.steps, 1)
        self.assertEqual(scene.last_generation, 0)

    def test_gate_requires_complete_physical_run(self):
        """Reject missing hold windows, excessive force, penetration and incomplete release."""
        rows = []
        for step in range(1, 1001):
            target = 0.1 if step <= 350 else 0.2 if step <= 700 else 0.0
            rows.append(
                {
                    "step": step,
                    "force_N": [target, target],
                    "target_force_N": target,
                    "joint_velocity_m_s": [0, 0],
                    "gap_m": [0.001, 0.001],
                    "converged": True,
                }
            )
        self.assertTrue(all(evaluate(rows)[1].values()))
        json.dumps(evaluate(rows))
        self.assertFalse(evaluate(rows[:-1])[1]["complete"])
        self.assertFalse(evaluate(rows[:100])[1]["low-force"])
        rows[300]["force_N"] = [1.0, 0.1]
        self.assertFalse(evaluate(rows)[1]["low-force"])
        rows[400]["gap_m"] = [-0.002, 0]
        self.assertFalse(evaluate(rows)[1]["penetration"])
        rows[-1]["gap_m"] = [0, 0]
        self.assertFalse(evaluate(rows)[1]["release"])

    def test_published_trace_is_serializable(self):
        """Serialize real observations after the first feedback-driven physical step."""
        scene = AdmittanceGripper(device="cpu")
        for _ in range(2):
            row = scene.step()
            self.assertTrue(row["converged"])
            self.assertEqual(json.loads(json.dumps(row)), row)


if __name__ == "__main__":
    unittest.main()
