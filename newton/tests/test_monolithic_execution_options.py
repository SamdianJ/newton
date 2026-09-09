# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check construction policies and shared stepping across P2 execution options."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic import solver_monolithic as impl
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests import test_solver_monolithic_friction as friction_tests
from newton.tests import test_solver_monolithic_sharpa as joint_tests
from newton.tests import test_solver_monolithic_step as step_tests
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _options():
    for owned, mode in itertools.product((False, True), ("diagnostic", "production")):
        yield {"use_optimized_articulation_mass_matrix": owned, "pcg_mode": mode}


class TestMonolithicExecutionOptions(unittest.TestCase):
    def test_invalid_options_before_allocation(self):
        """Reject malformed execution policies before constructing model buffers."""
        cases = [{"use_optimized_articulation_mass_matrix": v} for v in (0, 1, None, "true", np.bool_(True))]
        cases += [{"pcg_mode": v} for v in (None, True, 1, "unknown", "PRODUCTION")]
        for options in cases:
            with self.subTest(options=options), patch.object(impl, "_build_layout") as allocate:
                with self.assertRaises(ValueError):
                    SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, **options)
                with self.assertRaises(ValueError):
                    SolverMonolithic._create_joint_diagnostic(
                        None, joint_terms=SolverMonolithic.JointTerms(), **options
                    )
                allocate.assert_not_called()


def test_configuration_and_step_parity(test, device):
    """Preserve loaded physical stepping while freezing all four execution choices."""
    reference = None
    for options in _options():
        fixture, solver = step_tests._free_scene(device, gravity=(0, 0, -9.81), **options)
        test.assertEqual(solver.pcg_mode, SolverMonolithic.PCGMode(options["pcg_mode"]))
        test.assertEqual(solver._linear.pcg_mode, options["pcg_mode"])
        test.assertEqual(
            solver._articulation.use_optimized_articulation_mass_matrix,
            options["use_optimized_articulation_mass_matrix"],
        )
        for key, value in options.items():
            with test.assertRaises(AttributeError):
                setattr(solver, key, value)
        states, decisions = [], []
        state, state_next = fixture.state, fixture.state_next
        for _ in range(3):
            solver.step(state, state_next, fixture.control, None, 0.001)
            stats = solver.last_stats
            test.assertEqual(stats.status, SolverMonolithic.Status.SUCCESS)
            test.assertGreater(stats.linear_iterations, 0)
            test.assertGreaterEqual(stats.published_contact_generation, 0)
            decisions.append(
                (
                    stats.status,
                    stats.linear_iterations,
                    stats.true_residual_recomputations,
                    stats.residual_replacements,
                    stats.rolled_back,
                )
            )
            states.append(step_tests._snapshot(state_next))
            if options["pcg_mode"] == "production":
                for field in ("min_p_ap", "min_r_z", "initial_guess_norm", "recursive_true_residual_gap"):
                    test.assertTrue(np.isnan(getattr(stats, field)), field)
            state, state_next = state_next, state
        if reference is None:
            reference = states, decisions
        else:
            test.assertEqual(decisions, reference[1])
            for actual, expected in zip(states, reference[0], strict=True):
                for key in actual:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=5e-5, atol=1e-7)
    _, default = step_tests._free_scene(device)
    test.assertFalse(default.use_optimized_articulation_mass_matrix)
    test.assertIs(default.pcg_mode, SolverMonolithic.PCGMode.DIAGNOSTIC)


def test_shared_transactions(test, device):
    """Reuse rollback, final-publication and in-place gates in every execution mode."""
    original = SolverMonolithic.__init__
    for options in _options():

        def create(self, *args, options=options, **kwargs):
            original(self, *args, **kwargs, **options)
            test.assertEqual(self.pcg_mode.value, options["pcg_mode"])
            test.assertEqual(
                self.use_optimized_articulation_mass_matrix, options["use_optimized_articulation_mass_matrix"]
            )

        with test.subTest(options=options), patch.object(SolverMonolithic, "__init__", create):
            step_tests.test_stationary_current_convergence(test, device)
            step_tests.test_hard_failure_rollback(test, device)
            step_tests.test_final_failure_in_place_republication(test, device)
            step_tests.test_frozen_forces_and_in_place(test, device)
            step_tests.test_trial_recollision_preserves_owners(test, device)
            step_tests.test_regularization_budget_keeps_current_assembly(test, device)
            step_tests.test_permanent_final_failure_invalidates_cache(test, device)
            friction_tests.test_active_history_rollback(test, device)


def test_q_only_policies(test, device):
    """Pass execution policies through the private q-only factory and preserve final forces."""
    for options in _options():

        def create(device, options=options):
            model = joint_tests.make_model(device)
            solver = SolverMonolithic._create_joint_diagnostic(
                model,
                joint_terms=SolverMonolithic.JointTerms(
                    implicit_pd=True,
                    limits=True,
                    friction=True,
                    limit_width=(0.01, 0.01),
                    friction_velocity_scale=(0.01, 0.01),
                ),
                **options,
            )
            test.assertEqual(solver._linear.layout.dynamic_particle_count, 0)
            return model, solver

        with test.subTest(options=options), patch.object(joint_tests, "make_solver", side_effect=create):
            joint_tests.test_q_only_assembly(test, device)
            joint_tests.test_q_only_transactions(test, device)


for function in (test_configuration_and_step_parity, test_shared_transactions, test_q_only_policies):
    add_function_test(TestMonolithicExecutionOptions, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
