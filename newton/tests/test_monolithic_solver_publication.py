# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify opt-in coupled production policies without weakening strict publication."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.contact import ContactObjective, ContactStatus
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.tests.test_monolithic_contact_components import fixture, native, state
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def candidate(model, *, publish_policy="last_accepted", **options):
    return solver(model, alignment_policy="mask", publish_policy=publish_policy, **options)


def test_policy_binding(test, device):
    """Report heuristic energy under masking and bind all policies into state identity."""
    model = scene(device)
    strict = solver(model)
    original = strict.create_state(dt=0.001)
    run = candidate(model)
    with test.assertRaises(ValueError):
        run._states.validate(original)
    before = run.create_state(dt=0.001)
    with test.assertRaises(ValueError):
        strict._states.validate(before)
    run.step(before, before, model.control(), dt=0.001)
    test.assertFalse(run.last_stats.rolled_back, run.last_stats)
    test.assertFalse(run._conservative_energy())
    test.assertEqual(run.last_stats.variational["energy_acceptance"], "heuristic")
    for options in ({"alignment_policy": "invalid"}, {"publish_policy": "legacy"}):
        with test.assertRaises(ValueError):
            solver(model, **options)


def test_mask_contact(test, device):
    """Mask whole contact terms while retaining invalid-geometry and capacity gates."""
    model, xn = fixture(device, "plane")
    obj = ContactObjective(
        model,
        np.arange(model.particle_count, dtype=np.int32),
        q_dof_count=2,
        normal_stiffness=1e7,
        friction_coefficient=0.5,
        friction_velocity_scale=0.01,
        smoothing_width=1e-4,
        particle_radius=0.001,
        alignment_policy="mask",
    )
    batch = native(obj)
    inverted = xn.copy()
    inverted[[1, 2]] = inverted[[2, 1]]
    batch.begin_step(*state(model, inverted, np.zeros(2)))
    batch.evaluate(*state(model, inverted, np.zeros(2)), dt=0.001)
    test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
    test.assertGreater(batch.omitted.numpy()[:, 2].sum(), 0)
    test.assertEqual(np.count_nonzero(batch.pool.energy.result.numpy()), 0)
    test.assertEqual(np.count_nonzero(batch.pool.residual.result.numpy()), 0)
    test.assertEqual(np.count_nonzero(batch.weights.numpy()), 0)
    test.assertEqual(int(batch.pool.records.count.numpy()[0]), 0)
    for invalid in ("nan", "degenerate"):
        x = inverted.copy()
        if invalid == "nan":
            x[0, 0] = np.nan
        else:
            x[1] = x[0]
        batch.evaluate(*state(model, x, np.zeros(2)), dt=0.001)
        test.assertNotEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
    batch.begin_step(*state(model, xn, np.zeros(2)))
    batch.evaluate(*state(model, xn, np.zeros(2)), dt=0.001)
    test.assertEqual(np.count_nonzero(batch.omitted.numpy()), 0)
    test.assertGreater(int(batch.pool.records.count.numpy()[0]), 0)
    # A zero pool admits no partial contact; the drop is reported, not published.
    empty = _ContactBatch(obj, batch_capacity=obj.count, capacity=0)
    empty.begin_step(*state(model, xn, np.zeros(2)))
    empty.evaluate(*state(model, xn, np.zeros(2)), dt=0.001)
    test.assertEqual(int(empty.pool.records.count.numpy()[0]), 0)
    test.assertGreater(int(empty.pool.records.dropped.numpy()[0]), 0)


def test_last_accepted_publication(test, device):
    """Publish the accepted state after eligible stops and retain exact failure labels."""
    original_iterate = solver_monolithic.iterate
    for policy in ("last_accepted", "current_iterate"):
        for mode, graph in [("diagnostic", False), ("production", False)] + (
            [("production", True)] if device.is_cuda else []
        ):
            for reason in ("nonlinear_max_iterations", "nonlinear_stagnation", "regularization_exhausted"):
                model = scene(device)
                run = candidate(model, publish_policy=policy, newton_max_iterations=1, mode=mode, graph=graph)
                before = run.create_state(dt=0.001)
                after = run._states.copy(before)
                expected = snapshot(before)
                accepted = {}

                def stop(current, dt, accepted=accepted, reason=reason):
                    original_iterate(current, dt)
                    test.assertTrue(any(t.get("accepted") for t in current._variational["trials"]))
                    accepted.update(snapshot(current._accepted))
                    current._trial.particle_q.fill_(wp.vec3d(float("nan")))
                    raise _StepFailure(current.Status(reason), "eligible stop injection")

                with patch.object(solver_monolithic, "iterate", stop):
                    run.step(before, after, model.control(), dt=0.001)
                stats = run.last_stats
                test.assertFalse(stats.converged)
                test.assertFalse(stats.rolled_back, stats)
                test.assertEqual(stats.status.value, reason)
                test.assertEqual(stats.failure_reason, "eligible stop injection")
                test.assertEqual(stats.variational["publication_status"], "STOPPED_ACCEPTED")
                for field in ("joint_q", "particle_q"):
                    np.testing.assert_array_equal(getattr(after, field).numpy(), accepted[field])
                test.assertEqual(after._monolithic_coupled.force_valid.numpy()[0], 1)
                test.assertEqual(after._monolithic_coupled.force_generation.numpy()[0], 1)
                run._states.validate(after)
                assert_snapshot(test, before, expected)


def test_stopped_hard_gates(test, device):
    """Reject zero-accepted stops and invalid final states with complete output rollback."""
    original_iterate = solver_monolithic.iterate
    for policy in ("strict", "last_accepted", "current_iterate"):
        for failure in ("zero", "nan", "tet", "capacity", "force", "binding"):
            if failure == "zero" and policy == "current_iterate":
                continue
            model = scene(device)
            run = candidate(model, publish_policy=policy, newton_max_iterations=1)
            before = run.create_state(dt=0.001)
            after = run._states.copy(before)
            after.particle_f.fill_(wp.vec3(2.0))
            expected, expected_out = snapshot(before), snapshot(after)

            def stop(current, dt, failure=failure):
                if failure != "zero":
                    original_iterate(current, dt)
                if failure == "nan":
                    current._accepted.particle_q.fill_(wp.vec3d(float("nan")))
                elif failure == "tet":
                    current._accepted.particle_q.fill_(wp.vec3d(0.0))
                elif failure == "capacity":
                    raise _StepFailure(current.Status.CONTACT_OVERFLOW, "capacity injection")
                raise _StepFailure(current.Status.REGULARIZATION_EXHAUSTED, "stop injection")

            with patch.object(solver_monolithic, "iterate", stop):
                if failure in ("force", "binding"):
                    with patch.object(
                        run if failure == "force" else run._states,
                        "_finish_candidate" if failure == "force" else "bind",
                        side_effect=_StepFailure(run.Status.NONFINITE, "finalization injection"),
                    ):
                        run.step(before, after, model.control(), dt=0.001)
                else:
                    run.step(before, after, model.control(), dt=0.001)
            test.assertTrue(run.last_stats.rolled_back, (failure, run.last_stats))
            test.assertFalse(run.last_stats.converged)
            test.assertEqual(run.last_stats.variational["publication_status"], "HARD_FAILURE")
            assert_snapshot(test, before, expected)
            assert_snapshot(test, after, expected_out)


def test_current_iterate_publication(test, device):
    """Publish the initialized iterate on zero-update stops, never a rejected trial."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for mode, graph in modes:
        for reason in ("nonlinear_max_iterations", "nonlinear_stagnation", "regularization_exhausted"):
            model = scene(device)
            run = candidate(model, publish_policy="current_iterate", mode=mode, graph=graph)
            before = run.create_state(dt=0.001)
            after = run._states.copy(before)
            expected = snapshot(before)
            predictor = {}

            def stop(alpha, dt, reason=reason, current=run, predictor=predictor):
                predictor.update(snapshot(current._accepted))
                current._trial.particle_q.fill_(wp.vec3d(float("nan")))
                raise _StepFailure(current.Status(reason), "zero-update stop injection")

            with patch.object(run, "_take_trial", stop):
                run.step(before, after, model.control(), dt=0.001)
            stats = run.last_stats
            test.assertFalse(stats.rolled_back, stats)
            test.assertFalse(stats.converged)
            test.assertEqual(stats.status.value, reason)
            test.assertEqual(stats.variational["accepted_count"], 0)
            test.assertTrue(stats.variational["zero_accepted_publication"])
            test.assertEqual(stats.variational["publish_policy"], "current_iterate")
            for field in ("joint_q", "particle_q"):
                np.testing.assert_array_equal(getattr(after, field).numpy(), predictor[field])
            test.assertEqual(after._monolithic_coupled.force_valid.numpy()[0], 1)
            test.assertEqual(after._monolithic_coupled.force_generation.numpy()[0], 1)
            run._states.validate(after)
            assert_snapshot(test, before, expected)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.npz"
                run.save_checkpoint(after, path)
                restored = run.create_state(dt=0.001)
                run.load_checkpoint(restored, path)
                assert_snapshot(test, restored, snapshot(after))

            final_check = run._validate_final
            expected_output = snapshot(after)

            def invalid_final(dt, status, run=run, final_check=final_check):
                run._accepted.particle_q.fill_(wp.vec3d(float("nan")))
                return final_check(dt, status)

            with patch.object(run, "_take_trial", stop), patch.object(run, "_validate_final", invalid_final):
                run.step(before, after, model.control(), dt=0.001)
            test.assertTrue(run.last_stats.rolled_back)
            test.assertFalse(run.last_stats.variational["zero_accepted_publication"])
            assert_snapshot(test, before, expected)
            assert_snapshot(test, after, expected_output)


def test_publication_checkpoint_identity(test, device):
    """Reject cross-policy checkpoints before touching output."""
    model = scene(device)
    runs = {
        policy: candidate(model, publish_policy=policy) for policy in ("strict", "last_accepted", "current_iterate")
    }
    for policy, run in runs.items():
        state = run.create_state(dt=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            run.save_checkpoint(state, path)
            for other_policy, other in runs.items():
                if other_policy == policy:
                    continue
                output = other.create_state(dt=0.001)
                expected = snapshot(output)
                with test.assertRaises(ValueError):
                    other.load_checkpoint(output, path)
                assert_snapshot(test, output, expected)
                with test.assertRaises(ValueError):
                    other._states.validate(state)


class TestCoupledProduction(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_policy_binding,
        test_mask_contact,
        test_last_accepted_publication,
        test_stopped_hard_gates,
        test_current_iterate_publication,
        test_publication_checkpoint_identity,
    ):
        add_function_test(TestCoupledProduction, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
