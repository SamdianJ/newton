# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify real monolithic stepping and its public state/contact transaction."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import solver_monolithic as monolithic
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices

_STATE_ARRAYS = ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "particle_q", "particle_qd", "particle_f")
_LOGGER = "newton._src.solvers.monolithic.solver_monolithic"


def _free_scene(device, *, gravity=(0.0, 0.0, 0.0), request_force=True, **solver_options):
    fixture = build_tiny_cpu_fixture(device=device)
    fixture.model.gravity.assign(np.asarray([gravity], dtype=np.float32))
    fixture.state.joint_qd.zero_()
    fixture.state.body_qd.zero_()
    fixture.state.particle_qd.zero_()
    if request_force:
        fixture.model.request_contact_attributes("force")
    pipeline = MonolithicCollisionPipeline(fixture.model)
    solver = SolverMonolithic(fixture.model, collision_pipeline=pipeline, contact_stiffness=1.0e5, **solver_options)
    return fixture, solver


def _snapshot(state):
    return {name: getattr(state, name).numpy().copy() for name in _STATE_ARRAYS}


def _assert_state_equal(test, state, expected):
    for name, values in expected.items():
        with test.subTest(array=name):
            np.testing.assert_array_equal(getattr(state, name).numpy(), values)


def test_stationary_current_convergence(test, device):
    """Converge a stationary unforced real scene without starting PCG."""
    fixture, solver = _free_scene(device)
    initial = _snapshot(fixture.state)
    test.assertIsNone(solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001))
    stats = solver.last_stats
    test.assertEqual(stats.status, SolverMonolithic.Status.SUCCESS)
    test.assertTrue(stats.converged)
    test.assertFalse(stats.rolled_back)
    test.assertEqual(stats.linear_iterations, 0)
    test.assertEqual(stats.line_search_iterations, 0)
    test.assertGreaterEqual(stats.matrix_assembly_count, 1)
    test.assertTrue(np.isfinite(stats.merit_final))
    _assert_state_equal(test, fixture.state, initial)
    _assert_state_equal(test, fixture.state_next, initial)
    test.assertEqual(stats.published_contact_generation, int(solver.contacts.contact_generation.numpy()[0]))


def test_free_fall(test, device):
    """Advance a free tet by backward Euler gravity while keeping the anchored articulation stationary."""
    gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float32)
    fixture, solver = _free_scene(device, gravity=gravity)
    initial = _snapshot(fixture.state)
    dt = 0.001
    solver.step(fixture.state, fixture.state_next, fixture.control, None, dt)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
    test.assertGreater(solver.last_stats.linear_iterations, 0)
    np.testing.assert_allclose(
        fixture.state_next.particle_q.numpy(), initial["particle_q"] + dt * dt * gravity, atol=1e-7
    )
    np.testing.assert_allclose(fixture.state_next.particle_qd.numpy(), np.tile(dt * gravity, (4, 1)), atol=1e-4)
    np.testing.assert_allclose(fixture.state_next.joint_q.numpy(), initial["joint_q"], atol=1e-7)
    _assert_state_equal(test, fixture.state, initial)


def test_alpha_scaled_decrease_boundary(test, device):
    """Accept an actual smooth trial that separates alpha-scaled and fixed decrease gates."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -1000.0), newton_max_iterations=1)
    solver._config = replace(solver._config, merit_noise=0.0)
    recover = solver._linear.recover_delta
    evaluate = solver._evaluate_trial
    measured = []

    def overshoot(y, delta, **kwargs):
        status = recover(y, delta, **kwargs)
        # An intentionally overlong direction forces backtracking in a smooth
        # translating tet. Residual evaluation and candidate states remain real.
        delta.assign(delta.numpy() * np.float32(3.99985))
        y.assign(y.numpy() * np.float32(3.99985))
        return status

    def measure(candidate, dt):
        result = evaluate(candidate, dt)
        measured.append(result.merit)
        return result

    with (
        patch.object(solver._linear, "recover_delta", side_effect=overshoot),
        patch.object(solver, "_evaluate_trial", side_effect=measure),
        test.assertLogs(_LOGGER, level="WARNING"),
    ):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    stats = solver.last_stats
    test.assertEqual(stats.status, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
    test.assertFalse(stats.rolled_back)
    test.assertEqual(stats.accepted_alpha, 0.5)
    test.assertEqual(len(measured), 2)
    test.assertGreater(measured[0], stats.merit_initial)
    test.assertGreater(measured[1], (1.0 - 1.0e-4) * stats.merit_initial)
    test.assertLessEqual(measured[1], (1.0 - 1.0e-4 * 0.5) * stats.merit_initial)


def test_fixed_tet_rigid_step(test, device):
    """Advance a rigid subsystem while every tet node remains a static Dirichlet node."""
    fixture = build_tiny_cpu_fixture(device=device)
    model = fixture.model
    model.gravity.zero_()
    model.particle_mass.zero_()
    model.particle_inv_mass.zero_()
    fixture.state.joint_qd.zero_()
    fixture.state.body_qd.zero_()
    fixture.state.particle_qd.zero_()
    fixture.control.joint_f.assign(np.asarray([0.0, 0.5], dtype=np.float32))
    initial = _snapshot(fixture.state)
    solver = SolverMonolithic(model, collision_pipeline=MonolithicCollisionPipeline(model), contact_stiffness=1.0e5)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    test.assertTrue(solver.last_stats.converged)
    test.assertEqual(solver.last_stats.particle_block_count, 0)
    test.assertEqual(solver.last_stats.merit_x_final, 0.0)
    np.testing.assert_allclose(fixture.state_next.joint_q.numpy(), initial["joint_q"] + [0.0, 0.0001], atol=1e-7)
    np.testing.assert_array_equal(fixture.state_next.particle_q.numpy(), initial["particle_q"])
    np.testing.assert_array_equal(fixture.state_next.particle_qd.numpy(), initial["particle_qd"])


def test_optional_contact_force_attribute(test, device):
    """Finish a normal step without an optional force array and reject explicit force publication."""
    fixture, solver = _free_scene(device, request_force=False)
    test.assertIsNone(solver.contacts.force)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
    with test.assertRaises(ValueError):
        solver.update_contacts(solver.contacts, fixture.state_next)


def test_frozen_forces_and_in_place(test, device):
    """Preserve external input forces and produce the same physical step in separate and aliased states."""
    outputs = []
    dt = 0.001
    acceleration = np.asarray([1.0, -2.0, 3.0], dtype=np.float32)
    for in_place in (False, True):
        fixture, solver = _free_scene(device)
        fixture.state.particle_f.assign(fixture.model.particle_mass.numpy()[:, None] * acceleration)
        fixture.state.body_f.assign(
            np.asarray([[0.0, 0.0, 0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        fixture.control.joint_f.assign(np.asarray([0.0, 0.5], dtype=np.float32))
        initial = _snapshot(fixture.state)
        control = fixture.control.joint_f.numpy().copy()
        target = fixture.state if in_place else fixture.state_next
        solver.step(fixture.state, target, fixture.control, None, dt)
        test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
        np.testing.assert_allclose(target.particle_q.numpy(), initial["particle_q"] + dt * dt * acceleration, atol=1e-7)
        np.testing.assert_allclose(target.particle_qd.numpy(), np.tile(dt * acceleration, (4, 1)), atol=1e-4)
        np.testing.assert_allclose(target.joint_q.numpy(), initial["joint_q"] + [0.0, dt * dt], atol=1e-7)
        for name in ("body_f", "particle_f"):
            np.testing.assert_array_equal(getattr(target, name).numpy(), initial[name])
        np.testing.assert_array_equal(fixture.control.joint_f.numpy(), control)
        if not in_place:
            _assert_state_equal(test, fixture.state, initial)
        outputs.append(_snapshot(target))
    for name in _STATE_ARRAYS:
        np.testing.assert_allclose(outputs[0][name], outputs[1][name], rtol=1e-6, atol=1e-7)


def test_final_contact_generation_and_ownership(test, device):
    """Keep final publication idempotent and reject foreign or stale contact buffers."""
    fixture, solver = _free_scene(device)
    foreign = solver.collision_pipeline.contacts()
    initial = _snapshot(fixture.state)
    generation = int(solver.contacts.contact_generation.numpy()[0])
    for provided in (foreign, solver.contacts):
        with test.assertRaises(ValueError):
            solver.step(fixture.state, fixture.state_next, fixture.control, provided, 0.001)
        _assert_state_equal(test, fixture.state, initial)
        test.assertEqual(int(solver.contacts.contact_generation.numpy()[0]), generation)
    solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    generation = int(solver.contacts.contact_generation.numpy()[0])
    test.assertEqual(generation, 1)
    stats = solver.last_stats
    solver.update_contacts(solver.contacts, fixture.state_next)
    solver.update_contacts(solver.contacts)
    test.assertIs(solver.last_stats, stats)
    test.assertEqual(int(solver.contacts.contact_generation.numpy()[0]), generation)
    with test.assertRaises(ValueError):
        solver.update_contacts(foreign, fixture.state_next)
    with test.assertRaises(ValueError):
        solver.update_contacts(solver.contacts, fixture.state)
    solver.step(fixture.state_next, fixture.state_next, fixture.control, None, 0.001)
    test.assertEqual(int(solver.contacts.contact_generation.numpy()[0]), generation + 1)
    test.assertEqual(solver.last_stats.step_generation, 2)
    test.assertEqual(stats.step_generation, 1)
    solver.contacts.clear()
    with test.assertRaises((ValueError, RuntimeError)):
        solver.update_contacts(solver.contacts, fixture.state_next)


def _contact_scene(device, *, height=0.0005):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.Z,
        armature=0.0,
        damping=0.0,
        friction=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([joint])
    builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=builder.ShapeConfig(density=0.0))
    builder.add_soft_mesh(
        pos=(0.0, 0.0, height),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.02)],
        indices=[0, 1, 2, 3],
        density=1000.0,
        k_mu=1000.0,
        k_lambda=1000.0,
        k_damp=0.0,
        particle_radius=0.001,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    state, target = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = MonolithicCollisionPipeline(model)
    solver = SolverMonolithic(model, collision_pipeline=pipeline, contact_stiffness=1.0e5)
    return model, state, target, solver


def test_bilateral_contact_final_forces(test, device):
    """Move both contact actors and retain the final physical force cache across idempotent publication."""
    model, state, target, solver = _contact_scene(device)
    solver.step(state, target, model.control(), None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
    test.assertGreater(solver.last_stats.active_sample_count, 0)
    test.assertLess(float(target.joint_q.numpy()[0]), float(state.joint_q.numpy()[0]))
    momentum = model.body_mass.numpy()[0] * target.body_qd.numpy()[0, 2]
    momentum += np.sum(model.particle_mass.numpy() * target.particle_qd.numpy()[:, 2])
    test.assertAlmostEqual(float(momentum), 0.0, delta=1e-6)
    count = int(solver.contacts.soft_contact_count.numpy()[0])
    force = solver.contacts.force.numpy().copy()
    test.assertGreater(np.linalg.norm(force[:count, :3]), 0.0)
    test.assertLess(float(np.sum(force[:count, 2])), 0.0)
    generation = int(solver.contacts.contact_generation.numpy()[0])
    solver.update_contacts(solver.contacts, target)
    solver.update_contacts(solver.contacts)
    np.testing.assert_array_equal(solver.contacts.force.numpy(), force)
    test.assertEqual(int(solver.contacts.contact_generation.numpy()[0]), generation)


def test_safe_iteration_limit(test, device):
    """Commit the last safe accepted deformation and warn exactly once on nonlinear exhaustion."""
    fixture, solver = _free_scene(device, newton_max_iterations=1)
    positions = fixture.state.particle_q.numpy().copy()
    positions[1, 0] *= 1.15
    fixture.state.particle_q.assign(positions)
    initial = _snapshot(fixture.state)
    with test.assertLogs(_LOGGER, level="WARNING") as logs:
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    stats = solver.last_stats
    test.assertEqual(stats.status, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
    test.assertFalse(stats.converged)
    test.assertFalse(stats.rolled_back)
    test.assertEqual(len(logs.records), 1)
    test.assertGreater(stats.accepted_generation, 0)
    test.assertGreater(stats.min_det_f, 0.0)
    test.assertTrue(np.isfinite(stats.merit_final))
    reference_scaled = solver._linear.scale.numpy().astype(np.float64) * solver._residual_ref.numpy()
    test.assertLess(stats.merit_final, np.linalg.norm(reference_scaled) / np.sqrt(reference_scaled.size))
    test.assertGreater(np.linalg.norm(fixture.state_next.particle_q.numpy() - initial["particle_q"]), 0.0)
    _assert_state_equal(test, fixture.state, initial)
    for field in (
        "status=",
        "failure_reason=",
        "converged=",
        "rolled_back=",
        "step_generation=",
        "nonlinear_iterations=",
        "linear_iterations=",
        "rho=",
    ):
        test.assertIn(field, logs.output[0])


def test_hard_failure_rollback(test, device):
    """Restore the full input state and publish one warning after an evaluated current failure."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -9.81))
    initial = _snapshot(fixture.state)
    evaluate = solver._evaluate_current

    def fail_current(*args, **kwargs):
        evaluate(*args, **kwargs)
        raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "injected current failure")

    with patch.object(solver, "_evaluate_current", side_effect=fail_current):
        with test.assertLogs(_LOGGER, level="WARNING") as logs:
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONFINITE)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertFalse(solver.last_stats.converged)
    test.assertEqual(len(logs.records), 1)
    _assert_state_equal(test, fixture.state_next, initial)
    test.assertEqual(solver.last_stats.published_contact_generation, int(solver.contacts.contact_generation.numpy()[0]))
    solver.update_contacts(solver.contacts, fixture.state_next)


def test_final_failure_in_place_republication(test, device):
    """Undo a provisional in-place commit and republish contacts for the original state after final failure."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -9.81))
    initial = _snapshot(fixture.state)
    publish = solver._publish_final
    published = []

    def fail_first_publication(state, status):
        publish(state, status)
        published.append(_snapshot(state))
        if len(published) == 1:
            raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "injected final publication failure")

    with patch.object(solver, "_publish_final", side_effect=fail_first_publication):
        with test.assertLogs(_LOGGER, level="WARNING") as logs:
            solver.step(fixture.state, fixture.state, fixture.control, None, 0.001)
    test.assertEqual(len(published), 2)
    test.assertGreater(np.linalg.norm(published[0]["particle_q"] - initial["particle_q"]), 0.0)
    for name in _STATE_ARRAYS:
        np.testing.assert_array_equal(published[1][name], initial[name])
    _assert_state_equal(test, fixture.state, initial)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONFINITE)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertEqual(len(logs.records), 1)
    test.assertEqual(int(solver.contacts.contact_generation.numpy()[0]), 2)
    test.assertEqual(solver.last_stats.published_contact_generation, 2)
    solver.update_contacts(solver.contacts, fixture.state)


def _owner_snapshot(solver):
    linear = solver._linear
    arrays = {
        "accepted_residual": solver._transaction.accepted.residual,
        "aq": linear.aq_actor_dense,
        "diagonal": linear.diagonal,
        "scale": linear.scale,
        "dynamic_diagonal": linear.dynamic_diagonal,
    }
    for label, matrix in (("global", linear.k_global_scalar_bsr), ("internal", linear.ax_internal_bsr3)):
        for name in ("offsets", "columns", "values"):
            arrays[f"{label}_{name}"] = getattr(matrix, name)
    for name in ("gq", "gx_columns", "gx_values", "weights", "count"):
        arrays[f"factor_{name}"] = getattr(linear._factors, name)
    return {name: array.numpy().copy() for name, array in arrays.items()}


def _assert_owner_equal(test, solver, expected):
    actual = _owner_snapshot(solver)
    for name, values in expected.items():
        with test.subTest(owner=name):
            np.testing.assert_array_equal(actual[name], values)


def test_trial_recollision_preserves_owners(test, device):
    """Create and remove actual trial contacts while keeping accepted residuals, owners and factors unchanged."""
    for height, displacement, starts_active in ((0.03, -0.0295, False), (0.0005, 0.05, True)):
        model, state, target, solver = _contact_scene(device, height=height)
        evaluate_current = solver._evaluate_current
        observed = []

        def probe_trial(
            candidate,
            dt,
            *,
            evaluate_current=evaluate_current,
            solver=solver,
            starts_active=starts_active,
            displacement=displacement,
            observed=observed,
        ):
            evaluate_current(candidate, dt)
            owner = _owner_snapshot(solver)
            accepted = _snapshot(candidate.state)
            generation = solver._generation
            sequence = solver._assembly_sequence
            contacts_generation = int(solver._trial_contacts.contact_generation.numpy()[0])
            active_before = solver._contact._diagnostics(0)["active_sample_count"]
            test.assertEqual(active_before > 0, starts_active)
            delta = np.zeros(solver._layout.scalar_dof_count, dtype=np.float32)
            delta[solver._layout.q_dof_count + 2 :: 3] = displacement
            solver._delta.assign(delta)
            trial = solver._transaction.trial
            trial.form_trial(candidate, solver._delta, 1.0, dt)
            result = solver._evaluate_trial(trial, dt)
            active_after = solver._contact._diagnostics(1)["active_sample_count"]
            test.assertEqual(active_after > 0, not starts_active)
            test.assertTrue(np.isfinite(result.merit))
            test.assertGreater(result.min_det_f, solver._config.det_f_guard)
            test.assertEqual(int(solver._trial_contacts.contact_generation.numpy()[0]), contacts_generation + 1)
            _assert_state_equal(test, candidate.state, accepted)
            _assert_owner_equal(test, solver, owner)
            test.assertEqual(solver._generation, generation)
            test.assertEqual(solver._assembly_sequence, sequence)
            observed.append((active_before, active_after))
            raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "discard contact-transition probe trial")

        with patch.object(solver, "_evaluate_current", side_effect=probe_trial):
            with test.assertLogs(_LOGGER, level="WARNING"):
                solver.step(state, target, model.control(), None, 0.001)
        test.assertEqual(len(observed), 1)
        test.assertTrue(solver.last_stats.rolled_back)
        _assert_state_equal(test, target, _snapshot(state))


def test_small_step_stagnation(test, device):
    """Roll back when an accepted scaled step is small but the real residual remains unconverged."""
    fixture, solver = _free_scene(device)
    positions = fixture.state.particle_q.numpy().copy()
    positions[1, 0] *= 1.15
    fixture.state.particle_q.assign(positions)
    initial = _snapshot(fixture.state)
    solver._config = replace(
        solver._config,
        step_tolerance_global=1.0e8,
        step_tolerance_q=1.0e8,
        step_tolerance_x=1.0e8,
    )
    with test.assertLogs(_LOGGER, level="WARNING") as logs:
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.01)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONLINEAR_STAGNATION)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertFalse(solver.last_stats.converged)
    test.assertGreater(solver.last_stats.accepted_alpha, 0.0)
    test.assertGreater(solver.last_stats.linear_iterations, 0)
    test.assertEqual(solver.last_stats.nonlinear_iterations, 1)
    test.assertEqual(len(logs.records), 1)
    _assert_state_equal(test, fixture.state_next, initial)


def test_regularization_budget_keeps_current_assembly(test, device):
    """Exhaust rejected real trials with a finite retry budget and frozen current assembly and scaling."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -9.81), line_search_max_iterations=2)
    initial = _snapshot(fixture.state)
    evaluate_trial = solver._evaluate_trial
    snapshots = []

    def reject_trial(candidate, dt):
        before = _owner_snapshot(solver)
        generation, sequence = solver._generation, solver._assembly_sequence
        result = evaluate_trial(candidate, dt)
        test.assertTrue(np.isfinite(result.merit))
        _assert_owner_equal(test, solver, before)
        if snapshots:
            test.assertEqual((generation, sequence), snapshots[0][:2])
            _assert_owner_equal(test, solver, snapshots[0][2])
        snapshots.append((generation, sequence, before))
        raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "reject evaluated trial")

    with patch.object(solver, "_evaluate_trial", side_effect=reject_trial):
        with test.assertLogs(_LOGGER, level="WARNING") as logs:
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.REGULARIZATION_EXHAUSTED)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertEqual(solver.last_stats.matrix_assembly_count, 1)
    test.assertEqual(solver.last_stats.regularization_retries, len(solver._config.regularization_values) - 1)
    test.assertEqual(len(snapshots), len(solver._config.regularization_values) * solver.line_search_max_iterations)
    test.assertEqual(len(logs.records), 1)
    _assert_state_equal(test, fixture.state_next, initial)


def test_permanent_final_failure_invalidates_cache(test, device):
    """Restore input and invalidate force publication when both final publication attempts fail."""
    fixture, solver = _free_scene(device, gravity=(0.0, 0.0, -9.81))
    initial = _snapshot(fixture.state)
    publish = solver._publish_final

    def fail_publication(state, status):
        publish(state, status)
        raise monolithic._StepFailure(SolverMonolithic.Status.NONFINITE, "permanent final publication failure")

    with patch.object(solver, "_publish_final", side_effect=fail_publication) as final:
        with test.assertLogs(_LOGGER, level="WARNING") as logs:
            with test.assertRaisesRegex(RuntimeError, "rollback state"):
                solver.step(fixture.state, fixture.state, fixture.control, None, 0.001)
    test.assertEqual(final.call_count, 2)
    test.assertEqual(len(logs.records), 1)
    _assert_state_equal(test, fixture.state, initial)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertEqual(solver.last_stats.published_contact_generation, -1)
    test.assertIsNone(solver._contact.final_force_generation)
    with test.assertRaises(ValueError):
        solver.update_contacts(solver.contacts, fixture.state)
    solver.step(fixture.state, fixture.state, fixture.control, None, 0.001)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.SUCCESS)
    test.assertEqual(solver.last_stats.step_generation, 2)


def test_nonfinite_force_preserves_complete_rollback(test, device):
    """Preserve every original state array on a real nonfinite-force rollback, including non-FK body state."""
    for in_place in (False, True):
        with test.subTest(in_place=in_place):
            fixture, solver = _free_scene(device)
            body_q = fixture.state.body_q.numpy().copy()
            body_q[:, 0] += 0.123
            fixture.state.body_q.assign(body_q)
            fixture.state.body_qd.fill_(wp.spatial_vector(0.456))
            particle_force = fixture.state.particle_f.numpy().copy()
            particle_force[0, 0] = np.nan
            fixture.state.particle_f.assign(particle_force)
            original = _snapshot(fixture.state)
            target = fixture.state if in_place else fixture.state_next
            with test.assertLogs(_LOGGER, level="WARNING") as logs:
                solver.step(fixture.state, target, fixture.control, None, 0.001)
            test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NONFINITE)
            test.assertTrue(solver.last_stats.rolled_back)
            test.assertFalse(solver.last_stats.converged)
            test.assertEqual(len(logs.records), 1)
            for name, expected in original.items():
                with test.subTest(array=name):
                    np.testing.assert_allclose(
                        getattr(target, name).numpy(),
                        expected,
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
            if not in_place:
                _assert_state_equal(test, fixture.state, original)


class TestMonolithicStep(unittest.TestCase):
    """Exercise real CPU and CUDA physical steps through the experimental API."""


for _test in (
    test_stationary_current_convergence,
    test_free_fall,
    test_alpha_scaled_decrease_boundary,
    test_fixed_tet_rigid_step,
    test_optional_contact_force_attribute,
    test_frozen_forces_and_in_place,
    test_final_contact_generation_and_ownership,
    test_bilateral_contact_final_forces,
    test_safe_iteration_limit,
    test_hard_failure_rollback,
    test_final_failure_in_place_republication,
    test_trial_recollision_preserves_owners,
    test_small_step_stagnation,
    test_regularization_budget_keeps_current_assembly,
    test_permanent_final_failure_invalidates_cache,
    test_nonfinite_force_preserves_complete_rollback,
):
    add_function_test(TestMonolithicStep, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
