# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify bilateral P1Q3 factors and returned-state force publication."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    eval_articulation_passive_candidate,
)
from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.contact import (
    MonolithicContactWorkspace,
    _MonolithicReturnedStateRole,
    assemble_current_contacts,
    evaluate_final_contacts,
    evaluate_trial_contacts,
)
from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearWorkspace,
)
from newton.tests.test_solver_monolithic_collision import _scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _contact_scene(device, *, fixed=False, force=True, detection_gap=0.1):
    model, state = _scene(device, margin=0.006)
    model.joint_limit_ke.zero_()
    model.joint_limit_kd.zero_()
    model.joint_axis.assign([[0.0, 1.0, 0.0]])
    model.body_com.assign([[0.003, 0.004, 0.005]])
    if force:
        model.request_contact_attributes("force")
    mapping = np.arange(model.particle_count, dtype=np.int32)
    if fixed:
        mapping -= 1
        inv_mass = model.particle_inv_mass.numpy()
        inv_mass[0] = 0.0
        model.particle_inv_mass.assign(inv_mass)
    pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=detection_gap)
    linear = MonolithicLinearWorkspace(
        MonolithicLinearLayout(1, 3 if fixed else 4),
        MonolithicLinearCapacities(pipeline.soft_contact_max * 100, 0, pipeline.soft_contact_max, 20, 20),
        wp.array(mapping, dtype=int, device=device),
        device=device,
    )
    articulation = MonolithicArticulationWorkspace(model)
    eval_articulation_passive_candidate(model, state, articulation)
    workspace = MonolithicContactWorkspace(model, pipeline, linear, contact_stiffness=2e5)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, pipeline, linear, articulation, workspace, contacts


def _current(scene, sequence=0):
    model, state, pipeline, linear, articulation, workspace, contacts = scene
    generation = MonolithicLinearGeneration(0, sequence, int(contacts.contact_generation.numpy()[0]), sequence)
    assembly = linear.begin_assembly(generation)
    residual = wp.zeros(linear.layout.scalar_dof_count, dtype=float, device=model.device)
    assemble_current_contacts(
        model, state, contacts, pipeline, articulation, workspace, assembly, residual, generation=generation
    )
    return generation, assembly, residual


def _energy(scene):
    model, state, _, _, _, workspace, contacts = scene
    count = int(contacts.soft_contact_count.numpy()[0])
    x, rest = state.particle_q.numpy(), model.particle_q.numpy()
    indices, bary, shapes = (
        contacts.soft_contact_indices.numpy(),
        contacts.soft_contact_barycentric.numpy(),
        contacts.soft_contact_shape.numpy(),
    )
    normals, local = contacts.soft_contact_normal.numpy(), contacts.soft_contact_body_pos.numpy()
    poses = state.body_q.numpy()
    total = 0.0
    for record in range(count):
        ids, b = indices[record], bary[record]
        xr = np.asarray(wp.transform_point(wp.transform(*poses[0]), wp.vec3(local[record])))
        gap = float(
            normals[record] @ (b @ x[ids] - xr) - workspace.pipeline.r_soft - model.shape_margin.numpy()[shapes[record]]
        )
        area = np.linalg.norm(np.cross(rest[ids[1]] - rest[ids[0]], rest[ids[2]] - rest[ids[0]])) * 0.5
        total += 0.5 * workspace.contact_stiffness * area / 3.0 * min(gap, 0.0) ** 2
    return total


def test_contact_factors_and_fd(test, device):
    """Match energy gradients, all four blocks and physical-force projections."""
    scene = _contact_scene(device)
    model, state, pipeline, linear, articulation, workspace, contacts = scene
    generation, assembly, residual = _current(scene)
    test.assertEqual(int(assembly.contact_factors.status.numpy()[0]), 0)
    expected_residual = residual.numpy().astype(np.float64)
    count = int(assembly.contact_factors.count.numpy()[0])
    test.assertGreater(count, 0)
    factors = assembly.contact_factors
    G = np.zeros((count, linear.layout.scalar_dof_count))
    G[:, :1] = factors.gq.numpy()[:count]
    for factor in range(count):
        for column, value in zip(factors.gx_columns.numpy()[factor], factors.gx_values.numpy()[factor], strict=True):
            if column >= 0:
                G[factor, 1 + column] += value
    weights = factors.weights.numpy()[:count]
    matrix = G.T @ (weights[:, None] * G)
    np.testing.assert_allclose(
        expected_residual, G.T @ (weights * workspace.factor_gap.numpy()[:count]), rtol=2e-6, atol=1e-8
    )
    test.assertEqual(linear.finalize_assembly(generation=generation).status, 0)
    np.testing.assert_allclose(linear.densify_for_test(generation=generation).raw_matrix, matrix, rtol=2e-6, atol=1e-6)
    test.assertGreater(np.linalg.norm(matrix[:1, 1:]), 0.0)
    np.testing.assert_allclose(matrix[:1, 1:], matrix[1:, :1].T, atol=1e-8)
    test.assertGreater(np.linalg.norm(matrix[1:, 0]), 0.0)
    q0, x0 = state.joint_q.numpy().copy(), state.particle_q.numpy().copy()
    h = 1e-5
    for dof in range(13):
        energies = []
        for sign in (-1, 1):
            q, x = q0.copy(), x0.copy()
            if dof == 0:
                q[0] += sign * h
            else:
                x.flat[dof - 1] += sign * h
            state.joint_q.assign(q)
            state.particle_q.assign(x)
            eval_articulation_passive_candidate(model, state, articulation)
            energies.append(_energy(scene))
        derivative = (energies[1] - energies[0]) / (2 * h)
        test.assertAlmostEqual(derivative, expected_residual[dof], delta=3e-5 + 3e-3 * abs(expected_residual[dof]))
    state.joint_q.assign(q0)
    state.particle_q.assign(x0)
    eval_articulation_passive_candidate(model, state, articulation)
    out, status = wp.zeros_like(residual), wp.zeros(1, dtype=int, device=device)
    role = _MonolithicReturnedStateRole.STATE_OUT_CONVERGED
    evaluate_final_contacts(
        model,
        state,
        contacts,
        pipeline,
        articulation,
        workspace,
        out,
        status,
        step_generation=1,
        returned_state_role=role,
    )
    test.assertEqual(int(status.numpy()[0]), 0)
    np.testing.assert_allclose(out.numpy(), expected_residual, atol=1e-8)
    workspace._publish_final_forces(state, contacts, step_generation=1, returned_state_role=role)
    np.testing.assert_allclose(contacts.force.numpy()[:, :3], workspace.final_force_linear.numpy(), atol=0)
    np.testing.assert_allclose(contacts.force.numpy()[:, 3:], workspace.final_force_moment.numpy(), atol=0)
    test.assertLess(float(contacts.force.numpy()[:, 2].sum()), 0.0)
    diagnostics = workspace._diagnostics(2)
    test.assertEqual(diagnostics["active_sample_count"], count)
    for key in ("force_imbalance", "moment_imbalance", "generalized_projection_error", "contact_sign_error"):
        test.assertLess(diagnostics[key], 2e-6)


def test_modes_and_cache(test, device):
    """Keep rejected trials separate and reject stale final publication."""
    scene = _contact_scene(device)
    model, state, pipeline, linear, articulation, workspace, contacts = scene
    generation, assembly, residual = _current(scene)
    linear.finalize_assembly(generation=generation)
    matrix = linear.densify_for_test(generation=generation).raw_matrix.copy()
    factors = assembly.contact_factors.gq.numpy().copy()
    status = wp.zeros(1, dtype=int, device=device)
    role = _MonolithicReturnedStateRole.STATE_OUT_SOFT_STOP
    evaluate_final_contacts(
        model,
        state,
        contacts,
        pipeline,
        articulation,
        workspace,
        wp.zeros_like(residual),
        status,
        step_generation=3,
        returned_state_role=role,
    )
    forces = workspace.final_force_linear.numpy().copy()
    final_generation = workspace.final_force_generation
    trial = model.state()
    trial.particle_q.assign(state.particle_q.numpy() + np.array([0, 0, 1]))
    trial_contacts = pipeline.contacts()
    eval_articulation_passive_candidate(model, trial, articulation)
    pipeline.collide(trial, trial_contacts)
    out = wp.zeros_like(residual)
    evaluate_trial_contacts(
        model,
        trial,
        trial_contacts,
        pipeline,
        articulation,
        workspace,
        out,
        status,
        owner_generation=generation,
        trial_generation=1,
    )
    np.testing.assert_array_equal(out.numpy(), np.zeros(13))
    np.testing.assert_array_equal(workspace.final_force_linear.numpy(), forces)
    np.testing.assert_array_equal(assembly.contact_factors.gq.numpy(), factors)
    np.testing.assert_array_equal(linear.densify_for_test(generation=generation).raw_matrix, matrix)
    test.assertIs(workspace.final_force_generation, final_generation)
    test.assertEqual(workspace._diagnostics(1)["active_sample_count"], 0)
    for _ in range(2):
        workspace._publish_final_forces(state, contacts, step_generation=3, returned_state_role=role)
    np.testing.assert_array_equal(contacts.force.numpy()[:, :3], forces)
    for bad_state, bad_contacts, bad_step, bad_role in [
        (trial, contacts, 3, role),
        (state, trial_contacts, 3, role),
        (state, contacts, 4, role),
        (state, contacts, 3, _MonolithicReturnedStateRole.STATE_IN_ROLLBACK),
    ]:
        with test.assertRaisesRegex(ValueError, "stale"):
            workspace._publish_final_forces(
                bad_state, bad_contacts, step_generation=bad_step, returned_state_role=bad_role
            )
    pipeline.collide(state, contacts)
    with test.assertRaisesRegex(ValueError, "stale"):
        workspace._publish_final_forces(state, contacts, step_generation=3, returned_state_role=role)
    workspace.invalidate_final_force()
    np.testing.assert_array_equal(linear.densify_for_test(generation=generation).raw_matrix, matrix)


def test_fixed_slots_and_detection(test, device):
    """Eliminate fixed node columns and keep detection distance out of force."""
    scene = _contact_scene(device, fixed=True)
    _, _, _, linear, _, _, _ = scene
    generation, assembly, residual = _current(scene)
    factors = assembly.contact_factors
    count = int(factors.count.numpy()[0])
    columns, values = factors.gx_columns.numpy()[:count], factors.gx_values.numpy()[:count]
    test.assertTrue(np.any(columns == -1))
    np.testing.assert_array_equal(values[columns == -1], 0.0)
    test.assertEqual(linear.finalize_assembly(generation=generation).status, 0)
    test.assertTrue(np.isfinite(residual.numpy()).all())
    outputs = []
    for gap in (0.0, 0.1):
        other = _contact_scene(device, detection_gap=gap)
        _, _, other_residual = _current(other)
        outputs.append(other_residual.numpy())
    np.testing.assert_allclose(*outputs, atol=1e-8)


def test_contact_failures(test, device):
    """Reject malformed records, overflow and nonfinite force before publication."""
    for failure in ("overflow", "normal", "indices", "barycentric", "duplicate_tid", "force"):
        scene = _contact_scene(device)
        model, state, pipeline, _, articulation, workspace, contacts = scene
        if failure == "overflow":
            contacts.soft_contact_count.fill_(pipeline.soft_contact_max + 1)
        elif failure == "normal":
            contacts.soft_contact_normal.fill_(wp.vec3(float("nan"), 0, 1))
        elif failure == "indices":
            contacts.soft_contact_indices.fill_(wp.vec3i(-1))
        elif failure == "barycentric":
            contacts.soft_contact_barycentric.fill_(wp.vec3(0.5))
        elif failure == "duplicate_tid":
            contacts.soft_contact_tids.fill_(0)
        else:
            workspace.contact_stiffness = 1e39
        out = wp.zeros(13, dtype=float, device=device)
        status = wp.zeros(1, dtype=int, device=device)
        evaluate_final_contacts(
            model,
            state,
            contacts,
            pipeline,
            articulation,
            workspace,
            out,
            status,
            step_generation=0,
            returned_state_role=_MonolithicReturnedStateRole.STATE_IN_ROLLBACK,
        )
        test.assertNotEqual(int(status.numpy()[0]), 0)
        test.assertIsNone(workspace.final_force_generation)
    scene = _contact_scene(device, force=False)
    model, state, pipeline, _, articulation, workspace, contacts = scene
    evaluate_final_contacts(
        model,
        state,
        contacts,
        pipeline,
        articulation,
        workspace,
        wp.zeros(13, dtype=float, device=device),
        wp.zeros(1, dtype=int, device=device),
        step_generation=0,
        returned_state_role=_MonolithicReturnedStateRole.STATE_IN_ROLLBACK,
    )
    with test.assertRaisesRegex(ValueError, "force"):
        workspace._publish_final_forces(
            state, contacts, step_generation=0, returned_state_role=_MonolithicReturnedStateRole.STATE_IN_ROLLBACK
        )


def test_contact_tangent_and_inactive(test, device):
    """Match particle-direction residual differences and zero separated blocks."""
    scene = _contact_scene(device)
    _, state, pipeline, linear, _, _, contacts = scene
    generation, _, residual = _current(scene)
    linear.finalize_assembly(generation=generation)
    matrix = linear.densify_for_test(generation=generation).raw_matrix
    direction = np.arange(12, dtype=np.float32).reshape(4, 3) / 13.0
    x0 = state.particle_q.numpy().copy()
    samples = []
    for sequence, sign in enumerate((-1, 1), 1):
        state.particle_q.assign(x0 + sign * 1e-5 * direction)
        _, _, result = _current(scene, sequence)
        samples.append(result.numpy().copy())
    np.testing.assert_allclose(
        (samples[1] - samples[0]) / 2e-5, matrix @ np.r_[0.0, direction.ravel()], rtol=1e-3, atol=1e-4
    )
    state.particle_q.assign(x0 + np.array([0.0, 0.0, 1.0]))
    pipeline.collide(state, contacts)
    generation, assembly, residual = _current(scene, 3)
    test.assertEqual(int(assembly.contact_factors.count.numpy()[0]), 0)
    test.assertEqual(linear.finalize_assembly(generation=generation).status, 0)
    np.testing.assert_array_equal(residual.numpy(), 0.0)
    np.testing.assert_array_equal(linear.densify_for_test(generation=generation).raw_matrix, 0.0)


def test_workspace_and_publication_contract(test, device):
    """Reject stale assembly descriptors and changed final records without writes."""
    scene = _contact_scene(device)
    model, state, pipeline, linear, articulation, workspace, contacts = scene
    generation, assembly, residual = _current(scene)
    linear.finalize_assembly(generation=generation)
    with test.assertRaisesRegex(ValueError, "stale"):
        assemble_current_contacts(
            model, state, contacts, pipeline, articulation, workspace, assembly, residual, generation=generation
        )
    with test.assertRaisesRegex(ValueError, "stale"):
        evaluate_trial_contacts(
            model,
            state,
            contacts,
            pipeline,
            articulation,
            workspace,
            residual,
            wp.zeros(1, dtype=int, device=device),
            owner_generation=MonolithicLinearGeneration(10, 0, 0, 0),
            trial_generation=0,
        )
    for stiffness in (0.0, -1.0, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "stiffness"):
            MonolithicContactWorkspace(model, pipeline, linear, contact_stiffness=stiffness)
    role = _MonolithicReturnedStateRole.STATE_IN_ROLLBACK
    status = wp.zeros(1, dtype=int, device=device)
    for mutation in ("count", "face", "cache"):
        pipeline.collide(state, contacts)
        evaluate_final_contacts(
            model,
            state,
            contacts,
            pipeline,
            articulation,
            workspace,
            wp.zeros_like(residual),
            status,
            step_generation=4,
            returned_state_role=role,
        )
        test.assertIs(workspace.final_force_generation.returned_state, state)
        contacts.force.fill_(wp.spatial_vector(1.0))
        if mutation == "count":
            contacts.soft_contact_count.fill_(0)
        elif mutation == "face":
            contacts.soft_contact_indices.fill_(wp.vec3i(-1))
        else:
            workspace.final_force_linear.fill_(wp.vec3(float("nan")))
        with test.assertRaises(ValueError):
            workspace._publish_final_forces(state, contacts, step_generation=4, returned_state_role=role)
        np.testing.assert_array_equal(contacts.force.numpy(), 1.0)


class TestMonolithicContact(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_contact_factors_and_fd,
        test_modes_and_cache,
        test_fixed_slots_and_detection,
        test_contact_failures,
        test_contact_tangent_and_inactive,
        test_workspace_and_publication_contract,
    ):
        add_function_test(TestMonolithicContact, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
