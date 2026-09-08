# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""G3 contact mathematics and G4 history transactions."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import solver_monolithic as impl
from newton._src.solvers.monolithic.articulation import eval_articulation_passive_candidate
from newton._src.solvers.monolithic.contact import (
    MonolithicContactWorkspace,
    _evaluate,
    _normal_response,
    _tangent_response,
)
from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
    MonolithicPcgWarmStart,
)
from newton.examples.softbody.monolithic_contact_friction import FrictionCase
from newton.tests.test_solver_monolithic_contact import _contact_scene
from newton.tests.test_solver_monolithic_linear import _pcg_config
from newton.tests.test_solver_monolithic_step import _assert_state_equal, _free_scene, _snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _math_probe(
    u: wp.array[float],
    xi: wp.array[wp.vec3],
    eps: float,
    k: float,
    cap: float,
    normal: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    energy: wp.array[float],
    hessian: wp.array[wp.mat33],
):
    i = wp.tid()
    normal[i] = _normal_response(u[i], eps, k)
    f, e, h = _tangent_response(xi[i], wp.vec3(0.0, 0.0, 1.0), k, cap)
    force[i] = f
    energy[i] = e
    hessian[i] = h


def _probe(device, u, xi, *, eps=0.2, k=3.0, cap=0.6):
    count = len(u)
    arrays = [wp.zeros(count, dtype=dtype, device=device) for dtype in (wp.vec3, wp.vec3, float, wp.mat33)]
    wp.launch(
        _math_probe,
        count,
        [wp.array(u, dtype=float, device=device), wp.array(xi, dtype=wp.vec3, device=device), eps, k, cap, *arrays],
        device=device,
    )
    return [a.numpy().astype(float) for a in arrays]


def _normal_oracle(u, eps, k):
    if u <= -eps:
        return np.zeros(3)
    if u >= eps:
        return np.array([0.5 * k * u * u, k * u, k])
    t = (u + eps) / (2 * eps)
    p, dp, ddp = eps * (2 * t**3 - t**4), 3 * t * t - 2 * t**3, 3 * t * (1 - t) / eps
    return np.array([0.5 * k * p * p, k * p * dp, k * (dp * dp + p * ddp)])


def test_polyrelu(test, device):
    """Check all pieces, boundary continuity and a five-step FD plateau."""
    u = np.array([-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4])
    actual = _probe(device, u, np.zeros((len(u), 3)))[0]
    np.testing.assert_allclose(actual, [_normal_oracle(v, 0.2, 3) for v in u], rtol=2e-6, atol=1e-7)
    for center in (-0.2, -0.1, 0.0, 0.1, 0.2):
        errors = []
        for h in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
            result = _probe(device, [center - h, center, center + h], np.zeros((3, 3)))[0]
            fd = (result[2, :2] - result[0, :2]) / (2 * h)
            expected = result[1, 1:]
            errors.append(np.linalg.norm(fd - expected) / max(np.linalg.norm(expected), 1e-5))
        test.assertLess(min(errors), 5e-3)
    test.assertGreater(actual[2, 1], 0)  # Positive physical gap inside the shell.
    legacy = _probe(device, [-0.1, 0, 0.1], np.zeros((3, 3)), eps=0)[0]
    np.testing.assert_allclose(legacy, [[0, 0, 0], [0, 0, 0], [0.015, 0.3, 3]], rtol=1e-6)


def test_radial_return(test, device):
    """Verify zero, stick, slip, reversal, cap, potential and local PSD."""
    xi = np.array([[0, 0, 0], [0.01, 0.02, 0], [0.4, 0.3, 0], [-0.4, -0.3, 0], [0.2, 0, 0]])
    _, forces, energies, matrices = _probe(device, np.zeros(5), xi)
    for i, x in enumerate(xi):
        r = np.linalg.norm(x)
        k = 3.0
        cap = 0.6
        expected = -k * x if k * r < cap else -cap * x / r
        np.testing.assert_allclose(forces[i], expected, atol=1e-7)
        test.assertLessEqual(np.linalg.norm(forces[i]), cap + 1e-7)
        test.assertGreaterEqual(np.linalg.eigvalsh(matrices[i]).min(), -1e-6)
        np.testing.assert_allclose(matrices[i], matrices[i].T, atol=1e-8)
        test.assertAlmostEqual(
            energies[i], 0.5 * k * r * r if k * r < cap else cap * r - cap * cap / (2 * k), delta=1e-7
        )
    # Equality chooses the sliding generalized derivative.
    test.assertAlmostEqual(matrices[-1, 0, 0], 0, delta=1e-6)
    for x in xi[1:4]:
        errors = []
        for h in (2e-3, 1e-3, 5e-4, 2e-4, 1e-4):
            variations = np.array([x + np.eye(3)[j] * sign * h for j in range(2) for sign in (-1, 1)])
            _, f, e, _ = _probe(device, np.zeros(4), variations)
            _, fc, _, hc = _probe(device, [0], [x])
            grad = np.array([(e[2 * j + 1] - e[2 * j]) / (2 * h) for j in range(2)])
            derivative = np.stack([-(f[2 * j + 1] - f[2 * j]) / (2 * h) for j in range(2)], axis=1)
            errors.append(
                max(np.linalg.norm(grad + fc[0, :2]), np.linalg.norm(derivative - hc[0, :, :2]))
                / max(np.linalg.norm(hc), 1e-6)
            )
        test.assertLess(min(errors), 5e-3)
    _, f, e, h = _probe(device, np.zeros(5), xi, cap=0)
    for a in (f, e, h):
        np.testing.assert_array_equal(a, 0)
    # BE work + change in stored elastic energy + return dissipation <= 0.
    old = np.array([0.1, 0, 0])
    increment = np.array([0.3, 0.1, 0])
    trial = old + increment
    _, f, _, _ = _probe(device, [0], [trial])
    pending = -f[0] / 3
    work = f[0] @ increment
    delta = 0.5 * 3 * (pending @ pending - old @ old)
    plastic = 0.6 * np.linalg.norm(trial - pending)
    test.assertLessEqual(work + delta + plastic, 1e-7)


def _friction_scene(device, *, fixed=False):

    model, state, pipeline, old, articulation, _, contacts = _contact_scene(device, fixed=fixed)
    capacity = pipeline.soft_contact_max * 3
    linear = MonolithicLinearWorkspace(
        old.layout,
        MonolithicLinearCapacities(capacity * 100, 0, capacity, 20, 20),
        old.particle_to_dynamic,
        device=device,
    )
    workspace = MonolithicContactWorkspace(
        model,
        pipeline,
        linear,
        contact_stiffness=2e5,
        normal_smoothing_width=0.001,
        friction_coefficient=0.5,
        tangential_stiffness=1e5,
    )
    original = model.state()
    original.assign(state)
    workspace.begin_step(original, 0.01)
    return model, state, pipeline, linear, articulation, workspace, contacts


def _evaluate_scene(scene, *, mode=0, sequence=0):

    model, state, pipeline, linear, articulation, workspace, contacts = scene
    eval_articulation_passive_candidate(model, state, articulation)
    pipeline.collide(state, contacts)
    generation = MonolithicLinearGeneration(
        0, sequence, int(contacts.contact_generation.numpy()[0]), sequence, history_epoch=workspace.history_epoch
    )
    assembly = linear.begin_assembly(generation) if mode == 0 else None
    residual = wp.zeros(linear.layout.scalar_dof_count, dtype=float, device=model.device)
    status = assembly.contact_factors.status if assembly else wp.zeros(1, dtype=int, device=model.device)
    _evaluate(mode, model, state, contacts, pipeline, articulation, workspace, residual, status, assembly)
    if int(status.numpy()[0]):
        raise AssertionError(f"Contact error: {status.numpy()}")
    return generation, residual.numpy(), assembly


def test_friction_blocks(test, device):
    """Check four blocks, force moment balance, fixed elimination and sliding rank."""
    for fixed in (False, True):
        scene = _friction_scene(device, fixed=fixed)
        _, state, _, linear, _, workspace, _ = scene
        for sequence, distance in enumerate((1e-5, 0.04)):
            x = workspace._start_x.numpy().copy()
            x[:, 0] += distance
            state.particle_q.assign(x)
            generation, residual, assembly = _evaluate_scene(scene, sequence=sequence)
            count = int(assembly.contact_factors.count.numpy()[0])
            f = assembly.contact_factors
            g = np.zeros((count, linear.layout.scalar_dof_count))
            g[:, :1] = f.gq.numpy()[:count]
            columns, values = f.gx_columns.numpy()[:count], f.gx_values.numpy()[:count]
            for i in range(count):
                for col, value in zip(columns[i], values[i], strict=True):
                    if col >= 0:
                        g[i, 1 + col] += value
                    else:
                        test.assertEqual(value, 0)
            oracle = (g.T * f.weights.numpy()[:count]) @ g
            test.assertEqual(linear.finalize_assembly(generation=generation).status, 0)
            actual = linear.densify_for_test(generation=generation).raw_matrix
            np.testing.assert_allclose(actual, oracle, rtol=1e-5, atol=2e-5)
            test.assertLess(np.linalg.norm(actual - actual.T) / max(np.linalg.norm(actual), 1e-6), 1e-4)
            test.assertGreater(np.linalg.norm(actual[0, 1:]), 0)
            test.assertGreaterEqual(np.linalg.eigvalsh(actual.astype(float)).min(), -1e-6 * np.linalg.norm(actual))
            diagnostics = workspace._diagnostics(0)
            for key in ("force_imbalance", "moment_imbalance", "contact_sign_error"):
                test.assertLess(diagnostics[key], 1e-4)
            test.assertGreater(np.linalg.norm(residual), 0)
            test.assertIn(2, f.kind.numpy()[:count])
            # Each active sample has two tangent directions in stick, one in slip.
            sample_count = int(diagnostics["active_sample_count"])
            test.assertEqual(count, sample_count * (3 if distance < 0.001 else 2))


def test_true_residual_fd(test, device):
    """Compare changing-load raw residual FD to an independent plane-contact oracle."""
    scene = _friction_scene(device)
    model, state, _, _, _, workspace, contacts = scene
    rest = model.particle_q.numpy().astype(float)
    start = workspace._start_x.numpy().astype(float)

    # Independent NumPy evaluation at q=0; x derivatives include the changing cap N(x).
    def oracle(x):
        result = np.zeros(13)
        ids = contacts.soft_contact_indices.numpy()
        bary = contacts.soft_contact_barycentric.numpy()
        count = int(contacts.soft_contact_count.numpy()[0])
        for indices, b in zip(ids[:count], bary[:count], strict=True):
            p = b @ x[indices]
            area = 0.5 * np.linalg.norm(
                np.cross(rest[indices[1]] - rest[indices[0]], rest[indices[2]] - rest[indices[0]])
            )
            theta = float(state.joint_q.numpy()[0])
            n = np.array([np.sin(theta), 0, np.cos(theta)])
            normal = _normal_oracle(0.007 - n @ p, 0.001, 2e5 * area / 3)[1]
            if normal <= 0:
                continue
            xi = b @ (x[indices] - start[indices]) - workspace._dt * np.cross(
                [0, float(state.joint_qd.numpy()[0]), 0], p
            )
            xi -= n * (n @ xi)
            radius = np.linalg.norm(xi)
            kt = 1e5 * area / 3
            cap = 0.5 * normal
            ft = -kt * xi if kt * radius < cap else -cap * xi / radius
            rigid = -ft - normal * n
            # Revolute axis Y through the world origin, same point for both sides.
            result[0] -= np.cross(p, rigid)[1]
            for index, weight in zip(indices, b, strict=True):
                result[1 + 3 * index : 4 + 3 * index] += weight * rigid
        return result

    for shift in (1e-5, 0.02):
        state.joint_q.zero_()
        state.joint_qd.zero_()
        base = start.copy()
        base[:, 0] += shift
        state.particle_q.assign(base)
        _evaluate_scene(scene, mode=1)
        for axis in (-1, 0, 2):
            errors = []
            for h in (2e-5, 1e-5, 5e-6, 2e-6, 1e-6):
                evaluated = []
                references = []
                for sign in (-1, 1):
                    x = base.copy()
                    state.joint_q.zero_()
                    state.joint_qd.zero_()
                    if axis == -1:
                        state.joint_q.assign([sign * h])
                        state.joint_qd.assign([sign * h / workspace._dt])
                    else:
                        x[0, axis] += sign * h
                    state.particle_q.assign(x)
                    evaluated.append(_evaluate_scene(scene, mode=1)[1])
                    references.append(oracle(state.particle_q.numpy().astype(float)))
                fd = (evaluated[1] - evaluated[0]) / (2 * h)
                ref = (references[1] - references[0]) / (2 * h)
                errors.append(np.linalg.norm(fd - ref) / max(np.linalg.norm(ref), 1e-6))
            test.assertLess(min(errors), 5e-3)


def test_history_trials(test, device):
    """Keep rejected trials read-only and clear lost or rotated history before reentry."""

    scene = _friction_scene(device)
    model, state, pipeline, linear, articulation, workspace, contacts = scene
    start = state.particle_q.numpy().copy()
    x = start.copy()
    x[:, 0] += 0.0001
    state.particle_q.assign(x)
    _evaluate_scene(scene, mode=2)
    workspace._prepared = True
    workspace.commit_history()
    old = workspace.committed.xi_local.numpy().copy()
    valid = workspace.committed.valid.numpy().copy()
    test.assertGreater(valid.sum(), 0)
    test.assertGreater(np.linalg.norm(old), 0)
    original = model.state()
    original.assign(state)
    workspace.begin_step(original, 0.01)
    a = _evaluate_scene(scene, mode=1)[1]
    state.particle_q.assign(start + np.array([0.1, 0, 1], dtype=np.float32))
    _evaluate_scene(scene, mode=1)
    state.particle_q.assign(x)
    b = _evaluate_scene(scene, mode=1)[1]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(workspace.committed.xi_local.numpy(), old)
    # Warmed launches use preallocated scratch, including pending storage.
    out = wp.zeros(linear.layout.scalar_dof_count, dtype=float, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    with (
        patch.object(wp, "zeros", side_effect=AssertionError("allocation")),
        patch.object(wp, "empty", side_effect=AssertionError("allocation")),
    ):
        for _ in range(10):
            out.zero_()
            status.zero_()
            _evaluate(1, model, state, contacts, pipeline, articulation, workspace, out, status)
    # A changed previous normal is rejected by the frozen 45-degree transport rule.
    workspace.committed.normal_local.assign(np.tile([1, 0, 0], (pipeline.soft_contact_max, 1)))
    _evaluate_scene(scene, mode=2)
    test.assertGreater(np.count_nonzero(workspace._samples[2].numpy()[:, 5] == 2), 0)
    workspace.discard_history()
    workspace.begin_step(original, 0.01)
    state.particle_q.assign(start + np.array([0, 0, 1], dtype=np.float32))
    _evaluate_scene(scene, mode=2)
    workspace._prepared = True
    workspace.commit_history()
    np.testing.assert_array_equal(workspace.committed.valid.numpy(), 0)
    np.testing.assert_array_equal(workspace.committed.xi_local.numpy(), 0)
    test.assertEqual(workspace.history_epoch, 2)


def test_solver_history_transaction(test, device):
    """Commit zero-contact steps once and roll back injected publication failures."""

    fixture, solver = _free_scene(
        device, normal_smoothing_width=0.001, friction_coefficient=0.5, tangential_stiffness=1e5
    )
    solver.step(fixture.state, fixture.state, fixture.control, None, 0.001)
    test.assertTrue(solver.last_stats.converged)
    test.assertEqual(solver._history_epoch, 1)
    before = _snapshot(fixture.state)
    epoch = solver._history_epoch
    generation = solver._contact.final_force_generation
    solver.update_contacts(solver.contacts, fixture.state)
    solver.update_contacts(solver.contacts, fixture.state)
    test.assertEqual(solver._history_epoch, epoch)
    test.assertIs(solver._contact.final_force_generation, generation)
    original_publish = solver._publish_final
    calls = []

    def fail_once(state, status):
        original_publish(state, status)
        calls.append(status)
        if len(calls) == 1:
            raise impl._StepFailure(solver.Status.NONFINITE, "injected final failure")

    with patch.object(solver, "_publish_final", side_effect=fail_once):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertEqual(solver._history_epoch, epoch)
    _assert_state_equal(test, fixture.state_next, before)
    for where in ("_iterate", "_publish_final"):
        with patch.object(solver, where, side_effect=RuntimeError("injected")), test.assertRaises(RuntimeError):
            solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
        test.assertEqual(solver._history_epoch, epoch)
        _assert_state_equal(test, fixture.state_next, before)
        if where == "_publish_final":
            test.assertIsNone(solver._contact.final_force_generation)
        else:
            test.assertIs(solver._contact.final_force_generation.returned_state, fixture.state_next)
    # Soft stop is an accepted physical step, even if the candidate is unchanged.
    with patch.object(solver, "_iterate", return_value=solver.Status.NONLINEAR_MAX_ITERATIONS):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    test.assertFalse(solver.last_stats.rolled_back)
    test.assertEqual(solver._history_epoch, epoch + 1)


def test_rotating_history(test, device):
    """Measure BE rotation error under dt refinement and the 45-degree reset boundary."""
    errors = []
    for dt in (0.02, 0.01, 0.005):
        scene = _friction_scene(device)
        _, state, _, _, _, w, _ = scene
        rest = w._start_x.numpy().copy()
        w._dt = dt
        w.committed.valid.fill_(1)
        old = np.tile([0.00001, 0, 0], (w.committed.valid.shape[0], 1))
        w.committed.xi_local.assign(old)
        w.committed.normal_local.assign(np.tile([0, 0, 1], (len(old), 1)))
        theta = dt
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        state.particle_q.assign(rest @ rotation.T)
        state.joint_q.assign([theta])
        state.joint_qd.assign([1.0])
        _evaluate_scene(scene, mode=2)
        active = w.pending.valid.numpy() != 0
        error = np.linalg.norm(w.pending.xi_local.numpy()[active] - old[active])
        errors.append(error)
        np.testing.assert_array_equal(w.committed.xi_local.numpy(), old.astype(np.float32))
    test.assertLess(errors[1], 0.4 * errors[0])
    test.assertLess(errors[2], 0.4 * errors[1])
    for angle, reset in ((44.9, False), (45.1, True)):
        scene = _friction_scene(device)
        w = scene[5]
        size = w.committed.valid.shape[0]
        w.committed.valid.fill_(1)
        radians = np.deg2rad(angle)
        w.committed.normal_local.assign(np.tile([np.sin(radians), 0, np.cos(radians)], (size, 1)))
        w.committed.xi_local.assign(np.tile([0.00001, 0, 0], (size, 1)))
        _evaluate_scene(scene, mode=2)
        samples = w._samples[2].numpy()
        active = samples[:, 0] > 0
        np.testing.assert_array_equal(samples[active, 5], 2 if reset else 0)
        test.assertTrue(np.all(samples[active, 9] <= 1e-10))


def test_record_order_and_config(test, device):
    """Preserve stable history keys under record permutation and reject stale configuration."""

    scene = _friction_scene(device)
    model, state, pipeline, linear, articulation, w, contacts = scene
    w.committed.valid.fill_(1)
    w.committed.normal_local.assign(np.tile([0, 0, 1], (pipeline.soft_contact_max, 1)))
    w.committed.xi_local.assign(np.arange(pipeline.soft_contact_max * 3).reshape(-1, 3) * np.array([1e-6, 1e-6, 0]))
    _, reference, _ = _evaluate_scene(scene, mode=1)
    samples = w._samples[1].numpy().copy()
    count = int(contacts.soft_contact_count.numpy()[0])
    perm = np.arange(count)[::-1]
    for name in (
        "soft_contact_indices",
        "soft_contact_barycentric",
        "soft_contact_shape",
        "soft_contact_body_pos",
        "soft_contact_normal",
    ):
        a = getattr(contacts, name)
        data = a.numpy()
        data[:count] = data[:count][perm]
        a.assign(data)
    tids = contacts.soft_contact_tids.numpy()
    valid = tids >= 0
    tids[valid] = count - 1 - tids[valid]
    contacts.soft_contact_tids.assign(tids)
    output = wp.zeros(linear.layout.scalar_dof_count, dtype=float, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    _evaluate(1, model, state, contacts, pipeline, articulation, w, output, status)
    np.testing.assert_allclose(output.numpy(), reference, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(w._samples[1].numpy(), samples, rtol=1e-6, atol=1e-10)
    w.transport_cos = 0
    with test.assertRaisesRegex(ValueError, "stale"):
        _evaluate(1, model, state, contacts, pipeline, articulation, w, output, status)


def test_active_history_rollback(test, device):
    """Audit actual contacting steps, final-copy faults and rollback force without reintegration."""

    c = FrictionCase(device)
    solver = c.solver
    w = solver._contact
    for _ in range(150):
        c.step()
    # Compare the same production operator, with normal and tangent factors, to dense.

    linear, generation = solver._linear, solver._generation
    oracle = linear.densify_for_test(generation=generation)
    test.assertGreater(np.linalg.eigvalsh(oracle.scaled_matrix).min(), 0)
    test.assertEqual(linear.factor_actor_preconditioner(generation=generation, pivot_tolerance=0), 0)
    rhs = wp.array(np.linspace(0.1, 1, len(oracle.raw_matrix)), dtype=float, device=device)
    solution = wp.zeros_like(rhs)
    result = linear.solve_pcg(
        rhs,
        solution,
        generation=generation,
        warm_start=MonolithicPcgWarmStart.ZERO,
        config=_pcg_config(curvature_relative_tolerance=1e-12),
    )
    test.assertEqual(result.status, 0)
    expected = np.linalg.solve(oracle.scaled_matrix, rhs.numpy())
    test.assertLess(np.linalg.norm(solution.numpy() - expected) / np.linalg.norm(expected), 1e-4)
    committed_before = w.committed.xi_local.numpy().copy()
    solve = linear.solve_pcg
    calls_retry = []

    def retry_once(*args, **kwargs):
        np.testing.assert_array_equal(w.committed.xi_local.numpy(), committed_before)
        result = solve(*args, **kwargs)
        calls_retry.append(result.status)
        if len(calls_retry) == 1:
            return replace(result, status=MonolithicLinearStatus.NON_POSITIVE_CURVATURE)
        return result

    with patch.object(linear, "solve_pcg", side_effect=retry_once):
        solver.step(c.state, c.next, c.control, None, c.dt)
    test.assertFalse(solver.last_stats.rolled_back)
    test.assertGreaterEqual(solver.last_stats.regularization_retries, 1)
    c.state, c.next = c.next, c.state
    epoch = w.history_epoch
    old = w.committed.xi_local.numpy().copy()
    valid = w.committed.valid.numpy().copy()
    test.assertGreater(valid.sum(), 0)
    test.assertGreater(np.linalg.norm(old), 0)
    before = c.state.particle_q.numpy().copy()
    saved_publish = solver._publish_final
    calls = []

    def failed_final(state, status):
        saved_publish(state, status)
        calls.append(status)
        if len(calls) == 1:
            raise impl._StepFailure(solver.Status.NONFINITE, "after force publication")

    with patch.object(solver, "_publish_final", side_effect=failed_final):
        solver.step(c.state, c.state, c.control, None, c.dt)
    test.assertTrue(solver.last_stats.rolled_back)
    np.testing.assert_array_equal(c.state.particle_q.numpy(), before)
    np.testing.assert_array_equal(w.committed.xi_local.numpy(), old)
    np.testing.assert_array_equal(w.committed.valid.numpy(), valid)
    test.assertEqual(w.history_epoch, epoch)
    # All plate normals and rotations are constant in this prismatic fixture.
    samples = w._samples[2].numpy()
    keys = np.flatnonzero(samples[:, 0] > 0)
    ids = solver.collision_pipeline._face_pairs.numpy()
    faces = c.model.tri_indices.numpy()
    rest = c.rest
    for key in keys:
        face = faces[ids[key // 3, 0]]
        area = 0.5 * np.linalg.norm(np.cross(rest[face[1]] - rest[face[0]], rest[face[2]] - rest[face[0]]))
        expected = min(
            w.friction_coefficient * samples[key, 0], w.tangential_stiffness * area / 3 * np.linalg.norm(old[key])
        )
        test.assertAlmostEqual(samples[key, 1], expected, delta=1e-6 + 1e-5 * expected)
    # An output-copy exception also leaves committed history intact.
    candidate_type = type(solver._transaction.accepted)
    commit = candidate_type.commit

    def fail_copy(candidate, state):
        commit(candidate, state)
        raise RuntimeError("after State copy")

    with patch.object(candidate_type, "commit", new=fail_copy), test.assertRaisesRegex(RuntimeError, "State copy"):
        solver.step(c.state, c.next, c.control, None, c.dt)
    test.assertEqual(w.history_epoch, epoch)
    np.testing.assert_array_equal(w.committed.xi_local.numpy(), old)
    np.testing.assert_array_equal(c.next.particle_q.numpy(), before)
    # History commit follows joint-force publication too.
    publish_joint = solver._joint_terms.publish

    def fail_joint(generation):
        publish_joint(generation)
        raise RuntimeError("joint force publication")

    with (
        patch.object(solver._joint_terms, "publish", side_effect=fail_joint),
        test.assertRaisesRegex(RuntimeError, "joint force"),
    ):
        solver.step(c.state, c.next, c.control, None, c.dt)
    test.assertEqual(w.history_epoch, epoch)
    np.testing.assert_array_equal(w.committed.xi_local.numpy(), old)
    test.assertIsNone(w.final_force_generation)
    test.assertEqual(solver._joint_terms.final_generation, -1)


def test_contact_configuration(test, device):
    """Reject invalid friction, uncovered smoothing and stale arrays before stepping."""

    for options in (
        {"friction_coefficient": 0.5},
        {"friction_coefficient": 0.5, "tangential_stiffness": 0},
        {"normal_smoothing_width": 0.1},
        {"normal_smoothing_width": 1e-99},
        {"friction_coefficient": float("inf")},
        {"tangential_stiffness": -1},
    ):
        with test.assertRaises(ValueError):
            _free_scene(device, **options)
    fixture, solver = _free_scene(
        device, normal_smoothing_width=0.001, friction_coefficient=0.5, tangential_stiffness=1e5
    )
    initial = fixture.state_next.particle_q.numpy().copy()
    fixture.model.body_com = wp.clone(fixture.model.body_com)
    with test.assertRaisesRegex(ValueError, "stale"):
        solver.step(fixture.state, fixture.state_next, fixture.control, None, 0.001)
    np.testing.assert_array_equal(fixture.state_next.particle_q.numpy(), initial)
    test.assertEqual(solver._history_epoch, 0)


def test_common_translation_and_capacity(test, device):
    """Cancel common point velocity and fail bounded factor overflow without history mutation."""
    scene = _friction_scene(device)
    model, state, pipeline, linear, articulation, w, contacts = scene
    _evaluate_scene(scene, mode=1)
    normal = w._samples[1].numpy()[:, 0].copy()
    shift = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    state.particle_q.assign(w._start_x.numpy() + shift)
    state.body_q.assign([[*shift, 0.0, 0.0, 0.0, 1.0]])
    state.body_qd.assign([[*(shift / w._dt), 0.0, 0.0, 0.0]])
    pipeline.collide(state, contacts)
    out = wp.zeros(linear.layout.scalar_dof_count, dtype=float, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    _evaluate(1, model, state, contacts, pipeline, articulation, w, out, status)
    test.assertEqual(int(status.numpy()[0]), 0)
    samples = w._samples[1].numpy()
    test.assertLess(float(samples[:, 1].sum()), 1e-5)
    np.testing.assert_allclose(samples[:, 0], normal, rtol=1e-5, atol=1e-7)
    # The kernel must report overflow rather than truncate a sample's tangent factors.
    state.particle_q.assign(w._start_x)
    state.body_qd.zero_()
    generation = MonolithicLinearGeneration(0, 0, int(contacts.contact_generation.numpy()[0]), 0)
    assembly = linear.begin_assembly(generation)
    assembly.contact_factors.weights = wp.zeros(1, dtype=float, device=device)
    # Refresh real kinematics/contact records before the bounded assembly probe.
    eval_articulation_passive_candidate(model, state, articulation)
    pipeline.collide(state, contacts)
    _evaluate(0, model, state, contacts, pipeline, articulation, w, out, assembly.contact_factors.status, assembly)
    test.assertEqual(
        int(assembly.contact_factors.status.numpy()[0]), int(MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW)
    )
    np.testing.assert_array_equal(w.committed.valid.numpy(), 0)


class TestMonolithicFriction(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_polyrelu,
        test_radial_return,
        test_friction_blocks,
        test_true_residual_fd,
        test_history_trials,
        test_solver_history_transaction,
        test_rotating_history,
        test_record_order_and_config,
        test_active_history_rollback,
        test_contact_configuration,
        test_common_translation_and_capacity,
    ):
        add_function_test(TestMonolithicFriction, function.__name__, function, devices=[device])

if __name__ == "__main__":
    unittest.main()
