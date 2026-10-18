# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate native rigid dynamics, mixed layouts and complete continuation."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import ShapeFlags
from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.iteration import measure
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicPcgWarmStart,
)
from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_actor_layout import mixed_model
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, snapshot, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contract_oracle import rigid_terms, rotation
from scripts.monolithic_reference.articulation_oracle import skew, transform

CASE = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures/rigid_dynamics_short_v1.json"


def rigid_scene(device, *, before=False, contact=True, count=2):
    case = json.loads(CASE.read_text())
    specs = [case["rigids"][i % 2] for i in range(count)]
    return scene(device, contact=contact, rigid_specs=specs, rigids_before=before)


def rigid_solver(model, *, permuted=False, **options):
    table = ActorTable.from_model(model)
    if permuted:
        table = ActorTable.from_model(model, actor_order=(3, 2, 0, 1))
    with patch.object(ActorTable, "from_model", return_value=table):
        return solver(model, **options)


def apply_wrench(run, state, control, step=0, *, body=True, joint=True):
    specs = json.loads(CASE.read_text())["rigids"]
    forces, efforts = state.body_f.numpy(), control.joint_f.numpy()
    sign = 1 if step < 5 else -1
    for actor in run._rigid.actors:
        label = run.model.body_label[actor.body_indices[0]]
        spec = next(s for s in specs if s["label"] == label)
        forces[actor.body_indices[0]] = sign * np.asarray(spec["body_wrench"]) if body else 0
        efforts[list(actor.dof_indices)] = sign * np.asarray(spec["joint_wrench"]) if joint else 0
    state.body_f.assign(forces)
    control.joint_f.assign(efforts)


def independent_step(c, r, v, omega, sym, m, inertia, force, torque, h):
    """Solve the independent discrete rotational equation with a finite-difference Jacobian."""
    cp = c + h * v
    rp = r + h * (skew(omega) + sym) @ r
    current = r.copy()
    for _ in range(15):
        residual = rigid_terms(cp, current, cp, rp, m, inertia, h, force)[1][3:] - torque
        if np.linalg.norm(residual) < 1e-13:
            break
        eps = 1e-6
        jac = np.column_stack(
            [
                (
                    rigid_terms(cp, rotation(eps * axis) @ current, cp, rp, m, inertia, h, force)[1][3:]
                    - rigid_terms(cp, rotation(-eps * axis) @ current, cp, rp, m, inertia, h, force)[1][3:]
                )
                / (2 * eps)
                for axis in np.eye(3)
            ]
        )
        current = rotation(np.linalg.solve(jac, -residual)) @ current
    else:
        raise AssertionError("Independent rotational reference did not converge")
    result = cp + h * h * force / m
    a = (current - r) @ current.T / h
    rate = np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]]) / 2
    return result, current, (result - c) / h, rate, (a + a.T) / 2


def test_rigid_mapping_and_terms(test, device):
    """Match COM/twist conversion and conservative E/r/PSD B against independent NumPy."""
    for before in (False, True):
        model = rigid_scene(device, before=before)
        run = rigid_solver(model)
        state = run.create_state(dt=0.001)
        history = state._monolithic_coupled
        for i, actor in enumerate(run._rigid.actors):
            j, b = actor.joint_indices[0], actor.body_indices[0]
            parent, child = transform(model.joint_X_p.numpy()[j]), transform(model.joint_X_c.numpy()[j])
            pose = parent @ transform(state.joint_q.numpy()[list(actor.coord_indices)]) @ np.linalg.inv(child)
            np.testing.assert_allclose(
                history.rigid_com.numpy()[i], pose[:3, 3] + pose[:3, :3] @ model.body_com.numpy()[b], atol=1e-10
            )
            qd = state.joint_qd.numpy()[list(actor.dof_indices)]
            np.testing.assert_allclose(history.rigid_v_com.numpy()[i], parent[:3, :3] @ qd[:3], atol=1e-10)
            np.testing.assert_allclose(history.rigid_omega.numpy()[i], parent[:3, :3] @ qd[3:], atol=1e-10)
        control = model.control()
        apply_wrench(run, state, control)
        run._rigid.prepare(state, control, 0.001)
        run._rigid.torque.zero_()
        run._rigid.evaluate(state, 0.001, True)
        for i, actor in enumerate(run._rigid.actors):
            c, r = history.rigid_com.numpy()[i], run._rigid.rotations.numpy()[i]
            cp, rp = run._rigid.predicted_t.numpy()[i], run._rigid.predicted_r.numpy()[i]
            m = float(run._rigid.mass.numpy()[i])
            inertia = model.body_inertia.numpy()[actor.body_indices[0]].astype(float)
            force = run._rigid.force.numpy()[i]
            energy, residual, tangent = rigid_terms(c, r, cp, rp, m, inertia, 0.001, force, initial_com=c)
            np.testing.assert_allclose(run._rigid.energy.numpy()[i], energy, atol=1e-14)
            np.testing.assert_allclose(run._rigid.residual.numpy()[i], residual, atol=1e-13)
            np.testing.assert_allclose(run._rigid.tangent.numpy()[i], tangent, rtol=1e-6, atol=1e-6)
            test.assertGreaterEqual(np.linalg.eigvalsh(tangent).min(), 0)
            for axis in np.eye(6):
                eps = 1e-7

                energies = [
                    rigid_terms(
                        c + sign * eps * axis[:3],
                        rotation(sign * eps * axis[3:]) @ r,
                        cp,
                        rp,
                        m,
                        inertia,
                        0.001,
                        force,
                        initial_com=c,
                    )[0]
                    for sign in (-1, 1)
                ]
                test.assertAlmostEqual((energies[1] - energies[0]) / (2 * eps), residual @ axis, delta=2e-9)


def test_mixed_step_and_continuation(test, device):
    """Advance mixed actors with active A-T and recover complete schema 4 authority."""
    for permuted in (False, True):
        model = rigid_scene(device, before=True)
        run = rigid_solver(model, permuted=permuted)
        state = run.create_state(dt=0.001)
        control = model.control()
        initial = snapshot(state)
        apply_wrench(run, state, control)
        run.step(state, state, control, dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertEqual(len(run.last_stats.variational["rigid_blocks"]), 4)
        matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
        a, t = run._layout.indices("articulation"), run._layout.indices("tet")
        test.assertGreater(np.linalg.norm(matrix[np.ix_(a, t)]), 0)
        history = state._monolithic_coupled
        for i, actor in enumerate(run._rigid.actors):
            old_q = initial["history.rigid_rotation"][i]
            old_r = transform(np.r_[np.zeros(3), old_q])[:3, :3]
            sym_values = initial["history.rigid_vsym"][i]
            sym = np.array(
                [
                    [sym_values[0], sym_values[3], sym_values[4]],
                    [sym_values[3], sym_values[1], sym_values[5]],
                    [sym_values[4], sym_values[5], sym_values[2]],
                ]
            )
            expected = independent_step(
                initial["history.rigid_com"][i],
                old_r,
                initial["history.rigid_v_com"][i],
                initial["history.rigid_omega"][i],
                sym,
                run._rigid.mass.numpy()[i],
                model.body_inertia.numpy()[actor.body_indices[0]],
                run._rigid.force.numpy()[i],
                run._rigid.torque.numpy()[i],
                0.001,
            )
            np.testing.assert_allclose(history.rigid_com.numpy()[i], expected[0], atol=1e-7)
            np.testing.assert_allclose(
                transform(np.r_[np.zeros(3), history.rigid_rotation.numpy()[i]])[:3, :3], expected[1], atol=1e-7
            )
            np.testing.assert_allclose(history.rigid_v_com.numpy()[i], expected[2], atol=1e-6)
            np.testing.assert_allclose(history.rigid_omega.numpy()[i], expected[3], atol=1e-6)
        test.assertEqual(history.contact_body_force.shape, (model.body_count,))
        np.testing.assert_array_equal(history.contact_body_force.numpy()[run._rigid.body_indices], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mixed.npz"
            run.save_checkpoint(state, path)
            restored = run.create_state(dt=0.001)
            run.load_checkpoint(restored, path)
            assert_snapshot(test, restored, snapshot(state))
            other = rigid_solver(model, permuted=permuted)
            out = other.create_state(dt=0.001)
            other.load_checkpoint(out, path)
            run.step(state, state, control, dt=0.001)
            other.step(out, out, control, dt=0.001)
            test.assertTrue(run.last_stats.converged)
            test.assertTrue(other.last_stats.converged)
            assert_snapshot(test, out, snapshot(state))


def test_mixed_dense_linear(test, device):
    """Retain synthetic cross blocks and solve all rigid preconditioner blocks against dense NumPy."""
    for permuted in (False, True):
        run = rigid_solver(rigid_scene(device), permuted=permuted)
        layout, n = run._layout, run._layout.scalar_dof_count
        for mode in ("diagnostic", "production"):
            linear = MonolithicLinearWorkspace(
                layout,
                MonolithicLinearCapacities(n * n, 0, 0, 100, 102),
                run._tet.mapping,
                device=device,
                pcg_mode=mode,
                use_cuda_graph=bool(device.is_cuda and mode == "production"),
            )
            rows = np.arange(n, dtype=np.int32)
            empty = np.zeros(0, dtype=np.int32)
            linear._set_fixed_triplet_pattern(
                global_rows=np.repeat(rows, n),
                global_columns=np.tile(rows, n),
                internal_rows=empty,
                internal_columns=empty,
            )
            rng = np.random.default_rng(19)
            a = rng.normal(size=(n, n)) / n
            matrix = np.eye(n) + a @ a.T
            generation = MonolithicLinearGeneration(1, 0, 1, 1)
            assembly = linear.begin_assembly(generation)
            assembly.global_scalar_triplets.values.assign(matrix.astype(np.float32).ravel())
            test.assertEqual(linear.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
            diagonal = wp.ones(n, dtype=float, device=device)
            test.assertEqual(
                linear.build_scaling(diagonal, epsilon_d=1e-6, generation=generation), MonolithicLinearStatus.SUCCESS
            )
            test.assertEqual(linear.set_regularization(0.01, generation=generation), MonolithicLinearStatus.SUCCESS)
            test.assertEqual(
                linear.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
                MonolithicLinearStatus.SUCCESS,
            )
            oracle = linear.densify_for_test(generation=generation)
            block = np.zeros((n, n))
            parts = [layout.indices("articulation"), *linear.rigid_rows.numpy(), *layout.indices("tet").reshape(-1, 3)]
            for part in parts:
                block[np.ix_(part, part)] = oracle.scaled_matrix[np.ix_(part, part)]
            rhs_numpy = np.linspace(-0.2, 0.3, n).astype(np.float32)
            rhs = wp.array(rhs_numpy, dtype=float, device=device)
            out = wp.zeros_like(rhs)
            linear.preconditioner.matvec(rhs, out, out, 1.0, 0.0)
            np.testing.assert_allclose(out.numpy(), np.linalg.solve(block, rhs_numpy), atol=2e-7, rtol=2e-6)
            linear.operator.matvec(rhs, out, out, 1.0, 0.0)
            np.testing.assert_allclose(out.numpy(), oracle.scaled_matrix @ rhs_numpy, atol=2e-7, rtol=2e-6)
            result = linear.solve_pcg(
                rhs,
                out,
                generation=generation,
                warm_start=MonolithicPcgWarmStart.ZERO,
                config=_pcg_config(group_norm_limits=(1e-6,) * len(layout.groups)),
            )
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
            np.testing.assert_allclose(
                out.numpy(), np.linalg.solve(oracle.scaled_matrix, rhs_numpy), atol=2e-6, rtol=2e-5
            )
            for kind in ("_t", "_r"):
                run._metrics, run._variational = {}, {}
                residual = np.zeros(n)
                name, start, _ = next(g for g in layout.groups if g[0].startswith("rigid_") and g[0].endswith(kind))
                residual[start] = 3.2e-7 if kind == "_t" else 1e-7
                measured = measure(run, residual, np.zeros(n), np.ones(n))
                test.assertLess(measured["global"][0], measured["global"][2])
                test.assertGreater(measured[name][0], measured[name][2])
                # An isolated body's drift must trigger restart even when
                # the global and other actor denominators mask it.
                group = next(i for i, g in enumerate(layout.groups) if g[0] == name)
                norms = np.full(2 * linear.group_count, 10.0, dtype=np.float32)
                gap = np.zeros_like(norms)
                norms[group], gap[group], gap[0] = 1e-5, 2e-6, 2e-6
                linear._true_norms.assign(norms)
                linear._gap_norms.assign(gap)
                test.assertEqual(linear._should_restart(_pcg_config()), 1)
                gap[group] = 0
                linear._gap_norms.assign(gap)
                test.assertEqual(linear._should_restart(_pcg_config()), 0)
            test.assertEqual(
                linear.factor_actor_preconditioner(
                    generation=replace(generation, assembly_sequence=2), pivot_tolerance=0
                ),
                MonolithicLinearStatus.STALE_GENERATION,
            )
            old_factor = linear._rigid_cholesky
            linear._rigid_cholesky = wp.clone(old_factor)
            with test.assertRaisesRegex(ValueError, "refactored"):
                linear.preconditioner.matvec(rhs, out, out, 1.0, 0.0)
            test.assertEqual(
                linear.solve_pcg(
                    rhs, out, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
                ).status,
                MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED,
            )
            test.assertEqual(
                linear.factor_actor_preconditioner(generation=generation, pivot_tolerance=0),
                MonolithicLinearStatus.SUCCESS,
            )
            result = linear.solve_pcg(
                rhs, out, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
            )
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)


def test_rigid_failures_and_reset(test, device):
    """Reject unauthorized torque and roll back all actors on posterior, publication, reset and load failures."""
    model = rigid_scene(device)
    run = rigid_solver(model)
    state = run.create_state(dt=0.001)
    control = model.control()
    apply_wrench(run, state, control)
    output = run.create_state(dt=0.001)
    original, previous = snapshot(state), snapshot(output)
    evaluate = run._rigid.evaluate

    def nonfinite_tangent(candidate, dt, tangent):
        evaluate(candidate, dt, tangent)
        if tangent:
            run._rigid.tangent.fill_(float("nan"))

    with patch.object(run._rigid, "evaluate", nonfinite_tangent):
        run.step(state, output, control, dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, state, original)
    assert_snapshot(test, output, previous)
    with patch.object(run._rigid, "domain", return_value=False):
        run.step(state, output, control, dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, output, previous)
    for method in ("_assemble", "_finish_candidate"):
        with patch.object(run, method, side_effect=_StepFailure(run.Status.NONFINITE, "injected rigid failure")):
            run.step(state, output, control, dt=0.001)
        test.assertTrue(run.last_stats.rolled_back)
        assert_snapshot(test, state, original)
        assert_snapshot(test, output, previous)

    def posterior(*args, **kwargs):
        value = np.zeros(run._layout.scalar_dof_count)
        value[run._rigid.rows_numpy[1, 3]] = 1
        return value

    with patch.object(run, "_physical_linear_error", side_effect=posterior):
        run.step(state, output, control, dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, output, previous)
    original_assign = output.assign

    def publication(source):
        original_assign(source)
        if source is run._accepted:
            raise RuntimeError("injected output publication")

    with patch.object(output, "assign", side_effect=publication):
        with test.assertRaisesRegex(RuntimeError, "output publication"):
            run.step(state, output, control, dt=0.001)
    assert_snapshot(test, output, previous)
    rates = state.joint_qd.numpy()
    rates[run._rigid.dof_indices[0, 3]] = 2000
    state.joint_qd.assign(rates)
    edited = snapshot(state)
    with test.assertRaisesRegex(ValueError, "angular velocity"):
        run.reset_state(state, dt=0.001)
    assert_snapshot(test, state, edited)
    state.joint_qd.assign(original["joint_qd"])
    run.step(state, output, control, dt=0.001)
    test.assertTrue(run.last_stats.converged)
    before_reset = snapshot(output)
    original_assign = output.assign
    injected = []

    def partial_reset(source):
        original_assign(source)
        if not injected:
            injected.append(True)
            raise RuntimeError("injected reset publication")

    with patch.object(output, "assign", partial_reset):
        with test.assertRaisesRegex(RuntimeError, "reset publication"):
            run.reset_state(output, dt=0.002)
    assert_snapshot(test, output, before_reset)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bad.npz"
        run.save_checkpoint(output, path)
        with np.load(path) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        fields["history.rigid_rotation"][0] = 0
        np.savez(path, **fields)
        unchanged = snapshot(output)
        with test.assertRaises(ValueError):
            run.load_checkpoint(output, path)
        assert_snapshot(test, output, unchanged)


def test_rigid_count_and_trial(test, device):
    """Advance zero/one/many rigid actors and keep repeated trials independent of previous candidates."""
    for count in (0, 1, 2, 5):
        run = rigid_solver(rigid_scene(device, count=count, contact=False))
        state = run.create_state(dt=0.001)
        saved = snapshot(state)
        if count:
            trial = run._states.copy(state)
            direction = wp.full(run._layout.scalar_dof_count, 0.1, dtype=float, device=device)
            run._rigid.trial(state, trial, direction, 1)
            run._rigid.trial(state, trial, direction, 0.5)
            for i in range(count):
                base = transform(np.r_[np.zeros(3), saved["history.rigid_rotation"][i]])[:3, :3]
                got = transform(np.r_[np.zeros(3), trial._monolithic_coupled.rigid_rotation.numpy()[i]])[:3, :3]
                np.testing.assert_allclose(
                    got, rotation(np.full(3, 0.05, dtype=np.float32).astype(float)) @ base, atol=1e-12
                )
            assert_snapshot(test, state, saved)
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertEqual(state._monolithic_coupled.rigid_com.shape, (count,))


def test_rigid_force_channels(test, device):
    """Add world COM wrench channels once and preserve native history over display edits."""
    for body, joint in ((False, False), (True, False), (False, True), (True, True)):
        model = rigid_scene(device, contact=False)
        model.gravity.zero_()
        run = rigid_solver(model)
        state, control = run.create_state(dt=0.001), model.control()
        apply_wrench(run, state, control, body=body, joint=joint)
        c, v = state._monolithic_coupled.rigid_com.numpy().copy(), state._monolithic_coupled.rigid_v_com.numpy().copy()
        f = state.body_f.numpy()[run._rigid.body_indices] + control.joint_f.numpy()[run._rigid.dof_indices]
        run._rigid.prepare(state, control, 0.001)
        np.testing.assert_allclose(run._rigid.force.numpy(), f[:, :3], atol=1e-15)
        np.testing.assert_allclose(run._rigid.torque.numpy(), f[:, 3:], atol=1e-15)
        state.body_q.fill_(wp.transform((1, 2, 3), wp.quat_identity()))
        state.body_qd.zero_()
        run.step(state, state, control, dt=0.001)
        test.assertTrue(run.last_stats.converged)
        np.testing.assert_allclose(
            state._monolithic_coupled.rigid_com.numpy(),
            c + 0.001 * v + 1e-6 * f[:, :3] / run._rigid.mass.numpy()[:, None],
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            state._monolithic_coupled.rigid_v_com.numpy(),
            v + 0.001 * f[:, :3] / run._rigid.mass.numpy()[:, None],
            atol=1e-7,
            rtol=0,
        )


def test_rigid_eligibility(test, device):
    """Reject unsupported FREE construction and replacement of private rigid maps."""
    for field in (
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_friction",
        "joint_damping",
        "joint_armature",
    ):
        model = rigid_scene(device)
        actor = next(a for a in ActorTable.from_model(model).actors if a.kind == "rigid")
        value = getattr(model, field).numpy()
        value[actor.dof_indices[-1]] = 0.1
        getattr(model, field).assign(value)
        with test.assertRaisesRegex(ValueError, "Unsupported FREE"):
            rigid_solver(model)
    model = mixed_model(device)[0]
    run = rigid_solver(model)
    test.assertGreater(run._contact.actor_samples.count, 0)
    model.shape_flags.zero_()
    test.assertGreater(model.shape_contact_pairs.size, 0)
    with test.assertRaisesRegex(ValueError, "pair conflicts"):
        rigid_solver(model)
    model.shape_contact_pairs = wp.empty(0, dtype=wp.vec2i, device=device)
    model.shape_contact_pair_count = 0
    run = rigid_solver(model)
    state = run.create_state(dt=0.001)
    run.step(state, state, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged)
    original = snapshot(state)
    for field in ("rows", "qs", "vs", "bodies"):
        with patch.object(run._rigid, field, wp.clone(getattr(run._rigid, field))):
            with test.assertRaisesRegex(ValueError, "Rigid mapping"):
                run.step(state, state, model.control(), dt=0.001)
            assert_snapshot(test, state, original)
    flags = model.shape_flags.numpy()
    flags[run._rigid.actors[0].shape_indices[0]] |= int(ShapeFlags.COLLIDE_PARTICLES)
    model.shape_flags.assign(flags)
    # Model collision flags are static: rebuild after editing instead of relying
    # on the retired runtime mutation detector.
    rebuilt = rigid_solver(model)
    test.assertNotEqual(rebuilt._kinematics.digest, run._kinematics.digest)
    with patch.object(newton, "use_coord_layout_targets", False):
        legacy = rigid_scene(device, before=True, contact=False)
    run = rigid_solver(legacy)
    state = run.create_state(dt=0.001)
    run.step(state, state, legacy.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged)


def test_rigid_checkpoint_rejections(test, device):
    """Keep the complete target unchanged on invalid mixed schema and cross-layout loads."""
    model = rigid_scene(device)
    run = rigid_solver(model)
    state, control = run.create_state(dt=0.001), model.control()
    apply_wrench(run, state, control)
    run.step(state, state, control, dt=0.001)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.npz"
        run.save_checkpoint(state, path)
        with np.load(path) as saved:
            fields = {name: saved[name].copy() for name in saved.files}
        original = snapshot(state)
        for name in (
            "history.rigid_rotation",
            "history.rigid_vsym",
            "history.rigid_bound_com",
            "history.layout_digest",
            "history.actor_layout",
            "history.precision",
            "joint_q",
            "joint_qd",
        ):
            bad = {key: value.copy() for key, value in fields.items()}
            bad[name].flat[0] += 1
            np.savez(path, **bad)
            with test.assertRaises(ValueError, msg=name):
                run.load_checkpoint(state, path)
            assert_snapshot(test, state, original)
        for version in (2, 3):
            bad = {key: value.copy() for key, value in fields.items()}
            bad["history.schema_version"][:] = version
            np.savez(path, **bad)
            with test.assertRaisesRegex(ValueError, "canonical A"):
                run.load_checkpoint(state, path)
            assert_snapshot(test, state, original)
        del bad["history.rigid_com"]
        bad["history.schema_version"][:] = 4
        np.savez(path, **bad)
        with test.assertRaises(ValueError):
            run.load_checkpoint(state, path)
        assert_snapshot(test, state, original)
        np.savez(path, **fields)
        destination = run.create_state(dt=0.001)
        initial = snapshot(destination)
        original_assign = destination.assign
        injected = []

        def partial_load(source):
            original_assign(source)
            if not injected:
                injected.append(True)
                raise RuntimeError("injected checkpoint publication")

        with patch.object(destination, "assign", partial_load):
            with test.assertRaisesRegex(RuntimeError, "checkpoint publication"):
                run.load_checkpoint(destination, path)
        assert_snapshot(test, destination, initial)
        other = rigid_solver(model, permuted=True)
        target = other.create_state(dt=0.001)
        before = snapshot(target)
        with test.assertRaisesRegex(ValueError, "layout"):
            other.load_checkpoint(target, path)
        assert_snapshot(test, target, before)


def test_rigid_empty_blocks_and_stops(test, device):
    """Solve zero-length A/T blocks and preserve all three mixed Stopped publication policies."""
    builder = newton.ModelBuilder(gravity=0)
    anchor = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
    builder.add_articulation([builder.add_joint_prismatic(-1, anchor, limit_ke=0.0, limit_kd=0.0)])
    for _ in range(2):
        body = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
        builder.add_articulation([builder.add_joint_free(body)])
    add_contact_free_tet(builder)
    builder.particle_mass[:] = [0] * builder.particle_count
    model = builder.finalize(device=device)
    for mode in ("diagnostic", "production"):
        run = rigid_solver(model, mode=mode, graph=bool(device.is_cuda and mode == "production"), permuted=True)
        state, control = run.create_state(dt=0.001), model.control()
        effort = control.joint_f.numpy()
        effort[run._rigid.dof_indices[:, 0]] = 0.01
        control.joint_f.assign(effort)
        run.step(state, state, control, dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertEqual(run._layout.scalar_dof_count, 13)
        test.assertEqual(len(run._layout.groups), 6)
        layout = MonolithicLinearLayout(
            0,
            0,
            0,
            12,
            actor_ranges=(("articulation", 0, 0), ("rigid", 0, 6), ("rigid", 6, 6), ("tet", 12, 0)),
            rigid_ranges=((1, 0), (2, 6)),
        )
        linear = MonolithicLinearWorkspace(
            layout,
            MonolithicLinearCapacities(12, 0, 0, 100, 102),
            run._tet.mapping,
            device=device,
            pcg_mode=mode,
            use_cuda_graph=bool(device.is_cuda and mode == "production"),
        )
        rows = np.arange(12, dtype=np.int32)
        empty = np.zeros(0, dtype=np.int32)
        linear._set_fixed_triplet_pattern(
            global_rows=rows, global_columns=rows, internal_rows=empty, internal_columns=empty
        )
        generation = MonolithicLinearGeneration(1, 0, 0, 1)
        assembly = linear.begin_assembly(generation)
        assembly.global_scalar_triplets.values.fill_(1.0)
        test.assertEqual(linear.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
        rhs, out = wp.ones(12, dtype=float, device=device), wp.zeros(12, dtype=float, device=device)
        test.assertEqual(
            linear.build_scaling(rhs, epsilon_d=1e-6, generation=generation), MonolithicLinearStatus.SUCCESS
        )
        test.assertEqual(
            linear.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
            MonolithicLinearStatus.SUCCESS,
        )
        result = linear.solve_pcg(
            rhs, out, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
        )
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        np.testing.assert_allclose(out.numpy(), 1.0, atol=1e-6, rtol=0)
    original_iterate = solver_monolithic.iterate
    for policy in ("strict", "last_accepted", "current_iterate"):
        model = rigid_scene(device)
        run = rigid_solver(model, publish_policy=policy)
        state, output = run.create_state(dt=0.001), run.create_state(dt=0.001)
        before, previous = snapshot(state), snapshot(output)

        def stop(current, dt):
            test.assertEqual(original_iterate(current, dt), current.Status.SUCCESS)
            current._trial._monolithic_coupled.rigid_com.fill_(current._rigid.vector(float("nan")))
            raise _StepFailure(current.Status.NONLINEAR_MAX_ITERATIONS, "mixed stopped fixture")

        with patch.object(solver_monolithic, "iterate", stop):
            run.step(state, output, model.control(), dt=0.001)
        test.assertFalse(run.last_stats.converged)
        test.assertEqual(run.last_stats.rolled_back, policy == "strict")
        assert_snapshot(test, state, before)
        if policy == "strict":
            assert_snapshot(test, output, previous)
        else:
            run._states.validate(output)
            test.assertEqual(int(output._monolithic_coupled.force_generation.numpy()[0]), 1)


class TestRigidDynamics(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_rigid_mapping_and_terms,
        test_mixed_step_and_continuation,
        test_mixed_dense_linear,
        test_rigid_failures_and_reset,
        test_rigid_count_and_trial,
        test_rigid_force_channels,
        test_rigid_eligibility,
        test_rigid_checkpoint_rejections,
        test_rigid_empty_blocks_and_stops,
    ):
        add_function_test(TestRigidDynamics, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
