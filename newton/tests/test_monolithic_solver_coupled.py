# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coupled FP64 continuation, shared nonlinear iteration and atomic publication."""

import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import GeoType
from newton._src.geometry.sdf_texture import TextureSDFData, create_sparse_sdf_textures
from newton._src.solvers.monolithic.articulation_variational import ArticulationKinematics
from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic, _StepFailure
from newton.tests.test_monolithic_contact_mesh import synthetic
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.p3_articulation_oracle import Kinematics, joint_terms
from scripts.monolithic_reference.p3_contact_oracle import local
from scripts.monolithic_reference.p3_tet_oracle import TetOracle, material


def scene(device, *, contact=True, fixed=False, particles=True, articulation=True):
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    if articulation:
        inertia = wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01)
        root = builder.add_link(mass=1.0, inertia=inertia)
        body = builder.add_link(mass=1.0, inertia=inertia)
        joints = [
            builder.add_joint_revolute(-1, root, axis=(0, 1, 0)),
            builder.add_joint_prismatic(root, body, axis=(0, 0, 1)),
        ]
        builder.add_articulation(joints)
        builder.add_shape_plane(body=body, width=0, length=0, cfg=builder.ShapeConfig(density=0))
    if particles:
        builder.add_soft_mesh(
            pos=(0, 0, 0.0008),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02)],
            indices=[0, 1, 2, 3],
            density=1000,
            k_mu=10000 / 2.6,
            k_lambda=10000 * 0.3 / (1.3 * 0.4),
            k_damp=0,
            particle_radius=0.001,
            tri_ke=0,
            tri_ka=0,
            tri_kd=0,
            tri_drag=0,
            tri_lift=0,
            edge_ke=0,
            edge_kd=0,
        )
        if fixed:
            builder.particle_mass[0] = 0
    model = builder.finalize(device=device)
    if not contact:
        model.shape_flags.zero_()
    return model


def solver(model, law="C-EE", *, graph=False, mode="diagnostic", globalization="ResNorm", **kwargs):
    return SolverMonolithic(
        model,
        contact_model=law,
        globalization=globalization,
        friction_velocity_scale=0.01,
        tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
        pcg_mode=mode,
        use_cuda_graph=graph,
        joint_terms=SolverMonolithic.JointTerms(limits=True, limit_width=(0.01, 0.01)),
        **kwargs,
    )


def snapshot(state):
    result = {name: value.numpy().copy() for name, value in vars(state).items() if isinstance(value, wp.array)}
    for name, value in vars(state._monolithic_coupled).items():
        if isinstance(value, wp.array):
            result["history." + name] = value.numpy().copy()
    return result


def assert_snapshot(test, state, expected):
    got = snapshot(state)
    test.assertEqual(got.keys(), expected.keys())
    for name, value in expected.items():
        np.testing.assert_array_equal(got[name], value, err_msg=name)


def test_joint_particle_step(test, device):
    """Solve all laws with a nonzero q-x block and publish the complete final state."""
    for law in ("C-EE", "C-IE", "C-II"):
        model = scene(device)
        run = solver(model, law)
        before = run.create_state(dt=0.001)
        after = run._states.copy(before)
        for _ in range(3):
            old = snapshot(before)
            run.step(before, after, model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            assert_snapshot(test, before, old)
            for name in ("convergence_ratio", "convergence_ratio_q", "convergence_ratio_x"):
                test.assertLessEqual(getattr(run.last_stats, name), 1)
            matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
            test.assertGreater(np.linalg.norm(matrix[:2, 2:]), 0)
            test.assertLess(np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix), 1e-6)
            h = after._monolithic_coupled
            test.assertEqual(h.force_valid.numpy()[0], 1)
            test.assertEqual(h.force_generation.numpy()[0], run.last_stats.step_generation)
            np.testing.assert_allclose(
                h.contact_particle_force.numpy().sum(axis=0) + h.contact_body_force.numpy()[:, :3].sum(axis=0),
                0,
                atol=1e-12,
            )
            np.testing.assert_allclose(h.contact_joint_force.numpy(), -run._contact.residual.numpy()[:2], atol=1e-12)
            test.assertTrue(np.isfinite(h.link_vsym.numpy()).all())
            torque = np.cross(after.particle_q.numpy(), h.contact_particle_force.numpy()).sum(axis=0)
            torque += (
                np.cross(run._kinematics.com.numpy(), h.contact_body_force.numpy()[:, :3])
                + h.contact_body_force.numpy()[:, 3:]
            ).sum(axis=0)
            np.testing.assert_allclose(torque, 0, atol=1e-12)
            before, after = after, before


def test_state_and_rollback(test, device):
    """Preserve distinct outputs and every history/force array on failed publication."""
    model = scene(device, fixed=True)
    run = solver(model)
    before = run.create_state(dt=0.001)
    after = run._states.copy(before)
    run.step(before, after, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    # The output deliberately differs from input; rollback must restore that output.
    input_snapshot, output_snapshot = snapshot(before), snapshot(after)
    failure = _StepFailure(run.Status.NONFINITE, "injected final force failure")
    with patch.object(run, "_finish_candidate", side_effect=failure):
        run.step(before, after, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, before, input_snapshot)
    assert_snapshot(test, after, output_snapshot)
    with patch.object(run, "_finish_candidate", side_effect=failure):
        run.step(before, before, model.control(), dt=0.001)
    assert_snapshot(test, before, input_snapshot)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.npz"
        run.save_checkpoint(after, path)
        restored = run.create_state(dt=0.001)
        run.load_checkpoint(restored, path)
        assert_snapshot(test, restored, output_snapshot)
    after.particle_qd.assign(after.particle_qd.numpy() + 0.01)
    with test.assertRaises(ValueError):
        run._states.validate(after)
    with test.assertRaises(ValueError):
        ArticulationKinematics(model)
    with test.assertRaises(ValueError):
        SolverMonolithic(model, contact_model="C-IE", globalization="Armijo", friction_velocity_scale=0.01)


def test_modes_and_sparse_posterior(test, device):
    """Compare all execution modes and the true sparse posterior against dense FP64."""
    for law in ("C-EE", "C-IE", "C-II"):
        reference = None
        production = None
        for mode, graph in [("diagnostic", False), ("production", False)] + (
            [("production", True)] if device.is_cuda else []
        ):
            model = scene(device)
            run = solver(model, law, mode=mode, graph=graph)
            state = run.create_state(dt=0.001)
            for dt in (0.001, 0.001, 0.0005):
                run.step(state, state, model.control(), dt=dt)
                test.assertTrue(run.last_stats.converged, run.last_stats)
            coords = np.r_[state.joint_q.numpy(), state.particle_q.numpy().ravel()]
            if reference is None:
                reference = coords
            else:
                # Different PCG schedules may stop at different valid roots; use
                # the approved discrete gates, not a machine-equality assertion.
                np.testing.assert_allclose(coords[:2], reference[:2], atol=5e-4, rtol=0)
                np.testing.assert_allclose(coords[2:], reference[2:], atol=5e-5, rtol=0)
            if mode == "production":
                if production is None:
                    production = coords
                else:
                    np.testing.assert_allclose(coords, production, atol=2e-11, rtol=1e-9)
            matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix.astype(float)
            direction = np.linspace(-0.3, 0.7, len(matrix)).astype(np.float32)
            residual = np.linspace(0.2, -0.1, len(matrix))
            scale = run._linear.scale.numpy().astype(float)
            run._delta.assign(direction)
            actual = run._physical_linear_error(None, direction, residual, scale)
            np.testing.assert_allclose(
                actual, scale * (matrix @ direction.astype(float) + residual), atol=2e-12, rtol=2e-14
            )
            if graph:
                test.assertGreater(run._graphs.summary()["segments"]["coupled_objective"]["captures"], 1)
                test.assertGreater(run._graphs.summary()["segments"]["coupled_posterior"]["replays"], 0)
            if law != "C-EE":
                test.assertTrue(all(row.get("Armijo") is None for row in run.last_stats.variational["trials"]))
    for globalization in ("Armijo", "ArmijoOrResNorm"):
        model = scene(device)
        run = solver(model, globalization=globalization)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)


def test_failure_gates(test, device):
    """Refuse overflow, bad geometry/tets, false linear success and exhausted budgets."""
    for failure in ("capacity", "posterior", "budget", "trial", "tet", "geometry", "small", "final", "postpublish"):
        model = scene(device)
        options = {"contact_factor_capacity": 0} if failure == "capacity" else {}
        if failure == "budget":
            options["newton_max_iterations"] = 1
        run = solver(model, **options)
        state = run.create_state(dt=0.001)
        out = run._states.copy(state)
        expected, expected_out = snapshot(state), snapshot(out)
        if failure == "posterior":
            context = patch.object(run, "_physical_linear_error", return_value=np.full(14, np.nan))
        elif failure == "trial":
            context = patch(
                "newton._src.solvers.monolithic.variational_iteration.acceptance",
                return_value=(False, False, False, False),
            )
        elif failure == "small":
            context = ExitStack()
            context.enter_context(patch.object(run, "_physical_linear_error", return_value=np.zeros(14)))
            context.enter_context(
                patch.object(run, "_take_trial", side_effect=lambda *args, run=run: run._trial.assign(run._accepted))
            )
            context.enter_context(
                patch(
                    "newton._src.solvers.monolithic.variational_iteration.acceptance",
                    return_value=(True, True, True, True),
                )
            )
        elif failure == "geometry":
            evaluate = run._contact.evaluate

            def invalid(*args, evaluate=evaluate, run=run, **kwargs):
                evaluate(*args, **kwargs)
                run._contact.flags.fill_(2)

            context = patch.object(run._contact, "evaluate", side_effect=invalid)
        elif failure == "tet":
            evaluate = run._tet.evaluate

            def invalid(*args, evaluate=evaluate, run=run, **kwargs):
                evaluate(*args, **kwargs)
                run._tet.flags.fill_(1)

            context = patch.object(run._tet, "evaluate", side_effect=invalid)
        elif failure in ("final", "postpublish"):
            context = patch.object(
                run,
                "_finish_candidate" if failure == "final" else "_joint_force_metrics",
                side_effect=_StepFailure(run.Status.NONFINITE, "injected publication failure"),
            )
        else:
            context = nullcontext()
        with context:
            run.step(state, out, model.control(), dt=0.001)
        test.assertFalse(run.last_stats.converged, (failure, run.last_stats))
        test.assertTrue(run.last_stats.rolled_back)
        assert_snapshot(test, state, expected)
        assert_snapshot(test, out, expected_out)


def test_binding_and_resume(test, device):
    """Bind configuration and physics; resume force generations from the checkpoint."""
    model = scene(device)
    run = solver(model)
    state = run.create_state(dt=0.001)
    for _ in range(3):
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
    resumed = solver(model)
    resumed.step(state, state, model.control(), dt=0.001)
    test.assertTrue(resumed.last_stats.converged)
    test.assertEqual(state._monolithic_coupled.force_generation.numpy()[0], 4)
    for other in (solver(model, "C-II"), solver(model, newton_max_iterations=9)):
        with test.assertRaises(ValueError):
            other._states.validate(state)
    state.particle_q = wp.clone(state.particle_q)
    resumed.step(state, state, model.control(), dt=0.001)
    test.assertTrue(resumed.last_stats.converged)
    model.tet_materials.assign(model.tet_materials.numpy() * 1.01)
    before = snapshot(state)
    with test.assertRaises(ValueError):
        resumed.step(state, state, model.control(), dt=0.001)
    assert_snapshot(test, state, before)


def test_contact_off(test, device):
    """Contact-off agrees with independent q/x equations at fixed physics."""
    from scipy.optimize import root

    model = scene(device, contact=False)
    run = solver(model)
    state = run.create_state(dt=0.001)
    for _ in range(3):
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        # The contact-free system separates into independent q and x residuals.
        # Solve the independent NumPy equations instead of invoking retired solvers.
        initial = np.r_[run._original.joint_q.numpy(), run._original.particle_q.numpy().ravel()]
        reference = root(
            lambda z: independent_objective(run, z[:2], z[2:].reshape(-1, 3), 0.001)[1],
            initial,
            tol=1e-10,
        )
        test.assertTrue(reference.success, reference.message)
        np.testing.assert_allclose(reference.fun, 0, atol=1e-7, rtol=0)
        np.testing.assert_allclose(state.joint_q.numpy(), reference.x[:2], atol=5e-4, rtol=0)
        np.testing.assert_allclose(state.particle_q.numpy(), reference.x[2:].reshape(-1, 3), atol=5e-5, rtol=0)
        matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
        test.assertEqual(np.count_nonzero(matrix[:2, 2:]), 0)
        test.assertEqual(np.count_nonzero(state._monolithic_coupled.contact_particle_force.numpy()), 0)


def independent_objective(run, q, x, dt):
    """NumPy composition for this two-joint, one-plane, one-tet fixture."""

    model, obj = run.model, run._objective
    q, x, dt = np.asarray(q, dtype=np.float64), np.asarray(x, dtype=np.float64), float(dt)
    # Promote every frozen numeric input before reference arithmetic, including
    # scalar coefficients that NumPy otherwise keeps in FP32.
    frozen = {
        name: getattr(obj, name).numpy().astype(np.float64)
        for name in (
            "qn",
            "vn",
            "force",
            "predicted_t",
            "predicted_r",
            "params",
            "armature",
            "damping",
            "qt",
            "vt",
            "joint_force",
        )
    }
    k = Kinematics(model)
    com, rot, jt, jr, *_ = k.evaluate(q)
    old_com, old_rot, *_ = k.evaluate(frozen["qn"])
    force = frozen["force"].copy()
    energy, qr, _ = k.inertia(q, frozen["predicted_t"], frozen["predicted_r"], dt, force)
    energy += np.sum(force * old_com)
    je, jres, jb = joint_terms(
        q,
        frozen["qn"],
        frozen["vn"],
        dt,
        frozen["params"],
        frozen["armature"],
        frozen["damping"],
        frozen["qt"],
        frozen["vt"],
        frozen["joint_force"],
    )
    energy += je.sum()
    qr += jres
    qb = np.diag(jb)
    for body in range(k.b):
        c = rot[body] @ k.second[body] @ frozen["predicted_r"][body].T
        kr = (np.trace(c) * np.eye(3) - (c + c.T) / 2) / dt**2
        w, v = np.linalg.eigh(kr)
        kr = (v * np.maximum(w, 0)) @ v.T
        a = jr[body] @ rot[body].T
        jw = np.stack((a[:, 2, 1], a[:, 0, 2], a[:, 1, 0]), axis=1)
        qb += float(k.body_mass[body]) * jt[body] @ jt[body].T / dt**2 + jw @ kr @ jw.T
    tet = TetOracle(model, run._tet.physics)
    xn, vn = run._original.particle_q.numpy(), run._original.particle_qd.numpy()
    te, xr, _ = tet.evaluate(x, xn, vn, run._frozen_gravity.numpy()[0], run._frozen_force.numpy(), dt)
    energy += te.sum()
    residual = np.r_[qr, xr]
    b = np.zeros((len(residual), len(residual)))
    b[:2, :2] = qb
    xb = tet.mass / dt**2
    for ids, pose, params in zip(tet.indices, tet.poses, tet.materials, strict=True):
        f = (x[ids[1:]] - x[ids[0]]).T @ pose
        _, _, raw = material(f, *params[:2], tet.smith)
        grads = np.vstack((-pose.sum(axis=0), pose))
        g = np.column_stack([np.kron(ga[:, None], np.eye(3)) for ga in grads])
        nodal = g.T @ raw @ g / (6 * np.linalg.det(pose))
        w, v = np.linalg.eigh(nodal)
        nodal = (v * np.maximum(w, 0)) @ v.T
        dofs = (3 * ids[:, None] + np.arange(3)).ravel()
        xb[np.ix_(dofs, dofs)] += nodal
    b[2:, 2:] = xb[np.ix_(tet.dofs, tet.dofs)]
    if run._contact.count == 0:
        return energy, residual, b
    rest = model.particle_q.numpy().astype(float)
    mapping = run._tet.mapping.numpy()
    # The plane is attached at the second body's COM with identity local pose.
    for ids in model.tri_indices.numpy():
        area = np.linalg.norm(np.cross(rest[ids[1]] - rest[ids[0]], rest[ids[2]] - rest[ids[0]])) / 2
        for slot in range(3):
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            point = bary @ x[ids]
            y = rot[1].T @ (point - com[1])
            yn = old_rot[1].T @ (bary @ xn[ids] - old_com[1])
            j = np.zeros((3, len(residual)))
            for dof in range(2):
                j[:, dof] = jr[1, dof].T @ (point - com[1]) - rot[1].T @ jt[1, dof]
            for a, node in enumerate(ids):
                if mapping[node] >= 0:
                    col = 2 + 3 * mapping[node]
                    j[:, col : col + 3] = bary[a] * rot[1].T
            ce, cr, _, cb = local(
                y,
                yn,
                "plane",
                np.ones(3),
                run._contact.contact_model,
                offset=float(model.particle_radius.numpy()[0] + model.shape_margin.numpy()[0]),
                stiffness=1e7 * area / 3,
                a=0.01 * dt,
            )
            energy += ce.sum()
            residual += j.T @ cr
            b += j.T @ cb @ j
    return energy, residual, b


def test_complete_oracle_and_noise(test, device):
    """Check actual coupled E/r/B and 32-repeat noise at initial, trial and near-root states."""
    for law in ("C-EE", "C-IE", "C-II"):
        model = scene(device)
        run = solver(model, law)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        for kind in ("initial", "trial", "root"):
            candidate = run._states.copy(run._original if kind == "initial" else state)
            if kind == "trial":
                candidate.joint_q.assign(candidate.joint_q.numpy() + np.array([1e-5, -2e-6]))
                candidate.particle_q.assign(candidate.particle_q.numpy() + np.array([2e-6, -3e-6, 1e-6]))
            q, x = candidate.joint_q.numpy().copy(), candidate.particle_q.numpy().copy()
            expected_e, expected_r, expected_b = independent_objective(run, q, x, 0.001)
            energy_errors = []
            residual_errors = []
            for _ in range(32):
                energy, residual, _ = run._evaluate(candidate, 0.001)
                energy_errors.append(abs(energy - expected_e))
                residual_errors.append(abs(residual - expected_r))
            run._assemble(0)
            scale = run._linear.scale.numpy().astype(float)
            actual_b = run._linear.densify_for_test(generation=run._generation).raw_matrix
            np.testing.assert_allclose(actual_b, expected_b, rtol=2e-5, atol=2e-3)
            test.assertLessEqual(8 * max(energy_errors), 5.820766091346741e-11)
            envelope = 8 * np.max(residual_errors, axis=0)
            test.assertLessEqual(envelope.max(), 2.0935893019213836e-9)
            for name, part in [("global", slice(None)), ("q", slice(0, 2)), ("x", slice(2, None))]:
                reference = np.linalg.norm(scale[part] * run._reference_residual[part]) / np.sqrt(len(expected_r[part]))
                goal = (
                    getattr(run._config, "merit_absolute_" + name)
                    + getattr(run._config, "merit_relative_" + name) * reference
                )
                test.assertLess(
                    np.linalg.norm(scale[part] * envelope[part]) / np.sqrt(len(expected_r[part])), 0.1 * goal
                )
            if law == "C-EE" and kind == "trial":
                direction = np.linspace(-0.7, 0.9, len(expected_r))
                direction[:2] *= 0.01
                for h in (1e-7, 3e-8):
                    values = []
                    for sign in (1, -1):
                        candidate.joint_q.assign(q + sign * h * direction[:2])
                        candidate.particle_q.assign(x + sign * h * direction[2:].reshape(-1, 3))
                        values.append(run._evaluate(candidate, 0.001, build_tangent=False)[0])
                    test.assertAlmostEqual((values[0] - values[1]) / (2 * h), expected_r @ direction, delta=1e-7)


def test_mesh_binding_graph(test, device):
    """Bind owned texels and indirection and replay a real mesh coupled objective."""
    if device.is_cpu:
        test.skipTest("Mesh SDF textures require CUDA")
    data = synthetic()
    data["coarse_sdf"] = np.broadcast_to(
        np.linspace(-0.125, 0.125, 3, dtype=np.float32)[:, None, None], (3, 3, 3)
    ).copy()
    data["subgrid_data"] = np.broadcast_to(np.linspace(-0.125, 0, 3, dtype=np.float32)[:, None, None], (3, 3, 3)).copy()
    descriptor, coarse, fine = create_sparse_sdf_textures(data, device=device)
    model = scene(device)
    model.shape_type.assign(np.array([int(GeoType.MESH)], dtype=np.int32))
    model.shape_scale.assign(np.ones((1, 3), dtype=np.float32))
    model._shape_sdf_index.assign(np.array([0], dtype=np.int32))
    model._texture_sdf_data = wp.array([descriptor], dtype=TextureSDFData, device=device)
    model._texture_sdf_coarse_textures = [coarse]
    model._texture_sdf_subgrid_textures = [fine]
    model._texture_sdf_subgrid_start_slots = [descriptor.subgrid_start_slots]
    run = solver(model, mode="production", graph=True)
    state = run.create_state(dt=0.001)
    for _ in range(2):
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
    expected = snapshot(state)
    _name, _index, texture, pixels = run._textures[0]
    texture.copy_from(pixels + np.float32(0.001))
    with test.assertRaisesRegex(ValueError, "texels changed"):
        run.step(state, state, model.control(), dt=0.001)
    assert_snapshot(test, state, expected)
    texture.copy_from(pixels)
    slots = model._texture_sdf_subgrid_start_slots[0]
    model._texture_sdf_subgrid_start_slots[0] = wp.clone(slots)
    with test.assertRaisesRegex(ValueError, "indirection replaced"):
        run.step(state, state, model.control(), dt=0.001)
    assert_snapshot(test, state, expected)


class TestCoupled(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestCoupled, "test_joint_particle_step", test_joint_particle_step, devices=[_device])
    add_function_test(TestCoupled, "test_state_and_rollback", test_state_and_rollback, devices=[_device])
    for _test in (
        test_modes_and_sparse_posterior,
        test_failure_gates,
        test_binding_and_resume,
        test_contact_off,
        test_complete_oracle_and_noise,
        test_mesh_binding_graph,
    ):
        add_function_test(TestCoupled, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
