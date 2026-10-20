# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production FP64 contact/tet components against independent coordinate oracles."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.bsr_reduction import _FixedScalarBsrReduction
from newton._src.solvers.monolithic.contact import ContactObjective, ContactStatus
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton._src.solvers.monolithic.tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from newton._src.solvers.monolithic.tet_precision import TetObjective
from newton.tests.test_monolithic_linear_assembly import _dense_bsr
from newton.tests.test_monolithic_tet_derivatives import _dense
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_oracle import evaluate, penalty, rotation
from scripts.monolithic_reference.tet_fixtures import build_model
from scripts.monolithic_reference.tet_oracle import TetOracle


def _scene(device, *, shape="plane", margin=0.0):
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01))
    joint = builder.add_joint_revolute(-1, body)
    builder.add_articulation([joint])
    cfg = builder.ShapeConfig(margin=margin)
    if shape == "plane":
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=cfg)
    elif shape == "mixed":
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=cfg)
        builder.add_shape_box(
            body=body, xform=wp.transform((10.0, 10.0, 10.0), wp.quat_identity()), hx=0.02, hy=0.03, hz=0.01, cfg=cfg
        )
    elif shape == "sphere":
        builder.add_shape_sphere(body=body, radius=0.03, cfg=cfg)
    elif shape == "box":
        builder.add_shape_box(body=body, hx=0.02, hy=0.03, hz=0.01, cfg=cfg)
    elif shape == "mesh":
        mesh = newton.Mesh.create_box(0.02, 0.03, 0.01)
        mesh.build_sdf(max_resolution=32)
        builder.add_shape_mesh(body=body, mesh=mesh, cfg=cfg)
    elif shape == "ellipsoid":
        builder.add_shape_ellipsoid(body=body, rx=0.02, ry=0.03, rz=0.01, cfg=cfg)
    elif shape == "finite_plane":
        builder.add_shape_plane(body=body, width=1.0, length=1.0, cfg=cfg)
    else:
        getattr(builder, "add_shape_" + shape)(body=body, radius=0.02, half_height=0.03, cfg=cfg)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.002),
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
    return model, model.state()


def contact(model, *, fixed=False, mu=0.5, dtype=wp.float64):
    mapping = np.arange(model.particle_count, dtype=np.int32)
    if fixed:
        mapping[0], mapping[1:] = -1, np.arange(model.particle_count - 1)
        mass = model.particle_inv_mass.numpy()
        mass[0] = 0
        model.particle_inv_mass.assign(mass)
    return ContactObjective(
        model,
        mapping,
        q_dof_count=2,
        normal_stiffness=1e7,
        friction_coefficient=mu,
        friction_velocity_scale=0.01,
        smoothing_width=1e-4,
        particle_radius=0.001,
        dtype=dtype,
    )


def native(objective, *, batch_capacity=None):
    """Evaluate compact metadata through the production producer.

    The default single chunk keeps per-key geometry and law scratch inspectable
    in metadata key order.
    """
    count = objective.count if batch_capacity is None else batch_capacity
    return _ContactBatch(objective, batch_capacity=count, capacity=None)


def batch_matrix(batch):
    records = batch.pool.records
    builder = _FixedScalarBsrReduction(
        batch.contact.layout.scalar_dof_count,
        batch.pool.triplet_capacity,
        batch.device,
        use_cuda_graph=batch.device.is_cuda,
        dtype=records.triplet_values.dtype,
    )
    status = wp.zeros(1, dtype=int, device=batch.device)
    builder.build(
        SimpleNamespace(
            rows=records.triplet_rows,
            columns=records.triplet_columns,
            values=records.triplet_values,
            count=records.triplet_count,
            status=status,
        )
    )
    assert int(status.numpy()[0]) == 0
    return _dense_bsr(builder.matrix)


def state(model, x, q):
    device = model.device
    pose = np.array([[q[1], 0, 0, 0, np.sin(q[0] / 2), 0, np.cos(q[0] / 2)]])
    jt, jw = np.zeros((1, 2, 3)), np.zeros((1, 2, 3))
    jt[0, 1, 0], jw[0, 0, 1] = 1, 1
    return (
        wp.array(x, dtype=wp.vec3d, device=device),
        wp.array(pose, dtype=wp.transformd, device=device),
        wp.array([[q[1], 0, 0]], dtype=wp.vec3d, device=device),
        wp.array(jt, dtype=wp.vec3d, device=device),
        wp.array(jw, dtype=wp.vec3d, device=device),
    )


def fixture(device, kind):
    model, _ = _scene(device, shape=kind)
    transforms = model.shape_transform.numpy()
    if kind == "sphere":
        transforms[0, 2] = -0.03
    elif kind == "box":
        transforms[0, 2] = -0.01
    model.shape_transform.assign(transforms)
    x = model.particle_q.numpy().astype(float)
    # Place a small face near the top of the curved collider, with outward -Z normal.
    x[:, :2] *= 0.01
    x[:, 2] -= 0.0012
    return model, x


def energy(batch):
    # Copy: CPU numpy() views alias the producer output reused by the next evaluation.
    return batch.pool.energy.result.numpy()[:2].copy()


def residual(batch):
    return batch.pool.residual.result.numpy().copy()


def test_oracle_and_actual_trials(test, device):
    """Match the independent C-IE oracle and finite differences of the native producer."""
    rng = np.random.default_rng(893)
    for kind in ("plane", "sphere", "box"):
        for fixed in (False, True):
            with test.subTest(kind=kind, fixed=fixed):
                model, xn = fixture(device, kind)
                obj = contact(model, fixed=fixed)
                batch = native(obj)
                qn, q = np.zeros(2), np.array([0.003, 2e-6])
                x = xn + np.array([1e-6, 2e-6, 1e-6])
                if fixed:
                    x[0] = xn[0]
                batch.begin_step(*state(model, xn, qn))
                batch.evaluate(*state(model, x, q), dt=0.001)
                test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
                oracle = evaluate(model, q, x, qn, xn, kinds=[kind], mapping=obj.mapping.numpy())
                np.testing.assert_allclose(energy(batch), oracle[0].sum(axis=0), rtol=3e-10, atol=1e-15)
                np.testing.assert_allclose(residual(batch), oracle[1], rtol=3e-10, atol=1e-12)
                matrix = batch_matrix(batch)
                np.testing.assert_allclose(matrix, oracle[3], rtol=2e-5, atol=2e-4)
                test.assertLessEqual(
                    np.linalg.norm(matrix - matrix.T), np.finfo(np.float32).eps * np.linalg.norm(matrix)
                )
                test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * np.linalg.norm(matrix))
                direction = rng.normal(size=oracle[1].size)
                direction /= np.linalg.norm(direction)
                free = np.flatnonzero(obj.mapping.numpy() >= 0)
                errors = []
                for h in (2e-8, 5e-9, 1e-9):
                    values = []
                    for sign in (1, -1):
                        xt = x.copy()
                        xt[free] += sign * h * direction[2:].reshape(-1, 3)
                        batch.evaluate(*state(model, xt, q + sign * h * direction[:2]), dt=0.001)
                        test.assertEqual(int(batch.flags.numpy()[0]), 0)
                        values.append(residual(batch))
                    fd = (values[0] - values[1]) / (2 * h)
                    errors.append(
                        np.linalg.norm(fd - oracle[2] @ direction) / max(1, np.linalg.norm(oracle[2] @ direction))
                    )
                test.assertGreaterEqual(sum(e < 1e-5 for e in errors), 2, errors)
                test.assertFalse(obj.conservative_energy)


def test_friction_radial_tangent(test, device):
    """Match actual slip derivatives through zero, smoothing boundary and sliding."""
    along = np.array([0.6, 0.8, 0.0])
    across = np.array([-0.8, 0.6, 0.0])
    model, xn = fixture(device, "plane")
    batch = native(contact(model))
    q = np.zeros(2)
    batch.begin_step(*state(model, xn, q))
    # Revisit zero and reverse slip to detect stale weights/directions.
    for ratio in (0, 1e-10, 0.25, 0.9999, 1, 1.0001, 2, -2, 0):
        with test.subTest(ratio=ratio):
            x = xn + ratio * 1e-5 * along
            batch.evaluate(*state(model, x, q), dt=0.001)
            test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
            expected = evaluate(model, q, x, q, xn, kinds=["plane"])
            np.testing.assert_allclose(energy(batch), expected[0].sum(axis=0), atol=1e-15)
            np.testing.assert_allclose(residual(batch), expected[1], atol=1e-12)
            matrix = batch_matrix(batch)
            test.assertTrue(np.isfinite(matrix).all())
            test.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-6 * np.linalg.norm(matrix))
            for axis in (along, across):
                direction = np.concatenate((np.zeros(2), np.tile(axis, len(x))))
                values = []
                h = 1e-10
                for sign in (1, -1):
                    batch.evaluate(*state(model, x + sign * h * axis, q), dt=0.001)
                    test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
                    values.append(residual(batch))
                derivative = (values[0] - values[1]) / (2 * h)
                # Only the x block: the q rows also contain omitted FK geometry.
                error = np.linalg.norm((matrix @ direction - derivative)[2:])
                scale = max(1, np.linalg.norm(matrix[2:, 2:]) * np.linalg.norm(direction))
                test.assertLess(error / scale, 3e-5)
                if abs(ratio) >= 1.0001 and axis is along:
                    test.assertLess(np.linalg.norm((matrix @ direction)[2:]) / scale, 1e-6)


def test_friction_normal_motion_limit(test, device):
    """Keep the zero-slip tangent isotropic despite normal projection roundoff."""
    model, xn = fixture(device, "sphere")
    batch = native(contact(model))
    q = np.zeros(2)
    batch.begin_step(*state(model, xn, q))
    center = model.shape_transform.numpy()[0, :3].astype(float)
    for scale in (1.001, 1.0000001, 1.0):
        with test.subTest(scale=scale):
            x = center + scale * (xn - center)
            batch.evaluate(*state(model, x, q), dt=0.001)
            test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.SUCCESS)
            u = batch.current.y.numpy() - batch.start.y.numpy()
            n = batch.normal.numpy()
            slip = u - np.sum(n * u, axis=1)[:, None] * n
            test.assertLess(np.linalg.norm(slip), 1e-13)
            test.assertGreater(batch.weights.numpy()[:, 1].max(), 0)
            matrix = batch_matrix(batch)
            expected = evaluate(model, q, x, q, xn, kinds=["sphere"])[3]
            # FP32 factor sums can cancel in individual off-diagonal entries.
            test.assertLess(np.linalg.norm(matrix - expected), 5 * np.finfo(np.float32).eps * np.linalg.norm(expected))


def test_tet64_oracle(test, device):
    for material in ("kim_stable_no_log", "smith_log_stabilized"):
        for mass in ("lumped", "consistent"):
            for fixed in (False, True):
                with test.subTest(material=material, mass=mass, fixed=fixed):
                    model = build_model(device, fixed=fixed, gravity=(0, 0, -9.81))
                    obj = TetObjective(
                        model,
                        material_model=material,
                        mass_mode=mass,
                        density=wp.full(model.tet_count, 1000, dtype=float, device=device)
                        if mass == "consistent"
                        else None,
                    )
                    xn = model.particle_q.numpy().astype(float)
                    x = xn @ np.array([[1.02, 0.001, 0], [0, 0.99, 0.01], [0, 0, 1.01]])
                    vn = np.full_like(x, 0.02)
                    if fixed:
                        x[0], vn[0] = xn[0], 0
                    forces = np.full_like(x, 0.003, dtype=np.float32)
                    inputs = [
                        wp.array(a, dtype=dtype, device=device)
                        for a, dtype in ((x, wp.vec3d), (xn, wp.vec3d), (vn, wp.vec3d), (forces, wp.vec3))
                    ]
                    obj.evaluate(*inputs, model.gravity, dt=0.001)
                    test.assertEqual(obj.flags.numpy()[0], 0)
                    oracle = TetOracle(model, obj.physics)
                    parts, r, _ = oracle.evaluate(x, xn, vn, model.gravity.numpy()[0], forces, 0.001)
                    np.testing.assert_allclose(
                        obj.parts.numpy(), [parts[0], parts[1], parts[2] + parts[3]], rtol=1e-11, atol=1e-15
                    )
                    np.testing.assert_allclose(obj.residual.numpy().ravel(), r, rtol=1e-11, atol=1e-12)
                    # Sub-FP32 coordinate corrections remain visible to the direct residual.
                    dx = np.zeros_like(x)
                    dx[-1, 2] = 1e-11
                    inputs[0].assign(x + dx)
                    obj.evaluate(*inputs, model.gravity, dt=0.001)
                    test.assertGreater(np.linalg.norm(obj.residual.numpy().ravel() - r), 1e-9)


def test_law_branches_and_failures(test, device):
    """Keep frozen start references and reject invalid geometry without stale results."""
    model, xn = fixture(device, "mixed")
    obj = contact(model)
    test.assertEqual(obj.count, 24)
    batch = native(obj)
    q = np.zeros(2)
    batch.begin_step(*state(model, xn, q))
    frozen = batch.start.y.numpy().copy()
    for ratio in (0, 0.25, 1, 2):
        x = xn + np.array([ratio * 1e-5, 0, 0])
        batch.evaluate(*state(model, x, q), dt=0.001)
        oracle = evaluate(model, q, x, q, xn, kinds=["plane", "box"])
        np.testing.assert_allclose(residual(batch), oracle[1], atol=1e-12)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        # Far box is still queried; zero force does not remove a key.
        test.assertTrue(np.all(batch.current.distance.numpy()[1::2] > 10))
    np.testing.assert_array_equal(batch.start.y.numpy(), frozen)
    for bad in ("nan", "degenerate", "backface", "quaternion"):
        inputs = list(state(model, xn, q))
        invalid = xn.copy()
        if bad == "nan":
            invalid[0, 0] = np.nan
        elif bad == "degenerate":
            invalid[1] = invalid[0]
        elif bad == "backface":
            invalid[[1, 2]] = invalid[[2, 1]]
        else:
            pose = inputs[1].numpy()
            pose[0, 6] = 2
            inputs[1].assign(pose)
        inputs[0].assign(invalid)
        batch.evaluate(*inputs, dt=0.001)
        test.assertNotEqual(int(batch.flags.numpy()[0]), 0, bad)
        np.testing.assert_array_equal(batch.start.y.numpy(), frozen)
    batch.evaluate(*state(model, xn, q), dt=0.001)
    test.assertEqual(int(batch.flags.numpy()[0]), 0)
    model, xn = fixture(device, "plane")
    batch = native(contact(model, mu=0))
    batch.begin_step(*state(model, xn, q))
    batch.evaluate(*state(model, xn + np.array([1e-4, 0, 0]), q), dt=0.001)
    np.testing.assert_array_equal(batch.weights.numpy()[:, 1:], 0)
    for kind in ("finite_plane", "ellipsoid"):
        invalid, _ = _scene(device, shape=kind)
        with test.assertRaises(ValueError):
            contact(invalid)


def test_start_domain_and_boundary_branches(test, device):
    model, xn = fixture(device, "plane")
    batch = native(contact(model))
    q = np.zeros(2)
    bad = xn.copy()
    bad[[1, 2]] = bad[[2, 1]]
    batch.begin_step(*state(model, bad, q))
    batch.evaluate(*state(model, xn, q), dt=0.001)
    test.assertEqual(int(batch.flags.numpy()[0]), ContactStatus.INVALID_GEOMETRY)
    # Verify actual scalar branches on both sides of normal activation/smoothing.
    for gap in (-1.001e-4, -0.999e-4, 0, 0.999e-4, 1.001e-4):
        x = xn.copy()
        x[:, 2] += 0.001 + gap - xn[0, 2]
        batch.begin_step(*state(model, x, q))
        batch.evaluate(*state(model, x, q), dt=0.001)
        expected = evaluate(model, q, x, q, x, kinds=["plane"])
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        np.testing.assert_allclose(residual(batch), expected[1], atol=1e-12)
    # Common rigid motion gives zero relative slip even with rotation.
    batch.begin_step(*state(model, xn, q))
    moved_q = np.array([0.23, 0.12])
    moved_x = xn @ rotation(moved_q[0]).T + [moved_q[1], 0, 0]
    batch.evaluate(*state(model, moved_x, moved_q), dt=0.001)
    np.testing.assert_allclose(batch.current.y.numpy(), batch.start.y.numpy(), atol=3e-17)
    test.assertLess(energy(batch)[1], 1e-25)


def test_tet_tangent_and_domain(test, device):

    for material in ("kim_stable_no_log", "smith_log_stabilized"):
        model = build_model(device, fixed=True)
        obj = TetObjective(
            model,
            material_model=material,
            mass_mode="consistent",
            density=wp.full(model.tet_count, 1000, dtype=float, device=device),
        )
        xn = model.particle_q.numpy().astype(float)
        x = xn * [1.02, 0.97, 1.01]
        inputs = [wp.array(a, dtype=wp.vec3d, device=device) for a in (x, xn, np.zeros_like(x))]
        force = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        obj.evaluate(*inputs, force, model.gravity, dt=0.001)
        expected_residual = obj.residual.numpy().copy()
        pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), obj.mapping.numpy(), x_dof_start=2)
        workspace = create_tet_assembly_workspace(pattern, tet_count=model.tet_count, device=device)
        workspace.physics = obj.physics
        scatter = TetScatterBuffers(
            wp.empty(len(pattern.ax_rows), dtype=wp.mat33, device=device),
            wp.empty(len(pattern.global_rows), dtype=float, device=device),
        )
        assemble_tet_residual_tangent(
            model,
            candidate_particle_q=inputs[0],
            particle_q_n=inputs[1],
            particle_qd_n=inputs[2],
            frozen_particle_f=force,
            dynamic_particle_ids=obj.ids,
            particle_to_dynamic=obj.mapping,
            residual_x=wp.empty(obj.ids.size, dtype=wp.vec3, device=device),
            dt=0.001,
            min_det_f_guard=0.2,
            workspace=workspace,
            scatter=scatter,
            frozen_gravity=model.gravity,
        )
        test.assertEqual(workspace.failure_flags.numpy()[0], 0)
        owner, global_b = _dense(pattern, scatter, 3 * obj.ids.size)
        np.testing.assert_array_equal(owner, global_b)
        np.testing.assert_allclose(owner, owner.T, atol=1e-3)
        test.assertGreater(np.linalg.eigvalsh(owner).min(), 0)
        np.testing.assert_array_equal(obj.residual.numpy(), expected_residual)
        # Actual E/r finite differences with unchanged fixed node.
        direction = np.random.default_rng(811).normal(size=x.shape)
        direction[0] = 0
        for h in (1e-7, 3e-8):
            energies = []
            for sign in (1, -1):
                inputs[0].assign(x + sign * h * direction)
                obj.evaluate(*inputs, force, model.gravity, dt=0.001)
                energies.append(obj.parts.numpy().sum())
            test.assertAlmostEqual(
                (energies[0] - energies[1]) / (2 * h), np.sum(expected_residual * direction[1:]), delta=1e-7
            )
        for failure in ("fixed", "det", "nonfinite"):
            invalid = x.copy()
            if failure == "fixed":
                invalid[0, 0] += 1e-10
            elif failure == "det":
                invalid[3] = invalid[0]
            else:
                invalid[3, 0] = np.nan
            inputs[0].assign(invalid)
            obj.evaluate(*inputs, force, model.gravity, dt=0.001)
            test.assertNotEqual(obj.flags.numpy()[0], 0)
        inputs[0].assign(x)
        obj.evaluate(*inputs, force, model.gravity, dt=0.001)
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                obj.evaluate(*inputs, force, model.gravity, dt=0.001)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(obj.residual.numpy(), expected_residual, atol=1e-12)


def test_zero_smoothing(test, device):
    """Preserve the frozen quadratic normal law without introducing smoothing."""
    model, xn = fixture(device, "plane")
    kwargs = {
        "q_dof_count": 2,
        "normal_stiffness": 1e7,
        "friction_coefficient": 0.0,
        "friction_velocity_scale": 0.01,
        "particle_radius": 0.001,
    }
    with test.assertRaises(ValueError):
        ContactObjective(model, np.arange(4, dtype=np.int32), smoothing_width=-1e-4, **kwargs)
    batch = native(ContactObjective(model, np.arange(4, dtype=np.int32), smoothing_width=0.0, **kwargs))
    area = batch.all_samples.area.numpy()
    for height in (0.0008, 0.001, 0.0012):
        x = xn.copy()
        x[:, 2] += height - xn[0, 2]
        inputs = state(model, x, np.zeros(2))
        batch.begin_step(*inputs)
        batch.evaluate(*inputs, dt=0.001)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        gaps = batch.current.distance.numpy() - 0.001
        expected = np.array([penalty(g, 1e7 * a / 3, 0.0) for g, a in zip(gaps, area, strict=True)])
        np.testing.assert_allclose(energy(batch)[0], expected[:, 0].sum(), atol=1e-15)
        np.testing.assert_allclose(batch.weights.numpy()[:, 0], expected[:, 2], atol=1e-12)


class TestContactComponents(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_friction_normal_motion_limit,
        test_friction_radial_tangent,
        test_zero_smoothing,
        test_oracle_and_actual_trials,
        test_tet64_oracle,
        test_law_branches_and_failures,
        test_start_domain_and_boundary_branches,
        test_tet_tangent_and_domain,
    ):
        add_function_test(TestContactComponents, _test.__name__, _test, devices=[_device])

if __name__ == "__main__":
    unittest.main()
