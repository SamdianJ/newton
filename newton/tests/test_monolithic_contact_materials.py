# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify open material surfaces independently of closed tet contact geometry."""

import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from math import asin, cos, pi, sqrt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry import GeoType
from newton._src.solvers.monolithic import material_runtime
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.contact_pipeline import _SpatialContactObjective
from newton._src.solvers.monolithic.contact_sampling import ContactSamples
from newton._src.solvers.monolithic.contact_spatial import _ContactSpatial
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import MonolithicLinearCapacities, MonolithicLinearGeneration
from newton._src.solvers.monolithic.material_inputs import _MaterialInputs
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_material_state import snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_routes import pooled_audit_route


def scene(*, cable=False, body=False, device="cpu", mixed=False, fixed=False, mesh=False, plane=False):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    target = -1
    if body == "articulation":
        target = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.1)))
        builder.add_articulation(
            [builder.add_joint_prismatic(-1, target, axis=wp.vec3(0, 0, 1), limit_ke=0, limit_kd=0)]
        )
    elif body:
        target = builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(0.1)))
    if plane:
        builder.add_shape_plane(body=target, width=0.0, length=0.0, cfg=builder.ShapeConfig(density=0.0))
    elif mesh:
        source = newton.Mesh.create_box(2.1, 1.1, 0.1, compute_inertia=False)
        if wp.get_device(device).is_cuda:
            source.build_sdf(
                device=device, max_resolution=64, margin=0.2, narrow_band_range=(-0.2, 0.2), texture_format="float32"
            )
        builder.add_shape_mesh(
            target,
            mesh=source,
            xform=wp.transform(wp.vec3(0, 0, -0.1), wp.quat_identity()),
            cfg=builder.ShapeConfig(density=0.0),
        )
    elif body:
        builder.add_shape_box(
            target,
            hx=4.0,
            hy=2.0,
            hz=0.1,
            xform=wp.transform(wp.vec3(0, 0, -0.1), wp.quat_identity()),
            cfg=builder.ShapeConfig(density=0.0),
        )
    else:
        builder.add_shape_plane(width=0.0, length=0.0)
    cloth, cables = [], []
    if cable or mixed:
        ids = tuple(builder.add_particle((i, 0, 0.009), (0, 0, 0), m) for i, m in enumerate((0.5, 1, 0.5)))
        cables.append(
            SolverMonolithic.CableInput(
                particle_ids=ids,
                EA=100,
                bending=((1, 0), (0, 1)),
                GJ=1,
                radius=0.01,
                linear_density=1,
                rest_theta=(0, 0),
                rest_director=((0, 1, 0), (0, 1, 0)),
                fixed_theta=(fixed, fixed),
            )
        )
    if not cable or mixed:
        first = builder.particle_count
        face = builder.tri_count
        for p in ((0, 0, 0.004), (1, 0, 0.004), (0, 1, 0.004)):
            builder.add_particle(wp.vec3(p), wp.vec3(0), 1 / 6)
        builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
        cloth.append(
            SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 2), thickness=0.01, surface_density=1)
        )
    if mixed:
        first = builder.particle_count
        add_contact_free_tet(builder)
        for particle in range(first, builder.particle_count):
            builder.particle_radius[particle] = 0.0001
        builder.add_body(mass=1.0, inertia=wp.diag(wp.vec3(0.1)))
    if fixed:
        builder.particle_mass[:] = [0.0] * builder.particle_count
    model = builder.finalize(device=device)
    materials = _MaterialInputs(model, cloth, cables)
    actors = ActorTable.from_model(model, materials=materials)
    return model, materials, actors, cloth, cables


def contact_solver(
    *,
    cable=False,
    body=False,
    device="cpu",
    mu=0.0,
    smoothing=1e-5,
    mesh=False,
    plane=False,
    mixed=False,
    fixed=False,
    **options,
):
    model, _, _, cloth, cables = scene(
        cable=cable, body=body, device=device, mixed=mixed, fixed=fixed, mesh=mesh, plane=plane
    )
    options.setdefault("mass_mode", "lumped")
    if mesh and wp.get_device(device).is_cuda:
        options["smooth_sdf"] = build_smooth_sdf(model)
    solver = SolverMonolithic(
        model,
        cloth_inputs=cloth,
        cable_inputs=cables,
        friction_velocity_scale=0.1,
        contact_stiffness=100,
        friction_coefficient=mu,
        normal_smoothing=smoothing,
        **options,
    )

    material_runtime.initialize(solver)
    return solver


def query(solver, state, *, begin=False):
    c, k, h = solver._contact, solver._kinematics, state._monolithic_coupled
    q = k.gather_q(state)
    k.evaluate(q)
    c._batch.material_surface.update(state, begin=begin)
    inputs = (state.particle_q, k.poses, k.com, k.jt, k.jw)
    options = {"rigid_com": h.rigid_com, "rigid_rotation": h.rigid_rotation}
    spatial = isinstance(c, _SpatialContactObjective)
    if begin:
        c.begin_step(*inputs, **({"kinematics": k, "q0": q} if spatial else {}), **options)
        return
    if spatial:
        # Select candidates at this Newton point, as the solver does before evaluation.
        c.refresh(q, state.particle_q, rigid_com=h.rigid_com)
    c.evaluate(*inputs, dt=0.001, **options)
    if c.flags.numpy()[0]:
        raise AssertionError(c.flags.numpy())


class TestMaterialContactSampling(unittest.TestCase):
    def test_sampling_underflow(self):
        """Reject positive radii whose cap quadrature loses its reference area."""
        model, _, _, _, cables = scene(cable=True)
        with self.assertRaisesRegex(ValueError, "Unrepresentable material contact sampling"):
            SolverMonolithic(
                model,
                cable_inputs=[replace(cables[0], radius=1e-200)],
                mass_mode="lumped",
                friction_velocity_scale=0.1,
            )

    def test_cloth_two_faces(self):
        """Integrate both physical faces without an articulation or tet actor."""
        model, materials, actors, _, _ = scene()
        samples = ContactSamples(model, actors, radius=0, base_triplet_count=0, _materials=materials)
        self.assertEqual(samples.count, 6)
        np.testing.assert_allclose(samples.weights, 1 / 6)
        np.testing.assert_allclose(samples.offsets, 0.005 + model.shape_margin.numpy()[0])
        self.assertEqual(set(samples.source_shapes), {-2})
        self.assertEqual(sorted(samples.normals[:, 2]), [-1, -1, -1, 1, 1, 1])

    def test_cable_area_and_angle(self):
        """Integrate the cylindrical sides and two end disks while retaining spin."""
        model, materials, actors, _, _ = scene(cable=True, body=True)
        samples = ContactSamples(model, actors, radius=0, base_triplet_count=0, _materials=materials)
        self.assertFalse(materials.cable[0]["gauge"])
        self.assertTrue(all(row >= 0 for row in actors.maps["theta_to_global"]))
        self.assertEqual(samples.count, 512)
        self.assertAlmostEqual(samples.weights.sum(), 2 * np.pi * 0.01 * 2 + 2 * np.pi * 0.01**2)
        self.assertEqual(set(samples.source_shapes), {-3})
        self.assertEqual(samples.material_start, 0)

    def test_multiple_objects_world_filter(self):
        """Keep multiple objects independent and exclude shapes from another world."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        cloth, cables = [], []
        builder.begin_world()
        builder.add_shape_plane(width=0, length=0)
        for _ in range(2):
            first, face = builder.particle_count, builder.tri_count
            for point in ((0, 0, 0.004), (1, 0, 0.004), (0, 1, 0.004)):
                builder.add_particle(wp.vec3(point), wp.vec3(0), 1 / 6)
            builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0)
            cloth.append(
                SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 2), thickness=0.01, surface_density=1)
            )
            ids = tuple(builder.add_particle((i, 0, 0.009), (0, 0, 0), m) for i, m in enumerate((0.5, 1, 0.5)))
            cables.append(
                SolverMonolithic.CableInput(
                    particle_ids=ids,
                    EA=100,
                    bending=((1, 0), (0, 1)),
                    GJ=1,
                    radius=0.01,
                    linear_density=1,
                    rest_theta=(0, 0),
                    rest_director=((0, 1, 0), (0, 1, 0)),
                    fixed_theta=(False, False),
                )
            )
        builder.end_world()
        builder.begin_world()
        builder.add_shape_plane(width=0, length=0)
        builder.end_world()
        model = builder.finalize(device="cpu")
        solver = SolverMonolithic(
            model,
            cloth_inputs=cloth,
            cable_inputs=cables,
            mass_mode="lumped",
            contact_stiffness=100,
            friction_coefficient=0,
            friction_velocity_scale=0.1,
        )
        material_runtime.initialize(solver)
        samples = solver._contact.actor_samples
        self.assertEqual(samples.count, 2 * (6 + 512))
        for source, face, target in zip(samples.source_shapes, samples.faces, samples.targets, strict=True):
            particle = model.tri_indices.numpy()[face, 0] if source == -2 else samples.cable_edges[face, 0]
            self.assertEqual(model.particle_world.numpy()[particle], model.shape_world.numpy()[target])
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        solver.step(a, b, solver.create_control(), dt=0.001)
        self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
        force = solver.get_contact_forces(b).particle_force
        np.testing.assert_allclose(force[:6], force[6:], atol=1e-14)
        self.assertGreater(np.linalg.norm(force), 1e-8)
        query(solver, b, begin=True)
        c, k, h = solver._contact, solver._kinematics, b._monolithic_coupled
        selection = _ContactSpatial(
            c._batch, k, k.gather_q(b), b.particle_q, k.poses, rigid_com=h.rigid_com, capacity=c.count
        )
        selection.update(k.gather_q(b), b.particle_q, rigid_com=h.rigid_com)
        self.assertEqual(selection.check(), 0)
        self.assertEqual(int(selection.count.numpy()[0]), samples.count)

    def test_cloth_winding(self):
        """Preserve two-sided cloth loads under reversed triangle winding."""
        results = []
        for reverse in (False, True):
            model, _, _, cloth, _ = scene()
            if reverse:
                model.tri_indices.assign([[0, 2, 1]])
            solver = SolverMonolithic(
                model,
                cloth_inputs=cloth,
                mass_mode="lumped",
                contact_stiffness=100,
                friction_coefficient=0,
                friction_velocity_scale=0.1,
            )
            material_runtime.initialize(solver)
            state = solver.create_state(dt=0.001)
            query(solver, state, begin=True)
            query(solver, state)
            results.append(solver._contact.pool.force.result.numpy().copy())
        np.testing.assert_allclose(results[0], results[1], atol=1e-15)


class TestMaterialContactDerivatives(unittest.TestCase):
    def test_cylinder_plane_integral(self):
        """Compare side and end-cap contact loads with independent continuum integrals."""

        for ratio in (0.1, 0.5, 0.9):
            for angle in (0.0, 0.13):
                with self.subTest(height_ratio=ratio, theta=angle):
                    solver = contact_solver(cable=True, smoothing=1e-8)
                    state = solver.create_state(dt=0.001)
                    radius, stiffness, length = 0.01, 100.0, 2.0
                    height = radius * ratio
                    x = state.particle_q.numpy().copy()
                    x[:, 2] = height
                    state.particle_q.assign(x)
                    state._monolithic_coupled.cable_theta.fill_(angle)
                    query(solver, state, begin=True)
                    query(solver, state)
                    alpha = asin(ratio)
                    side = stiffness * radius * length * (2 * radius * cos(alpha) - height * (pi - 2 * alpha))
                    caps = (
                        2
                        * stiffness
                        * (
                            -height * (-height * sqrt(radius**2 - height**2) + radius**2 * (pi / 2 - alpha))
                            + 2 / 3 * (radius**2 - height**2) ** 1.5
                        )
                    )
                    actual = solver._contact.pool.force.result.numpy()[:9].reshape(3, 3).sum(axis=0)
                    np.testing.assert_allclose(actual[:2], 0, atol=1e-14)
                    self.assertLess(abs(actual[2] / (side + caps) - 1), 0.01)

    def test_sphere_geometry(self):
        """Compare material surface energy and force with independent spherical distances."""
        for cable in (False, True):
            model, _, _, cloth, cables = scene(cable=cable)
            model.shape_type.assign([int(GeoType.SPHERE)])
            model.shape_scale.assign([[1, 1, 1]])
            model.shape_transform.assign([[1, 0, -1, 0, 0, 0, 1]])
            solver = SolverMonolithic(
                model,
                cloth_inputs=cloth,
                cable_inputs=cables,
                friction_velocity_scale=0.1,
                friction_coefficient=0,
                contact_stiffness=100,
                normal_smoothing=1e-8,
            )
            material_runtime.initialize(solver)
            state = solver.create_state(dt=0.001)
            x = state.particle_q.numpy().copy()
            x[:, 2] = -0.2
            state.particle_q.assign(x)
            query(solver, state, begin=True)
            query(solver, state)
            c, samples = solver._contact, solver._contact.actor_samples
            if cable:
                points = c._batch.material_surface.points.numpy()[samples.faces]
                normals = c._batch.material_surface.directions.numpy()[samples.faces]
            else:
                bary = np.full((samples.count, 3), 1 / 6)
                bary[np.arange(samples.count), samples.slots % 3] = 2 / 3
                points = bary @ x
                normals = samples.normals
            relative = points - np.array([1, 0, -1])
            distance = np.linalg.norm(relative, axis=1)
            gradient = relative / distance[:, None]
            depth = np.maximum(1 + samples.offsets - distance, 0)
            depth[np.einsum("ij,ij->i", normals, gradient) > 0] = 0
            expected_energy = 0.5 * 100 * (samples.weights * depth**2).sum()
            expected_force = (100 * samples.weights * depth) @ gradient
            self.assertGreater(expected_energy, 1e-6)
            self.assertAlmostEqual(c.energy.numpy().sum(), expected_energy, delta=1e-12)
            force = c.pool.force.result.numpy()[:9].reshape(3, 3).sum(axis=0)
            np.testing.assert_allclose(force, expected_force, atol=1e-12)

    def test_surface_jacobian(self):
        """Differentiate off-anchor surface points with independent centered differences."""
        solver = contact_solver(cable=True)
        state = solver.create_state(dt=0.001)
        surface = solver._contact._batch.material_surface
        query(solver, state, begin=True)
        x = state.particle_q.numpy().copy()
        x[1] += (0.02, 0.01, -0.002)
        theta = np.array([0.13, -0.17])
        state.particle_q.assign(x)
        state._monolithic_coupled.cable_theta.assign(theta)
        surface.update(state)
        jacobian = surface.jacobian.numpy().copy()
        edges = surface.edges.numpy()
        epsilon = 1e-5
        for edge in (0, 1):
            selected = np.flatnonzero(edges[:, 2] == edge)
            for coordinate in range(7):
                values = []
                for sign in (-1, 1):
                    candidate_x, candidate_theta = x.copy(), theta.copy()
                    if coordinate < 6:
                        candidate_x[edge + coordinate // 3, coordinate % 3] += sign * epsilon
                    else:
                        candidate_theta[edge] += sign * epsilon
                    state.particle_q.assign(candidate_x)
                    state._monolithic_coupled.cable_theta.assign(candidate_theta)
                    surface.update(state)
                    values.append(surface.points.numpy()[selected])
                np.testing.assert_allclose(
                    (values[1] - values[0]) / (2 * epsilon), jacobian[selected, coordinate], rtol=2e-7, atol=3e-10
                )

    def test_friction_spin_and_wrench(self):
        """Transfer friction into cable spin and balance both physical endpoint wrenches."""
        solver = contact_solver(cable=True, body=True, mu=0.4)
        state = solver.create_state(dt=0.001)
        query(solver, state, begin=True)
        state._monolithic_coupled.cable_theta.assign([0.01, 0.01])
        query(solver, state)
        c = solver._contact
        forces = c.pool.force.result.numpy()
        particle = forces[:9].reshape(3, 3)
        body = forces[9:15]
        torque = forces[15:]
        self.assertLess(torque.sum(), -1e-8)
        np.testing.assert_allclose(particle.sum(axis=0) + body[:3], 0, atol=1e-14)
        angular = np.cross(state.particle_q.numpy(), particle).sum(axis=0)
        angular += np.array([torque.sum(), 0, 0])
        angular += body[3:] + np.cross(state._monolithic_coupled.rigid_com.numpy()[0], body[:3])
        np.testing.assert_allclose(angular, 0, atol=2e-14)
        for pair in c.pair_diagnostics():
            np.testing.assert_allclose(np.asarray(pair["world_wrench"]).sum(axis=0), 0, atol=1e-14)
        rows = np.asarray(solver._actors.maps["theta_to_global"])
        np.testing.assert_allclose(c.residual.numpy()[rows], -torque, atol=1e-15)
        h = 1e-6
        energies = []
        for sign in (-1, 1):
            state._monolithic_coupled.cable_theta.assign(np.full(2, 0.01 + sign * h))
            query(solver, state)
            energies.append(c.energy.numpy().sum())
        self.assertAlmostEqual((energies[1] - energies[0]) / (2 * h), -torque.sum(), delta=2e-10)

    def test_two_end_tangent_and_order(self):
        """Assemble both endpoint cross blocks and preserve the result under actor exchange."""

        original = ActorTable.from_model
        matrices, residuals, forces = [], [], []
        for reverse in (False, True):

            def table(model, *, reverse=reverse, **kwargs):
                return original(model, **kwargs, **({"actor_order": (1, 0)} if reverse else {}))

            with patch.object(ActorTable, "from_model", side_effect=table):
                solver = contact_solver(cable=True, body=True, mu=0.4)
            state = solver.create_state(dt=0.001)
            query(solver, state, begin=True)
            state._monolithic_coupled.cable_theta.fill_(0.01)
            query(solver, state)
            c = solver._contact
            linear = MonolithicLinearWorkspace(
                solver._layout,
                MonolithicLinearCapacities(c.triplet_capacity, 0, 0, 100, 102),
                solver._particle_mapping,
                device=solver.model.device,
                cable_row_order=np.concatenate([block.order for block in solver._cables]),
            )
            generation = MonolithicLinearGeneration(1, 1, c.evaluation_generation, 1)
            assembly = linear.begin_assembly(generation)
            c.assemble_tangent(assembly)
            with self.assertRaisesRegex(ValueError, "only once"):
                c.assemble_tangent(assembly)
            self.assertEqual(int(linear.finalize_assembly(generation=generation).status), 0)
            matrix = linear.densify_for_test(generation=generation).raw_matrix
            np.testing.assert_allclose(matrix, batch_matrix(c._batch), rtol=3e-6, atol=1e-9)
            rigid = solver._rigid.rows_numpy.ravel()
            cable = solver._layout.indices("cable")
            self.assertGreater(np.linalg.norm(matrix[np.ix_(rigid, cable)]), 1e-7)
            theta = np.asarray(solver._actors.maps["theta_to_global"])
            self.assertGreater(np.linalg.norm(matrix[np.ix_(rigid, theta)]), 1e-7)
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
            self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-5)
            order = np.r_[rigid, cable]
            matrices.append(matrix[np.ix_(order, order)])
            residuals.append(c.residual.numpy()[order].copy())
            forces.append(c.pool.force.result.numpy().copy())
            query(solver, state)
            with self.assertRaisesRegex(ValueError, "generation"):
                c.assemble_tangent(assembly)
            with self.assertRaisesRegex(ValueError, "finally assembled"):
                c.publish(state)
        np.testing.assert_allclose(matrices[0], matrices[1], atol=1e-9)
        np.testing.assert_allclose(residuals[0], residuals[1], atol=1e-15)
        np.testing.assert_allclose(forces[0], forces[1], atol=1e-15)

    def test_sparse_and_complete_candidates(self):
        """Match material B3 selection and capacity-independent full residuals."""

        for cable in (False, True):
            with self.subTest(cable=cable):
                with pooled_audit_route():
                    solver = contact_solver(cable=cable, body=True, mu=0.4)
                state = solver.create_state(dt=0.001)
                query(solver, state, begin=True)
                query(solver, state)
                c, k, h = solver._contact, solver._kinematics, state._monolithic_coupled
                energy, residual = c.energy.numpy().copy(), c.residual.numpy().copy()
                selection = _ContactSpatial(
                    c._batch,
                    k,
                    k.gather_q(state),
                    state.particle_q,
                    k.poses,
                    rigid_com=h.rigid_com,
                    capacity=c.count,
                )
                selection.update(k.gather_q(state), state.particle_q, rigid_com=h.rigid_com)
                self.assertEqual(selection.check(), 0)
                args = state.particle_q, k.poses, k.com, k.jt, k.jw
                kwargs = {"dt": 0.001, "rigid_com": h.rigid_com, "rigid_rotation": h.rigid_rotation}
                c._batch.evaluate(*args, selection=selection, **kwargs)
                self.assertEqual(c.flags.numpy()[0], 0)
                np.testing.assert_allclose(c.energy.numpy(), energy, atol=1e-16)
                np.testing.assert_allclose(c.residual.numpy(), residual, atol=1e-16)
                c._batch.initialize_reference()
                c._batch.evaluate_reference(*args, **kwargs)
                np.testing.assert_allclose(c._batch.reference_energy.numpy(), energy, atol=1e-16)
                np.testing.assert_allclose(c._batch.reference_residual.numpy(), residual, atol=1e-16)


class TestMaterialContactStep(unittest.TestCase):
    def test_cloth_plane_step(self):
        """Converge a penetrated standalone cloth and publish the integrated plane force."""
        m, _, _, cloth, cable = scene()
        solver = SolverMonolithic(
            m,
            cloth_inputs=cloth,
            cable_inputs=cable,
            friction_velocity_scale=0.1,
            contact_stiffness=100,
            friction_coefficient=0,
            normal_smoothing=1e-5,
        )
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        solver.step(a, b, solver.create_control(), dt=0.001)
        self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
        force = solver.get_contact_forces(b)
        gap = b.particle_q.numpy()[:, 2].mean() - 0.005 - m.shape_margin.numpy()[0]
        np.testing.assert_allclose(force.particle_force.sum(axis=0), (0, 0, -100 * 0.5 * gap), atol=1e-10)
        self.assertTrue(force.force_valid)

    def test_cable_step(self):
        """Converge coupled cable contact with a free spin mode and a rigid endpoint."""
        solver = contact_solver(cable=True, body=True, mu=0.4)
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        control = solver.create_control()
        for _ in range(3):
            solver.step(a, b, control, dt=0.001)
            self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
            force = solver.get_contact_forces(b)
            self.assertTrue(force.force_valid)
            np.testing.assert_allclose(
                force.particle_force.sum(axis=0) + force.body_wrench[:, :3].sum(axis=0), 0, atol=1e-13
            )
            a, b = b, a

    def test_checkpoint_and_rollback(self):
        """Preserve published contact torques across restart and failed assembly."""

        solver = contact_solver(cable=True, body=True, mu=0.4)
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        control = solver.create_control()
        control._monolithic_cable.torque.fill_(1e-7)
        solver.step(a, b, control, dt=0.001)
        self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.npz"
            solver.save_checkpoint(b, path)
            other = contact_solver(cable=True, body=True, mu=0.4)
            restored, continued = other.create_state(dt=0.001), other.create_state(dt=0.001)
            other.load_checkpoint(restored, path)
            np.testing.assert_array_equal(
                other.get_contact_forces(restored).cable_torque, solver.get_contact_forces(b).cable_torque
            )
            other.step(restored, continued, control, dt=0.001)
            solver.step(b, a, control, dt=0.001)
            self.assertTrue(other._last_stats.converged, other._last_stats.failure_reason)
            self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
            expected, actual = snapshot(a), snapshot(continued)
            for name in expected:
                np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)
        before = snapshot(b)
        with patch.object(solver._contact, "assemble_tangent", side_effect=ValueError("injected assembly failure")):
            with self.assertRaisesRegex(ValueError, "injected"):
                solver.step(a, b, control, dt=0.001)
        for name, value in before.items():
            np.testing.assert_array_equal(snapshot(b)[name], value, err_msg=name)

    def test_capacity_failure(self):
        """Reject incomplete contact admission without publishing partial forces."""

        solver = contact_solver(cable=True, mu=0.4, contact_factor_capacity=0)
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        before = snapshot(b)
        with self.assertWarnsRegex(RuntimeWarning, "capacity exceeded"):
            solver.step(a, b, solver.create_control(), dt=0.001)
        self.assertTrue(solver._last_stats.rolled_back)
        self.assertFalse(solver.get_contact_forces(b).force_valid)
        for name, value in before.items():
            np.testing.assert_array_equal(snapshot(b)[name], value, err_msg=name)

    def test_articulation_contact(self):
        """Converge both material kinds against a mechanical tree with friction."""
        for cable in (False, True):
            with self.subTest(cable=cable):
                solver = contact_solver(cable=cable, body="articulation", mu=0.4)
                a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
                solver.step(a, b, solver.create_control(), dt=0.001)
                self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
                force = solver.get_contact_forces(b)
                np.testing.assert_allclose(force.particle_force.sum(axis=0) + force.body_wrench[0, :3], 0, atol=1e-13)
                self.assertAlmostEqual(force.joint_force[0], force.body_wrench[0, 2], delta=1e-13)

    def test_mixed_and_fixed_contact(self):
        """Publish active A/R/T/cloth/cable contact and reactions with no free material rows."""
        for options in (
            {"mixed": True, "body": "articulation"},
            {"fixed": True, "body": True},
            {"fixed": True, "cable": True, "body": True},
        ):
            with self.subTest(options=options):
                solver = contact_solver(**options)
                a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
                solver.step(a, b, solver.create_control(), dt=0.001)
                self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
                force = solver.get_contact_forces(b)
                self.assertGreater(np.linalg.norm(force.particle_force), 1e-8)
                if options.get("fixed"):
                    np.testing.assert_array_equal(a.particle_q.numpy(), b.particle_q.numpy())

    def test_energy_acceptance_policy(self):
        """Accept material face selection through the nonconservative C-IE energy heuristic."""
        solver = contact_solver()
        a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
        solver.step(a, b, solver.create_control(), dt=0.001)
        self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
        self.assertFalse(solver._conservative_energy())

    def test_spatial_step(self):
        """Run material steps with a bounded spatial candidate capacity."""

        for cable in (False, True):
            with self.subTest(cable=cable):
                solver = contact_solver(cable=cable, body=True, mu=0.4, contact_candidate_capacity=1024)
                a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
                solver.step(a, b, solver.create_control(), dt=0.001)
                self.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)


def test_material_contact_native_devices(test, device):
    """Exercise production contact, material spin and graph reuse on each native device."""
    for cable in (False, True):
        with test.subTest(cable=cable):
            solver = contact_solver(
                cable=cable,
                body=True,
                device=device,
                mu=0.4,
                pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
                use_cuda_graph=device.is_cuda,
            )
            a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
            control = solver.create_control()
            for _ in range(3):
                solver.step(a, b, control, dt=0.001)
                test.assertTrue(solver._last_stats.converged, solver._last_stats.failure_reason)
                test.assertTrue(np.isfinite(solver.get_contact_forces(b).cable_torque).all())
                a, b = b, a
            if device.is_cuda:
                test.assertGreater(sum(solver._contact._batch.graphs.captures.values()), 0)
                test.assertGreater(sum(solver._contact._batch.graphs.replays.values()), 0)


def test_material_contact_precision(test, device):
    """Compare FP32 material surface laws and spin projections with the native FP64 path."""
    for cable in (False, True):
        results = []
        for precision in (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32):
            solver = contact_solver(cable=cable, body=True, device=device, mu=0.4, precision=precision)
            state = solver.create_state(dt=0.001)
            query(solver, state, begin=True)
            x = state.particle_q.numpy().copy()
            x += np.asarray((0, 2**-10, 0), dtype=x.dtype)
            state.particle_q.assign(x)
            state._monolithic_coupled.cable_theta.fill_(0.01)
            query(solver, state)
            results.append((solver._contact.energy.numpy().copy(), solver._contact.residual.numpy().copy()))
        for reference, actual in zip(results[0], results[1], strict=True):
            np.testing.assert_allclose(actual, reference, rtol=5e-4, atol=1e-9)


def test_material_contact_mesh(test, device):
    """Bind actual C2 mesh targets for cloth and cable without closed soft surfaces."""
    if not device.is_cuda:
        with test.assertRaisesRegex(ValueError, "CUDA texture SDF"):
            contact_solver(mesh=True)
        return
    for cable in (False, True):
        solver = contact_solver(cable=cable, body=True, mesh=True, device=device, mu=0.4)
        state = solver.create_state(dt=0.001)
        query(solver, state, begin=True)
        query(solver, state)
        test.assertGreater(solver._contact.energy.numpy().sum(), 0)
        forces = solver._contact.pool.force.result.numpy()
        np.testing.assert_allclose(forces[:9].reshape(3, 3).sum(axis=0) + forces[9:12], 0, atol=1e-12)


def test_material_moving_plane(test, device):
    """Differentiate the moving target and retain both contact endpoints in r and B."""
    for cable in (False, True):
        solver = contact_solver(cable=cable, body="articulation", plane=True, device=device, smoothing=1e-8)
        state = solver.create_state(dt=0.001)
        query(solver, state, begin=True)
        query(solver, state)
        contact = solver._contact
        row = solver._layout.indices("articulation")[0]
        residual = contact.residual.numpy().copy()
        matrix = batch_matrix(contact._batch)
        test.assertGreater(abs(residual[row]), 1e-8)
        test.assertGreater(np.linalg.norm(np.delete(matrix[row], row)), 1e-8)
        energy, gradients = [], []
        epsilon = 1e-7
        for sign in (-1, 1):
            state.joint_q.assign([sign * epsilon])
            query(solver, state)
            energy.append(contact.energy.numpy().sum())
            gradients.append(contact.residual.numpy().copy())
        test.assertAlmostEqual((energy[1] - energy[0]) / (2 * epsilon), residual[row], delta=1e-9)
        np.testing.assert_allclose((gradients[1] - gradients[0]) / (2 * epsilon), matrix[:, row], atol=1e-7)
        for pair in contact.pair_diagnostics():
            np.testing.assert_allclose(np.sum(pair["world_wrench"], axis=0), 0, atol=1e-12)


def test_material_plane_spatial_steps(test, device):
    """Match complete and spatial moving-plane steps with production PCG and graph."""
    for cable in (False, True):
        results = []
        for spatial in (False, True):
            with nullcontext() if spatial else pooled_audit_route():
                solver = contact_solver(
                    cable=cable,
                    body="articulation",
                    plane=True,
                    device=device,
                    pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
                    use_cuda_graph=device.is_cuda,
                    joint_residual_tolerance=1e-10,
                    cloth_position_tolerance=1e-10,
                )
            state = solver.create_state(dt=0.001)
            for _ in range(3):
                solver.step(state, state, solver.create_control(), dt=0.001)
                test.assertTrue(solver.last_stats.converged)
                test.assertLessEqual(solver.last_stats.raw_residual_q_norm, 1e-10)
            results.append(snapshot(state))
        for name in results[0]:
            if name == "history.config_digest":
                # Complete and spatial policies intentionally have different identities.
                continue
            np.testing.assert_allclose(results[0][name], results[1][name], atol=1e-10, rtol=1e-9, err_msg=name)


def test_material_plane_rotation(test, device):
    """Differentiate a free plane's angular work around every world rotation axis."""
    solver = contact_solver(body=True, plane=True, device=device, smoothing=1e-8)
    state = solver.create_state(dt=0.001)
    query(solver, state, begin=True)
    query(solver, state)
    rows = solver._rigid.rows_numpy[0]
    torque = solver._contact.residual.numpy()[rows[3:]].copy()
    epsilon = 1e-7
    for axis in range(3):
        energy = []
        for sign in (-1, 1):
            quaternion = np.array([[0.0, 0.0, 0.0, np.cos(epsilon / 2)]])
            quaternion[0, axis] = sign * np.sin(epsilon / 2)
            state._monolithic_coupled.rigid_rotation.assign(quaternion)
            query(solver, state)
            energy.append(solver._contact.energy.numpy().sum())
        test.assertAlmostEqual((energy[1] - energy[0]) / (2 * epsilon), torque[axis], delta=1e-9)


class TestMaterialContactDevices(unittest.TestCase):
    pass


add_function_test(
    TestMaterialContactDevices, "test_native_devices", test_material_contact_native_devices, devices=get_test_devices()
)

add_function_test(
    TestMaterialContactDevices, "test_native_precision", test_material_contact_precision, devices=get_test_devices()
)

add_function_test(TestMaterialContactDevices, "test_mesh", test_material_contact_mesh, devices=get_test_devices())
add_function_test(
    TestMaterialContactDevices, "test_moving_plane", test_material_moving_plane, devices=get_test_devices()
)
add_function_test(
    TestMaterialContactDevices,
    "test_plane_spatial_steps",
    test_material_plane_spatial_steps,
    devices=get_test_devices(),
)


add_function_test(
    TestMaterialContactDevices, "test_plane_rotation", test_material_plane_rotation, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
