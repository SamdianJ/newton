# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material ownership and transactional continuation through the solver API."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.linear_types import MonolithicLinearLayout
from newton._src.solvers.monolithic.pcg_common import group_data, group_policy
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.monolithic_test_utils import add_contact_free_tet


def scene(
    *,
    cloth=True,
    cables=2,
    fixed_theta=(False, True),
    bending=((2, 0.2), (0.2, 1)),
    mechanical=False,
    rigid=False,
    tet=False,
    precision=SolverMonolithic.Precision.FP64,
    fixed_particles=(),
    shape_radius=None,
    device="cpu",
    pcg_mode=SolverMonolithic.PCGMode.DIAGNOSTIC,
    use_cuda_graph=False,
    globalization="ResNorm",
    contact_model="C-IE",
    newton_max_iterations=10,
    linear_tolerance=1e-4,
    publish_policy="strict",
    mass_mode="lumped",
    cable_position_tolerance=1e-9,
    cable_theta_tolerance=1e-11,
    fp32_absolute_tolerance=None,
):
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    if mechanical:
        body = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.1)))
        builder.add_articulation([builder.add_joint_prismatic(-1, body, limit_ke=0, limit_kd=0)])
    if rigid:
        builder.add_body(mass=1, inertia=wp.diag(wp.vec3(0.1)))
    if shape_radius is not None:
        builder.add_shape_sphere(-1, radius=shape_radius)
    shells, rods = [], []
    for sheet in range(int(cloth)):
        first = builder.particle_count
        face = builder.tri_count
        for position, mass in zip(((0, 0, 0), (1, 0, 0), (0, 1, 0)), (0, 1 / 6, 1 / 6), strict=True):
            builder.add_particle(wp.vec3(position) + wp.vec3(0, 2 * sheet, 0), wp.vec3(0), mass)
        builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
        shells.append(
            SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 2), thickness=0.01, surface_density=1)
        )
    for index in range(cables):
        ids = tuple(
            builder.add_particle((2 + vertex, 2 * index, 0), (0, 0, 0), mass) for vertex, mass in enumerate((0, 1, 0.5))
        )
        rods.append(
            SolverMonolithic.CableInput(
                particle_ids=ids,
                EA=10,
                bending=bending,
                GJ=1,
                radius=0.01,
                linear_density=1,
                rest_theta=(7.0, 7.2),
                rest_director=((0, 1, 0), (0, 1, 0)),
                fixed_theta=fixed_theta,
            )
        )
    if tet:
        add_contact_free_tet(builder)
    for particle in fixed_particles:
        builder.particle_mass[particle] = 0
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        cloth_inputs=shells,
        cable_inputs=rods,
        contact_model=contact_model,
        globalization=globalization,
        newton_max_iterations=newton_max_iterations,
        linear_tolerance=linear_tolerance,
        publish_policy=publish_policy,
        friction_velocity_scale=0.1,
        mass_mode=mass_mode,
        cable_position_tolerance=cable_position_tolerance,
        cable_theta_tolerance=cable_theta_tolerance,
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=device)
        if mass_mode == "consistent"
        else None,
        precision=precision,
        fp32_absolute_tolerance=fp32_absolute_tolerance,
        pcg_mode=pcg_mode,
        use_cuda_graph=use_cuda_graph,
    )
    return model, solver


def snapshot(state):
    values = {name: value.numpy().copy() for name, value in vars(state).items() if isinstance(value, wp.array)}
    values.update(
        {
            "history." + name: value.numpy().copy()
            for name, value in vars(state._monolithic_coupled).items()
            if isinstance(value, wp.array)
        }
    )
    return values


class TestMonolithicMaterialState(unittest.TestCase):
    def test_independent_multi_object_layout(self):
        """Map each material coordinate exactly once without dummy bodies or tets."""
        model, solver = scene()
        table, layout = solver._actors, solver._layout
        self.assertEqual(model.body_count, 0)
        self.assertEqual(model.tet_count, 0)
        self.assertEqual([a.kind for a in table.actors], ["cloth", "cable", "cable"])
        self.assertEqual([a.dof_count for a in table.actors], [6, 7, 7])
        self.assertEqual(layout.scalar_dof_count, 20)
        self.assertEqual(table.maps["theta_to_global"], (12, -1, 19, -1))
        self.assertEqual(
            layout.groups,
            (
                ("global", 0, 20),
                ("cloth_0_x", 0, 6),
                ("cable_0_x", 6, 6),
                ("cable_0_theta", 12, 1),
                ("cable_1_x", 13, 6),
                ("cable_1_theta", 19, 1),
            ),
        )
        self.assertEqual(len({name for name, _, _ in layout.groups}), len(layout.groups))
        np.testing.assert_array_equal(layout.indices("cable", 1), np.arange(13, 20))
        with self.assertRaises(ValueError):
            layout.indices("cable")
        with self.assertRaises(ValueError):
            layout.indices("unknown")
        reordered = ActorTable.from_model(model, materials=solver._materials, actor_order=(2, 0, 1))
        other = MonolithicLinearLayout.from_actors(reordered)
        self.assertEqual(other.scalar_dof_count, layout.scalar_dof_count)
        self.assertEqual(reordered.maps["theta_to_global"], (19, -1, 6, -1))
        self.assertNotEqual(table.identity, reordered.identity)

    def test_single_and_empty_material_kinds(self):
        """Initialize standalone shells and cables with empty unrelated Model arrays."""
        for cloth, cables in ((True, 0), (False, 1)):
            with self.subTest(cloth=cloth, cables=cables):
                _, solver = scene(cloth=cloth, cables=cables)
                state = solver.create_state(dt=0.001)
                solver._states.validate(state)
                self.assertEqual(state.joint_q.size, 0)
                self.assertEqual(state.body_q.size, 0)
                self.assertEqual(state._monolithic_coupled.cable_theta.size, 2 * cables)

    def test_per_object_residual_policy(self):
        """Require independent xyz/theta floors and limits for every object."""
        _, solver = scene()
        layout = solver._layout
        membership, aliases = group_data(layout)
        np.testing.assert_array_equal(aliases, [0, -1, -1])
        for index, (_, start, count) in enumerate(layout.groups[1:], 1):
            np.testing.assert_array_equal(membership[start : start + count], index)
        config = SimpleNamespace(residual_floor_global=1, residual_floor_q=2, residual_floor_x=3)
        with self.assertRaises(ValueError):
            group_policy(layout, config)
        config.group_residual_floors = (1, 2, 3, 4, 5, 6)
        config.group_norm_limits = (1e-3, 1e-4, 1e-4, 1e-7, 1e-4, 1e-8)
        floors, limits = group_policy(layout, config)
        self.assertEqual(floors, list(config.group_residual_floors))
        self.assertEqual(limits, list(config.group_norm_limits))
        config.group_norm_limits = (1e-3,)
        with self.assertRaises(ValueError):
            group_policy(layout, config)

    def test_checkpoint_preserves_unwrapped_theta_and_frame(self):
        """Round-trip unwrapped material angles and a rotated reference frame."""
        _, solver = scene()
        state = solver.create_state(dt=0.001)
        history = state._monolithic_coupled
        theta = history.cable_theta.numpy()
        theta[0] += 4 * np.pi
        history.cable_theta.assign(theta)
        directors = history.cable_ref_director.numpy()
        directors[:, 1:] = [np.cos(0.4), np.sin(0.4)]
        history.cable_ref_director.assign(directors)
        twist = history.cable_ref_twist.numpy() + 2 * np.pi
        history.cable_ref_twist.assign(twist)
        solver.reset_state(state, dt=0.002)
        expected = snapshot(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            solver.save_checkpoint(state, path)
            target = solver.create_state(dt=0.001)
            solver.load_checkpoint(target, path)
        for name, value in snapshot(target).items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)
        np.testing.assert_allclose(solver._materials.cable[0]["rest_twist"], [0.2])

    def test_checkpoint_rejection_is_atomic(self):
        """Reject corrupt continuation fields without changing any target array."""
        _, solver = scene()
        state = solver.create_state(dt=0.001)
        history = state._monolithic_coupled
        history.force_generation.fill_(3)
        history.force_valid.fill_(1)
        history.contact_particle_force.fill_(wp.vec3d(2, 3, 4))
        expected = snapshot(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            for field, change in (
                ("history.cable_theta", lambda a: a.astype(np.float32)),
                ("history.cable_ref_director", lambda a: a * 2),
                ("history.cable_ref_twist", lambda a: a + 0.1),
                ("history.config_digest", lambda a: a ^ 1),
                ("history.layout_digest", lambda a: a ^ 1),
                ("history.schema_version", lambda a: np.array([4], dtype=a.dtype)),
            ):
                data = dict(expected)
                data[field] = change(data[field])
                np.savez(path, **data)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    solver.load_checkpoint(state, path)
                for name, value in snapshot(state).items():
                    np.testing.assert_array_equal(value, expected[name], err_msg=name)
            data = dict(expected)
            del data["history.cable_ref_twist"]
            np.savez(path, **data)
            with self.assertRaises(ValueError):
                solver.load_checkpoint(state, path)

    def test_reset_and_unavailable_step_preserve_state(self):
        """Preserve published history and forces after failed reset or unsupported step."""
        _, solver = scene()
        state = solver.create_state(dt=0.001)
        history = state._monolithic_coupled
        history.force_generation.fill_(2)
        history.force_valid.fill_(1)
        expected = snapshot(state)
        with self.assertRaises(ValueError):
            solver.reset_state(state, dt=-1)
        with self.assertRaises(NotImplementedError):
            solver.step(state, state, solver.create_control(), dt=0.001)
        for name, value in snapshot(state).items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)
        with self.assertRaises(ValueError):
            solver.create_state(dt=0.001, source=solver.model.state())

    def test_owned_controls(self):
        """Snapshot independent controls and reject torques on clamped material angles."""
        _, solver = scene()
        control, other = solver.create_control(), solver.create_control()
        torque = control._monolithic_cable.torque.numpy()
        torque[0] = 0.2
        control._monolithic_cable.torque.assign(torque)
        frozen = solver._snapshot_material_control(control)
        control._monolithic_cable.torque.zero_()
        self.assertEqual(frozen["torque"][0], 0.2)
        np.testing.assert_array_equal(other._monolithic_cable.torque.numpy(), 0)
        torque[1] = 0.1
        control._monolithic_cable.torque.assign(torque)
        with self.assertRaises(ValueError):
            solver._snapshot_material_control(control)

    def test_anisotropic_theta_not_permanently_pinned(self):
        """Retain both physical angles on an anisotropic straight chain."""
        _, solver = scene(cloth=False, cables=1, fixed_theta=(False, False))
        self.assertEqual(solver._actors.actors[0].dof_count, 8)
        self.assertEqual(solver._actors.maps["theta_to_global"], (6, 7))

    def test_mixed_mechanics_and_materials(self):
        """Initialize and restore A/R/T plus multiple material actors in one layout."""
        for mechanical, rigid, tet in ((True, True, True), (False, True, False), (False, False, True)):
            with self.subTest(mechanical=mechanical, rigid=rigid, tet=tet):
                model, solver = scene(mechanical=mechanical, rigid=rigid, tet=tet)
                state = solver.create_state(dt=0.001)
                solver._states.validate(state)
                table = solver._actors
                rows = list(table.maps["joint_dof_to_global"])
                for start in table.maps["particle_to_global"]:
                    if start >= 0:
                        rows.extend(range(start, start + 3))
                rows.extend(row for row in table.maps["theta_to_global"] if row >= 0)
                self.assertEqual(sorted(rows), list(range(table.scalar_dof_count)))
                self.assertEqual(table.scalar_dof_count, int(mechanical) + 6 * rigid + 12 * tet + 20)
                self.assertEqual(state.particle_q.size, model.particle_count)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "mixed.npz"
                    solver.save_checkpoint(state, path)
                    target = solver.create_state(dt=0.002)
                    solver.load_checkpoint(target, path)
                for name, value in snapshot(state).items():
                    np.testing.assert_array_equal(snapshot(target)[name], value, err_msg=name)

    def test_precision_and_physics_identity(self):
        """Restore FP32 history and reject a changed tensor or precision atomically."""
        _, solver = scene(precision=SolverMonolithic.Precision.FP32)
        source = solver.create_state(dt=0.001)
        self.assertEqual(source._monolithic_coupled.cable_theta.dtype, wp.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fp32.npz"
            solver.save_checkpoint(source, path)
            for precision, bending in (
                (SolverMonolithic.Precision.FP32, ((2, 0.2), (0.2, 1))),
                (SolverMonolithic.Precision.FP64, ((2, 0.2), (0.2, 1))),
                (SolverMonolithic.Precision.FP32, ((3, 0.2), (0.2, 1))),
            ):
                _, other = scene(precision=precision, bending=bending)
                target = other.create_state(dt=0.002)
                if precision == solver.precision and bending[0][0] == 2:
                    other.load_checkpoint(target, path)
                    expected = snapshot(source)
                else:
                    expected = snapshot(target)
                    with self.assertRaises(ValueError):
                        other.load_checkpoint(target, path)
                    with self.assertRaises(ValueError):
                        other.create_state(dt=0.001, source=source)
                for name, value in snapshot(target).items():
                    np.testing.assert_array_equal(value, expected[name], err_msg=name)

    def test_invalid_geometry_reset_is_atomic(self):
        """Reject collapsed shells and incompatible directors while preserving old forces."""
        _, solver = scene()
        for field in ("particle_q", "cable_ref_director"):
            state = solver.create_state(dt=0.001)
            history = state._monolithic_coupled
            history.force_valid.fill_(1)
            history.force_generation.fill_(4)
            old_forces = solver.get_contact_forces(state)
            if field == "particle_q":
                values = state.particle_q.numpy()
                values[2] = values[1]
                state.particle_q.assign(values)
            else:
                history.cable_ref_director.fill_(wp.vec3d(1, 0, 0))
            expected = snapshot(state)
            with self.assertRaises(ValueError):
                solver.reset_state(state, dt=0.001)
            for name, value in snapshot(state).items():
                np.testing.assert_array_equal(value, expected[name], err_msg=name)
            self.assertEqual(old_forces.force_generation, 4)

    def test_zero_dof_material_groups(self):
        """Skip fixed objects sharing a start offset without duplicating the next group's rows."""
        _, solver = scene(fixed_theta=(True, True), fixed_particles=(3, 4, 5))
        self.assertEqual(solver._layout.groups, (("global", 0, 12), ("cloth_0_x", 0, 6), ("cable_1_x", 6, 6)))
        state = solver.create_state(dt=0.001)
        solver._states.validate(state)
        with self.assertRaisesRegex(ValueError, "at least one dynamic"):
            scene(fixed_theta=(True, True), fixed_particles=range(9))

    def test_multiple_cloth_plain_initialization(self):
        """Initialize independent cloth objects from ordinary state without inventing theta history."""
        model, solver = scene(cloth=2, cables=0)
        self.assertEqual(solver._layout.groups, (("global", 0, 12), ("cloth_0_x", 0, 6), ("cloth_1_x", 6, 6)))
        state = solver.create_state(dt=0.001, source=model.state())
        solver._states.validate(state)
        self.assertEqual(state._monolithic_coupled.cable_theta.size, 0)

    def test_shape_geometry_checkpoint_identity(self):
        """Reject a checkpoint bound to a different static collision envelope."""
        _, first = scene(shape_radius=0.1)
        _, other = scene(shape_radius=0.2)
        source, target = first.create_state(dt=0.001), other.create_state(dt=0.001)
        expected = snapshot(target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shape.npz"
            first.save_checkpoint(source, path)
            with self.assertRaises(ValueError):
                other.load_checkpoint(target, path)
        for name, value in snapshot(target).items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
