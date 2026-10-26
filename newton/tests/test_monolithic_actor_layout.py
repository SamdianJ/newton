# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify actor ownership independently of Model creation order."""

import dataclasses
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import (
    ActorTable,
    TetBoundaryError,
    _checked_dof_count,
    _validate_and_collect_boundary_faces,
)
from newton._src.solvers.monolithic.articulation import _TreeKinematics
from newton._src.solvers.monolithic.joint_parameters import JointParameters
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.unittest_utils import add_function_test, get_test_devices


def mixed_model(device, rigid_count=2, *, before=False):
    """Build a noncontiguous two-link tree and independently owned FREE bodies."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    inertia = wp.mat33(0.002, 0, 0, 0, 0.003, 0, 0, 0, 0.004)

    def rigid():
        body = builder.add_body(mass=0.5, inertia=inertia, com=(0.01, 0.02, 0))
        builder.add_shape_box(body, hx=0.01, hy=0.02, hz=0.01, cfg=builder.ShapeConfig(density=0))
        return body

    rigid_ids = [rigid() for _ in range(rigid_count)] if before else []
    root = builder.add_link(mass=1, inertia=inertia)
    if not before:
        rigid_ids = [rigid() for _ in range(rigid_count)]
    child = builder.add_link(mass=1, inertia=inertia)
    options = {"limit_ke": 0, "limit_kd": 0}
    joints = [
        builder.add_joint_revolute(-1, root, axis=newton.Axis.Z, **options),
        builder.add_joint_prismatic(root, child, axis=newton.Axis.X, **options),
    ]
    builder.add_articulation(joints)
    add_contact_free_tet(builder)
    builder.particle_mass[0] = 0
    builder.add_ground_plane()
    return builder.finalize(device=device), (root, child), tuple(rigid_ids), tuple(joints)


def test_model_maps(test, device):
    """Round-trip owners and distinguish FREE coordinate, velocity and target offsets."""
    for count in (0, 1, 2, 5):
        for before in (False, True):
            model, bodies, rigids, joints = mixed_model(device, count, before=before)
            table = ActorTable.from_model(model)
            arm, tet = table.actors[0], table.actors[-1]
            test.assertEqual(arm.kind, "articulation")
            test.assertEqual(arm.body_indices, bodies)
            test.assertEqual(arm.joint_indices, joints)
            test.assertEqual(arm.coord_indices, tuple(range(7 * count, 7 * count + 2)))
            test.assertEqual(arm.dof_indices, tuple(range(6 * count, 6 * count + 2)))
            test.assertEqual(arm.target_indices, tuple(model.joint_target_q_start.numpy()[list(joints)]))
            test.assertEqual(tet.dof_count, 9)
            test.assertEqual(table.scalar_dof_count, 2 + 6 * count + 9)
            test.assertEqual(table.maps["particle_to_global"][0], -1)
            for actor_id, actor in enumerate(table.actors):
                for stem in ("body", "joint", "coord", "dof", "target", "particle", "shape"):
                    for local, index in enumerate(getattr(actor, stem + "_indices")):
                        test.assertEqual(table.maps[stem + "_owner"][index], actor_id)
                        test.assertEqual(table.maps[stem + "_local"][index], local)
            for actor, body in zip(table.actors[1:-1], rigids, strict=True):
                test.assertEqual(actor.kind, "rigid")
                test.assertEqual(actor.body_indices, (body,))
                test.assertEqual(len(actor.coord_indices), 7)
                test.assertEqual(actor.dof_count, 6)
                test.assertEqual(actor.state_count, 7)
            with test.assertRaises(dataclasses.FrozenInstanceError):
                arm.dof_start = 1
            with test.assertRaises(TypeError):
                table.maps["body_owner"] = ()
            np.testing.assert_array_equal(table.device_maps["body_owner"].numpy(), table.maps["body_owner"])
            test.assertEqual(table.identity, ActorTable.from_model(model).identity)
            test.assertEqual(table.static_shape_indices, (model.shape_count - 1,))
            test.assertEqual(table.maps["shape_owner"][-1], -1)


def test_layout_permutation(test, device):
    """Recover the same independently scattered operator after permuting actor blocks."""
    model, _, _, _ = mixed_model(device)
    table = ActorTable.from_model(model)
    other = ActorTable.from_model(model, actor_order=(3, 1, 0, 2))
    test.assertNotEqual(table.identity, other.identity)
    values, matrices, indices = [], [], []
    for current in (table, other):
        r = np.zeros(current.scalar_dof_count)
        b = np.zeros((len(r), len(r)))
        order = []
        for actor in current.actors:
            key = actor.source_index + (100 if actor.kind == "tet" else 10 if actor.kind == "rigid" else 1)
            local = np.arange(actor.dof_count) + key
            part = slice(actor.dof_start, actor.dof_start + actor.dof_count)
            r[part] = local
            b[part, part] = np.diag(local) + np.outer(local, local)
            order.extend((actor.kind, actor.source_index, i) for i in range(actor.dof_count))
        values.append(r)
        matrices.append(b)
        indices.append(order)
    permutation = [indices[1].index(key) for key in indices[0]]
    np.testing.assert_array_equal(values[0], values[1][permutation])
    np.testing.assert_array_equal(matrices[0], matrices[1][np.ix_(permutation, permutation)])
    np.testing.assert_allclose(
        np.linalg.solve(matrices[0], values[0]), np.linalg.solve(matrices[1], values[1])[permutation]
    )


def test_local_articulation(test, device):
    """Gather mechanical targets and evaluate local FK without consuming any FREE segment."""
    model, bodies, _, _ = mixed_model(device, before=True)
    table = ActorTable.from_model(model)
    arm = table.actors[0]
    ids = list(arm.dof_indices)
    for name, value in (("joint_target_ke", 4), ("joint_target_kd", 0.3)):
        data = getattr(model, name).numpy()
        data[ids] = value
        getattr(model, name).assign(data)
    mode = model.joint_target_mode.numpy()
    mode[ids] = int(newton.JointTargetMode.POSITION_VELOCITY)
    model.joint_target_mode.assign(mode)
    control = model.control()
    targets = control.joint_target_q.numpy()
    targets[list(arm.target_indices)] = (0.2, 0.015)
    control.joint_target_q.assign(targets)
    rates = control.joint_target_qd.numpy()
    rates[ids] = (0.1, -0.02)
    control.joint_target_qd.assign(rates)
    control.joint_f.fill_(3)
    efforts = control.joint_f.numpy()
    efforts[ids] = 0
    control.joint_f.assign(efforts)
    terms = JointParameters(model, actor=arm, implicit_pd=True)
    terms.snapshot_targets(control)
    np.testing.assert_array_equal(terms.target_q.numpy(), np.array([0.2, 0.015], dtype=np.float32))
    np.testing.assert_array_equal(terms.target_qd.numpy(), np.array([0.1, -0.02], dtype=np.float32))
    k = _TreeKinematics(model, actor=arm)
    state = model.state()
    q = state.joint_q.numpy()
    q[list(arm.coord_indices)] = terms.target_q.numpy()
    state.joint_q.assign(q)
    qd = state.joint_qd.numpy()
    qd[ids] = (0.1, -0.02)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    untouched = state.body_q.numpy().copy()
    velocities = state.body_qd.numpy().copy()
    k.evaluate(wp.array(terms.target_q.numpy(), dtype=wp.float64, device=device), state=state)
    np.testing.assert_allclose(k.poses.numpy(), untouched[list(bodies)], atol=2e-7)
    np.testing.assert_allclose(state.body_q.numpy(), untouched, atol=2e-7)
    np.testing.assert_allclose(state.body_qd.numpy(), velocities, atol=2e-7)


def test_input_rejection(test, device):
    """Reject duplicate owners, invalid trees, unsupported joints, worlds and coordinate layouts."""
    for field, edit in (
        ("joint_child", lambda x: x.__setitem__(-1, x[-2])),
        ("joint_parent", lambda x: x.__setitem__(-1, -1)),
        ("joint_parent", lambda x: x.__setitem__(0, 1)),
        ("joint_type", lambda x: x.__setitem__(0, int(newton.JointType.BALL))),
        ("body_world", lambda x: x.__setitem__(0, 1)),
        ("joint_world", lambda x: x.__setitem__(0, 1)),
        ("body_mass", lambda x: x.__setitem__(1, 0)),
        ("joint_q_start", lambda x: x.__setitem__(1, 6)),
        ("joint_qd_start", lambda x: x.__setitem__(1, 7)),
        ("articulation_end", lambda x: x.__setitem__(0, 2)),
        ("joint_enabled", lambda x: x.__setitem__(0, False)),
    ):
        model = mixed_model(device)[0]
        array = getattr(model, field)
        data = array.numpy()
        edit(data)
        array.assign(data)
        with test.assertRaises(ValueError, msg=field):
            ActorTable.from_model(model)


class TestActorLayout(unittest.TestCase):
    def test_legacy_target_layout(self):
        """Keep legacy target indices separate from FREE pose coordinate indices."""
        with patch.object(newton, "use_coord_layout_targets", False):
            model = mixed_model("cpu", before=True)[0]
        arm = ActorTable.from_model(model).actors[0]
        self.assertEqual(arm.coord_indices, (14, 15))
        self.assertEqual(arm.dof_indices, (12, 13))
        self.assertEqual(arm.target_indices, (12, 13))
        terms = JointParameters(model, actor=arm)
        np.testing.assert_array_equal(terms.target_map.numpy(), arm.target_indices)

    def test_capacity(self):
        """Reject int32 overflow before allocating layout storage."""
        self.assertEqual(_checked_dof_count((2, 6, 9)), 17)
        for counts in ((2**31,), (2**31 - 1, 1), (-1,), (True,)):
            with self.assertRaises(ValueError):
                _checked_dof_count(counts)

    def test_floating_tree(self):
        """Reject a FREE root with descendant links until coupled floating-base dynamics exist."""
        builder = newton.ModelBuilder()
        inertia = wp.diag(wp.vec3(0.01))
        first = builder.add_link(mass=1, inertia=inertia)
        second = builder.add_link(mass=1, inertia=inertia)
        builder.add_articulation([builder.add_joint_free(first), builder.add_joint_revolute(first, second)])
        add_contact_free_tet(builder)
        with self.assertRaises(ValueError):
            ActorTable.from_model(builder.finalize(device="cpu"))


for _test in (test_model_maps, test_layout_permutation, test_local_articulation, test_input_rejection):
    add_function_test(TestActorLayout, _test.__name__, _test, devices=get_test_devices())


def single_tet_model(points=None, tets=None):
    """Build one surface-meshed tet, optionally replacing its points or tets."""
    builder = newton.ModelBuilder()
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
    model = builder.finalize(device="cpu")
    if points is not None:
        model.particle_q = wp.array(points, dtype=wp.vec3, device="cpu")
    if tets is not None:
        model.tet_indices = wp.array(tets, dtype=int, device="cpu")
    return model


class TestTetBoundary(unittest.TestCase):
    def test_boundary_faces_and_particles(self):
        faces, particles = _validate_and_collect_boundary_faces(single_tet_model())
        self.assertEqual((len(faces), particles.tolist()), (4, [0, 1, 2, 3]))

    def test_invalid_boundaries_are_rejected(self):
        """Reject missing, extra, duplicate, degenerate and non-manifold tet boundaries."""
        for mode in ("missing", "duplicate", "repeated", "extra", "zero_area"):
            with self.subTest(mode=mode):
                model = single_tet_model()
                faces = model.tri_indices.numpy()
                if mode == "missing":
                    faces = faces[:-1]
                elif mode == "duplicate":
                    faces[0] = faces[1]
                elif mode == "repeated":
                    faces[0, 0] = faces[0, 1]
                elif mode == "extra":
                    faces = np.concatenate((faces, faces[:1]))
                else:
                    points = model.particle_q.numpy()
                    points[1] = points[0]
                    model.particle_q.assign(points)
                model.tri_indices = wp.array(faces, dtype=int, device="cpu")
                with self.assertRaisesRegex(TetBoundaryError, "INVALID_BOUNDARY"):
                    _validate_and_collect_boundary_faces(model)
        points = np.vstack((single_tet_model().particle_q.numpy(), [0, 0, -0.02], [0.01, 0.01, -0.03]))
        model = single_tet_model(points, [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]])
        with self.assertRaisesRegex(TetBoundaryError, "INVALID_BOUNDARY.*nonmanifold_tet_face"):
            _validate_and_collect_boundary_faces(model)


if __name__ == "__main__":
    unittest.main()
