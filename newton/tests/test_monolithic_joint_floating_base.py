# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Floating-base kinematics: root Jacobian columns, native pose increments, FREE mirrors and rejections."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contract_oracle import rotation

ANCHOR = wp.transform((0.3, -0.2, 1.0), wp.quat_rpy(0.4, -0.3, 0.7))
PRECISIONS = (("fp64", np.float64), ("fp32", np.float32))


def floating_chain(device, *, fixed_child=False, gravity=(0.0, 0.0, -9.81), edit=None):
    """A FREE root (offset COM, non-identity anchors) carrying a revolute link and a prismatic grandchild."""
    builder = newton.ModelBuilder(gravity=gravity)
    root = builder.add_link(
        mass=1.0, inertia=wp.diag(wp.vec3(2e-3, 3e-3, 4e-3)), com=wp.vec3(0.01, -0.02, 0.005), label="root"
    )
    child = builder.add_link(
        mass=0.5, inertia=wp.diag(wp.vec3(1e-3, 2e-3, 1.5e-3)), com=wp.vec3(0.01, 0.0, 0.0), label="child"
    )
    tip = builder.add_link(mass=0.2, inertia=wp.diag(wp.vec3(4e-4)), com=wp.vec3(0.0, 0.01, 0.0), label="tip")
    joints = [
        builder.add_joint_free(
            root,
            parent_xform=ANCHOR,
            child_xform=wp.transform((0.02, 0.01, -0.03), wp.quat_rpy(0.1, 0.0, -0.2)),
            label="free",
        )
    ]
    child_frames = {
        "parent_xform": wp.transform((0.1, 0.0, 0.0), wp.quat_rpy(0.0, 0.3, 0.0)),
        "child_xform": wp.transform((-0.05, 0.0, 0.0), wp.quat_identity()),
    }
    if fixed_child:
        joints.append(builder.add_joint_fixed(root, child, **child_frames, label="weld"))
        joints.append(
            builder.add_joint_fixed(
                child, tip, parent_xform=wp.transform((0.05, 0.02, 0.0), wp.quat_identity()), label="weld_tip"
            )
        )
    else:
        joints.append(
            builder.add_joint_revolute(
                root, child, **child_frames, axis=(0.0, 1.0, 0.0), limit_ke=0.0, limit_kd=0.0, label="hinge"
            )
        )
        joints.append(
            builder.add_joint_prismatic(
                child,
                tip,
                parent_xform=wp.transform((0.05, 0.02, 0.0), wp.quat_rpy(0.2, 0.0, 0.0)),
                axis=(1.0, 0.0, 0.0),
                limit_ke=0.0,
                limit_kd=0.0,
                label="slider",
            )
        )
    builder.add_articulation(joints)
    if edit is not None:
        edit(builder)
    model = builder.finalize(device=device)
    q, qd = model.joint_q.numpy(), model.joint_qd.numpy()
    q[:3] = (0.05, -0.02, 0.1)
    r = np.array([0.1, -0.2, 0.3, 0.9])
    q[3:7] = r / np.linalg.norm(r)
    qd[:6] = (0.1, 0.0, 0.2, 0.5, -1.0, 2.0)
    if not fixed_child:
        q[7:9] = (0.4, 0.03)
        qd[6:8] = (3.0, -0.2)
    model.joint_q.assign(q)
    model.joint_qd.assign(qd)
    return model


def make_solver(model, precision="fp64", **options):
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    return SolverMonolithic(model, friction_velocity_scale=0.01, precision=precision, **options)


def _quat(matrix):
    """Unit xyzw quaternion of a rotation matrix (NumPy)."""
    m = matrix
    w = np.sqrt(max(1.0 + np.trace(m), 1e-30)) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])


def _axial(a):
    return np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]]) / 2


def test_root_jacobian_matches_finite_differences(test, device):
    """Root columns translate along world e_k and rotate about world e_k through the root COM."""
    model = floating_chain(device)
    run = make_solver(model)
    state = run.create_state(dt=1e-3)
    k = run._kinematics
    q = wp.clone(k.gather_q(state))
    com0, rot0 = (a.numpy().astype(np.float64) for a in k.state_roots(state))
    roots = (wp.array(com0, dtype=k.vector, device=device), wp.array(rot0, dtype=k.quaternion, device=device))
    k.evaluate(q, roots=roots)
    jt, jw = k.jt.numpy().copy(), k.jw.numpy().copy()
    test.assertEqual(jt.shape[:2], (k.bodies, k.count))
    q_host = q.numpy().copy()

    def evaluate(dof, eps):
        com, rot, coordinates = com0.copy(), rot0.copy(), q_host.copy()
        if dof < 3:
            com[0, dof] += eps
        elif dof < 6:
            rot[0] = _quat(rotation(eps * np.eye(3)[dof - 3]) @ rotation_matrix(rot0[0]))
        else:
            coordinates[dof] += eps
        q.assign(coordinates)
        roots[0].assign(com)
        roots[1].assign(rot)
        k.evaluate(q, roots=roots)
        return k.com.numpy().copy(), k.rotations.numpy().copy()

    eps = 1e-6
    for dof in range(k.count):
        (cp, rp), (cm, rm) = evaluate(dof, eps), evaluate(dof, -eps)
        np.testing.assert_allclose((cp - cm) / (2 * eps), jt[:, dof], atol=1e-8, err_msg=f"translation column {dof}")
        angular = np.array([_axial(a @ b.T) for a, b in zip(rp, rm, strict=True)]) / (2 * eps)
        np.testing.assert_allclose(angular, jw[:, dof], atol=1e-8, err_msg=f"angular column {dof}")
    # Root rows: unit world columns.
    np.testing.assert_array_equal(jt[0, :3], np.eye(3))
    np.testing.assert_array_equal(jw[:, 3:6], np.tile(np.eye(3), (k.bodies, 1, 1)))


def rotation_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_pose_increment_round_trip(test, device):
    """A trial along d, its difference and the increments rederived from the trial pose agree."""
    for precision, numpy_dtype in PRECISIONS:
        with test.subTest(precision=precision):
            model = floating_chain(device)
            run = make_solver(model, precision)
            state = run.create_state(dt=1e-3)
            roots, k = run._floating, run._kinematics
            roots.tn.assign(state._monolithic_coupled.floating_com.numpy())
            roots.quat_n.assign(state._monolithic_coupled.floating_rotation.numpy())
            n = run._layout.scalar_dof_count
            direction = np.zeros(n, dtype=numpy_dtype)
            d = np.array([1e-3, -2e-3, 5e-4, 0.02, -0.01, 0.03], dtype=numpy_dtype)
            direction[roots.rows.numpy()[0]] = d
            direction = wp.array(direction, dtype=run._dtype, device=device)
            zero = (
                wp.zeros(1, dtype=k.vector, device=device),
                wp.full(1, wp.quat_identity(run._dtype), dtype=k.quaternion, device=device),
            )
            first = tuple(wp.zeros_like(a) for a in zero)
            second = tuple(wp.zeros_like(a) for a in zero)
            alpha = wp.array([0.5], dtype=run._dtype, device=device)
            roots.trial(zero, state, first, direction, alpha)
            roots.trial(first, state, second, direction, alpha)
            out = wp.zeros(n, dtype=run._dtype, device=device)
            roots.difference(zero, second, out)
            tolerance = 1e-12 if precision == "fp64" else 2e-6
            np.testing.assert_allclose(out.numpy()[roots.rows.numpy()[0]], d, atol=tolerance)
            derived = roots.derive_steps(state)
            np.testing.assert_allclose(derived[0].numpy(), second[0].numpy(), atol=tolerance)
            sign = np.sign(np.sum(derived[1].numpy() * second[1].numpy()))
            np.testing.assert_allclose(sign * derived[1].numpy(), second[1].numpy(), atol=tolerance)


def test_mirrors_match_newton_fk(test, device):
    """After stepping, joint_q/joint_qd follow Newton's FREE convention: eval_fk reproduces body_q/qd."""
    for precision, _dtype in PRECISIONS:
        with test.subTest(precision=precision):
            model = floating_chain(device)
            run = make_solver(model, precision)
            state = run.create_state(dt=1e-3)
            control = run.create_control()
            for _ in range(10):
                run.step(state, state, control, dt=1e-3)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
            reference = model.state()
            newton.eval_fk(
                model,
                wp.array(state.joint_q.numpy().astype(np.float32), device=device),
                wp.array(state.joint_qd.numpy().astype(np.float32), device=device),
                reference,
            )
            # Newton's FK is FP32: compare at its precision.
            np.testing.assert_allclose(reference.body_q.numpy(), state.body_q.numpy(), atol=2e-6)
            np.testing.assert_allclose(reference.body_qd.numpy(), state.body_qd.numpy(), atol=1e-5)


def test_rejections(test, device):
    """Unsupported floating-base inputs fail at construction or step with the joint or field named."""

    cases = []

    def free_below_root(builder):
        parent = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        child = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        builder.add_articulation(
            [
                builder.add_joint_revolute(-1, parent, axis=(0, 0, 1)),
                builder.add_joint_free(child, parent=parent, label="inner_free"),
            ]
        )

    cases.append((free_below_root, "FREE joint 'inner_free'.*not the world root"))

    def ball(builder):
        parent = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        child = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        builder.add_articulation(
            [builder.add_joint_free(parent), builder.add_joint_ball(parent, child, label="shoulder")]
        )

    cases.append((ball, "Joint 'shoulder'.*BALL"))

    def d6(builder):
        parent = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        child = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        axis = newton.ModelBuilder.JointDofConfig(axis=(1.0, 0.0, 0.0))
        builder.add_articulation(
            [builder.add_joint_free(parent), builder.add_joint_d6(parent, child, linear_axes=[axis], label="slide6")]
        )

    cases.append((d6, "Joint 'slide6'.*D6"))

    def distance(builder):
        parent = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        child = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)))
        builder.add_articulation(
            [builder.add_joint_free(parent), builder.add_joint_distance(parent, child, label="tether")]
        )

    cases.append((distance, "Joint 'tether'.*DISTANCE"))
    for edit, message in cases:
        with test.subTest(message=message):
            with test.assertRaisesRegex(ValueError, message):
                ActorTable.from_model(floating_chain(device, edit=edit))

    def kinematic(builder):
        builder.body_flags[0] = int(newton.BodyFlags.KINEMATIC)

    with test.assertRaisesRegex(ValueError, "FREE joint 'free'.*kinematic"):
        ActorTable.from_model(floating_chain(device, edit=kinematic))
    for name, value in (
        ("joint_armature", 0.01),
        ("joint_damping", 0.1),
        ("joint_limit_ke", 10.0),
        ("joint_limit_kd", 1.0),
        ("joint_friction", 0.1),
    ):
        with test.subTest(field=name):
            model = floating_chain(device)
            values = getattr(model, name).numpy()
            values[2] = value
            getattr(model, name).assign(values)
            with test.assertRaisesRegex(ValueError, f"Floating root joint has nonzero {name}"):
                make_solver(model)
    # Root drive gains need the implicit PD term (the 6D drive tests cover the enabled case).
    model = floating_chain(device)
    gains = model.joint_target_ke.numpy()
    gains[:3] = 10.0
    model.joint_target_ke.assign(gains)
    with test.assertRaisesRegex(ValueError, "Disabled joint term has nonzero parameters: joint_target_ke"):
        make_solver(model)
    # Invalid Model root coordinates fail at construction, before any root pose is derived.
    model = floating_chain(device)
    coordinates = model.joint_q.numpy()
    coordinates[3:7] = (0.0, 0.0, 0.0, 0.5)
    model.joint_q.assign(coordinates)
    with test.assertRaisesRegex(ValueError, "Floating root Model coordinates"):
        make_solver(model)


def test_failed_reset_leaves_state_unchanged(test, device):
    """A rejected reset of a floating-base state leaves every array untouched (the state owner resets
    in scratch); an incomplete history fails with a ValueError naming the missing field."""
    run = make_solver(floating_chain(device))
    for case in ("nonfinite", "fast"):
        with test.subTest(case=case):
            state = run.create_state(dt=1e-3)
            coordinates, rates = state.joint_q.numpy(), state.joint_qd.numpy()
            if case == "nonfinite":
                coordinates[3] = np.nan
                message = "finite|Nonfinite"
            else:
                # Within the unit-norm tolerance, so a reset would renormalize the mirror before the
                # angular-velocity check rejects the too-fast root.
                coordinates[3:7] *= 1 + 5e-6
                rates[5] = 2000.0
                message = "Initial angular velocity requires dt \\* \\|omega\\| < 1"
            state.joint_q.assign(coordinates)
            state.joint_qd.assign(rates)
            history = state._monolithic_coupled
            before = {
                name: getattr(state, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")
            }
            before.update({"history." + name: getattr(history, name).numpy().copy() for name in run._states.schema})
            with test.assertRaisesRegex(ValueError, message):
                run._states.reset(state, dt=1e-3)
            for name, expected in before.items():
                actual = getattr(history, name[8:]) if name.startswith("history.") else getattr(state, name)
                np.testing.assert_array_equal(actual.numpy(), expected, err_msg=name)
    state = run.create_state(dt=1e-3)
    del state._monolithic_coupled.floating_com
    with test.assertRaisesRegex(ValueError, "Invalid articulation history schema: missing floating_com"):
        run._states.reset(state, dt=1e-3)


def test_single_free_body_stays_rigid(test, device):
    """A FREE body without descendants keeps the rigid-actor path, with or without a FREE tree beside it."""

    def add_free_body(builder):
        body = builder.add_link(mass=0.3, inertia=wp.diag(wp.vec3(1e-3)), label="loose")
        builder.add_articulation([builder.add_joint_free(body, label="loose_free")])

    model = floating_chain(device, edit=add_free_body)
    table = ActorTable.from_model(model)
    test.assertEqual([a.kind for a in table.actors], ["articulation", "rigid"])
    test.assertEqual(table.actors[0].dof_count, 8)
    test.assertEqual(table.actors[0].state_count, 9)
    # A driven single FREE body keeps the rigid path and its existing rejection (PRD 2.10).
    gains = model.joint_target_ke.numpy()
    gains[list(table.actors[1].dof_indices)] = 10.0
    model.joint_target_ke.assign(gains)
    test.assertEqual([a.kind for a in ActorTable.from_model(model).actors], ["articulation", "rigid"])
    with test.assertRaisesRegex(ValueError, "Unsupported FREE joint term: joint_target_ke"):
        make_solver(model)


class TestJointFloatingBase(unittest.TestCase):
    pass


for _test in (
    test_root_jacobian_matches_finite_differences,
    test_pose_increment_round_trip,
    test_mirrors_match_newton_fk,
    test_rejections,
    test_single_free_body_stays_rigid,
    test_failed_reset_leaves_state_unchanged,
):
    add_function_test(TestJointFloatingBase, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
