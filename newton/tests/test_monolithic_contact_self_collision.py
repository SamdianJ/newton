# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-tree (self) contact: shared-column exclusion, energy-residual consistency, rest overlap and the option API."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_contact_multitree import contact_terms, set_q, trees_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices

H = 1e-3
RADIUS = 0.02


def two_branch_tree(device, *, gap=2e-3, rate=0.5, order=(0, 1), clear_flag=False, tip="sphere"):
    """A hub on a world revolute with two revolute branches whose tips (spheres, or boxes of half
    height ``RADIUS`` along y) face each other ``gap`` apart.

    ``order`` permutes the branch creation (labels stay with their geometry); ``rate`` [rad/s] turns
    the branches toward each other.
    """
    builder, branches = two_branch_builder(gap=gap, order=order, clear_flag=clear_flag, tip=tip)
    model = builder.finalize(device=device)
    rates = model.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    # The left branch (+y) turns clockwise and the right one counterclockwise: the tips close.
    rates[starts[branches["left"]]] = -rate
    rates[starts[branches["right"]]] = rate
    model.joint_qd.assign(rates)
    return model


def two_branch_builder(*, gap=2e-3, order=(0, 1), clear_flag=False, builder=None, tip="sphere"):
    """The unfinalized two-branch tree (see ``two_branch_tree``) and its branch joints by name."""
    builder = builder or newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    inertia = wp.diag(wp.vec3(1e-4, 1e-4, 1e-4))
    hub = builder.add_link(mass=0.5, inertia=inertia, label="hub")
    joints = [builder.add_joint_revolute(-1, hub, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0, label="base")]
    offset = RADIUS + gap / 2
    branches = {}
    for side in order:
        sign = 1.0 if side == 0 else -1.0
        name = "left" if side == 0 else "right"
        link = builder.add_link(mass=0.2, inertia=inertia, com=wp.vec3(0.05, 0.0, 0.0), label=f"{name}_arm")
        flags_cfg = (
            cfg
            if not (clear_flag and side == 1)
            else builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False, has_shape_collision=False)
        )
        tip_xform = wp.transform((0.1, 0.0, 0.0), wp.quat_identity())
        if tip == "sphere":
            builder.add_shape_sphere(link, xform=tip_xform, radius=RADIUS, cfg=flags_cfg, label=f"{name}_tip")
        else:
            builder.add_shape_box(
                link, xform=tip_xform, hx=0.015, hy=RADIUS, hz=0.012, cfg=flags_cfg, label=f"{name}_tip"
            )
        joints.append(
            builder.add_joint_revolute(
                hub,
                link,
                parent_xform=wp.transform((0.0, sign * (offset + 0.03), 0.0), wp.quat_identity()),
                # The child frame sits 0.03 m inboard of the link origin, which lands at y = +-offset.
                child_xform=wp.transform((0.0, sign * 0.03, 0.0), wp.quat_identity()),
                axis=(0, 0, 1),
                limit_ke=0.0,
                limit_kd=0.0,
                label=f"{name}_hinge",
            )
        )
        branches[name] = joints[-1]
    builder.add_articulation(joints)
    return builder, branches


def make_solver(model, precision="fp64", **options):
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    options.setdefault("self_collision", {})
    return SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        contact_stiffness=1e7,
        friction_coefficient=0.0,
        precision=precision,
        **options,
    )


def test_shared_columns_carry_no_self_force(test, device):
    """In contact, the self force on the shared hub row is exactly zero and the branch rows push apart."""
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            model = two_branch_tree(device)
            run = make_solver(model, precision)
            test.assertEqual(set(run._contact.actor_samples.pair_kinds.values()), {"A-self"})
            state, control = run.create_state(dt=H), run.create_control()
            touched = False
            for _ in range(60):
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                forces = run.get_contact_forces(state).joint_force
                test.assertEqual(float(forces[0]), 0.0)
                if np.abs(forces[1:]).max() > 0:
                    touched = True
                    # Equal and opposite generalized torques on the mirror-image branches.
                    test.assertLess(forces[1] * forces[2], 0.0)
            test.assertTrue(touched, "the tips never touched")


def test_energy_residual_consistency(test, device):
    """With zero friction the contact is a potential: its energy's slope along a trial is residual . d,
    including along the shared hub DoF (zero) and the branch DoFs."""
    model = two_branch_tree(device, gap=2e-3)
    run = make_solver(model, "fp64")
    state, control = run.create_state(dt=H), run.create_control()
    for _ in range(25):
        run.step(state, state, control, dt=H)
    test.assertTrue(run._conservative_energy())
    forces = run.get_contact_forces(state).joint_force
    test.assertGreater(np.abs(forces).max(), 0.0, "no contact in the check state")
    n = run._delta.size
    rng = np.random.default_rng(3)
    for direction in (np.eye(n)[0], np.eye(n)[1], rng.normal(size=n)):
        run._delta.assign(direction * 1e-4)
        eps = 1e-3
        energies = []
        for alpha in (eps, -eps):
            run._take_trial(alpha, H)
            energy, _residual, _ = run._evaluate(run._trial, H, build_tangent=False)
            energies.append(float(energy))
        _, residual, _ = run._evaluate(run._accepted, H, build_tangent=False)
        slope = (energies[0] - energies[1]) / (2 * eps)
        test.assertAlmostEqual(slope, float(residual @ (direction * 1e-4)), delta=1e-6 * (1 + abs(slope)))


def test_contact_terms_match_oracle(test, device):
    """Box tips turned into each other: E, r, the PSD tangent, both wrenches and residual differences
    match the independent two-ended oracle (both directions at half weight), whose relative Jacobian
    cancels the hub column; the native hub row is exactly zero."""
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            model = two_branch_tree(device, gap=5e-5, rate=0.0, tip="box")
            run = trees_solver(model, precision=precision, self_collision={})
            test.assertEqual(set(run._contact.actor_samples.pair_kinds.values()), {"A-self"})
            state = run.create_state(dt=H)
            batch = native_batch(run)
            qn = state.joint_q.numpy().astype(float)
            query(batch, run, state, start=True)
            # The hub turns too; the branches close about 0.2 mm each (the tips start 0.05 mm apart).
            q = qn + np.array([0.01, -2e-3, 2e-3])
            set_q(run, state, q)
            q = state.joint_q.numpy().astype(float)
            query(batch, run, state)
            energy, residual, matrix, forces = contact_terms(run.model, q, qn)
            rows = np.asarray(run._actors.maps["joint_dof_to_global"])
            actual = batch_matrix(batch)[np.ix_(rows, rows)]
            native_residual = batch.pool.residual.result.numpy()[rows]
            tol = 1e-3 if precision == "fp32" else 1e-8
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            test.assertGreater(abs(residual[1]), 1e-4, "the tips are not in contact")
            test.assertEqual(float(native_residual[0]), 0.0)
            np.testing.assert_array_equal(actual[0], 0.0)
            np.testing.assert_array_equal(actual[:, 0], 0.0)
            np.testing.assert_allclose(batch.pool.energy.result.numpy()[:2], energy, rtol=tol, atol=tol * 1e-4)
            np.testing.assert_allclose(native_residual, residual, rtol=tol, atol=tol)
            np.testing.assert_allclose(actual, matrix, rtol=max(tol, 1e-5), atol=max(tol, 1e-5))
            scale = max(1, np.linalg.norm(matrix))
            np.testing.assert_allclose(actual, actual.T, rtol=1e-6, atol=1e-6 * scale)
            test.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -1e-6 * scale)
            actual_forces = batch.pool.force.result.numpy()[: 6 * run.model.body_count].reshape(-1, 6)
            np.testing.assert_allclose(actual_forces, forces, rtol=tol, atol=tol)
            if precision == "fp64":
                native, reference = [], []
                for direction in np.eye(3):
                    values, expected = [], []
                    for sign in (-1, 1):
                        set_q(run, state, q + sign * 1e-7 * direction)
                        query(batch, run, state)
                        values.append(batch.pool.residual.result.numpy()[rows].copy())
                        expected.append(contact_terms(run.model, state.joint_q.numpy().astype(float), qn)[1])
                    native.append((values[1] - values[0]) / 2e-7)
                    reference.append((expected[1] - expected[0]) / 2e-7)
                np.testing.assert_allclose(native, reference, rtol=1e-6, atol=1e-4)
                np.testing.assert_array_equal(np.asarray(native)[:, 0], 0.0)


def test_rest_overlap_fails_everywhere(test, device):
    """Overlapping tips fail at construction, create_state and reset_state, with bodies, shapes and depth;
    the same threshold applies to FP32 and FP64 (the check is FP64 on the host)."""
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision, where="construction"):
            # The depth is the deepest P1Q3 sample's (0.97 mm of the exact 1 mm: no sample sits on the axis).
            with test.assertRaisesRegex(ValueError, r"rest overlap.*left_arm.*right_arm.*left_tip.*depth 9\.7"):
                make_solver(two_branch_tree(device, gap=-1e-3), precision)
        model = two_branch_tree(device, gap=2e-3, rate=0.0)
        run = make_solver(model, precision)
        source = model.state()
        q = source.joint_q.numpy()
        starts = model.joint_q_start.numpy()
        # Turning the left branch in by 0.06 rad drives its tip about 6 mm into the right one.
        q[starts[1]] = -0.06
        source.joint_q.assign(q)
        with test.subTest(precision=precision, where="create_state"):
            with test.assertRaisesRegex(ValueError, "rest overlap"):
                run.create_state(dt=H, source=source)
        with test.subTest(precision=precision, where="reset_state"):
            state = run.create_state(dt=H)
            before = state.joint_q.numpy().copy()
            coordinates = state.joint_q.numpy()
            coordinates[starts[1]] = -0.06
            state.joint_q.assign(coordinates)
            with test.assertRaisesRegex(ValueError, "rest overlap"):
                run.reset_state(state, dt=H)
            # reset_state restores the state it was given.
            np.testing.assert_array_equal(state.joint_q.numpy()[starts[1]], coordinates[starts[1]])
            test.assertNotEqual(before[starts[1]], coordinates[starts[1]])


def test_precompression_report(test, device):
    """Gaps below normal_smoothing but not overlapping are reported, not rejected."""
    run = make_solver(two_branch_tree(device, gap=5e-5, rate=0.0), normal_smoothing=1e-4)
    report = run.self_collision_report
    test.assertEqual(len(report), 1)
    test.assertTrue(0.0 <= report[0]["gap_m"] < 1e-4, report)


def test_option_errors(test, device):
    """Unknown keys, unknown, ambiguous or cross-tree labels, missing pairs and no pairs fail by name."""
    model = two_branch_tree(device)
    for option, message in (
        ({"excluded": []}, "Unknown self_collision keys"),
        ({"exclude": [["left_arm", "nowhere"]]}, "'nowhere' matches no body"),
        ({"exclude": [["left_arm", "hub"]]}, "has no same-tree Model contact pair"),
        ({"exclude": [["left_arm"]]}, "body label pairs"),
        ([], "must be None or a dict"),
    ):
        with test.subTest(message=message):
            with test.assertRaisesRegex(ValueError, message):
                make_solver(model, self_collision=option)
    with test.assertRaisesRegex(ValueError, "enable_self_collisions=True"):
        make_solver(model, self_collision={"exclude": [["left_arm", "right_arm"]]})
    # A second tree with a body of the same label makes the label ambiguous; across trees is not a pair.
    builder, _ = two_branch_builder()
    other = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)), label="hub")
    builder.add_articulation([builder.add_joint_revolute(-1, other, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0)])
    with test.assertRaisesRegex(ValueError, "'hub' matches 2 bodies"):
        make_solver(builder.finalize(device=device), self_collision={"exclude": [["hub", "left_arm"]]})


def test_label_order_and_cleared_flags(test, device):
    """The resolved pair set does not depend on builder order; a cleared COLLIDE_SHAPES flag drops the pair
    (like assets that clear flags after finalization), which then leaves no self pair."""
    sets = []
    for order in ((0, 1), (1, 0)):
        model = two_branch_tree(device, order=order)
        run = make_solver(model)
        labels = model.shape_label
        sets.append({tuple(sorted((labels[a], labels[b]))) for a, b in run._self_collision.pairs})
    test.assertEqual(sets[0], sets[1])
    test.assertEqual(sets[0], {("left_tip", "right_tip")})
    with test.assertRaisesRegex(ValueError, "enable_self_collisions=True"):
        make_solver(two_branch_tree(device, clear_flag=True))


def test_checkpoint_identity(test, device):
    """Self-collision on, off and with a different exclusion are different checkpoint identities."""
    model = two_branch_tree(device, rate=0.0)
    on = make_solver(model)
    off = make_solver(model, self_collision=None)
    test.assertNotEqual(on._kinematics.digest, off._kinematics.digest)
    state = on.create_state(dt=H)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "on.npz"
        on.save_checkpoint(state, path)
        other = off.create_state(dt=H)
        with test.assertRaisesRegex(ValueError, "Incompatible articulation state configuration"):
            off.load_checkpoint(other, path)
        restored = on.create_state(dt=H)
        on.load_checkpoint(restored, path)


class TestContactSelfCollision(unittest.TestCase):
    pass


for _test in (
    test_shared_columns_carry_no_self_force,
    test_energy_residual_consistency,
    test_contact_terms_match_oracle,
    test_rest_overlap_fails_everywhere,
    test_precompression_report,
    test_option_errors,
    test_label_order_and_cleared_flags,
    test_checkpoint_identity,
):
    add_function_test(TestContactSelfCollision, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
