# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-tree (self) contact: shared-column exclusion, energy-residual consistency, rest overlap and the option API."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests import test_monolithic_contact_multitree as multitree
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_contact_multitree import contact_terms, set_q, trees_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices

H = 1e-3
RADIUS = 0.02
CUBE = 0.015
SPACING = 4e-3  # the default self_collision box_sample_spacing


def box_grid_samples(half, spacing):
    """Independent P1Q3 samples of a box face grid: each face split into ceil(2 h / spacing) cells per
    edge, each cell split along the diagonal from its (-u, -v) corner (the solver's split)."""
    tris = []
    for axis in range(3):
        u, v = (a for a in range(3) if a != axis)
        # Edges that are an exact multiple of the spacing (up to FP32 scale rounding) keep that many cells.
        gu = np.linspace(-half[u], half[u], int(np.ceil(2 * half[u] / spacing * (1 - 1e-6))) + 1)
        gv = np.linspace(-half[v], half[v], int(np.ceil(2 * half[v] / spacing * (1 - 1e-6))) + 1)
        for sign in (-1, 1):
            for i in range(len(gu) - 1):
                for j in range(len(gv) - 1):
                    for corners in (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))):
                        tri = np.zeros((3, 3))
                        tri[:, axis] = sign * half[axis]
                        tri[:, u] = [gu[i + a] for a, _ in corners]
                        tri[:, v] = [gv[j + b] for _, b in corners]
                        tris.append(tri)
    tri = np.asarray(tris)
    bary = np.eye(3) / 2 + np.ones((3, 3)) / 6
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return np.einsum("qn,fni->fqi", bary, tri).reshape(-1, 3), np.repeat(area / 3, 3)


def two_branch_tree(device, *, gap=2e-3, rate=0.5, order=(0, 1), clear_flag=False, tip="sphere", root="revolute"):
    """A hub on a world revolute with two revolute branches whose tips face each other ``gap`` apart:
    spheres, boxes of half height ``RADIUS`` along y, or (``"cross"``) cubes turned 45 degrees about
    x and z so that two perpendicular edges face each other.

    ``order`` permutes the branch creation (labels stay with their geometry); ``rate`` [rad/s] turns
    the branches toward each other. ``root="free"`` makes the hub a floating (FREE) root.
    """
    builder, branches = two_branch_builder(gap=gap, order=order, clear_flag=clear_flag, tip=tip, root=root)
    model = builder.finalize(device=device)
    rates = model.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    # The left branch (+y) turns clockwise and the right one counterclockwise: the tips close.
    rates[starts[branches["left"]]] = -rate
    rates[starts[branches["right"]]] = rate
    model.joint_qd.assign(rates)
    return model


def two_branch_builder(
    *, gap=2e-3, order=(0, 1), clear_flag=False, builder=None, tip="sphere", third=False, root="revolute"
):
    """The unfinalized two-branch tree (see ``two_branch_tree``) and its branch joints by name;
    ``third`` adds a ``mid_arm`` branch whose small tip sphere is far from the others."""
    builder = builder or newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    inertia = wp.diag(wp.vec3(1e-4, 1e-4, 1e-4))
    hub = builder.add_link(mass=0.5, inertia=inertia, label="hub")
    if root == "free":
        joints = [builder.add_joint_free(hub, label="base")]
    else:
        joints = [builder.add_joint_revolute(-1, hub, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0, label="base")]
    offset = (CUBE * np.sqrt(2) if tip == "cross" else RADIUS) + gap / 2
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
        if tip == "cross":
            axis = wp.vec3(1.0, 0.0, 0.0) if side == 0 else wp.vec3(0.0, 0.0, 1.0)
            tip_xform = wp.transform((0.1, 0.0, 0.0), wp.quat_from_axis_angle(axis, np.pi / 4))
            builder.add_shape_box(link, xform=tip_xform, hx=CUBE, hy=CUBE, hz=CUBE, cfg=flags_cfg, label=f"{name}_tip")
        elif tip == "sphere":
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
    if third:
        link = builder.add_link(mass=0.1, inertia=inertia, label="mid_arm")
        builder.add_shape_sphere(
            link, xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), radius=0.005, cfg=cfg, label="mid_tip"
        )
        joints.append(
            builder.add_joint_revolute(
                hub,
                link,
                parent_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()),
                axis=(0, 0, 1),
                limit_ke=0.0,
                limit_kd=0.0,
                label="mid_hinge",
            )
        )
        branches["mid"] = joints[-1]
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
    """Box tips turned into each other: E, r, the tangent (with the field's normal curvature), both
    wrenches and residual differences match the independent two-ended oracle (both directions at half weight, on the box face grid),
    whose relative Jacobian cancels the hub column; the native hub row is exactly zero."""
    grid = lambda half: box_grid_samples(half, SPACING)  # noqa: E731
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision), patch.object(multitree, "box_face_samples", grid):
            model = two_branch_tree(device, gap=5e-5, rate=0.0, tip="box")
            run = trees_solver(model, precision=precision, self_collision={})
            test.assertEqual(set(run._contact.actor_samples.pair_kinds.values()), {"A-self"})
            # 30 x 40 x 24 mm boxes: 8, 10 and 6 cells per edge; 376 cells, 2256 samples per direction.
            test.assertEqual(run._contact.actor_samples.count, 2 * 2256)
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
            # Symmetric; not PSD in general: the curvature term is indefinite near box edges.
            np.testing.assert_allclose(actual, actual.T, rtol=1e-6, atol=1e-6 * scale)
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


def test_box_grid_samples(test, device):
    """The face grid covers the box surface exactly, with outward normals; a spacing beyond the box
    keeps two triangles per face; same-tree box sources use it and other box sources do not."""
    from newton._src.solvers.monolithic.contact_sampling import _surface_samples  # noqa: PLC0415

    half = np.array([0.015, 0.02, 0.012])
    for spacing, count in ((SPACING, 2256), (1.0, 36)):
        points, normals, weights = _surface_samples(newton.GeoType.BOX, half, box_spacing=spacing)
        test.assertEqual(len(points), count)
        test.assertAlmostEqual(weights.sum(), 8 * (half[0] * half[1] + half[1] * half[2] + half[0] * half[2]))
        axis = np.argmax(np.abs(points) / half, axis=1)
        np.testing.assert_allclose(np.abs(points[np.arange(count), axis]), half[axis], rtol=1e-12)
        np.testing.assert_allclose(normals[np.arange(count), axis], np.sign(points[np.arange(count), axis]))
        expected, area = box_grid_samples(half, spacing)
        # Sort whole points by coordinates rounded below the generators' last-bit differences.
        order, reference = np.lexsort(np.round(points, 12).T), np.lexsort(np.round(expected, 12).T)
        np.testing.assert_allclose(points[order], expected[reference], atol=1e-15)
        np.testing.assert_allclose(weights[order], area[reference], rtol=1e-12)
    with test.assertRaisesRegex(ValueError, "spacing must be positive"):
        _surface_samples(newton.GeoType.BOX, half, box_spacing=0.0)
    with test.assertRaisesRegex(ValueError, "needs 12000000 triangles"):
        _surface_samples(newton.GeoType.BOX, np.array([1.0, 1.0, 1.0]), box_spacing=2e-3)
    # The sampler refuses same-tree box pairs without a spacing (the rest check would sample a grid).
    from newton._src.solvers.monolithic.contact_sampling import ContactSamples  # noqa: PLC0415

    run = make_solver(two_branch_tree(device, tip="box"))
    with test.assertRaisesRegex(ValueError, "need a box sample spacing"):
        ContactSamples(run.model, run._actors, radius=0.0, base_triplet_count=0, _self_pairs=run._self_collision.pairs)


def test_box_grid_faces_in_steps(test, device):
    """A box tip that is the source of a self pair (face grid) and of a static pair (12 triangles)
    keeps the two face numberings apart: static faces 0-11, grid faces after them. The spatial sweep
    bounds every face, and both pairs carry force while the tips close (the static box sits 0.3 mm
    into the left tip's top face)."""
    builder, branches = two_branch_builder(tip="box")
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    top = 0.012 + 0.005 - 3e-4
    builder.add_shape_box(
        -1, xform=wp.transform((0.1, 0.03, top), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.005, cfg=cfg
    )
    model = builder.finalize(device=device)
    rates = model.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    rates[starts[branches["left"]]], rates[starts[branches["right"]]] = -0.5, 0.5
    model.joint_qd.assign(rates)
    run = make_solver(model)
    samples = run._contact.actor_samples
    test.assertEqual(set(samples.pair_kinds.values()), {"A-self", "A-static"})
    keys = np.asarray(samples.keys)
    self_pair = np.isin(keys[:, 1], [0, 1])
    test.assertTrue(np.all(keys[self_pair, 2] >= 12))
    test.assertTrue(np.all(keys[~self_pair, 2] < 12))
    state, control = run.create_state(dt=H), run.create_control()
    loaded = set()
    for _ in range(60):
        run.step(state, state, control, dt=H)
        test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
        for pair in run.last_stats.variational["actor_contact_pairs"]:
            if np.abs(pair["world_wrench"][0][:3]).max() > 0:
                loaded.add(pair["kind"])
    test.assertEqual(loaded, {"A-self", "A-static"})


def test_edge_contact_linear_solves(test, device):
    """Crossed cube tips closing edge on edge: the field's curvature near box edges makes the tangent
    indefinite, yet every step converges and every linear solve succeeds without regularization retries
    (no non-positive curvature; CUDA runs the production PCG). Measured: 150/150 SUCCESS, 0 retries in both precisions on CPU and CUDA."""
    options = {"pcg_mode": "production"} if wp.get_device(device).is_cuda else {}
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            run = make_solver(two_branch_tree(device, tip="cross", gap=2e-3, rate=0.5), precision, **options)
            state, control = run.create_state(dt=H), run.create_control()
            loaded, solves = 0.0, 0
            for _ in range(150):
                run.step(state, state, control, dt=H)
                stats = run.last_stats
                test.assertTrue(stats.converged, stats.failure_reason)
                # Every linear solve succeeds (no NON_POSITIVE_CURVATURE or other PCG failure) without retries.
                # A step converged at its predictor has no linear solve.
                attempts = stats.variational.get("linear_direction_attempts", [])
                solves += len(attempts)
                test.assertLessEqual({a["status"] for a in attempts}, {"SUCCESS"}, attempts)
                test.assertEqual(stats.regularization_retries, 0)
                loaded = max(loaded, float(np.abs(run.get_contact_forces(state).joint_force).max()))
            test.assertGreater(loaded, 0.0, "the edges never touched")
            test.assertGreater(solves, 0)


def test_crossed_box_edges(test, device):
    """Two cube edges crossed by 2 mm hold no sample of the 12-triangle box (it reads a 4.6 mm gap);
    the same-tree face grid finds the overlap at construction, and turning the cubes apart clears it."""
    with test.assertRaisesRegex(ValueError, r"rest overlap.*left_tip.*right_tip"):
        make_solver(two_branch_tree(device, gap=-2e-3, rate=0.0, tip="cross"))
    run = make_solver(two_branch_tree(device, gap=2e-3, rate=0.0, tip="cross"))
    test.assertEqual(run.self_collision_report, ())


def test_floating_root(test, device):
    """With a FREE hub the six root rows are the shared prefix: their self-contact force is exactly zero
    while the branches push apart. The rest check reads the floating root pose from the state: moving and
    turning the whole tree leaves the gap unchanged, and a branch overlap still fails."""
    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            run = make_solver(two_branch_tree(device, root="free"), precision)
            state, control = run.create_state(dt=H), run.create_control()
            touched = False
            for _ in range(60):
                run.step(state, state, control, dt=H)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                forces = run.get_contact_forces(state).joint_force
                np.testing.assert_array_equal(forces[:6], 0.0)
                if np.abs(forces[6:]).max() > 0:
                    touched = True
                    test.assertLess(forces[6] * forces[7], 0.0)
            test.assertTrue(touched, "the tips never touched")
    model = two_branch_tree(device, gap=5e-5, rate=0.0, root="free")
    run = make_solver(model, normal_smoothing=1e-4)
    at_model = run.self_collision_report[0]["gap_m"]
    source = model.state()
    q = source.joint_q.numpy()
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    q[:3] = (0.3, -0.2, 0.5)
    q[3:7] = (*(np.sin(0.35) * axis), np.cos(0.35))
    source.joint_q.assign(q)
    state = run.create_state(dt=H, source=source)
    np.testing.assert_allclose(state._monolithic_coupled.floating_com.numpy()[0][:2], (0.3, -0.2), atol=0.05)
    test.assertAlmostEqual(run.self_collision_report[0]["gap_m"], at_model, delta=1e-12)
    branch = model.joint_q_start.numpy()[1]
    q[branch] = -0.06
    source.joint_q.assign(q)
    with test.assertRaisesRegex(ValueError, "rest overlap"):
        run.create_state(dt=H, source=source)
    coordinates = state.joint_q.numpy()
    coordinates[branch] = -0.06
    state.joint_q.assign(coordinates)
    with test.assertRaisesRegex(ValueError, "rest overlap"):
        run.reset_state(state, dt=H)


def branchy_tree(device):
    """hub (world revolute) -> upper -> {fore_a, fore_b}, plus FIXED siblings weld_a, weld_b on the hub
    whose spheres overlap. Pairs: an ancestor-descendant pair (hub, fore_a), an LCA below the root
    (fore_a, fore_b) and the welded pair."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    inertia = wp.diag(wp.vec3(1e-4, 1e-4, 1e-4))
    bodies = {}

    def link(name, parent, offset, *, fixed=False, at=(0.0, 0.0, 0.0)):
        body = builder.add_link(mass=0.1, inertia=inertia, label=name)
        builder.add_shape_sphere(body, xform=wp.transform(at, wp.quat_identity()), radius=0.01, cfg=cfg)
        xform = wp.transform(offset, wp.quat_identity())
        if fixed:
            joint = builder.add_joint_fixed(parent, body, parent_xform=xform)
        else:
            joint = builder.add_joint_revolute(
                parent, body, parent_xform=xform, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0
            )
        bodies[name] = body
        return joint

    joints = [link("hub", -1, (0.0, 0.0, 0.0))]
    joints.append(link("upper", bodies["hub"], (0.0, 0.0, 0.2)))
    joints.append(link("fore_a", bodies["upper"], (0.1, 0.0, 0.0)))
    joints.append(link("fore_b", bodies["upper"], (-0.1, 0.0, 0.0)))
    joints.append(link("weld_a", bodies["hub"], (0.0, 0.3, 0.0), fixed=True))
    joints.append(link("weld_b", bodies["hub"], (0.0, 0.3, 0.0), fixed=True, at=(0.005, 0.0, 0.0)))
    builder.add_articulation(joints)
    return builder.finalize(device=device), bodies


def test_shared_columns_by_tree_shape(test, device):
    """Each self sample carries exactly the DoFs below the two bodies' lowest common ancestor (walked
    independently up joint_parent), for an ancestor-descendant pair, an LCA below the root and sibling
    branches; welded siblings, overlapping at rest, are no pair and construct."""
    from newton._src.solvers.monolithic.contact_sampling import ContactSamples  # noqa: PLC0415

    model, bodies = branchy_tree(device)
    run = make_solver(model)
    labels = {b: n for n, b in bodies.items()}
    shape_body = model.shape_body.numpy()
    pairs = {tuple(sorted((labels[shape_body[a]], labels[shape_body[b]]))) for a, b in run._self_collision.pairs}
    test.assertIn(("fore_a", "hub"), pairs)
    test.assertIn(("fore_a", "fore_b"), pairs)
    test.assertNotIn(("weld_a", "weld_b"), pairs)
    parent = dict(zip(model.joint_child.numpy().tolist(), model.joint_parent.numpy().tolist(), strict=True))
    qd_start = model.joint_qd_start.numpy()
    joint_of = {int(c): j for j, c in enumerate(model.joint_child.numpy())}

    def path(body):
        rows = set()
        while body >= 0:
            j = joint_of[body]
            rows.update(range(int(qd_start[j]), int(qd_start[j + 1])))
            body = parent[body]
        return rows

    samples = ContactSamples(
        model, run._actors, radius=0.0, base_triplet_count=0, _self_pairs=run._self_collision.pairs
    )
    rows = np.asarray(run._actors.maps["joint_dof_to_global"])
    checked = set()
    for (source, target, _, _), columns in zip(samples.keys, samples.supports, strict=True):
        a, b = int(shape_body[source]), int(shape_body[target])
        expected = {int(rows[d]) for d in path(a) ^ path(b)}
        test.assertEqual(set(columns), expected, (labels[a], labels[b]))
        checked.add(tuple(sorted((labels[a], labels[b]))))
    test.assertEqual(checked, pairs)
    test.assertEqual(samples.support_size, sum(3 * len(c) for c in samples.supports))
    # The pool's demand drops the same shared columns: its widest record and its budget match the samples.
    widths = [len(c) for c in samples.supports]
    test.assertEqual(run._contact.pool.max_support, max(widths))
    test.assertEqual(run._contact.pool.capacity, sum(max(1, k) for k in widths))


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
        ({"box_sample_spacing": 0.0}, "box_sample_spacing must be a positive finite length"),
        ({"box_sample_spacing": "4mm"}, "box_sample_spacing must be a positive finite length"),
        ({"box_sample_spacing": 3.0}, r"box_sample_spacing must lie in \[1e-05, 1\.0\] m"),
        ({"box_sample_spacing": 1e-6}, r"box_sample_spacing must lie in \[1e-05, 1\.0\] m"),
        ({"exclude": [["left_arm", "nowhere"]]}, "'nowhere' matches no body"),
        ({"exclude": [["left_arm", "hub"]]}, "has no same-tree Model contact pair"),
        ({"exclude": [["left_arm"]]}, "body label pairs"),
        ({"exclude": [["left_arm", "left_arm"]]}, "is not a pair of bodies of one tree"),
        ({"exclude": [["left_arm", 3]]}, "labels must be strings"),
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
    builder, _ = two_branch_builder()
    other = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(1e-4)), label="other")
    builder.add_articulation([builder.add_joint_revolute(-1, other, axis=(0, 0, 1), limit_ke=0.0, limit_kd=0.0)])
    with test.assertRaisesRegex(ValueError, "is not a pair of bodies of one tree"):
        make_solver(builder.finalize(device=device), self_collision={"exclude": [["other", "left_arm"]]})


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
    """Self-collision on, off, with another box spacing and with a different exclusion are different
    checkpoint identities: each rejects the others' checkpoints and restores its own."""

    def rejects(saver, loader):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            saver.save_checkpoint(saver.create_state(dt=H), path)
            with test.assertRaisesRegex(ValueError, "Incompatible articulation state configuration"):
                loader.load_checkpoint(loader.create_state(dt=H), path)
            saver.load_checkpoint(saver.create_state(dt=H), path)

    model = two_branch_tree(device, rate=0.0)
    on = make_solver(model)
    rejects(on, make_solver(model, self_collision=None))
    rejects(on, make_solver(model, self_collision={"box_sample_spacing": 8e-3}))
    # A third branch adds body pairs, so one exclusion leaves pairs behind.
    model = two_branch_builder(third=True)[0].finalize(device=device)
    every = make_solver(model)
    test.assertEqual(len(every._self_collision.pairs), 3)
    rejects(every, make_solver(model, self_collision={"exclude": [["left_arm", "mid_arm"]]}))


class TestContactSelfCollision(unittest.TestCase):
    pass


for _test in (
    test_shared_columns_carry_no_self_force,
    test_energy_residual_consistency,
    test_contact_terms_match_oracle,
    test_floating_root,
    test_shared_columns_by_tree_shape,
    test_box_grid_samples,
    test_crossed_box_edges,
    test_edge_contact_linear_solves,
    test_box_grid_faces_in_steps,
    test_rest_overlap_fails_everywhere,
    test_precompression_report,
    test_option_errors,
    test_label_order_and_cleared_flags,
    test_checkpoint_identity,
):
    add_function_test(TestContactSelfCollision, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
