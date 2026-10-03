# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise short, physically active mixed-actor contact and continuation."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.test_monolithic_solver_rigid import assert_physical, physical_frame
from newton.tests.unittest_utils import add_function_test, get_test_devices

CASE = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures/actor_contact_short_v2.json"


def contact_scene(device, kind="chain", *, before=False, mesh=False):
    """Construct supported A+R+T inputs with explicitly enabled pair-specific collision."""
    case = json.loads(CASE.read_text())
    builder = newton.ModelBuilder(gravity=case["gravity"])
    centers = {name: np.array(v) for name, v in case["centers"].items()}
    if kind in ("R-R", "R-T", "R-static"):
        centers["pusher"] += case["inactive_offset"]
    if kind == "R-T":
        centers["rigid_0"] = centers["rigid_1"].copy()
    dimensions = np.array(case["dimensions"])
    shapes = {}

    def add_shape(body, name, *, particles=False):
        cfg = builder.ShapeConfig(density=0, margin=case["shape_margin"], has_particle_collision=particles)
        if mesh and name == "rigid_0":
            source = newton.Mesh.create_box(*dimensions / 2, compute_inertia=False)
            if wp.get_device(device).is_cuda:
                source.build_sdf(
                    device=device,
                    max_resolution=32,
                    margin=0.06,
                    narrow_band_range=(-0.06, 0.06),
                    texture_format="float32",
                )
            shapes[name] = builder.add_shape_mesh(body, mesh=source, cfg=cfg)
        else:
            shapes[name] = builder.add_shape_box(
                body, **dict(zip(("hx", "hy", "hz"), dimensions / 2, strict=True)), cfg=cfg
            )

    def add_arm():
        body = builder.add_link(mass=case["mass"]["pusher"], inertia=wp.diag(wp.vec3(0.001)), label="pusher")
        pd = case["pusher_pd"]
        joint = builder.add_joint_prismatic(
            -1,
            body,
            axis=newton.Axis.X,
            parent_xform=wp.transform(centers["pusher"], wp.quat_identity()),
            target_ke=pd["ke"],
            target_kd=pd["kd"],
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            effort_limit=pd["effort_limit"],
            velocity_limit=pd["velocity_limit"],
            limit_ke=0,
            limit_kd=0,
            armature=0,
            damping=0,
            friction=0,
        )
        builder.add_articulation([joint])
        add_shape(body, "pusher")

    def add_rigids():
        for i in range(2 if kind in ("R-R", "chain") else 1):
            name = f"rigid_{i}"
            mass = case["mass"]["rigid"]
            inertia = mass / 12 * (np.sum(dimensions**2) - dimensions**2)
            body = builder.add_link(
                xform=wp.transform(centers[name], wp.quat_identity()),
                mass=mass,
                inertia=wp.diag(wp.vec3(inertia)),
                com=case["rigid_com"],
                label=name,
            )
            anchors = []
            for key in ("parent_anchor", "child_anchor"):
                spec = case[key]
                w = np.array(spec["rotation_vector"])
                angle = np.linalg.norm(w)
                anchors.append(
                    wp.transform(spec["position"], wp.quat_from_axis_angle(wp.vec3(w / angle), float(angle)))
                )
            joint = builder.add_joint_free(body, parent_xform=anchors[0], child_xform=anchors[1])
            builder.add_articulation([joint])
            add_shape(body, name, particles=kind == "R-T" or (kind == "chain" and i == 1))

    if before:
        add_rigids()
        add_arm()
    else:
        add_arm()
        add_rigids()
    if kind == "R-static":
        shapes["static"] = builder.add_shape_plane(
            body=-1,
            width=0,
            length=0,
            xform=wp.transform((case["static_plane_x"], 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)),
            cfg=builder.ShapeConfig(density=0, margin=case["shape_margin"], has_particle_collision=False),
        )
    allowed = {
        frozenset(p)
        for p in (
            {
                "A-R": [("pusher", "rigid_0")],
                "R-R": [("rigid_0", "rigid_1")],
                "R-static": [("rigid_0", "static")],
                "chain": [("pusher", "rigid_0"), ("rigid_0", "rigid_1")],
            }
        ).get(kind, [])
    }
    for a, b in itertools.combinations(shapes, 2):
        if frozenset((a, b)) not in allowed:
            builder.shape_collision_filter_pairs.append((shapes[a], shapes[b]))
    edge = case["tet"]["edge"]
    vertices = np.array(list(itertools.product((-edge / 2, edge / 2), repeat=3)))
    tets = []
    for axes in itertools.permutations((4, 2, 1)):
        ids = [0, axes[0], axes[0] + axes[1], 7]
        if np.linalg.det((vertices[ids[1:]] - vertices[ids[0]]).T) < 0:
            ids[1], ids[2] = ids[2], ids[1]
        tets.extend(ids)
    position = (0, 0, 0) if kind in ("R-T", "chain") else case["inactive_offset"]
    builder.add_soft_mesh(
        pos=position,
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        vertices=vertices.tolist(),
        indices=tets,
        density=case["tet"]["density"],
        k_mu=10000 / 2.6,
        k_lambda=10000 * 0.3 / (1.3 * 0.4),
        k_damp=0,
        particle_radius=case["particle_radius"],
        tri_ke=0,
        tri_ka=0,
        tri_kd=0,
        tri_drag=0,
        tri_lift=0,
        edge_ke=0,
        edge_kd=0,
    )
    for i in np.flatnonzero(vertices[:, 0] == edge / 2):
        builder.particle_mass[i] = 0
    model = builder.finalize(device=device)
    return model


def contact_solver(
    model, *, permuted=False, law="C-IE", globalization="ArmijoOrResNorm", mode="diagnostic", graph=False, **kwargs
):
    """Run the unchanged public solver with an internal layout permutation seam."""
    actors = ActorTable.from_model(model)
    if permuted:
        order = (3, 2, 0, 1) if len(actors.actors) == 4 else (2, 0, 1)
        actors = ActorTable.from_model(model, actor_order=order)
    options = {
        "allow_nonconservative_energy": True,
        "contact_stiffness": 1e7,
        "friction_coefficient": 0.5,
        "normal_smoothing": 1e-4,
        "friction_velocity_scale": 0.01,
        **json.loads(CASE.read_text())["solver"],
        **kwargs,
    }
    with patch.object(ActorTable, "from_model", return_value=actors):
        return SolverMonolithic(
            model,
            contact_model=law,
            globalization=globalization,
            pcg_mode=mode,
            use_cuda_graph=graph,
            joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
            tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
            **options,
        )


def contact_state(run, kind):
    """Initialize the versioned physical preload without changing rest material geometry."""
    case = json.loads(CASE.read_text())
    source = run.model.state()
    if kind in ("chain", "R-T"):
        x = source.particle_q.numpy()
        x[:, 0] += case["tet_initial_compression"] * (case["tet"]["edge"] / 2 - x[:, 0]) / case["tet"]["edge"]
        source.particle_q.assign(x)
    if kind == "R-static":
        forces = source.body_f.numpy().copy()
        forces[run.model.body_label.index("rigid_0"), :3] = case["static_body_force_world"]
        source.body_f.assign(forces)
    return run.create_state(dt=case["dt"], source=source)


def test_new_pair_step(test, device):
    """Publish both new contact endpoints through a genuine public solver step."""
    for kind in ("A-R", "R-R", "R-T", "R-static"):
        with test.subTest(kind=kind):
            run = contact_solver(contact_scene(device, kind))
            state = contact_state(run, kind)
            run.step(state, state, run.model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            forces = run.get_contact_forces(state)
            test.assertGreater(np.linalg.norm(forces.body_wrench[run._rigid.body_indices]), 1e-4)


def routes(device):
    return [("diagnostic", False), ("production", False)] + ([("production", True)] if device.is_cuda else [])


def frame(run, state):
    result = physical_frame(run, state)
    for actor in run._rigid.actors:
        body = actor.body_indices[0]
        label = run.model.body_label[body]
        result["force_r." + label] = state._monolithic_coupled.contact_body_force.numpy()[body].copy()
        result["force_rq." + label] = state._monolithic_coupled.contact_joint_force.numpy()[
            list(actor.dof_indices)
        ].copy()
    return result


def check_contact_step(test, run, state, control, kind, step):
    case = json.loads(CASE.read_text())
    targets = control.joint_target_q.numpy()
    targets[list(run._articulation_actor.target_indices)] = (step + 1) * case["target_increment_per_step"]
    control.joint_target_q.assign(targets)
    run.step(state, state, control, dt=case["dt"])
    test.assertTrue(run.last_stats.converged, run.last_stats)
    test.assertFalse(run.last_stats.rolled_back)
    pairs = run.last_stats.variational["actor_contact_pairs"]
    test.assertEqual(len(pairs), 3 if kind == "chain" else 1)
    test.assertEqual(
        {pair["kind"] for pair in pairs},
        {"A-R", "R-R", "R-T"} if kind == "chain" else {"rigid-static" if kind == "R-static" else kind},
    )
    for pair in pairs:
        test.assertEqual(pair["force_generation"], step + 1)
        wrench = np.asarray(pair["world_wrench"])
        test.assertGreater(np.linalg.norm(wrench[0, :3]), case["gates"]["min_pair_force"])
        force_scale = max(1, np.linalg.norm(wrench[:, :3], axis=1).sum())
        torque_scale = max(1e-3, np.linalg.norm(wrench[:, 3:], axis=1).sum())
        test.assertLessEqual(np.linalg.norm(wrench[:, :3].sum(axis=0)), 1e-5 * force_scale)
        test.assertLessEqual(np.linalg.norm(wrench[:, 3:].sum(axis=0)), 1e-5 * torque_scale)
    history = state._monolithic_coupled
    test.assertEqual(int(history.force_generation.numpy()[0]), step + 1)
    test.assertEqual(run.last_stats.variational["contact_force_generation"], step + 1)
    for i, actor in enumerate(run._rigid.actors):
        wrench = history.contact_body_force.numpy()[actor.body_indices[0]]
        np.testing.assert_allclose(
            history.contact_joint_force.numpy()[list(actor.dof_indices)], wrench, rtol=1e-12, atol=1e-12
        )
        angle = 2 * np.arctan2(
            np.linalg.norm(history.rigid_rotation.numpy()[i, :3]), abs(history.rigid_rotation.numpy()[i, 3])
        )
        test.assertGreater(angle, case["gates"]["min_rotation_rad"])
    if kind in ("chain", "R-T"):
        test.assertGreater(
            np.max(abs(state.particle_q.numpy() - run.model.particle_q.numpy())), case["gates"]["min_tet_displacement"]
        )
    matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
    if kind in ("A-R", "chain"):
        a = run._layout.indices("articulation")
        rigid = min(run._rigid.actors, key=lambda actor: run.model.body_label[actor.body_indices[0]])
        r = np.arange(rigid.dof_start, rigid.dof_start + rigid.dof_count)
        test.assertGreater(np.linalg.norm(matrix[np.ix_(a, r)]), 1e-4)
    if kind in ("R-R", "chain"):
        a, b = sorted(run._rigid.actors, key=lambda actor: run.model.body_label[actor.body_indices[0]])
        test.assertGreater(
            np.linalg.norm(matrix[a.dof_start : a.dof_start + a.dof_count, b.dof_start : b.dof_start + b.dof_count]),
            1e-4,
        )
    if kind in ("R-T", "chain"):
        r = max(run._rigid.actors, key=lambda actor: run.model.body_label[actor.body_indices[0]])
        test.assertGreater(
            np.linalg.norm(
                matrix[np.ix_(np.arange(r.dof_start, r.dof_start + r.dof_count), run._layout.indices("tet"))]
            ),
            1e-4,
        )


def case_record(run, **values):
    c = run._contact
    arrays = c.actor_contact.buffers()
    return {
        **values,
        "device": str(run.model.device),
        "layout": run._layout.identity,
        "physical_digest": run._kinematics.digest.hex(),
        "keys": c.count,
        "new_keys": c.actor_contact.count,
        "scalar_support": len(c.factor_support[1]),
        "contact_triplets": c.triplet_capacity,
        "native_contact_bytes": sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in arrays),
        "converged": run.last_stats.converged,
    }


def known_armijo_stop(test, run, kind):
    """Recognize only the retained slope-floor limitation with strict atomic rollback."""
    test.assertEqual(run._globalization, "Armijo")
    test.assertIn(
        (kind, run._contact.contact_model),
        (("A-R", "C-II"), ("R-T", "C-EE"), ("R-T", "C-IE"), ("R-T", "C-II"), ("R-static", "C-II")),
    )
    stats = run.last_stats
    test.assertEqual(stats.status, run.Status.REGULARIZATION_EXHAUSTED)
    test.assertTrue(stats.rolled_back)
    test.assertFalse(stats.converged)
    for ratio in (stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x):
        test.assertLessEqual(ratio, 1)
    blocks = stats.variational["rigid_blocks"]
    test.assertTrue(any(v["convergence_ratio_" + name] > 1 for name, v in blocks.items() if name.endswith("_r")))
    for name, values in blocks.items():
        if name.endswith("_t"):
            test.assertLessEqual(values["convergence_ratio_" + name], 1)
    trials = [t for t in stats.variational["trials"] if t.get("iteration") == stats.nonlinear_iterations]
    test.assertTrue(trials)
    for trial in trials:
        test.assertFalse(trial["accepted"])
        test.assertFalse(trial["energy_predicate"])
        test.assertTrue(trial["ResNorm"])
        test.assertLessEqual(-trial["slope_noise"], trial["acceptance_slope"])
        test.assertLessEqual(trial["acceptance_slope"], 0)
    return {
        "status": stats.status.value,
        "reason": "armijo_slope_noise",
        "rigid_blocks": dict(blocks),
        "final_trials": trials,
    }


def test_pair_matrix(test, device):
    """Exercise four isolated active pairs across laws, globalization, layout and execution."""
    test.integration_cases, test.integration_frames = [], {}
    test.integration_known_issues = []
    for kind, law, globalization in itertools.product(
        ("A-R", "R-R", "R-T", "R-static"), ("C-EE", "C-IE", "C-II"), ("Armijo", "ResNorm", "ArmijoOrResNorm")
    ):
        reference = None
        for (mode, graph), permuted in itertools.product(routes(device), (False, True)):
            with test.subTest(
                kind=kind, law=law, globalization=globalization, mode=mode, graph=graph, permuted=permuted
            ):
                run = contact_solver(
                    contact_scene(device, kind),
                    law=law,
                    globalization=globalization,
                    mode=mode,
                    graph=graph,
                    permuted=permuted,
                )
                state, control = contact_state(run, kind), run.model.control()
                frames = []
                for step in range(3):
                    previous = snapshot(state)
                    try:
                        check_contact_step(test, run, state, control, kind, step)
                    except AssertionError:
                        if run.last_stats.converged:
                            raise
                        issue = known_armijo_stop(test, run, kind)
                        assert_snapshot(test, state, previous)
                        test.integration_known_issues.append(
                            case_record(
                                run,
                                matrix="pair",
                                kind=kind,
                                law=law,
                                globalization=globalization,
                                mode=mode,
                                graph=graph,
                                permuted=permuted,
                                qualification="KNOWN_ISSUE",
                                completed_steps=len(frames),
                                failed_step=step + 1,
                                **issue,
                            )
                        )
                        break
                    frames.append(frame(run, state))
                if len(frames) != 3:
                    continue
                if reference is None:
                    reference = frames
                    test.integration_frames[f"{kind}/{law}/{globalization}"] = frames
                else:
                    for current, expected in zip(frames, reference, strict=True):
                        assert_physical(test, current, expected)
                test.integration_cases.append(
                    case_record(
                        run,
                        matrix="pair",
                        kind=kind,
                        law=law,
                        globalization=globalization,
                        mode=mode,
                        graph=graph,
                        permuted=permuted,
                        steps=3,
                    )
                )


def test_chain_matrix(test, device):
    """Keep all three force interfaces active through physical permutation and checkpoint continuation."""
    test.integration_cases, test.integration_frames = [], {}
    reference = None
    for (mode, graph), before, permuted in itertools.product(routes(device), (False, True), (False, True)):
        with test.subTest(mode=mode, graph=graph, before=before, permuted=permuted):
            model = contact_scene(device, before=before)
            options = {"mode": mode, "graph": graph, "permuted": permuted}
            run = contact_solver(model, **options)
            state, control = contact_state(run, "chain"), model.control()
            frames = []
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "midpoint.npz"
                for step in range(10):
                    check_contact_step(test, run, state, control, "chain", step)
                    frames.append(frame(run, state))
                    if step == 4:
                        run.save_checkpoint(state, checkpoint)
                for _ in range(2):
                    resumed = contact_solver(model, **options)
                    restored = contact_state(resumed, "chain")
                    resumed.load_checkpoint(restored, checkpoint)
                    for step in range(5, 10):
                        check_contact_step(test, resumed, restored, control, "chain", step)
                        assert_physical(test, frame(resumed, restored), frames[step])
            if reference is None:
                reference = frames
                test.integration_frames["chain"] = frames
            else:
                for current, expected in zip(frames, reference, strict=True):
                    assert_physical(test, current, expected)
            test.integration_cases.append(
                case_record(
                    run,
                    matrix="chain",
                    mode=mode,
                    graph=graph,
                    before=before,
                    permuted=permuted,
                    steps=10,
                    restored_steps=10,
                )
            )


def test_chain_acceptance(test, device):
    """Converge the active force chain with every existing law and acceptance rule."""
    test.integration_cases = []
    mode, graph = ("production", True) if device.is_cuda else ("diagnostic", False)
    for law, globalization in itertools.product(("C-EE", "C-IE", "C-II"), ("Armijo", "ResNorm", "ArmijoOrResNorm")):
        with test.subTest(law=law, globalization=globalization):
            run = contact_solver(contact_scene(device), law=law, globalization=globalization, mode=mode, graph=graph)
            state, control = contact_state(run, "chain"), run.model.control()
            for step in range(3):
                check_contact_step(test, run, state, control, "chain", step)
            test.integration_cases.append(
                case_record(run, matrix="chain_acceptance", law=law, globalization=globalization, steps=3)
            )


def test_contact_transactions(test, device):
    """Roll back all contact actors on query, assembly, posterior and partial final-copy failures."""
    for alias in (False, True):
        run = contact_solver(contact_scene(device), mode="production", graph=bool(device.is_cuda))
        state = contact_state(run, "chain")
        output = state if alias else contact_state(run, "chain")
        control = run.model.control()
        check_contact_step(test, run, state, control, "chain", 0)
        source, target = snapshot(state), snapshot(output)
        forces = run.get_contact_forces(state)
        previous_forces = forces.body_wrench.copy()
        for method in ("_assemble", "_finish_candidate"):
            with patch.object(run, method, side_effect=_StepFailure(run.Status.NONFINITE, "injected contact failure")):
                run.step(state, output, control, dt=0.001)
            test.assertTrue(run.last_stats.rolled_back)
            assert_snapshot(test, state, source)
            assert_snapshot(test, output, target)
            test.assertNotIn("actor_contact_pairs", run.last_stats.variational)
        with patch.object(run, "_physical_linear_error", return_value=np.full(run._layout.scalar_dof_count, np.nan)):
            run.step(state, output, control, dt=0.001)
        test.assertTrue(run.last_stats.rolled_back)
        assert_snapshot(test, state, source)
        assert_snapshot(test, output, target)
        test.assertNotIn("actor_contact_pairs", run.last_stats.variational)
        evaluate = run._contact.evaluate

        def bad_query(*args, evaluate=evaluate, run=run, **kwargs):
            evaluate(*args, **kwargs)
            run._contact.flags.fill_(1)

        with patch.object(run._contact, "evaluate", bad_query):
            run.step(state, output, control, dt=0.001)
        run._graphs.entries.clear()
        test.assertTrue(run.last_stats.rolled_back)
        assert_snapshot(test, state, source)
        assert_snapshot(test, output, target)
        original_assign = output.assign

        def partial_publish(candidate, original_assign=original_assign, run=run):
            original_assign(candidate)
            if candidate is run._accepted:
                raise RuntimeError("injected partial contact publication")

        with patch.object(output, "assign", partial_publish):
            with test.assertRaisesRegex(RuntimeError, "partial contact publication"):
                run.step(state, output, control, dt=0.001)
        assert_snapshot(test, state, source)
        assert_snapshot(test, output, target)
        test.assertNotIn("actor_contact_pairs", run.last_stats.variational)
        np.testing.assert_array_equal(forces.body_wrench, previous_forces)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "contact.npz"
            run.save_checkpoint(state, checkpoint)
            for method, args in ((run.load_checkpoint, {"path": checkpoint}), (run.reset_state, {"dt": 0.001})):
                assign_calls = 0

                def partial_load(candidate, original_assign=original_assign, output=output):
                    nonlocal assign_calls
                    original_assign(candidate)
                    assign_calls += 1
                    if assign_calls == 1:
                        output.particle_q.fill_(17)
                        output._monolithic_coupled.contact_body_force.fill_(17)
                        raise RuntimeError("injected partial contact load/reset")

                with patch.object(output, "assign", partial_load):
                    with test.assertRaisesRegex(RuntimeError, "partial contact load/reset"):
                        method(output, **args)
                assert_snapshot(test, output, target)
                np.testing.assert_array_equal(forces.body_wrench, previous_forces)
            with np.load(checkpoint) as archive:
                fields = dict(archive)
            fields["history.rigid_rotation"][0] = 0
            np.savez(checkpoint, **fields)
            with test.assertRaises(ValueError):
                run.load_checkpoint(output, checkpoint)
            assert_snapshot(test, output, target)


def test_mesh_contact_matrix(test, device):
    """Use an actual Model mesh source and bound C2 texture field on CUDA."""

    if device.is_cpu:
        with test.assertRaisesRegex(ValueError, "CUDA texture SDF"):
            contact_solver(contact_scene(device, "A-R", mesh=True))
        # Source-only mesh has no target-field requirement.
        run = contact_solver(contact_scene(device, "R-static", mesh=True))
        state = contact_state(run, "R-static")
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        return
    test.integration_cases = []
    for kind in ("A-R", "R-static"):
        model = contact_scene(device, kind, mesh=True)
        field = build_smooth_sdf(model)
        for law, (mode, graph), permuted in itertools.product(("C-EE", "C-IE", "C-II"), routes(device), (False, True)):
            with test.subTest(kind=kind, law=law, mode=mode, graph=graph, permuted=permuted):
                run = contact_solver(model, law=law, mode=mode, graph=graph, permuted=permuted, smooth_sdf=field)
                state, control = contact_state(run, kind), model.control()
                for step in range(3):
                    check_contact_step(test, run, state, control, kind, step)
                test.integration_cases.append(
                    case_record(
                        run,
                        matrix="mesh",
                        kind=kind,
                        law=law,
                        mode=mode,
                        graph=graph,
                        permuted=permuted,
                        steps=3,
                        sdf_identity=field.identity,
                    )
                )


def test_contact_precision_safety(test, device):
    """Keep experimental FP32 finite, native, atomic and precision-bound without claiming convergence."""
    test.integration_cases = []
    for kind, (mode, graph) in itertools.product(("A-R", "R-R", "R-T", "R-static"), routes(device)):
        with test.subTest(kind=kind, mode=mode, graph=graph):
            model = contact_scene(device, kind)
            run = contact_solver(model, mode=mode, graph=graph, precision="fp32")
            state = contact_state(run, kind)
            before = snapshot(state)
            run.step(state, state, model.control(), dt=0.001)
            if run.last_stats.rolled_back:
                assert_snapshot(test, state, before)
            for name, value in snapshot(state).items():
                if value.dtype.kind == "f":
                    test.assertTrue(np.isfinite(value).all(), name)
            test.assertEqual(state._monolithic_coupled.rigid_com.dtype, wp.vec3)
            run._states.validate(state)
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "fp32.npz"
                run.save_checkpoint(state, checkpoint)
                resumed = contact_solver(model, precision="fp32")
                restored = contact_state(resumed, kind)
                resumed.load_checkpoint(restored, checkpoint)
                assert_snapshot(test, restored, snapshot(state))
                wrong = contact_solver(model)
                output = contact_state(wrong, kind)
                previous = snapshot(output)
                with test.assertRaises(ValueError):
                    wrong.load_checkpoint(output, checkpoint)
                assert_snapshot(test, output, previous)
            test.integration_cases.append(
                {
                    "device": str(device),
                    "kind": kind,
                    "mode": mode,
                    "graph": graph,
                    "status": run.last_stats.status.value,
                    "converged": run.last_stats.converged,
                    "rolled_back": run.last_stats.rolled_back,
                    "qualification": "safety_only",
                }
            )


class TestSolverActorContacts(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_new_pair_step,
        test_pair_matrix,
        test_chain_matrix,
        test_chain_acceptance,
        test_contact_transactions,
        test_mesh_contact_matrix,
        test_contact_precision_safety,
    ):
        add_function_test(TestSolverActorContacts, function.__name__, function, devices=[device])

if __name__ == "__main__":
    unittest.main()
