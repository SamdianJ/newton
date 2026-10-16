# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete-candidate pooled contacts through the real solver lifecycle."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.contact import ContactObjective
from newton._src.solvers.monolithic.contact_pipeline import _PooledContactObjective
from newton._src.solvers.monolithic.linear_types import MonolithicLinearGeneration, MonolithicLinearStatus
from newton.tests.test_monolithic_contact_actor_pairs import physical
from newton.tests.test_monolithic_contact_batch import batch_matrix
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state, frame
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, independent_objective, scene, snapshot, solver
from newton.tests.test_monolithic_solver_rigid import assert_physical
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.actor_contact_oracle import fixture_terms
from scripts.monolithic_reference.compare_contact_dataflows import _check_contact, audit_contact_state, verify_contract
from scripts.monolithic_reference.contact_full_oracle import evaluate_stream


def pooled_solver(model, **options):
    """Select B2 at the existing private constructor seam, before any full contact allocation."""
    with patch.object(solver_monolithic, "ContactObjective", _PooledContactObjective):
        return contact_solver(model, **options)


def sphere_static_scene(device):
    """Keep the existing static recipe and loads while selecting its supported sphere source."""
    model = contact_scene(device, "R-static")
    shape = int(np.flatnonzero(model.shape_body.numpy() == model.body_label.index("rigid_0"))[0])
    kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
    kinds[shape], scales[shape] = 3, [0.01, 0, 0]
    model.shape_type.assign(kinds)
    model.shape_scale.assign(scales)
    return model


def check_sphere_publication(test, run, state, start):
    """Check the live accepted ledger and published endpoints against independent full mathematics."""
    c = run._contact
    stiffness, mu, velocity, width, radius = c.parameters
    full = evaluate_stream(
        run.model,
        run._actors,
        physical(state),
        start,
        c.contact_model,
        radius=radius,
        stiffness=stiffness,
        mu=mu,
        width=width,
        falloff=velocity * 0.001,
        alignment_policy=c.alignment_policy,
    )
    test.assertEqual(full["full_count"], 5952)
    test.assertEqual(c.count, full["full_count"])
    expected = full["full"]
    sparse = expected["tangent"]
    tangent = np.zeros(sparse["shape"])
    tangent[sparse["row"], sparse["column"]] = sparse["value"]
    force = run.get_contact_forces(state)
    errors = _check_contact(
        {
            "energy": c.energy.numpy().reshape(-1, 2).sum(axis=0),
            "residual": c.residual.numpy(),
            "tangent": batch_matrix(c._batch),
            "particle_force": force.particle_force,
            "body_wrench": force.body_wrench,
            "joint_force": force.joint_force,
        },
        {**expected, "tangent": tangent},
    )
    static = np.zeros_like(expected["static_wrench"])
    bodies = run.model.shape_body.numpy()
    for pair in run.last_stats.variational["actor_contact_pairs"]:
        for shape, wrench in zip((pair["source"], pair["target"]), pair["world_wrench"], strict=True):
            if bodies[shape] < 0:
                static[shape] += wrench
    np.testing.assert_allclose(static, expected["static_wrench"], rtol=1e-8, atol=1e-10)
    test.assertGreater(np.linalg.norm(force.body_wrench), 1e-4)
    test.assertEqual(force.force_generation, run.last_stats.step_generation)
    test.assertTrue(force.force_valid)
    test.assertEqual(c.pool.records.dropped.numpy()[0], 0)
    test.assertFalse(c.evaluation_diagnostics()["tainted"])
    return {"full_count": full["full_count"], "min_gap": full["min_gap"], "errors": errors}


def test_sphere_steps(test, device):
    """Advance sphere-source legacy/pool trajectories independently under original gates and all laws."""
    for graph in (False, True) if device.is_cuda else (False,):
        for law in ("C-EE", "C-IE", "C-II"):
            with test.subTest(graph=graph, law=law):
                options = {"law": law, "mode": "production", "graph": graph}
                with patch.object(solver_monolithic, "ContactObjective", ContactObjective):
                    legacy = contact_solver(sphere_static_scene(device), **options)
                run = pooled_solver(sphere_static_scene(device), **options)
                verify_contract(legacy, run)
                states = [contact_state(owner, "R-static") for owner in (legacy, run)]
                with patch.object(solver_monolithic, "ContactObjective", ContactObjective):
                    audit = contact_solver(sphere_static_scene(device), **options)
                for _ in range(2):
                    starts = [physical(state) for state in states]
                    for owner, state in zip((legacy, run), states, strict=True):
                        owner.step(state, state, owner.model.control(), dt=0.001)
                        test.assertTrue(owner.last_stats.converged, owner.last_stats)
                        test.assertFalse(owner.last_stats.rolled_back)
                        owner._states.validate(state)
                    assert_physical(test, frame(legacy, states[0]), frame(run, states[1]))
                    check_sphere_publication(test, run, states[1], starts[1])
                    result = audit_contact_state(audit, physical(states[0]), starts[0], 0.001)
                    test.assertEqual(result["full_count"], 5952)


def test_sphere_checkpoint(test, device):
    """Resume active sphere contacts into a distinct output and retain exact physical authority and forces."""
    for law in ("C-EE", "C-IE", "C-II"):
        options = {"law": law, "mode": "production", "graph": device.is_cuda}
        run = pooled_solver(sphere_static_scene(device), **options)
        state = contact_state(run, "R-static")
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sphere.npz"
            run.save_checkpoint(state, path)
            resumed = pooled_solver(sphere_static_scene(device), **options)
            restored = resumed.create_state(dt=0.001)
            resumed.load_checkpoint(restored, path)
            assert_snapshot(test, restored, snapshot(state))
            start = physical(state)
            run.step(state, state, run.model.control(), dt=0.001)
            output = resumed.create_state(dt=0.001)
            resumed.step(restored, output, resumed.model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            test.assertTrue(resumed.last_stats.converged, resumed.last_stats)
            assert_snapshot(test, output, snapshot(state))
            check_sphere_publication(test, resumed, output, start)


def test_pooled_steps(test, device):
    """Converge active mixed-actor steps with complete admission and final pool force publication."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for mode, graph in modes:
        for law in ("C-EE", "C-IE", "C-II"):
            run = pooled_solver(contact_scene(device, "chain"), permuted=True, law=law, mode=mode, graph=graph)
            c = run._contact
            test.assertFalse(hasattr(c, "factor_support"))
            test.assertIsNotNone(run._linear._fixed_bsr)
            s = contact_state(run, "chain")
            for _ in range(2):
                run.step(s, s, run.model.control(), dt=0.001)
                test.assertTrue(run.last_stats.converged, run.last_stats)
                test.assertFalse(run.last_stats.rolled_back)
                test.assertEqual(c.pool.records.dropped.numpy()[0], 0)
                force = run.get_contact_forces(s)
                expected = c.pool.force.result.numpy()
                n = 3 * run.model.particle_count
                np.testing.assert_array_equal(force.particle_force.ravel(), expected[:n])
                np.testing.assert_array_equal(force.body_wrench.ravel(), expected[n:])
                rows = np.asarray(run._actors.maps["joint_dof_to_global"])
                residual = c.pool.residual.result.numpy()
                valid = rows >= 0
                np.testing.assert_array_equal(force.joint_force[valid], -residual[rows[valid]])
                test.assertTrue(force.force_valid)
                test.assertEqual(force.force_generation, run.last_stats.step_generation)
                test.assertEqual(run._generation.contact_generation, c.evaluation_generation)
                test.assertEqual(
                    run.last_stats.variational["contact_pool"]["evaluation_generation"], c.evaluation_generation
                )
                test.assertFalse(run.last_stats.variational["contact_pool"]["tainted"])


def test_pooled_same_state(test, device):
    """Compare actual post-step complete contact E/r/B with an independently enumerated mathematical oracle."""
    for kind in ("A-R", "R-R", "R-T", "R-static", "chain"):
        for law in ("C-EE", "C-IE", "C-II"):
            run = pooled_solver(contact_scene(device, kind, before=True), permuted=True, law=law, mode="production")
            s = contact_state(run, kind)
            run.step(s, s, run.model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            expected_e, expected_r, expected_b = fixture_terms(
                run.model, run._actors, physical(s), physical(run._original), law
            )
            c = run._contact
            np.testing.assert_allclose(c.energy.numpy(), expected_e, rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(c.residual.numpy(), expected_r, rtol=1e-8, atol=1e-10)
            test.assertLess(
                np.linalg.norm(batch_matrix(c._batch) - expected_b) / max(1, np.linalg.norm(expected_b)), 1e-5
            )
            forces = run.get_contact_forces(s)
            total = forces.particle_force.sum(axis=0) + forces.body_wrench[:, :3].sum(axis=0)
            centers = np.zeros((run.model.body_count, 3))
            centers[list(run._articulation_actor.body_indices)] = run._kinematics.com.numpy()
            for i, actor in enumerate(run._rigid.actors):
                centers[actor.body_indices[0]] = s._monolithic_coupled.rigid_com.numpy()[i]
            torque = np.cross(s.particle_q.numpy(), forces.particle_force).sum(axis=0)
            torque += (forces.body_wrench[:, 3:] + np.cross(centers, forces.body_wrench[:, :3])).sum(axis=0)
            bodies = run.model.shape_body.numpy()
            for pair in run.last_stats.variational["actor_contact_pairs"]:
                for end, shape in enumerate((pair["source"], pair["target"])):
                    if shape >= 0 and bodies[shape] < 0:
                        total += np.asarray(pair["world_wrench"])[end, :3]
                        torque += np.asarray(pair["world_wrench"])[end, 3:]
            test.assertLess(np.linalg.norm(total), 1e-8)
            test.assertLess(np.linalg.norm(torque), 1e-8)


def test_pooled_assembly_generation(test, device):
    """Reject duplicate or stale contact appends without altering the new assembly."""
    run = pooled_solver(contact_scene(device, "chain"))
    s = contact_state(run, "chain")
    run.step(s, s, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    c = run._contact
    g = MonolithicLinearGeneration(1, 10, c.evaluation_generation, run._assembly_sequence + 1)
    a = run._linear.begin_assembly(g)
    with patch("newton._src.solvers.monolithic.contact_pipeline._CONTACT_WORKERS", 7):
        c.assemble_tangent(a)
    test.assertGreater(int(c.pool.records.triplet_count.numpy()[0]), 7)
    count = a.global_scalar_triplets.count.numpy().copy()
    with test.assertRaisesRegex(ValueError, "once"):
        c.assemble_tangent(a)
    np.testing.assert_array_equal(a.global_scalar_triplets.count.numpy(), count)
    run._linear.finalize_assembly(generation=g)
    next_g = MonolithicLinearGeneration(1, 11, c.evaluation_generation - 1, run._assembly_sequence + 2)
    a = run._linear.begin_assembly(next_g)
    count = a.global_scalar_triplets.count.numpy().copy()
    with test.assertRaisesRegex(ValueError, "evaluation generation"):
        c.assemble_tangent(a)
    np.testing.assert_array_equal(a.global_scalar_triplets.count.numpy(), count)


def test_pooled_capacity_and_failure(test, device):
    """Continue after complete capacity drops, while hard matrix errors roll back all published state."""
    run = pooled_solver(contact_scene(device, "A-R"), contact_factor_capacity=0)
    s = contact_state(run, "A-R")
    with test.assertWarnsRegex(RuntimeWarning, "discarded.*complete contacts"):
        run.step(s, s, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    detail = run.last_stats.variational["contact_pool"]
    test.assertTrue(detail["tainted"])
    test.assertGreater(detail["dropped"], 0)
    test.assertEqual(detail["admitted"], 0)
    for values in (run.get_contact_forces(s).particle_force, run.get_contact_forces(s).body_wrench):
        test.assertFalse(values.any())
    run = pooled_solver(contact_scene(device, "chain"), publish_policy="last_accepted")
    s = contact_state(run, "chain")
    before = snapshot(s)
    builder = run._linear._fixed_bsr[0]
    original = builder.build

    def bad_matrix(triplets, **options):
        original(triplets, **options)
        triplets.status.fill_(int(MonolithicLinearStatus.NONFINITE_CONTRIBUTION))

    with patch.object(builder, "build", bad_matrix):
        run.step(s, s, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    test.assertFalse(run.last_stats.converged)
    assert_snapshot(test, s, before)


def test_pooled_checkpoint(test, device):
    """Restore schema4 physical authority and pool identity, rejecting legacy identity without partial loads."""
    model = contact_scene(device, "chain")
    run = pooled_solver(model, mode="production", graph=device.is_cuda)
    s = contact_state(run, "chain")
    run.step(s, s, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        run.save_checkpoint(s, path)
        resumed = pooled_solver(model, mode="production", graph=device.is_cuda)
        restored = resumed.create_state(dt=0.001)
        resumed.load_checkpoint(restored, path)
        assert_snapshot(test, restored, snapshot(s))
        with patch.object(solver_monolithic, "ContactObjective", ContactObjective):
            legacy = contact_solver(model)
        target = legacy.create_state(dt=0.001)
        before = snapshot(target)
        with test.assertRaises(ValueError):
            legacy.load_checkpoint(target, path)
        assert_snapshot(test, target, before)
        run.step(s, s, model.control(), dt=0.001)
        resumed.step(restored, restored, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertTrue(resumed.last_stats.converged, resumed.last_stats)
        assert_snapshot(test, restored, snapshot(s))


def test_pooled_output_rebind(test, device):
    """Consume rebound pooled outputs and rebind result graphs before combining or publishing forces."""
    run = pooled_solver(contact_scene(device, "chain"), mode="production", graph=device.is_cuda)
    s = contact_state(run, "chain")
    run.step(s, s, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    c = run._contact
    expected = tuple(owner.result.numpy().copy() for owner in (c.pool.energy, c.pool.residual, c.pool.force))
    for name in ("energy", "residual", "force"):
        owner = getattr(c.pool, name)
        retired = owner.result
        owner.result = wp.zeros_like(retired)
        retired.fill_(np.nan)
        run._evaluate(run._accepted, 0.001)
        test.assertTrue(np.isfinite(c.residual.numpy()).all())
        for output, values in zip((c.pool.energy, c.pool.residual, c.pool.force), expected, strict=True):
            np.testing.assert_array_equal(output.result.numpy(), values)
        np.testing.assert_array_equal(c.residual.numpy(), expected[1])
        np.testing.assert_array_equal(c.energy.numpy(), expected[0][:2])
        run._assemble(run._metrics["nonlinear_iterations"])
        run._finish_candidate(run._accepted, 0.001)
        force = run.get_contact_forces(run._accepted)
        np.testing.assert_array_equal(force.particle_force.ravel(), expected[2][: 3 * run.model.particle_count])
    retired = c._batch.flags
    c._batch.flags = wp.zeros_like(retired)
    retired.fill_(1)
    run._evaluate(run._accepted, 0.001)
    test.assertEqual(c.flags.numpy()[0], 0)
    if device.is_cuda:
        test.assertGreaterEqual(run._graphs.captures["coupled_objective_result"], 5)


def test_pooled_device_energy(test, device):
    """Preserve legacy A+T native objective gates in device iteration and graph/plain energy packets."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for law in ("C-EE", "C-IE", "C-II"):
        for mode, graph in modes:
            with patch.object(solver_monolithic, "ContactObjective", _PooledContactObjective):
                run = solver(scene(device, fixed=True), law=law, mode=mode, graph=graph)
            s = run.create_state(dt=0.001)
            run.step(s, s, run.model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            expected_e, expected_r, expected_b = independent_objective(
                run, s.joint_q.numpy(), s.particle_q.numpy(), 0.001
            )
            energy, residual, absolute = run._evaluate(s, 0.001, device_result=True)
            host_e, host_r, host_abs = run._evaluate(s, 0.001, device_result=False)
            test.assertLessEqual(abs(energy - expected_e), 5.820766091346741e-11 / 8)
            np.testing.assert_allclose(residual.numpy(), expected_r, rtol=0, atol=2.0935893019213836e-9 / 8)
            np.testing.assert_array_equal(residual.numpy(), host_r)
            test.assertLessEqual(abs(energy - host_e), 5.820766091346741e-11 / 8)
            test.assertLessEqual(abs(absolute - host_abs), 5.820766091346741e-11 / 8)
            run._assemble(run._metrics["nonlinear_iterations"])
            np.testing.assert_allclose(
                run._linear.densify_for_test(generation=run._generation).raw_matrix, expected_b, rtol=2e-5, atol=2e-3
            )


def test_pooled_rejected_trial(test, device):
    """Recompute accepted E/r/B and force after a nonfinite trial and an eligible Stopped publication."""
    run = pooled_solver(
        contact_scene(device, "chain"), mode="production", graph=device.is_cuda, publish_policy="last_accepted"
    )
    before = contact_state(run, "chain")
    after = run._states.copy(before)
    original = solver_monolithic.iterate
    accepted = {}
    frozen = {}
    contact = {}

    def stop(current, dt):
        original(current, dt)
        test.assertTrue(any(t.get("accepted") for t in current._variational["trials"]))
        accepted.update(snapshot(current._accepted))
        contact.update(
            {
                name: getattr(current._contact.pool, name).result.numpy().copy()
                for name in ("energy", "residual", "force")
            }
        )
        frozen.update(
            {
                name: getattr(current._contact._batch.start, name).numpy().copy()
                for name in ("y", "gradient", "soft_normal", "distance")
            }
        )
        current._trial.assign(current._accepted)
        current._trial.particle_q.fill_(wp.vec3d(np.nan))
        test.assertIsNone(current._trial_evaluation(dt))
        assert_snapshot(test, current._accepted, accepted)
        raise solver_monolithic._StepFailure(
            current.Status.NONLINEAR_MAX_ITERATIONS, "stop after rejected nonfinite trial"
        )

    with patch.object(solver_monolithic, "iterate", stop):
        run.step(before, after, run.model.control(), dt=0.001)
    test.assertFalse(run.last_stats.converged)
    test.assertFalse(run.last_stats.rolled_back, run.last_stats)
    test.assertEqual(run.last_stats.variational["publication_status"], "STOPPED_ACCEPTED")
    np.testing.assert_array_equal(after.particle_q.numpy(), accepted["particle_q"])
    coordinates = run._kinematics.coord_map.numpy()
    np.testing.assert_array_equal(after.joint_q.numpy()[coordinates], accepted["joint_q"][coordinates])
    # R coordinates are recovered from the accepted COM/rotation at publication.
    for name in ("rigid_com", "rigid_rotation"):
        np.testing.assert_array_equal(getattr(after._monolithic_coupled, name).numpy(), accepted["history." + name])
    for name, values in contact.items():
        np.testing.assert_array_equal(getattr(run._contact.pool, name).result.numpy(), values)
    for name, values in frozen.items():
        np.testing.assert_array_equal(getattr(run._contact._batch.start, name).numpy(), values)
    forces = run.get_contact_forces(after)
    saved = forces.particle_force.copy()
    test.assertTrue(forces.force_valid)
    run.step(after, after, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    np.testing.assert_array_equal(forces.particle_force, saved)


class TestSolverContactPool(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_sphere_steps,
        test_sphere_checkpoint,
        test_pooled_steps,
        test_pooled_same_state,
        test_pooled_assembly_generation,
        test_pooled_capacity_and_failure,
        test_pooled_checkpoint,
        test_pooled_output_rebind,
        test_pooled_device_energy,
        test_pooled_rejected_trial,
    ):
        add_function_test(TestSolverContactPool, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
