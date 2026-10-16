# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard spatial candidates at the real Newton lifecycle boundaries."""

import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import contact_pipeline, solver_monolithic
from newton._src.solvers.monolithic.contact_pipeline import _SpatialContactObjective
from newton.tests.test_monolithic_contact_batch import batch_matrix, native_inputs
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state, frame
from newton.tests.test_monolithic_solver_contact_pool import pooled_solver
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.test_monolithic_solver_rigid import assert_physical
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.workload_scene import build_scene


def spatial_solver(model, *, capacity=4096, margin=0.0, **options):
    """Select B3 at the existing offline constructor seam with explicit experiment parameters."""
    factory = partial(_SpatialContactObjective, candidate_capacity=capacity, law_margin=margin)
    with patch.object(solver_monolithic, "ContactObjective", factory):
        return contact_solver(model, **options)


def test_spatial_default(test, device):
    """Match ordinary construction to an explicit spatial recipe and restore its checkpoint."""
    model = contact_scene(device, "chain")
    options = {"mode": "production", "graph": device.is_cuda}
    run = contact_solver(model, **options)
    test.assertTrue(run._contact_spatial)
    test.assertEqual(run._contact.candidate_capacity, run._contact.count)
    test.assertEqual(run._contact.law_margin, 0.0)
    explicit = spatial_solver(model, capacity=run._contact.count, **options)
    test.assertEqual(run._kinematics.digest, explicit._kinematics.digest)
    states = [contact_state(owner, "chain") for owner in (run, explicit)]
    for _ in range(2):
        for owner, state in zip((run, explicit), states, strict=True):
            owner.step(state, state, model.control(), dt=0.001)
            test.assertTrue(owner.last_stats.converged, owner.last_stats)
            test.assertFalse(owner.last_stats.variational["contact_spatial"]["tainted"])
        assert_snapshot(test, states[1], snapshot(states[0]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        run.save_checkpoint(states[0], path)
        restored = contact_solver(model, **options)
        target = restored.create_state(dt=0.001)
        restored.load_checkpoint(target, path)
        assert_snapshot(test, target, snapshot(states[0]))


def test_spatial_rounds(test, device):
    """Refresh once per Newton check, freeze trials/final, and preserve step-start authority."""
    modes = [("diagnostic", False), ("production", False)]
    if device.is_cuda:
        modes.append(("production", True))
    for law in ("C-EE", "C-IE", "C-II"):
        for mode, graph in modes:
            options = {"law": law, "mode": mode, "graph": graph, "permuted": True}
            run = spatial_solver(contact_scene(device, "chain"), **options)
            full = pooled_solver(contact_scene(device, "chain"), **options)
            states = [contact_state(owner, "chain") for owner in (run, full)]
            for _ in range(2):
                start = states[0].particle_q.numpy().copy()
                for owner, state in zip((run, full), states, strict=True):
                    owner.step(state, state, owner.model.control(), dt=0.001)
                    test.assertTrue(owner.last_stats.converged, owner.last_stats)
                test.assertEqual(
                    run.last_stats.variational["contact_spatial"]["refreshes"], run.last_stats.nonlinear_iterations + 1
                )
                np.testing.assert_array_equal(run._contact._spatial.x0.numpy(), start)
                test.assertFalse(run.last_stats.variational["contact_spatial"]["tainted"])
                assert_physical(test, frame(full, states[1]), frame(run, states[0]))
                np.testing.assert_allclose(run._reference_residual, full._reference_residual, rtol=1e-8, atol=1e-10)
                expected = run._contact.pool.force.result.numpy()
                force = run.get_contact_forces(states[0])
                n = 3 * run.model.particle_count
                np.testing.assert_array_equal(force.particle_force.ravel(), expected[:n])
                np.testing.assert_array_equal(force.body_wrench.ravel(), expected[n:])
                test.assertEqual(run._generation.contact_generation, run._contact.evaluation_generation)


def test_spatial_retry_freeze(test, device):
    """Retain one list through shrink and lambda retry, including the final budget check."""
    run = spatial_solver(
        contact_scene(device, "chain"),
        mode="production",
        graph=device.is_cuda,
        line_search_max_iterations=2,
        newton_max_iterations=1,
    )
    run._config = replace(run._config, regularization_values=(0.0, 0.01))
    state = contact_state(run, "chain")
    before = snapshot(state)
    calls, queries = [], []
    original_refresh, original_evaluate = run._refresh_candidates, run._evaluate
    original_acceptance = solver_monolithic.iterate.__globals__["acceptance"]

    def refresh():
        original_refresh()
        s = run._contact._spatial
        queries.append((run._contact.candidate_generation, s.identities.numpy()[: int(s.count.numpy()[0])].copy()))

    def evaluate(candidate, dt, **options):
        result = original_evaluate(candidate, dt, **options)
        if not run._contact._reference_mode:
            s = run._contact._spatial
            test.assertEqual(run._contact.candidate_generation, queries[-1][0])
            np.testing.assert_array_equal(s.identities.numpy()[: int(s.count.numpy()[0])], queries[-1][1])
            calls.append((candidate is run._accepted, result[0]))
        return result

    accepted_calls = 0

    def reject_first_retry(*args, **options):
        nonlocal accepted_calls
        accepted_calls += 1
        if accepted_calls <= 2:
            return False, False, False, False
        return original_acceptance(*args, **options)

    with (
        patch.object(run, "_refresh_candidates", refresh),
        patch.object(run, "_evaluate", evaluate),
        patch.dict(solver_monolithic.iterate.__globals__, acceptance=reject_first_retry),
    ):
        run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(any(t.get("retry", 0) > 0 for t in run.last_stats.variational["trials"]))
    test.assertEqual(len(queries), 2)  # Round zero and the budget's final check.
    test.assertEqual(sum(accepted for accepted, _ in calls), 2)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, state, before)


def test_spatial_miss_and_reference(test, device):
    """Detect a frozen-list miss without automatic handling and discover it using the original reference."""
    run = spatial_solver(contact_scene(device, "A-R"), law="C-II", mode="production", graph=device.is_cuda)
    state = contact_state(run, "A-R")
    touching = run._states.copy(state)
    state._monolithic_coupled.rigid_com.fill_(wp.vec3d(10))
    run._rigid.publish(state)
    run._states.bind(state, dt=0.001)
    before = snapshot(state)

    def probe(current, dt):
        c = current._contact
        current._initial_reference(dt)
        frozen = {name: getattr(c._batch.start, name).numpy().copy() for name in ("y", "gradient", "distance")}
        # Start and accepted can be far while this deliberately unbounded trial
        # touches. This is an accepted approximation, not an automatic reject.
        current._refresh_candidates()
        test.assertEqual(c._spatial.count.numpy()[0], 0)
        current._trial.assign(touching)
        generation = c.candidate_generation
        trial = current._trial_evaluation(dt)
        test.assertIsNotNone(trial)
        test.assertEqual(c.candidate_generation, generation)
        test.assertEqual(c.pool.records.count.numpy()[0], 0)
        inputs, options = native_inputs(current, current._trial)
        c._batch.evaluate(*inputs, dt=dt, **options)
        test.assertGreater(np.linalg.norm(c.residual.numpy()), 0)
        test.assertGreater(np.linalg.norm(batch_matrix(c._batch)), 0)
        # A new round discovers samples; their yn/gn still use the complete
        # original physical-start reference, never the discovery state.
        current._accepted.assign(current._trial)
        current._refresh_candidates()
        test.assertGreater(c._spatial.count.numpy()[0], 0)
        current._evaluate(current._accepted, dt)
        current._assemble(1)
        for name, value in frozen.items():
            np.testing.assert_array_equal(getattr(c._batch.start, name).numpy(), value)
        np.testing.assert_array_equal(c._spatial.com0.numpy(), np.full((*c._spatial.com0.shape, 3), 10))
        return current.Status.NONLINEAR_MAX_ITERATIONS

    with patch.object(solver_monolithic, "iterate", probe):
        run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.rolled_back)
    assert_snapshot(test, state, before)


def test_spatial_hard_and_stopped(test, device):
    """Roll back query hard errors and publish only the accepted ledger after rejected trial scratch."""
    for field in ("particle_q", "rigid_com"):
        run = spatial_solver(contact_scene(device, "chain"), mode="production", graph=device.is_cuda)
        state = contact_state(run, "chain")
        before = snapshot(state)

        def invalid(current, dt, field=field):
            current._initial_reference(dt)
            owner = current._accepted if field == "particle_q" else current._accepted._monolithic_coupled
            getattr(owner, field).fill_(wp.vec3d(np.nan))
            current._refresh_candidates()

        with patch.object(solver_monolithic, "iterate", invalid):
            run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.rolled_back)
        test.assertEqual(run.last_stats.status, run.Status.NONFINITE)
        test.assertEqual(run.last_stats.variational["contact_spatial"]["capacity_dropped"], 0)
        assert_snapshot(test, state, before)

    run = spatial_solver(
        contact_scene(device, "chain"), mode="production", graph=device.is_cuda, publish_policy="last_accepted"
    )
    state = contact_state(run, "chain")
    original = solver_monolithic.iterate
    ledger = {}

    def stopped(current, dt):
        original(current, dt)
        c, s = current._contact, current._contact._spatial
        generation = c.candidate_generation
        identities = s.identities.numpy().copy()
        ledger.update({n: getattr(c.pool, n).result.numpy().copy() for n in ("energy", "residual", "force")})
        current._trial.assign(current._accepted)
        current._trial.particle_q.fill_(wp.vec3d(np.nan))
        test.assertIsNone(current._trial_evaluation(dt))
        test.assertEqual(c.candidate_generation, generation)
        np.testing.assert_array_equal(s.identities.numpy(), identities)
        raise solver_monolithic._StepFailure(current.Status.NONLINEAR_MAX_ITERATIONS, "controlled stopped publication")

    with patch.object(solver_monolithic, "iterate", stopped):
        run.step(state, state, run.model.control(), dt=0.001)
    test.assertFalse(run.last_stats.rolled_back)
    test.assertFalse(run.last_stats.converged)
    test.assertEqual(run.last_stats.variational["publication_status"], "STOPPED_ACCEPTED")
    for n, values in ledger.items():
        np.testing.assert_array_equal(getattr(run._contact.pool, n).result.numpy(), values)


def test_spatial_reference_batches(test, device):
    """Retain full reference across batch tails in native FP64/FP32 even with both budgets zero."""
    original = contact_pipeline._ContactBatch

    def small_batch(*args, **options):
        options["batch_capacity"] = min(7, args[0].count)
        return original(*args, **options)

    for precision in ("fp64", "fp32"):
        references = []
        for factory in (pooled_solver, partial(spatial_solver, capacity=0, contact_factor_capacity=0)):
            with patch.object(contact_pipeline, "_ContactBatch", small_batch):
                run = factory(
                    contact_scene(device, "chain"),
                    precision=precision,
                    law="C-II",
                    mode="production",
                    graph=device.is_cuda,
                )
            state = contact_state(run, "chain")
            test.assertGreater(run._contact.count, run._contact._batch.batch_capacity)

            def probe(current, dt, results=references):
                results.append(current._initial_reference(dt).copy())
                return current.Status.NONLINEAR_MAX_ITERATIONS

            with patch.object(solver_monolithic, "iterate", probe):
                run.step(state, state, run.model.control(), dt=0.001)
        test.assertEqual(references[0].dtype, np.float64 if precision == "fp64" else np.float32)
        np.testing.assert_allclose(
            references[1],
            references[0],
            rtol=1e-8 if precision == "fp64" else 1e-3,
            atol=1e-10 if precision == "fp64" else 2e-5,
        )


def test_spatial_device_iteration(test, device):
    """Keep the full native reference and original energy gates in the A+T device loop."""
    runs, states = [], []
    for algorithm in ("pooled", "spatial"):
        model, run, state, _ = build_scene(
            "tiny-contact",
            device=device,
            graph=device.is_cuda,
            contact_dataflow=algorithm,
            spatial={"candidate_capacity": 256, "law_margin": 0.0} if algorithm == "spatial" else None,
        )
        test.assertTrue(run._device_iteration)
        run.step(state, state, model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        runs.append(run)
        states.append(state)
    np.testing.assert_allclose(
        runs[0]._reference_residual.numpy(), runs[1]._reference_residual.numpy(), rtol=1e-8, atol=1e-10
    )
    assert_physical(test, frame(runs[0], states[0]), frame(runs[1], states[1]))
    test.assertEqual(
        runs[1].last_stats.variational["contact_spatial"]["refreshes"], runs[1].last_stats.nonlinear_iterations + 1
    )


def test_spatial_capacity_reference(test, device):
    """Keep a complete native reference under query/pool overflow and distinguish both drop sources."""
    references = []
    for candidate_capacity, factor_capacity in ((4096, None), (0, None), (4096, 0)):
        run = spatial_solver(
            contact_scene(device, "A-R"),
            capacity=candidate_capacity,
            contact_factor_capacity=factor_capacity,
            mode="production",
            graph=device.is_cuda,
        )
        state = contact_state(run, "A-R")

        def probe(current, dt):
            references.append(current._initial_reference(dt).copy())
            current._refresh_candidates()
            current._evaluate(current._accepted, dt)
            return current.Status.NONLINEAR_MAX_ITERATIONS

        with patch.object(solver_monolithic, "iterate", probe):
            if candidate_capacity == 0 or factor_capacity == 0:
                with test.assertWarnsRegex(RuntimeWarning, "capacity exceeded"):
                    run.step(state, state, run.model.control(), dt=0.001)
            else:
                run.step(state, state, run.model.control(), dt=0.001)
        spatial, pool = run.last_stats.variational["contact_spatial"], run.last_stats.variational["contact_pool"]
        test.assertEqual(spatial["tainted"], candidate_capacity == 0)
        test.assertEqual(pool["dropped"] > 0, factor_capacity == 0)
        test.assertEqual(pool["tainted"], candidate_capacity == 0 or factor_capacity == 0)
    for r in references[1:]:
        np.testing.assert_array_equal(r, references[0])


def test_spatial_checkpoint(test, device):
    """Resume exact physical authority, bind spatial strategy, and reuse graphs after content changes."""
    model = contact_scene(device, "chain")
    options = {"mode": "production", "graph": device.is_cuda}
    run = spatial_solver(model, **options)
    state = contact_state(run, "chain")
    run.step(state, state, model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "spatial.npz"
        run.save_checkpoint(state, path)
        resumed = spatial_solver(model, **options)
        restored = resumed.create_state(dt=0.001)
        resumed.load_checkpoint(restored, path)
        for other in (pooled_solver(model, **options), spatial_solver(model, margin=0.001, **options)):
            target = other.create_state(dt=0.001)
            before = snapshot(target)
            with test.assertRaises(ValueError):
                other.load_checkpoint(target, path)
            assert_snapshot(test, target, before)
        for owner, s in ((run, state), (resumed, restored)):
            owner.step(s, s, model.control(), dt=0.001)
            test.assertTrue(owner.last_stats.converged, owner.last_stats)
        assert_snapshot(test, restored, snapshot(state))
        if device.is_cuda:
            captures = run._contact.query_graphs.captures["contact_candidates"]
            for _ in range(2):
                run._refresh_candidates()
            test.assertEqual(run._contact.query_graphs.captures["contact_candidates"], captures)


def test_spatial_graph_storage(test, device):
    """Reject a replaced BVH even when an existing query graph would bypass Python update validation."""
    run = spatial_solver(contact_scene(device, "chain"), mode="production", graph=device.is_cuda)
    state = contact_state(run, "chain")
    run.step(state, state, run.model.control(), dt=0.001)
    run._refresh_candidates()
    c, k = run._contact, run._kinematics
    retired = c._spatial.query.bvh
    c._spatial.query.bvh = None
    try:
        with test.assertRaisesRegex(ValueError, "storage changed"):
            c.refresh(
                k.gather_q(run._accepted),
                run._accepted.particle_q,
                rigid_com=run._accepted._monolithic_coupled.rigid_com,
            )
    finally:
        c._spatial.query.bvh = retired


class TestSolverContactSpatial(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_spatial_default,
        test_spatial_rounds,
        test_spatial_retry_freeze,
        test_spatial_miss_and_reference,
        test_spatial_capacity_reference,
        test_spatial_checkpoint,
        test_spatial_hard_and_stopped,
        test_spatial_reference_batches,
        test_spatial_graph_storage,
        test_spatial_device_iteration,
    ):
        add_function_test(TestSolverContactSpatial, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
