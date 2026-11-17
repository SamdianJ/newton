# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable contact runtime contracts on each device, precision and execution mode."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_preconditioner import CablePreconditioner
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_cable_contact import cable_contact_solver, cloth_and_cable_solver, hairpin, strands
from newton.tests.test_monolithic_material_state import snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices

DT = 0.001


def _solver(points, chains, device, precision, graph, **options):
    pcg = SolverMonolithic.PCGMode.PRODUCTION if graph else SolverMonolithic.PCGMode.DIAGNOSTIC
    return cable_contact_solver(
        points, chains, device=device, precision=precision, pcg_mode=pcg, use_cuda_graph=graph, **options
    )


def _producer(solver):
    return next(p for p in solver._problem.producers if p.name == "cable_contact")


def test_capacity_overflow_rolls_back(test, device, *, precision, graph):
    """Too few candidate or support slots report CONTACT_OVERFLOW and restore every published array."""
    points, chains = strands()
    for options in ({"candidate_capacity": 0}, {"candidate_capacity": 1}, {"support_capacity": 1}):
        with test.subTest(options=options):
            solver, state = _solver(points, chains, device, precision, graph, config=options)
            before = snapshot(state)
            solver.step(state, state, solver.create_control(), dt=DT)
            test.assertEqual(solver.last_stats.status, solver.Status.CONTACT_OVERFLOW)
            test.assertTrue(solver.last_stats.rolled_back)
            overflow, dropped, flags = (int(v) for v in solver._cable_contact.pool.status.numpy())
            test.assertEqual(flags, 0)
            test.assertGreater(overflow + dropped, 0)
            for name, value in before.items():
                np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)


def test_overflow_with_cloth_contact_rolls_back(test, device, *, precision):
    """With cloth contact also enabled, a cable contact overflow still reports CONTACT_OVERFLOW and restores the state."""
    for options in ({"candidate_capacity": 1}, {"support_capacity": 1}):
        with test.subTest(options=options):
            solver, state = cloth_and_cable_solver(device=device, precision=precision, cable_contact=options)
            before = snapshot(state)
            solver.step(state, state, solver.create_control(), dt=DT)
            test.assertEqual(solver.last_stats.status, solver.Status.CONTACT_OVERFLOW)
            test.assertTrue(solver.last_stats.rolled_back)
            overflow, dropped, _ = (int(v) for v in solver._cable_contact.pool.status.numpy())
            test.assertGreater(overflow + dropped, 0)
            for name, value in before.items():
                np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)
            test.assertEqual(state._monolithic_coupled.schema_version.numpy().tolist(), [10])


def test_graph_matches_eager(test, device, *, precision):
    """Production PCG under CUDA graphs and eagerly gives the same hairpin steps bit for bit."""
    results = {}
    for graph in (False, True):
        solver, state = cable_contact_solver(
            *hairpin(),
            config={},
            device=device,
            precision=precision,
            pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
            use_cuda_graph=graph,
        )
        control = solver.create_control()
        for _ in range(3):
            solver.step(state, state, control, dt=DT)
            test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        results[graph] = snapshot(state)
        if graph:
            segments = solver._cable_contact.graphs.summary()["segments"]
            for name in ("cable_contact_query", "cable_contact_response"):
                test.assertGreater(segments[name]["replays"], 0, name)
    for name, value in results[False].items():
        if name != "history.config_digest":
            np.testing.assert_array_equal(results[True][name], value, err_msg=name)


def test_anchor_makes_the_response_translation_exact(test, device, *, precision):
    """Dyadic strands moved 8 m away give bit-identical residual, force and energy: inputs are anchor-relative."""
    edge, radius = 2.0**-7, 2.0**-9
    line = np.column_stack([edge * np.arange(12), np.zeros(12), np.zeros(12)])
    points = np.vstack([line, line + np.array([0.0, 0.0, 2 * radius - 2.0**-12])])
    chains = [range(12), range(12, 24)]
    # Dyadic step increments, so the anchor-relative sums stay exact in FP32 too.
    increments = (np.arange(24)[:, None] % 5 - 2) * np.array([[2.0**-14, -(2.0**-15), 2.0**-16]])
    results = {}
    for offset in (np.zeros(3), np.array([8.0, 4.0, 2.0])):
        solver, state = _solver(
            points + offset, chains, device, precision, False, config={"self": False}, radii=[radius, radius]
        )
        contact = solver._cable_contact
        contact.begin_step(state)
        for moved in (False, True):
            steps = wp.zeros_like(state.particle_q)
            if moved:
                steps.assign(increments.astype(state.particle_q.numpy().dtype))
            contact.evaluate(steps, DT)
            test.assertEqual(tuple(int(v) for v in contact.pool.status.numpy()), (0, 0, 0))
            arrays = (contact.pool.residual.result, contact.pool.force.result, contact.pool.energy.result)
            results.setdefault(moved, []).append([a.numpy().copy() for a in arrays])
    for moved, (near, far) in results.items():
        with test.subTest(moved=moved):
            test.assertGreater(np.abs(near[0]).max(), 0.0)
            for expected, actual in zip(near, far, strict=True):
                np.testing.assert_array_equal(actual, expected)


def test_slot_order_does_not_change_results(test, device, *, precision):
    """Shuffling the candidate slots leaves residual, force, energy and the diagnostic summary unchanged bit for bit."""
    solver, state = _solver(*hairpin(), device, precision, False, config={})
    contact = solver._cable_contact
    contact.begin_step(state)
    count = contact._candidate_count
    test.assertGreater(count, 10)
    steps = wp.zeros_like(state.particle_q)
    results = []
    for seed in (None, 1, 2):
        if seed is not None:
            pairs = contact.pairs.numpy()
            pairs[:count] = pairs[:count][np.random.default_rng(seed).permutation(count)]
            contact.pairs.assign(pairs)
        contact.evaluate(steps, DT)
        test.assertEqual(tuple(int(v) for v in contact.pool.status.numpy()), (0, 0, 0))
        results.append(
            [
                a.numpy().copy()
                for a in (
                    contact.pool.residual.result,
                    contact.pool.force.result,
                    contact.pool.energy.result,
                    contact.summary,
                )
            ]
        )
    test.assertGreater(np.abs(results[0][0]).max(), 0.0)
    for other in results[1:]:
        for expected, actual in zip(results[0], other, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_one_readback_per_evaluation(test, device):
    """A cable contact evaluation reads one packed status array back to the host."""
    solver, state = _solver(*hairpin(), device, SolverMonolithic.Precision.FP64, False, config={})
    solver.step(state, state, solver.create_control(), dt=DT)
    producer = _producer(solver)
    array_type = type(state.particle_q)
    original = array_type.numpy
    reads = []

    def counted(array):
        reads.append(array.shape)
        return original(array)

    with mock.patch.object(array_type, "numpy", counted):
        producer.evaluate(solver, state, DT, solver._complete_r, build_tangent=True)
    test.assertEqual(len(reads), 1, reads)


def test_candidates_stay_frozen_within_trials(test, device, *, precision, graph):
    """Candidates refresh only at Newton iterates, never during a line-search trial."""
    solver, state = _solver(*hairpin(), device, precision, graph, config={})
    contact = solver._cable_contact
    calls = {"trials": 0, "refreshes": 0, "refreshes_in_trial": 0, "in_trial": False}
    trial, refresh, evaluate = solver._problem.evaluate_trial, contact.refresh, contact.evaluate
    seen = {}

    def recording(steps, dt):
        # Every evaluation of one candidate generation sees the same pair list.
        pairs = contact.pairs.numpy()[: contact._candidate_count].copy()
        seen.setdefault(contact.candidate_generation, []).append(pairs)
        evaluate(steps, dt)

    def trial_evaluation(*args, **kwargs):
        calls["trials"] += 1
        calls["in_trial"] = True
        try:
            return trial(*args, **kwargs)
        finally:
            calls["in_trial"] = False

    def counted_refresh(x):
        calls["refreshes"] += 1
        calls["refreshes_in_trial"] += int(calls["in_trial"])
        return refresh(x)

    with (
        mock.patch.object(solver._problem, "evaluate_trial", trial_evaluation),
        mock.patch.object(contact, "refresh", counted_refresh),
        mock.patch.object(contact, "evaluate", recording),
    ):
        solver.step(state, state, solver.create_control(), dt=DT)
    test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
    test.assertGreater(calls["trials"], 0)
    test.assertGreater(calls["refreshes"], 1)
    test.assertEqual(calls["refreshes_in_trial"], 0)
    test.assertGreater(max(len(lists) for lists in seen.values()), 1)
    for generation, lists in seen.items():
        for pairs in lists[1:]:
            np.testing.assert_array_equal(pairs, lists[0], err_msg=str(generation))


def test_shared_vertex_keeps_one_term(test, device, *, precision, graph):
    """Through the real anchors and candidates, one of the two segments at a shared vertex keeps its term.

    A short cable crosses a straight or convexly bent target with an edge middle exactly above a target
    vertex (dyadic coordinates, so the incoming end plane is an exact tie in both precisions).
    """
    edge = 2.0**-7  # 7.8125 mm
    radius = 2.0**-9  # edge = 4 radii
    height = 2 * radius - 2.0**-12
    for degrees in (0.0, 30.0):
        bend = np.radians(degrees)
        target = [np.array([k * edge, 0.0, 0.0]) for k in range(-3, 1)]
        target += [k * edge * np.array([np.cos(bend), 0.0, -np.sin(bend)]) for k in range(1, 4)]
        # The crossing cable runs along y; its middle edge's Gauss middle point sits above the vertex at 0.
        # It comes first: only the lower cable index samples the other cable.
        crossing = [np.array([0.0, (k - 0.5) * edge, height]) for k in range(-1, 3)]
        points = np.array(crossing + target)
        chains = [range(len(crossing)), range(len(crossing), len(points))]
        solver, state = _solver(
            points, chains, device, precision, graph, config={"self": False}, radii=[radius, radius]
        )
        contact = solver._cable_contact
        contact.begin_step(state)
        contact.evaluate(wp.zeros_like(state.particle_q), DT)
        test.assertEqual(tuple(int(v) for v in contact.pool.status.numpy()), (0, 0, 0))
        tables = contact.host_tables
        samples = tables["samples"]
        middle = int(np.flatnonzero((samples["edge"] == 1) & np.isclose(samples["xi"], 0.5))[0])
        pairs = contact.pairs.numpy()[: contact._candidate_count]
        weights = contact.pool.candidates.weights.numpy()[: contact._candidate_count, 0]
        # Target segments 5 (incoming) and 6 (outgoing) meet at the target vertex at the origin.
        kept = [
            float(weights[i]) for i, (s, j, kind, _) in enumerate(pairs) if s == middle and kind == 0 and j in (5, 6)
        ]
        with test.subTest(degrees=degrees):
            test.assertEqual(len(kept), 2)
            test.assertEqual(sum(w > 0 for w in kept), 1, kept)


def test_status_selects_the_failure(test, device):
    """Overflow or dropped contacts give CONTACT_OVERFLOW (first); record flags give NONFINITE; zeros pass."""
    solver, state = _solver(*strands(), device, SolverMonolithic.Precision.FP64, False, config={})
    solver.step(state, state, solver.create_control(), dt=DT)
    producer = _producer(solver)
    contact = producer.contact
    for status, expected in (
        ((1, 0, 0), solver.Status.CONTACT_OVERFLOW),
        ((0, 1, 0), solver.Status.CONTACT_OVERFLOW),
        ((0, 0, 1), solver.Status.NONFINITE),
        ((1, 0, 1), solver.Status.CONTACT_OVERFLOW),
        ((0, 0, 0), None),
    ):

        def evaluate(steps, dt, status=status):
            contact.pool.status.assign(np.asarray(status, dtype=np.int32))

        with test.subTest(status=status), mock.patch.object(contact, "evaluate", evaluate):
            if expected is None:
                producer.evaluate(solver, state, DT, solver._complete_r, build_tangent=True)
                continue
            with test.assertRaises(_StepFailure) as raised:
                producer.evaluate(solver, state, DT, solver._complete_r, build_tangent=True)
            test.assertEqual(raised.exception.status, expected)


def test_overstretched_segment_fails_the_step(test, device):
    """A segment longer than 1.5 rest lengths fails the refresh as invalid geometry and rolls back."""
    points, chains = strands()
    solver, state = _solver(points, chains, device, SolverMonolithic.Precision.FP64, False, config={})
    stretched = state.particle_q.numpy()
    stretched[5, 0] += 0.6 * 0.007
    state.particle_q.assign(stretched)
    solver.reset_state(state, dt=DT)
    before = snapshot(state)
    solver.step(state, state, solver.create_control(), dt=DT)
    test.assertEqual(solver.last_stats.status, solver.Status.NONFINITE)
    test.assertTrue(solver.last_stats.rolled_back)
    test.assertIn("cable contact", solver.last_stats.failure_reason)
    test.assertIn("[4]", solver.last_stats.failure_reason)  # the stretch flag
    for name, value in before.items():
        np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)


def test_diagnostics_and_checkpoint(test, device, *, precision, graph):
    """Publish self-contact diagnostics in schema 9, continue a checkpoint bit for bit and reject other schemas."""
    solver, state = _solver(*hairpin(), device, precision, graph, config={})
    control = solver.create_control()
    for _ in range(2):
        solver.step(state, state, control, dt=DT)
        test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        test.assertGreaterEqual(solver.last_stats.variational["cable_used_shift_max"], 0.0)
    history = state._monolithic_coupled
    test.assertEqual(history.schema_version.numpy().tolist(), [9])
    result = solver.get_cable_contact_diagnostics(state)
    test.assertEqual(result["inter"], {"candidates": 0, "active": 0, "compressive_load_N": 0.0, "max_overlap_m": 0.0})
    pressed = result["self"]
    test.assertTrue(0 < pressed["active"] <= pressed["candidates"], pressed)
    test.assertGreater(pressed["compressive_load_N"], 0.0)
    test.assertGreater(pressed["max_overlap_m"], 0.0)
    test.assertGreater(np.abs(history.cable_contact_particle_force.numpy()).max(), 0.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cable.npz"
        solver.save_checkpoint(state, path)
        restored_solver, restored = _solver(*hairpin(), device, precision, graph, config={})
        restored_solver.load_checkpoint(restored, path)
        # A schema-6 checkpoint (no cable contact) and this schema-9 one are not interchangeable.
        plain, plain_state = _solver(*hairpin(), device, precision, graph, config=None)
        other = Path(directory) / "plain.npz"
        plain.save_checkpoint(plain_state, other)
        for owner, owner_state, wrong in ((restored_solver, restored, other), (plain, plain_state, path)):
            with test.assertRaisesRegex(ValueError, "schema version"):
                owner.load_checkpoint(owner_state, wrong)
    for key, value in snapshot(state).items():
        np.testing.assert_array_equal(snapshot(restored)[key], value, err_msg=key)
    solver.step(state, state, control, dt=DT)
    restored_solver.step(restored, restored, restored_solver.create_control(), dt=DT)
    actual = snapshot(restored)
    for key, expected in snapshot(state).items():
        # The diagnostic loads are summed in candidate identity order, so they match bit for bit too.
        np.testing.assert_array_equal(actual[key], expected, err_msg=key)


def test_failed_evaluation_publishes_no_diagnostics(test, device):
    """An evaluation that fails leaves the summary at zero rather than partial counts."""
    solver, state = _solver(*hairpin(), device, SolverMonolithic.Precision.FP64, False, config={})
    contact = solver._cable_contact
    contact.begin_step(state)
    contact.evaluate(wp.zeros_like(state.particle_q), DT)
    test.assertGreater(np.abs(contact.summary.numpy()).max(), 0.0)
    broken = wp.full(state.particle_q.shape, wp.vec3d(np.nan, 0.0, 0.0), dtype=wp.vec3d, device=device)
    contact.evaluate(broken, DT)
    test.assertNotEqual(int(contact.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(contact.summary.numpy(), 0.0)


def test_used_shift_is_reported_per_step(test, device, *, precision, graph):
    """Stats report each step's own largest cable preconditioner shift: forced to 1e-3, then the normal ladder."""
    solver, state = _solver(*hairpin(), device, precision, graph, config={})
    pre = solver._linear._cable_preconditioner
    original = CablePreconditioner.factor
    seen = []

    def factor(self, workspace, tolerance):
        original(self, workspace, tolerance)
        seen.append(float(self.used_shift.numpy().max()))

    control = solver.create_control()
    with mock.patch.object(CablePreconditioner, "factor", factor):
        for forced in (True, False):
            seen.clear()
            pre.shifts.assign(np.full(len(pre.SHIFTS), 1e-3) if forced else np.asarray(pre.SHIFTS))
            solver.step(state, state, control, dt=DT)
            test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
            reported = solver.last_stats.variational["cable_used_shift_max"]
            with test.subTest(forced=forced):
                test.assertGreater(len(seen), 0)
                test.assertEqual(reported, max(seen))
                if forced:
                    test.assertEqual(reported, float(pre.shifts.numpy()[0]))
                else:
                    # Below the forced value: the previous step's maximum was cleared.
                    test.assertLess(reported, float(np.asarray(1e-3, dtype=pre.shifts.numpy().dtype)))


def test_failed_publication_restores_the_diagnostics(test, device):
    """A step that fails after writing its output restores the previous cable contact force and summary."""
    solver, state = _solver(*hairpin(), device, SolverMonolithic.Precision.FP64, False, config={})
    control = solver.create_control()
    solver.step(state, state, control, dt=DT)
    test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
    history = state._monolithic_coupled
    test.assertGreater(np.abs(history.cable_contact_summary.numpy()).max(), 0.0)
    test.assertGreater(np.abs(history.cable_contact_particle_force.numpy()).max(), 0.0)
    before = snapshot(state)
    failure = _StepFailure(solver.Status.NONFINITE, "forced after publication")
    with mock.patch.object(solver, "_joint_force_metrics", side_effect=failure) as forced:
        solver.step(state, state, control, dt=DT)
    # The step failed after it wrote state_out, not earlier inside the iteration.
    forced.assert_called_once()
    test.assertEqual(solver.last_stats.failure_reason, "forced after publication")
    test.assertTrue(solver.last_stats.rolled_back)
    for name, value in before.items():
        np.testing.assert_array_equal(snapshot(state)[name], value, err_msg=name)


def test_cloth_and_cable_checkpoint(test, device, *, precision):
    """Schema 10 (cloth and cable contact) continues a checkpoint bit for bit; schema 9 and 10 reject each other."""
    solver, state = cloth_and_cable_solver(device=device, precision=precision)
    control = solver.create_control()
    for _ in range(2):
        solver.step(state, state, control, dt=DT)
        test.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
    test.assertEqual(state._monolithic_coupled.schema_version.numpy().tolist(), [10])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "both.npz"
        solver.save_checkpoint(state, path)
        restored_solver, restored = cloth_and_cable_solver(device=device, precision=precision)
        restored_solver.load_checkpoint(restored, path)
        cable_only, cable_state = _solver(*strands(), device, precision, False, config={})
        other = Path(directory) / "cable.npz"
        cable_only.save_checkpoint(cable_state, other)
        for owner, owner_state, wrong in ((restored_solver, restored, other), (cable_only, cable_state, path)):
            with test.assertRaisesRegex(ValueError, "schema version"):
                owner.load_checkpoint(owner_state, wrong)
    for key, value in snapshot(state).items():
        np.testing.assert_array_equal(snapshot(restored)[key], value, err_msg=key)
    solver.step(state, state, control, dt=DT)
    restored_solver.step(restored, restored, restored_solver.create_control(), dt=DT)
    actual = snapshot(restored)
    for key, expected in snapshot(state).items():
        np.testing.assert_array_equal(actual[key], expected, err_msg=key)


class TestCableContactRuntime(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = [device for device in devices if device.is_cuda]
for _test in (
    test_one_readback_per_evaluation,
    test_status_selects_the_failure,
    test_overstretched_segment_fails_the_step,
    test_failed_evaluation_publishes_no_diagnostics,
    test_failed_publication_restores_the_diagnostics,
):
    add_function_test(TestCableContactRuntime, _test.__name__, _test, devices=devices)
for _precision in SolverMonolithic.Precision:
    for _test in (
        test_slot_order_does_not_change_results,
        test_anchor_makes_the_response_translation_exact,
        test_cloth_and_cable_checkpoint,
        test_overflow_with_cloth_contact_rolls_back,
    ):
        add_function_test(
            TestCableContactRuntime,
            f"{_test.__name__}_{_precision.value}",
            _test,
            devices=devices,
            precision=_precision,
        )
    add_function_test(
        TestCableContactRuntime,
        f"test_graph_matches_eager_{_precision.value}",
        test_graph_matches_eager,
        devices=cuda_devices,
        precision=_precision,
    )
    for _graph in (False, True):
        for _test in (
            test_capacity_overflow_rolls_back,
            test_candidates_stay_frozen_within_trials,
            test_shared_vertex_keeps_one_term,
            test_diagnostics_and_checkpoint,
            test_used_shift_is_reported_per_step,
        ):
            add_function_test(
                TestCableContactRuntime,
                f"{_test.__name__}_{_precision.value}_{'graph' if _graph else 'eager'}",
                _test,
                devices=cuda_devices if _graph else devices,
                precision=_precision,
                graph=_graph,
            )


if __name__ == "__main__":
    unittest.main()
