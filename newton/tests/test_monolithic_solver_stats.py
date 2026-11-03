# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check lifetime and export of final nonlinear diagnostic snapshots."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.examples.softbody.monolithic_recording import save, serial
from newton.tests.test_monolithic_material_state import scene
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def vectors(stats):
    """Enumerate present full-vector diagnostics without serializing them."""
    detail = stats.variational
    if "residual" in detail:
        yield detail["residual"]
    for name in ("cloth_reactions", "cable_reactions"):
        yield from detail.get(name, ())


def check_snapshot(test, stats, run):
    """Compare immutable native-precision diagnostics with their device sources."""
    sources = [run._stats_residual, *(c.raw for c in run._cloths), *(c.raw for c in run._cables)]
    arrays = list(vectors(stats))
    test.assertEqual(len(arrays), len(sources))
    for value, source in zip(arrays, sources, strict=True):
        test.assertIsInstance(value, np.ndarray)
        test.assertEqual(value.dtype, run._numpy_dtype)
        expected = source.numpy()
        np.testing.assert_array_equal(value, expected)
        test.assertFalse(np.shares_memory(value, expected))
        if value.size:
            with test.assertRaises(ValueError):
                value.flat[0] = 0


def test_final_snapshots(test, device):
    """Retain published vectors across later steps and solver scratch reuse."""
    for precision in ("fp32", "fp64"):
        for cloth, cables in ((0, 0), (1, 0), (0, 1), (2, 2)):
            with test.subTest(precision=precision, cloth=cloth, cables=cables):
                _, run = scene(
                    cloth=cloth,
                    cables=cables,
                    mechanical=True,
                    precision=precision,
                    device=device,
                    pcg_mode="production",
                    use_cuda_graph=device.is_cuda,
                )
                state, control = run.create_state(dt=0.001), run.create_control()
                control.joint_f.fill_(0.01)
                load = np.zeros((run.model.particle_count, 3), dtype=np.float32)
                load[:, 2] = 0.01
                state.particle_f.assign(load)
                run.step(state, state, control, dt=0.001)
                old = run.last_stats
                test.assertTrue(old.converged, old)
                check_snapshot(test, old, run)
                before = serial(old)
                control.joint_f.fill_(0.02)
                run.step(state, state, control, dt=0.001)
                test.assertTrue(run.last_stats.converged, run.last_stats)
                test.assertEqual(run.last_stats.step_generation, old.step_generation + 1)
                check_snapshot(test, run.last_stats, run)
                for value, other in zip(vectors(old), vectors(run.last_stats), strict=True):
                    test.assertFalse(np.shares_memory(value, other))
                run._stats_residual.fill_(17)
                for block in (*run._cloths, *run._cables):
                    block.raw.fill_(23)
                test.assertEqual(serial(old), before)


def test_stopped_snapshots(test, device):
    """Keep final candidate diagnostics on an eligible stopped publication."""
    for precision in ("fp32", "fp64"):
        _, run = scene(
            precision=precision,
            device=device,
            publish_policy="current_iterate",
            pcg_mode="production",
            use_cuda_graph=device.is_cuda,
        )
        state, control = run.create_state(dt=0.001), run.create_control()
        iterate = run._iterate_articulation

        def stop(dt, iterate=iterate, run=run):
            iterate(dt)
            return run.Status.NONLINEAR_MAX_ITERATIONS

        with patch.object(run, "_iterate_articulation", stop):
            run.step(state, state, control, dt=0.001)
        stats = run.last_stats
        test.assertFalse(stats.converged)
        test.assertFalse(stats.rolled_back)
        test.assertEqual(stats.variational["publication_status"], "STOPPED_ACCEPTED")
        test.assertEqual(int(state._monolithic_coupled.force_generation.numpy()[0]), stats.step_generation)
        check_snapshot(test, stats, run)


def test_failure_snapshots(test, device):
    """Preserve old Stats and label attempted diagnostics after early or late rollback."""
    for precision in ("fp32", "fp64"):
        _, run = scene(precision=precision, device=device, pcg_mode="production", use_cuda_graph=device.is_cuda)
        state, control = run.create_state(dt=0.001), run.create_control()
        run.step(state, state, control, dt=0.001)
        old, before = run.last_stats, serial(run.last_stats)
        output = snapshot(state)
        with test.assertRaises(ValueError):
            run.step(state, state, control, dt=-1)
        test.assertIs(run.last_stats, old)
        test.assertEqual(serial(old), before)
        finish = run._finish_candidate

        def fail_after_finish(candidate, dt, finish=finish, run=run):
            finish(candidate, dt)
            raise _StepFailure(run.Status.NONFINITE, "injected publication failure")

        for method, failure, has_vectors in (
            ("_iterate_articulation", _StepFailure(run.Status.NONFINITE, "injected early failure"), False),
            ("_finish_candidate", fail_after_finish, True),
        ):
            with patch.object(run, method, side_effect=failure):
                run.step(state, state, control, dt=0.001)
            failed = run.last_stats
            test.assertTrue(failed.rolled_back)
            test.assertEqual(failed.status, run.Status.NONFINITE)
            test.assertGreater(failed.step_generation, old.step_generation)
            assert_snapshot(test, state, output)
            test.assertEqual(serial(old), before)
            if has_vectors:
                check_snapshot(test, failed, run)
            else:
                test.assertEqual(list(vectors(failed)), [])
            failed_before = serial(failed)
            run.step(state, state, control, dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            test.assertEqual(serial(failed), failed_before)
            output = snapshot(state)
        with patch.object(run._states, "bind", side_effect=RuntimeError("injected bind failure")):
            with test.assertRaisesRegex(RuntimeError, "injected bind failure"):
                run.step(state, state, control, dt=0.001)
        test.assertTrue(run.last_stats.rolled_back)
        check_snapshot(test, run.last_stats, run)
        assert_snapshot(test, state, output)
        test.assertEqual(serial(old), before)


def test_snapshot_recording(test, device):
    """Export snapshots with the existing list schema and nonfinite-to-null policy."""
    for precision in ("fp32", "fp64"):
        _, run = scene(precision=precision, device=device, pcg_mode="production", use_cuda_graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        run.step(state, state, run.create_control(), dt=0.001)
        stats = run.last_stats
        check_snapshot(test, stats, run)
        expected = serial(stats)
        expected["variational"]["residual"] = run._stats_residual.numpy().tolist()
        for name, blocks in (("cloth_reactions", run._cloths), ("cable_reactions", run._cables)):
            expected["variational"][name] = [block.raw.numpy().tolist() for block in blocks]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            save(path, stats)
            test.assertEqual(json.loads(path.read_text()), expected)
        test.assertEqual(serial(np.array([np.nan, np.inf, -np.inf], dtype=run._numpy_dtype)), [None] * 3)
        test.assertEqual(serial(np.empty(0, dtype=run._numpy_dtype)), [])


class TestSolverStats(unittest.TestCase):
    pass


for function in (test_final_snapshots, test_stopped_snapshots, test_failure_snapshots, test_snapshot_recording):
    add_function_test(TestSolverStats, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
