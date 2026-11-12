# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise point-cloud candidate bounds and published-state lifecycle on each device."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cloth_contact_point_cloud import _finish_query, _query_points
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_point_cloud import _candidate_oracle, _pairs, _scene
from newton.tests.test_monolithic_material_state import snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_query_counter_limits(test, device):
    """Count beyond int32 without wrapping and publish bounded counts and overflow."""

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device=device)

    x = array(((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0.01, 0.01, 0)), wp.vec3f)
    limit = np.iinfo(np.int32).max
    for initial, capacity in ((0, 0), (0, 1), (0, 4), (limit - 1, 1), (limit + 1, 1)):
        with test.subTest(initial=initial, capacity=capacity):
            total = array((initial,), wp.uint64)
            count = wp.zeros(1, dtype=int, device=device)
            overflow = wp.zeros(1, dtype=int, device=device)
            flags = wp.zeros(1, dtype=int, device=device)
            pairs = wp.full(capacity, -1, dtype=wp.vec2i, device=device)
            wp.launch(
                _query_points,
                2,
                [
                    x,
                    x,
                    x,
                    array(((0, 1, 2),), wp.vec3i),
                    array((0,), int),
                    array((3,), int),
                    array((1,), int),
                    array((0, 0), wp.float32),
                    array((-1, -1), int),
                    array((0,), int),
                    array((-1,), int),
                    wp.float32(1),
                    wp.float32(1),
                    wp.float32(0),
                    wp.float32(0),
                    wp.float32(1.5),
                    True,
                    False,
                    False,
                    pairs,
                    total,
                    flags,
                ],
                device=device,
            )
            test.assertEqual(int(total.numpy()[0]), initial + 2)
            wp.launch(_finish_query, 1, [total, count, overflow, capacity], device=device)
            test.assertEqual(int(count.numpy()[0]), min(initial + 2, capacity))
            test.assertEqual(int(overflow.numpy()[0]), min(max(initial + 2 - capacity, 0), limit))
            if initial:
                np.testing.assert_array_equal(pairs.numpy(), -1)
            else:
                actual = set(map(tuple, pairs.numpy()[: min(2, capacity)]))
                test.assertEqual(len(actual), min(2, capacity))
                test.assertTrue(actual <= {(0, 0), (1, 0)})


def test_self_contact_lifecycle(test, device, *, precision, graph):
    """Publish self forces, replay graphs and continue through a fresh checkpoint owner."""
    _, solver = _scene(self_owner=True, precision=precision, device=device, graph=graph)
    dt = 0.0001
    state = solver.create_state(dt=dt)
    test.assertEqual(state.particle_q.device, device)
    test.assertEqual(state.particle_q.numpy().dtype, np.float32 if precision.value == "fp32" else np.float64)
    contact = solver._cloth_contact
    contact.begin_step(state)
    contact.evaluate(wp.zeros_like(state.particle_q), dt)
    np.testing.assert_array_equal(contact.pool.force.result.numpy(), 0)
    test.assertEqual(float(contact.summary.numpy()[5]), 0)
    # Bend without stretching the reference segments or starting at a sharp crease.
    lengths = np.array([0.015, 0.015, 0.005, 0.015, 0.015])
    angles = np.arange(5) * np.pi / 4
    columns = np.vstack(
        (np.zeros(2), np.cumsum(lengths[:, None] * np.array([np.cos(angles), np.sin(angles)]).T, axis=0))
    )
    state.particle_q.assign([(x, y, z) for x, z in columns for y in (0.0, 0.02)])
    solver.reset_state(state, dt=dt)
    contact.begin_step(state)
    x = state.particle_q.numpy()
    test.assertEqual(_pairs(contact), _candidate_oracle(solver, x, x))

    def step(owner, current):
        previous = owner.last_stats.step_generation if owner.last_stats is not None else 0
        owner.step(current, current, owner.create_control(), dt=dt)
        stats = owner.last_stats
        force = owner.get_contact_forces(current)
        test.assertFalse(stats.rolled_back, stats)
        test.assertTrue(force.force_valid)
        test.assertGreater(force.force_generation, previous)
        test.assertEqual(force.force_generation, stats.step_generation)
        test.assertGreater(owner.get_cloth_contact_diagnostics(current)["self"]["active"], 0)
        test.assertGreater(np.linalg.norm(force.particle_force), 0)

    for _ in range(2):
        step(solver, state)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cloth.npz"
        solver.save_checkpoint(state, path)
        _, restored_solver = _scene(self_owner=True, precision=precision, device=device, graph=graph)
        restored = restored_solver.create_state(dt=dt)
        restored_solver.load_checkpoint(restored, path)
    for key, value in snapshot(state).items():
        np.testing.assert_array_equal(snapshot(restored)[key], value, err_msg=key)
    step(solver, state)
    step(restored_solver, restored)
    actual = snapshot(restored)
    for key, expected in snapshot(state).items():
        # Diagnostic loads are summed in candidate identity order, so they match bit for bit too.
        np.testing.assert_array_equal(actual[key], expected, err_msg=key)
    if graph:
        for name in ("point_cloud_query", "point_cloud_response"):
            segment = contact.graphs.summary()["segments"][name]
            test.assertGreater(segment["captures"], 0)
            test.assertGreater(segment["replays"], segment["captures"])


def test_candidate_overflow_rollback(test, device, *, precision, graph):
    """Restore every published state and history array when zero or one slot cannot fit the candidates."""
    for capacity in (0, 1):
        with test.subTest(capacity=capacity):
            _, solver = _scene(capacity=capacity, precision=precision, device=device, graph=graph)
            state = solver.create_state(dt=0.001)
            before = snapshot(state)
            solver.step(state, state, solver.create_control(), dt=0.001)
            test.assertEqual(solver.last_stats.status, solver.Status.CONTACT_OVERFLOW)
            test.assertTrue(solver.last_stats.rolled_back)
            for key, value in before.items():
                np.testing.assert_array_equal(snapshot(state)[key], value, err_msg=key)


class TestPointCloudRuntime(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestPointCloudRuntime, "test_query_counter_limits", test_query_counter_limits, devices=devices)
for _precision in SolverMonolithic.Precision:
    for _graph in (False, True):
        for _test in (test_self_contact_lifecycle, test_candidate_overflow_rollback):
            add_function_test(
                TestPointCloudRuntime,
                f"{_test.__name__}_{_precision.value}_{'graph' if _graph else 'eager'}",
                _test,
                devices=[device for device in devices if not _graph or device.is_cuda],
                precision=_precision,
                graph=_graph,
            )


if __name__ == "__main__":
    unittest.main()
