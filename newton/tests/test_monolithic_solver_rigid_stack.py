# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise gravity-driven rigid stacks through the coupled contact solver."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_monolithic_solver_actor_contacts import frame
from newton.tests.test_monolithic_solver_rigid import assert_physical
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import transform
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit
from scripts.monolithic_reference.rigid_stack_scene import (
    BALANCE_RELATIVE,
    DT,
    GAP,
    HALF_EXTENT,
    HOLD_STEPS,
    PENETRATION_LIMIT,
    SETTLE_SPEED,
    STEPS,
    SUPPORT_RELATIVE,
    WEIGHT,
    stack_scene,
    stack_solver,
)

PAIR_LOADS = {
    (0, 1): -2 * WEIGHT,
    (1, 2): -WEIGHT,
    (0, 6): 3 * WEIGHT,
    (3, 4): -2 * WEIGHT,
    (4, 5): -WEIGHT,
    (3, 6): 3 * WEIGHT,
}


def check_stack_step(test, run, state):
    """Check publication, oriented vertical bounds and both sides of every pair."""
    test.assertTrue(run.last_stats.converged, run.last_stats)
    test.assertFalse(run.last_stats.rolled_back)
    for name, value in frame(run, state).items():
        test.assertTrue(np.isfinite(value).all(), name)
    pool = run.last_stats.variational["contact_pool"]
    test.assertEqual(pool["dropped"], 0)
    test.assertFalse(pool["tainted"])
    if run._contact_spatial:
        spatial = run.last_stats.variational["contact_spatial"]
        test.assertEqual(spatial["capacity_dropped"], 0)
        test.assertFalse(spatial["tainted"])
    history = state._monolithic_coupled
    test.assertEqual(history.force_valid.numpy()[0], 1)
    test.assertEqual(history.force_generation.numpy()[0], run.last_stats.step_generation)
    com = history.rigid_com.numpy().reshape(2, 3, 3)
    rotations = history.rigid_rotation.numpy()
    extents = np.asarray(
        [HALF_EXTENT * abs(transform(np.r_[np.zeros(3), q])[:3, :3][2]).sum() for q in rotations]
    ).reshape(2, 3)
    gaps = np.column_stack(
        (com[:, 0, 2] - extents[:, 0], np.diff(com[:, :, 2], axis=1) - extents[:, :-1] - extents[:, 1:])
    )
    test.assertGreaterEqual(float(gaps.min()), -PENETRATION_LIMIT)
    test.assertTrue((np.diff(com[:, :, 2], axis=1) > 0).all())
    loads = {}
    for pair in run.last_stats.variational["actor_contact_pairs"]:
        wrench = np.asarray(pair["world_wrench"])
        force_scale = max(1, np.linalg.norm(wrench[:, :3], axis=1).sum())
        torque_scale = max(1e-3, np.linalg.norm(wrench[:, 3:], axis=1).sum())
        test.assertLessEqual(np.linalg.norm(wrench[:, :3].sum(axis=0)), BALANCE_RELATIVE * force_scale)
        test.assertLessEqual(np.linalg.norm(wrench[:, 3:].sum(axis=0)), BALANCE_RELATIVE * torque_scale)
        loads[pair["source"], pair["target"]] = wrench[0, 2]
    return loads, max(0, -float(gaps.min()))


def test_rigid_stack_e2e(test, device):
    """Drop six boxes, transmit layer loads, settle, and restore an active midpoint."""
    routes = [(False, False), (True, False)] + ([(True, True)] if device.is_cuda else [])
    reference = {}
    test.stack_cases = []
    with tempfile.TemporaryDirectory() as directory:
        for spatial, graph in routes:
            with test.subTest(spatial=spatial, graph=graph):
                model = stack_scene(device)
                run = stack_solver(model, spatial=spatial, graph=graph)
                state, control = run.create_state(dt=DT), model.control()
                initial = state._monolithic_coupled.rigid_com.numpy().copy()
                audit = ContactSystemAudit(run, stack_solver(model, spatial=False, graph=graph)) if spatial else None
                checkpoint = Path(directory) / "midpoint.npz"
                seen, hold, peak_penetration = set(), [], 0.0
                midpoint_frame = None
                for step in range(1, STEPS + 1):
                    run.step(state, state, control, dt=DT)
                    loads, penetration = check_stack_step(test, run, state)
                    peak_penetration = max(peak_penetration, penetration)
                    seen.update(pair for pair in PAIR_LOADS if abs(loads[pair]) > 1e-4)
                    if step == 1:
                        test.assertTrue(all(abs(value) < 1e-12 for value in loads.values()))
                        np.testing.assert_allclose(
                            state._monolithic_coupled.rigid_v_com.numpy()[:, 2], -9.81 * DT, atol=1e-7, rtol=0
                        )
                    if step in (1, 20, 65, STEPS):
                        current = frame(run, state)
                        if not spatial:
                            reference[step] = current
                        else:
                            assert_physical(test, reference[step], current)
                        if audit is not None:
                            report = audit.evaluate()
                            test.assertEqual(report["status"], "PASS_FULL_RESIDUAL", report)
                    if step == 60:
                        run.save_checkpoint(state, checkpoint)
                    if step == 65:
                        midpoint_frame = frame(run, state)
                    if step > STEPS - HOLD_STEPS:
                        values = [loads[pair] for pair in PAIR_LOADS]
                        np.testing.assert_allclose(values, list(PAIR_LOADS.values()), rtol=SUPPORT_RELATIVE, atol=0)
                        forces = run.get_contact_forces(state).body_wrench[run._rigid.body_indices, :3]
                        np.testing.assert_allclose(forces[:, 2], WEIGHT, rtol=SUPPORT_RELATIVE, atol=0)
                        speed = np.linalg.norm(state._monolithic_coupled.rigid_v_com.numpy(), axis=1).max()
                        test.assertLessEqual(speed, SETTLE_SPEED)
                        hold.append(state._monolithic_coupled.rigid_com.numpy().copy())
                test.assertEqual(seen, set(PAIR_LOADS))
                final = state._monolithic_coupled.rigid_com.numpy()
                test.assertTrue(((initial - final)[:, 2] > GAP / 2).all())
                test.assertLessEqual(float(np.ptp(np.asarray(hold), axis=0).max()), GAP / 10)
                np.testing.assert_allclose(final[:, :2], initial[:, :2], atol=GAP / 10, rtol=0)
                resumed = stack_solver(model, spatial=spatial, graph=graph)
                restored = resumed.create_state(dt=DT)
                resumed.load_checkpoint(restored, checkpoint)
                for _ in range(5):
                    resumed.step(restored, restored, control, dt=DT)
                    check_stack_step(test, resumed, restored)
                assert_physical(test, midpoint_frame, frame(resumed, restored))
                test.stack_cases.append(
                    {
                        "device": str(device),
                        "spatial": spatial,
                        "graph": graph,
                        "steps": STEPS,
                        "restored_steps": 5,
                        "peak_penetration": peak_penetration,
                        "final_z": final[:, 2].tolist(),
                    }
                )


def test_rigid_stack_query_capture(test, device):
    """Preserve unique spatial candidates and free fall across capture and replay."""
    columns = [((i % 4) * (2 * HALF_EXTENT + GAP), (i // 4) * (2 * HALF_EXTENT + GAP)) for i in range(8)]
    for graph in (False, True) if device.is_cuda else (False,):
        with test.subTest(graph=graph):
            model = stack_scene(device, columns=columns, layers=4)
            run = stack_solver(model, spatial=True, graph=graph, query_capacity=36864, support_capacity=32768)
            state, control = run.create_state(dt=DT), model.control()
            initial = state._monolithic_coupled.rigid_com.numpy().copy()
            for step in range(1, 4):
                run.step(state, state, control, dt=DT)
                test.assertTrue(run.last_stats.converged, run.last_stats.failure_reason)
                test.assertFalse(run.last_stats.rolled_back)
                selection = run._contact._spatial
                count = int(selection.count.numpy()[0])
                identities = selection.identities.numpy()[:count]
                test.assertGreater(count, run._contact._batch.batch_capacity)
                test.assertTrue(np.all(identities[1:] > identities[:-1]))
                test.assertEqual(int(selection.query.dropped.numpy()[0]), 0)
                np.testing.assert_allclose(
                    state._monolithic_coupled.rigid_v_com.numpy()[:, 2], -9.81 * DT * step, atol=1e-7, rtol=0
                )
                np.testing.assert_allclose(
                    state._monolithic_coupled.rigid_com.numpy()[:, 2],
                    initial[:, 2] - 9.81 * DT**2 * step * (step + 1) / 2,
                    atol=1e-9,
                    rtol=0,
                )


class TestMonolithicRigidStack(unittest.TestCase):
    pass


add_function_test(TestMonolithicRigidStack, "test_rigid_stack_e2e", test_rigid_stack_e2e, devices=get_test_devices())
add_function_test(
    TestMonolithicRigidStack,
    "test_rigid_stack_query_capture",
    test_rigid_stack_query_capture,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
