# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise gravity-driven rigid stacks through the coupled contact solver."""

import tempfile
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.contact_pipeline import _PooledContactObjective, _SpatialContactObjective
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_solver_actor_contacts import frame
from newton.tests.test_monolithic_solver_rigid import assert_physical
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import transform
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit

DT = 0.001
STEPS = 120
HALF_EXTENT = 0.02
MASS = 0.01
GAP = 0.0005
# Retain the existing contact E2E penetration and balance envelopes. The
# settling speed allows at most one initial gap of motion in the 20 ms hold.
PENETRATION_LIMIT = 0.002
BALANCE_RELATIVE = 1e-5
SUPPORT_RELATIVE = 0.05
HOLD_STEPS = 20
SETTLE_SPEED = GAP / (HOLD_STEPS * DT)
WEIGHT = MASS * 9.81
PAIR_LOADS = {
    (0, 1): -2 * WEIGHT,
    (1, 2): -WEIGHT,
    (0, 6): 3 * WEIGHT,
    (3, 4): -2 * WEIGHT,
    (4, 5): -WEIGHT,
    (3, 6): 3 * WEIGHT,
}


def stack_scene(device):
    """Build two initially separated three-box stacks above an infinite plane."""
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    # The supported coupled entry requires an articulation and a tet. Neither
    # has a collision shape or moves in this rigid-only contact experiment.
    auxiliary = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)), label="auxiliary")
    joint = builder.add_joint_prismatic(-1, auxiliary, axis=newton.Axis.X, limit_ke=0, limit_kd=0)
    builder.add_articulation([joint])
    add_contact_free_tet(builder)
    builder.particle_mass[:] = [0.0] * builder.particle_count
    cfg = builder.ShapeConfig(density=0, margin=0, has_particle_collision=False)
    for column, x in enumerate((-0.03, 0.03)):
        for layer in range(3):
            z = HALF_EXTENT + GAP + layer * (2 * HALF_EXTENT + GAP)
            body = builder.add_link(
                xform=wp.transform((x, 0, z), wp.quat_identity()),
                mass=MASS,
                inertia=wp.diag(wp.vec3(MASS * (2 * HALF_EXTENT) ** 2 / 6)),
                label=f"box_{column}_{layer}",
            )
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])
            builder.add_shape_box(body, hx=HALF_EXTENT, hy=HALF_EXTENT, hz=HALF_EXTENT, cfg=cfg)
    builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg)
    return builder.finalize(device=device)


def stack_solver(model, *, spatial, graph=False):
    """Select an explicit contact route without changing production defaults."""
    factory = (
        partial(_SpatialContactObjective, candidate_capacity=4096, law_margin=0.0)
        if spatial
        else _PooledContactObjective
    )
    with patch.object(solver_monolithic, "ContactObjective", factory):
        run = SolverMonolithic(
            model,
            contact_model="C-IE",
            globalization="ArmijoOrResNorm",
            allow_nonconservative_energy=True,
            friction_velocity_scale=0.01,
            pcg_mode="production",
            use_cuda_graph=graph,
            precision="fp64",
            publish_policy="strict",
            tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
        )
    # Match the currently selected numerical experiment; retain all physical
    # convergence gates and the normal Newton/line-search/PCG budgets.
    run._config = replace(run._config, regularization_values=(0.0,))
    return run


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


class TestMonolithicRigidStack(unittest.TestCase):
    pass


add_function_test(TestMonolithicRigidStack, "test_rigid_stack_e2e", test_rigid_stack_e2e, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
