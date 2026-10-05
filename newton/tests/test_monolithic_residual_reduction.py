# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed native contact and tet residual accumulation on repeated evaluations."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.residual import _accumulate_rows, _row_gather_map
from newton._src.solvers.monolithic.tet_precision import TetObjective
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_ordered_gather(test, device):
    """Preserve cancellation order, initial particle terms, fixed slots and empty rows."""
    columns = np.array([2, 0, -1, 2, 0, 2], dtype=np.int32)
    offsets, slots = _row_gather_map(columns, 4, device)
    for dtype, native, vector in ((wp.float32, np.float32, wp.vec3), (wp.float64, np.float64, wp.vec3d)):
        big = native(2 ** (24 if dtype == wp.float32 else 53))
        for element_type in (dtype, vector):
            values = np.array([big, 3, np.nan, -big, -2, 1], dtype=native)
            initial = np.array([1, 4, 0, 7], dtype=native)
            expected = np.array([2, 4, 1, 7], dtype=native)
            if element_type == vector:
                values, initial, expected = (np.repeat(a[:, None], 3, axis=1) for a in (values, initial, expected))
            terms = wp.array(values, dtype=element_type, device=device)
            residual = wp.array(initial, dtype=element_type, device=device)
            wp.launch(_accumulate_rows, 4, [offsets, slots, terms, residual], device=device)
            np.testing.assert_array_equal(residual.numpy(), expected)
    empty_offsets, empty_slots = _row_gather_map(np.empty(0, dtype=np.int32), 0, device)
    test.assertEqual(empty_slots.size, 0)
    np.testing.assert_array_equal(empty_offsets.numpy(), [0])
    with patch.object(wp, "array", side_effect=AssertionError("allocated before capacity check")):
        with test.assertRaisesRegex(ValueError, "capacity"):
            _row_gather_map(columns, 2**31 - 1, device)
        oversized = np.broadcast_to(np.array([0], dtype=np.int32), (2**31,))
        with test.assertRaisesRegex(ValueError, "capacity"):
            _row_gather_map(oversized, 1, device)
    for bad in (np.array([-2]), np.array([4])):
        with test.assertRaisesRegex(ValueError, "rows"):
            _row_gather_map(bad, 4, device)


def test_frozen_tet(test, device):
    """Keep shared-vertex residuals stable in both native precisions and mass modes."""
    builder = newton.ModelBuilder(gravity=(0.2, -0.3, -9.81))
    builder.add_soft_grid(
        pos=wp.vec3(0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1000,
        k_mu=10000 / 2.6,
        k_lambda=10000 * 0.3 / (1.3 * 0.4),
        k_damp=0,
        fix_left=True,
        add_surface_mesh_edges=False,
    )
    mass = np.zeros(len(builder.particle_q))
    for ids, pose in zip(builder.tet_indices, builder.tet_poses, strict=True):
        volume = 1 / (6 * np.linalg.det(np.asarray(pose, dtype=float).reshape(3, 3)))
        np.add.at(mass, list(ids), 1000 * volume / 4)
    for i, point in enumerate(builder.particle_q):
        builder.particle_mass[i] = 0.0 if point[0] == 0 else float(mass[i])
    model = builder.finalize(device=device)
    initial = model.particle_q.numpy().astype(np.float64)
    positions = initial.copy()
    positions[:, 1] += 0.03 * positions[:, 0] ** 2
    positions[:, 2] += 0.01 * positions[:, 0] * positions[:, 1]
    for dtype, vector in ((wp.float64, wp.vec3d), (wp.float32, wp.vec3)):
        for mass_mode in ("lumped", "consistent"):
            with test.subTest(dtype=dtype, mass_mode=mass_mode):
                obj = TetObjective(
                    model,
                    material_model="kim_stable_no_log",
                    mass_mode=mass_mode,
                    density=wp.full(model.tet_count, 1000.0, device=device) if mass_mode == "consistent" else None,
                    dtype=dtype,
                )
                x = wp.array(positions, dtype=vector, device=device)
                xn = wp.array(initial, dtype=vector, device=device)
                vn = wp.zeros_like(x)
                forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
                repeated = []
                for _ in range(32):
                    obj.evaluate(x, xn, vn, forces, model.gravity, dt=0.01)
                    repeated.append(obj.residual.numpy().tobytes())
                test.assertEqual(int(obj.flags.numpy()[0]), 0)
                test.assertEqual(len(set(repeated)), 1)


def test_frozen_residual(test, device):
    """Keep native residual bits stable when evaluating the same accepted contact state."""
    for permuted in (False, True):
        with test.subTest(permuted=permuted):
            run = contact_solver(contact_scene(device), permuted=permuted)
            state = contact_state(run, "chain")
            run.step(state, state, run.model.control(), dt=0.001)
            test.assertTrue(run.last_stats.converged)
            values = {"contact": [], "tet": []}
            for _ in range(32):
                run._evaluate(run._accepted, 0.001)
                values["contact"].append(run._contact.residual.numpy())
                values["tet"].append(run._tet.residual.numpy())
            for component, repeated in values.items():
                with test.subTest(component=component):
                    test.assertGreater(np.linalg.norm(repeated[0]), 0)
                    test.assertEqual(len({value.tobytes() for value in repeated}), 1)


def test_reduction_failure(test, device):
    """Roll back complete state and retain old forces when a residual sum is nonfinite."""
    run = contact_solver(contact_scene(device), mode="production", graph=device.is_cuda)
    state = contact_state(run, "chain")
    run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged)
    output = run.create_state(dt=0.001)
    before, output_before = snapshot(state), snapshot(output)
    force = run.get_contact_forces(state).body_wrench.copy()
    launch = wp.launch
    for owner in (run._contact, run._tet):
        run._graphs.entries.clear()

        def inject(kernel, dim, inputs=None, *args, owner=owner, **kwargs):
            if kernel is _accumulate_rows and inputs[2] is owner._residual_terms:
                owner._residual_terms.fill_(np.finfo(np.float64).max)
            return launch(kernel, dim, inputs, *args, **kwargs)

        with patch.object(wp, "launch", inject):
            run.step(state, output, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.rolled_back)
        test.assertEqual(run.last_stats.status, run.Status.NONFINITE)
        assert_snapshot(test, state, before)
        assert_snapshot(test, output, output_before)
        np.testing.assert_array_equal(run.get_contact_forces(state).body_wrench, force)


def test_graph_rebinding(test, device):
    """Replay changed values and recapture replacement residual storage and row maps."""
    run = contact_solver(contact_scene(device), mode="production", graph=True)
    state = contact_state(run, "chain")
    run.step(state, state, run.model.control(), dt=0.001)
    test.assertTrue(run.last_stats.converged)
    expected = run._evaluate(run._accepted, 0.001)[1]
    for owner in (run._contact, run._tet):
        names = ("_residual_terms", "_residual_offsets", "_residual_slots")
        if owner is run._contact:
            names += ("_support_offsets", "_support_local")
        for name in names:
            count = run._graphs.captures["coupled_objective"]
            setattr(owner, name, wp.clone(getattr(owner, name)))
            np.testing.assert_array_equal(run._evaluate(run._accepted, 0.001)[1], expected)
            test.assertEqual(run._graphs.captures["coupled_objective"], count + 1)
    x = run._accepted.particle_q.numpy()
    dynamic = run.model.particle_inv_mass.numpy() > 0
    x[dynamic, 1] += 1e-6
    run._accepted.particle_q.assign(x)
    count = run._graphs.captures["coupled_objective"]
    changed = run._evaluate(run._accepted, 0.001)[1]
    test.assertFalse(np.array_equal(changed, expected))
    test.assertEqual(run._graphs.captures["coupled_objective"], count)
    run._graphs.enabled = False
    np.testing.assert_array_equal(run._evaluate(run._accepted, 0.001)[1], changed)


class TestResidualReduction(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (test_ordered_gather, test_frozen_tet, test_frozen_residual, test_reduction_failure):
        add_function_test(TestResidualReduction, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestResidualReduction, "test_graph_rebinding", test_graph_rebinding, devices=[_device])


if __name__ == "__main__":
    unittest.main()
