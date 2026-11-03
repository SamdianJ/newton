# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check numerical updates and storage invalidation across nonlinear graph replay."""

import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import contact_batch, contact_material, problem
from newton.tests.test_monolithic_contact_actor_pairs import native_inputs, perturb, physical
from newton.tests.test_monolithic_contact_spatial import spatial_fixture
from newton.tests.test_monolithic_material_state import scene, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_candidate_parameters(test, device):
    """Reuse both state roles across more alpha values than the resident graph budget."""
    for precision in ("fp32", "fp64"):
        _, run = scene(
            mechanical=True,
            rigid=True,
            tet=True,
            precision=precision,
            device=device,
            pcg_mode="production",
            use_cuda_graph=device.is_cuda,
        )
        state = run.create_state(dt=0.001)
        run.step(state, state, run.create_control(), dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        direction = np.linspace(-0.0001, 0.0002, run._delta.size, dtype=run._numpy_dtype)
        run._delta.assign(direction)
        before = run._graphs.captures["problem_candidate"]
        for i in range(32):
            alpha = run._numpy_dtype(2.0 ** -(i % 12))
            run._take_trial(alpha, 0.001)
            moved = run._effective_displacement(device_result=True).numpy()
            np.testing.assert_allclose(moved, alpha * direction, atol=32 * np.finfo(run._numpy_dtype).eps)
            run._problem.accept()
        if device.is_cuda:
            test.assertLessEqual(run._graphs.captures["problem_candidate"] - before, 2)
            test.assertEqual(run._graphs.evictions["problem_candidate"], 0)


def test_registered_bindings(test, device):
    """Resolve declared storage without reflection and reject incompatible replacements."""
    _, run = scene(
        mechanical=True,
        rigid=True,
        tet=True,
        device=device,
        pcg_mode="production",
        use_cuda_graph=device.is_cuda,
    )
    state = run.create_state(dt=0.001)
    run.step(state, state, run.create_control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    with (
        patch.object(problem, "vars", side_effect=AssertionError("hot reflection"), create=True),
        patch.object(contact_batch, "vars", side_effect=AssertionError("hot reflection"), create=True),
        patch.object(contact_material, "vars", side_effect=AssertionError("hot reflection"), create=True),
    ):
        buffers = run._problem._buffers(run._accepted)
        run._contact.buffers()
    test.assertTrue(any(a is run._objective.residual for a in buffers))
    old = run._objective.residual
    run._objective.residual = wp.clone(old)
    buffers = run._problem._buffers(run._accepted)
    test.assertTrue(any(a is run._objective.residual for a in buffers))
    for replacement in (
        wp.zeros(old.size + 1, dtype=old.dtype, device=device),
        wp.zeros(old.size, dtype=wp.float32, device=device),
    ):
        run._objective.residual = replacement
        with test.assertRaisesRegex(ValueError, "binding"):
            run._problem._buffers(run._accepted)
    run._objective.residual = old
    expected = run._evaluate(run._accepted, 0.001, build_tangent=False)
    captures = run._graphs.captures["problem_owners"]
    cable = run._cables[0]
    backing = wp.full(2 * cable.raw.size, np.nan, dtype=cable.raw.dtype, device=device)
    cable.raw = backing[::2]
    actual = run._evaluate(run._accepted, 0.001, build_tangent=False)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    test.assertTrue(np.isfinite(cable.raw.numpy()).all())
    test.assertTrue(np.isnan(backing.numpy()[1::2]).all())
    if device.is_cuda:
        test.assertEqual(run._graphs.captures["problem_owners"], captures + 1)
    for name in ("dt", "alpha", "inv_dt_sq"):
        original = getattr(run._problem, name)
        with patch.object(run._problem, name, wp.zeros(0, dtype=run._dtype, device=device)):
            with test.assertRaisesRegex(ValueError, "binding"):
                run._take_trial(0.25, 0.001)
        replacement = wp.zeros_like(original)
        with patch.object(run._problem, name, replacement):
            run._take_trial(0.25, 0.001)
            expected = 0.25 if name == "alpha" else 0.001 if name == "dt" else 1e6
            np.testing.assert_allclose(replacement.numpy(), expected, rtol=1e-6)


def test_timestep_validation_scope(test, device):
    """Preserve the failed-step boundary when non-tet inertia underflows."""
    _, run = scene(cloth=False, cables=0, mechanical=True, precision="fp32", device=device)
    state = run.create_state(dt=0.001)
    before = snapshot(state)
    run.step(state, state, run.create_control(), dt=1e23)
    test.assertEqual(run.last_stats.status, run.Status.NONFINITE)
    test.assertEqual(run.last_stats.failure_reason, "INVALID_DIAGONAL")
    test.assertTrue(run.last_stats.rolled_back)
    for name, value in snapshot(state).items():
        np.testing.assert_array_equal(value, before[name])


def test_timestep_rounding(test, device):
    """Distinguish host timesteps that share one FP32 dt but have different inertial scales."""
    _, run = scene(cloth=False, cables=0, mechanical=True, precision="fp32", device=device)
    for dt, inverse in ((np.float32(0.001), 999999.875), (0.001, 1000000.0), (np.float32(0.001), 999999.875)):
        run._problem._set_timestep(dt)
        np.testing.assert_array_equal(run._problem.dt.numpy(), np.float32(0.001))
        np.testing.assert_array_equal(run._problem.inv_dt_sq.numpy(), np.float32(inverse))


def test_timestep_replay(test, device):
    """Match eager mixed-owner dynamics under A/B/A time steps without new scalar graphs."""
    for precision, mass_mode in product(("fp32", "fp64"), ("lumped", "consistent")):
        runs = [
            scene(
                mechanical=True,
                rigid=True,
                tet=True,
                precision=precision,
                device=device,
                pcg_mode="production",
                use_cuda_graph=graph,
                mass_mode=mass_mode,
            )[1]
            for graph in (False, device.is_cuda)
        ]
        states = [run.create_state(dt=0.001) for run in runs]
        for run, state in zip(runs, states, strict=True):
            state.joint_qd.fill_(run._dtype(0.0001))
            velocity = np.zeros((run.model.particle_count, 3), dtype=run._numpy_dtype)
            velocity[run.model.particle_inv_mass.numpy() > 0, 2] = 0.0001
            state.particle_qd.assign(velocity)
            run.reset_state(state, dt=0.001)
        controls = [run.create_control() for run in runs]
        for control in controls:
            control.joint_f.fill_(0.0001)
        captures = None
        for dt in (0.001, 0.001, 0.001, 0.0017, 0.001, 0.0008, 0.001):
            for run, state, control in zip(runs, states, controls, strict=True):
                run.step(state, state, control, dt=dt)
                test.assertTrue(run.last_stats.converged, run.last_stats)
            for name, values in snapshot(states[0]).items():
                np.testing.assert_array_equal(values, snapshot(states[1])[name], err_msg=f"{precision}/{dt}/{name}")
            np.testing.assert_array_equal(runs[0]._complete_r.numpy(), runs[1]._complete_r.numpy())
            np.testing.assert_array_equal(
                runs[0]._linear._global.values.numpy(),
                runs[1]._linear._global.values.numpy(),
            )
            if device.is_cuda:
                current = {
                    name: value for name, value in runs[1]._graphs.captures.items() if name.startswith("problem_")
                }
                if captures is not None and dt != 0.001:
                    test.assertEqual(current, captures)
                captures = current


def test_contact_timestep_replay(test, device):
    """Update friction displacement scales in reference and regular contact graphs."""
    for precision in ("fp32", "fp64"):
        run, state, batch, _ = spatial_fixture(device, "chain", precision=precision)
        start = physical(state)
        perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-5)
        inputs, options = native_inputs(run, state)
        test.assertGreater(batch.contact.parameters[1], 0)
        batch.initialize_reference()
        for reference in (False, True):
            evaluate = batch.evaluate_reference if reference else batch.evaluate
            outputs = (
                (batch.reference_energy, batch.reference_residual)
                if reference
                else (
                    batch.pool.energy.result,
                    batch.pool.residual.result,
                    batch.pool.magnitude,
                    batch.pool.records.triplet_values,
                )
            )
            energies = []
            for dt in (0.001, 0.0017, 0.001):
                batch.graphs.enabled = False
                evaluate(*inputs, dt=dt, **options)
                expected = [a.numpy().copy() for a in outputs]
                batch.graphs.enabled = device.is_cuda
                evaluate(*inputs, dt=dt, **options)
                for a, value in zip(outputs, expected, strict=True):
                    np.testing.assert_array_equal(a.numpy(), value)
                energies.append(expected[0])
            test.assertFalse(np.array_equal(energies[0], energies[1]))
            np.testing.assert_array_equal(energies[0], energies[2])
            if device.is_cuda:
                name = "contact_reference" if reference else "contact_batch_eval"
                test.assertEqual(batch.graphs.captures[name], 1)


class TestGraphParameters(unittest.TestCase):
    pass


for name, function in (
    ("test_candidate_parameters", test_candidate_parameters),
    ("test_registered_bindings", test_registered_bindings),
    ("test_timestep_replay", test_timestep_replay),
    ("test_timestep_validation_scope", test_timestep_validation_scope),
    ("test_timestep_rounding", test_timestep_rounding),
    ("test_contact_timestep_replay", test_contact_timestep_replay),
):
    add_function_test(TestGraphParameters, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
