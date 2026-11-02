# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete device evaluation of the composed nonlinear problem."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.problem import _MonolithicProblem
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.tests.test_monolithic_material_state import scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def energy_arrays(solver):
    """Independent inventory of the current physical energy buffers."""
    arrays = [solver._objective.link_energy, solver._objective.joint_energy, solver._rigid.energy, solver._tet.parts]
    for cloth in solver._cloths:
        arrays.extend((cloth.objective.energy, cloth.external_energy))
    for cable in solver._cables:
        arrays.extend((cable.objective.stretch_energy, cable.objective.bend_twist_energy, cable.external_energy))
    return arrays


def test_complete_energy(test, device):
    """Include rigid inertia and every material term in the device objective."""
    for precision in ("fp32", "fp64"):
        for materials in (False, True):
            _, run = scene(
                cloth=materials,
                cables=2 if materials else 0,
                mechanical=materials,
                rigid=True,
                tet=materials,
                precision=precision,
                device=device,
                pcg_mode="production",
                use_cuda_graph=device.is_cuda,
            )
            state = run.create_state(dt=0.001)
            run.step(state, state, run.create_control(), dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            # A known nonzero COM step makes omission of rigid inertia observable.
            step, _ = run._rigid_steps(run._accepted)
            step.assign(np.full((run._rigid.count, 3), 0.002, dtype=run._numpy_dtype))
            energy, residual, absolute = run._evaluate(run._accepted, 0.001, build_tangent=False, device_result=True)
            parts = np.concatenate([array.numpy().ravel() for array in energy_arrays(run)])
            expected = parts.sum() + run._contact.energy.numpy().sum()
            expected_absolute = abs(parts).sum() + run._contact.absolute_energy.numpy().sum()
            tolerance = 64 * np.finfo(run._numpy_dtype).eps
            np.testing.assert_allclose(energy, expected, rtol=tolerance)
            np.testing.assert_allclose(absolute, expected_absolute, rtol=tolerance)
            test.assertIsInstance(residual, wp.array)


def test_owner_evaluation_stays_on_device(test, device):
    """Do not collect energy or residual payloads between physical producers."""
    _, run = scene(mechanical=True, rigid=True, tet=True, device=device)
    state = run.create_state(dt=0.001)
    run.step(state, state, run.create_control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    arrays = [*energy_arrays(run), run._complete_r, run._frozen_gravity]
    pointers = {array.ptr for array in arrays if array.size}
    numpy = wp.array.numpy

    def read(array, *args, **kwargs):
        if array.size and array.ptr in pointers:
            raise AssertionError("Evaluation read a physical contribution payload")
        return numpy(array, *args, **kwargs)

    with patch.object(wp.array, "numpy", read):
        run._evaluate(run._accepted, 0.001, build_tangent=False, device_result=True)


def test_state_geometry(test, device):
    """Use owner maps and the left rotation logarithm for actual mixed-state motion."""
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
        accepted, trial = run._accepted, run._trial
        _, rotation = run._rigid_steps(accepted)
        quaternion = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.2)
        rotation.assign(np.array([quaternion], dtype=run._numpy_dtype))
        direction = np.linspace(-0.03, 0.04, run._delta.size, dtype=run._numpy_dtype)
        run._delta.assign(direction)
        run._take_trial(0.25, 0.001)
        expected = 0.25 * direction
        # Independently recover rotation from R_trial R_accepted^T.
        qa, qb = rotation.numpy()[0], run._rigid_steps(trial)[1].numpy()[0]
        ra, rb = (np.asarray(wp.quat_to_matrix(wp.quatd(*map(float, q)))).reshape(3, 3) for q in (qa, qb))
        relative = rb @ ra.T
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        axial = np.array(
            [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
        )
        expected[run._rigid.rows_numpy[0, 3:]] = axial * angle / (2 * np.sin(angle))
        arrays = [array for state_steps in run._steps.values() for array in state_steps]
        pointers = {array.ptr for array in arrays if array.size}
        numpy = wp.array.numpy

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("State geometry read a complete increment array")
            return numpy(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", read):
            result = run._effective_displacement(device_result=True)
            test.assertTrue(run._domain(trial))
        np.testing.assert_allclose(result.numpy(), expected, atol=64 * np.finfo(run._numpy_dtype).eps, rtol=1e-5)


def test_required_contributions(test, device):
    """Reject incomplete or duplicate physical composition at the binding boundary."""
    _, run = scene(cloth=False, cables=0, rigid=True, device=device)
    producer = run._producers[0]
    with patch.object(producer, "energies", ()), test.assertRaisesRegex(ValueError, "Missing energy"):
        _MonolithicProblem(run)
    with patch.object(producer, "candidate", None), test.assertRaisesRegex(ValueError, "Missing nonlinear phase"):
        _MonolithicProblem(run)
    with (
        patch.object(run, "_producers", [*run._producers, run._producers[-1]]),
        test.assertRaisesRegex(ValueError, "Duplicate"),
    ):
        _MonolithicProblem(run)


def test_invalid_contributions(test, device):
    """Retain nonfinite fixed-row reactions and geometry guards in the shared result."""
    _, run = scene(cables=0, device=device)
    state = run.create_state(dt=0.001)
    run.step(state, state, run.create_control(), dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    producer = run._producers[0]
    evaluate = producer.evaluate
    block = producer.block
    test.assertEqual(block.rows_host[0], -1)
    for target, value in ((block.raw[:1], np.nan), (block.objective.energy, np.nan), (block.objective.flags, 1)):

        def invalid(*args, target=target, value=value, **kwargs):
            evaluate(*args, **kwargs)
            target.fill_(target.dtype(value))

        with patch.object(producer, "evaluate", invalid), test.assertRaises(_StepFailure) as caught:
            run._evaluate(run._accepted, 0.001, build_tangent=False, device_result=True)
        test.assertEqual(caught.exception.status, run.Status.NONFINITE)
        energy, _, _ = run._evaluate(run._accepted, 0.001, build_tangent=False, device_result=True)
        test.assertTrue(np.isfinite(energy))


class TestProblem(unittest.TestCase):
    pass


for name, function in (
    ("test_complete_energy", test_complete_energy),
    ("test_owner_evaluation_stays_on_device", test_owner_evaluation_stays_on_device),
    ("test_state_geometry", test_state_geometry),
    ("test_required_contributions", test_required_contributions),
    ("test_invalid_contributions", test_invalid_contributions),
):
    add_function_test(TestProblem, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
