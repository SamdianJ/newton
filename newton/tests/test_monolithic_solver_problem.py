# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete device evaluation of the composed nonlinear problem."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.problem import _MonolithicProblem
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_multitree import contact_trees, trees_solver
from newton.tests.test_monolithic_material_state import scene, snapshot
from newton.tests.test_monolithic_solver_producers import free_box_scene
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
    with test.subTest(binding="duplicate"), patch.object(producer, "energies", producer.energies * 2):
        with test.assertRaisesRegex(ValueError, "Duplicate energy"):
            _MonolithicProblem(run)
    with test.subTest(phase="inherited"):
        owner = type(producer)
        difference = owner.difference
        del owner.difference
        try:
            with test.assertRaisesRegex(ValueError, "Missing nonlinear phase difference"):
                _MonolithicProblem(run)
        finally:
            owner.difference = difference
    _, run = scene(cloth=False, cables=0, mechanical=True, device=device)
    producer = run._producers[0]
    with test.subTest(binding="partial"), patch.object(producer, "energies", producer.energies[:1]):
        with test.assertRaisesRegex(ValueError, "Missing energy"):
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


def test_mixed_iteration_payloads(test, device):
    """Use one device iteration for every current actor kind, including material gauges."""
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
        control = run.create_control()
        force = control.joint_f.numpy().copy()
        force[0] = 1e-3
        control.joint_f.assign(force)
        arrays = [
            run._complete_r,
            run._complete_a,
            run._linear.scale,
            run._delta,
            run._coordinates,
            run._accepted_r,
            run._reference_device,
            run._effective_device,
            run._stats_residual,
            run._post_error,
            run._post_magnitude,
            run._post_original,
            run._post_regularization,
            run._criterion.tau,
            run._criterion.magnitude,
            run._criterion.squares,
            *energy_arrays(run),
        ]
        arrays.extend(a for block in run._cloths + run._cables for a in (block.raw, block.absolute))
        pointers = {a.ptr for a in arrays if a.size}
        numpy = wp.array.numpy
        original = run._iterate_articulation

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("Mixed iteration read a physical payload")
            return numpy(array, *args, **kwargs)

        def iterate(dt, run=run, original=original, pointers=pointers, read=read):
            pointers.update(a.ptr for steps in run._steps.values() for a in steps if a.size)
            with patch.object(wp.array, "numpy", read):
                return original(dt)

        with patch.object(run, "_iterate_articulation", iterate):
            run.step(state, state, control, dt=0.001)
        test.assertTrue(run.last_stats.converged, run.last_stats)
        test.assertGreater(run.last_stats.nonlinear_iterations, 0)


def test_step_inputs(test, device):
    """Freeze material controls and rigid wrenches without downloading physical vectors."""
    for precision in ("fp32", "fp64"):
        _, run = scene(rigid=True, precision=precision, device=device)
        state, control = run.create_state(dt=0.001), run.create_control()
        body = np.arange(state.body_f.size * 6, dtype=np.float32).reshape(-1, 6) * 0.1
        joint = np.arange(control.joint_f.size, dtype=np.float32) * 0.2
        gravity = np.array([[0.1, -0.2, -0.3]], dtype=np.float32)
        state.body_f.assign(body)
        control.joint_f.assign(joint)
        run.model.gravity.assign(gravity)
        cable = control._monolithic_cable
        torque = cable.torque.numpy().copy()
        torque[0] = 0.2
        cable.torque.assign(torque)
        arrays = [
            state.body_f,
            control.joint_f,
            run.model.gravity,
            run._rigid.mass,
            cable.target_position,
            cable.target_theta,
            cable.torque,
        ]
        pointers, numpy = {a.ptr for a in arrays if a.size}, wp.array.numpy

        def read(array, *args, pointers=pointers, numpy=numpy, **kwargs):
            if array.size and array.ptr in pointers:
                raise AssertionError("Step preparation read a physical payload")
            return numpy(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", read):
            frozen = run._snapshot_material_control(control)
            run._rigid.prepare(state, control, 0.001)
        cable.torque.zero_()
        np.testing.assert_array_equal(frozen["torque"].numpy(), torque)
        rigid, native = run._rigid, run._numpy_dtype
        expected = body[rigid.body_indices].astype(native) + joint[rigid.dof_indices].astype(native)
        expected[:, :3] += rigid.mass.numpy()[:, None] * gravity.astype(native)
        np.testing.assert_allclose(rigid.force.numpy(), expected[:, :3], rtol=4 * np.finfo(native).eps)
        np.testing.assert_array_equal(rigid.torque.numpy(), expected[:, 3:])
        test.assertTrue(rigid.has_torque)
        for field in ("target_position", "target_theta", "torque"):
            array = getattr(cable, field)
            before = array.numpy().copy()
            array.fill_(array.dtype(np.nan))
            with test.assertRaisesRegex(ValueError, "Nonfinite cable control"):
                run._snapshot_material_control(control)
            array.assign(before)
        # Fixed physical angles remain constraints, including when rows are omitted.
        torque[1] = 1
        cable.torque.assign(torque)
        with test.assertRaisesRegex(ValueError, "physically free theta"):
            run._snapshot_material_control(control)


def test_step_group_units(test, device):
    """Keep global scaled RMS and rigid physical RMS distinct in the same reduction."""
    for precision in ("fp32", "fp64"):
        _, run = scene(mechanical=True, rigid=True, tet=True, precision=precision, device=device)
        native, n = run._numpy_dtype, run._delta.size
        residual = np.linspace(-1, 1, n, dtype=native)
        effective = np.linspace(0.1, 0.2, n, dtype=native)
        direction = -2 * effective
        scale = np.geomspace(0.5, 2, n).astype(native)
        run._accepted_r.assign(residual)
        run._effective_device.assign(effective)
        run._delta.assign(direction)
        run._linear.scale.assign(scale)
        dot, noise, moved = run._problem.trial_summary(run._accepted_r, run._effective_device)
        tolerance = 64 * np.finfo(native).eps
        np.testing.assert_allclose(
            [dot, noise], [residual @ effective, abs(residual * direction).sum()], rtol=tolerance, atol=tolerance
        )
        expected = []
        for kind, start, count in run._layout.groups:
            part = slice(start, start + count)
            values = effective[part] if kind.startswith("rigid_") else effective[part] / scale[part]
            expected.append(np.sqrt(np.mean(values * values)))
        np.testing.assert_allclose(moved, expected, rtol=tolerance)


def test_composed_continuation(test, device):
    """Run load/hold/unload through mixed owners, checkpoint continuation and rollback."""
    for precision in ("fp32", "fp64"):
        for graph in (False, True) if device.is_cuda else (False,):
            options = {"precision": precision, "pcg_mode": "production", "use_cuda_graph": graph}
            factories = {
                "trees": lambda options=options: trees_solver(
                    contact_trees(device, mixed=True), mass_mode="lumped", **options
                ),
                "materials": lambda options=options: scene(
                    mechanical=True, rigid=True, tet=True, device=device, **options
                )[1],
                "rigid": lambda options=options: SolverMonolithic(
                    free_box_scene(device), friction_velocity_scale=0.01, **options
                ),
            }
            for kind, factory in factories.items():
                with (
                    test.subTest(precision=precision, graph=graph, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    run = factory()
                    state, control = run.create_state(dt=0.001), run.create_control()
                    fixed = np.flatnonzero(run.model.particle_inv_mass.numpy() == 0)
                    fixed_position = state.particle_q.numpy()[fixed].copy()
                    joint_load = np.zeros(control.joint_f.shape, dtype=np.float32)
                    if kind == "trees":
                        joint_load[:3] = [0.02, -0.04, -0.01]
                    else:
                        joint_load[0] = 1e-3
                    particle_load = np.zeros((run.model.particle_count, 3), dtype=np.float32)
                    if kind == "materials":
                        particle_load[:, 1] = 1e-4
                        particle_load[fixed] = 0
                    restored = restored_state = restored_control = None
                    path = Path(directory) / "step32.npz"
                    updates, contact_seen = 0, False
                    for step in range(64):
                        factor = (step + 1) / 16 if step < 16 else 1.0 if step < 48 else (63 - step) / 16
                        active = [(run, state, control)]
                        if restored is not None:
                            active.append((restored, restored_state, restored_control))
                        for current, current_state, current_control in active:
                            current_control.joint_f.assign(factor * joint_load)
                            current_state.particle_f.assign(factor * particle_load)
                            current.step(current_state, current_state, current_control, dt=0.001)
                            stats = current.last_stats
                            test.assertTrue(stats.converged, (kind, precision, graph, step, stats))
                            test.assertFalse(stats.rolled_back)
                            test.assertLessEqual(stats.convergence_ratio, 1)
                            test.assertEqual(stats.variational["contact_spatial"]["capacity_dropped"], 0)
                            current._states.validate(current_state)
                            np.testing.assert_array_equal(current_state.particle_q.numpy()[fixed], fixed_position)
                        updates += run.last_stats.nonlinear_iterations
                        contact_seen |= bool(run.last_stats.variational.get("actor_contact_pairs"))
                        if restored is not None:
                            expected = snapshot(state)
                            for name, values in snapshot(restored_state).items():
                                np.testing.assert_array_equal(values, expected[name], err_msg=f"{kind}/{step}/{name}")
                        if step == 31:
                            run.save_checkpoint(state, path)
                            restored = factory()
                            restored_state, restored_control = (
                                restored.create_state(dt=0.001),
                                restored.create_control(),
                            )
                            restored.load_checkpoint(restored_state, path)
                    test.assertGreater(updates, 0)
                    if kind == "trees":
                        test.assertTrue(contact_seen)
                    # Independent failed steps must not publish any rejected state or history.
                    control.joint_f.fill_(0.01)
                    original, output = snapshot(state), run.create_state(dt=0.001)
                    previous = snapshot(output)
                    with patch(
                        "newton._src.solvers.monolithic.iteration.acceptance", return_value=(False,) * 4
                    ) as reject:
                        run.step(state, output, control, dt=0.001)
                    test.assertGreater(reject.call_count, 0)
                    test.assertTrue(run.last_stats.rolled_back)
                    for current, expected in ((state, original), (output, previous)):
                        for name, values in snapshot(current).items():
                            np.testing.assert_array_equal(values, expected[name], err_msg=name)
                    with patch.object(
                        run, "_finish_candidate", side_effect=_StepFailure(run.Status.NONFINITE, "publish")
                    ):
                        run.step(state, output, control, dt=0.001)
                    test.assertTrue(run.last_stats.rolled_back)
                    for current, expected in ((state, original), (output, previous)):
                        for name, values in snapshot(current).items():
                            np.testing.assert_array_equal(values, expected[name], err_msg=name)


class TestProblem(unittest.TestCase):
    pass


for name, function in (
    ("test_complete_energy", test_complete_energy),
    ("test_owner_evaluation_stays_on_device", test_owner_evaluation_stays_on_device),
    ("test_state_geometry", test_state_geometry),
    ("test_required_contributions", test_required_contributions),
    ("test_invalid_contributions", test_invalid_contributions),
    ("test_mixed_iteration_payloads", test_mixed_iteration_payloads),
    ("test_step_inputs", test_step_inputs),
    ("test_step_group_units", test_step_group_units),
    ("test_composed_continuation", test_composed_continuation),
):
    add_function_test(TestProblem, name, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
