# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check material phase costs and typed device batches."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_material_state import scene, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def phase_calls(run, control):
    """Expose existing material dependency boundaries for bounded cost checks."""
    contact = next(producer for producer in run._producers if producer.name == "contact")
    operator, preconditioner = run._linear._material_operator, run._linear._cable_preconditioner
    calls = {
        "control": lambda: run._snapshot_material_control(control),
        "reaction": lambda: contact.evaluate(run, run._accepted, 0.001, run._complete_r, build_tangent=False),
    }
    if run._cables:
        calls.update(
            gauge=run._settle_cable_gauges,
            absolute=lambda: operator.add_abs_coordinates(run._coordinates, run._post_material_magnitude),
            product=lambda: operator.add_product(
                run._y, run._linear.scale, run._post_material, magnitude=run._post_material_magnitude
            ),
            band=lambda: preconditioner.factor(run._linear, 1e-12),
        )
    return calls


def test_submission_count(test, device):
    """Keep steady material phase launches and readbacks independent of actor count."""
    for kind in ("cloth", "cable"):
        baseline = {}
        for count in (1, 2, 4, 8):
            _, run = scene(cloth=count if kind == "cloth" else 0, cables=count if kind == "cable" else 0, device=device)
            state, control = run.create_state(dt=0.001), run.create_control()
            run.step(state, state, control, dt=0.001)
            test.assertTrue(run.last_stats.converged, run.last_stats)
            run._evaluate(run._accepted, 0.001)
            run._assemble(0, device_result=True)
            calls = phase_calls(run, control)
            for name, call in calls.items():
                call()
                launches, reads = [], []
                original_launch, original_read = wp.launch, wp.array.numpy

                def launch(kernel, *args, name=name, launches=launches, original_launch=original_launch, **kwargs):
                    if name != "reaction" or kernel.key == "_contact_reaction":
                        launches.append(kernel.key)
                    return original_launch(kernel, *args, **kwargs)

                def read(array, *args, reads=reads, original_read=original_read, **kwargs):
                    reads.append(array.size)
                    return original_read(array, *args, **kwargs)

                with patch.object(wp, "launch", launch), patch.object(wp.array, "numpy", read):
                    call()
                actual = (len(launches), len(reads))
                if count == 1:
                    baseline[name] = actual
                with test.subTest(kind=kind, count=count, phase=name):
                    test.assertEqual(actual, baseline[name])


def heterogeneous_scene(device, precision):
    """Build disjoint cloth/cable owners with interleaved ids and a zero-DoF cable."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.add_body(mass=1, inertia=wp.diag(wp.vec3(0.1)))
    sizes = (3, 3, 2, 4, 7)
    ids = [[] for _ in sizes]
    for vertex in range(max(sizes)):
        for actor, count in enumerate(sizes):
            if vertex >= count:
                continue
            if actor < 2:
                point = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0))[vertex])
                point[1] += 3 * actor
                mass = 0 if vertex == 0 else 1 / 6
            else:
                point = (vertex, 3 * actor, 0)
                mass = 0 if vertex == 0 or actor == 2 else 0.5 if vertex == count - 1 else 1
            ids[actor].append(builder.add_particle(wp.vec3(point), wp.vec3(0), mass))
    cloths = []
    for group in ids[:2]:
        face = builder.tri_count
        builder.add_triangle(*group, tri_ke=2, tri_ka=3, tri_kd=0, tri_drag=0, tri_lift=0)
        cloths.append(
            SolverMonolithic.ClothInput(triangle_ids=(face,), bending=(1, 2), thickness=0.01, surface_density=1)
        )
    cables = [
        SolverMonolithic.CableInput(
            particle_ids=group,
            EA=10,
            bending=((2, 0.2), (0.2, 1)),
            GJ=1,
            radius=0.01,
            linear_density=1,
            rest_theta=(0,) * (len(group) - 1),
            rest_director=((0, 1, 0),) * (len(group) - 1),
            fixed_theta=(True,) if index == 0 else (True, False, True) if index == 1 else (False,) * (len(group) - 1),
        )
        for index, group in enumerate(ids[2:])
    ]
    run = SolverMonolithic(
        builder.finalize(device=device),
        cloth_inputs=cloths,
        cable_inputs=cables,
        mass_mode="lumped",
        friction_velocity_scale=0.1,
        precision=precision,
        pcg_mode="production",
        use_cuda_graph=device.is_cuda,
    )
    state, control = run.create_state(dt=0.001), run.create_control()
    state.body_f.assign(np.array([[1e-3, 0, 0, 0, 0, 0]], dtype=np.float32))
    run.step(state, state, control, dt=0.001)
    if not run.last_stats.converged:
        raise AssertionError(run.last_stats)
    return run, state, control


def test_reaction_mapping(test, device):
    """Add contact contributions once with fixed rows, interleaved ids and separate theta offsets."""
    for precision in ("fp32", "fp64"):
        run, state, _ = heterogeneous_scene(device, precision)
        producer = next(p for p in run._producers if p.name == "contact")
        contact, blocks = run._contact, run._cloths + run._cables
        residual = np.linspace(-0.5, 0.7, run._complete_r.size, dtype=run._numpy_dtype)
        force = np.linspace(-1, 1, contact.pool.force.result.size, dtype=run._numpy_dtype)
        original = [np.arange(block.raw.size, dtype=run._numpy_dtype) / 8 for block in blocks]
        for reference in (False, True, False):
            for replay in range(2):
                # Same-shape storage changes must not leave a captured descriptor stale.
                if replay:
                    old = blocks[-1].raw
                    blocks[-1].raw = wp.zeros_like(old)
                    old_values = old.numpy().copy()
                for block, values in zip(blocks, original, strict=True):
                    block.raw.assign(values)
                run._complete_r.fill_(0.25)
                contact.pool.force.result.assign(force)
                with (
                    patch.object(contact, "evaluate"),
                    patch.object(contact, "evaluation_diagnostics", return_value={}),
                    patch.object(contact, "_reference_mode", reference),
                ):
                    contact.residual.assign(residual)
                    producer.evaluate(run, state, 0.001, run._complete_r, build_tangent=False)
                np.testing.assert_array_equal(run._complete_r.numpy(), residual + run._numpy_dtype(0.25))
                for block, values in zip(blocks, original, strict=True):
                    expected = values.copy()
                    if reference:
                        mask = block.rows_host >= 0
                        expected[mask] += residual[block.rows_host[mask]]
                    else:
                        n = 3 * block.ids.size
                        expected[:n] -= force[: 3 * run.model.particle_count].reshape(-1, 3)[block.ids.numpy()].ravel()
                        offset = (
                            3 * run.model.particle_count
                            + 6 * run.model.body_count
                            + int(block.reference.get("theta_offset", 0))
                        )
                        expected[n:] -= force[offset : offset + expected.size - n]
                    np.testing.assert_array_equal(block.raw.numpy(), expected)
                if replay:
                    np.testing.assert_array_equal(old.numpy(), old_values)


def test_operator_and_band(test, device):
    """Match independent element sums and Cholesky while rebinding heterogeneous cable buffers."""
    rng = np.random.default_rng(304)
    for precision in ("fp32", "fp64"):
        run, _, _ = heterogeneous_scene(device, precision)
        operator, linear = run._linear._material_operator, run._linear
        n, native = operator.size, run._numpy_dtype
        x = rng.normal(size=n).astype(native)
        scale = rng.uniform(0.5, 2, n).astype(native)
        x_device = wp.array(x, dtype=run._dtype, device=device)
        scale_device = wp.array(scale, dtype=run._dtype, device=device)
        out, magnitude = wp.zeros_like(x_device), wp.zeros_like(x_device)
        active = wp.ones(1, dtype=int, device=device)
        old_plan = linear._masked_pcg_plan
        test.assertIsNotNone(old_plan)
        old_plan._validate_storage()
        retained = []
        for binding in range(3):
            if binding:
                owner = run._cables[-1].objective
                field = "stretch_b" if binding == 1 else "bend_twist_b"
                old = getattr(owner, field)
                retained.append((old, old.numpy().copy()))
                setattr(owner, field, wp.array(old.numpy() * native(1.5), dtype=run._dtype, device=device))
                with test.assertRaisesRegex(ValueError, "storage changed"):
                    old_plan._validate_storage()
            dense, absolute = np.zeros((n, n)), np.zeros((n, n))
            for cable, block in zip(operator.cables, run._cables, strict=True):
                matrices = block.objective.stretch_b.numpy(), block.objective.bend_twist_b.numpy()
                for kind, element, a, b, row_a, row_b in cable.entries():
                    value = float(matrices[kind][element, a, b])
                    dense[row_a, row_b] += value
                    absolute[row_a, row_b] += abs(value)

            def execute(
                operator=operator,
                out=out,
                magnitude=magnitude,
                x_device=x_device,
                scale_device=scale_device,
                active=active,
            ):
                out.zero_()
                magnitude.zero_()
                operator.add_product(x_device, scale_device, out, magnitude=magnitude, active=active)

            buffers = [*operator.buffers(), x_device, scale_device, out, magnitude, active]
            for replay in range(2):
                run._graphs.run("test_material_product", (), buffers, execute)
                expected = scale.astype(float) * (dense @ (scale.astype(float) * x.astype(float)))
                np.testing.assert_allclose(
                    out.numpy(), expected, rtol=64 * np.finfo(native).eps, atol=64 * np.finfo(native).eps
                )
                np.testing.assert_allclose(
                    magnitude.numpy(),
                    scale * (absolute @ abs(scale * x)),
                    rtol=64 * np.finfo(native).eps,
                    err_msg=f"{precision}, binding={binding}, replay={replay}",
                )
            coordinates = wp.array(abs(x), dtype=run._dtype, device=device)
            out.zero_()
            operator.add_abs_coordinates(coordinates, out)
            np.testing.assert_allclose(out.numpy(), absolute @ abs(x), rtol=64 * np.finfo(native).eps)
            active.zero_()
            out.fill_(3)
            magnitude.fill_(4)
            operator.add_product(x_device, scale_device, out, magnitude=magnitude, active=active)
            np.testing.assert_array_equal(out.numpy(), 3)
            np.testing.assert_array_equal(magnitude.numpy(), 4)
            active.fill_(1)
            # Factor the same physical element matrix plus the assembled inertia.
            linear.factor_actor_preconditioner(generation=run._generation, pivot_tolerance=0.0)
            preconditioner = linear._cable_preconditioner
            rows = preconditioner.rows.numpy()
            factors, scales, shifts = (
                preconditioner.factor_values.numpy(),
                linear.scale.numpy(),
                preconditioner.used_shift.numpy(),
            )
            matrix = dense.copy()
            sparse = linear.k_global_scalar_bsr
            offsets, columns, values = sparse.offsets.numpy(), sparse.columns.numpy(), sparse.values.numpy()
            for row in range(n):
                matrix[row, columns[offsets[row] : offsets[row + 1]]] += values[offsets[row] : offsets[row + 1]]
            matrix = scales[:, None] * matrix * scales[None, :]
            band = preconditioner.band.numpy().reshape(-1, 11)
            expected_band, band_bound = np.zeros_like(band, dtype=float), np.zeros_like(band, dtype=float)
            for actor, (start, end) in enumerate(preconditioner.spans.numpy()):
                indices = rows[start:end]
                expected = matrix[np.ix_(indices, indices)].copy()
                expected[np.diag_indices_from(expected)] += linear._lambda + shifts[actor] * abs(np.diag(expected))
                pin = preconditioner.gauge_rows.numpy()[actor]
                if pin >= 0:
                    expected[np.flatnonzero(indices == pin)[0], np.flatnonzero(indices == pin)[0]] += 1
                lower = np.zeros_like(expected)
                for i in range(end - start):
                    for j in range(max(0, i - 10), i + 1):
                        lower[i, j] = factors[start + i, i - j]
                        row, column = indices[i], indices[j]
                        scale_product = float(scales[row]) * float(scales[column])
                        expected_band[start + i, i - j] = scale_product * dense[row, column]
                        band_bound[start + i, i - j] = 64 * np.finfo(native).eps * scale_product * absolute[row, column]
                np.testing.assert_allclose(
                    lower @ lower.T, expected, rtol=256 * np.finfo(native).eps, atol=256 * np.finfo(native).eps
                )
            # Check element terms directly: the inertial diagonal can hide FP32 band errors.
            np.testing.assert_array_less(np.abs(band - expected_band), band_bound + np.finfo(native).tiny)
            for old, values in retained:
                np.testing.assert_array_equal(old.numpy(), values)
        if device.is_cuda:
            test.assertEqual(run._graphs.summary()["segments"]["test_material_product"]["captures"], 3)
        owner = run._cables[-1].objective
        bad = owner.stretch_b
        owner.stretch_b = wp.zeros((1, 6, 6), dtype=run._dtype, device=device)
        with test.assertRaisesRegex(ValueError, "layout"):
            operator.buffers()
        owner.stretch_b = bad


def test_control_boundaries(test, device):
    """Preserve frozen controls and published states across repeated updates and invalid inputs."""
    for precision in ("fp32", "fp64"):
        for count in (0, 1, 8):
            _, run = scene(cloth=count, cables=count, rigid=True, device=device, precision=precision)
            state, control = run.create_state(dt=0.001), run.create_control()
            if count == 0:
                run.step(state, state, control, dt=0.001)
                test.assertTrue(run.last_stats.converged)
                continue
            cable = control._monolithic_cable
            for value in (0.1, -0.2, 0.1):
                torque = np.zeros(cable.torque.size, dtype=run._numpy_dtype)
                torque[::2] = value
                cable.torque.assign(torque)
                frozen = run._snapshot_material_control(control)
                cable.torque.zero_()
                np.testing.assert_array_equal(frozen["torque"].numpy(), torque)
            frozen_before = {name: array.numpy().copy() for name, array in frozen.items()}
            state_before = snapshot(state)
            for field in ("target_position", "target_theta", "torque"):
                original = getattr(cable, field)
                values = original.numpy().copy()
                strided = wp.zeros(2 * original.size, dtype=original.dtype, device=device)[::2]
                strided.assign(original)
                setattr(cable, field, strided)
                run._snapshot_material_control(control)
                setattr(cable, field, original)
                frozen_before = {name: array.numpy().copy() for name, array in frozen.items()}
                invalid = [
                    wp.zeros(original.size + 1, dtype=original.dtype, device=device),
                    wp.zeros(original.size, dtype=int, device=device),
                ]
                if device.is_cuda:
                    invalid.append(wp.array(values, dtype=original.dtype, device="cpu"))
                for bad in invalid:
                    setattr(cable, field, bad)
                    with test.assertRaisesRegex(ValueError, "Invalid cable control"):
                        run._snapshot_material_control(control)
                setattr(cable, field, original)
                for bad in (np.nan, np.inf):
                    original.fill_(original.dtype(bad))
                    with test.assertRaisesRegex(ValueError, "Nonfinite cable control"):
                        run.step(state, state, control, dt=0.001)
                    for name, array in frozen.items():
                        np.testing.assert_array_equal(array.numpy(), frozen_before[name])
                    for name, array in snapshot(state).items():
                        np.testing.assert_array_equal(array, state_before[name])
                original.assign(values)
            cable.torque.fill_(1)
            with test.assertRaisesRegex(ValueError, "physically free theta"):
                run.step(state, state, control, dt=0.001)
            for name, array in snapshot(state).items():
                np.testing.assert_array_equal(array, state_before[name])


def test_gauge_binding(test, device):
    """Rebind raw/Jacobian views and preserve exact null checks and accepted pins."""
    for precision in ("fp32", "fp64"):
        run, _, _ = heterogeneous_scene(device, precision)
        cable, gauges = run._cables[-1], run._gauges
        native = run._numpy_dtype
        run._contact.pool.records.factor_count.zero_()
        accepted = gauges.accepted_pins.numpy().copy()
        raw = np.zeros(cable.raw.size, dtype=native)
        start = 3 * cable.ids.size
        raw[start : start + 2] = (0.25, -0.25)
        jacobian = np.zeros(cable.objective.jacobian.shape, dtype=native)
        jacobian[0, 0, 9:11] = (1, -1)
        for binding in range(2):
            if binding:
                cable.raw = wp.zeros(2 * cable.raw.size, dtype=run._dtype, device=device)[::2]
                cable.objective.jacobian = wp.zeros_like(cable.objective.jacobian)
            cable.raw.assign(raw)
            for null in (True, False, True):
                jacobian[0, 0, 10] = -1 if null else np.nextafter(native(-1), native(0))
                cable.objective.jacobian.assign(jacobian)
                run._settle_cable_gauges()
                expected = cable.rows_host[start] if null else -1
                test.assertEqual(gauges.pins.numpy()[-1], expected)
                test.assertAlmostEqual(
                    run._cable_raw_theta_norms[f"cable_{cable.actor.source_index}_theta"],
                    float(native(np.sqrt(native(0.125)))),
                )
                np.testing.assert_array_equal(gauges.accepted_pins.numpy(), accepted)
            if device.is_cuda:
                captures = run._graphs.summary()["segments"]["cable_gauge_qualify"]["captures"]
                test.assertEqual(captures, binding + 1)
        raw[start] = 1
        cable.raw.assign(raw)
        with test.assertRaisesRegex(_StepFailure, "incompatible net torque"):
            run._settle_cable_gauges()
        np.testing.assert_array_equal(gauges.accepted_pins.numpy(), accepted)


class TestMaterialBatching(unittest.TestCase):
    pass


for function in (
    test_submission_count,
    test_reaction_mapping,
    test_operator_and_band,
    test_control_boundaries,
    test_gauge_binding,
):
    add_function_test(TestMaterialBatching, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
