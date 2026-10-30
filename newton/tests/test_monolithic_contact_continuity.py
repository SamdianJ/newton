# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact feature and orientation continuity in native arithmetic."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact import _freeze_orientation, _geometry_type, _law, _response
from newton._src.solvers.monolithic.contact_actor import _point_column, _point_column_magnitude
from newton._src.solvers.monolithic.contact_batch import _ContactBatch, _project_contact_column
from newton._src.solvers.monolithic.contact_pool import _ContactPool
from newton._src.solvers.monolithic.contact_smoothing import box_field
from newton._src.solvers.monolithic.contact_spatial import _ContactSpatial
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, native_inputs, perturb, physical, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _box_query(
    points: wp.array[wp.vec3d],
    values: wp.array[wp.float64],
    gradients: wp.array[wp.vec3d],
):
    i = wp.tid()
    d, g, _h = box_field(wp.vec3d(0.02), points[i], wp.float64(25e-6))
    values[i] = d
    gradients[i] = g


def local_law(device, alignment, current_alignment=None):
    """One penetrating sample, with a continuously rotating source normal."""
    geometry = _geometry_type(wp.float64)()
    for name, value in (
        ("y", [0.0, 0.0, -0.001]),
        ("slip", [0.0, 0.0, 0.0]),
        ("gradient", [0.0, 0.0, 1.0]),
        ("soft_normal", [np.sqrt(1.0 - alignment**2), 0.0, alignment]),
    ):
        setattr(geometry, name, wp.array([value], dtype=wp.vec3d, device=device))
    geometry.distance = wp.array([-0.001], dtype=wp.float64, device=device)
    geometry.uncertified = wp.zeros(1, dtype=int, device=device)
    geometry.orientation = wp.zeros(1, dtype=wp.float64, device=device)
    for name in ("orientation_magnitude", "orientation_condition", "field_magnitude"):
        setattr(geometry, name, wp.zeros(1, dtype=wp.float64, device=device))
    for name in ("gradient_magnitude", "normal_magnitude", "slip_magnitude"):
        setattr(geometry, name, wp.zeros(1, dtype=wp.vec3d, device=device))
    wp.launch(
        _freeze_orientation,
        1,
        [geometry, wp.float64(0.1), wp.float64(np.finfo(float).eps), wp.array([1], dtype=int, device=device)],
        device=device,
    )
    if current_alignment is not None:
        geometry.soft_normal.assign([[np.sqrt(1 - current_alignment**2), 0, current_alignment]])
    energy = wp.zeros((1, 2), dtype=wp.float64, device=device)
    residual = wp.zeros(1, dtype=wp.vec3d, device=device)
    flags = wp.zeros(2, dtype=int, device=device)
    empty = wp.empty(0, dtype=wp.float64, device=device)
    ints = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        _law,
        1,
        [
            geometry,
            geometry,
            ints,
            wp.zeros(1, dtype=wp.float64, device=device),
            wp.ones(1, dtype=wp.float64, device=device),
            wp.float64(0),
            wp.float64(1),
            wp.float64(0),
            wp.float64(0),
            wp.float64(1e-5),
            2,
            flags,
            energy,
            empty,
            wp.zeros_like(residual),
            wp.zeros_like(residual),
            wp.zeros((1, 3), dtype=wp.float64, device=device),
            residual,
            wp.zeros((1, 4), dtype=wp.float64, device=device),
            wp.zeros(1, dtype=int, device=device),
            3,
            0,
            wp.zeros(1, dtype=wp.float64, device=device),
            ints,
            ints,
            wp.empty(0, dtype=wp.vec3d, device=device),
            wp.array([1], dtype=int, device=device),
            _response(wp.float64, 1, device),
            wp.zeros((1, 2), dtype=wp.float64, device=device),
            wp.float64(np.finfo(float).eps),
        ],
        device=device,
    )
    if flags.numpy()[0]:
        raise AssertionError(flags.numpy())
    return energy.numpy()[0, 0], residual.numpy()[0]


def test_box_feature_continuity(test, device):
    points = wp.array([[-0.019 + 1e-10, 0.0, -0.019], [-0.019 - 1e-10, 0.0, -0.019]], dtype=wp.vec3d, device=device)
    values = wp.zeros(2, dtype=wp.float64, device=device)
    gradients = wp.zeros(2, dtype=wp.vec3d, device=device)
    wp.launch(_box_query, 2, [points, values, gradients], device=device)
    test.assertLess(np.linalg.norm(gradients.numpy()[0] - gradients.numpy()[1]), 1e-4)


def test_orientation_continuity(test, device):
    first = local_law(device, -1e-10)
    second = local_law(device, 1e-10)
    test.assertLess(abs(first[0] - second[0]), 1e-12)
    test.assertLess(np.linalg.norm(first[1] - second[1]), 1e-8)


@wp.kernel
def _field(
    points: wp.array[Any],
    scales: wp.array[Any],
    width: Any,
    fields: wp.array[Any],
    gradients: wp.array[Any],
    hessians: wp.array[Any],
):
    i = wp.tid()
    phi, g, h = box_field(scales[i], points[i], width)
    fields[i] = phi
    gradients[i] = g
    hessians[i] = h


def reference(point, normal, scale, width, eta):
    """Independent extended-precision scalar definition, including pressure smoothing."""
    point, normal, scale = (np.asarray(x, dtype=np.longdouble) for x in (point, normal, scale))
    width, eta = np.longdouble(width), np.longdouble(eta)
    planes = np.r_[point - scale, -point - scale]
    p = np.exp((planes - planes.max()) / width)
    p /= p.sum()
    phi = planes.max() - width * np.log(p[planes.argmax()])
    g = p[:3] - p[3:]
    h = (np.diag(p[:3] + p[3:]) - np.outer(g, g)) / width
    alignment = normal @ g / np.linalg.norm(g)
    t = np.clip(alignment / eta, 0, 1)
    w = (1 - t) ** 3 * (1 + 3 * t + 6 * t * t)
    pressure = -phi
    if pressure <= -np.longdouble(1e-4):
        pressure = np.longdouble(0)
    elif pressure < np.longdouble(1e-4):
        u = (pressure + np.longdouble(1e-4)) / np.longdouble(2e-4)
        pressure = np.longdouble(1e-4) * u**3 * (2 - u)
    return phi, g, h, np.longdouble(5e3) * w * pressure**2


def test_field_and_potential_derivatives(test, device):
    """Match the field and its derivatives on identical representable inputs."""
    points = np.array([[0.019, 0.019, 0], [0.019, 0.019, 0.019], [0.019, 0, 0], [0.01999, 0.01999, 0]])
    for dtype, vector, matrix, native in (
        (wp.float64, wp.vec3d, wp.mat33d, np.float64),
        (wp.float32, wp.vec3, wp.mat33, np.float32),
    ):
        x = points.astype(native)
        b = np.tile([0.02] * 3, (len(x), 1)).astype(native)
        ell = native(25e-6)
        outputs = [wp.empty(len(x), dtype=t, device=device) for t in (dtype, vector, matrix)]
        wp.launch(
            _field,
            len(x),
            [wp.array(x, dtype=vector, device=device), wp.array(b, dtype=vector, device=device), dtype(ell), *outputs],
            device=device,
        )
        phi, g, h = (a.numpy() for a in outputs)
        tol = 3e-5 if native == np.float32 else 1e-10
        for i in range(len(x)):
            expected = reference(x[i], [1, 0, 0], b[i], ell, 0.1)
            for actual, wanted in zip((phi[i], g[i], h[i]), expected[:3], strict=True):
                np.testing.assert_allclose(actual, np.asarray(wanted, dtype=float), rtol=tol, atol=tol * 1e-5)
            for axis in range(3):
                step = np.zeros(3, dtype=np.longdouble)
                step[axis] = 1e-9
                plus = reference(x[i] + step, [1, 0, 0], b[i], ell, 0.1)
                minus = reference(x[i] - step, [1, 0, 0], b[i], ell, 0.1)
                np.testing.assert_allclose(
                    g[i, axis], float((plus[0] - minus[0]) / 2e-9), rtol=max(tol, 1e-7), atol=tol
                )
                np.testing.assert_allclose(
                    h[i, :, axis], np.asarray((plus[1] - minus[1]) / 2e-9, dtype=float), rtol=max(tol, 1e-7), atol=tol
                )


def test_step_frozen_orientation(test, device):
    """Changing the current source normal cannot change this step's coefficient."""
    expected = local_law(device, 0.05)
    for alignment in (-1.0, -0.1, 0.0, 0.05, 0.1, 1.0):
        actual = local_law(device, 0.05, current_alignment=alignment)
        test.assertAlmostEqual(actual[0], 2.5e-7, places=15)
        np.testing.assert_array_equal(actual[1], expected[1])
        test.assertAlmostEqual(actual[1][2], -0.0005, places=15)


def test_conservative_projection(test, device):
    """Free rotations and deforming source normals contribute their complete virtual work."""
    for kind in ("A-R", "R-R", "R-T", "R-static"):
        with test.subTest(kind=kind):
            run = contact_solver(
                contact_scene(device, kind),
                friction_coefficient=0.0,
                alignment_policy="frozen",
                box_smoothing=25e-6,
                alignment_smoothing=0.1,
            )
            state = contact_state(run, kind)
            batch = native_batch(run)
            values = physical(state)
            direction = np.zeros(run._layout.scalar_dof_count)
            rigid = run._rigid.actors[0]
            direction[rigid.dof_start + 5] = 1
            perturb(run, state, values, direction, 0.05)
            values = physical(state)
            query(batch, run, state, start=True)
            query(batch, run, state)
            test.assertEqual(int(batch.flags.numpy()[0]), 0)
            r = batch.pool.residual.result.numpy().copy()
            b = batch_matrix(batch)
            test.assertGreater(np.linalg.norm(r), 1e-5)
            np.testing.assert_allclose(b, b.T, rtol=1e-10, atol=1e-8)
            test.assertGreaterEqual(np.linalg.eigvalsh(b).min(), -1e-9 * max(1, np.linalg.norm(b)))
            pairs = batch.pool.records.pairs.numpy()[: int(batch.pool.records.count.numpy()[0])]
            np.testing.assert_allclose(pairs.sum(axis=1), 0, atol=1e-10)
            for axis in range(len(r)):
                direction = np.eye(len(r))[axis]
                energies = []
                for sign in (-1, 1):
                    perturb(run, state, values, direction, sign * 1e-8)
                    query(batch, run, state)
                    test.assertEqual(int(batch.flags.numpy()[0]), 0)
                    energies.append(batch.pool.energy.result.numpy()[0])
                derivative = (energies[1] - energies[0]) / 2e-8
                np.testing.assert_allclose(r[axis], derivative, rtol=2e-5, atol=2e-6)


@wp.kernel
def _projection_probe(
    candidates: Any,
    kinds: wp.array[int],
    indices: wp.array[int],
    centers: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    local_point: Any,
):
    point = centers[0] + local_point
    column = _point_column(0, 4, point, kinds, indices, centers, jt, jw, centers)
    magnitude = _point_column_magnitude(0, 4, point, kinds, indices, centers, jt, jw, centers)
    zero = type(point)(0.0)
    force = type(point)(1.0, 0.0, 0.0)
    _project_contact_column(
        candidates,
        0,
        0,
        0,
        column,
        force,
        force,
        zero,
        zero,
        False,
        type(point[0])(0.0),
        0,
        wp.identity(n=3, dtype=type(point[0])),
        zero,
        magnitude,
    )


def test_translated_torque_magnitude(test, device):
    """Identical input precision: a short arm at large COM retains its projection error estimate."""
    for dtype, vector, native in ((wp.float32, wp.vec3, np.float32), (wp.float64, wp.vec3d, np.float64)):
        for kind in (0, 1):
            for translation in (0, 1000):
                pool = _ContactPool(1, 1, 1, 1, 1, 1, dtype, device)
                jt = wp.zeros((1, 6), dtype=vector, device=device)
                angular = np.zeros((1, 6, 3), dtype=native)
                angular[0, 4, 1] = 1
                wp.launch(
                    _projection_probe,
                    1,
                    [
                        pool.candidates,
                        wp.array([kind], dtype=int, device=device),
                        wp.array([0], dtype=int, device=device),
                        wp.array([[0, 0, translation]], dtype=vector, device=device),
                        jt,
                        wp.array(angular, dtype=vector, device=device),
                        vector(0, 0, native(0.01)),
                    ],
                    device=device,
                )
                actual = pool.candidates.residual.numpy()[0, 0]
                magnitude = pool.candidates.magnitude.numpy()[0, 0]
                expected = np.longdouble(native(0.01))
                test.assertLessEqual(abs(np.longdouble(actual) - expected), np.finfo(native).eps * magnitude)


def test_frozen_candidate_refresh(test, device):
    """New spatial candidates retain their original step coefficient, including after restart."""
    run = contact_solver(
        contact_scene(device, "A-R"),
        alignment_policy="frozen",
        box_smoothing=25e-6,
        alignment_smoothing=0.1,
        friction_coefficient=0.0,
    )
    state = contact_state(run, "A-R")
    com = state._monolithic_coupled.rigid_com
    original = com.numpy().copy()
    com.assign(original + np.array([0.3, 0, 0]))
    inputs, options = native_inputs(run, state)
    batch = _ContactBatch(run._contact, batch_capacity=7, capacity=None, use_cuda_graph=device.is_cuda)
    batch.begin_step(*inputs, **options)
    frozen = batch.start.orientation.numpy().copy()
    spatial = _ContactSpatial(
        batch,
        run._kinematics,
        run._kinematics.gather_q(state),
        inputs[0],
        inputs[1],
        **options,
        capacity=run._contact.count,
    )
    spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=com)
    spatial.check()
    test.assertEqual(int(spatial.count.numpy()[0]), 0)
    for _ in range(2):
        com.assign(original)
        inputs, options = native_inputs(run, state)
        batch.evaluate(*inputs, dt=0.001, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        reference = [p.result.numpy().copy() for p in (batch.pool.energy, batch.pool.residual, batch.pool.force)]
        test.assertGreater(np.linalg.norm(reference[1]), 1e-4)
        spatial.update(run._kinematics.gather_q(state), inputs[0], rigid_com=com)
        spatial.check()
        test.assertGreater(int(spatial.count.numpy()[0]), 0)
        batch.evaluate(*inputs, dt=0.001, selection=spatial, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
        for actual, expected in zip((batch.pool.energy, batch.pool.residual, batch.pool.force), reference, strict=True):
            np.testing.assert_allclose(actual.result.numpy(), expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(batch.start.orientation.numpy(), frozen)
        com.assign(original + np.array([0.3, 0, 0]))
        inputs, options = native_inputs(run, state)
        batch.begin_step(*inputs, **options)
        np.testing.assert_array_equal(batch.start.orientation.numpy(), frozen)


def test_smoothing_identity_and_budget(test, device):
    """Widths change physics/checkpoint identity and obey the frozen local geometry budget."""
    model = contact_scene(device, "A-R")
    original = contact_solver(model)
    state = original.create_state(dt=0.001)
    smooth = contact_solver(
        model, alignment_policy="frozen", box_smoothing=25e-6, alignment_smoothing=0.1, friction_coefficient=0.0
    )
    test.assertTrue(smooth._contact.conservative_energy)
    with test.assertRaises(ValueError):
        smooth._states.validate(state)
    for options in (
        {"box_smoothing": -0.1},
        {"box_smoothing": 0.1},
        {"alignment_smoothing": 1.0},
        {"alignment_smoothing": float("nan")},
        {"alignment_policy": "guard", "box_smoothing": 25e-6},
        {"alignment_policy": "mask", "box_smoothing": 25e-6},
        {"alignment_policy": "guard", "alignment_smoothing": 0.1},
        {"alignment_policy": "mask", "alignment_smoothing": 0.1},
        {"alignment_policy": "frozen", "box_smoothing": 0.1},
    ):
        with test.assertRaises(ValueError):
            contact_solver(model, **options)
    ell = 25e-6
    for dimension in (1, 2, 3):
        # Actual zero set on face, edge and corner diagonals, not the LSE max bound.
        lo, hi = 0.019, 0.02
        for _ in range(50):
            coordinate = (lo + hi) / 2
            point = np.zeros(3)
            point[:dimension] = coordinate
            phi = reference(point, [1, 0, 0], [0.02] * 3, ell, 0.1)[0]
            if phi > 0:
                hi = coordinate
            else:
                lo = coordinate
        offset = np.sqrt(dimension) * (0.02 - (lo + hi) / 2)
        test.assertLess(offset, 1e-4)
        test.assertAlmostEqual(offset, np.sqrt(dimension) * ell * np.log(dimension), delta=1e-14)
    # An ordinary face, with competing planes at least ten widths farther away.
    point = np.array([0.0199, 0.0199 - 10 * ell, 0])
    phi, g, _, _ = reference(point, [-1, 0, 0], [0.02] * 3, ell, 0.1)
    force = -float(phi) * np.asarray(g, dtype=float)
    test.assertLess(np.linalg.norm(force - np.array([0.0001, 0, 0])) / 0.0001, 0.01)


class TestContactContinuity(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_box_feature_continuity,
        test_orientation_continuity,
        test_step_frozen_orientation,
        test_field_and_potential_derivatives,
        test_conservative_projection,
        test_frozen_candidate_refresh,
        test_translated_torque_magnitude,
        test_smoothing_identity_and_budget,
    ):
        add_function_test(TestContactContinuity, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
