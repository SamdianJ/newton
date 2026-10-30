# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify local contact arithmetic separately from frozen-coefficient error."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.geometry import GeoType
from newton._src.solvers.monolithic.contact import _geometry_type, _law, _response
from newton._src.solvers.monolithic.contact_pool import _ContactPool
from newton._src.solvers.monolithic.contact_roundoff import (
    analytic_geometry_magnitude,
    friction_magnitude,
    frozen_weight_magnitude,
    mesh_geometry_magnitude,
    normal_magnitude,
)
from newton._src.solvers.monolithic.contact_smoothing import box_field, orientation_weight, penetration
from newton._src.solvers.monolithic.smooth_sdf import _sdf_type, smooth_query
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _analytic_samples(
    points: wp.array[Any],
    normals: wp.array[Any],
    slips: wp.array[Any],
    coefficients: wp.array[Any],
    scale: Any,
    parameters: wp.array[Any],
    eps: Any,
    values: wp.array2d[Any],
    magnitudes: wp.array2d[Any],
):
    i = wp.tid()
    ell, eta, k = parameters[0], parameters[1], parameters[2]
    smoothing, mu, falloff = parameters[3], parameters[4], parameters[5]
    phi, g, _h = box_field(scale, points[i], ell)
    p, dp, _ddp = penetration(phi, smoothing)
    n = g / wp.length(g)
    w, _dw = orientation_weight(wp.dot(normals[i], n), eta)
    pp, dpp, _ddpp = penetration(phi - wp.dot(g, slips[i]), smoothing)
    v = slips[i] - wp.dot(n, slips[i]) * n
    rho = wp.length(v)
    factor, psi = type(ell)(0.0), type(ell)(0.0)
    if rho < falloff:
        factor = type(ell)(2.0) / falloff - rho / (falloff * falloff)
        psi = rho * rho / falloff - rho * rho * rho / (type(ell)(3.0) * falloff * falloff)
    else:
        factor = type(ell)(1.0) / rho
        psi = rho - falloff / type(ell)(3.0)
    c = coefficients[i]
    rn = -c * k * p * dp * g
    amplitude = c * wp.length(g) * mu * k * pp * dpp
    rf = amplitude * factor * v
    ap, ag = analytic_geometry_magnitude(int(GeoType.BOX), scale, points[i], type(g)(0.0), phi, g, ell)
    ac, _condition = frozen_weight_magnitude(g, normals[i], eta, ag, type(g)(0.0), eps)
    # The stored coefficient is identical in both arithmetic precisions. Its
    # generation error is checked in column 12, never mixed into the force A.
    ae, ar = normal_magnitude(phi, g, k, smoothing, c, ap, ag, type(ell)(0.0), eps)
    ad, arf = friction_magnitude(
        phi, g, slips[i], k, smoothing, mu, falloff, c, ap, ag, type(g)(0.0), type(ell)(0.0), eps
    )
    values[i, 0] = phi
    magnitudes[i, 0] = ap
    values[i, 4] = c * k * p * p / type(ell)(2.0)
    magnitudes[i, 4] = ae
    values[i, 8] = amplitude * psi
    magnitudes[i, 8] = ad
    values[i, 12] = w
    magnitudes[i, 12] = ac
    for axis in range(3):
        values[i, 1 + axis] = g[axis]
        magnitudes[i, 1 + axis] = ag[axis]
        values[i, 5 + axis] = rn[axis]
        magnitudes[i, 5 + axis] = ar[axis]
        values[i, 9 + axis] = rf[axis]
        magnitudes[i, 9 + axis] = arf[axis]


def _analytic_reference(point, normal, slip, coefficient, scale, parameters):
    """Longdouble scalar definitions, independent of the Warp implementation."""
    ell, eta, stiffness, smoothing, mu, falloff = parameters
    planes = np.r_[point - scale, -point - scale]
    probabilities = np.exp((planes - planes.max()) / ell)
    total = probabilities.sum()
    probabilities /= total
    phi = planes.max() + ell * np.log(total)
    gradient = probabilities[:3] - probabilities[3:]
    length = np.sqrt(gradient @ gradient)
    direction = gradient / length
    t = np.clip((normal @ direction) / eta, 0, 1)
    weight = (1 - t) ** 3 * (1 + 3 * t + 6 * t * t)

    def pressure(gap):
        if gap <= -smoothing:
            return -gap, np.longdouble(1)
        if gap >= smoothing:
            return np.longdouble(0), np.longdouble(0)
        u = (smoothing - gap) / (2 * smoothing)
        return smoothing * u**3 * (2 - u), u * u * (3 - 2 * u)

    p, dp = pressure(phi)
    friction_p, friction_dp = pressure(phi - gradient @ slip)
    tangent = slip - (direction @ slip) * direction
    rho = np.sqrt(tangent @ tangent)
    if rho < falloff:
        factor = 2 / falloff - rho / falloff**2
        psi = rho**2 / falloff - rho**3 / (3 * falloff**2)
    else:
        factor, psi = 1 / rho, rho - falloff / 3
    energy = coefficient * stiffness * p * p / 2
    residual = -coefficient * stiffness * p * dp * gradient
    amplitude = coefficient * length * mu * stiffness * friction_p * friction_dp
    return np.r_[phi, gradient, energy, residual, amplitude * psi, amplitude * factor * tangent, weight]


def _check_magnitudes(test, actual, expected, magnitudes, dtype):
    test.assertTrue(np.isfinite(actual).all())
    test.assertTrue(np.isfinite(magnitudes).all())
    test.assertTrue((magnitudes >= 0).all())
    error = np.abs(actual.astype(np.longdouble) - expected)
    allowance = np.longdouble(np.finfo(dtype).eps) * magnitudes.astype(np.longdouble)
    test.assertTrue((error <= allowance).all(), f"Uncovered error: {np.max(error - allowance)}")


def test_lse_magnitudes(test, device, dtype):
    native, vector = (np.float64, wp.vec3d) if dtype == wp.float64 else (np.float32, wp.vec3)
    # Quantize inputs once, then use those exact values for the independent
    # reference. This excludes asset/state quantization from evaluation error.
    scale = np.array([0.02, 0.025, 0.03], dtype=np.float32)
    parameters = np.array([25e-6, 0.1, 1e4, 1e-4, 0.5, 1e-5], dtype=np.float32)
    rng = np.random.default_rng(731)
    points = []
    for count in (1, 2, 3):
        for gap in (-3e-4, -1e-4, -1e-7, 0, 1e-7, 1e-4):
            for sign in (-1, 1):
                point = np.zeros(3)
                point[:count] = sign * (scale[:count].astype(float) + gap)
                points.append(point)
    for _ in range(12):
        points.append(scale + rng.normal(0, 25e-6, 3) - 1e-4)
    points = np.asarray(points, dtype=np.float32)
    normals = []
    alignments = (-1e-6, 0, 1e-6, 0.05, 0.1 - 1e-6, 0.1, 0.1 + 1e-6)
    for i, point in enumerate(points):
        result = _analytic_reference(
            point.astype(np.longdouble),
            np.array([1, 0, 0]),
            np.zeros(3),
            1,
            scale.astype(np.longdouble),
            parameters.astype(np.longdouble),
        )
        normal = np.asarray(result[1:4], dtype=float)
        normal /= np.linalg.norm(normal)
        tangent = np.cross(normal, [0, 1, 0])
        tangent /= np.linalg.norm(tangent)
        a = alignments[i % len(alignments)]
        normals.append(a * normal + np.sqrt(1 - a * a) * tangent)
    normals = np.asarray(normals, dtype=np.float32)
    slips = (rng.normal(size=points.shape) * 1e-5).astype(np.float32)
    slips[::8] = 0
    coefficients = np.resize(np.array([0, 0.2, 0.5, 1], dtype=np.float32), len(points))
    values = wp.empty((len(points), 13), dtype=dtype, device=device)
    magnitudes = wp.empty_like(values)
    wp.launch(
        _analytic_samples,
        len(points),
        [
            *(wp.array(a, dtype=vector, device=device) for a in (points, normals, slips)),
            wp.array(coefficients, dtype=dtype, device=device),
            vector(scale),
            wp.array(parameters, dtype=dtype, device=device),
            dtype(np.finfo(native).eps),
            values,
            magnitudes,
        ],
        device=device,
    )
    expected = np.array(
        [
            _analytic_reference(
                *(a.astype(np.longdouble) for a in (point, normal, slip)),
                np.longdouble(c),
                scale.astype(np.longdouble),
                parameters.astype(np.longdouble),
            )
            for point, normal, slip, c in zip(points, normals, slips, coefficients, strict=True)
        ]
    )
    actual, amounts = values.numpy(), magnitudes.numpy()
    _check_magnitudes(test, actual, expected, amounts, native)
    np.testing.assert_array_equal(actual[coefficients == 0, 4:12], 0)
    np.testing.assert_array_equal(amounts[coefficients == 0, 4:12], 0)


@wp.kernel
def _mesh_samples(field: Any, points: wp.array[Any], values: wp.array2d[Any], magnitudes: wp.array2d[Any]):
    i = wp.tid()
    value, gradient, _h = smooth_query(field, points[i])
    av, ag = mesh_geometry_magnitude(field, points[i], type(points[i])(0.0))
    values[i, 0] = value
    magnitudes[i, 0] = av
    for axis in range(3):
        values[i, 1 + axis] = gradient[axis]
        magnitudes[i, 1 + axis] = ag[axis]


def _mesh_reference(point, lower, inv_dx, dims, coefficients):
    f = (point - lower) * inv_dx
    cell = np.clip(np.floor(f).astype(int), 0, dims - 2)
    weights, derivatives = [], []
    for t in f - cell:
        weights.append(
            [(1 - t) ** 3 / 6, (3 * t**3 - 6 * t * t + 4) / 6, (-3 * t**3 + 3 * t * t + 3 * t + 1) / 6, t**3 / 6]
        )
        derivatives.append([-((1 - t) ** 2) / 2, 1.5 * t * t - 2 * t, -1.5 * t * t + t + 0.5, t * t / 2])
    value, gradient = np.longdouble(0), np.zeros(3, dtype=np.longdouble)
    for index in np.ndindex(4, 4, 4):
        x, y, z = index
        c = coefficients[cell[2] + z, cell[1] + y, cell[0] + x]
        w = np.array([weights[a][index[a]] for a in range(3)])
        d = np.array([derivatives[a][index[a]] * inv_dx[a] for a in range(3)])
        value += c * np.prod(w)
        for a in range(3):
            gradient[a] += c * d[a] * w[(a + 1) % 3] * w[(a + 2) % 3]
    return np.r_[value, gradient]


def test_mesh_magnitudes(test, device, dtype):
    native, vector = (np.float64, wp.vec3d) if dtype == wp.float64 else (np.float32, wp.vec3)
    rng = np.random.default_rng(901)
    dims = np.array([6, 7, 5])
    lower = np.array([-0.03, -0.04, -0.05], dtype=np.float32)
    upper = -lower
    inv_dx = ((dims - 1) / (upper - lower)).astype(np.float32)
    coefficients = rng.normal(0, 0.002, tuple((dims + 2)[::-1])).astype(np.float32)
    points = rng.uniform(lower * 0.8, upper * 0.8, (12, 3)).astype(np.float32)
    for i, delta in enumerate((-1e-6, 0, 1e-6)):
        points[i] = lower + (np.array([2, 3, 2]) + delta) / inv_dx
    field = _sdf_type(dtype)()
    field.kind, field.dims = 2, wp.vec3i(dims)
    field.lower, field.upper, field.inv_dx = vector(lower), vector(upper), vector(inv_dx)
    field.coefficients = wp.array(coefficients.ravel(), dtype=dtype, device=device)
    values = wp.empty((len(points), 4), dtype=dtype, device=device)
    magnitudes = wp.empty_like(values)
    wp.launch(
        _mesh_samples,
        len(points),
        [field, wp.array(points, dtype=vector, device=device), values, magnitudes],
        device=device,
    )
    expected = np.array(
        [
            _mesh_reference(
                p.astype(np.longdouble),
                lower.astype(np.longdouble),
                inv_dx.astype(np.longdouble),
                dims,
                coefficients.astype(np.longdouble),
            )
            for p in points
        ]
    )
    _check_magnitudes(test, values.numpy(), expected, magnitudes.numpy(), native)


@wp.kernel
def _weight_diagnostics(normals: wp.array[Any], eps: Any, output: wp.array2d[Any]):
    i = wp.tid()
    g = type(normals[i])(0.0, 0.0, 1.0)
    ac, condition = frozen_weight_magnitude(g, normals[i], type(eps)(0.0), type(g)(1.0), type(g)(0.0), eps)
    output[i, 0] = ac
    output[i, 1] = condition
    # The diagnostic ambiguity does not enter the equation for stored c=0.
    ae, ar = normal_magnitude(
        type(eps)(-0.001),
        g,
        type(eps)(1e4),
        type(eps)(1e-4),
        type(eps)(0.0),
        type(eps)(1.0),
        type(g)(1.0),
        type(eps)(0.0),
        eps,
    )
    output[i, 2] = ae
    output[i, 3] = wp.length(ar)


def test_frozen_weight_classification(test, device, dtype):
    native, vector = (np.float64, wp.vec3d) if dtype == wp.float64 else (np.float32, wp.vec3)
    eps = np.finfo(native).eps
    normals = wp.array([[1, 0, 0], [1, 0, eps], [0, 0, -1], [0, 0, 1]], dtype=vector, device=device)
    output = wp.empty((4, 4), dtype=dtype, device=device)
    wp.launch(_weight_diagnostics, 4, [normals, dtype(eps), output], device=device)
    values = output.numpy()
    np.testing.assert_array_equal(values[:2, 0] * eps, 1)
    test.assertTrue(np.isinf(values[:2, 1]).all())
    np.testing.assert_array_equal(values[2:, 0], 0)
    test.assertTrue(np.isfinite(values[2:, 1]).all())
    np.testing.assert_array_equal(values[:, 2:], 0)


def _law_probe(device, dtype, distance, gradient, *, start_uncertified=0, smoothing=0.0, offset=0.0, stiffness=1.0):
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    native = np.float64 if dtype == wp.float64 else np.float32
    current, start = _geometry_type(dtype)(), _geometry_type(dtype)()
    for geometry, gap, uncertified in ((current, distance, 0), (start, 1.0, start_uncertified)):
        for name, value in (
            ("y", [0, 0, gap]),
            ("slip", [0, 0, 0]),
            ("gradient", gradient),
            ("soft_normal", [0, 0, -1]),
            ("gradient_magnitude", [100, 100, 100]),
            ("slip_magnitude", [0, 0, 0]),
        ):
            setattr(geometry, name, wp.array([value], dtype=vector, device=device))
        geometry.distance = wp.array([gap], dtype=dtype, device=device)
        geometry.field_magnitude = wp.array([0.1], dtype=dtype, device=device)
        geometry.uncertified = wp.array([uncertified], dtype=int, device=device)
        geometry.orientation = wp.ones(1, dtype=dtype, device=device)
    integer = wp.zeros(1, dtype=int, device=device)
    scalar = wp.zeros(1, dtype=dtype, device=device)
    response = _response(dtype, 1, device)
    flags = wp.zeros(2, dtype=int, device=device)
    energy = wp.zeros((1, 2), dtype=dtype, device=device)
    energy_magnitude = wp.zeros_like(energy)
    residual = wp.zeros(1, dtype=vector, device=device)
    wp.launch(
        _law,
        1,
        [
            current,
            start,
            integer,
            scalar,
            wp.ones_like(scalar),
            dtype(0),
            dtype(stiffness),
            dtype(smoothing),
            dtype(0),
            dtype(1e-5),
            2,
            flags,
            energy,
            wp.empty(0, dtype=dtype, device=device),
            wp.zeros_like(residual),
            wp.zeros_like(residual),
            wp.zeros((1, 3), dtype=dtype, device=device),
            residual,
            wp.zeros((1, 4), dtype=dtype, device=device),
            integer,
            3,
            0,
            wp.array([offset], dtype=dtype, device=device),
            integer,
            integer,
            wp.empty(0, dtype=vector, device=device),
            wp.array([1], dtype=int, device=device),
            response,
            energy_magnitude,
            dtype(np.finfo(native).eps),
        ],
        device=device,
    )
    return (
        flags.numpy(),
        energy.numpy(),
        residual.numpy(),
        energy_magnitude.numpy(),
        response.residual_magnitude.numpy(),
    )


def test_frozen_start_domain(test, device, dtype):
    """Reject a loaded frozen normal whose step-start field was uncertified, even without friction."""
    flags, energy, residual, _, _ = _law_probe(device, dtype, -0.001, [0, 0, 1], start_uncertified=1)
    test.assertEqual(int(flags[0]), 5)
    np.testing.assert_array_equal(energy, 0)
    np.testing.assert_array_equal(residual, 0)


@wp.kernel
def _corner_query(width: Any, distance: wp.array[Any], gradient: wp.array[Any]):
    point = gradient.dtype(type(width)(0.02))
    value, g, _h = box_field(point, point, width)
    distance[0] = value
    gradient[0] = g


def test_zero_residual_magnitude(test, device, dtype):
    """Retain near-support arithmetic uncertainty when the computed residual is exactly zero."""
    native, vector = (np.float64, wp.vec3d) if dtype == wp.float64 else (np.float32, wp.vec3)
    distance = wp.empty(1, dtype=dtype, device=device)
    gradient = wp.empty(1, dtype=vector, device=device)
    width = native(25e-6)
    wp.launch(_corner_query, 1, [dtype(width), distance, gradient], device=device)
    phi, g = distance.numpy()[0], gradient.numpy()[0]
    flags, energy, residual, ae, ar = _law_probe(device, dtype, phi, g, offset=phi, stiffness=1e8)
    test.assertEqual(int(flags[0]), 0)
    np.testing.assert_array_equal(residual, 0)
    with test.subTest(component="local law"):
        test.assertGreater(float(ar[0, 0]), 0)
        reference_gap = np.longdouble(width) * np.log(np.longdouble(3)) - np.longdouble(phi)
        reference_r = np.longdouble(1e8) * min(reference_gap, 0) / 3
        test.assertLessEqual(abs(reference_r), np.longdouble(np.finfo(native).eps) * np.longdouble(ar[0, 0]))
    with test.subTest(component="pool admission"):
        pool = _ContactPool(1, 1, 1, 1, 1, 1, dtype, device)
        pool.candidates.count.fill_(1)
        pool.candidates.support_counts.fill_(1)
        pool.candidates.energy.assign(energy)
        pool.candidates.energy_magnitude.assign(ae)
        # Exercise admission independently of the local-law assertion above.
        amount = max(float(ar[0, 0]), 1.0)
        pool.candidates.magnitude.fill_(amount)
        pool.evaluate()
        test.assertEqual(int(pool.records.flags.numpy()[0]), 0)
        test.assertEqual(int(pool.records.count.numpy()[0]), 1)
        test.assertEqual(float(pool.magnitude.numpy()[0]), native(amount))
        np.testing.assert_array_equal(pool.residual.result.numpy(), 0)


class TestContactRoundoff(unittest.TestCase):
    pass


for _dtype, _suffix in ((wp.float32, "fp32"), (wp.float64, "fp64")):
    for _test in (
        test_lse_magnitudes,
        test_mesh_magnitudes,
        test_frozen_weight_classification,
        test_frozen_start_domain,
        test_zero_residual_magnitude,
    ):
        add_function_test(
            TestContactRoundoff, f"{_test.__name__}_{_suffix}", _test, devices=get_test_devices(), dtype=_dtype
        )


if __name__ == "__main__":
    unittest.main()
