# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic magnitudes for contact with a frozen step-start weight.

Outputs have the units of the corresponding quantities and do not contain
machine epsilon. These are first-order operation/sensitivity estimates, not
global interval bounds. They assume normal floating-point values, ordinary
round-to-nearest operations, exp relative error <= 4 eps, and log absolute
error <= 4 eps max(1, abs(log(x))). CPU/CUDA qualification must check these
assumptions and that the perturbations resolve the geometric transition.

Input quantization and smoothing bias are separate budgets. In particular,
the Newton equation uses the *stored* frozen coefficient, so pass zero for
its magnitude when estimating that equation's evaluation error. The error
in generating the coefficient belongs to a separate step-start assessment.
"""

from typing import Any

import warp as wp

from ...geometry import GeoType
from .contact_smoothing import orientation_weight, penetration
from .smooth_sdf import _bspline


@wp.func
def _abs_vector(v: Any):
    return type(v)(wp.abs(v[0]), wp.abs(v[1]), wp.abs(v[2]))


@wp.func
def _unit_magnitude(v: Any, magnitude: Any):
    length = wp.length(v)
    n = v / length
    alength = wp.dot(_abs_vector(n), magnitude) + type(length)(5.0) * length
    an = (magnitude + _abs_vector(n) * alength) / length + _abs_vector(n)
    return n, length, an, alength


@wp.func
def analytic_geometry_magnitude(geo: int, scale: Any, y: Any, point_magnitude: Any, phi: Any, g: Any, ell: Any):
    """Return A_phi [m], A_gradient [1]; point_magnitude is componentwise [m].

    BOX ell=0 estimates only an already selected feature. It does not certify
    a sharp-box feature switch. MESH needs its own field evaluation estimate.
    """
    value, gradient = point_magnitude[2], type(y)(0.0)
    if geo == GeoType.SPHERE:
        if wp.length(y) == type(phi)(0.0):
            return type(phi)(wp.inf), type(y)(wp.inf)
        _n, _length, gradient, alength = _unit_magnitude(y, point_magnitude)
        value = alength + wp.abs(phi)
    elif geo == GeoType.BOX and ell > type(ell)(0.0):
        planes = wp.vector(type(ell)(0.0), length=6)
        z = wp.vector(type(ell)(0.0), length=6)
        p = wp.vector(type(ell)(0.0), length=6)
        ad = wp.vector(type(ell)(0.0), length=6)
        ap = wp.vector(type(ell)(0.0), length=6)
        largest = y[0] - scale[0]
        for j in range(6):
            axis = j / 2
            planes[j] = type(ell)(2 * (j % 2) - 1) * y[axis] - scale[axis]
            ad[j] = point_magnitude[axis] + wp.abs(y[axis]) + wp.abs(scale[axis])
            largest = wp.max(largest, planes[j])
        total = type(ell)(0.0)
        for j in range(6):
            z[j] = (planes[j] - largest) / ell
            p[j] = wp.exp(z[j])
            total += p[j]
        p /= total
        value, exponent = type(ell)(0.0), type(ell)(0.0)
        for j in range(6):
            # Softmax cancels the common max shift. The centered sensitivity
            # tends to zero on a resolved face instead of always scaling D/ell.
            mixed = type(ell)(0.0)
            for k in range(6):
                if j != k:
                    mixed += p[k] * (
                        (ad[j] + ad[k]) / ell + type(ell)(2.0) * (wp.abs(z[j]) + wp.abs(z[k])) + type(ell)(8.0)
                    )
            ap[j] = p[j] * (mixed + type(ell)(8.0))
            value += p[j] * ad[j]
            exponent += p[j] * wp.abs(z[j])
        value += ell * (type(ell)(2.0) * exponent + type(ell)(12.0) + type(ell)(5.0) * wp.abs(wp.log(total)))
        value += wp.abs(phi)
        for axis in range(3):
            gradient[axis] = ap[2 * axis] + ap[2 * axis + 1] + p[2 * axis] + p[2 * axis + 1]
    elif geo == GeoType.BOX:
        operands = point_magnitude + _abs_vector(y) + _abs_vector(scale)
        value = wp.dot(_abs_vector(g), operands) + wp.abs(phi)
        if phi > type(phi)(0.0):
            gradient = (operands + _abs_vector(g) * value) / phi + _abs_vector(g)
    elif geo != GeoType.PLANE:
        return type(phi)(wp.inf), type(y)(wp.inf)
    return value, gradient


@wp.func
def _basis_magnitude(index: int, t: Any, magnitude: Any):
    values = _bspline(index, t)
    u = wp.abs(t)
    absolute, derivative = type(t)(0.0), type(t)(0.0)
    if index == 0:
        s = wp.abs(type(t)(1.0) - t)
        absolute, derivative = s * s * s / type(t)(6.0), s * s / type(t)(2.0)
    elif index == 1:
        absolute = (type(t)(3.0) * u * u * u + type(t)(6.0) * u * u + type(t)(4.0)) / type(t)(6.0)
        derivative = type(t)(1.5) * u * u + type(t)(2.0) * u
    elif index == 2:
        absolute = (type(t)(3.0) * u * u * u + type(t)(3.0) * u * u + type(t)(3.0) * u + type(t)(1.0)) / type(t)(6.0)
        derivative = type(t)(1.5) * u * u + u + type(t)(0.5)
    else:
        absolute, derivative = u * u * u / type(t)(6.0), u * u / type(t)(2.0)
    aw = wp.abs(values[1]) * magnitude + type(t)(12.0) * absolute
    ad = wp.abs(values[2]) * magnitude + type(t)(12.0) * derivative
    return values[0], values[1], aw, ad


@wp.func
def _product_magnitude(a: Any, aa: Any, b: Any, ab: Any, c: Any, ac: Any):
    # Three multiplies and serial accumulation over 64 coefficients. Keep the
    # coefficient contributions separate even when the field/gradient cancels.
    return wp.abs(b * c) * aa + wp.abs(a * c) * ab + wp.abs(a * b) * ac + type(a)(68.0) * wp.abs(a * b * c)


@wp.func
def mesh_geometry_magnitude(field: Any, point: Any, point_magnitude: Any):
    """C2 B-spline A_phi [m], A_gradient [1], for interior grid queries.

    Basis derivatives propagate coordinate arithmetic through B and B'. The
    B'' terms are needed for gradient sensitivity, not a contact Hessian.
    These first-order estimates require a resolved grid coordinate. Exterior
    extension and C1 Hermite fields are not qualified by this function.
    """
    value, gradient = type(point[0])(0.0), type(point)(0.0)
    t, at = type(point)(0.0), type(point)(0.0)
    cell = wp.vec3i(0)
    valid = field.kind == 2
    for axis in range(3):
        valid = valid and point[axis] > field.lower[axis] and point[axis] < field.upper[axis]
        f = (point[axis] - field.lower[axis]) * field.inv_dx[axis]
        cell[axis] = wp.clamp(int(wp.floor(f)), 0, field.dims[axis] - 2)
        t[axis] = f - type(point[0])(cell[axis])
        at[axis] = point_magnitude[axis] * field.inv_dx[axis] + type(point[0])(2.0) * wp.abs(f) + wp.abs(t[axis])
    if not valid:
        return type(point[0])(wp.inf), type(point)(wp.inf)
    stride = field.dims + wp.vec3i(2)
    for index in range(64):
        basis = wp.vec3i(index % 4, (index / 4) % 4, index / 16)
        w, d, aw, ad = type(point)(0.0), type(point)(0.0), type(point)(0.0), type(point)(0.0)
        for axis in range(3):
            wi, di, awi, adi = _basis_magnitude(basis[axis], t[axis], at[axis])
            w[axis] = wi
            d[axis] = di * field.inv_dx[axis]
            aw[axis] = awi
            ad[axis] = (adi + wp.abs(di)) * field.inv_dx[axis]
        position = cell + basis
        offset = (position[2] * stride[1] + position[1]) * stride[0] + position[0]
        coefficient = wp.abs(field.coefficients[offset])
        value += coefficient * _product_magnitude(w[0], aw[0], w[1], aw[1], w[2], aw[2])
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            gradient[i] += coefficient * _product_magnitude(d[i], ad[i], w[j], aw[j], w[k], aw[k])
    return value, gradient


@wp.func
def frozen_weight_magnitude(g: Any, ns: Any, eta: Any, gradient_magnitude: Any, source_magnitude: Any, eps: Any):
    """Return A_c [1] and conditioning for generating the step-start coefficient.

    A hard-mask uncertainty interval crossing zero reports infinite condition
    and unit absolute coefficient uncertainty (A_c=1/eps), not smoothness.
    """
    if wp.length(g) == type(eta)(0.0):
        return type(eta)(wp.inf), type(eta)(wp.inf)
    n, length, an, _alength = _unit_magnitude(g, gradient_magnitude)
    a = wp.dot(ns, n)
    aa = wp.dot(_abs_vector(n), source_magnitude) + wp.dot(_abs_vector(ns), an)
    aa += type(eta)(3.0) * wp.dot(_abs_vector(ns), _abs_vector(n))
    condition = eps * wp.length(gradient_magnitude) / length
    if eta == type(eta)(0.0):
        if wp.abs(a) <= eps * aa:
            return type(eta)(1.0) / eps, type(eta)(wp.inf)
        return type(eta)(0.0), condition
    # Include the division's rounding when finding the local derivative bound.
    magnitude = aa + wp.abs(a)
    lo = wp.clamp((a - eps * magnitude) / eta, type(eta)(0.0), type(eta)(1.0))
    hi = wp.clamp((a + eps * magnitude) / eta, type(eta)(0.0), type(eta)(1.0))
    t = wp.clamp(type(eta)(0.5), lo, hi)
    z = type(eta)(1.0) - t
    error = type(eta)(30.0) * t * t * z * z / eta * magnitude
    if a > type(eta)(0.0) and a < eta:
        w, _dw = orientation_weight(a, eta)
        error += type(eta)(12.0) * w
    return error, wp.max(condition, eps * aa / eta)


@wp.func
def _pressure_magnitude(gap: Any, k: Any, smoothing: Any, magnitude: Any, eps: Any):
    p, dp, _ddp = penetration(gap, smoothing)
    energy, load = k * p * p / type(gap)(2.0), k * p * dp
    ap, adp = type(gap)(0.0), type(gap)(0.0)
    if -gap >= smoothing:
        ap = magnitude
    elif smoothing > type(gap)(0.0) and -gap > -smoothing:
        t = (-gap + smoothing) / (type(gap)(2.0) * smoothing)
        at = magnitude / (type(gap)(2.0) * smoothing) + type(gap)(2.0) * wp.abs(t)
        ap = type(gap)(2.0) * smoothing * dp * at + type(gap)(6.0) * p
        adp = type(gap)(6.0) * t * (type(gap)(1.0) - t) * at + type(gap)(6.0) * dp
    ae = k * p * ap + type(gap)(4.0) * energy
    al = k * (dp * ap + p * adp) + type(gap)(4.0) * wp.abs(load)
    # A finite interval also captures the nonzero tail across a support endpoint
    # where the central first derivative vanishes. It does not certify the rest
    # of the floating-point computation as an interval calculation.
    for sign in range(2):
        displaced = gap + type(gap)(2 * sign - 1) * eps * magnitude
        other_p, other_dp, _other_ddp = penetration(displaced, smoothing)
        other_e = k * other_p * other_p / type(gap)(2.0)
        other_l = k * other_p * other_dp
        ae = wp.max(ae, wp.abs(other_e - energy) / eps)
        al = wp.max(al, wp.abs(other_l - load) / eps)
    return energy, load, ae, al


@wp.func
def normal_magnitude(
    gap: Any,
    g: Any,
    k: Any,
    smoothing: Any,
    coefficient: Any,
    gap_magnitude: Any,
    gradient_magnitude: Any,
    coefficient_magnitude: Any,
    eps: Any,
):
    """Return A_E [J], A_r [N] for E=cP and r=-cL*g.

    Set coefficient_magnitude=0 to assess the equation with the stored c.
    A second assessment with A_c includes coefficient-generation uncertainty.
    """
    energy, load, ae, al = _pressure_magnitude(gap, k, smoothing, gap_magnitude, eps)
    ae = coefficient * ae + energy * coefficient_magnitude + wp.abs(coefficient * energy)
    ar = coefficient * load * gradient_magnitude
    ar += _abs_vector(g) * (coefficient * al + load * coefficient_magnitude)
    ar += type(gap)(3.0) * wp.abs(coefficient * load) * _abs_vector(g)
    return ae, ar


@wp.func
def friction_magnitude(
    gap: Any,
    g: Any,
    slip: Any,
    k: Any,
    smoothing: Any,
    mu: Any,
    falloff: Any,
    coefficient: Any,
    gap_magnitude: Any,
    gradient_magnitude: Any,
    slip_magnitude: Any,
    coefficient_magnitude: Any,
    eps: Any,
):
    """Return A_D [J], A_r_f [N] for c*|g| times the C-IE friction law.

    The frozen-weight uncertainty has the same separate role as in
    normal_magnitude. Gradient normalization requires a resolved nonzero g.
    """
    if mu == type(gap)(0.0):
        return type(gap)(0.0), type(g)(0.0)
    if wp.length(g) == type(gap)(0.0):
        return type(gap)(wp.inf), type(g)(wp.inf)
    n, length, an, alength = _unit_magnitude(g, gradient_magnitude)
    linear_gap = gap - wp.dot(g, slip)
    aline = gap_magnitude + wp.dot(_abs_vector(slip), gradient_magnitude) + wp.dot(_abs_vector(g), slip_magnitude)
    aline += type(gap)(3.0) * wp.dot(_abs_vector(g), _abs_vector(slip)) + wp.abs(linear_gap)
    _p, load, _ap, al = _pressure_magnitude(linear_gap, k, smoothing, aline, eps)
    projection = wp.dot(n, slip)
    aprojection = wp.dot(_abs_vector(slip), an) + wp.dot(_abs_vector(n), slip_magnitude)
    aprojection += type(gap)(3.0) * wp.dot(_abs_vector(n), _abs_vector(slip))
    v = slip - projection * n
    av = slip_magnitude + _abs_vector(n) * aprojection + wp.abs(projection) * an
    av += _abs_vector(slip) + wp.abs(projection) * _abs_vector(n)
    rho = wp.length(v)
    arho = wp.length(av) + type(gap)(5.0) * rho
    factor, psi = type(gap)(0.0), type(gap)(0.0)
    if rho < falloff:
        factor = type(gap)(2.0) / falloff - rho / (falloff * falloff)
        psi = rho * rho / falloff - rho * rho * rho / (type(gap)(3.0) * falloff * falloff)
    else:
        factor = type(gap)(1.0) / rho
        psi = rho - falloff / type(gap)(3.0)
    lower = wp.max(falloff, rho - eps * arho)
    afactor = arho / (lower * lower) + type(gap)(6.0) * wp.abs(factor)
    max_dpsi = wp.min(type(gap)(1.0), (rho + eps * arho) * type(gap)(2.0) / falloff)
    apsi = max_dpsi * arho + type(gap)(10.0) * wp.abs(psi)
    amplitude = coefficient * length * mu * load
    aamplitude = mu * (length * load * coefficient_magnitude + coefficient * load * alength + coefficient * length * al)
    aamplitude += type(gap)(6.0) * wp.abs(amplitude)
    ar = wp.abs(amplitude * factor) * av + _abs_vector(v) * (wp.abs(factor) * aamplitude + wp.abs(amplitude) * afactor)
    ar += type(gap)(5.0) * wp.abs(amplitude * factor) * _abs_vector(v)
    ae = wp.abs(amplitude) * apsi + wp.abs(psi) * aamplitude + type(gap)(3.0) * wp.abs(amplitude * psi)
    return ae, ar
