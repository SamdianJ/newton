# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Finite-EA DER strains and analytic first derivatives in a committed frame chart.

The native-precision residual is exact. The FP32 tangent is the full
strain-Jacobian Gauss-Newton matrix, including all position/theta blocks.
No finite differences, numerical differentiation or theta inertia are used.
"""

from typing import Any

import numpy as np
import warp as wp

from .material_inputs import _cable_geometry

wp.set_module_options({"enable_backward": False})


@wp.func
def _rotate(angle: Any, increment: Any):
    """Return the cosine and sine of ``angle + increment``."""
    c, s = wp.cos(angle), wp.sin(angle)
    dc, ds = wp.cos(increment), wp.sin(increment)
    return c * dc - s * ds, s * dc + c * ds


@wp.func
def _abs_cross(a: Any, b: Any):
    """Return the componentwise sum of absolute products forming ``a x b``."""
    return type(a)(
        wp.abs(a[1] * b[2]) + wp.abs(a[2] * b[1]),
        wp.abs(a[2] * b[0]) + wp.abs(a[0] * b[2]),
        wp.abs(a[0] * b[1]) + wp.abs(a[1] * b[0]),
    )


@wp.func
def _abs_dot(a: Any, b: Any):
    return wp.abs(a[0] * b[0]) + wp.abs(a[1] * b[1]) + wp.abs(a[2] * b[2])


@wp.func
def _transport(a: Any, b: Any, d: Any):
    v = wp.cross(a, b)
    first = wp.cross(v, d)
    return d + first + wp.cross(v, first) / (type(a[0])(1.0) + wp.dot(a, b))


@wp.func
def _transport_differential(a: Any, b: Any, d: Any, da: Any, db: Any, dd: Any):
    v = wp.cross(a, b)
    dv = wp.cross(da, b) + wp.cross(a, db)
    denominator = type(a[0])(1.0) + wp.dot(a, b)
    derivative = wp.dot(da, b) + wp.dot(a, db)
    first = wp.cross(v, d)
    dfirst = wp.cross(dv, d) + wp.cross(v, dd)
    return (
        dd
        + dfirst
        + (wp.cross(dv, first) + wp.cross(v, dfirst)) / denominator
        - wp.cross(v, first) * derivative / (denominator * denominator)
    )


@wp.kernel
def _difference(x: wp.array[Any], anchor: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = x[i] - anchor[i]


@wp.kernel
def _edges(
    step: wp.array[Any],
    anchor_x: wp.array[Any],
    anchor_d: wp.array[Any],
    lengths0: wp.array[Any],
    ea: Any,
    tangent: wp.array[Any],
    director: wp.array[Any],
    lengths: wp.array[Any],
    energy: wp.array[Any],
    residual: wp.array2d[Any],
    matrix: wp.array3d[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    anchor_edge = anchor_x[i + 1] - anchor_x[i]
    change = step[i + 1] - step[i]
    edge = anchor_edge + change
    length = wp.length(edge)
    if not wp.isfinite(length) or length <= type(ea)(1e-8) * lengths0[i]:
        wp.atomic_max(flags, 0, 1)
        return
    t = edge / length
    old_t = wp.normalize(anchor_edge)
    if type(ea)(1.0) + wp.dot(old_t, t) <= type(ea)(1e-10):
        wp.atomic_max(flags, 0, 1)
        return
    tangent[i] = t
    director[i] = _transport(old_t, t, anchor_d[i])
    lengths[i] = length
    stiffness = ea / lengths0[i]
    # |e| - L0 = (|e|^2 - L0^2) / (|e| + L0) with |e|^2 formed from the
    # step-start edge, fixed within the step, plus the size of the change.
    squares = (wp.dot(anchor_edge, anchor_edge) - lengths0[i] * lengths0[i]) + (
        type(ea)(2.0) * wp.dot(anchor_edge, change) + wp.dot(change, change)
    )
    extension = squares / (length + lengths0[i])
    energy[i] = type(ea)(0.5) * stiffness * extension * extension
    for a in range(6):
        gradient = t[a % 3]
        if a < 3:
            gradient = -gradient
        residual[i, a] = stiffness * extension * gradient
        for b in range(6):
            other = t[b % 3]
            if b < 3:
                other = -other
            matrix[i, a, b] = stiffness * gradient * other


@wp.kernel
def _bend_twist(
    anchor_theta: wp.array[Any],
    theta_step: wp.array[Any],
    anchor_x: wp.array[Any],
    anchor_d: wp.array[Any],
    anchor_phi: wp.array[Any],
    tangent: wp.array[Any],
    director: wp.array[Any],
    lengths: wp.array[Any],
    dual: wp.array[Any],
    rest_strain: wp.array[Any],
    bending: wp.array2d[Any],
    gj: Any,
    phi_out: wp.array[Any],
    energy: wp.array[Any],
    jacobian: wp.array3d[Any],
    residual: wp.array2d[Any],
    rounding: wp.array2d[Any],
    matrix: wp.array3d[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    if flags[0] != 0:
        return
    t0, t1 = tangent[i], tangent[i + 1]
    d0, d1 = director[i], director[i + 1]
    denominator = type(gj)(1.0) + wp.dot(t0, t1)
    if denominator <= type(gj)(1e-10):
        wp.atomic_max(flags, 0, 1)
        return
    kb = type(gj)(2.0) * wp.cross(t0, t1) / denominator
    carried = _transport(t0, t1, d0)
    cosine = wp.dot(carried, d1)
    sine = wp.dot(t1, wp.cross(carried, d1))
    phi = wp.atan2(sine, cosine)
    tau = type(gj)(6.283185307179586)
    phi += tau * wp.round((anchor_phi[i] - phi) / tau)
    if not wp.isfinite(phi) or wp.abs(phi - anchor_phi[i]) >= type(gj)(1.5707963267948966):
        wp.atomic_max(flags, 0, 1)
        return
    phi_out[i] = phi
    # Angle addition keeps the step-start angle's rounding a fixed bias of the
    # step; the increment enters exactly however large the accumulated angle.
    c0, s0 = _rotate(anchor_theta[i], theta_step[i])
    c1, s1 = _rotate(anchor_theta[i + 1], theta_step[i + 1])
    perpendicular0, perpendicular1 = wp.cross(t0, d0), wp.cross(t1, d1)
    m10, m11 = c0 * d0 + s0 * perpendicular0, c1 * d1 + s1 * perpendicular1
    m20, m21 = -s0 * d0 + c0 * perpendicular0, -s1 * d1 + c1 * perpendicular1
    strain = (
        type(t0)(
            type(gj)(0.5) * wp.dot(kb, m20 + m21),
            -type(gj)(0.5) * wp.dot(kb, m10 + m11),
            (anchor_theta[i + 1] - anchor_theta[i]) + (theta_step[i + 1] - theta_step[i]) + phi,
        )
        - rest_strain[i]
    )
    stress = (
        type(t0)(
            bending[0, 0] * strain[0] + bending[0, 1] * strain[1],
            bending[1, 0] * strain[0] + bending[1, 1] * strain[1],
            gj * strain[2],
        )
        / dual[i]
    )
    energy[i] = type(t0)(
        type(gj)(0.5) * (strain[0] * stress[0] + strain[1] * stress[1]),
        type(gj)(0.5) * strain[2] * stress[2],
        type(gj)(0.0),
    )
    # Near rest each strain is a small sum of larger operands: the products forming
    # kb and its projections on the material frames, and the angle differences plus
    # the frame angle, whose atan2 rounds with the products forming its sine. The
    # rounding follows those operands, so bound the stress from them. Componentwise
    # products keep the bound at zero where a straight axis-aligned rod is exact.
    curvature = type(gj)(2.0) * _abs_cross(t0, t1) / denominator
    absolute0, absolute1 = wp.abs(m10) + wp.abs(m11), wp.abs(m20) + wp.abs(m21)
    sine_operands = _abs_dot(t1, _abs_cross(carried, d1))
    operands = type(t0)(
        type(gj)(0.5) * _abs_dot(curvature, absolute1) + wp.abs(rest_strain[i][0]),
        type(gj)(0.5) * _abs_dot(curvature, absolute0) + wp.abs(rest_strain[i][1]),
        wp.abs(anchor_theta[i + 1] - anchor_theta[i])
        + wp.abs(theta_step[i + 1] - theta_step[i])
        + wp.abs(phi)
        + sine_operands
        + wp.abs(rest_strain[i][2]),
    )
    stress_bound = (
        type(t0)(
            wp.abs(bending[0, 0]) * operands[0] + wp.abs(bending[0, 1]) * operands[1],
            wp.abs(bending[1, 0]) * operands[0] + wp.abs(bending[1, 1]) * operands[1],
            wp.abs(gj) * operands[2],
        )
        / dual[i]
    )
    old_t0 = wp.normalize(anchor_x[i + 1] - anchor_x[i])
    old_t1 = wp.normalize(anchor_x[i + 2] - anchor_x[i + 1])
    zero = type(t0)(type(gj)(0.0))
    for a in range(11):
        de0 = type(t0)(type(gj)(0.0))
        de1 = type(t0)(type(gj)(0.0))
        dtheta0, dtheta1 = type(gj)(0.0), type(gj)(0.0)
        if a < 9:
            axis = a % 3
            if a < 3:
                de0[axis] = type(gj)(-1.0)
            elif a < 6:
                de0[axis] = type(gj)(1.0)
                de1[axis] = type(gj)(-1.0)
            else:
                de1[axis] = type(gj)(1.0)
        elif a == 9:
            dtheta0 = type(gj)(1.0)
        else:
            dtheta1 = type(gj)(1.0)
        dt0 = (de0 - wp.dot(t0, de0) * t0) / lengths[i]
        dt1 = (de1 - wp.dot(t1, de1) * t1) / lengths[i + 1]
        dd0 = _transport_differential(old_t0, t0, anchor_d[i], zero, dt0, zero)
        dd1 = _transport_differential(old_t1, t1, anchor_d[i + 1], zero, dt1, zero)
        dp0, dp1 = wp.cross(dt0, d0) + wp.cross(t0, dd0), wp.cross(dt1, d1) + wp.cross(t1, dd1)
        dm10 = c0 * dd0 + s0 * dp0 + dtheta0 * m20
        dm11 = c1 * dd1 + s1 * dp1 + dtheta1 * m21
        dm20 = -s0 * dd0 + c0 * dp0 - dtheta0 * m10
        dm21 = -s1 * dd1 + c1 * dp1 - dtheta1 * m11
        ddenominator = wp.dot(dt0, t1) + wp.dot(t0, dt1)
        dkb = (type(gj)(2.0) * (wp.cross(dt0, t1) + wp.cross(t0, dt1)) - kb * ddenominator) / denominator
        dcarried = _transport_differential(t0, t1, d0, dt0, dt1, dd0)
        dcosine = wp.dot(dcarried, d1) + wp.dot(carried, dd1)
        dsine = wp.dot(dt1, wp.cross(carried, d1)) + wp.dot(t1, wp.cross(dcarried, d1) + wp.cross(carried, dd1))
        dphi = (cosine * dsine - sine * dcosine) / (cosine * cosine + sine * sine)
        ds = type(t0)(
            type(gj)(0.5) * (wp.dot(dkb, m20 + m21) + wp.dot(kb, dm20 + dm21)),
            -type(gj)(0.5) * (wp.dot(dkb, m10 + m11) + wp.dot(kb, dm10 + dm11)),
            dtheta1 - dtheta0 + dphi,
        )
        residual[i, a] = wp.dot(ds, stress)
        rounding[i, a] = wp.abs(residual[i, a]) + (
            wp.abs(ds[0]) * stress_bound[0] + wp.abs(ds[1]) * stress_bound[1] + wp.abs(ds[2]) * stress_bound[2]
        )
        for row in range(3):
            jacobian[i, row, a] = ds[row]
    for a in range(11):
        for b in range(11):
            matrix[i, a, b] = (
                jacobian[i, 0, a] * (bending[0, 0] * jacobian[i, 0, b] + bending[0, 1] * jacobian[i, 1, b])
                + jacobian[i, 1, a] * (bending[1, 0] * jacobian[i, 0, b] + bending[1, 1] * jacobian[i, 1, b])
                + gj * jacobian[i, 2, a] * jacobian[i, 2, b]
            ) / dual[i]


class CableObjective:
    """Own one cable's reference strains and fixed-size device element buffers."""

    def __init__(self, positions, theta, directors, stiffness, *, dtype, device):
        self.dtype, self.device = dtype, wp.get_device(device)
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        lengths, tangents, kb, phi = _cable_geometry(np.asarray(positions), np.asarray(directors))
        theta = np.asarray(theta)
        perpendicular = np.cross(tangents, directors)
        m1 = np.cos(theta[:, None]) * directors + np.sin(theta[:, None]) * perpendicular
        m2 = -np.sin(theta[:, None]) * directors + np.cos(theta[:, None]) * perpendicular
        strain = np.column_stack(
            (
                np.sum(kb * (m2[:-1] + m2[1:]), axis=1) / 2,
                -np.sum(kb * (m1[:-1] + m1[1:]), axis=1) / 2,
                np.diff(theta) + phi,
            )
        )
        self.ea, self.gj = dtype(stiffness[0]), dtype(stiffness[2])
        self.lengths0 = wp.array(lengths, dtype=dtype, device=device)
        self.dual = wp.array((lengths[:-1] + lengths[1:]) / 2, dtype=dtype, device=device)
        self.rest_strain = wp.array(strain, dtype=self.vector, device=device)
        self.bending = wp.array(stiffness[1], dtype=dtype, device=device)
        ne, ni = len(lengths), len(lengths) - 1
        self.lengths = wp.zeros(ne, dtype=dtype, device=device)
        self.tangent = wp.zeros(ne, dtype=self.vector, device=device)
        self.director = wp.zeros_like(self.tangent)
        self.phi = wp.zeros(ni, dtype=dtype, device=device)
        self.stretch_energy = wp.zeros(ne, dtype=dtype, device=device)
        self.bend_twist_energy = wp.zeros(ni, dtype=self.vector, device=device)
        self.stretch_r = wp.zeros((ne, 6), dtype=dtype, device=device)
        self.stretch_b = wp.zeros((ne, 6, 6), dtype=dtype, device=device)
        self.jacobian = wp.zeros((ni, 3, 11), dtype=dtype, device=device)
        self.bend_twist_r = wp.zeros((ni, 11), dtype=dtype, device=device)
        # Per-entry operand magnitude whose eps multiple bounds the residual rounding.
        self.bend_twist_rounding = wp.zeros((ni, 11), dtype=dtype, device=device)
        self.bend_twist_b = wp.zeros((ni, 11, 11), dtype=dtype, device=device)
        self.flags = wp.zeros(1, dtype=int, device=device)
        self._step = self._theta_step = None

    def evaluate(self, positions, theta, anchor_positions, anchor_directors, anchor_twist):
        """Overwrite element terms and candidate frames, leaving all anchors unchanged."""
        if self._step is None or self._step.size != positions.size:
            self._step = wp.zeros_like(positions)
            self._theta_step = wp.zeros_like(theta)
        wp.launch(_difference, positions.size, [positions, anchor_positions, self._step], device=self.device)
        self._theta_step.zero_()
        self.evaluate_step(self._step, theta, self._theta_step, anchor_positions, anchor_directors, anchor_twist)

    def evaluate_step(self, step, anchor_theta, theta_step, anchor_positions, anchor_directors, anchor_twist):
        """Evaluate the configuration ``anchor + step`` from its exact increments.

        Edges and twist are formed as anchor differences plus step differences,
        so a small step is not added to large absolute coordinates or angles.
        """
        self.flags.zero_()
        wp.launch(
            _edges,
            self.lengths.size,
            [
                step,
                anchor_positions,
                anchor_directors,
                self.lengths0,
                self.ea,
                self.tangent,
                self.director,
                self.lengths,
                self.stretch_energy,
                self.stretch_r,
                self.stretch_b,
                self.flags,
            ],
            device=self.device,
        )
        wp.launch(
            _bend_twist,
            self.phi.size,
            [
                anchor_theta,
                theta_step,
                anchor_positions,
                anchor_directors,
                anchor_twist,
                self.tangent,
                self.director,
                self.lengths,
                self.dual,
                self.rest_strain,
                self.bending,
                self.gj,
                self.phi,
                self.bend_twist_energy,
                self.jacobian,
                self.bend_twist_r,
                self.bend_twist_rounding,
                self.bend_twist_b,
                self.flags,
            ],
            device=self.device,
        )

    def evaluate_host(self, positions, theta, frame, *, step=None, theta_step=None):
        """Return dense diagnostic E/r/B for component validation, never for stepping.

        With ``step`` and ``theta_step``, ``positions`` and ``theta`` are the
        step-start values and the configuration is formed from the increments.
        """

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=self.device)

        anchors = (
            array(frame.positions, self.vector),
            array(frame.directors, self.vector),
            array(frame.reference_twist, self.dtype),
        )
        if step is None:
            self.evaluate(array(positions, self.vector), array(theta, self.dtype), *anchors)
        else:
            if not np.array_equal(np.asarray(positions), np.asarray(frame.positions)):
                raise ValueError("A cable step starts from the committed frame positions")
            self.evaluate_step(
                array(step, self.vector), array(theta, self.dtype), array(theta_step, self.dtype), *anchors
            )
        if self.flags.numpy()[0]:
            raise ValueError("Invalid cable candidate geometry or frame chart")
        count = len(positions) * 3 + len(theta)
        residual, matrix = np.zeros(count), np.zeros((count, count))
        for i, (r, b) in enumerate(zip(self.stretch_r.numpy(), self.stretch_b.numpy(), strict=True)):
            rows = np.arange(3 * i, 3 * i + 6)
            residual[rows] += r
            matrix[np.ix_(rows, rows)] += b
        for i, (r, b) in enumerate(zip(self.bend_twist_r.numpy(), self.bend_twist_b.numpy(), strict=True)):
            rows = np.r_[np.arange(3 * i, 3 * i + 9), 3 * len(positions) + np.arange(i, i + 2)]
            residual[rows] += r
            matrix[np.ix_(rows, rows)] += b
        parts = self.bend_twist_energy.numpy().sum(axis=0)
        energy = np.array([self.stretch_energy.numpy().sum(), parts[0], parts[1]])
        if not np.isfinite(energy).all() or not np.isfinite(residual).all() or not np.isfinite(matrix).all():
            raise ValueError("Nonfinite cable energy or derivatives")
        return energy, residual, matrix
