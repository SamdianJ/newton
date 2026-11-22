# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native COM-and-quaternion pose increments shared by FREE rigid actors and floating roots.

A pose is advanced from its step start by an exact COM increment and a left rotation increment
q_step with R = R(q_step) R_n; prediction, line-search trials, derived increments, rate recovery,
increment differences and coordinate magnitudes all act on that pair.
"""

from typing import Any

import warp as wp

from .articulation import _rotation_step
from .articulation_state import _store_rates
from .numeric_parameters import scalar_value


@wp.kernel
def _predict_pose(
    predicted_com: wp.array[Any],
    predicted_step: wp.array[Any],
    quat_n: wp.array[Any],
    omega: wp.array[Any],
    h_arg: Any,
    com: wp.array[Any],
    rotation: wp.array[Any],
    step: wp.array[Any],
    rotation_step: wp.array[Any],
):
    h = scalar_value(h_arg)
    i = wp.tid()
    dr = h * omega[i]
    angle = wp.length(dr)
    increment = type(quat_n[0])(type(h)(0.0), type(h)(0.0), type(h)(0.0), type(h)(1.0))
    # Keep the previous initial guess if extrapolation leaves the rigid domain.
    if angle > type(h)(0.0) and angle < type(h)(1.5707963267948966):
        increment = wp.quat_from_axis_angle(dr / angle, angle)
    com[i] = predicted_com[i]
    step[i] = predicted_step[i]
    rotation_step[i] = increment
    rotation[i] = wp.normalize(increment * quat_n[i])


@wp.kernel
def _trial(
    step: wp.array[Any],
    rotation_step: wp.array[Any],
    rows: wp.array2d[int],
    direction: wp.array[Any],
    alpha_arg: Any,
    tn: wp.array[Any],
    quat_n: wp.array[Any],
    out_step: wp.array[Any],
    out_rotation_step: wp.array[Any],
    out_com: wp.array[Any],
    out_rotation: wp.array[Any],
):
    alpha = scalar_value(alpha_arg)
    i = wp.tid()
    dc = type(tn[0])(
        type(alpha)(direction[rows[i, 0]]), type(alpha)(direction[rows[i, 1]]), type(alpha)(direction[rows[i, 2]])
    )
    dr = alpha * type(tn[0])(
        type(alpha)(direction[rows[i, 3]]), type(alpha)(direction[rows[i, 4]]), type(alpha)(direction[rows[i, 5]])
    )
    angle = wp.length(dr)
    increment = type(quat_n[0])(type(alpha)(0.0), type(alpha)(0.0), type(alpha)(0.0), type(alpha)(1.0))
    if angle != type(alpha)(0.0):
        increment = wp.quat_from_axis_angle(dr / angle, angle)
    # Advance the exact step and the rotation from the step start, then form
    # the pose from them once.
    out_step[i] = step[i] + alpha * dc
    out_rotation_step[i] = wp.normalize(increment * rotation_step[i])
    out_com[i] = tn[i] + out_step[i]
    out_rotation[i] = wp.normalize(out_rotation_step[i] * quat_n[i])


@wp.kernel
def _derive_steps(
    com: wp.array[Any],
    rotation: wp.array[Any],
    tn: wp.array[Any],
    quat_n: wp.array[Any],
    step: wp.array[Any],
    rotation_step: wp.array[Any],
):
    i = wp.tid()
    step[i] = com[i] - tn[i]
    q = rotation[i] * wp.quat_inverse(quat_n[i])
    if q[3] < type(q[3])(0.0):
        q = -q
    rotation_step[i] = q


@wp.kernel
def _recover(
    step: wp.array[Any],
    rotation_step: wp.array[Any],
    h_arg: Any,
    v: wp.array[Any],
    omega: wp.array[Any],
    sym: wp.array2d[Any],
):
    h = scalar_value(h_arg)
    i = wp.tid()
    _store_rates(step[i], _rotation_step(rotation_step[i]), h, i, v, omega, sym)


@wp.kernel
def _pose_difference(
    ta: wp.array[Any],
    qa: wp.array[Any],
    tb: wp.array[Any],
    qb: wp.array[Any],
    rows: wp.array2d[int],
    out: wp.array[Any],
):
    i = wp.tid()
    # Left relative quaternion logarithm, using the short representative.
    relative = qb[i] * wp.quat_inverse(qa[i])
    vector = wp.vector(relative[0], relative[1], relative[2])
    scalar = relative[3]
    if scalar < out.dtype(0):
        vector = -vector
    length = wp.length(vector)
    factor = out.dtype(2)
    if length > out.dtype(0):
        factor = out.dtype(2) * wp.atan2(length, wp.abs(scalar)) / length
    for axis in range(3):
        out[rows[i, axis]] = tb[i][axis] - ta[i][axis]
        out[rows[i, axis + 3]] = factor * vector[axis]


@wp.kernel
def _pose_coordinates(com: wp.array[Any], rotation: wp.array[Any], rows: wp.array2d[int], output: wp.array[Any]):
    i, j = wp.tid()
    # A rotation step q q_n^-1 near identity turns by about twice its vector part.
    value = type(output[0])(2.0) * type(output[0])(wp.abs(rotation[i][j % 3]))
    if j < 3:
        value = type(output[0])(wp.abs(com[i][j]))
    output[rows[i, j]] = value
