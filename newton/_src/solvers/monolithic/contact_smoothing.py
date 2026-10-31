# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smooth BOX field and step-start orientation coefficients."""

from typing import Any

import warp as wp


@wp.func
def box_field(scale: Any, point: Any, width: Any):
    """Six-plane LSE field, its gradient and Hessian (not a Euclidean SDF)."""
    largest = point[0] - scale[0]
    for j in range(6):
        axis = j / 2
        sign = type(width)(2 * (j % 2) - 1)
        largest = wp.max(largest, sign * point[axis] - scale[axis])
    total = type(width)(0.0)
    gradient = type(point)(0.0)
    positive, negative = type(point)(0.0), type(point)(0.0)
    for j in range(6):
        axis = j / 2
        sign = type(width)(2 * (j % 2) - 1)
        value = wp.exp((sign * point[axis] - scale[axis] - largest) / width)
        total += value
        gradient[axis] += sign * value
        if j % 2 == 0:
            negative[axis] = value
        else:
            positive[axis] = value
    gradient /= total
    positive /= total
    negative /= total
    probabilities = positive + negative
    hessian = -wp.outer(gradient, gradient)
    for axis in range(3):
        # Covariance diagonal without q_i - g_i^2 cancellation on a face.
        hessian[axis, axis] = (
            probabilities[axis] * (probabilities[(axis + 1) % 3] + probabilities[(axis + 2) % 3])
            + type(width)(4.0) * positive[axis] * negative[axis]
        )
    hessian /= width
    return largest + width * wp.log(total), gradient, hessian


@wp.func
def orientation_weight(alignment: Any, width: Any):
    """Return w and w' without endpoint cancellation."""
    w, dw = type(width)(1.0), type(width)(0.0)
    if alignment >= width:
        w = type(width)(0.0)
    elif alignment > type(width)(0.0):
        t = alignment / width
        z = type(width)(1.0) - t
        polynomial = type(width)(1.0) + type(width)(3.0) * t + type(width)(6.0) * t * t
        w = z * z * z * polynomial
        dw = -type(width)(30.0) * t * t * z * z / width
    return w, dw


@wp.func
def penetration(gap: Any, smoothing: Any):
    u = -gap
    p, dp, ddp = type(gap)(0.0), type(gap)(0.0), type(gap)(0.0)
    if u > type(gap)(0.0) and u >= smoothing:
        p, dp = u, type(gap)(1.0)
    elif smoothing > type(gap)(0.0) and u > -smoothing:
        t = (u + smoothing) / (type(gap)(2.0) * smoothing)
        p = smoothing * t * t * t * (type(gap)(2.0) - t)
        dp = t * t * (type(gap)(3.0) - type(gap)(2.0) * t)
        ddp = type(gap)(3.0) * t * (type(gap)(1.0) - t) / smoothing
    return p, dp, ddp
