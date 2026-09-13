# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared block residual acceptance and gap policies."""

import warp as wp


@wp.func
def residual_converged(norms: wp.vec3, rhs_norms: wp.vec3, floors: wp.vec3, rtol: float, atol: float):
    accepted = bool(True)
    for k in range(3):
        target = wp.max(wp.float64(rtol) * wp.float64(wp.max(rhs_norms[k], floors[k])), wp.float64(atol))
        if not wp.isfinite(norms[k]) or wp.float64(norms[k]) > target:
            accepted = False
    return accepted


@wp.func
def block_drift(gap: float, true_norm: float, floor: float):
    return wp.float64(gap) / wp.float64(wp.max(true_norm, floor))


@wp.func
def max_block_drift(gap: wp.vec3, true_norm: wp.vec3, floors: wp.vec3):
    drift = wp.float64(0.0)
    for k in range(3):
        if not wp.isfinite(gap[k]) or not wp.isfinite(true_norm[k]):
            return wp.float64(wp.nan)
        drift = wp.max(drift, block_drift(gap[k], true_norm[k], floors[k]))
    return drift


@wp.kernel
def decide_restart(
    gap: wp.array[float],
    true_norm: wp.array[float],
    floors: wp.vec3,
    threshold: wp.float64,
    error: wp.array[int],
    result: wp.array[int],
):
    drift = max_block_drift(wp.vec3(gap[0], gap[1], gap[2]), wp.vec3(true_norm[0], true_norm[1], true_norm[2]), floors)
    if error[0] != 0 or not wp.isfinite(drift):
        result[0] = -1
    else:
        result[0] = int(drift >= threshold)
