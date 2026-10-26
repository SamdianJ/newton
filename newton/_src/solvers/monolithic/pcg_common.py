# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared layout groups, residual acceptance and gap policies."""

from typing import Any

import numpy as np
import warp as wp

LINEAR_ROUNDING_MULTIPLIER = 4.0
"""Multiplier ``c`` of the attainable linear residual ``c eps |A| |y|`` in the solver precision.

A direction ``y`` stored in a precision with unit ``eps`` leaves a residual of
up to about ``eps |A| |y|`` even when it is the exact solution; finite-precision
PCG reaches about that floor. The value matches the nonlinear rounding
multiplier and covers the measured FP32 floors of ill-conditioned tet systems.
"""


LINEAR_ROUNDING_CAP = 1e-2
"""Largest residual, relative to a block's right-hand side, that the rounding floor may accept.

Inexact Newton needs a direction whose residual is a fixed fraction of the
right-hand side. A nearly singular block can return a huge ``y`` whose
``|A| |y|`` rounding would excuse any residual; such a block still fails.
"""

_ROUNDING_CAP = wp.constant(LINEAR_ROUNDING_CAP)


def linear_rounding(dtype):
    """Return ``c eps`` of the attainable linear residual for a native dtype."""
    native = np.float64 if dtype in (wp.float64, np.float64) else np.float32
    return LINEAR_ROUNDING_MULTIPLIER * float(np.finfo(native).eps)


def group_data(layout):
    """Build row membership and compatibility views from the active layout."""
    groups = layout.groups
    membership = np.zeros(layout.scalar_dof_count, dtype=np.int32)
    aliases = [0, -1, -1]
    for index, (kind, start, count) in enumerate(groups[1:], 1):
        membership[start : start + count] = index
        if kind == "articulation":
            aliases[1] = index
        elif kind == "tet":
            aliases[2] = index
    return membership, np.asarray(aliases, dtype=np.int32)


def group_policy(layout, config, *, require_limits=True):
    """Preserve existing q/x floors and caps while ordering by layout groups."""
    names = {"global": 0, "articulation": 1, "tet": 2}
    floors = (config.residual_floor_global, config.residual_floor_q, config.residual_floor_x)
    limits = getattr(config, "residual_norm_limits", None) or (float("inf"),) * 3
    keys = [names.get(kind, 2 if kind.endswith(("_t", "_x")) else 1) for kind, _, _ in layout.groups]
    complete_limits = getattr(config, "group_norm_limits", None)
    complete_floors = getattr(config, "group_residual_floors", None)
    if layout.material_ranges and (complete_floors is None or (require_limits and complete_limits is None)):
        raise ValueError("Material xyz/theta groups require explicit per-group residual floors and limits")
    if complete_limits is not None and len(complete_limits) != len(keys):
        raise ValueError("Residual limits must match all layout groups")
    if complete_floors is not None and len(complete_floors) != len(keys):
        raise ValueError("Residual floors must match all layout groups")
    return (
        list(complete_floors) if complete_floors is not None else [floors[key] for key in keys],
        list(complete_limits) if complete_limits is not None else [limits[key] for key in keys],
    )


@wp.func
def residual_converged(
    norms: wp.array[Any],
    rhs_offset: int,
    floors: wp.array[float],
    rtol: float,
    atol: float,
    limits: wp.array[float],
    rounding_offset: int,
    rounding: float,
):
    accepted = bool(True)
    for k in range(floors.shape[0]):
        target = wp.max(
            wp.float64(rtol) * wp.max(wp.float64(norms[rhs_offset + k]), wp.float64(floors[k])), wp.float64(atol)
        )
        target = wp.min(target, wp.float64(limits[k]))
        # No stored solution resolves its residual below the rounding of |A| |y|,
        # which may relax the target to at most a fixed fraction of the rhs.
        floor = wp.min(
            wp.float64(rounding) * wp.float64(norms[rounding_offset + k]),
            wp.float64(_ROUNDING_CAP) * wp.float64(norms[rhs_offset + k]),
        )
        target = wp.max(target, floor)
        if not wp.isfinite(norms[k]) or wp.float64(norms[k]) > target:
            accepted = False
    return accepted


@wp.func
def block_drift(gap: Any, true_norm: Any, floor: float):
    return wp.float64(gap) / wp.max(wp.float64(true_norm), wp.float64(floor))


@wp.func
def max_block_drift(norms: wp.array[Any], gap_offset: int, floors: wp.array[float]):
    drift = wp.float64(0.0)
    for k in range(floors.shape[0]):
        if not wp.isfinite(norms[gap_offset + k]) or not wp.isfinite(norms[k]):
            return wp.float64(wp.nan)
        drift = wp.max(drift, block_drift(norms[gap_offset + k], norms[k], floors[k]))
    return drift
