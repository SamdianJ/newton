# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared layout groups, residual acceptance and gap policies."""

import numpy as np
import warp as wp


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


def group_policy(layout, config):
    """Preserve existing q/x floors and caps while ordering by layout groups."""
    names = {"global": 0, "articulation": 1, "tet": 2}
    floors = (config.residual_floor_global, config.residual_floor_q, config.residual_floor_x)
    limits = getattr(config, "residual_norm_limits", None) or (float("inf"),) * 3
    keys = [names.get(kind, 2 if kind.endswith(("_t", "_x")) else 1) for kind, _, _ in layout.groups]
    complete_limits = getattr(config, "group_norm_limits", None)
    complete_floors = getattr(config, "group_residual_floors", None)
    if layout.material_ranges and (complete_floors is None or complete_limits is None):
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
    norms: wp.array[float], rhs_offset: int, floors: wp.array[float], rtol: float, atol: float, limits: wp.array[float]
):
    accepted = bool(True)
    for k in range(floors.shape[0]):
        target = wp.max(wp.float64(rtol) * wp.float64(wp.max(norms[rhs_offset + k], floors[k])), wp.float64(atol))
        target = wp.min(target, wp.float64(limits[k]))
        if not wp.isfinite(norms[k]) or wp.float64(norms[k]) > target:
            accepted = False
    return accepted


@wp.func
def block_drift(gap: float, true_norm: float, floor: float):
    return wp.float64(gap) / wp.float64(wp.max(true_norm, floor))


@wp.func
def max_block_drift(norms: wp.array[float], gap_offset: int, floors: wp.array[float]):
    drift = wp.float64(0.0)
    for k in range(floors.shape[0]):
        if not wp.isfinite(norms[gap_offset + k]) or not wp.isfinite(norms[k]):
            return wp.float64(wp.nan)
        drift = wp.max(drift, block_drift(norms[gap_offset + k], norms[k], floors[k]))
    return drift


@wp.kernel
def decide_restart(
    gap: wp.array[float],
    true_norm: wp.array[float],
    floors: wp.array[float],
    threshold: wp.float64,
    error: wp.array[int],
    result: wp.array[int],
):
    drift = wp.float64(0.0)
    for k in range(floors.shape[0]):
        if not wp.isfinite(gap[k]) or not wp.isfinite(true_norm[k]):
            error[0] = 16
        drift = wp.max(drift, block_drift(gap[k], true_norm[k], floors[k]))
    if error[0] != 0 or not wp.isfinite(drift):
        result[0] = -1
    else:
        result[0] = int(drift >= threshold)
