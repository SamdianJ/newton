# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure threading, draw, contact geometry and free-end slip on published hitch states."""

import numpy as np

from .cable_contact_criteria import segment_distance
from .cable_hitch_fixture import CORE_RADIUS, DT, END, HALF_LENGTH, RADIUS


def core_clearance(q):
    """Return the minimum signed distance from the cable centerline to the capped cylinder."""
    p = q[:-1]
    direction = np.diff(q, axis=0)
    if np.max(np.abs(q[:, 2])) <= HALF_LENGTH:
        # Inside the axial slab, this radial distance also lower-bounds the signed distance.
        xy = direction[:, :2]
        s = np.clip(-np.sum(p[:, :2] * xy, axis=1) / np.maximum(np.sum(xy * xy, axis=1), 1e-30), 0, 1)
        return np.linalg.norm(p[:, :2] + s[:, None] * xy, axis=1).min() - CORE_RADIUS

    def signed_distance(s):
        point = p + s[:, None] * direction
        delta = np.column_stack((np.linalg.norm(point[:, :2], axis=1) - CORE_RADIUS, np.abs(point[:, 2]) - HALF_LENGTH))
        return np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)

    lo, hi = np.zeros(len(p)), np.ones(len(p))
    for _ in range(42):
        a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
        left = signed_distance(a) < signed_distance(b)
        lo, hi = np.where(left, lo, a), np.where(left, b, hi)
    return signed_distance((lo + hi) * 0.5).min()


def clearances(positions):
    first, second = np.triu_indices(positions.shape[1] - 1, 4)
    rope_min = core_min = crossing_bound = np.inf
    previous = previous_distance = None
    for q in positions:
        distance, _, _ = segment_distance(q[first], q[first + 1], q[second], q[second + 1])
        rope_min = min(rope_min, distance.min())
        core_min = min(core_min, core_clearance(q))
        if previous is not None:
            motion = np.linalg.norm(q - previous, axis=1)
            edge_motion = np.maximum(motion[:-1], motion[1:])
            bound = 0.5 * (distance + previous_distance - edge_motion[first] - edge_motion[second])
            crossing_bound = min(crossing_bound, bound.min())
        previous, previous_distance = q, distance
    return 1000 * np.array([max(0, 2 * RADIUS - rope_min), max(0, RADIUS - core_min), crossing_bound])


def evaluate_hitch(fixture, t, q, grip, load, contact_y):
    """Evaluate both friction settings using the same geometric and 12 mN retention gates."""
    q = q.astype(np.float64, copy=False)
    if abs(t[0]) > 1e-9 or abs(t[-1] - END) > 1e-9 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > DT + 1e-9:
        raise ValueError("The entire 0-5 s trajectory must consist of published steps of at most 3 ms")
    passages = fixture.turn_passages(t, q, RADIUS, CORE_RADIUS)
    rope_mm, core_mm, crossing_mm = clearances(q)

    def nearest(stamp):
        return np.abs(t - stamp).argmin()

    draw = (t >= 1.26 - 1e-9) & (t <= 1.46 + 1e-9)
    ends = np.linalg.norm(q[:, 0] - q[:, -1], axis=1)
    gain_mm = 1000 * (ends[nearest(1.46)] - ends[nearest(1.26)])
    outward = q[:, [0, -1]] - q[:, [1, -2]]
    outward /= np.linalg.norm(outward, axis=2, keepdims=True)
    tension = np.sum(outward * grip[:, [0, -1]], axis=2)
    tension_mN = 1000 * tension[draw].min(axis=1).max()
    released = t >= 1.78 - 1e-9
    loaded = t >= 1.96 - 1e-9
    tail = t >= END - 0.5 - 1e-9
    fixture_force = np.max(np.linalg.norm(grip[released], axis=2))
    angles = np.unwrap(np.arctan2(q[:, :, 1], q[:, :, 0]), axis=1)
    turns = (angles[released, -1] - angles[released, 0]) / (2 * np.pi)
    change = np.max(np.abs(turns - turns[0]))
    excursion_mm = 1000 * np.ptp(q[tail, 0, 1])
    reaction_mN = -1000 * np.average(contact_y[tail][1:], weights=np.diff(t[tail]))
    return {
        "passages": [row["turn_from_start_of_wrap"] for row in passages],
        "rope_overlap_mm": float(rope_mm),
        "core_overlap_mm": float(core_mm),
        "centerline_bound_mm": float(crossing_mm),
        "end_separation_gain_mm": float(gain_mm),
        "simultaneous_end_tension_mN": float(tension_mN),
        "max_released_fixture_force_N": float(fixture_force),
        "full_load_12mN": bool(np.allclose(load[loaded], [0, 0.012, 0], rtol=0, atol=1e-8)),
        "full_load_duration_s": float(t[-1] - t[loaded][0]),
        "min_released_turns": float(np.min(np.abs(turns))),
        "max_wrap_change_turns": float(change),
        "end_turns": float(abs(turns[-1])),
        "end_excursion_last_half_s_mm": float(excursion_mm),
        "opposing_reaction_last_half_s_mN": float(reaction_mN),
    }
