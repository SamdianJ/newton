# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure clove and square knot formation, non-crossing and passive load retention."""

import numpy as np

from .cable_contact_criteria import segment_distance
from .cable_hitch_fixture import arc_lengths, smooth
from .cable_spot_tie_fixture import (
    CORD_RADIUS,
    END,
    HALF_SPAN,
    RELEASE,
    SERVICE_START,
    WIRE_COUNT,
    WIRE_NODES,
    WIRE_RADIUS,
    SpotTie,
)


def piece(q, bounds):
    a, b = bounds
    ids = np.r_[a, np.arange(np.floor(a) + 1, np.ceil(b)), b]
    return np.column_stack([np.interp(ids, np.arange(len(q)), axis) for axis in q.T])


def intersections(a, b):
    """Intersect two measured XY polylines; interpolate signed separation along the third axis."""

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    u, v = np.diff(a, axis=0)[:, None], np.diff(b, axis=0)[None]
    delta = b[None, :-1] - a[:-1, None]
    det = cross(u, v)
    den = np.where(abs(det) > 1e-14, det, np.inf)
    r, s = cross(delta, v) / den, cross(delta, u) / den
    ii, jj = np.nonzero((abs(det) > 1e-14) & (r >= 0) & (r < 1) & (s >= 0) & (s < 1))
    hits = []
    for i, j in zip(ii, jj, strict=True):
        p = a[i] + r[i, j] * u[i, 0]
        other = b[j] + s[i, j] * v[0, j]
        hits.append((float(p[0]), float(p[2] - other[2])))
    return sorted(hits)


def certify_spot(q, t, w):
    fixture = SpotTie()
    parts = {name: piece(q, fixture.material_span(t, name)) for name in fixture.marks}

    def face(p):
        relative = p - w[:, 8].mean(axis=0)
        return np.column_stack(
            (
                np.arctan2(relative[:, 0], relative[:, 1]),
                relative[:, 2],
                np.linalg.norm(relative[:, :2], axis=1),
            )
        )

    bridge = face(parts["bridge"])
    clove = {
        side: [1000 * gap for _, gap in intersections(bridge, face(parts[side]))] for side in ("clove_in", "clove_out")
    }
    section = piece(q, [fixture.material_span(t, "clove_in")[0], fixture.material_span(t, "clove_out")[1]])
    local = section - w[:, 8].mean(axis=0)
    winding = abs(np.diff(np.unwrap(np.arctan2(local[:, 0], local[:, 1]))).sum() / (2 * np.pi))
    braids = {}
    for level in ("lower", "upper"):
        a, b = [parts[f"{s}_{level}"][:, [2, 1, 0]] for s in "ab"]
        hits = intersections(a, b)
        a, b = [p[np.argsort(p[:, 0])] for p in (a, b)]
        grid = np.linspace(max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0]), 100)
        delta = [np.interp(grid, a[:, 0], a[:, i]) - np.interp(grid, b[:, 0], b[:, i]) for i in (1, 2)]
        twist = np.diff(np.unwrap(np.arctan2(delta[1], delta[0]))).sum() / np.pi
        braids[level] = {"depth_gaps_mm": [1000 * gap for _, gap in hits], "half_twists": float(twist)}
    valid = all(len(g) == 1 and g[0] > 2.88 for g in clove.values()) and winding >= 1.8
    for a in braids.values():
        g = np.array(a["depth_gaps_mm"])
        valid = valid and len(g) == 3 and np.array_equal(np.sign(g), [-1, 1, -1]) and min(abs(g)) >= 2.88
    valid = valid and braids["lower"]["half_twists"] * braids["upper"]["half_twists"] < 0
    return {
        "valid": bool(valid),
        "clove_turns": float(winding),
        "bridge_over_legs_mm": clove,
        "square_half_knots": braids,
    }


def winding(cord, centers):
    aa = np.unwrap(np.arctan2(cord[:, 1] - centers[..., 1, :], cord[:, 0] - centers[..., 0, :]), axis=-1)
    return (aa[..., -1] - aa[..., 0]) / (2 * np.pi)


def closure_clearance(positions, starts, cord_start):
    """Check an exterior closing path from formation onward, including locally adjacent nonincident edges."""
    n = positions.shape[1]
    tip_pair = positions[:, [cord_start, -1]].copy()
    cap = tip_pair.copy()
    cap[:, :, 1] = 1.0
    cap[:, :, 2] += (1 - tip_pair[:, :, 1]) * [-0.35, 0.35]
    nodes = np.concatenate((positions, cap), axis=1)
    helpers = [(cord_start, n), (n, n + 1), (n + 1, n - 1)]
    pairs = [(a, b, c, c + 1) for a, b in helpers for c in starts if a not in (c, c + 1) and b not in (c, c + 1)]
    pairs.append((*helpers[0], *helpers[2]))
    pairs.extend((a, a + 1, a + gap, a + gap + 1) for gap in (2, 3) for a in range(cord_start, n - gap - 1))
    a, b, c, d = np.asarray(pairs).T
    lower = np.inf
    previous = last_dist = None
    for xyz in nodes:
        dist = segment_distance(xyz[a], xyz[b], xyz[c], xyz[d])[0]
        if previous is not None:
            speed = np.linalg.norm(xyz - previous, axis=1)
            lower = min(
                lower,
                0.5 * np.min(last_dist + dist - np.maximum(speed[a], speed[b]) - np.maximum(speed[c], speed[d])),
            )
        previous, last_dist = xyz, dist
    return float(1000 * lower)


def contact_geometry(q, cord_start, formed):
    """Bound every pair's centerline separation between adjacent published frames."""
    chains = np.r_[np.repeat(np.arange(WIRE_COUNT), WIRE_NODES), np.full(q.shape[1] - cord_start, WIRE_COUNT)]
    starts = np.flatnonzero(chains[:-1] == chains[1:])
    i, j = np.triu_indices(len(starts), 1)
    a, b = starts[i], starts[j]
    keep = (chains[a] != chains[b]) | (b - a >= 4)
    a, b = a[keep], b[keep]
    group = np.where(chains[b] != WIRE_COUNT, 2, np.where(chains[a] == WIRE_COUNT, 0, 1))
    radii = np.where(chains == WIRE_COUNT, CORD_RADIUS, WIRE_RADIUS)
    radius_sum = radii[a] + radii[b]
    minimum = np.full(3, np.inf)
    maximum_overlap = np.zeros(3)
    previous = previous_dist = None
    for xyz in q:
        dist, _, along_b = segment_distance(xyz[a], xyz[a + 1], xyz[b], xyz[b + 1])
        overlap = radius_sum - dist
        maximum_overlap = np.maximum(maximum_overlap, [overlap[group == g].max() for g in range(3)])
        if previous is not None:
            movement = np.linalg.norm(xyz - previous, axis=1)
            edge = np.maximum(movement[:-1], movement[1:])
            bound = 0.5 * (dist + previous_dist - edge[a] - edge[b])
            minimum = np.minimum(minimum, [bound[group == g].min() for g in range(3)])
        previous, previous_dist = xyz, dist
    # Use the terminal contact's material coordinate on the initially straight cord.
    self_touch = (group == 0) & (overlap > 0)
    reserve = 0.0
    if self_touch.any():
        arc = arc_lengths(q[0, cord_start:])
        idx = b[self_touch] - cord_start
        end_contact = np.max(arc[idx] + along_b[self_touch] * np.diff(arc)[idx])
        reserve = 1000 * (arc[-1] - end_contact)
    return {
        "overlap_mm": (1000 * maximum_overlap).tolist(),
        "centerline_bound_mm": float(1000 * minimum.min()),
        "closure_bound_mm": closure_clearance(q[formed:], starts, cord_start),
        "terminal_tail_mm": float(reserve),
    }


def wrap_geometry(wires, cord):
    """Trace both physical back crossings against each wire's current centerline."""
    minimum = np.inf
    change = 0.0
    inside = True
    first_turns = None
    for wire, strand in zip(wires, cord, strict=True):
        if np.any(np.diff(wire[:, :, 2], axis=1) <= 0):
            raise ValueError("Wire doubles back along the supported span")
        outlines = np.array([[np.interp(strand[:, 2], w[:, 2], w[:, k]) for k in (0, 1)] for w in wire])
        minimum = min(minimum, np.abs(winding(strand, outlines)).min())
        center = outlines.mean(axis=0)
        turns = abs(winding(strand, center))
        if first_turns is None:
            first_turns = turns
        change = max(change, abs(turns - first_turns))
        dx, dy = strand[:, :2].T - center
        edges = np.flatnonzero((dx[:-1] > 0) != (dx[1:] > 0))
        frac = dx[edges] / (dx[edges] - dx[edges + 1])
        back = dy[edges] + frac * (dy[edges + 1] - dy[edges]) < 0
        edges, frac = edges[back], frac[back]
        crossing_z = strand[edges, 2] + frac * (strand[edges + 1, 2] - strand[edges, 2])
        inside = inside and len(edges) == 2 and np.all(np.abs(crossing_z) < HALF_SPAN - CORD_RADIUS)
    return {"min_wire_turns": float(minimum), "max_turn_change": float(change), "both_wraps_on_span": bool(inside)}


def tip_excursions_mm(t, cord, bundle_y, service_start):
    """Include the entire ramp and hold so a stable final frame cannot hide early slip."""
    ramp = np.searchsorted(t, service_start - 1e-9)
    tips = 1000 * (cord[:, [0, -1], 1] - bundle_y[:, None])
    return np.ptp(tips[ramp - 1 :], axis=0).tolist()


def evaluate_spot_tie(t, q, fixture_force, end_grips, load, contact_y):
    """Score the complete rollout, including startup motion and contact during dressing."""
    if t[0] != 0 or abs(t[-1] - END) > 1e-9 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > 0.003 + 1e-9:
        raise ValueError("Expected a complete straight-start spot tie trajectory")
    q = q.astype(np.float64, copy=False)
    cord_start = WIRE_COUNT * WIRE_NODES
    wires = q[:, :cord_start].reshape(len(t), WIRE_COUNT, WIRE_NODES, 3)
    cord = q[:, cord_start:]
    formed = int(np.abs(t - 1.25).argmin())
    release, ramp, full, last = np.searchsorted(
        t, [RELEASE - 1e-9, SERVICE_START - 1e-9, SERVICE_START + 0.04 - 1e-9, END - 0.5 - 1e-9]
    )
    bundle_y = wires[:, :, WIRE_NODES // 2, 1].mean(axis=1)
    loaded_y = 1000 * (cord[:, 0, 1] - bundle_y)
    plateau_ranges, rest_forces = [], []
    proof_y = np.zeros_like(t)
    for start, magnitude in ((6.12, 0.012), (11.10, 0.016), (16.08, 0.008)):
        age = t - start
        proof_y += magnitude * smooth(age / 0.04) * (1 - smooth((age - 4.04) / 0.80))
        hold = (t >= start + 0.04) & (t <= start + 4.04)
        rest = (t >= start + 4.84) & (t < start + 4.98)
        force_y = 1000 * end_grips[hold, 0, 1]
        plateau_ranges.append((float(force_y.min()), float(force_y.max())))
        rest_forces.append(float(fixture_force[rest].max()))
    proof_grip = np.zeros_like(end_grips[:, 0])
    proof_grip[:, 1] = proof_y
    proof_window = t >= 6.12
    return {
        "proof_profile_error_N": float(np.max(np.abs((end_grips[:, 0] - proof_grip)[proof_window]))),
        "service_profile_error_N": float(np.max(np.abs(load - 0.012 * smooth((t - 21.06) / 0.04)))),
        "formation": certify_spot(cord[formed], t[formed], wires[formed]),
        "proof_plateaus_mN": plateau_ranges,
        "proof_unloaded_N": rest_forces,
        "free_jaw_during_proofs_N": float(np.linalg.norm(end_grips[t >= 6.12, 1], axis=1).max()),
        "max_released_fixture_force_N": float(fixture_force[release:].max()),
        "no_early_service_load": bool(np.all(load[:ramp] == 0)),
        "full_load_12mN": bool(np.allclose(load[full:], 0.012, atol=1e-9, rtol=0)),
        "full_load_duration_s": float(t[-1] - t[full]),
        "tip_excursions_mm": tip_excursions_mm(t, cord, bundle_y, SERVICE_START),
        "end_excursion_last_half_s_mm": float(np.ptp(loaded_y[last:])),
        "opposing_reaction_mN": float(-1000 * np.average(contact_y[last + 1 :], weights=np.diff(t[last:]))),
        "support_motion_mm": float(1000 * np.linalg.norm(wires[:, :, [0, -1]] - wires[:1, :, [0, -1]], axis=-1).max()),
        "final_bundle_deflection_mm": float(1000 * (bundle_y[-1] - bundle_y[0])),
        **wrap_geometry(wires[release:], cord[release:]),
        **contact_geometry(q, cord_start, formed),
    }
