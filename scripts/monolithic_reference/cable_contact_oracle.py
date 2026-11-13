# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent FP64 NumPy reference for cable-cable contact (P9c method A').

Sample points of one cable part are tested against the *segments* of another.
A sample-segment pair has the energy

    E = w * psi * phi(d - (rA + rB)),

with w the sample weight [m], d the distance from the sample point to the
segment's line (clamped only at a free end, a spherical cap), phi the C-IE
penalty k_l P^2 / 2 with line stiffness k_l [N/m^2] and the C2-smoothed overlap
P, and psi a C1 partition of unity over the target polyline. At an interior
vertex v both adjacent segments use the same window coordinate
s = (p - v) . tau_v, with tau_v a vertex tangent frozen for the step (the
bisector of the adjacent edges), so their windows sum to one for every sample
position, on or off the axis; the window half width is a quarter of the shorter
adjacent rest edge. A free end does not decay.

Samples: three Gauss-Legendre points per edge with weight Gauss weight x rest
length, plus one point at each free end of a cable with weight equal to the
cable radius (the hemispherical tip), so a tip pushed into another strand loads
from zero overlap. Counting is one way: inside a cable only a lower edge index
samples a higher segment index, between cables only the lower cable index samples.

The gradient is exact (window derivative included, tau held fixed); the tangent
is the Gauss-Newton term w psi phi'' (c n)(c n)^T with c = (1 - xi, xi, -(1 - t), -t).
This module never imports the solver; tests compare the Warp kernels with it.
"""

import numpy as np

GAUSS = {
    1: (np.array([0.5]), np.array([1.0])),
    2: (np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)]), np.array([0.5, 0.5])),
    3: (
        np.array([0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)]),
        np.array([5.0, 8.0, 5.0]) / 18.0,
    ),
}
#: Partition-of-unity half width at an interior vertex: this fraction of the shorter adjacent rest edge.
WINDOW_FRACTION = 0.25


def smoothstep(u):
    """C1 ramp S(u) = 3u^2 - 2u^3 on [0, 1], clamped outside, and its derivative; S(u) + S(1 - u) = 1."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


def penalty(gap, stiffness, smoothing):
    """phi = k P^2 / 2 of the C2-smoothed overlap P(u), u = -gap, and its first two u-derivatives."""
    u = -np.asarray(gap, dtype=float)
    p, dp, ddp = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
    deep = u >= smoothing
    p[deep], dp[deep] = u[deep], 1.0
    band = (u > -smoothing) & ~deep
    t = (u[band] + smoothing) / (2.0 * smoothing)
    p[band] = smoothing * t**3 * (2.0 - t)
    dp[band] = t * t * (3.0 - 2.0 * t)
    ddp[band] = 3.0 * t * (1.0 - t) / smoothing
    return stiffness * p * p / 2.0, stiffness * p * dp, stiffness * (dp * dp + p * ddp)


def window(s_start, s_end, eps_start, eps_end):
    """psi and its derivatives with respect to the two vertex coordinates; eps = 0 marks a free end.

    ``s_start = (p - y0) . tau0`` rises through the start ramp, ``s_end = (p - y1) . tau1`` falls
    through the end ramp. The neighbour across a vertex sees the same coordinate, so the two windows
    sum to one there for every p.
    """
    s_start, s_end = np.asarray(s_start, dtype=float), np.asarray(s_end, dtype=float)
    eps_start, eps_end = np.asarray(eps_start, dtype=float), np.asarray(eps_end, dtype=float)
    start, end = eps_start > 0, eps_end > 0
    es, ee = np.where(start, eps_start, 1.0), np.where(end, eps_end, 1.0)
    a, da = smoothstep((s_start + es) / (2.0 * es))
    b, db = smoothstep((ee - s_end) / (2.0 * ee))
    a, da = np.where(start, a, 1.0), np.where(start, da / (2.0 * es), 0.0)
    b, db = np.where(end, b, 1.0), np.where(end, -db / (2.0 * ee), 0.0)
    return a * b, da * b, a * db


def pair_response(
    xa, xb, y0, y1, tau0, tau1, xi, weight, radius_sum, stiffness, smoothing, eps_start, eps_end, *, split=False
):
    """Energy [J], gradient (..., 4, 3) [N], Gauss-Newton coefficient [N/m] and direction (..., 4, 3).

    Positions have shape (..., 3): sample edge (xa, xb), target segment (y0, y1); ``tau0``/``tau1`` are
    the frozen unit tangents of the target's vertices (ignored at a free end). The pair tangent is
    coefficient * outer(direction, direction) over the 12 coordinates. With ``split`` the gradient is
    returned as its normal part and its window-derivative part.
    """
    xa, xb, y0, y1, tau0, tau1 = (np.asarray(v, dtype=float) for v in (xa, xb, y0, y1, tau0, tau1))
    xi = np.asarray(xi, dtype=float)
    eps_start, eps_end = np.asarray(eps_start, dtype=float), np.asarray(eps_end, dtype=float)
    p = (1.0 - xi)[..., None] * xa + xi[..., None] * xb
    e = y1 - y0
    length = np.linalg.norm(e, axis=-1)
    if np.any(length <= 0):
        raise ValueError("Degenerate target segment")
    # Inside a window the closest point runs on the segment's line, so two collinear segments give the
    # same distance across their shared vertex; only a free end clamps (a spherical cap).
    t = np.sum((p - y0) * e, axis=-1) / (length * length)
    t = np.where(eps_start > 0, t, np.maximum(t, 0.0))
    t = np.where(eps_end > 0, t, np.minimum(t, 1.0))
    y = (p - y0) - t[..., None] * e
    distance = np.linalg.norm(y, axis=-1)
    psi, dpsi_start, dpsi_end = window(
        np.sum((p - y0) * tau0, axis=-1), np.sum((p - y1) * tau1, axis=-1), eps_start, eps_end
    )
    # A pair outside the target's window contributes nothing, even when the sample is collinear with it.
    active = psi > 0
    if np.any(active & ~(distance > 0)):
        raise ValueError("Sample point lies on the target axis")
    safe = np.where(active, distance, 1.0)
    n = np.where(active[..., None], y / safe[..., None], 0.0)
    phi, dphi, ddphi = (np.where(active, v, 0.0) for v in penalty(safe - radius_sum, stiffness, smoothing))
    c = np.stack([1.0 - xi, xi, -(1.0 - t), -t], axis=-1)
    direction = c[..., None] * n[..., None, :]  # d(distance)/dx, shape (..., 4, 3)
    dpsi_dp = dpsi_start[..., None] * tau0 + dpsi_end[..., None] * tau1
    dpsi = np.stack(
        [
            (1.0 - xi)[..., None] * dpsi_dp,
            xi[..., None] * dpsi_dp,
            -dpsi_start[..., None] * tau0,
            -dpsi_end[..., None] * tau1,
        ],
        axis=-2,
    )
    w = np.asarray(weight, dtype=float)
    energy = w * psi * phi
    normal = w[..., None, None] * (-(psi * dphi)[..., None, None] * direction)
    windowed = w[..., None, None] * phi[..., None, None] * dpsi
    coefficient = w * psi * ddphi
    if split:
        return energy, normal, windowed, coefficient, direction
    return energy, normal + windowed, coefficient, direction


# ---- static tables over a set of cables ------------------------------------------------------


def segment_table(rest_positions, vertex_ranges, radii, *, window_fraction=WINDOW_FRACTION):
    """Per-segment first vertex, cable, local index, rest arc interval, rest length and window widths."""
    rows = []
    for cable, (a, b) in enumerate(vertex_ranges):
        rest = np.asarray(rest_positions[a:b], dtype=float)
        lengths = np.linalg.norm(np.diff(rest, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        count = len(lengths)
        for k in range(count):
            start = window_fraction * min(lengths[k - 1], lengths[k]) if k > 0 else 0.0
            end = window_fraction * min(lengths[k], lengths[k + 1]) if k + 1 < count else 0.0
            rows.append((a + k, cable, k, arc[k], arc[k + 1], lengths[k], start, end, radii[cable]))
    table = np.asarray(rows, dtype=float)
    return {
        "first": table[:, 0].astype(np.int64),
        "cable": table[:, 1].astype(np.int64),
        "index": table[:, 2].astype(np.int64),
        "arc0": table[:, 3],
        "arc1": table[:, 4],
        "rest_length": table[:, 5],
        "eps_start": table[:, 6],
        "eps_end": table[:, 7],
        "radius": table[:, 8],
    }


def sample_table(segments, samples_per_edge):
    """Gauss samples (edge-major), then one tip sample at each free end of each cable (weight = radius)."""
    nodes, weights = GAUSS[samples_per_edge]
    count = len(segments["first"])
    edge = np.repeat(np.arange(count), len(nodes))
    xi = np.tile(nodes, count)
    weight = np.tile(weights, count) * segments["rest_length"][edge]
    tips = []
    for cable in np.unique(segments["cable"]):
        own = np.flatnonzero(segments["cable"] == cable)
        tips += [(own[0], 0.0), (own[-1], 1.0)]
    tip_edge = np.asarray([k for k, _ in tips], dtype=np.int64)
    edge = np.r_[edge, tip_edge]
    xi = np.r_[xi, [x for _, x in tips]]
    weight = np.r_[weight, segments["radius"][tip_edge]]
    arc = segments["arc0"][edge] + xi * segments["rest_length"][edge]
    tip = np.r_[np.zeros(count * len(nodes), dtype=bool), np.ones(len(tips), dtype=bool)]
    return {"edge": edge, "xi": xi, "weight": weight, "arc": arc, "tip": tip}


def vertex_tangents(positions, segments):
    """Unit tangent at the start vertex of each segment: bisector of the adjacent edges, the edge at a free end."""
    positions = np.asarray(positions, dtype=float)
    first = segments["first"]
    e = positions[first + 1] - positions[first]
    e /= np.linalg.norm(e, axis=1)[:, None]
    tau = e.copy()
    k = np.flatnonzero(segments["eps_start"] > 0)
    bisector = e[k - 1] + e[k]
    size = np.linalg.norm(bisector, axis=1)
    # A fully folded vertex has no bisector; any tangent shared by both sides keeps the partition.
    ok = size > 1e-9
    tau[k] = np.where(ok[:, None], bisector / np.where(ok, size, 1.0)[:, None], e[k])
    return tau


def allowed(samples, segments, sample, target, *, index_span, arc_factor):
    """One-way counting and the near-neighbour exclusion of PRD decision 5, for (sample, target segment) pairs."""
    edge = samples["edge"][sample]
    cable_a, cable_b = segments["cable"][edge], segments["cable"][target]
    i, j = segments["index"][edge], segments["index"][target]
    reach = arc_factor * (segments["radius"][edge] + segments["radius"][target])
    gap = np.maximum.reduce(
        [
            np.zeros(np.shape(sample)),
            segments["arc0"][target] - samples["arc"][sample],
            samples["arc"][sample] - segments["arc1"][target],
        ]
    )
    same = cable_a == cable_b
    keep_same = (i < j) & (j - i > index_span) & (gap >= reach)
    return np.where(same, keep_same, cable_a < cable_b)


def all_pairs(samples, segments, *, index_span, arc_factor):
    """Every allowed (sample, target segment) pair, without any distance culling."""
    sample, target = np.meshgrid(np.arange(len(samples["edge"])), np.arange(len(segments["first"])), indexing="ij")
    sample, target = sample.ravel(), target.ravel()
    keep = allowed(samples, segments, sample, target, index_span=index_span, arc_factor=arc_factor)
    return sample[keep], target[keep]


def evaluate(positions, segments, samples, sample, target, *, stiffness, smoothing, tangents=None):
    """Total energy [J], per-vertex gradient (n, 3) [N] and per-pair results for the given pairs.

    ``tangents`` are the frozen vertex tangents (``vertex_tangents`` of the step-start positions);
    by default they are taken from ``positions`` themselves.
    """
    positions = np.asarray(positions, dtype=float)
    tau = vertex_tangents(positions, segments) if tangents is None else np.asarray(tangents, dtype=float)
    edge = samples["edge"][sample]
    a, b = segments["first"][edge], segments["first"][target]
    # The end tangent of segment j is the start tangent of segment j + 1 (only read when j has a successor).
    tau1 = tau[np.minimum(target + 1, len(segments["first"]) - 1)]
    energy, gradient, coefficient, direction = pair_response(
        positions[a],
        positions[a + 1],
        positions[b],
        positions[b + 1],
        tau[target],
        tau1,
        samples["xi"][sample],
        samples["weight"][sample],
        segments["radius"][edge] + segments["radius"][target],
        stiffness,
        smoothing,
        segments["eps_start"][target],
        segments["eps_end"][target],
    )
    total = np.zeros_like(positions)
    for slot, vertex in enumerate((a, a + 1, b, b + 1)):
        np.add.at(total, vertex, gradient[:, slot])
    return float(np.sum(energy)), total, {"energy": energy, "coefficient": coefficient, "direction": direction}


def onset_overlap(samples_per_edge, length_over_radius):
    """Worst-phase overlap / r at which a perpendicular crossing first loads (two equal radii r).

    The crossing point sits midway in the largest along-cable gap between consecutive sample
    points (within an edge or across a vertex); force starts when the sample is within 2r.
    """
    nodes, _ = GAUSS[samples_per_edge]
    gaps = np.r_[np.diff(nodes), 1.0 - nodes[-1] + nodes[0]] * length_over_radius
    half = np.max(gaps) / 2.0
    return 2.0 - np.sqrt(max(4.0 - half * half, 0.0))


def contact_energy(
    positions,
    rest_positions,
    vertex_ranges,
    radii,
    *,
    stiffness,
    smoothing,
    samples_per_edge,
    index_span,
    arc_factor,
    window_fraction=WINDOW_FRACTION,
    pairs=None,
    tangents=None,
):
    """Total cable-cable penalty energy [J] of one configuration, for the offline energy criterion.

    Pairs farther apart than every window and radius allow are culled; the result equals the sum over
    all allowed pairs. Pass ``pairs`` (from a previous call's tables) to reuse the static tables.
    """
    if pairs is None:
        segments = segment_table(rest_positions, vertex_ranges, radii, window_fraction=window_fraction)
        samples = sample_table(segments, samples_per_edge)
        pairs = (segments, samples, *all_pairs(samples, segments, index_span=index_span, arc_factor=arc_factor))
    segments, samples, sample, target = pairs
    positions = np.asarray(positions, dtype=float)
    edge = samples["edge"][sample]
    a, b = segments["first"][edge], segments["first"][target]
    xi = samples["xi"][sample][:, None]
    point = (1.0 - xi) * positions[a] + xi * positions[a + 1]
    middle = 0.5 * (positions[b] + positions[b + 1])
    half = 0.5 * np.linalg.norm(positions[b + 1] - positions[b], axis=1)
    reach = half + np.maximum(segments["eps_start"][target], segments["eps_end"][target])
    reach += segments["radius"][edge] + segments["radius"][target] + smoothing
    near = np.linalg.norm(point - middle, axis=1) <= reach
    if not np.any(near):
        return 0.0, pairs
    energy, _, _ = evaluate(
        positions,
        segments,
        samples,
        sample[near],
        target[near],
        stiffness=stiffness,
        smoothing=smoothing,
        tangents=tangents,
    )
    return energy, pairs
