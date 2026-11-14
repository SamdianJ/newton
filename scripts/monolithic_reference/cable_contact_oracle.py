# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent FP64 NumPy reference for cable-cable contact (P9c method A', segment minus vertex).

Sample points of one cable part are tested against the polyline of another. For
each sample the energy is

    E = sum_segments w * phi(d_seg) - sum_interior_vertices w * phi(d_vertex),

with w the sample weight [m], d_seg the distance from the sample point to the
segment (clamped to its end points), d_vertex the distance to the vertex, and phi
the C-IE penalty k_l P^2 / 2 with line stiffness k_l [N/m^2] and the C2-smoothed
overlap P = R_sum - d. Two segments that meet at a vertex both clamp to it on
the convex side and beyond their ends on a straight line; the vertex term removes
that double count. On a straight line and on the convex side of a bend this is
exactly phi of the distance to the polyline; on the concave side both segments
count (a strand in a groove), never less than the nearest one. The energy is C1
for d > 0, needs no frozen data, and is non-negative: a vertex term only enters
when both adjacent segments are counted, and each of them is at least as close.

Samples: three Gauss-Legendre points per edge with weight Gauss weight x rest
length, plus one point at each free end of a cable with weight equal to the cable
radius (the hemispherical tip). Counting is one way: inside a cable only a lower
edge index samples a higher segment index, between cables only the lower cable
index samples.

The gradient is exact. The tangent is the Gauss-Newton term w phi'' (c n)(c n)^T
of each segment pair with c = (1 - xi, xi, -(1 - t), -t); every coefficient is
non-negative. A vertex pair has no tangent term: a segment that clamps to a
vertex whose pair is counted drops its own term instead (the outgoing segment
only when the incoming one does not clamp there too). A clamped segment's term
equals the vertex term, so on a straight line and on the convex side the tangent
is that of the distance to the polyline; on the concave side the subtracted
vertex term is left out.
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


def penalty(gap, stiffness, smoothing):
    """phi = k P^2 / 2 of the C2-smoothed overlap P(u), u = -gap, and its first two u-derivatives."""
    u = -np.asarray(gap, dtype=float)
    p, dp, ddp = np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)
    deep = u >= smoothing
    p[deep], dp[deep] = u[deep], 1.0
    band = (u > -smoothing) & ~deep
    if smoothing > 0:
        t = (u[band] + smoothing) / (2.0 * smoothing)
        p[band] = smoothing * t**3 * (2.0 - t)
        dp[band] = t * t * (3.0 - 2.0 * t)
        ddp[band] = 3.0 * t * (1.0 - t) / smoothing
    return stiffness * p * p / 2.0, stiffness * p * dp, stiffness * (dp * dp + p * ddp)


def _sample_point(xa, xb, xi):
    xi = np.asarray(xi, dtype=float)
    return (1.0 - xi)[..., None] * np.asarray(xa, dtype=float) + xi[..., None] * np.asarray(xb, dtype=float)


def _projection(p, y0, y1):
    e = y1 - y0
    return np.sum((p - y0) * e, axis=-1), np.sum(e * e, axis=-1)


def segment_response(
    xa, xb, y0, y1, xi, weight, radius_sum, stiffness, smoothing, *, previous=None, start_vertex=False, end_vertex=False
):
    """Segment pair: energy [J], gradient (..., 4, 3) [N] on (xa, xb, y0, y1), GN coefficient [N/m], direction, t.

    ``start_vertex`` / ``end_vertex`` say whether the vertex pair of this sample at y0 / y1 is counted;
    the coefficient is then zero where the segment clamps to that vertex (at y0 only if the incoming
    segment ``previous -> y0`` does not clamp to it as well).
    """
    xi = np.asarray(xi, dtype=float)
    y0, y1 = np.asarray(y0, dtype=float), np.asarray(y1, dtype=float)
    p = _sample_point(xa, xb, xi)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y0)) and np.all(np.isfinite(y1))):
        raise ValueError("Nonfinite sample or target position")
    along, length2 = _projection(p, y0, y1)
    if np.any(length2 <= 0):
        raise ValueError("Degenerate target segment")
    at_start, at_end = along <= 0.0, along >= length2
    t = np.where(at_start, 0.0, np.where(at_end, 1.0, along / length2))
    y = np.where(at_start[..., None], p - y0, np.where(at_end[..., None], p - y1, (p - y0) - t[..., None] * (y1 - y0)))
    distance = np.linalg.norm(y, axis=-1)
    if np.any(~(distance > 0)):
        raise ValueError("Sample point lies on the target segment")
    n = y / distance[..., None]
    phi, dphi, ddphi = penalty(distance - radius_sum, stiffness, smoothing)
    c = np.stack([1.0 - xi, xi, -(1.0 - t), -t], axis=-1)
    direction = c[..., None] * n[..., None, :]
    w = np.asarray(weight, dtype=float)
    gradient = -(w * dphi)[..., None, None] * direction
    drop = at_end & np.asarray(end_vertex, dtype=bool)
    start = at_start & np.asarray(start_vertex, dtype=bool)
    if np.any(start):
        incoming_along, incoming_length2 = _projection(p, np.asarray(previous, dtype=float), y0)
        if not np.all(np.where(start, np.isfinite(incoming_along) & np.isfinite(incoming_length2), True)):
            raise ValueError("Nonfinite previous vertex read by a start clamp")
        drop = np.where(start, ~(incoming_along >= incoming_length2), drop)
    return w * phi, gradient, np.where(drop, 0.0, w * ddphi), direction, t


def vertex_response(xa, xb, vertex, xi, weight, radius_sum, stiffness, smoothing):
    """Vertex pair (the subtracted double count): energy [J] <= 0, gradient (..., 3, 3) on (xa, xb, vertex),
    direction. It has no tangent term; the clamped segments drop theirs instead."""
    xi = np.asarray(xi, dtype=float)
    vertex = np.asarray(vertex, dtype=float)
    p = _sample_point(xa, xb, xi)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(vertex))):
        raise ValueError("Nonfinite sample or vertex position")
    y = p - vertex
    distance = np.linalg.norm(y, axis=-1)
    if np.any(~(distance > 0)):
        raise ValueError("Sample point lies on the target vertex")
    n = y / distance[..., None]
    phi, dphi, _ = penalty(distance - radius_sum, stiffness, smoothing)
    c = np.stack([1.0 - xi, xi, -np.ones_like(xi)], axis=-1)
    direction = c[..., None] * n[..., None, :]
    w = np.asarray(weight, dtype=float)
    gradient = (w * dphi)[..., None, None] * direction
    return -w * phi, gradient, direction


# ---- static tables over a set of cables ------------------------------------------------------


def segment_table(rest_positions, vertex_ranges, radii):
    """Per-segment first vertex, cable, local index, rest arc interval, rest length and radius."""
    rows = []
    for cable, (a, b) in enumerate(vertex_ranges):
        rest = np.asarray(rest_positions[a:b], dtype=float)
        lengths = np.linalg.norm(np.diff(rest, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        for k in range(len(lengths)):
            rows.append((a + k, cable, k, arc[k], arc[k + 1], lengths[k], radii[cable]))
    table = np.asarray(rows, dtype=float)
    return {
        "first": table[:, 0].astype(np.int64),
        "cable": table[:, 1].astype(np.int64),
        "index": table[:, 2].astype(np.int64),
        "arc0": table[:, 3],
        "arc1": table[:, 4],
        "rest_length": table[:, 5],
        "radius": table[:, 6],
    }


def vertex_table(segments):
    """Interior vertices: the segment ending there (``incoming``); the outgoing segment is incoming + 1."""
    same = segments["cable"][1:] == segments["cable"][:-1]
    incoming = np.flatnonzero(same)
    return {"incoming": incoming, "vertex": segments["first"][incoming] + 1}


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
    """Every allowed (sample, segment) pair and every (sample, interior vertex) pair whose two segments are allowed."""
    count = len(segments["first"])
    sample, target = np.meshgrid(np.arange(len(samples["edge"])), np.arange(count), indexing="ij")
    keep = allowed(samples, segments, sample.ravel(), target.ravel(), index_span=index_span, arc_factor=arc_factor)
    keep = keep.reshape(sample.shape)
    vertices = vertex_table(segments)
    both = keep[:, vertices["incoming"]] & keep[:, vertices["incoming"] + 1]
    vertex_sample, vertex = np.nonzero(both)
    return {
        "segment": (sample[keep], target[keep]),
        "vertex": (vertex_sample, vertex),
    }


def vertex_flags(segments, pairs):
    """For each segment pair: whether the vertex pair of its sample at the segment start / end is in ``pairs``."""
    sample, target = pairs["segment"]
    vertices = vertex_table(segments)
    rows = len(vertices["vertex"])
    row_of = np.full(len(segments["first"]) + 1, -1)
    row_of[vertices["incoming"]] = np.arange(rows)
    counted = np.asarray(pairs["vertex"][0]) * rows + np.asarray(pairs["vertex"][1])
    end_row = row_of[target]
    start_row = np.where(target > 0, row_of[target - 1], -1)
    end = (end_row >= 0) & np.isin(sample * rows + end_row, counted)
    start = (start_row >= 0) & np.isin(sample * rows + start_row, counted)
    return start, end


def evaluate(positions, segments, samples, pairs, *, stiffness, smoothing):
    """Total energy [J], per-vertex gradient (n, 3) [N] and per-pair results for the given pairs.

    The segment results carry the Gauss-Newton coefficient under the counted vertex pairs of ``pairs``.
    """
    positions = np.asarray(positions, dtype=float)
    total = np.zeros_like(positions)
    energy = 0.0
    sample, target = pairs["segment"]
    edge = samples["edge"][sample]
    a, b = segments["first"][edge], segments["first"][target]
    start, end = vertex_flags(segments, pairs)
    seg = segment_response(
        positions[a],
        positions[a + 1],
        positions[b],
        positions[b + 1],
        samples["xi"][sample],
        samples["weight"][sample],
        segments["radius"][edge] + segments["radius"][target],
        stiffness,
        smoothing,
        previous=positions[np.where(start, b - 1, b)],
        start_vertex=start,
        end_vertex=end,
    )
    energy += float(np.sum(seg[0]))
    for slot, index in enumerate((a, a + 1, b, b + 1)):
        np.add.at(total, index, seg[1][:, slot])
    vertices = vertex_table(segments)
    sample, row = pairs["vertex"]
    edge = samples["edge"][sample]
    a, v = segments["first"][edge], vertices["vertex"][row]
    incoming = vertices["incoming"][row]
    ver = vertex_response(
        positions[a],
        positions[a + 1],
        positions[v],
        samples["xi"][sample],
        samples["weight"][sample],
        segments["radius"][edge] + segments["radius"][incoming],
        stiffness,
        smoothing,
    )
    energy += float(np.sum(ver[0]))
    for slot, index in enumerate((a, a + 1, v)):
        np.add.at(total, index, ver[1][:, slot])
    return energy, total, {"segment": seg, "vertex": ver}


def onset_overlap(samples_per_edge, length_over_radius, radius_sum_over_radius=2.0):
    """Worst-phase overlap / r at which a perpendicular crossing first loads, all lengths in units of r.

    The crossing point sits midway in the largest along-cable gap between consecutive sample
    points (within an edge or across a vertex); force starts when the sample is within the radius
    sum (2 r for two equal radii r).
    """
    nodes, _ = GAUSS[samples_per_edge]
    gaps = np.r_[np.diff(nodes), 1.0 - nodes[-1] + nodes[0]] * length_over_radius
    half = np.max(gaps) / 2.0
    reach = radius_sum_over_radius
    return reach - np.sqrt(max(reach * reach - half * half, 0.0))


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
    tables=None,
):
    """Total cable-cable penalty energy [J] of one configuration, for the offline energy criterion.

    Pairs farther apart than the radii allow are culled; the result equals the sum over all allowed
    pairs. Pass ``tables`` (returned by a previous call) to reuse the static tables.
    """
    if tables is None:
        segments = segment_table(rest_positions, vertex_ranges, radii)
        samples = sample_table(segments, samples_per_edge)
        tables = (segments, samples, all_pairs(samples, segments, index_span=index_span, arc_factor=arc_factor))
    segments, samples, pairs = tables
    positions = np.asarray(positions, dtype=float)

    def point(sample):
        edge, xi = samples["edge"][sample], samples["xi"][sample][:, None]
        a = segments["first"][edge]
        return (1.0 - xi) * positions[a] + xi * positions[a + 1], edge

    sample, target = pairs["segment"]
    p, edge = point(sample)
    b = segments["first"][target]
    middle, half = (
        0.5 * (positions[b] + positions[b + 1]),
        0.5 * np.linalg.norm(positions[b + 1] - positions[b], axis=1),
    )
    reach = half + segments["radius"][edge] + segments["radius"][target] + smoothing
    near_segment = np.linalg.norm(p - middle, axis=1) <= reach
    sample_v, row = pairs["vertex"]
    p, edge = point(sample_v)
    vertices = vertex_table(segments)
    reach = segments["radius"][edge] + segments["radius"][vertices["incoming"][row]] + smoothing
    near_vertex = np.linalg.norm(p - positions[vertices["vertex"][row]], axis=1) <= reach
    near = {
        "segment": (sample[near_segment], target[near_segment]),
        "vertex": (sample_v[near_vertex], row[near_vertex]),
    }
    energy, _, _ = evaluate(positions, segments, samples, near, stiffness=stiffness, smoothing=smoothing)
    return energy, tables
