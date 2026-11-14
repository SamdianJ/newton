# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-to-polyline geometry and local C-IE response for cable-cable contact (method A').

Sample points of one cable part are tested against the polyline of another. For
each sample the energy is

    E = sum_segments w * phi(d_seg) - sum_interior_vertices w * phi(d_vertex),

where ``w`` is the sample weight [m], ``d_seg`` the distance from the sample point
to the segment (clamped to its end points), ``d_vertex`` the distance to the
vertex, and ``phi`` the C-IE penalty ``k_l P^2 / 2`` (``contact._penalty64``) with
line stiffness ``k_l`` [N/m^2]. Two segments meeting at a vertex both clamp to it
on the convex side of a bend and past their ends on a straight line; the vertex
term removes that double count. On a straight line and on the convex side this
is exactly ``phi`` of the distance to the polyline; on the concave side both
segments count (a strand in a groove), never less than the nearest one. The
energy is C1 for ``d > 0`` (point-to-segment distance has no closest-point jump,
also for near-parallel strands), needs no frozen data, and is non-negative: a
vertex pair only exists when both adjacent segment pairs do.

Samples are three Gauss-Legendre points per edge (weight: Gauss weight times the
rest edge length) plus one point at each free end of a cable (weight: the cable
radius, the hemispherical tip). The gradient is exact. The tangent is the
Gauss-Newton term ``w phi'' (c n)(c n)^T`` of each segment pair with
``c = (1 - xi, xi, -(1 - t), -t)``; a vertex pair contributes its negative term,
``c = (1 - xi, xi, -1)``, only when an adjacent segment clamps to that vertex for
the same sample, where it cancels that segment's identical term, so the assembled
tangent stays positive semidefinite. Callers pass positions relative to a nearby
anchor (start-of-step relative position plus the exact increment) so FP32 keeps
sub-millimetre gaps. Counting is one way: inside a cable only the lower edge
index samples, between cables only the lower cable index samples.
"""

from typing import Any

import numpy as np
import warp as wp

from .contact import _penalty64

#: Gauss-Legendre nodes and weights on [0, 1] by samples per edge.
GAUSS_RULES = {
    2: ((0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)), (0.5, 0.5)),
    3: ((0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)), (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)),
}
#: Identity strings of the discrete rule; they enter the checkpoint identity once the method is wired.
RULES = {
    "sampling": "gauss-legendre-per-edge, length weighted; free-end tip sample weighted by the radius",
    "target": "clamped segments minus interior vertices (no double count at shared vertices)",
    "tangent": "Gauss-Newton; vertex term only where an adjacent segment clamps to it",
    "counting": "one way: lower edge index within a cable, lower cable index between cables",
}


@wp.func
def cable_segment_response(
    xa: Any, xb: Any, y0: Any, y1: Any, xi: Any, weight: Any, radius_sum: Any, stiffness: Any, smoothing: Any
):
    """Segment pair: energy, the gradient on (xa, xb, y0, y1), Gauss-Newton coefficient, normal n and parameter t.

    The pair tangent is ``coefficient * outer(J, J)`` with ``J = (c0 n, c1 n, c2 n, c3 n)`` and
    ``c = (1 - xi, xi, -(1 - t), -t)``. Status 0 is valid, 1 nonfinite, 2 a degenerate segment or a
    sample on the segment.
    """
    zero, one = type(xi)(0.0), type(xi)(1.0)
    vzero = type(xa)(zero)
    p = (one - xi) * xa + xi * xb
    e = y1 - y0
    length2 = wp.dot(e, e)
    if not (length2 > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, zero
    t = wp.clamp(wp.dot(p - y0, e) / length2, zero, one)
    y = (p - y0) - t * e
    distance = wp.length(y)
    if not (distance > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, t
    n = y / distance
    phi, dphi, ddphi = _penalty64(distance - radius_sum, stiffness, smoothing)
    # d(distance)/dx = c n, and the overlap u = R - distance decreases with distance.
    load = -weight * dphi
    energy = weight * phi
    coefficient = weight * ddphi
    ga = load * (one - xi) * n
    gb = load * xi * n
    g0 = -load * (one - t) * n
    g1 = -load * t * n
    status = int(0)
    if not wp.isfinite(energy) or not wp.isfinite(coefficient) or not wp.isfinite(load):
        status = int(1)
    return status, energy, ga, gb, g0, g1, coefficient, n, t


@wp.func
def cable_vertex_response(
    xa: Any,
    xb: Any,
    previous: Any,
    vertex: Any,
    following: Any,
    xi: Any,
    weight: Any,
    radius_sum: Any,
    stiffness: Any,
    smoothing: Any,
):
    """Vertex pair (the subtracted double count): energy <= 0, the gradient on (xa, xb, vertex), GN coefficient, n.

    The coefficient is the negative Gauss-Newton weight with ``c = (1 - xi, xi, -1)`` when the incoming
    (previous -> vertex) or outgoing (vertex -> following) segment clamps to the vertex for this sample,
    and zero otherwise; ``previous`` and ``following`` only decide that. Status as for a segment pair.
    """
    zero, one = type(xi)(0.0), type(xi)(1.0)
    vzero = type(xa)(zero)
    p = (one - xi) * xa + xi * xb
    y = p - vertex
    distance = wp.length(y)
    if not (distance > zero):
        return int(2), zero, vzero, vzero, vzero, zero, vzero
    n = y / distance
    phi, dphi, ddphi = _penalty64(distance - radius_sum, stiffness, smoothing)
    load = weight * dphi
    incoming = vertex - previous
    outgoing = following - vertex
    clamps = wp.dot(p - previous, incoming) >= wp.dot(incoming, incoming) or wp.dot(y, outgoing) <= zero
    coefficient = wp.where(clamps, -weight * ddphi, zero)
    energy = -weight * phi
    ga = load * (one - xi) * n
    gb = load * xi * n
    gv = -load * n
    status = int(0)
    if not wp.isfinite(energy) or not wp.isfinite(ddphi) or not wp.isfinite(load):
        status = int(1)
    return status, energy, ga, gb, gv, coefficient, n


@wp.func
def cable_pair_allowed(
    cable_a: int,
    cable_b: int,
    edge_index: int,
    segment_index: int,
    sample_arc: Any,
    segment_arc0: Any,
    segment_arc1: Any,
    reach: Any,
    index_span: int,
):
    """One-way counting and the near-neighbour exclusion of PRD decision 5 for one (sample, segment) pair.

    ``reach`` is ``alpha * (r_A + r_B)``. Inside one cable only a lower edge index samples a higher
    segment index, more than ``index_span`` edges away and at least ``reach`` away in rest arc length.
    A (sample, interior vertex) pair is allowed when both segments at the vertex are.
    """
    if cable_a != cable_b:
        return cable_a < cable_b
    if segment_index - edge_index <= index_span:
        return False
    gap = wp.max(segment_arc0 - sample_arc, sample_arc - segment_arc1)
    return wp.max(gap, type(gap)(0.0)) >= reach


def build_tables(rest_positions, vertex_ranges, radii, *, samples_per_edge=3):
    """Static host tables over a set of cables (NumPy, FP64): segments, interior vertices and samples.

    ``vertex_ranges`` lists (first, end) particle indices of each cable in input order; edges are
    consecutive particles. Samples are the Gauss points of every edge (edge-major) followed by one
    tip sample at each free end of each cable, weighted by the cable radius.
    """
    if samples_per_edge not in GAUSS_RULES:
        raise ValueError(f"Cable contact supports {sorted(GAUSS_RULES)} samples per edge, got {samples_per_edge}")
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    first, cable, index, arc0, arc1, rest, radius = ([] for _ in range(7))
    for owner, (a, b) in enumerate(vertex_ranges):
        if b - a < 2:
            raise ValueError(f"Cable {owner} needs at least two vertices")
        r = float(radii[owner])
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f"Cable {owner} contact radius must be finite and positive, got {r}")
        lengths = np.linalg.norm(np.diff(rest_positions[a:b], axis=0), axis=1)
        if np.any(~(lengths > 0)):
            raise ValueError(f"Cable {owner} has a zero-length rest edge")
        if r > lengths.min():
            raise ValueError(f"Cable {owner} radius exceeds its shortest rest edge, which contact does not support")
        arc = np.r_[0.0, np.cumsum(lengths)]
        for k in range(len(lengths)):
            first.append(a + k)
            cable.append(owner)
            index.append(k)
            arc0.append(arc[k])
            arc1.append(arc[k + 1])
            rest.append(lengths[k])
            radius.append(r)
    segments = {
        "first": np.asarray(first, dtype=np.int32),
        "cable": np.asarray(cable, dtype=np.int32),
        "index": np.asarray(index, dtype=np.int32),
        "arc0": np.asarray(arc0),
        "arc1": np.asarray(arc1),
        "rest_length": np.asarray(rest),
        "radius": np.asarray(radius),
    }
    incoming = np.flatnonzero(segments["cable"][1:] == segments["cable"][:-1]).astype(np.int32)
    vertices = {"incoming": incoming, "vertex": segments["first"][incoming] + 1}
    nodes, weights = (np.asarray(v) for v in GAUSS_RULES[samples_per_edge])
    count = len(first)
    tip_edge, tip_xi = [], []
    for owner in range(len(vertex_ranges)):
        own = np.flatnonzero(segments["cable"] == owner)
        tip_edge += [own[0], own[-1]]
        tip_xi += [0.0, 1.0]
    edge = np.r_[np.repeat(np.arange(count), len(nodes)), tip_edge].astype(np.int32)
    xi = np.r_[np.tile(nodes, count), tip_xi]
    weight = np.r_[
        np.tile(weights, count) * np.repeat(segments["rest_length"], len(nodes)), segments["radius"][tip_edge]
    ]
    samples = {
        "edge": edge,
        "xi": xi,
        "weight": weight,
        "arc": segments["arc0"][edge] + xi * segments["rest_length"][edge],
        "tip": np.r_[np.zeros(count * len(nodes), dtype=bool), np.ones(len(tip_edge), dtype=bool)],
    }
    return segments, vertices, samples
