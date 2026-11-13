# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-to-segment geometry and local C-IE response for cable-cable contact (method A').

Sample points of one cable part are tested against the *segments* of another. One
sample-segment pair has

    E = w * psi * phi(d - (r_A + r_B)),

where ``w`` is the sample weight [m], ``d`` the distance from the sample point to
the segment's line (clamped only at a free end, a spherical cap), ``phi`` the
C-IE penalty ``k_l P^2 / 2`` (``contact._penalty64``) with line stiffness ``k_l``
[N/m^2], and ``psi`` a C1 partition of unity over the target polyline. At an
interior vertex ``v`` both adjacent segments use the same window coordinate
``s = (p - v) . tau_v``, with ``tau_v`` a vertex tangent frozen for the step, so
their windows sum to one for every sample position, on or off the axis. A free
end does not decay. The energy is C1 in all twelve coordinates for ``d > 0``,
including near-parallel strands.

Samples are three Gauss-Legendre points per edge (weight: Gauss weight times the
rest edge length) plus one point at each free end of a cable (weight: the cable
radius, the hemispherical tip). The gradient is exact for fixed ``tau``, window
derivative included. The tangent keeps only the Gauss-Newton term
``w psi phi'' (c n)(c n)^T`` with ``c = (1 - xi, xi, -(1 - t), -t)``, which is
positive semidefinite. Because ``tau`` is frozen, the window part of the force
carries a small net torque; translation balance is exact. Callers pass positions
relative to a nearby anchor (start-of-step relative position plus the exact
increment) so FP32 keeps sub-millimetre gaps. Counting is one way: inside a
cable only the lower edge index samples, between cables only the lower cable
index samples.
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
#: Window half width at an interior vertex as a fraction of the shorter adjacent rest edge (K5).
WINDOW_FRACTION = 0.25
#: Identity strings of the discrete rule; they enter the checkpoint identity once the method is wired.
RULES = {
    "sampling": "gauss-legendre-per-edge, length weighted; free-end tip sample weighted by the radius",
    "target": "segment line, clamped at free ends; C1 smoothstep partition of unity on shared frozen vertex tangents",
    "window": f"half width {WINDOW_FRACTION} x shorter adjacent rest edge, at most the cable radius",
    "counting": "one way: lower edge index within a cable, lower cable index between cables",
}


@wp.func
def _smoothstep(u: Any):
    """C1 ramp 3u^2 - 2u^3 clamped to [0, 1] and its slope; S(u) + S(1 - u) = 1."""
    zero, one = type(u)(0.0), type(u)(1.0)
    if u <= zero:
        return zero, zero
    if u >= one:
        return one, zero
    return u * u * (type(u)(3.0) - type(u)(2.0) * u), type(u)(6.0) * u * (one - u)


@wp.func
def cable_window(s_start: Any, s_end: Any, eps_start: Any, eps_end: Any):
    """Window psi and its derivatives in the two vertex coordinates; eps = 0 marks an undecayed free end.

    ``s_start = (p - y0) . tau0`` rises through the start ramp and ``s_end = (p - y1) . tau1`` falls
    through the end ramp; the neighbour across a vertex sees the same coordinate.
    """
    zero, one, two = type(s_start)(0.0), type(s_start)(1.0), type(s_start)(2.0)
    a, da, b, db = one, zero, one, zero
    if eps_start > zero:
        value, slope = _smoothstep((s_start + eps_start) / (two * eps_start))
        a, da = value, slope / (two * eps_start)
    if eps_end > zero:
        value, slope = _smoothstep((eps_end - s_end) / (two * eps_end))
        b, db = value, -slope / (two * eps_end)
    return a * b, da * b, a * db


@wp.func
def cable_pair_response(
    xa: Any,
    xb: Any,
    y0: Any,
    y1: Any,
    tau0: Any,
    tau1: Any,
    xi: Any,
    weight: Any,
    radius_sum: Any,
    stiffness: Any,
    smoothing: Any,
    eps_start: Any,
    eps_end: Any,
):
    """Energy, the gradient on (xa, xb, y0, y1), the Gauss-Newton coefficient, normal n and line parameter t.

    ``tau0``/``tau1`` are the frozen unit tangents of the target's vertices (unused at a free end). The
    pair tangent is ``coefficient * outer(J, J)`` with ``J = (c0 n, c1 n, c2 n, c3 n)`` and
    ``c = (1 - xi, xi, -(1 - t), -t)``. Status 0 is valid, 1 nonfinite, 2 a degenerate segment or a
    sample on the target axis inside its window.
    """
    zero, one = type(xi)(0.0), type(xi)(1.0)
    vzero = type(xa)(zero)
    p = (one - xi) * xa + xi * xb
    e = y1 - y0
    length2 = wp.dot(e, e)
    if not (length2 > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, zero
    psi, dpsi_start, dpsi_end = cable_window(wp.dot(p - y0, tau0), wp.dot(p - y1, tau1), eps_start, eps_end)
    if psi == zero:
        # Outside the target's window: no contribution, even for a sample collinear with the segment.
        return int(0), zero, vzero, vzero, vzero, vzero, zero, vzero, zero
    # Inside a window the closest point runs on the segment's line, so collinear neighbours give the same
    # distance across their shared vertex; only a free end (eps = 0) clamps, giving a spherical cap.
    t = wp.dot(p - y0, e) / length2
    if eps_start == zero:
        t = wp.max(t, zero)
    if eps_end == zero:
        t = wp.min(t, one)
    y = (p - y0) - t * e
    distance = wp.length(y)
    if not (distance > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, t
    n = y / distance
    phi, dphi, ddphi = _penalty64(distance - radius_sum, stiffness, smoothing)
    # Normal part: d(distance)/dx = c n, and the overlap u = -gap decreases with distance.
    normal = -weight * psi * dphi
    # Window part: phi * dpsi/dx through s_start(p, y0) and s_end(p, y1) with fixed tangents.
    window = weight * phi
    dp = window * (dpsi_start * tau0 + dpsi_end * tau1)
    ga = normal * (one - xi) * n + (one - xi) * dp
    gb = normal * xi * n + xi * dp
    g0 = -normal * (one - t) * n - window * dpsi_start * tau0
    g1 = -normal * t * n - window * dpsi_end * tau1
    energy = weight * psi * phi
    coefficient = weight * psi * ddphi
    status = int(0)
    if not wp.isfinite(energy) or not wp.isfinite(coefficient) or not wp.isfinite(wp.length(ga) + wp.length(g1)):
        status = int(1)
    return status, energy, ga, gb, g0, g1, coefficient, n, t


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
    """
    if cable_a != cable_b:
        return cable_a < cable_b
    if segment_index - edge_index <= index_span:
        return False
    gap = wp.max(segment_arc0 - sample_arc, sample_arc - segment_arc1)
    return wp.max(gap, type(gap)(0.0)) >= reach


@wp.func
def cable_vertex_tangent(previous: Any, vertex: Any, following: Any):
    """Frozen window tangent at an interior vertex: the bisector of its two edges.

    A fully folded vertex has no bisector; the following edge is used, which still keeps the two
    windows complementary because both sides read the same tangent.
    """
    a = wp.normalize(vertex - previous)
    b = wp.normalize(following - vertex)
    bisector = a + b
    size = wp.length(bisector)
    if size > type(size)(1e-9):
        return bisector / size
    return b


def build_tables(rest_positions, vertex_ranges, radii, *, samples_per_edge=3, window_fraction=WINDOW_FRACTION):
    """Static host tables over a set of cables (NumPy, FP64): segments and samples.

    ``vertex_ranges`` lists (first, end) particle indices of each cable in input order; edges are
    consecutive particles. Samples are the Gauss points of every edge (edge-major) followed by one
    tip sample at each free end of each cable, weighted by the cable radius. A window half width
    above the cable radius is rejected: inside a window the target's line extends past its vertex,
    and a longer extension could load samples outside the tube at a sharp bend.
    """
    if samples_per_edge not in GAUSS_RULES:
        raise ValueError(f"Cable contact supports {sorted(GAUSS_RULES)} samples per edge, got {samples_per_edge}")
    if not 0.0 < window_fraction <= 0.5:
        raise ValueError("Cable contact window fraction must lie in (0, 0.5]")
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    first, cable, index, arc0, arc1, rest, eps_start, eps_end, radius = ([] for _ in range(9))
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
        if len(lengths) > 1 and window_fraction * np.minimum(lengths[:-1], lengths[1:]).max() > r:
            raise ValueError(
                f"Cable {owner} edges are too long for contact: the window half width {window_fraction} x edge "
                f"exceeds the radius {r} (rest edges up to {r / window_fraction} m)"
            )
        arc = np.r_[0.0, np.cumsum(lengths)]
        for k in range(len(lengths)):
            first.append(a + k)
            cable.append(owner)
            index.append(k)
            arc0.append(arc[k])
            arc1.append(arc[k + 1])
            rest.append(lengths[k])
            eps_start.append(window_fraction * min(lengths[k - 1], lengths[k]) if k > 0 else 0.0)
            eps_end.append(window_fraction * min(lengths[k], lengths[k + 1]) if k + 1 < len(lengths) else 0.0)
            radius.append(r)
    segments = {
        "first": np.asarray(first, dtype=np.int32),
        "cable": np.asarray(cable, dtype=np.int32),
        "index": np.asarray(index, dtype=np.int32),
        "arc0": np.asarray(arc0),
        "arc1": np.asarray(arc1),
        "rest_length": np.asarray(rest),
        "eps_start": np.asarray(eps_start),
        "eps_end": np.asarray(eps_end),
        "radius": np.asarray(radius),
    }
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
        np.tile(weights, count) * segments["rest_length"][:count].repeat(len(nodes)), segments["radius"][tip_edge]
    ]
    samples = {
        "edge": edge,
        "xi": xi,
        "weight": weight,
        "arc": segments["arc0"][edge] + xi * segments["rest_length"][edge],
        "tip": np.r_[np.zeros(count * len(nodes), dtype=bool), np.ones(len(tip_edge), dtype=bool)],
    }
    return segments, samples
