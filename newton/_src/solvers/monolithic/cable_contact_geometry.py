# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-to-segment geometry and local C-IE response for cable-cable contact (method A').

Each cable edge carries Gauss sample points; a sample is tested against the
*segments* of another cable part, not its vertices. One sample-segment pair has

    E = w * psi(sigma, L) * phi(d - (r_A + r_B)),

where ``w`` is the Gauss weight times the rest edge length [m], ``d`` the distance
from the sample point to the closest point on the segment's line (clamped only at
a free end, so collinear neighbours agree across their shared vertex), ``phi`` the
C-IE penalty ``k_l P^2 / 2`` (``contact._penalty64``) with the line stiffness
``k_l`` [N/m^2], and ``psi`` a C1 partition of unity over the target polyline:
at an interior vertex the windows of the two adjacent segments overlap over
``+-eps`` and sum to one on a straight line, while a free end does not decay
(a spherical end cap). ``sigma = (p - y0) . e_hat`` is the projected arc length
along the target segment and ``L`` its current length, so the energy is C1 in
all twelve coordinates for ``d > 0``, including near-parallel strands.

The gradient is exact, window derivative included. The tangent keeps only the
Gauss-Newton term ``w psi phi'' (c n)(c n)^T`` with ``c = (1 - xi, xi, -(1 - t), -t)``,
which is positive semidefinite. Callers pass positions relative to a nearby
anchor (start-of-step relative position plus the exact increment) so FP32 keeps
the sub-millimetre gaps. Counting is one way: inside a cable only the lower edge
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
#: Window half width at an interior vertex as a fraction of the shorter adjacent rest edge (K5).
WINDOW_FRACTION = 0.25
#: Identity strings of the discrete rule; they enter the checkpoint identity once the method is wired.
RULES = {
    "sampling": "gauss-legendre-per-edge, length weighted",
    "target": "clamped segment, C1 smoothstep partition of unity, free ends undecayed",
    "window": f"half width {WINDOW_FRACTION} x shorter adjacent rest edge",
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
def cable_window(sigma: Any, length: Any, eps_start: Any, eps_end: Any):
    """Window psi(sigma, L) and its partial derivatives; eps = 0 marks an undecayed free end."""
    zero, one, two = type(sigma)(0.0), type(sigma)(1.0), type(sigma)(2.0)
    s0, d0, s1, d1 = one, zero, one, zero
    if eps_start > zero:
        value, slope = _smoothstep((sigma + eps_start) / (two * eps_start))
        s0, d0 = value, slope / (two * eps_start)
    if eps_end > zero:
        value, slope = _smoothstep((length - sigma + eps_end) / (two * eps_end))
        s1, d1 = value, slope / (two * eps_end)
    return s0 * s1, d0 * s1 - s0 * d1, s0 * d1


@wp.func
def cable_pair_response(
    xa: Any,
    xb: Any,
    y0: Any,
    y1: Any,
    xi: Any,
    weight: Any,
    radius_sum: Any,
    stiffness: Any,
    smoothing: Any,
    eps_start: Any,
    eps_end: Any,
):
    """Energy, the gradient on (xa, xb, y0, y1), the Gauss-Newton coefficient, normal n and segment parameter t.

    The pair tangent is ``coefficient * outer(J, J)`` with ``J = (c0 n, c1 n, c2 n, c3 n)`` and
    ``c = (1 - xi, xi, -(1 - t), -t)``. Status 0 is valid, 1 nonfinite, 2 a degenerate segment
    or a sample on the target axis.
    """
    zero, one = type(xi)(0.0), type(xi)(1.0)
    vzero = type(xa)(zero)
    p = (one - xi) * xa + xi * xb
    e = y1 - y0
    length = wp.length(e)
    if not (length > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, zero
    e_hat = e / length
    sigma = wp.dot(p - y0, e_hat)
    # Inside a window the closest point runs on the segment's line, so collinear neighbours give the same
    # distance across their shared vertex; only a free end (eps = 0) clamps, giving a spherical cap.
    t = sigma / length
    if eps_start == zero:
        t = wp.max(t, zero)
    if eps_end == zero:
        t = wp.min(t, one)
    psi, dpsi_dsigma, dpsi_dlength = cable_window(sigma, length, eps_start, eps_end)
    if psi == zero:
        # Outside the target's window: no contribution, even for a sample collinear with the segment.
        return int(0), zero, vzero, vzero, vzero, vzero, zero, vzero, t
    y = p - (y0 + t * e)
    distance = wp.length(y)
    if not (distance > zero):
        return int(2), zero, vzero, vzero, vzero, vzero, zero, vzero, t
    n = y / distance
    phi, dphi, ddphi = _penalty64(distance - radius_sum, stiffness, smoothing)
    # Normal part: d(distance)/dx = c n; the overlap u = -gap decreases with distance.
    normal = -weight * psi * dphi
    # Window part: phi * dpsi/dx through sigma(p, y0, y1) and L(y0, y1).
    h = (p - y0) - sigma * e_hat
    window = weight * phi
    ws, wl = window * dpsi_dsigma, window * dpsi_dlength
    ga = normal * (one - xi) * n + ws * (one - xi) * e_hat
    gb = normal * xi * n + ws * xi * e_hat
    g0 = -normal * (one - t) * n + ws * (-e_hat - h / length) - wl * e_hat
    g1 = -normal * t * n + ws * (h / length) + wl * e_hat
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


def build_tables(rest_positions, vertex_ranges, radii, *, samples_per_edge=3, window_fraction=WINDOW_FRACTION):
    """Static host tables over a set of cables (NumPy, FP64): segments and Gauss samples.

    ``vertex_ranges`` lists (first, end) particle indices of each cable in input order; edges are
    consecutive particles. Returns two dicts of arrays keyed like the device tables PR2b uploads.
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
        lengths = np.linalg.norm(np.diff(rest_positions[a:b], axis=0), axis=1)
        if np.any(~(lengths > 0)):
            raise ValueError(f"Cable {owner} has a zero-length rest edge")
        if radii[owner] > lengths.min():
            raise ValueError(f"Cable {owner} radius exceeds its shortest rest edge, which contact does not support")
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
            radius.append(float(radii[owner]))
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
    edge = np.repeat(np.arange(len(first), dtype=np.int32), len(nodes))
    samples = {
        "edge": edge,
        "xi": np.tile(nodes, len(first)),
        "weight": np.tile(weights, len(first)) * segments["rest_length"][edge],
        "arc": segments["arc0"][edge] + np.tile(nodes, len(first)) * segments["rest_length"][edge],
    }
    return segments, samples
