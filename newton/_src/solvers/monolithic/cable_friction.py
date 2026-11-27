# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lagged Coulomb friction for Gauss samples contacting a cable polyline."""

from typing import Any

import warp as wp

from .cable_contact_geometry import cable_projection
from .contact import _penalty64


@wp.func
def _closest(tables: Any, start: wp.array[Any], anchor: int, point: Any, seg: int):
    first = start[tables.seg_a[seg]] - start[anchor]
    last = start[tables.seg_b[seg]] - start[anchor]
    along, length2 = cable_projection(point, first, last)
    zero, one = type(along)(0.0), type(along)(1.0)
    t = wp.clamp(along / length2, zero, one)
    offset = point - first - t * (last - first)
    # Exact endpoint arithmetic makes ownership ties independent of segment orientation.
    if t == one:
        offset = point - last
    return t, offset


@wp.kernel
def freeze_patches(
    tables: Any,
    start: wp.array[Any],
    pairs: wp.array[wp.vec4i],
    count: wp.array[int],
    stiffness: Any,
    smoothing: Any,
    flags: wp.array[int],
    normal_load: wp.array[Any],
    bary: wp.array[Any],
):
    """Assign one shear contact to each connected, loaded arc seen by a sample.

    Method A' sums segment loads and subtracts each shared vertex load. Walking
    outwards adds only nonnegative segment-minus-vertex increments. The closest
    feature owns the net load; disconnected arcs (including a second strand of
    the same cable) retain separate contact frames.
    """
    i = wp.tid()
    if i >= count[0] or i >= pairs.shape[0] or flags[0] != 0:
        return
    zero = type(stiffness)(0.0)
    normal_load[i] = normal_load.dtype(zero)
    sample, seg, kind = pairs[i][0], pairs[i][1], pairs[i][2]
    if kind != 0:
        return
    e = tables.sample_edge[sample]
    a, b = tables.seg_a[e], tables.seg_b[e]
    p = tables.sample_xi[sample] * (start[b] - start[a])
    radii = tables.seg_radius[e] + tables.seg_radius[seg]
    t, delta = _closest(tables, start, a, p, seg)
    distance = wp.length(delta)
    if not (distance > zero) or not wp.isfinite(distance):
        wp.atomic_max(flags, 0, 2)
        return
    _, load, _ = _penalty64(distance - radii, stiffness, smoothing)
    if load <= zero:
        return
    for direction in range(2):
        current = int(seg)
        while True:
            neighbor = tables.seg_next[current] + 0
            joint = tables.seg_b[current] + 0
            if direction == 0:
                neighbor = tables.seg_prev[current] + 0
                joint = tables.seg_a[current] + 0
            if neighbor < 0:
                break
            if tables.seg_cable[e] == tables.seg_cable[seg] and neighbor < tables.sample_min_self[sample]:
                break
            vertex_distance = wp.length(p - (start[joint] - start[a]))
            _, vertex_load, _ = _penalty64(vertex_distance - radii, stiffness, smoothing)
            if vertex_load <= zero:
                break
            _, other = _closest(tables, start, a, p, neighbor)
            other_distance = wp.length(other)
            if other_distance < distance or (other_distance == distance and neighbor < seg):
                return
            _, neighbor_load, _ = _penalty64(other_distance - radii, stiffness, smoothing)
            load += wp.max(zero, neighbor_load - vertex_load)
            current = neighbor
    bary[i] = t
    normal_load[i] = (delta / distance) * (load * tables.sample_weight[sample])


@wp.kernel
def add_friction(
    tables: Any,
    steps: wp.array[Any],
    rows: wp.array[int],
    pairs: wp.array[wp.vec4i],
    normal_load: wp.array[Any],
    bary: wp.array[Any],
    mu: Any,
    falloff: Any,
    candidates: Any,
    flags: wp.array[int],
):
    i = wp.tid()
    if i >= candidates.count[0] or flags[0] != 0:
        return
    load_vector = normal_load[i]
    load = wp.length(load_vector)
    zero, one = type(mu)(0.0), type(mu)(1.0)
    if load == zero:
        return
    n = load_vector / load
    sample, target = pairs[i][0], pairs[i][1]
    e = tables.sample_edge[sample]
    a, b, c, d = tables.seg_a[e], tables.seg_b[e], tables.seg_a[target], tables.seg_b[target]
    xi, t = tables.sample_xi[sample], bary[i]
    slip = (steps[a] - steps[c]) + xi * (steps[b] - steps[a]) - t * (steps[d] - steps[c])
    across = wp.cross(n, slip)
    rho = wp.length(across)
    axis = type(n)(one, zero, zero)
    if wp.abs(n[0]) > type(mu)(0.8):
        axis = type(n)(zero, one, zero)
    v = wp.normalize(wp.cross(n, axis))
    if rho > zero:
        v = across / rho
    u = wp.cross(v, n)
    # Same cubic regularization as rigid/cloth contact, with a fixed normal
    # plane, material coordinates and load throughout this step.
    cap = mu * load
    slope, radial, transverse = cap, zero, zero
    psi = rho - falloff / type(mu)(3.0)
    if rho < falloff:
        r = rho / falloff
        slope = cap * r * (type(mu)(2.0) - r)
        radial = (cap / falloff) * type(mu)(2.0) * (one - r)
        transverse = (cap / falloff) * (type(mu)(2.0) - r)
        psi = rho * r * (one - r / type(mu)(3.0))
    else:
        transverse = cap / rho
    gradient = slope * u
    candidates.energy[i, 1] = cap * psi
    candidates.energy_magnitude[i, 1] = cap * psi
    candidates.weights[i, 1] = radial
    candidates.weights[i, 2] = transverse
    support = int(0)
    particles = wp.vec4i(a, b, c, d)
    for slot in range(4):
        weight = one - xi
        if slot == 1:
            weight = xi
        elif slot == 2:
            weight = t - one
        elif slot == 3:
            weight = -t
        for axis_index in range(3):
            g = weight * gradient[axis_index]
            candidates.forces[i, 3 * slot + axis_index] -= g
            if rows[particles[slot]] >= 0:
                candidates.residual[i, support] += g
                candidates.magnitude[i, support] += wp.abs(g)
                candidates.factors[i, 1, support] = weight * u[axis_index]
                candidates.factors[i, 2, support] = weight * v[axis_index]
                support += 1
