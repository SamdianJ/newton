# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frictionless cable self and inter-cable contact (method A') inside the monolithic transaction.

Gauss samples of each edge, plus one sample at each free end, are tested against
the segments and interior vertices of other cable parts (``cable_contact_geometry``).
Candidates are found by discrete detection at the Newton refresh, from the current
and the step-start positions, and stay frozen through the trials of that iterate.
A sample-segment pair carries four vertices and a non-negative Gauss-Newton term;
a sample-vertex pair subtracts the double count at a shared vertex and carries
energy and gradient only. Both enter one ``_ContactPool``; the response acts on
centerline positions only and adds nothing to the cable twist rows.

Each evaluation also writes an 8-value summary, inter-cable then self contact, each as
[sample-segment candidates, active sample-segment pairs, compressive load [N], deepest overlap [m]].
The load sums every pair's normal load with its sign (a sample-vertex pair subtracts its double
count) in candidate-identity order; the overlap is the largest ``radius sum - distance`` of a
sample-segment pair, at least zero. A failed evaluation leaves the summary at zero.
"""

import json
from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .cable_contact_geometry import (
    GAUSS_RULES,
    MAX_EDGE_OVER_RADIUS,
    RULES,
    build_tables,
    cable_segment_response,
    cable_vertex_response,
)
from .cloth_contact_point_cloud import _bucket, _cell, _finish_query
from .contact_pipeline import _copy_pool_triplets, _reserve_pool_triplets
from .contact_pool import _ContactPool, _ContactRows
from .segment_graph import _SegmentGraphCache

#: Longest current segment over its rest length that the candidate grid covers; a longer one fails the refresh.
MAX_STRETCH = 1.5
#: Default line stiffness [N/m^2]; PR-02b gives the basis (static overlap of a ten-layer stack << 0.2 r).
DEFAULT_LINE_STIFFNESS = 1.0e5
_ALLOWED = {
    "self",
    "inter",
    "normal_stiffness",
    "index_exclusion",
    "arc_exclusion_factor",
    "samples_per_edge",
    "candidate_capacity",
    "support_capacity",
    "candidate_padding",
}
# Query flags; evaluation adds the response statuses 1 (nonfinite) and 2 (degenerate segment or sample on
# the target). The problem's checks report any of them as invalid cable contact geometry.
_FLAG_COORDINATE = 3
_FLAG_STRETCH = 4


@cache
def _table_type(dtype):
    @wp.struct
    class Tables:
        # Segment j joins particles seg_a[j] -> seg_b[j]; prev/next are the neighbours in its cable or -1.
        seg_a: wp.array[int]
        seg_b: wp.array[int]
        seg_cable: wp.array[int]
        seg_prev: wp.array[int]
        seg_next: wp.array[int]
        seg_radius: wp.array[dtype]
        seg_limit: wp.array[dtype]
        # A sample lies on segment sample_edge at parameter sample_xi; in its own cable it may only
        # target segment indices >= sample_min_self (precomputed in FP64; equal to the segment count
        # when nothing is allowed).
        sample_edge: wp.array[int]
        sample_cable: wp.array[int]
        sample_min_self: wp.array[int]
        sample_xi: wp.array[dtype]
        sample_weight: wp.array[dtype]

    return Tables


@wp.func
def _position(x: wp.array[Any], start: wp.array[Any], from_start: bool, particle: int):
    return wp.where(from_start, start[particle], x[particle])


@wp.func
def _sample_point(tables: Any, x: wp.array[Any], start: wp.array[Any], from_start: bool, sample: int):
    edge = tables.sample_edge[sample]
    xi = tables.sample_xi[sample]
    a = _position(x, start, from_start, tables.seg_a[edge])
    b = _position(x, start, from_start, tables.seg_b[edge])
    return (type(xi)(1.0) - xi) * a + xi * b


@wp.func
def _middle(tables: Any, x: wp.array[Any], start: wp.array[Any], from_start: bool, segment: int):
    a = _position(x, start, from_start, tables.seg_a[segment])
    b = _position(x, start, from_start, tables.seg_b[segment])
    return type(a[0])(0.5) * (a + b)


@wp.func
def _allowed(tables: Any, sample: int, segment: int, inter: bool):
    cable, other = tables.sample_cable[sample], tables.seg_cable[segment]
    if cable == other:
        return segment >= tables.sample_min_self[sample]
    return inter and other > cable


@wp.func
def _near(
    tables: Any, x: wp.array[Any], start: wp.array[Any], from_start: bool, sample: int, segment: int, margin: Any
):
    # The point-to-segment distance is at least |p - middle| - half length.
    a = _position(x, start, from_start, tables.seg_a[segment])
    b = _position(x, start, from_start, tables.seg_b[segment])
    reach = type(margin)(0.5) * wp.length(b - a) + tables.seg_radius[tables.sample_edge[sample]]
    reach = reach + tables.seg_radius[segment] + margin
    point = _sample_point(tables, x, start, from_start, sample)
    return wp.length(point - _middle(tables, x, start, from_start, segment)) <= reach


@wp.func
def _candidate(
    tables: Any, x: wp.array[Any], start: wp.array[Any], sample: int, segment: int, inter: bool, margin: Any
):
    """The candidate predicate both queries enumerate: allowed and near at the current or step-start positions."""
    if segment < 0:
        return False
    if not _allowed(tables, sample, segment, inter):
        return False
    near = _near(tables, x, start, False, sample, segment, margin)
    return near or _near(tables, x, start, True, sample, segment, margin)


@wp.kernel
def _check_segments(tables: Any, x: wp.array[Any], width: Any, flags: wp.array[int]):
    j = wp.tid()
    a, b = x[tables.seg_a[j]], x[tables.seg_b[j]]
    for axis in range(3):
        for point in range(2):
            value = wp.where(point == 0, a[axis], b[axis])
            if not wp.isfinite(value) or wp.abs(value / width) >= type(width)(536870912.0):
                wp.atomic_max(flags, 0, _FLAG_COORDINATE)
    if not (wp.length(b - a) <= tables.seg_limit[j]):
        wp.atomic_max(flags, 0, _FLAG_STRETCH)


@wp.kernel
def _build_segment_grid(
    tables: Any, x: wp.array[Any], width: Any, heads: wp.array[int], links: wp.array[int], flags: wp.array[int]
):
    if flags[0] != 0:
        return
    j = wp.tid()
    middle = _middle(tables, x, x, False, j)
    links[j] = wp.atomic_exch(heads, _bucket(_cell(middle, width), heads.shape[0] - 1), j)


@wp.func
def _emit(pairs: wp.array[wp.vec4i], count: wp.array[wp.uint64], pair: wp.vec4i):
    index = wp.atomic_add(count, 0, wp.uint64(1))
    if index < wp.uint64(pairs.shape[0]):
        pairs[int(index)] = pair


@wp.kernel
def _query_samples(
    tables: Any,
    x: wp.array[Any],
    start: wp.array[Any],
    heads: wp.array[int],
    links: wp.array[int],
    width: Any,
    margin: Any,
    inter: bool,
    from_start: bool,
    pairs: wp.array[wp.vec4i],
    count: wp.array[wp.uint64],
    flags: wp.array[int],
):
    if flags[0] != 0:
        return
    sample = wp.tid()
    cell = _cell(_sample_point(tables, x, start, from_start, sample), width)
    # One dynamic loop over the 27 neighbour cells: unrolled nested loops made the CUDA kernel too large to compile.
    for neighbor_index in range(27):
        neighbor = cell + wp.vec3i(neighbor_index // 9 - 1, (neighbor_index // 3) % 3 - 1, neighbor_index % 3 - 1)
        segment = heads[_bucket(neighbor, heads.shape[0] - 1)] + 0
        while segment >= 0:
            middle = _middle(tables, x, start, from_start, segment)
            if _cell(middle, width) == neighbor and _allowed(tables, sample, segment, inter):
                near = _near(tables, x, start, from_start, sample, segment, margin)
                # The step-start grid contributes only pairs the current grid misses.
                if near and not (from_start and _near(tables, x, start, False, sample, segment, margin)):
                    previous, following = tables.seg_prev[segment] + 0, tables.seg_next[segment] + 0
                    start_vertex = _candidate(tables, x, start, sample, previous, inter, margin)
                    end_vertex = _candidate(tables, x, start, sample, following, inter, margin)
                    bits = wp.where(start_vertex, 1, 0) + wp.where(end_vertex, 2, 0)
                    _emit(pairs, count, wp.vec4i(sample, segment, 0, bits))
                    # A vertex pair exists where both of its segments are candidates; the
                    # incoming segment emits it, so it is emitted once.
                    if end_vertex:
                        _emit(pairs, count, wp.vec4i(sample, segment, 1, 0))
            segment = links[segment] + 0


@wp.func
def _relative(start: wp.array[Any], steps: wp.array[Any], particle: int, anchor: int):
    # Anchor-relative position: the step-start offset plus the exact step increments.
    return (start[particle] - start[anchor]) + (steps[particle] - steps[anchor])


@wp.func
def _offset_size(start: wp.array[Any], steps: wp.array[Any], particle: int, anchor: int):
    return wp.length(start[particle] - start[anchor]) + wp.length(steps[particle] - steps[anchor])


@wp.func
def _write_slot(
    candidates: Any,
    i: int,
    slot: int,
    support: int,
    particle: int,
    row: int,
    gradient: Any,
    scale: Any,
    normal: Any,
):
    zero = type(scale)(0.0)
    for axis in range(3):
        candidates.force_rows[i, 3 * slot + axis] = 3 * particle + axis
        candidates.forces[i, 3 * slot + axis] = -gradient[axis]
    if row < 0:
        return support
    for axis in range(3):
        candidates.columns[i, support + axis] = row + axis
        candidates.residual[i, support + axis] = gradient[axis]
        candidates.magnitude[i, support + axis] = wp.abs(gradient[axis])
        candidates.factors[i, 0, support + axis] = scale * normal[axis]
        candidates.factors[i, 1, support + axis] = zero
        candidates.factors[i, 2, support + axis] = zero
    return support + 3


@wp.kernel
def _respond(
    tables: Any,
    start: wp.array[Any],
    steps: wp.array[Any],
    rows: wp.array[int],
    pairs: wp.array[wp.vec4i],
    stiffness: Any,
    smoothing: Any,
    candidates: Any,
    flags: wp.array[int],
    counts: wp.array[int],
    overlaps: wp.array[Any],
    load_rows: wp.array[int],
    loads: wp.array[Any],
):
    i = wp.tid()
    if i >= candidates.count[0] or flags[0] != 0:
        return
    pair = pairs[i]
    sample, target, kind, bits = pair[0], pair[1], pair[2], pair[3]
    edge = tables.sample_edge[sample]
    xi, weight = tables.sample_xi[sample], tables.sample_weight[sample]
    zero, one = type(xi)(0.0), type(xi)(1.0)
    a, b = tables.seg_a[edge], tables.seg_b[edge]
    c, d = tables.seg_a[target], tables.seg_b[target]
    # Every pair of a sample uses the same anchor, so the pairs that share a vertex
    # decide its clamp from identical inputs.
    xa = _relative(start, steps, a, a)
    xb = _relative(start, steps, b, a)
    radius_sum = tables.seg_radius[edge] + tables.seg_radius[target]
    rounding = wp.max(_offset_size(start, steps, b, a), _offset_size(start, steps, d, a))
    status = int(0)
    energy, coefficient = zero, zero
    ga, gb, g0, g1 = type(xa)(zero), type(xa)(zero), type(xa)(zero), type(xa)(zero)
    normal = type(xa)(zero)
    c0, c1 = zero, zero
    if kind == 0:
        y0 = _relative(start, steps, c, a)
        y1 = _relative(start, steps, d, a)
        rounding = wp.max(rounding, _offset_size(start, steps, c, a))
        previous = y0
        start_vertex = bits % 2
        if start_vertex != 0:
            previous = _relative(start, steps, tables.seg_a[tables.seg_prev[target]], a)
        status, energy, ga, gb, g0, g1, coefficient, normal, t = cable_segment_response(
            xa, xb, previous, y0, y1, xi, weight, radius_sum, stiffness, smoothing, start_vertex, bits / 2
        )
        c0, c1 = -(one - t), -t
    else:
        # The vertex between this incoming segment and the next one.
        status, energy, ga, gb, g1, normal = cable_vertex_response(
            xa, xb, _relative(start, steps, d, a), xi, weight, radius_sum, stiffness, smoothing
        )
    if status != 0:
        wp.atomic_max(flags, 0, status)
        return
    support = _write_slot(candidates, i, 0, 0, a, rows[a], ga, (one - xi), normal)
    support = _write_slot(candidates, i, 1, support, b, rows[b], gb, xi, normal)
    force_count = int(9)
    if kind == 0:
        support = _write_slot(candidates, i, 2, support, c, rows[c], g0, c0, normal)
        support = _write_slot(candidates, i, 3, support, d, rows[d], g1, c1, normal)
        force_count = 12
    else:
        support = _write_slot(candidates, i, 2, support, d, rows[d], g1, zero, normal)
    segments = wp.uint64(tables.seg_a.shape[0])
    candidates.identities[i] = wp.uint64(sample) * (wp.uint64(2) * segments) + wp.uint64(target)
    if kind != 0:
        candidates.identities[i] = candidates.identities[i] + segments
    candidates.support_counts[i] = support
    candidates.force_counts[i] = force_count
    candidates.energy[i, 0] = energy
    candidates.energy[i, 1] = zero
    candidates.energy_magnitude[i, 0] = wp.abs(energy)
    candidates.energy_magnitude[i, 1] = zero
    candidates.weights[i, 0] = coefficient
    candidates.weights[i, 1] = zero
    candidates.weights[i, 2] = zero
    candidates.rounding[i] = rounding + radius_sum
    # Diagnostics: channel 0 between cables, 1 within one cable. The gradient on the sample is
    # ga + gb = -load n, with the compressive load non-negative for a segment pair and the
    # subtracted double count (non-positive) for a vertex pair.
    channel = wp.where(tables.sample_cable[sample] == tables.seg_cable[target], 1, 0)
    load = -wp.dot(ga + gb, normal)
    load_rows[i] = channel
    loads[i] = load
    if kind == 0:
        wp.atomic_add(counts, 2 * channel, 1)
        if load > zero:
            wp.atomic_add(counts, 2 * channel + 1, 1)
        point = (one - xi) * xa + xi * xb
        first = _relative(start, steps, c, a)
        # The closest point y0 + t (y1 - y0), with c1 = -t.
        overlap = radius_sum - wp.length(point - (first - c1 * (_relative(start, steps, d, a) - first)))
        wp.atomic_max(overlaps, channel, wp.max(overlap, zero))


@wp.kernel
def _summary_count(flags: wp.array[int], count: wp.array[int], load_count: wp.array[int]):
    # A failed evaluation leaves stale slots; reduce none of them.
    load_count[0] = wp.where(flags[0] == 0, count[0], 0)


@wp.kernel
def _finish_summary(
    flags: wp.array[int],
    counts: wp.array[int],
    loads: wp.array[Any],
    overlaps: wp.array[Any],
    summary: wp.array[Any],
):
    # A failed evaluation is never published; leave no partial values behind.
    valid = flags[0] == 0
    zero = summary.dtype(0.0)
    for channel in range(2):
        summary[4 * channel] = wp.where(valid, summary.dtype(counts[2 * channel]), zero)
        summary[4 * channel + 1] = wp.where(valid, summary.dtype(counts[2 * channel + 1]), zero)
        summary[4 * channel + 2] = wp.where(valid, loads[channel], zero)
        summary[4 * channel + 3] = wp.where(valid, overlaps[channel], zero)


def _positive(value, name, numpy_dtype, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Cable contact {name} must be a number, got {value!r}")
    native = numpy_dtype(value)
    if not np.isfinite(native) or native < 0 or (native == 0 and not allow_zero):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"Cable contact {name} must be finite and {bound} in native precision, got {value!r}")
    return float(value)


def _integer(value, name, minimum, maximum=2**30 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Cable contact {name} must be an integer in [{minimum}, {maximum}], got {value!r}")
    return value


def self_bounds(segments, samples, *, index_span, arc_factor):
    """Lowest segment index each sample may target in its own cable (the segment count when none), in FP64.

    Inside a cable a sample on edge ``i`` targets segment ``j`` when ``j - i > index_span`` and the rest
    arc gap ``arc0_j - arc_sample`` (``j > i``) reaches ``arc_factor * 2 r``; both grow with ``j``.
    """
    count = len(segments["first"])
    bounds = np.full(len(samples["edge"]), count, dtype=np.int32)
    for cable in np.unique(segments["cable"]):
        own = np.flatnonzero(segments["cable"] == cable)
        reach = arc_factor * 2.0 * segments["radius"][own[0]]
        for sample in np.flatnonzero(segments["cable"][samples["edge"]] == cable):
            i = segments["index"][samples["edge"][sample]]
            gap = segments["arc0"][own] - samples["arc"][sample]
            keep = (segments["index"][own] > i) & (segments["index"][own] - i > index_span) & (gap >= reach)
            if keep.any():
                bounds[sample] = own[int(np.argmax(keep))]
    return bounds


class _CableContact:
    """Bounded sample-polyline cable contacts evaluated inside the monolithic transaction."""

    #: Frictionless (v1): the contact energy is a potential.
    mu = 0.0

    def __init__(self, solver, config, *, smoothing, use_cuda_graph):
        if not isinstance(config, dict):
            raise ValueError("Cable contact configuration must be a dict")
        unknown = sorted(set(config) - _ALLOWED)
        if unknown:
            raise ValueError(f"Unknown cable contact fields: {unknown}")
        self.model, self.dtype = solver._contact.model, solver._dtype
        self.device = self.model.device
        self.vector = wp.vec3d if self.dtype == wp.float64 else wp.vec3
        numpy_dtype = np.float64 if self.dtype == wp.float64 else np.float32
        materials = solver._materials
        cables = [] if materials is None else list(materials.cable)
        if not cables:
            raise ValueError("Cable contact requires cable_inputs")
        self.self_enabled, self.inter_enabled = config.get("self", True), config.get("inter", True)
        if type(self.self_enabled) is not bool or type(self.inter_enabled) is not bool:
            raise ValueError("Cable contact switches self and inter must be bool")
        if not (self.self_enabled or self.inter_enabled):
            raise ValueError("Cable contact needs self or inter enabled")
        self.stiffness = _positive(
            config.get("normal_stiffness", DEFAULT_LINE_STIFFNESS), "normal_stiffness", numpy_dtype
        )
        self.index_exclusion = _integer(config.get("index_exclusion", 3), "index_exclusion", 3)
        self.arc_factor = _positive(config.get("arc_exclusion_factor", np.pi / 2), "arc_exclusion_factor", numpy_dtype)
        self.samples_per_edge = config.get("samples_per_edge", 3)
        if type(self.samples_per_edge) is not int or self.samples_per_edge not in GAUSS_RULES:
            raise ValueError(f"Cable contact samples_per_edge must be one of {sorted(GAUSS_RULES)}")
        self.smoothing = float(smoothing)

        ranges, positions, particles, radii = [], [], [], []
        for obj in cables:
            ids = np.asarray(obj["particle_ids"], dtype=np.int64)
            ranges.append((len(particles), len(particles) + len(ids)))
            particles.extend(ids.tolist())
            positions.append(np.asarray(obj["positions"], dtype=np.float64))
            radii.append(float(obj["radius"]))
        segments, vertices, samples = build_tables(
            np.vstack(positions), ranges, radii, samples_per_edge=self.samples_per_edge, inter=self.inter_enabled
        )
        self.particles = np.asarray(particles, dtype=np.int32)
        self.segment_count, self.sample_count = len(segments["first"]), len(samples["edge"])
        self.padding = _positive(
            config.get("candidate_padding", max(radii)), "candidate_padding", numpy_dtype, allow_zero=True
        )
        capacity = config.get("candidate_capacity", min(1 << 20, max(64, 16 * self.sample_count)))
        capacity = _integer(capacity, "candidate_capacity", 0)
        support_capacity = _integer(config.get("support_capacity", 12 * capacity), "support_capacity", 0)

        same = segments["cable"][1:] == segments["cable"][:-1]
        previous = np.full(self.segment_count, -1, dtype=np.int32)
        following = np.full(self.segment_count, -1, dtype=np.int32)
        previous[1:][same] = np.flatnonzero(same)
        following[:-1][same] = np.flatnonzero(same) + 1
        bounds = (
            self_bounds(segments, samples, index_span=self.index_exclusion, arc_factor=self.arc_factor)
            if self.self_enabled
            else np.full(self.sample_count, self.segment_count, dtype=np.int32)
        )
        limit = MAX_STRETCH * segments["rest_length"]
        # Every segment a sample can reach has its middle within one cell of the sample.
        self.width = 0.5 * float(limit.max()) + 2.0 * max(radii) + self.smoothing + self.padding
        self.margin = self.smoothing + self.padding

        def array(data, dtype):
            return wp.array(np.asarray(data), dtype=dtype, device=self.device)

        tables = _table_type(self.dtype)()
        tables.seg_a = array(self.particles[segments["first"]], int)
        tables.seg_b = array(self.particles[segments["first"] + 1], int)
        tables.seg_cable = array(segments["cable"], int)
        tables.seg_prev = array(previous, int)
        tables.seg_next = array(following, int)
        tables.seg_radius = array(segments["radius"], self.dtype)
        tables.seg_limit = array(limit, self.dtype)
        tables.sample_edge = array(samples["edge"], int)
        tables.sample_cable = array(segments["cable"][samples["edge"]], int)
        tables.sample_min_self = array(bounds, int)
        tables.sample_xi = array(samples["xi"], self.dtype)
        tables.sample_weight = array(samples["weight"], self.dtype)
        self.tables = tables
        self.host_tables = {"segments": segments, "vertices": vertices, "samples": samples, "self_bounds": bounds}
        self.rows = array(solver._actors.maps["particle_to_global"], int)
        self.start = wp.zeros(self.model.particle_count, dtype=self.vector, device=self.device)
        self.pairs = wp.zeros(capacity, dtype=wp.vec4i, device=self.device)
        self._query_count = wp.zeros(1, dtype=wp.uint64, device=self.device)
        buckets = 1 << max(1, (2 * self.segment_count - 1).bit_length())
        self.heads = wp.zeros(buckets, dtype=int, device=self.device)
        self.links = wp.zeros(self.segment_count, dtype=int, device=self.device)
        self.pool = _ContactPool(
            solver._layout.scalar_dof_count,
            3 * self.model.particle_count,
            capacity,
            support_capacity,
            12,
            max(1, 2 * self.sample_count * self.segment_count),
            self.dtype,
            self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.flags = wp.zeros(1, dtype=int, device=self.device)
        self.query_flags = wp.zeros(1, dtype=int, device=self.device)
        self._append_base = wp.zeros(1, dtype=int, device=self.device)
        # Diagnostics (see the module docstring): per-channel counts and overlaps, and per-slot
        # signed loads summed by candidate identity, so the sum does not depend on slot order.
        self._summary_counts = wp.zeros(4, dtype=int, device=self.device)
        self._overlaps = wp.zeros(2, dtype=self.dtype, device=self.device)
        self._load_rows = wp.zeros(capacity, dtype=int, device=self.device)
        self._load_locals = wp.zeros(capacity, dtype=int, device=self.device)
        self._loads = wp.zeros(capacity, dtype=self.dtype, device=self.device)
        self._load_count = wp.zeros(1, dtype=int, device=self.device)
        self._load_sums = _ContactRows(
            2, capacity, max(1, 2 * self.sample_count * self.segment_count), 1, self.dtype, self.device
        )
        self._load_sums._validate_inputs(
            self._load_rows, self.pool.candidates.identities, self._load_locals, self._loads, self._load_count
        )
        self.summary_flags = self._load_sums.flags
        self.summary = wp.zeros(8, dtype=self.dtype, device=self.device)
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)
        self.evaluation_generation = 0
        self.assembled_generation = -1
        self.candidate_generation = 0
        self._candidate_count = None
        self._last_assembly = None
        for rows in (self.pool._identity, self.pool.energy, self.pool.residual, self.pool.force):
            for buffer in rows.buffers():
                buffer.zero_()

        self.config = {
            "self": self.self_enabled,
            "inter": self.inter_enabled,
            "normal_stiffness": self.stiffness,
            "index_exclusion": self.index_exclusion,
            "arc_exclusion_factor": self.arc_factor,
            "samples_per_edge": self.samples_per_edge,
            "candidate_capacity": capacity,
            "support_capacity": support_capacity,
            "candidate_padding": self.padding,
            "rules": dict(RULES),
            "exclusion": "same cable: index gap > index_exclusion and rest arc gap >= arc_exclusion_factor * 2r",
            "max_edge_over_radius": MAX_EDGE_OVER_RADIUS,
            "max_stretch": MAX_STRETCH,
            "coverage": "newton-frozen-start-and-current-DCD",
            "friction": "none",
            "dofs": "centerline-position",
        }
        self.identity = np.frombuffer(json.dumps(self.config, sort_keys=True).encode(), dtype=np.uint8)

    @property
    def energy(self):
        return self.pool.energy.result[:2]

    @property
    def absolute_energy(self):
        return self.pool.energy.result[2:]

    def _table_buffers(self):
        t = self.tables
        return (
            t.seg_a,
            t.seg_b,
            t.seg_cable,
            t.seg_prev,
            t.seg_next,
            t.seg_radius,
            t.seg_limit,
            t.sample_edge,
            t.sample_cable,
            t.sample_min_self,
            t.sample_xi,
            t.sample_weight,
        )

    def begin_step(self, state):
        wp.copy(self.start, state.particle_q)
        self.refresh(state.particle_q)

    def refresh(self, x):
        metadata = (
            *self._table_buffers(),
            self.start,
            self.heads,
            self.links,
            self.pairs,
            self._query_count,
            self.query_flags,
            self.pool.candidates.count,
            self.pool.candidates.overflow,
        )
        self.graphs.run("cable_contact_query", (), (x, *metadata), lambda: self._refresh(x), warmup=False)
        # Freeze a conservative work bound with the candidate set, once per Newton refresh.
        count = int(self.pool.candidates.count.numpy()[0])
        self._candidate_count = max(0, min(self.pool.candidate_capacity, count))
        self.candidate_generation += 1
        self.assembled_generation = -1

    def _refresh(self, x):
        self._query_count.zero_()
        self.query_flags.zero_()
        width = self.dtype(self.width)
        for source in (x, self.start):
            wp.launch(
                _check_segments, self.segment_count, [self.tables, source, width, self.query_flags], device=self.device
            )
        for from_start, source in ((False, x), (True, self.start)):
            self.heads.fill_(-1)
            wp.launch(
                _build_segment_grid,
                self.segment_count,
                [self.tables, source, width, self.heads, self.links, self.query_flags],
                device=self.device,
            )
            wp.launch(
                _query_samples,
                self.sample_count,
                [
                    self.tables,
                    x,
                    self.start,
                    self.heads,
                    self.links,
                    width,
                    self.dtype(self.margin),
                    self.inter_enabled,
                    from_start,
                    self.pairs,
                    self._query_count,
                    self.query_flags,
                ],
                device=self.device,
            )
        c = self.pool.candidates
        wp.launch(_finish_query, 1, [self._query_count, c.count, c.overflow, self.pairs.size], device=self.device)

    def evaluate(self, steps, dt):
        """Write every frozen candidate's local response into the pool; ``dt`` is unused (no friction)."""
        buffers = (
            *self._table_buffers(),
            self.start,
            steps,
            self.rows,
            self.pairs,
            self.flags,
            self.query_flags,
            self._summary_counts,
            self._overlaps,
            self._load_rows,
            self._loads,
            self._load_count,
            self.summary,
            *self._load_sums.buffers(),
            *self.pool.buffers(),
        )
        count, capacity = self._candidate_count, self.pairs.size
        span = capacity if count is None else min(capacity, 1 << (max(1, count) - 1).bit_length()) if count else 0
        self.graphs.run("cable_contact_response", (span,), buffers, lambda: self._evaluate(steps, span), warmup=False)
        self.pool.evaluate(candidate_count=self._candidate_count)
        self.evaluation_generation += 1
        self.assembled_generation = -1

    def _evaluate(self, steps, span):
        wp.copy(self.flags, self.query_flags)
        self._summary_counts.zero_()
        self._overlaps.zero_()
        if span:
            wp.launch(
                _respond,
                span,
                [
                    self.tables,
                    self.start,
                    steps,
                    self.rows,
                    self.pairs,
                    self.dtype(self.stiffness),
                    self.dtype(self.smoothing),
                    self.pool.candidates,
                    self.flags,
                    self._summary_counts,
                    self._overlaps,
                    self._load_rows,
                    self._loads,
                ],
                device=self.device,
            )
        wp.launch(_summary_count, 1, [self.flags, self.pool.candidates.count, self._load_count], device=self.device)
        sums = self._load_sums
        sums._reduce(
            self._load_rows, self.pool.candidates.identities, self._load_locals, self._loads, self._load_count, span
        )
        wp.launch(
            _finish_summary,
            1,
            [self.flags, self._summary_counts, sums.result, self._overlaps, self.summary],
            device=self.device,
        )

    def assemble(self, assembly):
        if self._last_assembly == assembly.generation:
            raise ValueError("Cable contact COO may be appended only once per assembly")
        wp.launch(
            _reserve_pool_triplets,
            1,
            [self.pool.records, self.flags, assembly.global_scalar_triplets, self._append_base],
            device=self.device,
        )
        if self.pool.triplet_capacity:
            workers = min(self.pool.triplet_capacity, 65536)
            wp.launch(
                _copy_pool_triplets,
                workers,
                [self.pool.records, assembly.global_scalar_triplets, self._append_base, workers],
                device=self.device,
            )
        self.assembled_generation = self.evaluation_generation
        self._last_assembly = assembly.generation
