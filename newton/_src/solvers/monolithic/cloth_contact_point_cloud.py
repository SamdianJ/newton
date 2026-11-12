# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Weighted shell point contacts with native C-IE response and four-node support."""

import json
from typing import Any

import numpy as np
import warp as wp

from .contact import _penalty64
from .contact_pipeline import _copy_pool_triplets, _reserve_pool_triplets
from .contact_pool import _ContactPool, _ContactRows
from .segment_graph import _SegmentGraphCache


@wp.func
def _point_response(y: Any, slip: Any, radius: Any, stiffness: Any, smoothing: Any, mu: Any, falloff: Any):
    zero = type(radius)(0.0)
    n = type(y)(zero)
    matrix = wp.outer(n, n)
    length = wp.length(y)
    if length == zero or not wp.isfinite(length):
        return zero, zero, n, matrix, zero, zero, int(2)
    n = y / length
    gap = length - radius
    energy, load, curvature = _penalty64(gap, stiffness, smoothing)
    _unused, friction_load, _unused_curvature = _penalty64(gap - wp.dot(n, slip), stiffness, smoothing)
    if mu == zero:
        friction_load = zero
    projection = wp.diag(type(y)(type(radius)(1.0))) - wp.outer(n, n)
    v = slip - wp.dot(n, slip) * n
    rho = wp.length(v)
    psi, factor, radial = zero, zero, zero
    if friction_load > zero:
        if rho < falloff:
            psi = rho * rho / falloff - rho * rho * rho / (type(radius)(3.0) * falloff * falloff)
            factor = type(radius)(2.0) / falloff - rho / (falloff * falloff)
            radial = type(radius)(2.0) * mu * friction_load * (falloff - rho) / (falloff * falloff)
        else:
            psi = rho - falloff / type(radius)(3.0)
            factor = type(radius)(1.0) / rho
    beta = mu * friction_load * factor
    residual = -load * n + beta * v
    tangent = type(y)(zero)
    transverse = wp.cross(n, v)
    if wp.length(transverse) > zero:
        tangent = wp.cross(transverse / wp.length(transverse), n)
    tt = wp.outer(tangent, tangent)
    matrix = curvature * wp.outer(n, n) + beta * (projection - tt) + radial * tt
    dissipation = mu * friction_load * psi
    status = int(0)
    if not wp.isfinite(energy) or not wp.isfinite(dissipation) or not wp.isfinite(wp.length(residual)):
        status = 1
    return energy, dissipation, residual, matrix, load, friction_load, status


@wp.kernel
def _evaluate_local(
    separation: wp.array[Any],
    slip: wp.array[Any],
    thickness: wp.array2d[Any],
    areas: wp.array2d[Any],
    length: Any,
    stiffness: Any,
    smoothing: Any,
    mu: Any,
    falloff: Any,
    energy: wp.array2d[Any],
    residual: wp.array[Any],
    tangent: wp.array[Any],
    loads: wp.array2d[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    radius = length + (thickness[i, 0] + thickness[i, 1]) / type(length)(2.0)
    k = stiffness * areas[i, 0] * areas[i, 1] / (length * length)
    en, ed, r, h, normal_load, friction_load, status = _point_response(
        separation[i], slip[i], radius, k, smoothing, mu, falloff
    )
    energy[i, 0] = en
    energy[i, 1] = ed
    residual[i] = r
    tangent[i] = h
    loads[i, 0] = normal_load
    loads[i, 1] = friction_load
    flags[i] = status


@wp.func
def _bary(slot: int, axis: int, scalar: Any):
    return wp.where(slot == axis, type(scalar)(2.0) / type(scalar)(3.0), type(scalar)(1.0) / type(scalar)(6.0))


@wp.func
def _cell(point: Any, width: Any):
    return wp.vec3i(int(wp.floor(point[0] / width)), int(wp.floor(point[1] / width)), int(wp.floor(point[2] / width)))


@wp.func
def _bucket(cell: wp.vec3i, mask: int):
    value = wp.uint32(cell[0]) * wp.uint32(73856093)
    value = value ^ (wp.uint32(cell[1]) * wp.uint32(19349663))
    value = value ^ (wp.uint32(cell[2]) * wp.uint32(83492791))
    return int(value & wp.uint32(mask))


@wp.kernel
def _check_grid_coordinates(x: wp.array[Any], ids: wp.array[int], width: Any, flags: wp.array[int]):
    point = x[ids[wp.tid()]]
    for axis in range(3):
        if not wp.isfinite(point[axis]) or wp.abs(point[axis] / width) >= type(width)(536870912.0):
            wp.atomic_max(flags, 0, 2)


@wp.kernel
def _build_grid(
    x: wp.array[Any], ids: wp.array[int], width: Any, heads: wp.array[int], links: wp.array[int], flags: wp.array[int]
):
    if flags[0] != 0:
        return
    node = wp.tid()
    bucket = _bucket(_cell(x[ids[node]], width), heads.shape[0] - 1)
    links[node] = wp.atomic_exch(heads, bucket, node)


@wp.kernel
def _query_points(
    x: wp.array[Any],
    start: wp.array[Any],
    reference: wp.array[Any],
    triangles: wp.array[wp.vec3i],
    triangle_owners: wp.array[int],
    nodes: wp.array[int],
    node_owners: wp.array[int],
    thickness: wp.array[Any],
    worlds: wp.array[int],
    heads: wp.array[int],
    links: wp.array[int],
    width: Any,
    length: Any,
    padding: Any,
    smoothing: Any,
    self_exclusion_ratio: Any,
    inter_enabled: bool,
    self_enabled: bool,
    from_start: bool,
    pairs: wp.array[wp.vec2i],
    count: wp.array[wp.uint64],
    flags: wp.array[int],
):
    if flags[0] != 0:
        return
    sample = wp.tid()
    face, slot = sample // 3, sample % 3
    tri = triangles[face]
    point = type(x[0])(type(length)(0.0))
    point_current = type(x[0])(type(length)(0.0))
    point_reference = type(x[0])(type(length)(0.0))
    for j in range(3):
        b = _bary(slot, j, length)
        point += b * wp.where(from_start, start[tri[j]], x[tri[j]])
        point_current += b * x[tri[j]]
        point_reference += b * reference[tri[j]]
    cell = _cell(point, width)
    owner = triangle_owners[face]
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                neighbor = cell + wp.vec3i(dx, dy, dz)
                node = heads[_bucket(neighbor, heads.shape[0] - 1)] + 0
                while node >= 0:
                    particle = nodes[node] + 0
                    other = node_owners[node] + 0
                    target = wp.where(from_start, start[particle], x[particle])
                    same_cell = _cell(target, width) == neighbor
                    allowed = (owner != other and inter_enabled) or (owner == other and self_enabled)
                    allowed = allowed and (worlds[owner] == -1 or worlds[other] == -1 or worlds[owner] == worlds[other])
                    if owner == other and allowed:
                        # Match SuperDex's strict reference-distance rule with tau = h_bar + epsilon.
                        exclusion_radius = (
                            self_exclusion_ratio * length
                            + (thickness[owner] + thickness[other]) / type(length)(2.0)
                            + smoothing
                        )
                        delta_reference = point_reference - reference[particle]
                        if wp.dot(delta_reference, delta_reference) < exclusion_radius * exclusion_radius:
                            allowed = False
                    cutoff = length + (thickness[owner] + thickness[other]) / type(length)(2.0) + padding
                    inside = wp.length(point - target) <= cutoff
                    # The second grid contributes only pairs absent from the current query.
                    duplicate = from_start and wp.length(point_current - x[particle]) <= cutoff
                    if same_cell and allowed and inside and not duplicate:
                        index = wp.atomic_add(count, 0, wp.uint64(1))
                        if index < wp.uint64(pairs.shape[0]):
                            pairs[int(index)] = wp.vec2i(sample, node)
                    node = links[node] + 0


@wp.kernel
def _finish_query(total: wp.array[wp.uint64], count: wp.array[int], overflow: wp.array[int], capacity: int):
    count[0] = int(wp.min(total[0], wp.uint64(capacity)))
    overflow[0] = 0
    if total[0] > wp.uint64(capacity):
        overflow[0] = int(wp.min(total[0] - wp.uint64(capacity), wp.uint64(2147483647)))


@wp.kernel
def _point_candidates(
    start: wp.array[Any],
    steps: wp.array[Any],
    triangles: wp.array[wp.vec3i],
    triangle_owners: wp.array[int],
    sample_area: wp.array[Any],
    nodes: wp.array[int],
    node_owners: wp.array[int],
    node_area: wp.array[Any],
    thickness: wp.array[Any],
    row_map: wp.array[int],
    pairs: wp.array[wp.vec2i],
    length: Any,
    stiffness: Any,
    smoothing: Any,
    mu: Any,
    falloff: Any,
    candidates: Any,
    flags: wp.array[int],
    counts: wp.array[int],
    load_rows: wp.array[int],
    loads: wp.array[Any],
):
    i = wp.tid()
    if i >= candidates.count[0] or flags[0] != 0:
        return
    pair = pairs[i]
    face, slot, node = pair[0] // 3, pair[0] % 3, pair[1]
    tri = triangles[face]
    p = nodes[node] + 0
    owner = triangle_owners[face] + 0
    other = node_owners[node] + 0
    y0 = type(start[0])(type(length)(0.0))
    u = type(start[0])(type(length)(0.0))
    for j in range(3):
        b = _bary(slot, j, length)
        y0 += b * (start[tri[j]] - start[p])
        u += b * (steps[tri[j]] - steps[p])
    y = y0 + u
    radius = length + (thickness[owner] + thickness[other]) / type(length)(2.0)
    k = stiffness * sample_area[face] * node_area[node] / (length * length)
    en, ed, r, matrix, load, friction_load, status = _point_response(y, u, radius, k, smoothing, mu, falloff)
    if status != 0:
        wp.atomic_max(flags, 0, status)
        return
    n = y / wp.length(y)
    v = u - wp.dot(n, u) * n
    axis = type(n)(type(length)(1.0), type(length)(0.0), type(length)(0.0))
    if wp.abs(n[0]) > type(length)(0.8):
        axis = type(n)(type(length)(0.0), type(length)(1.0), type(length)(0.0))
    t = wp.normalize(wp.cross(n, axis))
    cross = wp.cross(n, v)
    if wp.length(cross) > type(length)(0.0):
        t = wp.cross(cross / wp.length(cross), n)
    across = wp.cross(n, t)
    candidates.identities[i] = wp.uint64(pair[0]) * wp.uint64(nodes.shape[0]) + wp.uint64(node)
    candidates.energy[i, 0] = en
    candidates.energy[i, 1] = ed
    candidates.energy_magnitude[i, 0] = wp.abs(en)
    candidates.energy_magnitude[i, 1] = wp.abs(ed)
    candidates.weights[i, 0] = wp.max(type(length)(0.0), wp.dot(n, matrix * n))
    candidates.weights[i, 1] = wp.max(type(length)(0.0), wp.dot(t, matrix * t))
    candidates.weights[i, 2] = wp.max(type(length)(0.0), wp.dot(across, matrix * across))
    candidates.rounding[i] = wp.length(y0) + wp.length(u) + radius
    support = int(0)
    for j in range(4):
        particle = p
        b = type(length)(-1.0)
        if j < 3:
            particle = tri[j]
            b = _bary(slot, j, length)
        for axis_index in range(3):
            candidates.force_rows[i, 3 * j + axis_index] = 3 * particle + axis_index
            candidates.forces[i, 3 * j + axis_index] = -b * r[axis_index]
            if row_map[particle] >= 0:
                candidates.columns[i, support] = row_map[particle] + axis_index
                candidates.residual[i, support] = b * r[axis_index]
                candidates.magnitude[i, support] = wp.abs(b) * (
                    wp.abs(load * n[axis_index]) + wp.abs(r[axis_index] + load * n[axis_index])
                )
                candidates.factors[i, 0, support] = b * n[axis_index]
                candidates.factors[i, 1, support] = b * t[axis_index]
                candidates.factors[i, 2, support] = b * across[axis_index]
                support += 1
    candidates.support_counts[i] = support
    candidates.force_counts[i] = 12
    channel = wp.where(owner == other, 1, 0)
    wp.atomic_add(counts, 2 * channel, 1)
    if load > type(length)(0.0) or friction_load > type(length)(0.0):
        wp.atomic_add(counts, 2 * channel + 1, 1)
    # Slots are reserved atomically; loads are summed later in identity order.
    load_rows[i] = channel
    loads[i] = load


@wp.kernel
def _summary_count(flags: wp.array[int], count: wp.array[int], load_count: wp.array[int]):
    # A failed evaluation leaves stale slots; reduce none of them.
    load_count[0] = wp.where(flags[0] == 0, count[0], 0)


@wp.kernel
def _finish_summary(flags: wp.array[int], counts: wp.array[int], loads: wp.array[Any], summary: wp.array[Any]):
    # A failed evaluation is never published; leave no partial counts behind.
    valid = flags[0] == 0
    for channel in range(2):
        summary[3 * channel] = wp.where(valid, summary.dtype(counts[2 * channel]), summary.dtype(0))
        summary[3 * channel + 1] = wp.where(valid, summary.dtype(counts[2 * channel + 1]), summary.dtype(0))
        summary[3 * channel + 2] = wp.where(valid, loads[channel], summary.dtype(0))


class _PointCloudContact:
    """Bounded point pairs evaluated inside the existing monolithic transaction."""

    def __init__(self, solver, config, *, mu, smoothing, velocity_scale, use_cuda_graph):
        allowed = {
            "inter",
            "self",
            "radius",
            "normal_stiffness",
            "candidate_capacity",
            "support_capacity",
            "candidate_padding",
            "self_exclusion_ratio",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("Invalid cloth_contact fields")
        self.model, self.dtype = solver._contact.model, solver._dtype
        self.device = self.model.device
        self.vector = wp.vec3d if self.dtype == wp.float64 else wp.vec3
        self.numpy_dtype = np.float64 if self.dtype == wp.float64 else np.float32
        self.length = float(config.get("radius", 0.01))
        self.stiffness = float(config.get("normal_stiffness", 1e5))
        self.padding = float(config.get("candidate_padding", self.length))
        if (
            any(not np.isfinite(v) or v <= 0 for v in (self.length, self.stiffness))
            or not np.isfinite(self.padding)
            or self.padding < 0
        ):
            raise ValueError("Cloth contact requires positive radius/stiffness and nonnegative candidate padding")
        for name, value in (("radius", self.length), ("normal_stiffness", self.stiffness)):
            native = self.numpy_dtype(value)
            if isinstance(config.get(name), (bool, np.bool_)) or not np.isfinite(native) or native <= 0:
                raise ValueError(f"Cloth contact {name} must be positive in native precision")
        self.inter_enabled, self.self_enabled = config.get("inter", True), config.get("self", True)
        if type(self.inter_enabled) is not bool or type(self.self_enabled) is not bool:
            raise ValueError("Cloth contact switches must be bool")
        self.self_exclusion_ratio = float(config.get("self_exclusion_ratio", 1.5))
        native_ratio = self.numpy_dtype(self.self_exclusion_ratio)
        if (
            isinstance(config.get("self_exclusion_ratio"), (bool, np.bool_))
            or not np.isfinite(native_ratio)
            or (self.self_enabled and native_ratio <= 1)
        ):
            raise ValueError("Cloth contact self_exclusion_ratio must be finite and > 1 when self is enabled")
        self.mu, self.smoothing, self.velocity_scale = mu, smoothing, velocity_scale
        materials = solver._materials
        if materials is None or not materials.cloth:
            raise ValueError("Cloth contact requires cloth_inputs")
        triangles, owners, areas, nodes, node_owners, node_areas = [], [], [], [], [], []
        thickness, worlds = [], []
        reference = np.zeros((self.model.particle_count, 3), dtype=self.numpy_dtype)
        for owner, obj in enumerate(materials.cloth):
            reference[obj["particle_ids"]] = obj["positions"]
            faces, reference_areas = obj["triangles"], obj["areas"]
            thickness.append(obj["thickness"])
            worlds.append(obj["world"])
            weights = {}
            for tri, area in zip(faces, reference_areas, strict=True):
                for particle in tri:
                    weights[int(particle)] = weights.get(int(particle), 0) + area / 3
            for tri, area in zip(faces, reference_areas, strict=True):
                triangles.append(tri)
                owners.append(owner)
                areas.append(area / 3)
            for particle, area in sorted(weights.items()):
                nodes.append(particle)
                node_owners.append(owner)
                node_areas.append(area)
        self.sample_count, self.node_count = 3 * len(triangles), len(nodes)
        capacity = config.get("candidate_capacity", min(65536, max(64, self.sample_count * 16)))
        support_capacity = config.get("support_capacity", 12 * capacity)
        for name, value in (("candidate_capacity", capacity), ("support_capacity", support_capacity)):
            if type(value) is not int or not 0 <= value < 2**30:
                raise ValueError(f"Invalid cloth contact {name}")
        self.config = {
            "inter": self.inter_enabled,
            "self": self.self_enabled,
            "radius": self.length,
            "normal_stiffness": self.stiffness,
            "candidate_padding": self.padding,
            "candidate_capacity": capacity,
            "support_capacity": support_capacity,
            "thickness_rule": "L+(hA+hB)/2",
            "quadrature": "P1Q3-node",
            "self_exclusion": "material-reference-distance",
            "self_exclusion_ratio": self.self_exclusion_ratio,
            "self_exclusion_radius": "alpha*L+(hA+hB)/2+normal_smoothing",
            "self_exclusion_comparison": "reference-distance-squared<radius-squared",
            "coverage": "newton-frozen-start-and-current-DCD",
        }
        self.identity = np.frombuffer(json.dumps(self.config, sort_keys=True).encode(), dtype=np.uint8)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.triangles = array(np.asarray(triangles), wp.vec3i)
        self.reference = array(reference, self.vector)
        self.triangle_owners = array(owners, int)
        self.sample_area = array(areas, self.dtype)
        self.nodes = array(nodes, int)
        self.node_owners = array(node_owners, int)
        self.node_area = array(node_areas, self.dtype)
        self.thickness = array(thickness, self.dtype)
        self.worlds = array(worlds, int)
        self.rows = array(solver._actors.maps["particle_to_global"], int)
        self.width = self.length + max(thickness) + self.smoothing + self.padding
        self.start = wp.zeros(self.model.particle_count, dtype=self.vector, device=self.device)
        self.pairs = wp.zeros(capacity, dtype=wp.vec2i, device=self.device)
        # Each sample/node identity is admitted at most once across both grids.
        # Match the pool's checked uint64 identity domain before narrowing counts.
        self._query_count = wp.zeros(1, dtype=wp.uint64, device=self.device)
        buckets = 1 << max(1, (2 * self.node_count - 1).bit_length())
        self.heads = wp.zeros(buckets, dtype=int, device=self.device)
        self.links = wp.zeros(self.node_count, dtype=int, device=self.device)
        identity_count = max(1, self.sample_count * self.node_count)
        self.pool = _ContactPool(
            solver._layout.scalar_dof_count,
            3 * self.model.particle_count,
            capacity,
            support_capacity,
            12,
            identity_count,
            self.dtype,
            self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.flags = wp.zeros(1, dtype=int, device=self.device)
        self.query_flags = wp.zeros(1, dtype=int, device=self.device)
        self.summary = wp.zeros(6, dtype=self.dtype, device=self.device)
        # Inter/self diagnostics: integer counts, and compressive loads reduced in
        # candidate identity order so the sums do not depend on slot order.
        self._summary_counts = wp.zeros(4, dtype=int, device=self.device)
        self._load_rows = wp.zeros(capacity, dtype=int, device=self.device)
        self._load_locals = wp.zeros(capacity, dtype=int, device=self.device)
        self._loads = wp.zeros(capacity, dtype=self.dtype, device=self.device)
        self._load_count = wp.zeros(1, dtype=int, device=self.device)
        # Reduced inside the response segment, like the contact pool's own rows.
        self._load_sums = _ContactRows(2, capacity, identity_count, 1, self.dtype, self.device)
        self._load_sums._validate_inputs(
            self._load_rows, self.pool.candidates.identities, self._load_locals, self._loads, self._load_count
        )
        self.summary_flags = self._load_sums.flags
        self._append_base = wp.zeros(1, dtype=int, device=self.device)
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)
        self.evaluation_generation = 0
        self.assembled_generation = -1
        self.candidate_generation = 0
        self._candidate_count = None
        self._last_assembly = None
        for rows in (self.pool._identity, self.pool.energy, self.pool.residual, self.pool.force):
            for buffer in rows.buffers():
                buffer.zero_()

    @property
    def energy(self):
        return self.pool.energy.result[:2]

    @property
    def absolute_energy(self):
        return self.pool.energy.result[2:]

    def begin_step(self, state):
        wp.copy(self.start, state.particle_q)
        self.refresh(state.particle_q)

    def refresh(self, x):
        metadata = (
            self.start,
            self.reference,
            self.triangles,
            self.triangle_owners,
            self.nodes,
            self.node_owners,
            self.thickness,
            self.worlds,
            self.heads,
            self.links,
            self.pairs,
            self._query_count,
            self.query_flags,
            self.pool.candidates.count,
            self.pool.candidates.overflow,
        )
        self.graphs.run("point_cloud_query", (), (x, *metadata), lambda: self._refresh(x), warmup=False)
        # Freeze a conservative work bound with the candidate set, once per
        # Newton refresh. Trial evaluations need no additional count readback.
        count = int(self.pool.candidates.count.numpy()[0])
        self._candidate_count = max(0, min(self.pool.candidate_capacity, count))
        self.candidate_generation += 1
        self.assembled_generation = -1

    def _refresh(self, x):
        c = self.pool.candidates
        self._query_count.zero_()
        self.query_flags.zero_()
        for source in (x, self.start):
            wp.launch(
                _check_grid_coordinates,
                self.node_count,
                [source, self.nodes, self.dtype(self.width), self.query_flags],
                device=self.device,
            )
        for from_start, source in ((False, x), (True, self.start)):
            self.heads.fill_(-1)
            wp.launch(
                _build_grid,
                self.node_count,
                [source, self.nodes, self.dtype(self.width), self.heads, self.links, self.query_flags],
                device=self.device,
            )
            wp.launch(
                _query_points,
                self.sample_count,
                [
                    x,
                    self.start,
                    self.reference,
                    self.triangles,
                    self.triangle_owners,
                    self.nodes,
                    self.node_owners,
                    self.thickness,
                    self.worlds,
                    self.heads,
                    self.links,
                    self.dtype(self.width),
                    self.dtype(self.length),
                    self.dtype(self.smoothing + self.padding),
                    self.dtype(self.smoothing),
                    self.dtype(self.self_exclusion_ratio),
                    self.inter_enabled,
                    self.self_enabled,
                    from_start,
                    self.pairs,
                    self._query_count,
                    self.query_flags,
                ],
                device=self.device,
            )
        wp.launch(_finish_query, 1, [self._query_count, c.count, c.overflow, self.pairs.size], device=self.device)

    def evaluate(self, steps, dt):
        buffers = (
            self.start,
            steps,
            self.triangles,
            self.triangle_owners,
            self.sample_area,
            self.nodes,
            self.node_owners,
            self.node_area,
            self.thickness,
            self.rows,
            self.pairs,
            self.flags,
            self.query_flags,
            self._summary_counts,
            self._load_rows,
            self._loads,
            self._load_count,
            self.summary,
            *self._load_sums.buffers(),
            *self.pool.buffers(),
        )
        # Bound the diagnostic sort by the frozen candidate count, as the pool does.
        count, capacity = self._candidate_count, self.pairs.size
        span = capacity if count is None else min(capacity, 1 << (max(1, count) - 1).bit_length()) if count else 0
        self.graphs.run(
            "point_cloud_response", (float(dt), span), buffers, lambda: self._evaluate(steps, dt, span), warmup=False
        )
        self.pool.evaluate(candidate_count=self._candidate_count)
        self.evaluation_generation += 1
        self.assembled_generation = -1

    def _evaluate(self, steps, dt, span):
        wp.copy(self.flags, self.query_flags)
        self._summary_counts.zero_()
        wp.launch(
            _point_candidates,
            self.pairs.size,
            [
                self.start,
                steps,
                self.triangles,
                self.triangle_owners,
                self.sample_area,
                self.nodes,
                self.node_owners,
                self.node_area,
                self.thickness,
                self.rows,
                self.pairs,
                self.dtype(self.length),
                self.dtype(self.stiffness),
                self.dtype(self.smoothing),
                self.dtype(self.mu),
                self.dtype(self.velocity_scale * dt),
                self.pool.candidates,
                self.flags,
                self._summary_counts,
                self._load_rows,
                self._loads,
            ],
            device=self.device,
        )
        wp.launch(_summary_count, 1, [self.flags, self.pool.candidates.count, self._load_count], device=self.device)
        loads = self._load_sums
        loads._reduce(
            self._load_rows, self.pool.candidates.identities, self._load_locals, self._loads, self._load_count, span
        )
        wp.launch(
            _finish_summary, 1, [self.flags, self._summary_counts, loads.result, self.summary], device=self.device
        )

    def assemble(self, assembly):
        if self._last_assembly == assembly.generation:
            raise ValueError("Cloth contact COO may be appended only once per assembly")
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
