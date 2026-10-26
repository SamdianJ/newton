# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded native contact generation; solver publication stays with its owner."""

import math
from functools import cache, partial
from typing import Any

import numpy as np
import warp as wp

from .actors import articulation_support
from .contact import _geometry_type, _law, _reset_flags, _validate_start
from .contact_actor import _actor_geometry, _center, _point_column
from .contact_pool import _ContactPool, _ContactRows
from .segment_graph import _SegmentGraphCache


@cache
def _batch_types(dtype):
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3

    @wp.struct
    class Samples:
        sources: wp.array[int]
        targets: wp.array[int]
        faces: wp.array[int]
        slots: wp.array[int]
        points: wp.array[vector]
        normals: wp.array[vector]
        area: wp.array[dtype]
        gaps: wp.array[dtype]

    @wp.struct
    class Support:
        kinds: wp.array[int]
        indices: wp.array[int]
        offsets: wp.array[int]
        columns: wp.array[int]
        local: wp.array[int]
        particles: wp.array[int]
        com: wp.array[vector]

    return Samples, Support


@wp.kernel
def _select_contact_chunk(first: int, end: int, capacity: int, begin: wp.array[int], count: wp.array[int]):
    begin[0] = first
    count[0] = wp.min(capacity, end - first)


@wp.kernel(enable_backward=False)
def _validate_selection(
    ids: wp.array[wp.uint64], count: wp.array[int], status: wp.array[int], limit: int, records: Any, workers: int
):
    i = wp.tid()
    n = count[0]
    if status[0] != 0 or n < 0 or n > ids.shape[0]:
        wp.atomic_or(records.flags, 0, 2)
        return
    while i < n:
        if ids[i] >= wp.uint64(limit) or (i > 0 and ids[i] <= ids[i - 1]):
            wp.atomic_or(records.flags, 0, 2)
        i += workers


@wp.kernel(enable_backward=False)
def _select_sparse_chunk(
    ids: wp.array[wp.uint64],
    size: wp.array[int],
    records: Any,
    lower: int,
    upper: int,
    offset: int,
    capacity: int,
    begin: wp.array[int],
    count: wp.array[int],
):
    n = wp.clamp(size[0], 0, ids.shape[0])
    lo, hi = int(0), n + 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if ids[mid] < wp.uint64(lower):
            lo = mid + 1
        else:
            hi = mid
    start = lo + 0
    hi = n + 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if ids[mid] < wp.uint64(upper):
            lo = mid + 1
        else:
            hi = mid
    end = lo + 0
    begin[0] = start + offset
    count[0] = wp.max(0, wp.min(capacity, end - start - offset))
    if records.flags[0] != 0:
        count[0] = 0


@wp.kernel
def _load_contact_chunk(
    all_samples: Any, batch: Any, first: wp.array[int], candidates: Any, selected: wp.array[wp.uint64], sparse: bool
):
    i = wp.tid()
    if i < candidates.count[0]:
        key = first[0] + i
        if sparse:
            key = int(selected[key])
        candidates.identities[i] = wp.uint64(key)
        candidates.support_counts[i] = 0
        candidates.force_counts[i] = 0
        for axis in range(2):
            candidates.energy[i, axis] = candidates.energy.dtype(0)
            for j in range(6):
                candidates.pairs[i, axis, j] = candidates.pairs.dtype(0)
        for axis in range(3):
            candidates.weights[i, axis] = candidates.weights.dtype(0)
        batch.sources[i] = all_samples.sources[key]
        batch.targets[i] = all_samples.targets[key]
        batch.faces[i] = all_samples.faces[key]
        batch.slots[i] = all_samples.slots[key]
        batch.points[i] = all_samples.points[key]
        batch.normals[i] = all_samples.normals[key]
        batch.area[i] = all_samples.area[key]
        batch.gaps[i] = all_samples.gaps[key]


@wp.kernel
def _derive_contact_steps(
    x: wp.array[Any],
    poses: wp.array[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
    start_x: wp.array[Any],
    start_poses: wp.array[Any],
    start_rigid_com: wp.array[Any],
    start_rigid_rotation: wp.array[Any],
    x_steps: wp.array[Any],
    link_translation_steps: wp.array[Any],
    link_rotation_steps: wp.array[Any],
    rigid_steps: wp.array[Any],
    rigid_rotation_steps: wp.array[Any],
):
    i = wp.tid()
    if i < x.shape[0]:
        x_steps[i] = x[i] - start_x[i]
    if i < poses.shape[0]:
        link_translation_steps[i] = wp.transform_get_translation(poses[i]) - wp.transform_get_translation(
            start_poses[i]
        )
        r = wp.quat_to_matrix(wp.transform_get_rotation(poses[i]))
        rn = wp.quat_to_matrix(wp.transform_get_rotation(start_poses[i]))
        link_rotation_steps[i] = r * wp.transpose(rn) - wp.identity(n=3, dtype=type(r[0, 0]))
    if i < rigid_com.shape[0]:
        rigid_steps[i] = rigid_com[i] - start_rigid_com[i]
        q = rigid_rotation[i] * wp.quat_inverse(start_rigid_rotation[i])
        if q[3] < type(q[3])(0.0):
            q = -q
        rigid_rotation_steps[i] = q


@wp.kernel
def _copy_contact_geometry(source: Any, target: Any, ids: wp.array[wp.uint64], count: wp.array[int], store: bool):
    i = wp.tid()
    if i < count[0]:
        a, b = i, int(ids[i])
        if not store:
            a, b = b, a
        target.y[b] = source.y[a]
        target.slip[b] = source.slip[a]
        target.gradient[b] = source.gradient[a]
        target.soft_normal[b] = source.soft_normal[a]
        target.distance[b] = source.distance[a]
        target.uncertified[b] = source.uncertified[a]


@wp.kernel
def _merge_contact_chunk_flags(flags: wp.array[int], chunk: wp.array[int], ids: wp.array[wp.uint64], records: Any):
    if chunk[0] != 0:
        wp.atomic_max(flags, 0, chunk[0])
        identity = int(-1)
        if chunk[1] >= 0 and chunk[1] < ids.shape[0]:
            identity = int(ids[chunk[1]])
        wp.atomic_min(flags, 1, identity)
        wp.atomic_or(records.flags, 0, wp.where(chunk[0] == 1, 4, 2))


@wp.kernel
def _contact_pool_start_flags(records: Any, flags: wp.array[int]):
    if flags[0] != 0:
        wp.atomic_or(records.flags, 0, wp.where(flags[0] == 1, 4, 2))


@wp.kernel
def _contact_pool_failure(flags: wp.array[int], records: Any):
    if records.flags[0] != 0:
        status = wp.where((records.flags[0] & 4) != 0, 1, 2)
        wp.atomic_max(flags, 0, status)
        wp.atomic_min(flags, 1, -1)


@wp.kernel
def _reference_rows(
    candidates: Any,
    rows: wp.array[int],
    identities: wp.array[wp.uint64],
    local: wp.array[int],
    values: wp.array[Any],
    count: wp.array[int],
    flags: wp.array[int],
    row_count: int,
):
    i, j = wp.tid()
    if i < candidates.count[0]:
        if j == 0 and (not wp.isfinite(candidates.energy[i, 0]) or not wp.isfinite(candidates.energy[i, 1])):
            wp.atomic_max(flags, 0, 2)
        if j < candidates.support_counts[i]:
            slot = wp.atomic_add(count, 0, 1)
            rows[slot] = candidates.columns[i, j]
            identities[slot] = candidates.identities[i]
            local[slot] = j
            values[slot] = candidates.residual[i, j]

        if j == 0:
            for axis in range(2):
                slot = wp.atomic_add(count, 0, 1)
                rows[slot] = row_count + axis
                identities[slot] = candidates.identities[i]
                local[slot] = 0
                values[slot] = candidates.energy[i, axis]


@wp.kernel
def _add_reference(source: wp.array[Any], result: wp.array[Any], status: wp.array[int], flags: wp.array[int]):
    i = wp.tid()
    value = result[i] + source[i]
    if status[0] != 0 or not wp.isfinite(value):
        wp.atomic_max(flags, 0, 2)
    result[i] = value


@wp.func
def _project_contact_column(
    candidates: Any,
    contact: int,
    local: int,
    row: int,
    jacobian: Any,
    residual: Any,
    gradient: Any,
    radial: Any,
    transverse: Any,
    legacy_tet: bool,
    bary: Any,
    axis: int,
    rotation: Any,
):
    candidates.columns[contact, local] = row
    candidates.residual[contact, local] = wp.dot(jacobian, residual)
    candidates.factors[contact, 0, local] = wp.dot(jacobian, gradient)
    candidates.factors[contact, 1, local] = wp.dot(jacobian, radial)
    candidates.factors[contact, 2, local] = wp.dot(jacobian, transverse)
    if legacy_tet:
        # Preserve legacy bary * world-coordinate arithmetic rather than using
        # an equivalent dot.
        candidates.residual[contact, local] = bary * (rotation * residual)[axis]
        candidates.factors[contact, 0, local] = bary * (rotation * gradient)[axis]
        candidates.factors[contact, 1, local] = bary * (rotation * radial)[axis]
        candidates.factors[contact, 2, local] = bary * (rotation * transverse)[axis]


@wp.kernel
def _project_contact_chunk(
    samples: Any,
    support: Any,
    geometry: Any,
    points: wp.array[Any],
    bodies: wp.array[int],
    triangles: wp.array2d[int],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
    normal: wp.array[Any],
    tangent: wp.array[Any],
    weights: wp.array2d[Any],
    local_r: wp.array[Any],
    candidates: Any,
    flags: wp.array[int],
    legacy: bool,
    particle_count: int,
):
    i = wp.tid()
    if i >= candidates.count[0] or flags[0] != 0:
        return
    source, target = samples.sources[i] + 0, samples.targets[i] + 0
    if source == -3:
        return
    point, rotation, n = points[i], geometry.rotation[i], normal[i]
    axis = type(point)(1.0, 0.0, 0.0)
    if wp.abs(n[1]) < wp.abs(n[0]) and wp.abs(n[1]) <= wp.abs(n[2]):
        axis = type(point)(0.0, 1.0, 0.0)
    elif wp.abs(n[2]) < wp.abs(n[0]):
        axis = type(point)(0.0, 0.0, 1.0)
    t = tangent[i]
    if weights[i, 1] > type(weights[0, 0])(0.0) and wp.length(t) == type(weights[0, 0])(0.0):
        t = wp.normalize(wp.cross(n, axis))
    transverse = wp.cross(n, t)
    candidates.weights[i, 0] = weights[i, 0]
    candidates.weights[i, 1] = weights[i, 2]
    candidates.weights[i, 2] = weights[i, 1]
    candidates.rounding[i] = geometry.rounding[i]
    r = local_r[i]
    k = int(0)
    for end in range(2):
        tet = source < 0 and end == 0
        if legacy:
            tet = end == 1
        if tet:
            for node in range(3):
                particle = triangles[samples.faces[i], node]
                row = support.particles[particle]
                if row >= 0:
                    bary = type(point[0])(1.0) / type(point[0])(6.0)
                    if node == samples.slots[i] % 3:
                        bary = type(point[0])(2.0) / type(point[0])(3.0)
                    for coordinate in range(3):
                        vector = type(point)(0.0)
                        vector[coordinate] = bary
                        jacobian = wp.transpose(rotation) * vector
                        _project_contact_column(
                            candidates,
                            i,
                            k,
                            row + coordinate,
                            jacobian,
                            r,
                            geometry.gradient[i],
                            t,
                            transverse,
                            legacy,
                            bary,
                            coordinate,
                            rotation,
                        )
                        k += 1
        else:
            shape = target
            sign = type(point[0])(-1.0)
            if not legacy and end == 0:
                shape, sign = source, type(point[0])(1.0)
            body = bodies[shape]
            if body >= 0:
                for index in range(support.offsets[body], support.offsets[body + 1]):
                    column = _point_column(
                        body, support.local[index], point, support.kinds, support.indices, com, jt, jw, rigid_com
                    )
                    jacobian = wp.transpose(rotation) * (sign * column)
                    _project_contact_column(
                        candidates,
                        i,
                        k,
                        support.columns[index],
                        jacobian,
                        r,
                        geometry.gradient[i],
                        t,
                        transverse,
                        False,
                        type(point[0])(0.0),
                        0,
                        rotation,
                    )
                    k += 1
    candidates.support_counts[i] = k
    world = rotation * r
    f = int(0)
    for end in range(2):
        shape = source
        force = -world
        if end == 1:
            shape, force = target, world
        torque = wp.cross(point, force)
        for coordinate in range(3):
            candidates.pairs[i, end, coordinate] = force[coordinate]
            candidates.pairs[i, end, coordinate + 3] = torque[coordinate]
        if shape < 0:
            for node in range(3):
                bary = type(point[0])(1.0) / type(point[0])(6.0)
                if node == samples.slots[i] % 3:
                    bary = type(point[0])(2.0) / type(point[0])(3.0)
                for coordinate in range(3):
                    candidates.force_rows[i, f] = 3 * triangles[samples.faces[i], node] + coordinate
                    candidates.forces[i, f] = bary * force[coordinate]
                    f += 1
        else:
            body = bodies[shape]
            if body >= 0:
                torque = wp.cross(point - _center(body, support.kinds, support.indices, com, rigid_com), force)
                for coordinate in range(6):
                    candidates.force_rows[i, f] = 3 * particle_count + 6 * body + coordinate
                    value = force[coordinate % 3]
                    if coordinate >= 3:
                        value = torque[coordinate % 3]
                    candidates.forces[i, f] = value
                    f += 1
    candidates.force_counts[i] = f


def _storage_interval(array):
    offsets = [(size - 1) * stride for size, stride in zip(array.shape, array.strides, strict=True)]
    begin = array.ptr + sum(min(0, offset) for offset in offsets)
    end = array.ptr + sum(max(0, offset) for offset in offsets) + wp.types.type_size_in_bytes(array.dtype)
    return begin, end


class _ContactBatch:
    """Generate complete supported candidates from validated contact metadata.

    The metadata owner supplies Model identity and the unchanged native law
    parameters. No owner factor_support, per-contact selector/J table, residual
    gather map or evaluated local output is consumed. Owned projection storage
    is batch x max_support; the complete physical-step reference is O(M).
    This generator does not publish physical state. A private spatial selection
    can restrict evaluation while preserving original physical-step references.
    """

    METADATA_ARRAYS = (
        "bodies",
        "transforms",
        "triangles",
        "types",
        "scales",
        "sdf_indices",
        "table",
        "smooth_table",
        "margins",
    )

    def __init__(self, contact, *, batch_capacity, capacity, contact_factor_capacity=None, use_cuda_graph=False):
        self.contact, self.device = contact, contact.model.device
        self.dtype, self.vector, self.matrix = contact.dtype, contact.vector, contact.matrix
        self.batch_capacity = batch_capacity
        model, m = contact.model, contact.count
        if (
            isinstance(batch_capacity, (bool, np.bool_))
            or not isinstance(batch_capacity, (int, np.integer))
            or not (0 if m == 0 else 1) <= batch_capacity <= m
        ):
            raise ValueError("Invalid native contact batch capacity")
        if capacity is not None and (
            isinstance(capacity, (bool, np.bool_))
            or not isinstance(capacity, (int, np.integer))
            or not 0 <= capacity < 2**31
        ):
            raise ValueError("Invalid native contact pool capacity")
        samples_type, support_type = _batch_types(self.dtype)
        self.support = support_type()
        actors = None if contact.actor_samples is None else contact.actor_samples.actors
        kinds = np.zeros(model.body_count, np.int32)
        indices = contact.body_local.numpy()
        body_support = {}
        if actors is not None:
            kinds.fill(-1)
            indices = np.full(model.body_count, -1, np.int32)
            arms = tuple(a for a in actors.actors if a.kind == "articulation")
            for body, (local, columns) in articulation_support(model, arms).items():
                kinds[body], indices[body] = 0, local
                body_support[body] = columns
            for local, actor in enumerate(a for a in actors.actors if a.kind == "rigid"):
                body = actor.body_indices[0]
                kinds[body], indices[body] = 1, local
                body_support[body] = [(actor.dof_start + j, j) for j in range(6)]
            particles = np.asarray(actors.maps["particle_to_global"], np.int32)
        else:
            body_support = {
                body: [
                    (contact.layout.q_scalar_offset + j, indices[body] * contact.nq + j)
                    for j in contact._legacy_ancestors.get(body, range(contact.nq))
                ]
                for body in range(model.body_count)
                if indices[body] >= 0
            }
            mapping = contact.mapping.numpy()
            particles = np.where(mapping >= 0, contact.layout.x_scalar_offset + 3 * mapping, -1).astype(np.int32)
        self.rigid_count = int(np.count_nonzero(kinds == 1))
        offsets, columns, local = [0], [], []
        for body in range(model.body_count):
            for row, j in body_support.get(body, ()):
                columns.append(row)
                local.append(j)
            offsets.append(len(columns))
        for name, values, scalar in (
            ("kinds", kinds, int),
            ("indices", indices, int),
            ("offsets", offsets, int),
            ("columns", columns, int),
            ("local", local, int),
            ("particles", particles, int),
            ("com", model.body_com.numpy().astype(contact.numpy_dtype), self.vector),
        ):
            setattr(self.support, name, wp.array(values, dtype=scalar, device=self.device))
        shapes, triangles = contact.shapes.numpy(), contact.triangles.numpy()
        n = contact.legacy_count
        legacy = {
            "sources": np.full(n, -1, np.int32),
            "targets": np.tile(shapes, len(triangles) * 3),
            "faces": np.repeat(np.arange(len(triangles)), 3 * len(shapes)),
            "slots": np.tile(np.repeat(np.arange(3), len(shapes)), len(triangles)),
            "points": np.zeros((n, 3)),
            "normals": np.zeros((n, 3)),
            "area": np.repeat(contact.area.numpy(), 3 * len(shapes)),
            "gaps": np.tile(contact.parameters[4] + contact.margins.numpy()[shapes], len(triangles) * 3),
        }
        a = contact.actor_samples
        self.all_samples, self.samples = samples_type(), samples_type()
        fields = {
            "sources": "source_shapes",
            "targets": "targets",
            "faces": "faces",
            "slots": "slots",
            "points": "points",
            "normals": "normals",
            "area": "weights",
            "gaps": "offsets",
        }
        arrays = {}
        for name, base_values in legacy.items():
            values = base_values
            if a is not None and a.count:
                values = np.concatenate((values, getattr(a, fields[name])), axis=0)
            scalar = self.vector if name in ("points", "normals") else self.dtype if name in ("area", "gaps") else int
            arrays[name] = values
            setattr(self.all_samples, name, wp.array(values, dtype=scalar, device=self.device))
        bodies = contact.bodies.numpy()
        demands = []
        for source, target, face in zip(arrays["sources"], arrays["targets"], arrays["faces"], strict=True):
            k = len(body_support.get(int(bodies[target]), ()))
            if source == -3:
                edge = a.cable_edges[face]
                k += 3 * int(np.count_nonzero(particles[edge[:2]] >= 0))
                k += int(actors.maps["theta_to_global"][edge[2]] >= 0)
            elif source < 0:
                k += 3 * int(np.count_nonzero(particles[triangles[face]] >= 0))
            else:
                k += len(body_support.get(int(bodies[source]), ()))
            demands.append(k)
        if capacity is None:
            capacity = sum(max(1, k) for k in demands)
        self.pool = _ContactPool(
            contact.layout.scalar_dof_count,
            3 * model.particle_count
            + 6 * model.body_count
            + (0 if contact._materials is None else contact._materials.theta_count),
            m,
            capacity,
            max(1, max(demands, default=0)),
            max(1, m),
            self.dtype,
            self.device,
            batch_capacity=batch_capacity,
            contact_factor_capacity=contact_factor_capacity,
            use_cuda_graph=use_cuda_graph,
        )
        for name in fields:
            array = getattr(self.all_samples, name)
            setattr(
                self.samples, name, wp.zeros((batch_capacity, *array.shape[1:]), dtype=array.dtype, device=self.device)
            )
        self.current, self._start_chunk = (
            self._geometry(batch_capacity, rotation=True),
            self._geometry(batch_capacity, rotation=False),
        )
        self.start = self._geometry(m, rotation=False)
        self.first = wp.zeros(1, dtype=int, device=self.device)
        self.points = wp.zeros(batch_capacity, dtype=self.vector, device=self.device)
        self.normal, self.tangent, self.local_r = (wp.zeros_like(self.points) for _ in range(3))
        self.weights = wp.zeros((batch_capacity, 3), dtype=self.dtype, device=self.device)
        self.omitted = wp.zeros((batch_capacity, 4), dtype=self.dtype, device=self.device)
        self.mask = wp.zeros(batch_capacity, dtype=int, device=self.device)
        self.flags, self.start_flags, self._flags = (wp.zeros(2, dtype=int, device=self.device) for _ in range(3))
        self._empty_scalar = wp.empty(0, dtype=self.dtype, device=self.device)
        self._empty_int = wp.empty(0, dtype=int, device=self.device)
        self._empty_selector = wp.empty(0, dtype=wp.vec2i, device=self.device)
        self._empty_vector = wp.empty(0, dtype=self.vector, device=self.device)
        self._empty_rotation = wp.empty(0, dtype=wp.quatd if self.dtype == wp.float64 else wp.quat, device=self.device)
        self._key_ids = wp.array(np.arange(batch_capacity, dtype=np.int32), dtype=int, device=self.device)
        self._empty_ids = wp.empty(0, dtype=wp.uint64, device=self.device)
        # Step-start poses, and step increments derived from them for callers
        # that do not supply exact increments.
        nr, nb, n = self.rigid_count, contact.body_count, contact.model.particle_count
        self._start_x = wp.zeros(n, dtype=self.vector, device=self.device)
        self._start_poses = wp.zeros(nb, dtype=contact.transform, device=self.device)
        self._start_rigid_com = wp.zeros(nr, dtype=self.vector, device=self.device)
        self._start_rigid_rotation = wp.zeros(nr, dtype=self._empty_rotation.dtype, device=self.device)
        self._derived_steps = (
            wp.zeros(n, dtype=self.vector, device=self.device),
            wp.zeros(nb, dtype=self.vector, device=self.device),
            wp.zeros(nb, dtype=self.matrix, device=self.device),
            wp.zeros(nr, dtype=self.vector, device=self.device),
            wp.zeros(nr, dtype=self._empty_rotation.dtype, device=self.device),
        )
        self.graphs = _SegmentGraphCache(self.device, enabled=use_cuda_graph and self.device.is_cuda)
        self._started = False
        self.material_surface = None
        if contact._materials is not None:
            from .contact_material import _MaterialSurface  # noqa: PLC0415

            self.material_surface = _MaterialSurface(self)
        self._storage_bindings = None
        self._storage_intervals = ()

    def _geometry(self, count, *, rotation):
        geometry = _geometry_type(self.dtype)()
        for name in ("y", "slip", "gradient", "soft_normal"):
            setattr(geometry, name, wp.zeros(count, dtype=self.vector, device=self.device))
        geometry.distance = wp.zeros(count, dtype=self.dtype, device=self.device)
        geometry.rounding = wp.zeros(count, dtype=self.dtype, device=self.device)
        geometry.uncertified = wp.zeros(count, dtype=int, device=self.device)
        geometry.rotation = wp.zeros(count if rotation else 0, dtype=self.matrix, device=self.device)
        geometry.jacobian_q = wp.empty((count, 0), dtype=self.vector, device=self.device)
        return geometry

    def buffers(self):
        arrays = [a for a in vars(self).values() if isinstance(a, wp.array)]
        for owner in (self.all_samples, self.samples, self.support, self.current, self.start, self._start_chunk):
            arrays.extend(a for a in vars(owner).values() if isinstance(a, wp.array))
        # Shared metadata is a graph binding too: rebinding a shape/SDF array
        # must not replay a graph containing its retired pointer.
        if self.material_surface is not None:
            arrays.extend(self.material_surface.buffers())
        metadata = tuple(getattr(self.contact, name) for name in self.METADATA_ARRAYS)
        if hasattr(self, "reference_rows"):
            arrays.extend(self.reference_rows.buffers())
            arrays.extend(self.reference_inputs)
        return tuple(arrays) + self.pool.buffers() + metadata

    def initialize_reference(self):
        """Reserve batch-sized full-reference scratch, independent of contact admission budgets."""
        k = self.pool.max_support
        capacity = self.batch_capacity * (k + 2)
        self.reference_rows = _ContactRows(
            self.pool.row_count + 2,
            capacity,
            max(1, self.contact.count),
            k,
            self.dtype,
            self.device,
            use_cuda_graph=self.graphs.enabled,
        )
        self.reference_inputs = (
            *(wp.zeros(capacity, dtype=dtype, device=self.device) for dtype in (int, wp.uint64, int, self.dtype)),
            wp.zeros(1, dtype=int, device=self.device),
        )
        self.reference_residual = wp.zeros(self.pool.row_count, dtype=self.dtype, device=self.device)
        self.reference_energy = wp.zeros(2, dtype=self.dtype, device=self.device)

    def evaluate_reference(self, x, poses, com, jt, jw, *, dt, rigid_com=None, rigid_rotation=None, steps=None):
        """Stream every sample into a native residual without capacity truncation or tangent storage."""
        if not self._started or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Freeze a physical step before its full reference")
        dt = self.contact.numpy_dtype(dt)
        falloff = self.contact.numpy_dtype(self.contact.parameters[2]) * dt
        if (
            not np.isfinite(dt)
            or dt <= 0
            or (self.contact.parameters[1] > 0 and (not np.isfinite(falloff) or falloff <= 0))
        ):
            raise ValueError("Invalid native contact displacement scale")
        state = self._state(x, poses, com, jt, jw, rigid_com, rigid_rotation, steps)
        self.reference_rows._validate_inputs(*self.reference_inputs)
        self.graphs.run(
            "contact_reference",
            (float(dt),),
            (*self.buffers(), *state),
            partial(self._evaluate_reference, state, dt),
        )

    def _evaluate_reference(self, state, dt):
        wp.copy(self.flags, self.start_flags)
        self.pool.records.flags.zero_()
        self.reference_residual.zero_()
        self.reference_energy.zero_()
        for mode in self._chunks():
            self._stage(state, mode, dt, begin=False)
            wp.launch(
                _merge_contact_chunk_flags,
                1,
                [self.flags, self._flags, self.pool.candidates.identities, self.pool.records],
                device=self.device,
            )
            self.reference_inputs[-1].zero_()
            wp.launch(
                _reference_rows,
                (self.batch_capacity, self.pool.max_support),
                [self.pool.candidates, *self.reference_inputs, self.flags, self.pool.row_count],
                device=self.device,
            )
            self.reference_rows._reduce(*self.reference_inputs)
            # The full graph clears the accumulator before processing any chunk.
            wp.launch(
                _add_reference,
                self.reference_residual.size,
                [self.reference_rows.result, self.reference_residual, self.reference_rows.flags, self.flags],
                device=self.device,
            )
            wp.launch(
                _add_reference,
                2,
                [
                    self.reference_rows.result[self.pool.row_count :],
                    self.reference_energy,
                    self.reference_rows.flags,
                    self.flags,
                ],
                device=self.device,
            )

    def _state(self, x, poses, com, jt, jw, rigid_com, rigid_rotation, steps=None):
        """Validate a contact state; append its step increments.

        ``steps`` holds the exact particle, link origin, link rotation ``E``,
        rigid COM and rigid rotation increments from the step start. Without
        them the increments are derived from the step-start poses.
        """
        c = self.contact
        nr = self.rigid_count
        if nr == 0:
            rigid_com, rigid_rotation = self._empty_vector, self._empty_rotation
        provided = steps is not None
        if provided:
            steps = tuple(steps)
            if len(steps) != 5:
                raise ValueError("Invalid native contact batch steps")
            if nr == 0:
                steps = (*steps[:3], self._empty_vector, self._empty_rotation)
        else:
            steps = self._derived_steps
        checks = [
            (x, self.vector, (c.model.particle_count,)),
            (poses, c.transform, (c.body_count,)),
            (com, self.vector, (c.body_count,)),
            (jt, self.vector, c.jacobian_shape),
            (jw, self.vector, c.jacobian_shape),
            (rigid_com, self.vector, (nr,)),
            (rigid_rotation, self._empty_rotation.dtype, (nr,)),
        ]
        if provided:
            checks += [
                (steps[0], self.vector, (c.model.particle_count,)),
                (steps[1], self.vector, (c.body_count,)),
                (steps[2], self.matrix, (c.body_count,)),
                (steps[3], self.vector, (nr,)),
                (steps[4], self._empty_rotation.dtype, (nr,)),
            ]
        for a, dtype, shape in checks:
            if not isinstance(a, wp.array) or (a.dtype, a.shape, a.device) != (dtype, shape, self.device):
                raise ValueError("Invalid native contact batch state")
        state = tuple(a for a, _, _ in checks)
        owned = self.buffers()[: -len(self.METADATA_ARRAYS)]
        bindings = tuple((a.ptr, a.shape, a.strides, a.dtype) for a in owned if a.size)
        if bindings != self._storage_bindings:
            self._storage_intervals = tuple(_storage_interval(a) for a in owned if a.size)
            self._storage_bindings = bindings
        intervals = self._storage_intervals
        for a in state:
            if a.size:
                begin, end = _storage_interval(a)
                if any(begin < hi and lo < end for lo, hi in intervals):
                    raise ValueError("Native contact state aliases producer storage")
        if not provided:
            state = (*state, *steps)
            self._derive(state)
        return state

    def _snapshot(self, state):
        """Freeze the step-start poses; the start is its own zero step."""
        x, poses, _, _, _, rigid_com, rigid_rotation = state[:7]
        for target, source in zip(
            (self._start_x, self._start_poses, self._start_rigid_com, self._start_rigid_rotation),
            (x, poses, rigid_com, rigid_rotation),
            strict=True,
        ):
            if target.size:
                wp.copy(target, source)
        for step in self._derived_steps:
            step.zero_()
        if self._derived_steps[4].size:
            self._derived_steps[4].fill_(wp.quat_identity(self.dtype))
        return (*state[:7], *self._derived_steps)

    def _derive(self, state):
        """Derive step increments of a state from its rounded poses."""
        x, poses, _, _, _, rigid_com, rigid_rotation = state[:7]
        steps = self._derived_steps
        size = max(x.size, poses.size, rigid_com.size)
        if size:
            wp.launch(
                _derive_contact_steps,
                size,
                [
                    x,
                    poses,
                    rigid_com,
                    rigid_rotation,
                    self._start_x,
                    self._start_poses,
                    self._start_rigid_com,
                    self._start_rigid_rotation,
                    *steps,
                ],
                device=self.device,
            )

    def _chunks(self, selection=None):
        if selection is not None:
            if selection.mode_counts is None:
                raise ValueError("Check spatial selection before scheduling batches")
            middle = (
                self.contact.count
                if self.material_surface is None
                else self.contact.legacy_count + self.contact.actor_samples.material_start
            )
            bounds = ((0, self.contact.legacy_count), (self.contact.legacy_count, middle))
            if self.material_surface is not None:
                bounds += ((middle, self.contact.count),)
            for mode, ((lower, upper), count) in enumerate(zip(bounds, selection.mode_counts, strict=True), start=2):
                for offset in range(0, count, max(1, self.batch_capacity)):
                    wp.launch(
                        _select_sparse_chunk,
                        1,
                        [
                            selection.identities,
                            selection.count,
                            self.pool.records,
                            lower,
                            upper,
                            offset,
                            self.batch_capacity,
                            self.first,
                            self.pool.candidates.count,
                        ],
                        device=self.device,
                    )
                    yield mode
            return
        middle = self.contact.count
        if self.material_surface is not None:
            middle = self.contact.legacy_count + self.contact.actor_samples.material_start
        for begin, end, mode in (
            (0, self.contact.legacy_count, 2),
            (self.contact.legacy_count, middle, 3),
            (middle, self.contact.count, 4),
        ):
            for first in range(begin, end, max(1, self.batch_capacity)):
                wp.launch(
                    _select_contact_chunk,
                    1,
                    [first, end, self.batch_capacity, self.first, self.pool.candidates.count],
                    device=self.device,
                )
                yield mode

    def _query(self, state, begin):
        c, s, t = self.contact, self.samples, self.support
        x, poses, com, jt, jw, rigid_com, rigid_rotation = state[:7]
        surface = self._empty_vector if self.material_surface is None else self.material_surface.point_steps
        wp.launch(
            _actor_geometry,
            self.batch_capacity,
            [
                x,
                poses,
                com,
                jt,
                jw,
                rigid_com,
                rigid_rotation,
                *state[7:],
                surface,
                self._start_chunk.y,
                not begin,
                t.com,
                t.kinds,
                t.indices,
                c.bodies,
                c.transforms,
                s.sources,
                s.targets,
                s.faces,
                s.slots,
                s.points,
                s.normals,
                c.triangles,
                c.types,
                c.scales,
                c.sdf_indices,
                c.table,
                c.smooth_table,
                self._empty_int,
                self._empty_selector,
                self._empty_vector,
                0,
                self.dtype(c.quaternion_norm_tolerance),
                self.current,
                self.points,
                self._flags,
                False,
                self.pool.candidates.count,
            ],
            device=self.device,
        )

    def _stage(self, state, mode, dt, *, begin, selection=None):
        c, out = self.contact, self.pool.candidates
        wp.launch(_reset_flags, 1, [self._flags], device=self.device)
        wp.launch(
            _load_contact_chunk,
            self.batch_capacity,
            [
                self.all_samples,
                self.samples,
                self.first,
                out,
                self._empty_ids if selection is None else selection.identities,
                selection is not None,
            ],
            device=self.device,
        )
        if not begin:
            # The query forms each local point from its step-start point.
            wp.launch(
                _copy_contact_geometry,
                self.batch_capacity,
                [self.start, self._start_chunk, out.identities, out.count, False],
                device=self.device,
            )
        self._query(state, begin)
        if begin:
            wp.launch(
                _validate_start,
                self.batch_capacity,
                [
                    self.current,
                    self.samples.targets if mode == 2 else self._key_ids,
                    c.margins if mode == 2 else self.samples.gaps,
                    self.dtype(c.parameters[4] if mode == 2 else 0),
                    self.dtype(c.parameters[3]),
                    int(mode == 4 or c._alignment_policy == "mask"),
                    self._flags,
                    True,
                    out.count,
                ],
                device=self.device,
            )
            wp.launch(
                _copy_contact_geometry,
                self.batch_capacity,
                [self.current, self.start, out.identities, out.count, True],
                device=self.device,
            )
            return
        self.weights.zero_()
        self.local_r.zero_()
        self.omitted.zero_()
        self.mask.zero_()
        wp.launch(
            _law,
            self.batch_capacity,
            [
                self.current,
                self._start_chunk,
                self.samples.targets,
                c.margins,
                self.samples.area,
                self.dtype(c.parameters[4] if mode == 2 else 0),
                self.dtype(c.parameters[0]),
                self.dtype(c.parameters[3]),
                self.dtype(c.parameters[1]),
                self.dtype(c.numpy_dtype(c.parameters[2]) * c.numpy_dtype(dt)),
                int(mode == 4 or c._alignment_policy == "mask"),
                self._flags,
                out.energy,
                self._empty_scalar,
                self.normal,
                self.tangent,
                self.weights,
                self.local_r,
                self.omitted,
                self.mask,
                mode,
                0,
                self.samples.gaps,
                self._empty_int,
                self._empty_int,
                self._empty_vector,
                out.count,
            ],
            device=self.device,
        )
        wp.launch(
            _project_contact_chunk,
            self.batch_capacity,
            [
                self.samples,
                self.support,
                self.current,
                self.points,
                c.bodies,
                c.triangles,
                state[2],
                state[3],
                state[4],
                state[5],
                self.normal,
                self.tangent,
                self.weights,
                self.local_r,
                out,
                self._flags,
                mode == 2,
                c.model.particle_count,
            ],
            device=self.device,
        )

        if self.material_surface is not None:
            self.material_surface.project(state)

    def begin_step(self, x, poses, com, jt, jw, *, rigid_com=None, rigid_rotation=None):
        state = self._snapshot(self._state(x, poses, com, jt, jw, rigid_com, rigid_rotation))
        self.graphs.run("contact_start", (), (*self.buffers(), *state), partial(self._begin_step, state))
        self._started = True

    def _begin_step(self, state):
        wp.launch(_reset_flags, 1, [self.start_flags], device=self.device)
        for mode in self._chunks():
            self._stage(state, mode, 0, begin=True)
            wp.launch(
                _merge_contact_chunk_flags,
                1,
                [self.start_flags, self._flags, self.pool.candidates.identities, self.pool.records],
                device=self.device,
            )

    def evaluate(self, x, poses, com, jt, jw, *, dt, rigid_com=None, rigid_rotation=None, steps=None, selection=None):
        if not self._started or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Freeze a physical step and use positive finite dt")
        dt = self.contact.numpy_dtype(dt)
        falloff = self.contact.numpy_dtype(self.contact.parameters[2]) * dt
        if (
            not np.isfinite(dt)
            or dt <= 0
            or (self.contact.parameters[1] > 0 and (not np.isfinite(falloff) or falloff <= 0))
        ):
            raise ValueError("Invalid native contact displacement scale")
        state = self._state(x, poses, com, jt, jw, rigid_com, rigid_rotation, steps)
        bindings = ()
        if selection is not None:
            if selection.batch is not self:
                raise ValueError("Spatial selection belongs to another contact batch")
            selection.validate_storage()
            bindings = selection.buffers()
        chunks = None
        if selection is not None:
            if selection.mode_counts is None:
                raise ValueError("Check spatial selection before scheduling batches")
            chunks = tuple(
                (count + max(1, self.batch_capacity) - 1) // max(1, self.batch_capacity)
                for count in selection.mode_counts
            )
        self.pool._validate_storage()
        self.graphs.run(
            "contact_batch_eval",
            (float(dt), chunks),
            (*self.buffers(), *state, *bindings),
            partial(self._evaluate_batches, state, dt, selection),
        )
        self.pool._stream_started = True
        self.pool.finish()
        wp.launch(_contact_pool_failure, 1, [self.flags, self.pool.records], device=self.device)

    def _evaluate_batches(self, state, dt, selection):
        # Reset the entire stream inside the graph before any additive append.
        self.pool._begin_stream()
        wp.copy(self.flags, self.start_flags)
        wp.launch(_contact_pool_start_flags, 1, [self.pool.records, self.start_flags], device=self.device)
        if selection is not None:
            if selection.mode_counts is None:
                raise ValueError("Check spatial selection before scheduling batches")
            workers = max(1, min(selection.capacity, 256))
            wp.launch(
                _validate_selection,
                workers,
                [
                    selection.identities,
                    selection.count,
                    selection.status,
                    self.contact.count,
                    self.pool.records,
                    workers,
                ],
                device=self.device,
            )
        for mode in self._chunks(selection):
            self._stage(state, mode, dt, begin=False, selection=selection)
            wp.launch(
                _merge_contact_chunk_flags,
                1,
                [self.flags, self._flags, self.pool.candidates.identities, self.pool.records],
                device=self.device,
            )
            self.pool._append_stream()
