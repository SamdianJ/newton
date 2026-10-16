# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model sampling adapter for private swept patch queries and stable batch selections."""

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...geometry import GeoType
from .aabb import _CandidateBounds
from .cable import _transport
from .contact_pool import _CONTACT_WORKERS, _merge_contact_keys
from .contact_query import _ContactQuery
from .contact_sampling import _surface_samples
from .contact_sweep import _ArticulationSweep, _length_upper, _product_upper, _sum_upper


@wp.func_native("""
    wp::uint64 bits;
    memcpy(&bits, &x, sizeof(bits));
    if (x == 0.0) bits = 0x8000000000000001ULL;
    else if (x > 0.0) --bits;
    else if (x > -INFINITY) ++bits;
    wp::float64 result;
    memcpy(&result, &bits, sizeof(result));
    return result;
""")
def _lower(x: wp.float64) -> wp.float64: ...


@wp.kernel(enable_backward=False)
def _material_reach(
    samples: wp.array[int],
    radii: wp.array[wp.float64],
    edges: wp.array[wp.vec3i],
    x0: wp.array[Any],
    x1: wp.array[Any],
    directors: wp.array[Any],
    gamma: wp.float64,
    reach: wp.array[wp.float64],
    status: wp.array[int],
):
    i = wp.tid()
    sample = samples[i]
    if sample < 0:
        return
    a, b, angle = edges[sample][0], edges[sample][1], edges[sample][2]
    old = wp.normalize(x0[b] - x0[a])
    tangent = wp.normalize(x1[b] - x1[a])
    if type(old[0])(1.0) + wp.dot(old, tangent) <= type(old[0])(1e-10):
        wp.atomic_or(status, 0, 1)
        return
    d = _transport(old, tangent, directors[angle])
    v, w = wp.vec3d(d), wp.vec3d(wp.cross(tangent, d))
    # Frobenius norm bounds every angular phase, including native chart drift.
    current = wp.sqrt(wp.dot(v, v) + wp.dot(w, w))
    v, w = wp.vec3d(directors[angle]), wp.vec3d(wp.cross(old, directors[angle]))
    initial = wp.sqrt(wp.dot(v, v) + wp.dot(w, w))
    value = radii[i] * wp.max(current, initial) * (wp.float64(1.0) + gamma)
    reach[i] = -_lower(-value)
    if not wp.isfinite(value):
        wp.atomic_or(status, 0, 1)


@wp.kernel(enable_backward=False)
def _patch_bounds(
    shapes: wp.array[int],
    faces: wp.array[int],
    local_lo: wp.array[wp.vec3d],
    local_hi: wp.array[wp.vec3d],
    radii: wp.array[wp.float64],
    reach: wp.array[wp.float64],
    bodies: wp.array[int],
    transforms: wp.array[Any],
    kinds: wp.array[int],
    indices: wp.array[int],
    triangles: wp.array2d[int],
    x0: wp.array[Any],
    x1: wp.array[Any],
    poses0: wp.array[Any],
    com0: wp.array[Any],
    com1: wp.array[Any],
    displacement: wp.array[wp.float64],
    numerical: wp.array[wp.float64],
    gamma: wp.float64,
    tiny: wp.float64,
    lower: wp.array[wp.vec3d],
    upper: wp.array[wp.vec3d],
    status: wp.array[int],
):
    i = wp.tid()
    shape = shapes[i]
    if shape < -1 or shape >= bodies.shape[0]:
        wp.atomic_or(status, 0, 1)
        return
    lo, hi = wp.vec3d(wp.inf), wp.vec3d(-wp.inf)
    magnitude, motion = wp.float64(0.0), wp.float64(0.0)
    if shape < 0:
        if faces[i] < 0 or faces[i] >= triangles.shape[0]:
            wp.atomic_or(status, 0, 1)
            return
        for node in range(3):
            particle = triangles[faces[i], node]
            if particle < 0 or particle >= x0.shape[0]:
                wp.atomic_or(status, 0, 1)
                return
            a, b = wp.vec3d(x0[particle]), wp.vec3d(x1[particle])
            if not wp.isfinite(a) or not wp.isfinite(b):
                wp.atomic_or(status, 0, 1)
                return
            lo, hi = wp.min(lo, wp.min(a, b)), wp.max(hi, wp.max(a, b))
            magnitude = wp.max(magnitude, wp.max(wp.length(a), wp.length(b)))
    else:
        body = bodies[shape]
        if body < -1 or body >= kinds.shape[0]:
            wp.atomic_or(status, 0, 1)
            return
        local = int(-1)
        kind = int(-1)
        if body >= 0:
            if body >= kinds.shape[0]:
                wp.atomic_or(status, 0, 1)
                return
            kind, local = kinds[body] + 0, indices[body] + 0
            limit = poses0.shape[0]
            if kind == 1:
                limit = com0.shape[0]
            if local < 0 or local >= limit or (kind != 0 and kind != 1):
                wp.atomic_or(status, 0, 1)
                return
        if kind == 1:
            a, b = wp.vec3d(com0[local]), wp.vec3d(com1[local])
            if not wp.isfinite(a) or not wp.isfinite(b):
                wp.atomic_or(status, 0, 1)
                return
            lo, hi = wp.min(a, b) - wp.vec3d(radii[i]), wp.max(a, b) + wp.vec3d(radii[i])
            magnitude = wp.max(wp.length(a), wp.length(b)) + radii[i]
        else:
            t = transforms[shape]
            pose = wp.transformd(wp.vec3d(wp.transform_get_translation(t)), wp.quatd(wp.transform_get_rotation(t)))
            if body >= 0:
                p = poses0[local]
                pose = (
                    wp.transformd(wp.vec3d(wp.transform_get_translation(p)), wp.quatd(wp.transform_get_rotation(p)))
                    * pose
                )
                motion = _sum_upper(displacement[local], numerical[local])
            for corner in range(8):
                point = local_lo[i]
                for axis in range(3):
                    if (corner & (1 << axis)) != 0:
                        point[axis] = local_hi[i][axis]
                world = wp.transform_point(pose, point)
                lo, hi = wp.min(lo, world), wp.max(hi, world)
                # Absolute operand magnitudes retain cancellation error at zero.
                magnitude = wp.max(magnitude, wp.length(wp.transform_get_translation(pose)) + wp.length(point))
            if body >= 0:
                magnitude += wp.length(wp.vec3d(wp.transform_get_translation(poses0[local])))
            magnitude += wp.length(wp.vec3d(wp.transform_get_translation(t)))
    padding = _sum_upper(_sum_upper(motion, reach[i]), _sum_upper(_product_upper(gamma, magnitude), tiny))
    for axis in range(3):
        lo[axis] = _lower(lo[axis] - padding)
        hi[axis] = -_lower(-hi[axis] - padding)
    if not wp.isfinite(lo) or not wp.isfinite(hi):
        wp.atomic_or(status, 0, 1)
        lo, hi = wp.vec3d(), wp.vec3d()
    lower[i] = lo
    upper[i] = hi


@wp.kernel(enable_backward=False)
def _map_query_keys(
    keys: wp.array[wp.uint64],
    count: wp.array[int],
    query_status: wp.array[int],
    sweep_status: wp.array[int],
    dictionary: wp.array[wp.uint64],
    original: wp.array[wp.uint64],
    identities: wp.array[wp.uint64],
    indices: wp.array[int],
    status: wp.array[int],
):
    i = wp.tid()
    if query_status[0] != 0 or sweep_status[0] != 0:
        wp.atomic_or(status, 0, 1)
        return
    if i < count[0] and status[0] == 0:
        lo, hi = int(0), dictionary.shape[0]
        key = keys[i]
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if dictionary[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo == dictionary.shape[0] or dictionary[lo] != key:
            wp.atomic_or(status, 0, 1)
        else:
            identities[i] = original[lo]
            indices[i] = i


@wp.kernel(enable_backward=False)
def _selection_count(source: wp.array[int], status: wp.array[int], count: wp.array[int]):
    count[0] = source[0]
    if status[0] != 0:
        count[0] = 0


@wp.kernel
def _selection_packet(
    ids: wp.array[wp.uint64],
    count: wp.array[int],
    status: wp.array[int],
    query_status: wp.array[int],
    dropped: wp.array[wp.int64],
    legacy: int,
    material: int,
    packet: wp.array[wp.int64],
):
    n = count[0]
    flag = status[0] | query_status[0]
    if n < 0 or n > ids.shape[0]:
        flag = flag | 1
        n = 0
    lo, hi = int(0), n + 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if ids[mid] < wp.uint64(legacy):
            lo = mid + 1
        else:
            hi = mid
    packet[0] = wp.int64(flag)
    packet[1] = wp.int64(lo)
    packet[2] = wp.int64(n - lo)
    packet[3] = dropped[0]
    lo, hi = int(0), n + 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if ids[mid] < wp.uint64(material):
            lo = mid + 1
        else:
            hi = mid
    packet[4] = wp.int64(lo)


def _corners(lower, upper):
    return np.array([np.where(np.array([i & 1, i & 2, i & 4]) != 0, upper, lower) for i in range(8)])


def _transform_points(transform, points):
    """Evaluate stored binary transforms in FP64 for static radius construction."""
    t, v, w = transform[:3].astype(float), transform[3:6].astype(float), float(transform[6])
    return t + (2 * w * w - 1) * points + 2 * (points @ v)[:, None] * v + 2 * w * np.cross(v, points)


class _ContactSpatial:
    """Build swept candidates from the existing directed sampling owner.

    Construction and begin_step snapshot the physical-step start. Updates query q0→q1,
    vertex segments and COM segments. Trials consume the resulting frozen IDs;
    they can miss contacts outside these envelopes, without automatic repair.
    ``law_margin`` is an explicit experimental length, not a qualified default.
    The solver lifecycle owner controls refresh/freeze and rollback.
    """

    def __init__(self, batch, kinematics, q0, x0, poses0, *, rigid_com=None, rigid_rotation=None, capacity, law_margin):
        c, m, k = batch.contact, batch.contact.model, kinematics
        if (
            k.model is not m
            or k.dtype != c.dtype
            or k.count != c.nq
            or k.bodies != c.body_count
            or not np.isfinite(law_margin)
            or law_margin < 0
        ):
            raise ValueError("Invalid spatial contact model or law margin")
        self.batch, self.kinematics, self.device, self.capacity = batch, k, m.device, capacity
        self.law_margin = float(law_margin)
        if rigid_com is None:
            rigid_com = batch._empty_vector
        for a, dtype, shape in (
            (q0, k.dtype, (k.count,)),
            (x0, c.vector, (m.particle_count,)),
            (poses0, c.transform, (k.bodies,)),
            (rigid_com, c.vector, (batch.rigid_count,)),
        ):
            if (
                not isinstance(a, wp.array)
                or (a.dtype, a.shape, a.device) != (dtype, shape, m.device)
                or not a.is_contiguous
            ):
                raise ValueError("Invalid native spatial contact start")
        fields = [getattr(batch.all_samples, n).numpy() for n in ("sources", "targets", "faces", "slots")]
        sources, targets, faces, _slots = fields
        material = c._materials is not None
        slots_per_patch = 1 if material else 3
        sample_stride = int(max(_slots, default=0)) + 1
        if material:
            fields[2] = faces * sample_stride + _slots
            fields[3] = np.zeros_like(_slots)
            faces = fields[2]
        unique_faces = sorted(set(zip(sources.tolist(), faces.tolist(), strict=True)))
        unique_targets = sorted(set(targets.tolist()))
        self.face_index = {key: i for i, key in enumerate(unique_faces)}
        target_index = {key: i for i, key in enumerate(unique_targets)}
        allowed = sorted(set(zip(sources.tolist(), targets.tolist(), strict=True)))
        keys = np.array(
            [
                (self.face_index[s, f] * len(unique_targets) + target_index[t]) * slots_per_patch + slot
                for s, t, f, slot in zip(*fields, strict=True)
            ],
            dtype=np.uint64,
        )
        if len(np.unique(keys)) != c.count:
            raise ValueError("Duplicate spatial sampling identity")
        order = np.argsort(keys)
        self.dictionary = wp.array(keys[order], dtype=wp.uint64, device=self.device)
        self.original = wp.array(order.astype(np.uint64), dtype=wp.uint64, device=self.device)
        self.q0, self.x0, self.poses0 = wp.clone(q0), wp.clone(x0), wp.clone(poses0)
        self.com0 = wp.clone(rigid_com)
        self.status, self.count = (wp.zeros(1, dtype=int, device=self.device) for _ in range(2))
        self.packet = wp.zeros(5, dtype=wp.int64, device=self.device)
        self.mode_counts = None
        kinds, indices, body_com = (getattr(batch.support, n).numpy() for n in ("kinds", "indices", "com"))
        transforms, bodies, types, scales = (getattr(c, n).numpy() for n in ("transforms", "bodies", "types", "scales"))
        local_lo, local_hi, source_shapes, source_faces = [], [], [], []
        surface = {}
        bound_triangles = c.triangles.numpy().tolist()
        face_reach, face_material = [], []
        for source, ordinal in unique_faces:
            face = ordinal // sample_stride if material else ordinal
            source_shapes.append(max(-1, source))
            face_material.append(face if source == -3 else -1)
            reach = 0.0
            bound_face = face
            if source == -3:
                edge = c.actor_samples.cable_edges[face]
                bound_face = len(bound_triangles)
                bound_triangles.append([edge[0], edge[1], edge[1]])
                reach = np.linalg.norm(c.actor_samples.cable_coordinates[face, 1:])
                reach = np.nextafter(reach * (1 + 128 * np.finfo(c.numpy_dtype).eps), np.inf)
            source_faces.append(bound_face)
            face_reach.append(reach)
            if source < 0:
                lo, hi = np.zeros(3), np.zeros(3)
            else:
                if source not in surface:
                    p, _, _, tri = _surface_samples(
                        types[source], scales[source].astype(float), m.shape_source[source], _return_triangles=True
                    )
                    surface[source] = (p.reshape(-1, 3, 3), tri)
                p, tri = surface[source]
                # Include both chord vertices and the actual native projected
                # samples; sphere barycentres lie outside their chord triangle.
                native = p[face].astype(c.numpy_dtype).astype(float)
                vertices = np.concatenate((tri[face], p[face], native))
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            local_lo.append(lo)
            local_hi.append(hi)
        # Mesh targets use SDF domains and negative boundary extension, never
        # the source mesh vertices. Smooth fields additionally bound their jets.
        bounds = _CandidateBounds(m, (), unique_targets, ())
        target_lo = bounds.local_lo.numpy()[unique_targets].astype(float)
        target_hi = bounds.local_hi.numpy()[unique_targets].astype(float)
        if c.smooth_sdf is not None:
            descriptors, sdf_indices = c.table.numpy(), c.sdf_indices.numpy()
            for ordinal, target in enumerate(unique_targets):
                if types[target] != GeoType.MESH:
                    continue
                index = int(sdf_indices[target])
                field = c.smooth_sdf.fields[index]
                coefficients = field.data.coefficients.numpy().astype(float)
                if field.representation == "c2-bspline":
                    minimum = min(0.0, float(coefficients.min()))
                else:
                    # On [0,1], endpoint Hermite bases <=1 and derivative
                    # bases <=4/27. Sum absolute jet bounds for all 8 corners.
                    jets = np.abs(coefficients.reshape(-1, 8)).max(axis=0)
                    dx = 1 / np.array(field.data.inv_dx, dtype=float)
                    factors = np.array(
                        [np.prod([4 / 27 * dx[a] for a in range(3) if jet & (1 << a)]) for jet in range(8)]
                    )
                    minimum = -float(8 * (jets @ factors))
                u = np.finfo(c.numpy_dtype).eps / 2
                # <=4096 value-evaluation operations, including basis weights.
                magnitude = max(abs(minimum), float(np.max(np.abs(coefficients))))
                extension = -minimum + (4096 * u / (1 - 4096 * u)) * magnitude
                scale = 1.0 if descriptors[index]["scale_baked"] else float(scales[target, 0])
                target_lo[ordinal] = (np.array(field.data.lower) - extension) * scale
                target_hi[ordinal] = (np.array(field.data.upper) + extension) * scale
        u = np.finfo(c.numpy_dtype).eps / 2
        gamma = 128 * u / (1 - 128 * u)
        self.gamma = wp.float64(gamma)
        self.tiny = wp.float64(128 * np.finfo(c.numpy_dtype).tiny)
        arm_radii = np.zeros(k.bodies)

        def geometry(prefix, shapes, face_ids, lo, hi, reach):
            radii = []
            for shape, a, b in zip(shapes, lo, hi, strict=True):
                radius = 0.0
                if shape >= 0:
                    body = int(bodies[shape])
                    points = _transform_points(transforms[shape], _corners(a, b))
                    local_radius = float(_length_upper(_corners(a, b)).max())
                    operands = local_radius + float(_length_upper(transforms[shape : shape + 1, :3]).max())
                    if body >= 0 and kinds[body] == 1:
                        points -= body_com[body]
                        operands += float(_length_upper(body_com[body : body + 1]).max())
                    radius = float(_length_upper(points).max())
                    # Account for all rotations accepted by the existing
                    # quaternion norm contract, plus native point arithmetic.
                    factor = 2 * (1 + c.quaternion_norm_tolerance) ** 2 - 1
                    # For a nonunit accepted quaternion, composed rotation
                    # differs from successive rotations by <=(factor-1)|p|.
                    radius = np.nextafter(
                        (radius + gamma * operands + float(self.tiny)) * factor + (factor - 1) * local_radius,
                        np.inf,
                    )
                    if body >= 0 and kinds[body] == 0:
                        arm_radii[indices[body]] = max(arm_radii[indices[body]], radius)
                radii.append(radius)
            for name, values, dtype in (
                ("shapes", shapes, int),
                ("faces", face_ids, int),
                ("local_lo", lo, wp.vec3d),
                ("local_hi", hi, wp.vec3d),
                ("radii", radii, wp.float64),
                ("reach", reach, wp.float64),
            ):
                setattr(self, prefix + "_" + name, wp.array(values, dtype=dtype, device=self.device))
            setattr(self, prefix + "_lo", wp.zeros(len(shapes), dtype=wp.vec3d, device=self.device))
            setattr(self, prefix + "_hi", wp.zeros(len(shapes), dtype=wp.vec3d, device=self.device))

        self.bound_triangles = wp.array(
            np.asarray(bound_triangles, dtype=np.int32).reshape(-1, 3), dtype=int, device=self.device
        )
        geometry("face", source_shapes, source_faces, local_lo, local_hi, face_reach)
        self.face_material = wp.array(face_material, dtype=int, device=self.device)
        self.face_radius = wp.array(face_reach, dtype=wp.float64, device=self.device)
        gaps = batch.all_samples.gaps.numpy().astype(float)
        reach = [float(gaps[targets == target].max()) + c.parameters[3] + self.law_margin for target in unique_targets]
        geometry("target", unique_targets, np.zeros(len(unique_targets), np.int32), target_lo, target_hi, reach)
        self.sweep = _ArticulationSweep(k, arm_radii)
        self._bounds(q0, x0, self.com0)
        if self.status.numpy()[0] or self.sweep.status.numpy()[0]:
            raise ValueError("Invalid initial contact envelopes")
        query_faces = unique_faces
        if material:
            query_faces = [(i, 0) for i in range(len(unique_faces))]
            allowed = sorted({(self.face_index[s, f], int(t)) for s, t, f in zip(sources, targets, faces, strict=True)})
        self.query = _ContactQuery(
            query_faces,
            unique_targets,
            allowed,
            planes=[t for t in unique_targets if types[t] == GeoType.PLANE],
            capacity=capacity,
            device=self.device,
            initial_bounds=(self.face_lo, self.face_hi, self.target_lo, self.target_hi),
            _samples_per_patch=slots_per_patch,
        )
        self._ids, self._scratch_ids = (wp.empty(capacity, dtype=wp.uint64, device=self.device) for _ in range(2))
        self._indices, self._scratch_indices = (wp.empty(capacity, dtype=int, device=self.device) for _ in range(2))
        self.identities = self._scratch_ids if (max(1, capacity) - 1).bit_length() % 2 else self._ids
        self._owned = {name: a for name, a in vars(self).items() if isinstance(a, wp.array)}
        self._metadata = (batch.all_samples, batch.support)
        if batch.material_surface is not None:
            self._metadata += (batch.material_surface,)
        self._borrowed = tuple((c, name, getattr(c, name)) for name in batch.METADATA_ARRAYS)
        self._borrowed += tuple(
            (owner, name, a) for owner in self._metadata for name, a in vars(owner).items() if isinstance(a, wp.array)
        )

    def buffers(self):
        return tuple(self._owned.values()) + self.sweep.buffers() + self.query.buffers()

    def validate_storage(self):
        metadata = (self.batch.all_samples, self.batch.support)
        if self.batch.material_surface is not None:
            metadata += (self.batch.material_surface,)
        if (
            metadata != self._metadata
            or any(getattr(self, name) is not a for name, a in self._owned.items())
            or any(getattr(owner, name) is not a for owner, name, a in self._borrowed)
        ):
            raise ValueError("Spatial contact storage changed")
        self.query.validate_storage()
        self.sweep.validate_storage()

    def begin_step(self, q, x, poses, *, rigid_com=None):
        """Overwrite fixed physical-start snapshots; never reset them on Newton refresh."""
        self.validate_storage()
        com = self.com0 if rigid_com is None else rigid_com
        pairs = ((q, self.q0), (x, self.x0), (poses, self.poses0), (com, self.com0))
        for source, target in pairs:
            if (
                not isinstance(source, wp.array)
                or (source.dtype, source.shape, source.device) != (target.dtype, target.shape, self.device)
                or not source.is_contiguous
            ):
                raise ValueError("Invalid native spatial contact start")
            if source.size and any(
                b.size and source.ptr < b.ptr + b.capacity and b.ptr < source.ptr + source.capacity
                for b in self.buffers()
                if b is not source
            ):
                raise ValueError("Spatial contact start aliases owned storage")
        for source, target in pairs:
            wp.copy(target, source)
        self.count.zero_()
        self.status.zero_()
        self.mode_counts = None

    def _bounds(self, q, x, com):
        self.status.zero_()
        self.sweep.update(self.q0, q)
        surface = self.batch.material_surface
        if surface is not None:
            wp.launch(
                _material_reach,
                self.face_material.size,
                [
                    self.face_material,
                    self.face_radius,
                    surface.edges,
                    self.x0,
                    x,
                    surface.anchor_d,
                    self.gamma,
                    self.face_reach,
                    self.status,
                ],
                device=self.device,
            )
        for prefix in ("face", "target"):
            data = [
                getattr(self, prefix + "_" + n) for n in ("shapes", "faces", "local_lo", "local_hi", "radii", "reach")
            ]
            c, s = self.batch.contact, self.batch.support
            wp.launch(
                _patch_bounds,
                len(data[0]),
                [
                    *data,
                    c.bodies,
                    c.transforms,
                    s.kinds,
                    s.indices,
                    self.bound_triangles,
                    self.x0,
                    x,
                    self.poses0,
                    self.com0,
                    com,
                    self.sweep.displacement,
                    self.sweep.numerical,
                    self.gamma,
                    self.tiny,
                    getattr(self, prefix + "_lo"),
                    getattr(self, prefix + "_hi"),
                    self.status,
                ],
                device=self.device,
            )

    def update(self, q, x, *, rigid_com=None):
        """Overwrite the fixed query and selection, without runtime allocations or host reads."""
        self.validate_storage()
        self.mode_counts = None
        com = self.com0 if rigid_com is None else rigid_com
        for a, original in ((q, self.q0), (x, self.x0), (com, self.com0)):
            if (
                not isinstance(a, wp.array)
                or (a.dtype, a.shape, a.device) != (original.dtype, original.shape, self.device)
                or not a.is_contiguous
            ):
                raise ValueError("Invalid native spatial contact state")
            if a.size and any(
                b.size and a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity
                for b in self.buffers()
                if b is not a
            ):
                raise ValueError("Spatial contact input aliases output")
        self._bounds(q, x, com)
        self.query.update(self.face_lo, self.face_hi, self.target_lo, self.target_hi)
        wp.launch(
            _map_query_keys,
            max(1, self.capacity),
            [
                self.query.keys,
                self.query.count,
                self.query.status,
                self.sweep.status,
                self.dictionary,
                self.original,
                self._ids,
                self._indices,
                self.status,
            ],
            device=self.device,
        )
        wp.launch(_selection_count, 1, [self.query.count, self.status, self.count], device=self.device)
        ids, scratch, indices, scratch_indices = self._ids, self._scratch_ids, self._indices, self._scratch_indices
        run, workers = 1, min(self.capacity, _CONTACT_WORKERS)
        while run < self.capacity:
            wp.launch(
                _merge_contact_keys,
                workers,
                [ids, indices, scratch, scratch_indices, self.count, run, workers],
                device=self.device,
            )
            ids, scratch, indices, scratch_indices = scratch, ids, scratch_indices, indices
            run *= 2

    def check(self, *, warn=True):
        """Freeze host batch counts once with the existing Newton status readback."""
        wp.launch(
            _selection_packet,
            1,
            [
                self.identities,
                self.count,
                self.status,
                self.query.status,
                self.query.dropped,
                self.batch.contact.legacy_count,
                self.batch.contact.legacy_count + self.batch.contact.actor_samples.material_start
                if self.batch.material_surface is not None
                else self.batch.contact.count,
                self.packet,
            ],
            device=self.device,
        )
        flags, left, right, dropped, middle = map(int, self.packet.numpy())
        self.mode_counts = (left, right) if flags == 0 else (0, 0)
        if self.batch.material_surface is not None:
            self.mode_counts = (left, middle - left, left + right - middle) if flags == 0 else (0, 0, 0)
        if flags:
            raise ValueError("Invalid spatial contact envelope or identity")
        if dropped and warn:
            warnings.warn(f"Contact query capacity dropped {dropped} candidates", RuntimeWarning, stacklevel=2)
        return dropped
