# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative candidate bounds for the fixed monolithic contact table."""

import math

import numpy as np
import warp as wp

from ...geometry.sdf_texture import TextureSDFData, _read_cell_corners
from ...geometry.soft_contacts_sdf import _shape_frames
from ...geometry.types import GeoType


@wp.kernel(enable_backward=False)
def _sdf_boundary_lower(
    descriptors: wp.array[TextureSDFData], index: int, size: wp.vec3i, lower: wp.array[float], status: wp.array[int]
):
    i, j, k = wp.tid()
    if i != 0 and j != 0 and k != 0 and i != size[0] - 1 and j != size[1] - 1 and k != size[2] - 1:
        return
    corners, _tx, _ty, _tz = _read_cell_corners(
        descriptors[index], wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)
    )
    for c in range(8):
        v = corners[c]
        if not wp.isfinite(v):
            wp.atomic_max(status, 0, 1)
        # The interpolation is a convex combination, up to float32 rounding.
        wp.atomic_min(lower, 0, v - 32.0 * 1.1920928955078125e-7 * wp.max(1.0, wp.abs(v)))


@wp.func
def _overlap(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    return a[0] <= d[0] and c[0] <= b[0] and a[1] <= d[1] and c[1] <= b[1] and a[2] <= d[2] and c[2] <= b[2]


@wp.kernel(enable_backward=False)
def _update_particles(
    particles: wp.array[int], x: wp.array[wp.vec3], aggregate: wp.array2d[float], status: wp.array[int]
):
    position = x[particles[wp.tid()]]
    if not wp.isfinite(position):
        wp.atomic_max(status, 0, 11)
        return
    for axis in range(3):
        wp.atomic_min(aggregate, 0, axis, position[axis])
        wp.atomic_max(aggregate, 1, axis, position[axis])


@wp.kernel(enable_backward=False)
def _update_faces(
    faces: wp.array[int],
    triangles: wp.array2d[int],
    x: wp.array[wp.vec3],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    status: wp.array[int],
):
    face = faces[wp.tid()]
    a, b, c = x[triangles[face, 0]], x[triangles[face, 1]], x[triangles[face, 2]]
    if not wp.isfinite(a) or not wp.isfinite(b) or not wp.isfinite(c):
        wp.atomic_max(status, 0, 11)
        return
    lo, hi = wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))
    lower[face] = lo
    upper[face] = hi


@wp.kernel(enable_backward=False)
def _update_shapes(
    shapes: wp.array[int],
    bodies: wp.array[int],
    types: wp.array[int],
    transforms: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    local_lo: wp.array[wp.vec3],
    local_hi: wp.array[wp.vec3],
    margin: wp.array[float],
    reach: float,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    aggregate: wp.array2d[float],
    status: wp.array[int],
):
    shape = shapes[wp.tid()]
    _bs, ws, _sw = _shape_frames(bodies, body_q, transforms, shape)
    if not wp.isfinite(wp.transform_get_translation(ws)) or not wp.isfinite(wp.length(wp.transform_get_rotation(ws))):
        wp.atomic_max(status, 0, 11)
        return
    if types[shape] == GeoType.PLANE:
        return
    lo, hi = local_lo[shape], local_hi[shape]
    center, half = (lo + hi) * 0.5, (hi - lo) * 0.5
    world_center = wp.transform_point(ws, center)
    world_half = (
        wp.abs(wp.transform_vector(ws, wp.vec3(half[0], 0.0, 0.0)))
        + wp.abs(wp.transform_vector(ws, wp.vec3(0.0, half[1], 0.0)))
        + wp.abs(wp.transform_vector(ws, wp.vec3(0.0, 0.0, half[2])))
    )
    # Account for forward/inverse transforms, interpolation and contact comparisons.
    slack = 64.0 * 1.1920928955078125e-7 * (wp.vec3(1.0) + wp.abs(world_center) + world_half)
    extension = wp.vec3(reach + margin[shape]) + slack
    a, b = world_center - world_half - extension, world_center + world_half + extension
    if not wp.isfinite(a) or not wp.isfinite(b):
        wp.atomic_max(status, 0, 11)
        return
    lower[shape] = a
    upper[shape] = b
    for axis in range(3):
        wp.atomic_min(aggregate, 2, axis, a[axis])
        wp.atomic_max(aggregate, 3, axis, b[axis])


@wp.kernel(enable_backward=False)
def _reset_aggregate(aggregate: wp.array2d[float]):
    i, j = wp.tid()
    value = float(wp.inf)
    if i == 1 or i == 3:
        value = -value
    aggregate[i, j] = value


@wp.func
def aggregate_overlap(aggregate: wp.array2d[float]):
    return _overlap(
        wp.vec3(aggregate[0, 0], aggregate[0, 1], aggregate[0, 2]),
        wp.vec3(aggregate[1, 0], aggregate[1, 1], aggregate[1, 2]),
        wp.vec3(aggregate[2, 0], aggregate[2, 1], aggregate[2, 2]),
        wp.vec3(aggregate[3, 0], aggregate[3, 1], aggregate[3, 2]),
    )


class _CandidateBounds:
    """Own immutable local bounds and scratch overwritten on every collision."""

    def __init__(self, model, faces, shapes, particles):
        self.model = model
        self.faces = wp.array(faces, dtype=int, device=model.device)
        self.shapes = wp.array(shapes, dtype=int, device=model.device)
        self.particles = wp.array(particles, dtype=int, device=model.device)
        lo = np.zeros((model.shape_count, 3), dtype=np.float64)
        hi = lo.copy()
        types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        descriptors, indices = model._texture_sdf_data.numpy(), model._shape_sdf_index.numpy()
        sdf_bounds = {}
        for shape in shapes:
            geo, scale = types[shape], scales[shape].astype(float)
            half = np.zeros(3)
            if geo == GeoType.PLANE:
                continue
            if geo == GeoType.SPHERE:
                half[:] = scale[0]
            elif geo == GeoType.BOX:
                half[:] = scale
            elif geo in (GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE):
                half[:] = (scale[0], scale[0], scale[1])
                if geo == GeoType.CAPSULE:
                    half[2] += scale[0]
                elif geo == GeoType.CYLINDER and scale[2] > 0:
                    # Match the rounded cylinder SDF, including its clamped radius.
                    radius = max(scale[2], scale[1])
                    half[:2] += scale[1] ** 2 / (radius + np.sqrt(max(0, radius**2 - scale[1] ** 2)))
                lo[shape], hi[shape] = -half, half
            elif geo in (GeoType.MESH, GeoType.CONVEX_MESH):
                index = int(indices[shape])
                d = descriptors[index]
                if index not in sdf_bounds:
                    extent = d["sdf_box_upper"].astype(float) - d["sdf_box_lower"].astype(float)
                    cells = np.maximum(1, np.ceil(extent * d["inv_sdf_dx"].astype(float)))
                    if not np.isfinite(cells).all() or np.any(cells > np.iinfo(np.int32).max):
                        raise ValueError("SDF boundary validation exceeds int32 capacity")
                    size = tuple(int(v) for v in cells)
                    if math.prod(size) > np.iinfo(np.int32).max:
                        raise ValueError("SDF boundary validation exceeds int32 capacity")
                    minimum = wp.zeros(1, dtype=float, device=model.device)
                    status = wp.zeros(1, dtype=int, device=model.device)
                    wp.launch(
                        _sdf_boundary_lower,
                        dim=tuple(int(v) for v in size),
                        inputs=[model._texture_sdf_data, index, wp.vec3i(*size), minimum, status],
                        device=model.device,
                    )
                    if status.numpy()[0]:
                        raise ValueError("Nonfinite SDF boundary corner")
                    extension = -float(minimum.numpy()[0])
                    sdf_bounds[index] = (
                        d["sdf_box_lower"].astype(float) - extension,
                        d["sdf_box_upper"].astype(float) + extension,
                    )
                a, b = sdf_bounds[index]
                multiplier = 1.0 if d["scale_baked"] else scale[0]
                lo[shape], hi[shape] = a * multiplier, b * multiplier
                continue
            lo[shape], hi[shape] = -half, half
        with np.errstate(over="ignore"):
            lo = np.nextafter(lo.astype(np.float32), -np.inf)
            hi = np.nextafter(hi.astype(np.float32), np.inf)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError("Local contact bounds exceed finite float32 range")
        self.local_lo = wp.array(lo, dtype=wp.vec3, device=model.device)
        self.local_hi = wp.array(hi, dtype=wp.vec3, device=model.device)
        self.face_lo = wp.empty(model.tri_count, dtype=wp.vec3, device=model.device)
        self.face_hi = wp.empty_like(self.face_lo)
        self.shape_lo = wp.empty(model.shape_count, dtype=wp.vec3, device=model.device)
        self.shape_hi = wp.empty_like(self.shape_lo)
        self.aggregate = wp.empty((4, 3), dtype=float, device=model.device)

    def update(self, state, reach, status):
        """Refresh scratch without allocations or host reads."""
        m = self.model
        wp.launch(_reset_aggregate, dim=(4, 3), inputs=[self.aggregate], device=m.device)
        wp.launch(
            _update_particles,
            dim=len(self.particles),
            inputs=[self.particles, state.particle_q, self.aggregate, status],
            device=m.device,
        )
        wp.launch(
            _update_faces,
            dim=len(self.faces),
            inputs=[self.faces, m.tri_indices, state.particle_q, self.face_lo, self.face_hi, status],
            device=m.device,
        )
        if len(self.shapes):
            wp.launch(
                _update_shapes,
                dim=len(self.shapes),
                inputs=[
                    self.shapes,
                    m.shape_body,
                    m.shape_type,
                    m.shape_transform,
                    state.body_q,
                    self.local_lo,
                    self.local_hi,
                    m.shape_margin,
                    reach,
                    self.shape_lo,
                    self.shape_hi,
                    self.aggregate,
                    status,
                ],
                device=m.device,
            )
