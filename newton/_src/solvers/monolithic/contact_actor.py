# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native directed actor geometry and force projection; local laws live in contact."""

from typing import Any

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...geometry.sdf_texture import TextureSDFData
from .articulation import _packed_column
from .contact import _failure
from .contact_geometry import _shape64
from .rigid import _rotation_step
from .smooth_sdf import smooth_query


@wp.func
def _body_pose(
    body: int,
    kinds: wp.array[int],
    indices: wp.array[int],
    body_com: wp.array[Any],
    poses: wp.array[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
):
    result = type(poses[0])(type(rigid_com[0])(0.0), type(rigid_rotation[0])(0.0, 0.0, 0.0, 1.0))
    if body >= 0:
        local = indices[body]
        if kinds[body] == 1:
            rotation = rigid_rotation[local]
            result = type(poses[0])(rigid_com[local] - wp.quat_rotate(rotation, body_com[body]), rotation)
        else:
            result = poses[local]
    return result


@wp.func
def _body_step(
    body: int,
    kinds: wp.array[int],
    indices: wp.array[int],
    body_com: wp.array[Any],
    rotation: Any,
    link_translation_steps: wp.array[Any],
    link_rotation_steps: wp.array[Any],
    rigid_steps: wp.array[Any],
    rigid_rotation_steps: wp.array[Any],
):
    """Return the body frame origin step and ``E`` with ``R = (I + E) R_n``."""
    zero = type(body_com[0])(0.0)
    translation, e = zero, wp.outer(zero, zero)
    if body >= 0:
        local = indices[body]
        if kinds[body] == 1:
            # The origin is c - R c_local, and R - R_n = -E^T R.
            e = _rotation_step(rigid_rotation_steps[local])
            translation = rigid_steps[local] + wp.transpose(e) * (rotation * body_com[body])
        else:
            translation = link_translation_steps[local]
            e = link_rotation_steps[local]
    return translation, e


@wp.func
def _center(body: int, kinds: wp.array[int], indices: wp.array[int], com: wp.array[Any], rigid_com: wp.array[Any]):
    local = indices[body]
    result = com[0]
    if kinds[body] == 1:
        result = rigid_com[local]
    else:
        result = com[local]
    return result


@wp.func
def _point_column(
    body: int,
    column: int,
    point: Any,
    kinds: wp.array[int],
    indices: wp.array[int],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
):
    result = type(point)(0.0)
    local = indices[body]
    if kinds[body] == 0:
        result = _packed_column(jt, column) + wp.cross(_packed_column(jw, column), point - com[local])
    else:
        basis = type(point)(0.0)
        basis[column % 3] = type(point[0])(1.0)
        result = basis
        if column >= 3:
            result = wp.cross(basis, point - rigid_com[local])
    return result


@wp.kernel
def _actor_geometry(
    x: wp.array[Any],
    poses: wp.array[Any],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
    x_steps: wp.array[Any],
    link_translation_steps: wp.array[Any],
    link_rotation_steps: wp.array[Any],
    rigid_steps: wp.array[Any],
    rigid_rotation_steps: wp.array[Any],
    surface_steps: wp.array[Any],
    start_y: wp.array[Any],
    incremental: bool,
    body_com: wp.array[Any],
    kinds: wp.array[int],
    indices: wp.array[int],
    bodies: wp.array[int],
    transforms: wp.array[Any],
    source_shapes: wp.array[int],
    target_shapes: wp.array[int],
    faces: wp.array[int],
    slots: wp.array[int],
    local_points: wp.array[Any],
    local_normals: wp.array[Any],
    triangles: wp.array2d[int],
    types: wp.array[int],
    scales: wp.array[Any],
    sdf_indices: wp.array[int],
    table: wp.array[TextureSDFData],
    smooth_table: wp.array[Any],
    support_offsets: wp.array[int],
    selectors: wp.array[wp.vec2i],
    jacobians: wp.array[Any],
    key_offset: int,
    quaternion_tolerance: Any,
    out: Any,
    world_points: wp.array[Any],
    flags: wp.array[int],
    project: bool,
    batch_count: wp.array[int],
):
    key = wp.tid()
    if not project and key >= batch_count[0]:
        return
    source, target = source_shapes[key], target_shapes[key]
    point, normal = type(x[0])(0.0), type(x[0])(0.0)
    # The point's step from the step start, from increments only.
    point_step = type(x[0])(0.0)
    if source == -3:
        point = local_points[key] + type(point)(0.0)
        normal = local_normals[key] + type(normal)(0.0)
        point_step = surface_steps[faces[key]]
    elif source < 0:
        face, quadrature = faces[key], slots[key] % 3
        for node in range(3):
            bary = type(x[0][0])(1.0) / type(x[0][0])(6.0)
            if node == quadrature:
                bary = type(x[0][0])(2.0) / type(x[0][0])(3.0)
            point += bary * x[triangles[face, node]]
            point_step += bary * x_steps[triangles[face, node]]
        normal = wp.cross(x[triangles[face, 1]] - x[triangles[face, 0]], x[triangles[face, 2]] - x[triangles[face, 0]])
        if source == -2:
            normal *= local_normals[key][2]
    else:
        source_body = _body_pose(bodies[source], kinds, indices, body_com, poses, rigid_com, rigid_rotation)
        pose = source_body * transforms[source]
        point = wp.transform_point(pose, local_points[key])
        normal = wp.transform_vector(pose, local_normals[key])
        origin, e = _body_step(
            bodies[source],
            kinds,
            indices,
            body_com,
            wp.quat_to_matrix(wp.transform_get_rotation(source_body)),
            link_translation_steps,
            link_rotation_steps,
            rigid_steps,
            rigid_rotation_steps,
        )
        # A body point o + R w moves by the origin step and (R - R_n) w = -E^T R w.
        point_step = origin - wp.transpose(e) * (point - wp.transform_get_translation(source_body))
    length = wp.length(normal)
    if not wp.isfinite(length) or length <= type(x[0][0])(0.0):
        _failure(flags, key + key_offset, 2)
        return
    target_body = _body_pose(bodies[target], kinds, indices, body_com, poses, rigid_com, rigid_rotation)
    target_pose = target_body * transforms[target]
    q = wp.transform_get_rotation(target_pose)
    if not wp.isfinite(wp.length(q)) or wp.abs(wp.length(q) - type(x[0][0])(1.0)) > quaternion_tolerance:
        _failure(flags, key + key_offset, 2)
        return
    rotation = wp.quat_to_matrix(q)
    relative = point - wp.transform_get_translation(target_pose)
    target_origin, e = _body_step(
        bodies[target],
        kinds,
        indices,
        body_com,
        wp.quat_to_matrix(wp.transform_get_rotation(target_body)),
        link_translation_steps,
        link_rotation_steps,
        rigid_steps,
        rigid_rotation_steps,
    )
    # The shape frame origin moves with its body; its rotation step is the body's.
    target_step = target_origin - wp.transpose(e) * (
        wp.transform_get_translation(target_pose) - wp.transform_get_translation(target_body)
    )
    # y - y_n = R^T ((I + E)(dp - dt) - E (p - t)): the relative step is a
    # difference of increments and E multiplies only the relative position.
    step = point_step - target_step
    slip = wp.transpose(rotation) * (step + e * step - e * relative)
    # After the step start the local point is the step-start point plus the
    # slip, so the gap reads the increments; absolute positions round only y_n,
    # which is fixed within the step.
    y = wp.transpose(rotation) * relative
    if incremental:
        y = start_y[key] + slip
    d, g = _shape64(types[target], scales[target], y, sdf_indices[target], table)
    out.uncertified[key] = 0
    if types[target] == GeoType.MESH and smooth_table.shape[0] > 0:
        index = sdf_indices[target]
        field = smooth_table[index]
        query = y
        if not table[index].scale_baked:
            query /= scales[target][0]
        d, g, _unused_hessian = smooth_query(field, query)
        if not table[index].scale_baked:
            d *= scales[target][0]
        for axis in range(3):
            if query[axis] <= field.lower[axis] or query[axis] >= field.upper[axis]:
                out.uncertified[key] = 1
    if not wp.isfinite(d) or not wp.isfinite(wp.length(g)) or not wp.isfinite(wp.length(point)):
        _failure(flags, key + key_offset, 1)
        return
    world_points[key] = point
    out.y[key] = y
    out.slip[key] = slip
    out.rounding[key] = wp.abs(g[0] * y[0]) + wp.abs(g[1] * y[1]) + wp.abs(g[2] * y[2])
    out.gradient[key] = g
    out.distance[key] = d
    out.rotation[key] = rotation
    out.soft_normal[key] = wp.transpose(rotation) * normal / length
    if not project:
        return
    for component in range(3):
        factor = 3 * (key + key_offset) + component
        for index in range(support_offsets[factor], support_offsets[factor + 1]):
            selector = selectors[index]
            j = selector[1]
            value = type(point)(0.0)
            if selector[0] == 0:
                if source < 0:
                    bary = type(point[0])(1.0) / type(point[0])(6.0)
                    if j / 3 == slots[key]:
                        bary = type(point[0])(2.0) / type(point[0])(3.0)
                    value[j % 3] = bary
                else:
                    value = _point_column(bodies[source], j, point, kinds, indices, com, jt, jw, rigid_com)
            else:
                value = -_point_column(bodies[target], j, point, kinds, indices, com, jt, jw, rigid_com)
            value = wp.transpose(rotation) * value
            if not wp.isfinite(wp.length(value)):
                _failure(flags, key + key_offset, 1)
            jacobians[index] = value


def _validate_actor_targets(contact, samples):
    """Apply the actor target SDF contract before either storage path is allocated."""
    for shape in samples.targets_used:
        if contact.values["shape_type"][shape] == GeoType.MESH:
            index = int(contact.values["_shape_sdf_index"][shape])
            if (
                contact.model.device.is_cpu
                or contact._descriptors is None
                or index < 0
                or index >= len(contact._descriptors)
            ):
                raise ValueError("Actor mesh target requires CUDA texture SDF")
            if contact.smooth_sdf is None or contact.smooth_sdf.fields[index].representation != "c2-bspline":
                raise ValueError("Actor mesh target requires bound C2 SDF")
            scale = contact.values["shape_scale"][shape]
            if not contact._descriptors["scale_baked"][index] and (scale[0] <= 0 or not np.all(scale == scale[0])):
                raise ValueError("Unbaked actor mesh scale must be positive and uniform")
