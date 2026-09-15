# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal contact continuity experiment; disabled in normal solver construction."""

import numpy as np
import warp as wp

from ...geometry.sdf_texture import TextureSDFData, texture_sample_sdf, texture_sample_sdf_at_voxel


@wp.func
def cubic_weights(t: float):
    """Return interpolating Catmull-Rom weights and their derivatives."""
    t2, t3 = t * t, t * t * t
    return (
        wp.vec4(
            -0.5 * t + t2 - 0.5 * t3, 1.0 - 2.5 * t2 + 1.5 * t3, 0.5 * t + 2.0 * t2 - 1.5 * t3, -0.5 * t2 + 0.5 * t3
        ),
        wp.vec4(-0.5 + 2.0 * t - 1.5 * t2, -5.0 * t + 4.5 * t2, 0.5 + 4.0 * t - 4.5 * t2, -t + 1.5 * t2),
    )


@wp.func
def sample_patch(values: wp.array3d[float], coordinate: wp.vec3, inverse_dx: wp.vec3):
    """Evaluate one shared C1 scalar field and its unnormalized gradient."""
    if not wp.isfinite(wp.length(coordinate)):
        return float(0.0), wp.vec3(), False
    cell = wp.vec3i(int(wp.floor(coordinate[0])), int(wp.floor(coordinate[1])), int(wp.floor(coordinate[2])))
    value, gradient = float(0.0), wp.vec3()
    valid = True
    for axis in range(3):
        if cell[axis] < 1 or cell[axis] + 2 >= values.shape[axis]:
            valid = False
    if valid:
        wx, dx = cubic_weights(coordinate[0] - float(cell[0]))
        wy, dy = cubic_weights(coordinate[1] - float(cell[1]))
        wz, dz = cubic_weights(coordinate[2] - float(cell[2]))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    v = values[cell[0] + i - 1, cell[1] + j - 1, cell[2] + k - 1]
                    value += v * wx[i] * wy[j] * wz[k]
                    gradient += v * wp.vec3(dx[i] * wy[j] * wz[k], wx[i] * dy[j] * wz[k], wx[i] * wy[j] * dz[k])
        gradient = wp.cw_mul(gradient, inverse_dx)
        valid = wp.isfinite(value) and wp.isfinite(wp.length(gradient)) and wp.length(gradient) > 1.0e-8
    return value, gradient, valid


@wp.func
def transport_history(old: wp.vec3, old_normal: wp.vec3, normal: wp.vec3, reset_cos: float, retain_cos: float):
    """Contract committed history without changing it during trial evaluation."""
    c = wp.dot(old_normal, normal)
    t = wp.clamp((c - reset_cos) / (retain_cos - reset_cos), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    return weight * (old - wp.dot(normal, old) * normal)


@wp.struct
class CubicField:
    values: wp.array3d[float]
    lower: wp.vec3
    inverse_dx: wp.vec3
    scale_baked: bool


@wp.kernel(enable_backward=False)
def export_nodes(table: wp.array[TextureSDFData], index: int, origin: wp.vec3i, values: wp.array3d[float]):
    i, j, k = wp.tid()
    sdf = table[index]
    node = origin + wp.vec3i(i, j, k)
    p = sdf.sdf_box_lower + wp.cw_div(wp.vec3(float(node[0]), float(node[1]), float(node[2])), sdf.inv_sdf_dx)
    outside = False
    for a in range(3):
        if p[a] < sdf.sdf_box_lower[a] or p[a] > sdf.sdf_box_upper[a]:
            outside = True
    if outside:
        values[i, j, k] = texture_sample_sdf(sdf, p)
    else:
        values[i, j, k] = texture_sample_sdf_at_voxel(sdf, node[0], node[1], node[2])


@wp.func
def query_field(field: CubicField, position: wp.vec3, scale: wp.vec3):
    """Query the certified contact domain; outside it no contact is emitted."""
    distance_scale = float(1.0)
    if not field.scale_baked:
        position = wp.cw_div(position, scale)
        distance_scale = scale[0]  # Monolithic mesh validation requires positive uniform scale.
    coordinate = wp.cw_mul(position - field.lower, field.inverse_dx)
    interior = True
    for a in range(3):
        if coordinate[a] < 1.0 or coordinate[a] >= float(field.values.shape[a] - 2):
            interior = False
    if not interior:
        # Setup certifies that the entire outer layer is beyond the detection
        # shell. These sentinels are not used as a distance or normal force.
        return float(1.0e10), wp.vec3(1.0, 0.0, 0.0)
    phi, gradient, valid = sample_patch(field.values, coordinate, field.inverse_dx)
    if not valid:
        gradient = wp.vec3()  # The existing collision failure path rejects this.
    return phi * distance_scale, gradient


def build_fields(model, shapes, shell):
    """Preallocate fixed C1 fields and conservatively certify their outer layer."""
    descriptors = model._texture_sdf_data.numpy()
    indices = model._shape_sdf_index.numpy()
    scales, margins = model.shape_scale.numpy(), model.shape_margin.numpy()
    fields = [CubicField() for _ in descriptors]
    storage, evidence = [], []
    padding = 16
    for index in sorted({int(indices[s]) for s in shapes if indices[s] >= 0}):
        d = descriptors[index]
        inv = d["inv_sdf_dx"].astype(float)
        dimensions = np.rint((d["sdf_box_upper"] - d["sdf_box_lower"]) * inv).astype(int) + 1 + 2 * padding
        grid = wp.empty(tuple(int(n) for n in dimensions), dtype=float, device=model.device)
        wp.launch(
            export_nodes, grid.shape, [model._texture_sdf_data, index, wp.vec3i(-padding), grid], device=model.device
        )
        nodes = grid.numpy().astype(float)
        if not np.isfinite(nodes).all():
            raise ValueError("Nonfinite cubic SDF nodes")
        error = sum(0.125 * 1.25**a * float(np.max(np.abs(np.diff(nodes, n=2, axis=a)))) for a in range(3))
        boundary = min(float(np.min(np.take(nodes, [1, 2, -3, -2], axis=a))) for a in range(3)) - error
        owners = [int(s) for s in shapes if indices[s] == index]
        threshold = max(
            (shell + float(margins[s])) / (1.0 if d["scale_baked"] else float(scales[s, 0])) for s in owners
        )
        if boundary <= threshold:
            raise ValueError(f"Uncertified cubic SDF boundary: {index}: {boundary} <= {threshold}")
        field = fields[index]
        field.values = grid
        field.lower = wp.vec3(*(d["sdf_box_lower"].astype(float) - padding / inv))
        field.inverse_dx = wp.vec3(*inv)
        field.scale_baked = bool(d["scale_baked"])
        storage.append(grid)
        evidence.append(
            {
                "sdf_index": index,
                "shapes": owners,
                "lower": list(field.lower),
                "upper": (d["sdf_box_upper"].astype(float) + padding / inv).tolist(),
                "scale_baked": bool(d["scale_baked"]),
                "bytes": int(grid.size * 4),
                "padding": padding,
                "boundary_lower_bound_m": boundary,
                "shell_m": threshold,
            }
        )
    return wp.array(fields, dtype=CubicField, device=model.device), storage, evidence
