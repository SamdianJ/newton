# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native scalar/gradient queries, preserving the existing SDF distance extension."""

from typing import Any

import warp as wp

from ...geometry.sdf_texture import SLOT_LINEAR, TextureSDFData
from ...geometry.types import GeoType


@wp.func
def _mesh64(sdf: TextureSDFData, point: Any):
    lo, hi = type(point)(sdf.sdf_box_lower), type(point)(sdf.sdf_box_upper)
    clamped, mask = type(point)(0.0), type(point)(1.0)
    for a in range(3):
        clamped[a] = wp.clamp(point[a], lo[a], hi[a])
        if point[a] < lo[a] or point[a] > hi[a]:
            mask[a] = type(point[0])(0.0)
    inv = type(point)(sdf.inv_sdf_dx)
    raw_f = wp.cw_mul(clamped - lo, inv)
    dims = wp.vec3i(sdf.coarse_texture.width - 1, sdf.coarse_texture.height - 1, sdf.coarse_texture.depth - 1)
    cell, base = wp.vec3i(0), wp.vec3i(0)
    f = type(point)(0.0)
    for a in range(3):
        f[a] = wp.clamp(raw_f[a], type(point[0])(0.0), type(point[0])(dims[a] * sdf.subgrid_size))
        if f[a] != raw_f[a]:
            inv[a] = type(point[0])(0.0)
        cell[a] = wp.clamp(int(wp.floor(f[a])), 0, dims[a] * sdf.subgrid_size - 1)
        base[a] = cell[a] / sdf.subgrid_size
    slot = sdf.subgrid_start_slots[base[0], base[1], base[2]]
    fine = slot < SLOT_LINEAR
    index = base
    t = f / type(point[0])(sdf.subgrid_size) - type(point)(base)
    derivative = inv / type(point[0])(sdf.subgrid_size)
    texture = sdf.coarse_texture
    if fine:
        block = wp.vec3i(
            int(slot & wp.uint32(1023)),
            int((slot >> wp.uint32(10)) & wp.uint32(1023)),
            int((slot >> wp.uint32(20)) & wp.uint32(1023)),
        )
        index = block * (sdf.subgrid_size + 1) + cell - base * sdf.subgrid_size
        t = f - type(point)(cell)
        derivative = inv
        texture = sdf.subgrid_texture
    value, gradient = type(point[0])(0.0), type(point)(0.0)
    for j in range(8):
        bit = wp.vec3i(j % 2, (j / 2) % 2, j / 4)
        # Only exact texel-center addresses are FP32. Query coordinates,
        # weights and derivatives use the candidate precision.
        address = wp.vec3(
            float(index[0] + bit[0]) + 0.5, float(index[1] + bit[1]) + 0.5, float(index[2] + bit[2]) + 0.5
        )
        sample = float(0.0)
        if sdf.paired_samples:
            sample = wp.texture_sample(texture, address, dtype=wp.vec2)[0]
        else:
            sample = wp.texture_sample(texture, address, dtype=float)
        weights, signs = type(point)(0.0), type(point)(0.0)
        for a in range(3):
            signs[a] = type(point[0])(2 * bit[a] - 1)
            weights[a] = type(point[0])(1 - bit[a]) + signs[a] * t[a]
        value += type(point[0])(sample) * weights[0] * weights[1] * weights[2]
        for a in range(3):
            gradient[a] += (
                type(point[0])(sample) * signs[a] * weights[(a + 1) % 3] * weights[(a + 2) % 3] * derivative[a]
            )
    if fine:
        value = value * type(point[0])(sdf.subgrids_sdf_value_range) + type(point[0])(sdf.subgrids_min_sdf_value)
        gradient *= type(point[0])(sdf.subgrids_sdf_value_range)
    delta = point - clamped
    length = wp.length(delta)
    if length > type(point[0])(0.0):
        value += length
        gradient = wp.cw_mul(mask, gradient) + delta / length
    return value, gradient


@wp.func
def _shape64(geo: int, scale: Any, point: Any, index: int, table: wp.array[TextureSDFData]):
    value = point[2]
    gradient = type(point)(0.0, 0.0, 1.0)
    if geo == GeoType.SPHERE:
        length = wp.length(point)
        value = length - scale[0]
        gradient = type(point)(0.0)
        if length > type(point[0])(0.0):
            gradient = point / length
    elif geo == GeoType.BOX:
        d = type(point)(wp.abs(point[0]), wp.abs(point[1]), wp.abs(point[2])) - scale
        outside = type(point)(
            wp.max(d[0], type(point[0])(0.0)), wp.max(d[1], type(point[0])(0.0)), wp.max(d[2], type(point[0])(0.0))
        )
        length = wp.length(outside)
        value = length + wp.min(wp.max(d[0], wp.max(d[1], d[2])), type(point[0])(0.0))
        gradient = type(point)(0.0)
        if length > type(point[0])(0.0):
            gradient = outside / length
        else:
            axis = int(0)
            if d[1] > d[axis]:
                axis = 1
            if d[2] > d[axis]:
                axis = 2
            gradient[axis] = type(point[0])(1.0)
        for a in range(3):
            if point[a] < type(point[0])(0.0):
                gradient[a] = -gradient[a]
    elif geo == GeoType.MESH:
        sdf = table[index]
        query = point
        if not sdf.scale_baked:
            query /= scale[0]
        value, gradient = _mesh64(sdf, query)
        if not sdf.scale_baked:
            value *= scale[0]
    return value, gradient
