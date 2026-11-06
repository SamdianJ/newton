# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only NumPy triangle geometry, independent of production contact candidates.

This diagnostic checks saved configurations, not motion between configurations.
Distances exclude triangles sharing vertices; intersections beyond their shared
feature are still checked. Coplanar contact is reported without claiming crossing.
"""

import numpy as np


def _segment_points(a, b, c, d):
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = u @ u, v @ v, u @ v
    uw, vw = u @ w, v @ w
    normal = np.cross(u, v)
    denominator = normal @ normal
    s = np.clip(np.cross(-w, v) @ normal / denominator, 0, 1) if denominator > 1e-28 * uu * vv else 0.0
    t = np.clip((uv * s + vw) / vv, 0, 1) if vv > 0 else 0.0
    s = np.clip((uv * t - uw) / uu, 0, 1) if uu > 0 else 0.0
    return a + s * u, c + t * v


def _point_triangle(p, triangle):
    a, b, c = triangle
    u, v = b - a, c - a
    n = np.cross(u, v)
    nn = n @ n
    if nn > 0:
        q = p - n * ((p - a) @ n / nn)
        w = q - a
        s, t = np.cross(w, v) @ n / nn, np.cross(u, w) @ n / nn
        if s >= 0 and t >= 0 and s + t <= 1:
            return q
    candidates = [_segment_points(p, p, triangle[i], triangle[(i + 1) % 3])[1] for i in range(3)]
    return min(candidates, key=lambda q: np.linalg.norm(q - p))


def _triangle_pair(a, b, tolerance):
    points = []
    minimum = np.inf
    coplanar = False
    for face, other in ((a, b), (b, a)):
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        heights = (other - face[0]) @ (normal / length)
        coplanar |= bool(np.max(abs(heights)) <= tolerance)
        for i, p in enumerate(other):
            q = _point_triangle(p, face)
            distance = np.linalg.norm(q - p)
            minimum = min(minimum, distance)
            if distance <= tolerance:
                points.append((p + q) / 2)
            j = (i + 1) % 3
            if heights[i] * heights[j] < 0:
                hit = p + heights[i] / (heights[i] - heights[j]) * (other[j] - p)
                if np.linalg.norm(hit - _point_triangle(hit, face)) <= tolerance:
                    points.append(hit)
                    minimum = 0.0
    for i in range(3):
        for j in range(3):
            p, q = _segment_points(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])
            distance = np.linalg.norm(p - q)
            minimum = min(minimum, distance)
            if distance <= tolerance:
                points.append((p + q) / 2)
    return float(minimum), points, coplanar


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _segment_points_batch(a, b, c, d):
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    u, v, w = b - a, d - c, a - c
    uu, vv, uv, uw, vw = _dot(u, u), _dot(v, v), _dot(u, v), _dot(u, w), _dot(v, w)
    normal = np.cross(u, v)
    denominator = _dot(normal, normal)
    s = np.clip(
        np.divide(
            _dot(np.cross(-w, v), normal),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-28 * uu * vv,
        ),
        0,
        1,
    )
    t = np.clip(np.divide(uv * s + vw, vv, out=np.zeros_like(vv), where=vv > 0), 0, 1)
    s = np.clip(np.divide(uv * t - uw, uu, out=np.zeros_like(uu), where=uu > 0), 0, 1)
    return a + s[..., None] * u, c + t[..., None] * v


def _point_triangle_batch(p, triangle):
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    u, v = b - a, c - a
    normal = np.cross(u, v)
    nn = _dot(normal, normal)
    q = p - normal * np.divide(_dot(p - a, normal), nn, out=np.zeros_like(nn), where=nn > 0)[:, None]
    w = q - a
    s = np.divide(_dot(np.cross(w, v), normal), nn, out=np.zeros_like(nn), where=nn > 0)
    t = np.divide(_dot(np.cross(u, w), normal), nn, out=np.zeros_like(nn), where=nn > 0)
    inside = (nn > 0) & (s >= 0) & (t >= 0) & (s + t <= 1)
    # The scalar oracle uses the first minimum on ties; argmin preserves that rule.
    edge_points = np.stack(
        [_segment_points_batch(p, p, triangle[:, i], triangle[:, (i + 1) % 3])[1] for i in range(3)], axis=1
    )
    closest = np.argmin(np.linalg.norm(edge_points - p[:, None], axis=2), axis=1)
    return np.where(inside[:, None], q, edge_points[np.arange(len(p)), closest])


def _triangle_pairs(a, b, tolerance):
    """Evaluate the scalar oracle's same 6 VF, 6 crossing and 9 EE witnesses in bulk."""
    minimum = np.full(len(a), np.inf)
    coplanar = np.zeros(len(a), dtype=bool)
    points, masks, distances = [], [], []
    for face, other in ((a, b), (b, a)):
        normal = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        length = np.linalg.norm(normal, axis=1)
        heights = _dot(other - face[:, None, 0], (normal / length[:, None])[:, None])
        coplanar |= np.max(abs(heights), axis=1) <= tolerance
        for i in range(3):
            p = other[:, i]
            q = _point_triangle_batch(p, face)
            distance = np.linalg.norm(q - p, axis=1)
            minimum = np.minimum(minimum, distance)
            points.append((p + q) / 2)
            masks.append(distance <= tolerance)
            distances.append(distance)
            j = (i + 1) % 3
            crossing = heights[:, i] * heights[:, j] < 0
            fraction = np.divide(heights[:, i], heights[:, i] - heights[:, j], out=np.zeros(len(a)), where=crossing)
            hit = p + fraction[:, None] * (other[:, j] - p)
            distance = np.linalg.norm(hit - _point_triangle_batch(hit, face), axis=1)
            mask = crossing & (distance <= tolerance)
            points.append(hit)
            masks.append(mask)
            distances.append(np.where(crossing, distance, np.inf))
            minimum[mask] = 0.0
    for i in range(3):
        for j in range(3):
            p, q = _segment_points_batch(a[:, i], a[:, (i + 1) % 3], b[:, j], b[:, (j + 1) % 3])
            distance = np.linalg.norm(p - q, axis=1)
            minimum = np.minimum(minimum, distance)
            points.append((p + q) / 2)
            masks.append(distance <= tolerance)
            distances.append(distance)
    # Near the tolerance boundary, use the retained scalar operation order.
    rounding = 64 * np.finfo(float).eps * np.maximum(np.max(abs(a), axis=(1, 2)), np.max(abs(b), axis=(1, 2)))
    sensitive = np.any(abs(np.stack(distances, axis=1) - tolerance) <= rounding[:, None], axis=1)
    return minimum, np.stack(points, axis=1), np.stack(masks, axis=1), coplanar, sensitive, rounding


def inspect_geometry(positions, triangles, owners, thicknesses, *, search_distance=0.01, tolerance=1e-9):
    """Measure nearby nonadjacent layers and surface intersections in SI units.

    Unknown distances outside the search radius remain null, not zero. AABB
    candidates are independently rebuilt from all faces. No result alters state.
    """
    x = np.asarray(positions, dtype=float)
    faces = np.asarray(triangles)
    owner, thickness = np.asarray(owners), np.asarray(thicknesses, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or not np.isfinite(x).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype.kind not in "iu"
        or np.any(faces < 0)
        or np.any(faces >= len(x))
        or owner.shape != (len(faces),)
        or owner.dtype.kind not in "iu"
        or thickness.shape != (len(faces),)
        or not np.isfinite(thickness).all()
        or np.any(thickness <= 0)
        or not np.isfinite(search_distance)
        or search_distance <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Require finite SI geometry, valid triangle/owner indices and positive thickness/radius")
    points = x[faces]
    area2 = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    degenerate = np.flatnonzero(area2 <= tolerance**2)
    edge2 = np.max(np.sum((points - np.roll(points, 1, axis=1)) ** 2, axis=2), axis=1)
    ill_conditioned = area2 <= np.sqrt(np.finfo(float).eps) * edge2
    lo, hi = points.min(axis=1), points.max(axis=1)
    radius = max(search_distance, float(thickness.max(initial=0)))
    minimum, compression, checked = np.inf, 0.0, 0
    intersections, kinds, uncertain = [], [], []
    order = np.argsort(lo[:, 0], kind="stable")
    ordered_x = lo[order, 0]
    candidates = []
    for index, i in enumerate(order):
        stop = np.searchsorted(ordered_x, hi[i, 0] + radius, side="right")
        later = order[index + 1 : stop]
        later = later[np.all(lo[later] <= hi[i] + radius, axis=1) & np.all(hi[later] >= lo[i] - radius, axis=1)]
        if len(later):
            candidates.append(np.column_stack((np.full(len(later), i), later)))
    candidates = np.concatenate(candidates) if candidates else np.empty((0, 2), dtype=int)
    for start in range(0, len(candidates), 4096):
        batch = candidates[start : start + 4096]
        bad = np.any((area2[batch] <= tolerance**2) | ill_conditioned[batch], axis=1)
        uncertain.extend(np.sort(batch[bad], axis=1).tolist())
        batch = batch[~bad]
        if not len(batch):
            continue
        checked += len(batch)
        i, j = batch.T
        distance, hits, hit_mask, coplanar, sensitive, rounding = _triangle_pairs(points[i], points[j], tolerance)
        shared = np.any(faces[i, :, None] == faces[j, None, :], axis=2)
        count = shared.sum(axis=1)
        shared_ids = np.sort(np.where(shared, faces[i], len(x)), axis=1)
        first, second = x[np.minimum(shared_ids[:, 0], len(x) - 1)], x[np.minimum(shared_ids[:, 1], len(x) - 1)]
        shared_distance = np.linalg.norm(hits - first[:, None], axis=2)
        edge = count == 2
        if np.any(edge):
            projected = _segment_points_batch(hits[edge], hits[edge], first[edge, None], second[edge, None])[1]
            shared_distance[edge] = np.linalg.norm(hits[edge] - projected, axis=2)
        filtered = (count == 1) | edge
        sensitive |= filtered & np.any(hit_mask & (abs(shared_distance - tolerance) <= rounding[:, None]), axis=1)
        hit_mask &= ~filtered[:, None] | (shared_distance > tolerance)
        events = np.any(hit_mask, axis=1)
        for k in np.flatnonzero(sensitive):
            distance[k], scalar_hits, coplanar[k] = _triangle_pair(points[i[k]], points[j[k]], tolerance)
            ids = shared_ids[k, : count[k]]
            if count[k] == 1:
                scalar_hits = [p for p in scalar_hits if np.linalg.norm(p - x[ids[0]]) > tolerance]
            elif count[k] == 2:
                scalar_hits = [
                    p for p in scalar_hits if np.linalg.norm(p - _segment_points(p, p, *x[ids])[1]) > tolerance
                ]
            events[k] = bool(scalar_hits)
        unshared = count == 0
        minimum = min(minimum, float(np.min(distance[unshared], initial=np.inf)))
        compression = max(
            compression,
            float(np.max((thickness[i[unshared]] + thickness[j[unshared]]) / 2 - distance[unshared], initial=0)),
        )
        intersections.extend(np.sort(batch[events], axis=1).tolist())
        kinds.extend(np.where(coplanar[events], "coplanar_overlap_or_touch", "surface_intersection_or_touch").tolist())
    return {
        "coverage": "published_configuration_only",
        "continuous_check": False,
        "distance_search_radius_m": radius,
        "minimum_distance_m": minimum if minimum <= radius and not uncertain else None,
        "minimum_observed_distance_m": minimum if minimum <= radius else None,
        "maximum_thickness_compression_m": float(compression) if not uncertain else None,
        "maximum_observed_thickness_compression_m": float(compression),
        "intersections": intersections,
        "intersection_kinds": kinds,
        "inter_owner_events": sum(owner[i] != owner[j] for i, j in intersections),
        "self_events": sum(owner[i] == owner[j] for i, j in intersections),
        "checked_pairs": checked,
        "degenerate_faces": degenerate.tolist(),
        "ill_conditioned_faces": np.flatnonzero(ill_conditioned).tolist(),
        "uncertain_pairs": uncertain,
        "coverage_complete": not uncertain and not len(degenerate) and not np.any(ill_conditioned),
        "numerical_tolerance_m": tolerance,
        "certified_predicates": False,
    }
