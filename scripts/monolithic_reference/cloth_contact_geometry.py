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
    for index, i in enumerate(order):
        later = order[index + 1 :]
        later = later[lo[later, 0] <= hi[i, 0] + radius]
        later = later[np.all(lo[later] <= hi[i] + radius, axis=1) & np.all(hi[later] >= lo[i] - radius, axis=1)]
        for j in later:
            if area2[i] <= tolerance**2 or area2[j] <= tolerance**2 or ill_conditioned[i] or ill_conditioned[j]:
                uncertain.append([int(min(i, j)), int(max(i, j))])
                continue
            checked += 1
            distance, hits, coplanar = _triangle_pair(points[i], points[j], tolerance)
            shared = np.intersect1d(faces[i], faces[j])
            if len(shared):
                if len(shared) == 1:
                    hits = [p for p in hits if np.linalg.norm(p - x[shared[0]]) > tolerance]
                elif len(shared) == 2:
                    hits = [p for p in hits if np.linalg.norm(p - _segment_points(p, p, *x[shared])[1]) > tolerance]
            else:
                minimum = min(minimum, distance)
                compression = max(compression, (thickness[i] + thickness[j]) / 2 - distance)
            if hits:
                intersections.append([int(min(i, j)), int(max(i, j))])
                kinds.append("coplanar_overlap_or_touch" if coplanar else "surface_intersection_or_touch")
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
