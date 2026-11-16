# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FP64 offline criteria for floating-hand runs, independent of the solver's contact code.

Every collision shape is a convex core plus a radius: a sphere is a point, a
capsule a segment, a box or convex hull a polytope with radius zero. For convex
sets the translation (penetration) depth is exact:

    depth(A, B) = max(0, r_A + r_B - sd(core_A, core_B)),

where ``sd`` is the core distance (GJK) when the cores are disjoint and minus
their SAT overlap when they intersect. Box pairs reproduce the rigid-stack SAT
depth (``run_rigid_stack.box_penetration``).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

import newton

_TOL = 1e-13


@dataclass(frozen=True)
class Convex:
    """Shape-local convex core: vertices, face normals, edge directions and an inflation radius."""

    vertices: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    radius: float = 0.0

    def placed(self, rotation, translation):
        """Return the core in world: rotation is 3x3, translation a 3-vector."""
        rotation = np.asarray(rotation, dtype=np.float64)
        return Convex(
            self.vertices @ rotation.T + np.asarray(translation, dtype=np.float64),
            self.normals @ rotation.T,
            self.edges @ rotation.T,
            self.radius,
        )


_EMPTY = np.zeros((0, 3))


def point(radius=0.0):
    return Convex(np.zeros((1, 3)), _EMPTY, _EMPTY, float(radius))


def segment(half_height, radius=0.0):
    """Capsule core along local z, Newton's convention (scale = radius, half_height)."""
    ends = np.array([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]])
    return Convex(ends, _EMPTY, np.array([[0.0, 0.0, 1.0]]), float(radius))


def box(half_extents):
    h = np.asarray(half_extents, dtype=np.float64)
    if h.shape != (3,) or np.any(h <= 0):
        raise ValueError(f"Box half extents must be three positive values, got {half_extents}")
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64) * h
    return Convex(corners, np.eye(3), np.eye(3))


def hull(points):
    """Convex hull of a point cloud: unique face normals and unique edge directions."""
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=np.float64)
    h = ConvexHull(points)
    vertices = points[h.vertices]
    normals = _unique_directions(h.equations[:, :3])
    edges = []
    for simplex in h.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.append(points[simplex[b]] - points[simplex[a]])
    edges = np.asarray(edges)
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    return Convex(vertices, normals, _unique_directions(edges))


def _unique_directions(directions, tol=1e-9):
    """Keep one of each pair of parallel or anti-parallel unit directions."""
    kept = []
    for direction in directions:
        unit = direction / np.linalg.norm(direction)
        if not any(abs(abs(float(unit @ k)) - 1.0) < tol for k in kept):
            kept.append(unit)
    return np.asarray(kept) if kept else _EMPTY


def _closest_on_simplex(simplex):
    """Point of the simplex's convex hull nearest the origin, and the vertices that support it."""
    best, best_index = None, None
    n = len(simplex)
    for size in range(1, n + 1):
        for index in combinations(range(n), size):
            q = simplex[list(index)]
            if size == 1:
                x, weights = q[0], np.ones(1)
            else:
                gram = q @ q.T
                kkt = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
                rhs = np.r_[np.zeros(size), 1.0]
                solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
                weights = solution[:size]
                if np.any(weights < -1e-12):
                    continue
                x = weights @ q
            if best is None or x @ x < best @ best - 1e-30:
                best, best_index = x, index
    return best, best_index


def core_distance(a, b, iterations=200):
    """GJK distance between the convex hulls of two vertex sets (0 when they intersect)."""
    support = lambda d: a.vertices[np.argmax(a.vertices @ d)] - b.vertices[np.argmin(b.vertices @ d)]  # noqa: E731
    simplex = np.array([a.vertices[0] - b.vertices[0]])
    scale = max(1.0, float(np.abs(a.vertices).max(initial=0)), float(np.abs(b.vertices).max(initial=0)))
    previous = np.inf
    for _ in range(iterations):
        x, index = _closest_on_simplex(simplex)
        distance = float(np.linalg.norm(x))
        if distance <= _TOL * scale:
            return 0.0
        s = support(-x)
        kept = simplex[list(index)]
        # Stop when the support adds no progress, repeats a simplex vertex, or the distance stalls.
        if (
            distance * distance - float(x @ s) <= 1e-12 * distance * scale
            or np.any(np.all(np.abs(kept - s) <= 1e-15 * scale, axis=1))
            or distance >= previous * (1 - 1e-14)
        ):
            return distance
        previous = distance
        simplex = np.vstack((kept, s))
    raise RuntimeError(f"GJK did not converge in {iterations} iterations (distance {distance})")


def core_overlap(a, b):
    """SAT overlap of two intersecting polytopes over face normals and edge-edge crosses (0 if none)."""
    axes = [a.normals, b.normals]
    if len(a.edges) and len(b.edges):
        crosses = np.cross(a.edges[:, None, :], b.edges[None, :, :]).reshape(-1, 3)
        lengths = np.linalg.norm(crosses, axis=1)
        axes.append(crosses[lengths > 1e-9] / lengths[lengths > 1e-9, None])
    axes = np.concatenate(axes)
    if not len(axes):
        return 0.0
    pa, pb = a.vertices @ axes.T, b.vertices @ axes.T
    overlap = np.minimum(pa.max(axis=0) - pb.min(axis=0), pb.max(axis=0) - pa.min(axis=0))
    return max(0.0, float(overlap.min()))


def signed_distance(a, b):
    """Signed distance of the inflated shapes: positive gap, negative penetration depth."""
    d = core_distance(a, b)
    core = d if d > 0 else -core_overlap(a, b)
    return core - a.radius - b.radius


def penetration(a, b):
    return max(0.0, -signed_distance(a, b))


def plane_signed_distance(a, normal=(0.0, 0.0, 1.0), offset=0.0):
    """Signed distance of an inflated core to the half-space ``n.x >= offset``."""
    normal = np.asarray(normal, dtype=np.float64)
    return float((a.vertices @ normal).min() - offset - a.radius)


def model_core(model, shape, *, hull_points=None):
    """Body-local convex core of one Model shape; MESH and CONVEX_MESH use the hull of their vertices."""
    geo = int(model.shape_type.numpy()[shape])
    scale = model.shape_scale.numpy()[shape].astype(np.float64)
    if geo == newton.GeoType.SPHERE:
        core = point(scale[0])
    elif geo == newton.GeoType.CAPSULE:
        core = segment(scale[1], scale[0])
    elif geo == newton.GeoType.BOX:
        core = box(scale)
    elif geo in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        source = model.shape_source[shape] if hull_points is None else None
        points = hull_points if hull_points is not None else np.asarray(source.vertices, dtype=np.float64) * scale
        core = hull(points)
    else:
        raise ValueError(f"Shape {shape}: no convex core for {newton.GeoType(geo).name}")
    x = model.shape_transform.numpy()[shape].astype(np.float64)
    return core.placed(quaternion_matrix(x[3:]), x[:3])


def quaternion_matrix(q):
    """Rotation matrix of a unit quaternion stored xyzw (Warp order)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_core(core, body_pose):
    """Place a body-local core with a body pose ``(px, py, pz, qx, qy, qz, qw)``."""
    pose = np.asarray(body_pose, dtype=np.float64)
    return core.placed(quaternion_matrix(pose[3:]), pose[:3])
