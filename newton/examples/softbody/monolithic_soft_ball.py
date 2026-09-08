# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline, deterministic volume assets for the internal PR-6D fixture."""

import itertools

import numpy as np

import newton

RADIUS = 0.020
DENSITY = 1000.0
YOUNG_MODULUS = 10000.0
POISSON_RATIO = 0.3


def generate_ball(refinement, *, radius=RADIUS):
    """Map a conforming Freudenthal cube grid onto a sphere, in SI units.

    Each refinement increases both boundary and interior resolution. This is
    an explicit geometric discretization, not an independently refined FEM
    experiment. No external tetrahedralizer or random seed is needed.
    """
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement not in (1, 2, 3):
        raise ValueError("Ball refinement must be 1, 2, or 3")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Ball radius must be positive and finite")
    n = 2 * refinement
    xyz = np.asarray(list(itertools.product(np.linspace(-1, 1, n + 1), repeat=3)))
    x, y, z = xyz.T
    points = (
        radius
        * xyz
        * np.sqrt(
            np.column_stack(
                (
                    1 - y * y / 2 - z * z / 2 + y * y * z * z / 3,
                    1 - z * z / 2 - x * x / 2 + z * z * x * x / 3,
                    1 - x * x / 2 - y * y / 2 + x * x * y * y / 3,
                )
            )
        )
    )
    ids = np.arange(len(points)).reshape((n + 1,) * 3)
    tets = []
    for cell in itertools.product(range(n), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = np.array(cell)
            tet = [ids[tuple(corner)]]
            for axis in order:
                corner[axis] += 1
                tet.append(ids[tuple(corner)])
            tets.append(tet)
    tets = np.asarray(tets, dtype=np.int32)
    p = points[tets]
    negative = np.linalg.det(p[:, 1:] - p[:, :1]) < 0
    tets[negative, :2] = tets[negative, 1::-1]
    points = points.astype(np.float32)
    validate_ball(points, tets, radius=radius)
    mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
    lam = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
    return newton.TetMesh(points, tets.flatten(), density=DENSITY, k_mu=mu, k_lambda=lam, k_damp=0.0)


def validate_ball(vertices, tets, *, radius=RADIUS):
    """Reject invalid volume topology and return independent rest-quality data."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Ball radius must be positive and finite")
    x = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(tets)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid ball coordinates")
    if t.ndim != 2 or t.shape[1] != 4 or not len(t) or t.dtype.kind not in "iu":
        raise ValueError("Expected integer tet connectivity (T, 4)")
    if t.min() < 0 or t.max() >= len(x) or len(np.unique(t)) != len(x):
        raise ValueError("Invalid connectivity or isolated ball node")
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise ValueError("Duplicate tet")
    p = x[t]
    volumes = np.linalg.det(p[:, 1:] - p[:, :1]) / 6
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise ValueError("Nonpositive rest volume")
    # Outward orientation for a positive tet, retained for a winding check.
    faces = t[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    _, inverse, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError("Non-manifold volume face")
    owners = np.repeat(np.arange(len(t)), 4)
    groups = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    adjacency = [[] for _ in t]
    for start in starts[counts == 2]:
        a, b = owners[groups[start : start + 2]]
        adjacency[a].append(b)
        adjacency[b].append(a)
        f, g = faces[groups[start : start + 2]]
        if any(np.array_equal(f, np.roll(g, k)) for k in range(3)):
            raise ValueError("Inconsistent interior face orientation")
    seen, stack = {0}, [0]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if len(seen) != len(t):
        raise ValueError("Disconnected ball volume")
    boundary = faces[counts[inverse] == 1]
    edges = boundary[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    _, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError("Open or non-manifold ball boundary")
    surface_nodes = np.unique(boundary)
    if len(surface_nodes) - len(edge_counts) + len(boundary) != 2:
        raise ValueError("Ball boundary is not a topological sphere")
    radii = np.linalg.norm(x, axis=1)
    if not np.allclose(radii[surface_nodes], radius, rtol=1e-6, atol=1e-10) or np.any(radii > radius * (1 + 1e-6)):
        raise ValueError("Ball boundary radius mismatch")
    squared_edges = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1) for i, j in itertools.combinations(range(4), 2))
    quality = 12 * (3 * volumes) ** (2 / 3) / squared_edges
    if quality.min() < 0.1:
        raise ValueError("Ball tet mean-ratio quality below 0.1")
    volume = float(volumes.sum())
    return {
        "radius_m": radius,
        "node_count": len(x),
        "tet_count": len(t),
        "boundary_count": len(boundary),
        "interior_node_count": len(x) - len(surface_nodes),
        "minimum_rest_volume_m3": float(volumes.min()),
        "volume_m3": volume,
        "relative_volume_error": abs(volume / (4 * np.pi * radius**3 / 3) - 1),
        "minimum_mean_ratio": float(quality.min()),
        "mass_kg": volume * DENSITY,
        "all_nodes_dynamic": True,
    }


def load_ball(path, *, radius=RADIUS):
    """Validate the frozen volume before passing it to the Newton loader."""
    with np.load(path, allow_pickle=False) as data:
        validate_ball(data["vertices"], data["tet_indices"].reshape(-1, 4), radius=radius)
        mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
        lam = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
        for name, expected in (("density", DENSITY), ("k_mu", mu), ("k_lambda", lam), ("k_damp", 0)):
            if name not in data or not np.allclose(data[name], expected, rtol=1e-6, atol=0):
                raise ValueError(f"Frozen ball material mismatch: {name}")
    return newton.TetMesh.create_from_file(str(path))
