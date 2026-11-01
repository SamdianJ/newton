# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A free 40 mm open paper shell and per-frame geometry diagnostics."""

from typing import Any

import numpy as np
import warp as wp

EDGE = 0.04
CELLS = 6
THICKNESS = 0.0003
SURFACE_DENSITY = 0.35
YOUNG_MODULUS = 1e8
POISSON_RATIO = 0.3


def add_paper_box(builder, position):
    """Add five welded outward-facing panels, with all particles dynamic; return SI material data."""
    n = CELLS
    corners = np.array(
        [
            [(-n, -n, 0), (-n, n, 0), (n, n, 0), (n, -n, 0)],
            [(-n, -n, 0), (n, -n, 0), (n, -n, n), (-n, -n, n)],
            [(n, -n, 0), (n, n, 0), (n, n, n), (n, -n, n)],
            [(n, n, 0), (-n, n, 0), (-n, n, n), (n, n, n)],
            [(-n, n, 0), (-n, -n, 0), (-n, -n, n), (-n, n, n)],
        ]
    )
    vertices, triangles = {}, []
    for face in corners:
        grid = np.empty((n + 1, n + 1), dtype=int)
        for i in range(n + 1):
            for j in range(n + 1):
                p = tuple(face[0] + i * (face[1] - face[0]) // n + j * (face[3] - face[0]) // n)
                grid[i, j] = vertices.setdefault(p, len(vertices))
        for i in range(n):
            for j in range(n):
                a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                triangles.extend(((a, b, c), (a, c, d)))
    x = np.array(list(vertices)) * [EDGE / (2 * n), EDGE / (2 * n), EDGE / n]
    x += np.asarray(position) - [0, 0, EDGE / 2]
    # Compute masses from the same rounded rest geometry stored by ModelBuilder.
    x = x.astype(np.float32).astype(float)
    faces = x[np.array(triangles)]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
    mass = np.zeros(len(x))
    np.add.at(mass, np.array(triangles).ravel(), np.repeat(area * SURFACE_DENSITY / 3, 3))
    first = builder.particle_count
    for p, m in zip(x, mass, strict=True):
        builder.add_particle(wp.vec3(p), wp.vec3(0), m, radius=THICKNESS / 2)
    mu = YOUNG_MODULUS * THICKNESS / (2 * (1 + POISSON_RATIO))
    lam = YOUNG_MODULUS * THICKNESS * POISSON_RATIO / (1 - POISSON_RATIO**2)
    for face in triangles:
        builder.add_triangle(*(first + i for i in face), tri_ke=mu, tri_ka=lam, tri_kd=0, tri_drag=0, tri_lift=0)
    return {
        "name": "paper-box",
        "young_modulus_pa": YOUNG_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "thickness_m": THICKNESS,
        "surface_density_kg_m2": SURFACE_DENSITY,
        "mu_2d_N_m": mu,
        "lambda_2d_N_m": lam,
        "bending_Nm": (lam * THICKNESS**2 / 12, 2 * mu * THICKNESS**2 / 12),
    }


@wp.kernel
def _intersections(x: wp.array[Any], triangles: wp.array2d[int], count: wp.array[int]):
    i, j = wp.tid()
    if i >= j:
        return
    for a in range(3):
        for b in range(3):
            if triangles[i, a] == triangles[j, b]:
                return
    # Warp's Moller predicate has an absolute epsilon: use local millimeters in FP64.
    origin = wp.vec3d(x[triangles[i, 0]])
    scale = wp.float64(1000.0)
    a0 = wp.vec3d(0.0)
    a1 = (wp.vec3d(x[triangles[i, 1]]) - origin) * scale
    a2 = (wp.vec3d(x[triangles[i, 2]]) - origin) * scale
    b0 = (wp.vec3d(x[triangles[j, 0]]) - origin) * scale
    b1 = (wp.vec3d(x[triangles[j, 1]]) - origin) * scale
    b2 = (wp.vec3d(x[triangles[j, 2]]) - origin) * scale
    if wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2):
        wp.atomic_add(count, 0, 1)


class PaperBoxMetrics:
    """Inspect every published shell frame; this supplies no self-contact forces or CCD."""

    def __init__(self, model):
        self.triangles = model.tri_indices
        self.ids = self.triangles.numpy()
        self.rest_area = self.area(model.particle_q.numpy().astype(float))
        self.count = wp.zeros(1, dtype=int, device=model.device)

    def area(self, x):
        """Return twice the triangle areas [m²]."""
        p = x[self.ids]
        return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    def measure(self, positions):
        """Return extremal area ratios and nonadjacent triangle intersection count."""
        ratio = self.area(positions.numpy().astype(float)) / self.rest_area
        self.count.zero_()
        wp.launch(
            _intersections,
            (len(self.ids), len(self.ids)),
            [positions, self.triangles, self.count],
            device=positions.device,
        )
        return {
            "area_ratio_min": float(ratio.min()),
            "area_ratio_max": float(ratio.max()),
            "self_intersections": int(self.count.numpy()[0]),
        }
