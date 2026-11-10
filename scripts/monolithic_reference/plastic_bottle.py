# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural open, empty cola-shaped elastic shell; not a calibrated PET model."""

import numpy as np
import warp as wp

from scripts.monolithic_reference.material_recipes import shell_coefficients


def bottle_mesh():
    """Return outward triangles of a 160 mm bottle with a 60 mm belly and open neck."""
    # y is the bottle axis; a welded bottom disk closes only the lower end.
    profile = (
        np.array(
            [
                (-80, 13),
                (-80, 24),
                (-77, 28),
                (-72, 30),
                (-64, 30),
                (-56, 29),
                (-48, 28),
                (-40, 27),
                (-32, 26),
                (-24, 26),
                (-16, 27),
                (-8, 28),
                (0, 29),
                (8, 30),
                (16, 30),
                (24, 30),
                (32, 29),
                (40, 27),
                (48, 23),
                (56, 18),
                (64, 12),
                (70, 10),
                (76, 10),
                (80, 11),
            ],
            dtype=float,
        )
        / 1000
    )
    segments = 24
    theta = np.arange(segments) * 2 * np.pi / segments
    rings = np.array([np.c_[r * np.cos(theta), np.full(segments, y), r * np.sin(theta)] for y, r in profile])
    points = np.concatenate(([[0, profile[0, 0], 0]], rings.reshape(-1, 3)))
    faces = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    for i in range(len(profile) - 1):
        for j in range(segments):
            a, b = 1 + i * segments + j, 1 + i * segments + (j + 1) % segments
            c, d = b + segments, a + segments
            faces.extend(((a, d, c), (a, c, b)))
    return points, np.array(faces, dtype=np.int32)


def add_bottle(builder, *, center=(-0.15, 0, -0.064)):
    """Add a free shell with assumed E=1 GPa, h=0.25 mm and density=1380 kg/m³."""
    points, faces = bottle_mesh()
    points = (points + center).astype(np.float32).astype(float)
    material = {"young_modulus_Pa": 1e9, "poisson_ratio": 0.35, "thickness_m": 0.00025}
    lam, mu, *bending = shell_coefficients(material)
    surface_density = 1380 * material["thickness_m"]
    vertices = points[faces]
    area = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1) / 2
    mass = np.zeros(len(points))
    np.add.at(mass, faces.ravel(), np.repeat(area * surface_density / 3, 3))
    first = builder.particle_count
    for point, value in zip(points, mass, strict=True):
        builder.add_particle(wp.vec3(point), wp.vec3(0), value, radius=material["thickness_m"] / 2)
    for face in faces + first:
        builder.add_triangle(*face, tri_ke=mu, tri_ka=lam, tri_kd=0, tri_drag=0, tri_lift=0)
    return material | {
        "name": "empty-open-elastic-plastic-bottle-proxy",
        "density_kg_m3": 1380,
        "surface_density_kg_m2": surface_density,
        "bending_Nm": bending,
        "mass_kg": float(mass.sum()),
        "center_m": list(center),
        "limitations": "uncalibrated elastic shell; no liquid, gas pressure, plasticity or self-contact",
    }
