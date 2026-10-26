# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-area quadrature for open shells and circular material rods."""

import numpy as np

from ...geometry import ShapeFlags
from .contact_sampling import _canonical_shapes

_CABLE_ANGULAR_SAMPLES = 64


def _append_material_samples(samples, materials, support, append):
    model, actors = samples.model, samples.actors
    targets = np.asarray(
        _canonical_shapes(actors, np.flatnonzero(model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES))),
        dtype=np.int64,
    )
    if not len(targets):
        samples.cable_edges = np.empty((0, 3), dtype=np.int32)
        samples.cable_coordinates = samples.cable_normals = np.empty((0, 3))
        return set()
    worlds, bodies, margins = (getattr(model, name).numpy() for name in ("shape_world", "shape_body", "shape_margin"))
    margins = margins.astype(np.float64)
    triangles, rest = (
        (model.tri_indices.numpy() if model.tri_indices is not None else np.empty((0, 3), dtype=np.int32)),
        model.particle_q.numpy().astype(float),
    )
    particles, angles = actors.maps["particle_to_global"], actors.maps["theta_to_global"]
    used = set()
    edges, coordinates, normals = [], [], []

    def emit(source, obj, face, slot, point, normal, area, gap, rows):
        for target in targets:
            if int(obj["world"]) != -1 and worlds[target] not in (-1, int(obj["world"])):
                continue
            used.add(int(target))
            columns = list(rows) + [col for col, _ in support.get(int(bodies[target]), ())]
            # Only the bounded producer consumes material samples.
            append(source, int(target), face, slot, point, normal, area, gap + margins[target], columns, [])
            samples.pair_kinds[source, int(target)] = "cloth-shape" if source == -2 else "cable-shape"

    for obj in materials.cloth:
        for face in obj["triangle_ids"]:
            tri = triangles[face]
            area = np.linalg.norm(np.cross(rest[tri[1]] - rest[tri[0]], rest[tri[2]] - rest[tri[0]])) / 2
            rows = [particles[p] + j for p in tri if particles[p] >= 0 for j in range(3)]
            for side in (-1, 1):
                for slot in range(3):
                    emit(
                        -2,
                        obj,
                        int(face),
                        slot + (3 if side < 0 else 0),
                        np.zeros(3),
                        (0, 0, side),
                        area / 3,
                        float(obj["thickness"]) / 2,
                        rows,
                    )
    gauss = (1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2
    for obj in materials.cable:
        ids, radius = obj["particle_ids"], float(obj["radius"])
        for edge, length in enumerate(obj["rest_lengths"]):
            theta = int(obj["theta_offset"]) + edge
            rows = [particles[p] + j for p in ids[edge : edge + 2] if particles[p] >= 0 for j in range(3)]
            if angles[theta] >= 0:
                rows.append(angles[theta])
            sections = [(a, radius, 0, 2 * np.pi * radius * length / (2 * _CABLE_ANGULAR_SAMPLES)) for a in gauss]
            for endpoint, enabled in ((0, edge == 0), (1, edge == len(ids) - 2)):
                if enabled:
                    sections.extend(
                        (
                            endpoint,
                            radius * np.sqrt(a),
                            2 * endpoint - 1,
                            np.pi * radius**2 / (2 * _CABLE_ANGULAR_SAMPLES),
                        )
                        for a in gauss
                    )
            for axial, radial, cap, area in sections:
                for slot in range(_CABLE_ANGULAR_SAMPLES):
                    phi = 2 * np.pi * slot / _CABLE_ANGULAR_SAMPLES
                    face = len(edges)
                    edges.append((int(ids[edge]), int(ids[edge + 1]), theta))
                    coordinates.append((axial, radial * np.cos(phi), radial * np.sin(phi)))
                    normals.append((cap, np.cos(phi), np.sin(phi)))
                    emit(-3, obj, face, slot, np.zeros(3), np.zeros(3), area, 0, rows)
    samples.cable_edges = np.asarray(edges, dtype=np.int32).reshape(-1, 3)
    samples.cable_coordinates = np.asarray(coordinates).reshape(-1, 3)
    samples.cable_normals = np.asarray(normals).reshape(-1, 3)
    return used
