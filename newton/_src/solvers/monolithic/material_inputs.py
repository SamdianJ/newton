# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Owned host topology, reference geometry and material data for shells and rods."""

import hashlib
from types import MappingProxyType

import numpy as np

from ...geometry.flags import ParticleFlags
from .solver_types import CableInput, ClothInput


def _ids(value, size, name):
    ids = np.asarray(value)
    if (
        ids.ndim != 1
        or not ids.size
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= size)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError(f"{name} requires nonempty, unique, in-range integer IDs")
    return ids.astype(np.int32)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite scalar") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _freeze(fields):
    # Bytes-backed storage cannot be made writable by a caller retaining a view.
    return MappingProxyType(
        {
            name: np.frombuffer(np.asarray(value).tobytes(), dtype=np.asarray(value).dtype).reshape(np.shape(value))
            for name, value in fields.items()
        }
    )


def _cable_geometry(positions, director):
    edges = np.diff(positions, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Cable requires finite nonzero edges")
    tangent = edges / lengths[:, None]
    if (
        director.shape != tangent.shape
        or not np.isfinite(director).all()
        or np.any(abs(np.linalg.norm(director, axis=1) - 1) > 1e-6)
        or np.any(abs(np.sum(tangent * director, axis=1)) > 1e-6)
    ):
        raise ValueError("Cable directors must be unit and perpendicular to their edges")
    dot = np.sum(tangent[:-1] * tangent[1:], axis=1)
    if np.any(1 + dot <= 1e-10):
        raise ValueError("Cable has antiparallel adjacent edges")
    cross = np.cross(tangent[:-1], tangent[1:])
    carried = director[:-1] + np.cross(cross, director[:-1])
    carried += np.cross(cross, np.cross(cross, director[:-1])) / (1 + dot[:, None])
    twist = np.arctan2(
        np.sum(tangent[1:] * np.cross(carried, director[1:]), axis=1),
        np.sum(carried * director[1:], axis=1),
    )
    return lengths, tangent, 2 * cross / (1 + dot[:, None]), twist


def _shell_adjacency(triangles, positions):
    if any(len(set(face)) != 3 for face in triangles) or len({tuple(sorted(f)) for f in triangles}) != len(triangles):
        raise ValueError("Cloth has duplicate or degenerate triangles")
    edges, neighbors = {}, [set() for _ in triangles]
    stencils = np.full((len(triangles), 6), -1, dtype=np.int32)
    stencils[:, :3] = triangles
    for face, tri in enumerate(triangles):
        for opposite in range(3):
            a, b = tri[(opposite + 1) % 3], tri[(opposite + 2) % 3]
            key = tuple(sorted((a, b)))
            incident = edges.setdefault(key, [])
            if len(incident) == 2 or (incident and incident[0][2:] != (b, a)):
                raise ValueError("Cloth requires consistently oriented manifold edges")
            if incident:
                other, slot, _, _ = incident[0]
                neighbors[face].add(other)
                neighbors[other].add(face)
                stencils[face, 3 + opposite] = triangles[other, slot]
                stencils[other, 3 + slot] = tri[opposite]
            incident.append((face, opposite, a, b))
    reached, pending = set(), [0]
    while pending:
        face = pending.pop()
        if face not in reached:
            reached.add(face)
            pending.extend(neighbors[face] - reached)
    if len(reached) != len(triangles):
        raise ValueError("Each cloth input must be edge-connected")
    # Reject pinched vertex fans even when the mesh is connected elsewhere.
    for vertex in np.unique(triangles):
        incident = set(np.flatnonzero(np.any(triangles == vertex, axis=1)))
        fan, pending = set(), [next(iter(incident))]
        while pending:
            face = pending.pop()
            if face not in fan:
                fan.add(face)
                pending.extend((neighbors[face] & incident) - fan)
        if fan != incident:
            raise ValueError("Cloth has a nonmanifold vertex")
    xyz = positions[triangles]
    e1, e2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
    normal = np.cross(e1, e2)
    twice_area = np.linalg.norm(normal, axis=1)
    if not np.isfinite(twice_area).all() or np.any(twice_area <= 0):
        raise ValueError("Cloth requires finite nondegenerate reference triangles")
    for corner in range(3):
        left, right = xyz[:, (corner + 1) % 3] - xyz[:, corner], xyz[:, (corner + 2) % 3] - xyz[:, corner]
        if np.any(twice_area / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)) <= 1e-8):
            raise ValueError("Cloth reference triangle is ill-conditioned")
    normals = normal / twice_area[:, None]
    for face, adjacent in enumerate(neighbors):
        if any(np.sum((normals[face] + normals[other]) ** 2) <= 1e-8 for other in adjacent):
            raise ValueError("Cloth reference normals are antiparallel")
    return stencils, twice_area / 2


class _MaterialInputs:
    """Validate once and freeze all material-owned Model data without device work."""

    def __init__(self, model, cloth_inputs=(), cable_inputs=()):
        self.cloth, self.cable = [], []
        self.theta_count = 0
        positions = model.particle_q.numpy().astype(np.float64)
        inv_mass, masses = model.particle_inv_mass.numpy(), model.particle_mass.numpy()
        worlds = model.particle_world.numpy()
        flags = model.particle_flags.numpy()
        occupied = set() if model.tet_indices is None else set(model.tet_indices.numpy().ravel())
        triangles = np.empty((0, 3), dtype=np.int32) if model.tri_indices is None else model.tri_indices.numpy()

        def common(ids, mass):
            if occupied.intersection(ids):
                raise ValueError("Tet, cloth and cable particle ownership must be disjoint")
            occupied.update(ids)
            if not np.isfinite(positions[ids]).all():
                raise ValueError("Material reference positions must be finite")
            world = worlds[ids]
            if len(np.unique(world)) != 1 or world[0] < -1 or world[0] >= model.world_count:
                raise ValueError("A material object must belong to one valid world")
            dynamic = (inv_mass[ids] > 0) & ((flags[ids] & int(ParticleFlags.ACTIVE)) != 0)
            if not np.isin(flags[ids], [0, int(ParticleFlags.ACTIVE)]).all():
                raise ValueError("Material particles require supported non-proxy flags")
            if (
                not np.isfinite(inv_mass[ids]).all()
                or np.any(inv_mass[ids] < 0)
                or not np.isfinite(masses[ids]).all()
                or np.any(masses[ids] < 0)
                or not np.isfinite(mass).all()
                or np.any(mass <= 0)
                or not np.allclose(masses[ids][dynamic], mass[dynamic], rtol=5e-6, atol=0)
                or not np.allclose(inv_mass[ids][dynamic] * masses[ids][dynamic], 1, rtol=5e-6, atol=0)
            ):
                raise ValueError("Dynamic particle mass must match material reference lumped mass")
            if model.spring_indices is not None:
                spring_ids = model.spring_indices.numpy()
                selected = np.any(np.isin(spring_ids, ids), axis=1)
                for name in ("spring_stiffness", "spring_damping"):
                    if np.any(getattr(model, name).numpy()[selected] != 0):
                        raise ValueError("Material particles cannot carry native spring forces")
            if model.edge_indices is not None:
                native_edges = model.edge_indices.numpy()
                selected = np.any(np.isin(native_edges, ids), axis=1)
                if np.any(model.edge_bending_properties.numpy()[selected] != 0):
                    raise ValueError("Material native hinge stiffness and damping must be zero")
            return {
                "particle_ids": ids,
                "positions": positions[ids],
                "mass": mass,
                "fixed": ~dynamic,
                "world": world[0],
            }

        for item in cloth_inputs:
            if not isinstance(item, ClothInput):
                raise ValueError("cloth_inputs requires SolverMonolithic.ClothInput objects")
            face_ids = _ids(item.triangle_ids, len(triangles), "triangle_ids")
            faces = triangles[face_ids]
            if faces.shape != (len(face_ids), 3) or np.any(faces < 0) or np.any(faces >= len(positions)):
                raise ValueError("Invalid cloth triangle vertices")
            stencils, areas = _shell_adjacency(faces, positions)
            bending = np.asarray(item.bending, dtype=float)
            if (
                bending.shape != (2,)
                or not np.isfinite(bending).all()
                or bending[1] <= 0
                or bending[0] <= -bending[1] / 2
            ):
                raise ValueError("Cloth bending requires beta > 0 and alpha > -beta/2")
            thickness = _positive(item.thickness, "thickness")
            density = _positive(item.surface_density, "surface_density")
            material = model.tri_materials.numpy()[face_ids].astype(float)
            if (
                material.shape != (len(faces), 5)
                or not np.isfinite(material).all()
                or np.any(material[:, 0] <= 0)
                or np.any(material[:, 1] <= -material[:, 0])
                or np.any(material[:, 2:] != 0)
                or np.any(model.tri_activations.numpy()[face_ids] != 0)
            ):
                raise ValueError(
                    "Cloth requires valid membrane coefficients and zero native damping/drag/lift/activation"
                )
            ids, local = np.unique(faces, return_inverse=True)
            mass = np.zeros(len(ids))
            np.add.at(mass, local.ravel(), np.repeat(areas * density / 3, 3))
            fields = common(ids, mass)
            fields.update(
                triangle_ids=face_ids,
                triangles=faces,
                stencils=stencils,
                areas=areas,
                membrane=material[:, :2],
                bending=bending,
                thickness=thickness,
                surface_density=density,
            )
            self.cloth.append(_freeze(fields))

        for item in cable_inputs:
            if not isinstance(item, CableInput):
                raise ValueError("cable_inputs requires SolverMonolithic.CableInput objects")
            ids = _ids(item.particle_ids, len(positions), "particle_ids")
            if len(ids) < 2:
                raise ValueError("Cable requires at least two vertices")
            fields = {name: _positive(getattr(item, name), name) for name in ("EA", "GJ", "radius", "linear_density")}
            bending = np.asarray(item.bending, dtype=float)
            if (
                bending.shape != (2, 2)
                or not np.isfinite(bending).all()
                or not np.array_equal(bending, bending.T)
                or np.linalg.eigvalsh(bending).min() <= 0
            ):
                raise ValueError("Cable bending must be a symmetric positive definite 2x2 tensor")
            theta, director = np.asarray(item.rest_theta, dtype=float), np.asarray(item.rest_director, dtype=float)
            fixed = np.asarray(item.fixed_theta)
            ne = len(ids) - 1
            if theta.shape != (ne,) or not np.isfinite(theta).all() or fixed.shape != (ne,) or fixed.dtype != np.bool_:
                raise ValueError("Cable requires finite per-edge theta and boolean fixed_theta")
            lengths, tangent, kb, twist = _cable_geometry(positions[ids], director)
            d2 = np.cross(tangent, director)
            m1 = np.cos(theta[:, None]) * director + np.sin(theta[:, None]) * d2
            m2 = -np.sin(theta[:, None]) * director + np.cos(theta[:, None]) * d2
            curvature = (
                np.stack((np.sum(kb * (m2[:-1] + m2[1:]), axis=1), -np.sum(kb * (m1[:-1] + m1[1:]), axis=1)), axis=1)
                / 2
            )
            mass = np.zeros(len(ids))
            mass[:-1] += lengths * fields["linear_density"] / 2
            mass[1:] += lengths * fields["linear_density"] / 2
            fields.update(common(ids, mass))
            gauge = np.array_equal(bending, np.eye(2) * bending[0, 0]) and np.all(curvature == 0) and not fixed.any()
            eliminated = fixed.copy()
            if gauge:
                eliminated[0] = True
            fields.update(
                bending=bending,
                rest_theta=theta,
                rest_director=director,
                fixed_theta=fixed,
                rest_lengths=lengths,
                rest_curvature=curvature,
                rest_twist=np.diff(theta) + twist,
                ref_twist=twist,
                free_theta=np.flatnonzero(~eliminated).astype(np.int32),
                gauge=gauge,
                theta_offset=self.theta_count,
                length_scale=np.median(lengths),
            )
            self.theta_count += ne
            self.cable.append(_freeze(fields))
        self.cloth, self.cable = tuple(self.cloth), tuple(self.cable)
        digest = hashlib.sha256()
        for kind, objects in (("cloth", self.cloth), ("cable", self.cable)):
            for index, fields in enumerate(objects):
                for name, value in sorted(fields.items()):
                    digest.update(f"{kind}/{index}/{name}/{value.dtype.str}/{value.shape}:".encode())
                    digest.update(value.tobytes())
        self.identity = digest.hexdigest()
