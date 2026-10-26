# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed directed surface queries and actual two-endpoint scalar supports."""

import numpy as np

from ...geometry import GeoType, Mesh, ShapeFlags


def _surface_samples(geo, scale, mesh=None, *, _return_triangles=False):
    """Return local P1Q3 points, outward normals and scaled surface weights."""
    bary = np.eye(3) * 0.5 + np.full((3, 3), 1 / 6)
    if geo == GeoType.SPHERE:
        if not np.isfinite(scale).all() or scale[0] <= 0:
            raise ValueError("Invalid source sphere scale")
        theta, phi = np.linspace(0, np.pi, 33), np.arange(32) * (2 * np.pi / 32)
        grid = np.stack(
            (
                np.sin(theta[:, None]) * np.cos(phi),
                np.sin(theta[:, None]) * np.sin(phi),
                np.broadcast_to(np.cos(theta[:, None]), (33, 32)),
            ),
            axis=-1,
        )
        faces = []
        for i in range(32):
            for j in range(32):
                a, b, c, d = grid[i, j], grid[i, (j + 1) % 32], grid[i + 1, j], grid[i + 1, (j + 1) % 32]
                if i > 0:
                    faces.append((a, b, c))
                if i < 31:
                    faces.append((b, d, c))
        tri = np.asarray(faces)
        u, v, w = tri[:, 0], tri[:, 1], tri[:, 2]
        area = (
            2
            * scale[0] ** 2
            * np.arctan2(abs(np.einsum("ij,ij->i", u, np.cross(v, w))), 1 + np.sum(u * v + v * w + w * u, axis=1))
        )
        points = np.einsum("qn,fni->fqi", bary, tri)
        normals = points / np.linalg.norm(points, axis=2)[..., None]
        points = normals * scale[0]
        tri = tri * scale[0]
    else:
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid source shape scale")
        if geo == GeoType.BOX:
            mesh = Mesh.create_box(1.0, compute_inertia=False)
        elif geo != GeoType.MESH or not isinstance(mesh, Mesh):
            raise ValueError("Unsupported contact source shape")
        indices = np.asarray(mesh.indices)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if indices.size % 3 or vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Invalid source mesh topology")
        if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError("Invalid source mesh indices")
        tri = (vertices * scale)[indices.reshape(-1, 3)]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(cross, axis=1) / 2
        if not len(area) or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Source faces must have positive finite area")
        points = np.einsum("qn,fni->fqi", bary, tri)
        normals = np.repeat((cross / (2 * area[:, None]))[:, None], 3, axis=1)
    result = points.reshape(-1, 3), normals.reshape(-1, 3), np.repeat(area / 3, 3)
    return (*result, tri) if _return_triangles else result


def _canonical_shapes(actors, shapes):
    """Order shapes by owning actor and actor-local index, then unowned shapes in Model order.

    Contact sample identities, and so the order in which reductions add contact
    terms, follow this order rather than Model shape creation order.
    """
    owner, local = actors.maps["shape_owner"], actors.maps["shape_local"]
    unowned = len(actors.actors)
    return sorted((int(s) for s in shapes), key=lambda s: (owner[s], local[s]) if owner[s] >= 0 else (unowned, s))


def _checked_contact_capacity(count, support, triplets, base=0):
    if min(count, support, triplets, base) < 0 or max(3 * count + 1, support, base + triplets) >= 2**31:
        raise ValueError("Actor contact capacity exceeds int32")


class ContactSamples:
    """Freeze Model-selected directed keys without rebuilding collision filters."""

    def __init__(self, model, actors, *, radius, base_triplet_count, _expand_support=True, _materials=None):
        if type(_expand_support) is not bool:
            raise ValueError("Contact support expansion mode must be boolean")
        self.model = model
        self.actors = actors
        self.meshes = {}
        self.sources = {}
        self.count = 0
        self.triplet_capacity = 0
        self.support_size = 0
        self.fields = {}
        self.pair_kinds = {}
        self.keys, self.supports, self.selectors = [], [], []
        self.body_kind = np.full(model.body_count, -1, dtype=np.int32)
        self.body_local = np.full(model.body_count, -1, dtype=np.int32)
        rigids = [a for a in actors.actors if a.kind == "rigid"]
        if not rigids and _materials is None:
            self.rigid_shapes = ()
            return
        arm = next((a for a in actors.actors if a.kind == "articulation"), None)
        arm_bodies = () if arm is None else arm.body_indices
        arm_joints = () if arm is None else arm.joint_indices
        arm_dofs = () if arm is None else arm.dof_indices
        self.body_kind[list(arm_bodies)] = 0
        self.body_local[list(arm_bodies)] = np.arange(len(arm_bodies))
        for i, actor in enumerate(rigids):
            self.body_kind[actor.body_indices[0]], self.body_local[actor.body_indices[0]] = 1, i
        self.rigid_shapes = tuple(s for a in rigids for s in a.shape_indices)
        for name in ("shape_flags", "shape_contact_pairs", "shape_world", "shape_collision_group"):
            self.sources[name] = getattr(model, name)
        self.values = {name: value.numpy().copy() for name, value in self.sources.items()}
        self.pair_count = model.shape_contact_pair_count
        self.filter_store = model._shape_collision_filter_store()
        self.filters = model.shape_collision_filter_pairs_array().copy()
        flags, pairs = self.values["shape_flags"], self.values["shape_contact_pairs"]
        if pairs.shape != (self.pair_count, 2) or np.any(pairs < 0) or np.any(pairs >= model.shape_count):
            raise ValueError("Invalid Model shape contact pairs")
        owner = actors.maps["shape_owner"]
        bodies, types, scales, margins = (
            getattr(model, n).numpy() for n in ("shape_body", "shape_type", "shape_scale", "shape_margin")
        )
        margins = margins.astype(np.float64)
        incoming = {int(model.joint_child.numpy()[j]): j for j in arm_joints}
        parent, starts = model.joint_parent.numpy(), model.joint_qd_start.numpy()
        local_dof = {d: i for i, d in enumerate(arm_dofs)}
        support = {}
        for b in arm_bodies:
            js, cursor = [], b
            while cursor >= 0:
                joint = incoming[cursor]
                js.extend(local_dof[d] for d in range(int(starts[joint]), int(starts[joint + 1])))
                cursor = int(parent[joint])
            support[b] = [(arm.dof_start + j, j) for j in sorted(js)]
        for a in rigids:
            support[a.body_indices[0]] = [(a.dof_start + j, j) for j in range(6)]
        points, normals, weights, offsets, source_shapes, targets, faces, slots = [], [], [], [], [], [], [], []

        def append(source, target, face, slot, point, normal, area, gap_offset, columns, selectors):
            k = len(columns)
            self.count += 1
            self.support_size += 3 * k
            self.triplet_capacity += 3 * k * k
            _checked_contact_capacity(self.count, self.support_size, self.triplet_capacity, base_triplet_count)
            self.keys.append((source, target, face, slot))
            if _expand_support:
                self.supports.append(tuple(columns))
                self.selectors.append(tuple(selectors))
            points.append(point)
            normals.append(normal)
            weights.append(area)
            offsets.append(gap_offset)
            source_shapes.append(source)
            targets.append(target)
            faces.append(face)
            slots.append(slot)

        triangles = model.tri_indices.numpy() if model.tri_indices is not None else np.empty((0, 3), dtype=np.int32)
        rest = model.particle_q.numpy().astype(np.float64)
        rest_area = (
            np.linalg.norm(
                np.cross(rest[triangles[:, 1]] - rest[triangles[:, 0]], rest[triangles[:, 2]] - rest[triangles[:, 0]]),
                axis=1,
            )
            / 2
        )
        tet_targets = _canonical_shapes(
            actors, [s for s in self.rigid_shapes if flags[s] & int(ShapeFlags.COLLIDE_PARTICLES)]
        )
        cloth_faces = set() if _materials is None else {int(f) for obj in _materials.cloth for f in obj["triangle_ids"]}
        if _materials is not None:
            tet_targets = _canonical_shapes(actors, np.flatnonzero(flags & int(ShapeFlags.COLLIDE_PARTICLES)))
        particle_world = model.particle_world.numpy() if _materials is not None else None
        for face, tri in enumerate(triangles):
            if face in cloth_faces:
                continue
            for slot in range(3):
                for target in tet_targets:
                    if _materials is not None:
                        world = int(particle_world[tri[0]])
                        target_world = int(self.values["shape_world"][target])
                        if world >= 0 and target_world >= 0 and world != target_world:
                            continue
                    self.pair_kinds[-1, target] = "R-T" if _materials is None else "tet-shape"
                    cols, sels = [], []
                    for node, particle in enumerate(tri):
                        row = actors.maps["particle_to_global"][particle]
                        if row >= 0:
                            for j in range(3):
                                cols.append(row + j)
                                sels.append((0, 3 * node + j))
                    for col, j in support.get(int(bodies[target]), ()):
                        cols.append(col)
                        sels.append((1, j))
                    append(
                        -1,
                        target,
                        face,
                        slot,
                        np.zeros(3),
                        np.zeros(3),
                        rest_area[face] / 3,
                        radius + margins[target],
                        cols,
                        sels,
                    )
        surfaces, seen = {}, set()
        targets_used = set(tet_targets)
        # Pairs, and each pair's two directions, follow the canonical shape order.
        rank = {s: i for i, s in enumerate(_canonical_shapes(actors, range(model.shape_count)))}
        ordered = sorted(
            (sorted(map(int, pair), key=rank.get) for pair in pairs), key=lambda p: (rank[p[0]], rank[p[1]])
        )
        for a, b in ordered:
            if owner[a] == owner[b] or not (a in self.rigid_shapes or b in self.rigid_shapes):
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                raise ValueError("Duplicate effective Model shape pair")
            seen.add(pair)
            if not flags[a] & int(ShapeFlags.COLLIDE_SHAPES) or not flags[b] & int(ShapeFlags.COLLIDE_SHAPES):
                raise ValueError("Model shape pair conflicts with collision flags")
            dynamic = [bool(support.get(int(bodies[s]), ())) for s in (a, b)]
            if not any(dynamic):
                continue
            self.pair_kinds[pair] = (
                "rigid-static" if not all(dynamic) else "R-R" if all(s in self.rigid_shapes for s in (a, b)) else "A-R"
            )
            for source, target, enabled in ((a, b, dynamic[0]), (b, a, dynamic[1])):
                if not enabled:
                    continue
                if source not in surfaces:
                    surfaces[source] = _surface_samples(
                        types[source], scales[source].astype(float), model.shape_source[source]
                    )
                if types[source] == GeoType.MESH:
                    mesh = model.shape_source[source]
                    self.meshes[source] = (mesh, np.array(mesh.vertices, copy=True), np.array(mesh.indices, copy=True))
                targets_used.add(target)
                ps, ns, ws = surfaces[source]
                cols, sels = [], []
                for end, s in enumerate((source, target)):
                    for col, j in support.get(int(bodies[s]), ()):
                        cols.append(col)
                        sels.append((end, j))
                for index, (p, n, w) in enumerate(zip(ps, ns, ws, strict=True)):
                    append(
                        source,
                        target,
                        index // 3,
                        index % 3,
                        p,
                        n,
                        w * (0.5 if all(dynamic) else 1),
                        margins[source] + margins[target],
                        cols,
                        sels,
                    )
        self.material_start = self.count
        if _materials is not None:
            from .contact_material_sampling import _append_material_samples  # noqa: PLC0415

            targets_used.update(_append_material_samples(self, _materials, support, append))
        used_shapes = targets_used | {s for s in source_shapes if s >= 0}
        for s in used_shapes:
            geo, scale = types[s], scales[s]
            if geo not in (GeoType.PLANE, GeoType.SPHERE, GeoType.BOX, GeoType.MESH):
                raise ValueError("Unsupported actor contact target")
            if not np.isfinite(scale).all() or not np.isfinite(margins[s]) or margins[s] < 0:
                raise ValueError("Invalid target shape scale or margin")
            if geo == GeoType.PLANE and (
                s not in targets_used
                or np.any(scale[:2] != 0)
                or (_materials is None and support.get(int(bodies[s]), ()))
            ):
                raise ValueError("New plane target must be infinite and static outside material contact")
            if (geo == GeoType.SPHERE and scale[0] <= 0) or (geo == GeoType.BOX and np.any(scale <= 0)):
                raise ValueError("Invalid analytic target scale")
            if geo == GeoType.MESH:
                mesh = model.shape_source[s]
                _surface_samples(geo, scale.astype(float), mesh)
                self.meshes[s] = (mesh, np.array(mesh.vertices, copy=True), np.array(mesh.indices, copy=True))
        self.targets_used = tuple(sorted(targets_used))
        self.source_shapes = np.asarray(source_shapes, dtype=np.int32)
        self.targets = np.asarray(targets, dtype=np.int32)
        self.faces, self.slots = np.asarray(faces, dtype=np.int32), np.asarray(slots, dtype=np.int32)
        self.points, self.normals = np.asarray(points).reshape(-1, 3), np.asarray(normals).reshape(-1, 3)
        self.weights, self.offsets = np.asarray(weights), np.asarray(offsets)
        if self.count:
            self.fields = {
                "actor_contact.rule": np.frombuffer(b"directed-p1q3/v1", dtype=np.uint8),
                "actor_contact.keys": np.asarray(self.keys, dtype=np.int32),
                "actor_contact.points": self.points,
                "actor_contact.normals": self.normals,
                "actor_contact.weights": self.weights,
                "actor_contact.offsets": self.offsets,
                "actor_contact.flags": flags,
                "actor_contact.pairs": pairs,
                "actor_contact.world": self.values["shape_world"],
                "actor_contact.group": self.values["shape_collision_group"],
                "actor_contact.filters": self.filters,
            }
            if _materials is not None:
                self.fields["material_contact.rule"] = np.frombuffer(b"shell-p1q3-rod-gauss2-ring64/v1", dtype=np.uint8)
                for name in ("cable_edges", "cable_coordinates", "cable_normals"):
                    self.fields["material_contact." + name] = getattr(self, name)
            for s, (_, vertices, indices) in self.meshes.items():
                self.fields[f"actor_contact.mesh.{s}.vertices"] = vertices
                self.fields[f"actor_contact.mesh.{s}.indices"] = indices

    def validate_identity(self):
        if not self.sources:
            return
        for name, source in self.sources.items():
            if getattr(self.model, name) is not source or not np.array_equal(source.numpy(), self.values[name]):
                raise ValueError(f"Actor contact model changed: {name}")
        if (
            self.model.shape_contact_pair_count != self.pair_count
            or self.model._shape_collision_filter_store() is not self.filter_store
        ):
            raise ValueError("Actor contact pair/filter storage changed")
        if not np.array_equal(self.model.shape_collision_filter_pairs_array(), self.filters):
            raise ValueError("Actor contact filters changed")
        for s, (mesh, vertices, indices) in self.meshes.items():
            if (
                self.model.shape_source[s] is not mesh
                or not np.array_equal(mesh.vertices, vertices)
                or not np.array_equal(mesh.indices, indices)
            ):
                raise ValueError("Actor contact source mesh changed")
