# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed P1Q3 boundary sampling for the experimental monolithic solver."""

from __future__ import annotations

import enum
import math
from collections import Counter, defaultdict
from weakref import WeakKeyDictionary

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...geometry.sdf_texture import TextureSDFData
from ...geometry.soft_contacts_sdf import _shape_frames, eval_shape_sdf
from ...geometry.types import GeoType
from ...sim import Contacts, Model, State


def _validate_and_collect_boundary_faces(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic model face indices and all referenced particle ids."""
    error = MonolithicCollisionPipeline.Error
    status = MonolithicCollisionPipeline.Status.INVALID_BOUNDARY
    tets = model.tet_indices.numpy()
    positions = model.particle_q.numpy().astype(np.float64)
    if len(tets) == 0 or tets.shape[1:] != (4,) or np.any(tets < 0) or np.any(tets >= len(positions)):
        raise error(status, "invalid_tets")
    if any(len(set(tet)) != 4 for tet in tets) or len({tuple(sorted(tet)) for tet in tets}) != len(tets):
        raise error(status, "duplicate_or_degenerate_tet")
    referenced = np.unique(tets).astype(np.int32)
    if not np.isfinite(positions[referenced]).all():
        raise error(status, "nonfinite_reference_positions")
    connected = set(tets[0])
    remaining = list(tets[1:])
    while remaining:
        pending = []
        for tet in remaining:
            if connected.intersection(tet):
                connected.update(tet)
            else:
                pending.append(tet)
        if len(pending) == len(remaining):
            raise error(status, "disconnected_tets")
        remaining = pending
    incidence = Counter()
    for tet in tets:
        for opposite in range(4):
            incidence[tuple(sorted(np.delete(tet, opposite)))] += 1
    if any(count > 2 for count in incidence.values()):
        raise error(status, "nonmanifold_tet_face")
    boundary = sorted(face for face, count in incidence.items() if count == 1)
    edges = Counter()
    vertex_links = defaultdict(list)
    for a, b, c in boundary:
        edges.update(((a, b), (a, c), (b, c)))
        vertex_links[a].append({b, c})
        vertex_links[b].append({a, c})
        vertex_links[c].append({a, b})
    if not boundary or any(count != 2 for count in edges.values()):
        raise error(status, "nonmanifold_boundary_edge")
    # Each boundary vertex must have one connected ring, including at pinched vertices.
    for edges_at_vertex in vertex_links.values():
        incident = list(edges_at_vertex)
        ring = incident.pop()
        while incident:
            pending = []
            for edge in incident:
                if ring.intersection(edge):
                    ring.update(edge)
                else:
                    pending.append(edge)
            if len(pending) == len(incident):
                raise error(status, "nonmanifold_boundary_vertex")
            incident = pending
    triangles = model.tri_indices.numpy()
    if triangles.shape != (len(boundary), 3):
        raise error(status, "missing_or_extra_faces")
    face_to_index = {}
    for index, face in enumerate(triangles):
        key = tuple(sorted(face))
        if len(set(face)) != 3 or key in face_to_index:
            raise error(status, "duplicate_or_repeated_vertices")
        face_to_index[key] = index
    if set(face_to_index) != set(boundary):
        raise error(status, "boundary_set_mismatch")
    for face in boundary:
        a, b, c = positions[list(face)]
        if np.linalg.norm(np.cross(b - a, c - a)) == 0.0:
            raise error(status, "zero_area")
    return np.asarray([face_to_index[face] for face in boundary], dtype=np.int32), referenced


def _validate_and_collect_shapes(model: Model, target_bodies: np.ndarray) -> np.ndarray:
    """Validate participating SDF assets without changing default collision behavior."""
    error, status = MonolithicCollisionPipeline.Error, MonolithicCollisionPipeline.Status
    flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
    shapes = np.flatnonzero(np.isin(bodies, target_bodies) & ((flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0))
    types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
    indices = model._shape_sdf_index.numpy()
    descriptors = model._texture_sdf_data.numpy()
    coarse = model._texture_sdf_coarse_textures
    fine = model._texture_sdf_subgrid_textures
    margins = model.shape_margin.numpy()
    transforms = model.shape_transform.numpy()
    analytic = {GeoType.SPHERE, GeoType.BOX, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE}
    for shape in shapes:
        geo, scale = types[shape], scales[shape]
        if geo == GeoType.ELLIPSOID:
            raise error(status.UNSUPPORTED_SHAPE_SDF_ACCURACY, f"shape_{shape}")
        if geo == GeoType.PLANE:
            if not np.isfinite(scale).all() or np.any(scale[:2] != 0.0):
                raise error(status.UNSUPPORTED_FINITE_PLANE, f"shape_{shape}")
        elif geo in analytic:
            dims = scale[:1] if geo == GeoType.SPHERE else scale[:2] if geo != GeoType.BOX else scale
            if not np.isfinite(scale).all() or np.any(dims <= 0.0):
                raise error(status.INVALID_SHAPE, f"shape_{shape}_dimensions")
        elif geo in (GeoType.MESH, GeoType.CONVEX_MESH):
            index = int(indices[shape])
            if (
                index < 0
                or index >= len(descriptors)
                or coarse is None
                or index >= len(coarse)
                or coarse[index] is None
            ):
                raise error(status.INVALID_SDF_DESCRIPTOR, f"shape_{shape}_unprovisioned")
            descriptor = descriptors[index]
            if (
                not model.device.is_cuda
                or not np.isfinite(descriptor["inv_sdf_dx"]).all()
                or np.any(descriptor["inv_sdf_dx"] <= 0.0)
                or not np.isfinite(descriptor["sdf_box_lower"]).all()
                or not np.isfinite(descriptor["sdf_box_upper"]).all()
                or np.any(descriptor["sdf_box_upper"] <= descriptor["sdf_box_lower"])
                or descriptor["subgrid_start_slots"]["data"] == 0
                or descriptor["subgrid_start_slots"]["ndim"] != 3
                or np.any(descriptor["subgrid_start_slots"]["shape"][:3] <= 0)
                or descriptor["num_subgrids"] < 0
                or int.from_bytes(descriptor["coarse_texture"].tobytes()[:8], "little") == 0
            ):
                raise error(status.INVALID_SDF_DESCRIPTOR, f"shape_{shape}_empty_or_invalid")
            if descriptor["num_subgrids"] > 0:
                fine_handle = int.from_bytes(descriptor["subgrid_texture"].tobytes()[:8], "little")
                if (
                    fine_handle == 0
                    or fine is None
                    or index >= len(fine)
                    or fine[index] is None
                    or fine[index].id != fine_handle
                ):
                    raise error(status.INVALID_SDF_DESCRIPTOR, f"shape_{shape}_missing_fine_texture")
            if not descriptor["scale_baked"]:
                reason = ""
                if not np.isfinite(scale).all():
                    reason = "nonfinite"
                elif np.any(scale == 0.0):
                    reason = "zero"
                elif np.any(scale < 0.0):
                    reason = "mirrored"
                elif not np.all(scale == scale[0]):
                    reason = "nonuniform"
                if reason:
                    raise error(status.UNSUPPORTED_SDF_RUNTIME_SCALE, reason)
            elif not np.isfinite(scale).all():
                raise error(status.INVALID_SHAPE, f"shape_{shape}_nonfinite_baked_scale")
        else:
            raise error(status.INVALID_SHAPE, f"shape_{shape}_unsupported_type")
        if not np.isfinite(transforms[shape]).all() or not np.isclose(
            np.linalg.norm(transforms[shape, 3:]), 1.0, atol=1e-5
        ):
            raise error(status.INVALID_SHAPE, f"shape_{shape}_transform")
        margin = margins[shape]
        if not np.isfinite(margin) or margin < 0.0:
            raise error(status.INVALID_SHAPE, f"shape_{shape}_margin")
    return shapes.astype(np.int32)


def _build_face_shape_pairs(model: Model, face_indices: np.ndarray, shape_indices: np.ndarray) -> np.ndarray:
    """Build stable face-major pairs after the single participating world gate."""
    particles = np.unique(model.tet_indices.numpy())
    labels = np.concatenate(
        (model.particle_world.numpy()[particles], model.body_world.numpy(), model.shape_world.numpy()[shape_indices])
    )
    if len(np.unique(labels)) != 1 or np.any(labels < -1) or np.any(labels >= model.world_count):
        raise MonolithicCollisionPipeline.Error(
            MonolithicCollisionPipeline.Status.INVALID_WORLD, "participating_labels"
        )
    return np.asarray(
        [(int(face), int(shape)) for face in face_indices for shape in shape_indices], dtype=np.int32
    ).reshape((-1, 2))


@wp.kernel(enable_backward=False)
def create_monolithic_p1q3_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_flags: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[int],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    r_soft: float,
    soft_contact_gap: float,
    soft_contact_max: int,
    soft_contact_count: wp.array[int],
    soft_contact_tids: wp.array[int],
    out_particle: wp.array[int],
    out_indices: wp.array[wp.vec3i],
    out_barycentric: wp.array[wp.vec3],
    out_shape: wp.array[int],
    out_body_pos: wp.array[wp.vec3],
    out_body_vel: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
    out_status: wp.array[int],
):
    pair_id = wp.tid()
    pair = face_pairs[pair_id]
    face, shape = pair[0], pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    ia, ib, ic = tri_indices[face, 0], tri_indices[face, 1], tri_indices[face, 2]
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    a = wp.transform_point(X_sw, particle_q[ia])
    b = wp.transform_point(X_sw, particle_q[ib])
    c = wp.transform_point(X_sw, particle_q[ic])
    centroid = (a + b + c) / 3.0
    lower, _centroid_phi, _centroid_grad = eval_shape_sdf(
        shape_type[shape], shape_scale[shape], centroid, shape_sdf_index[shape], texture_sdf_table
    )
    reach = wp.max(wp.length(a - centroid), wp.max(wp.length(b - centroid), wp.length(c - centroid)))
    threshold = r_soft + shape_margin[shape] + soft_contact_gap
    if not wp.isfinite(lower) or not wp.isfinite(reach):
        wp.atomic_max(out_status, 0, int(MonolithicCollisionPipeline.Status.INVALID_SDF_GRADIENT))
        return
    if lower > threshold + reach:
        return
    for slot in range(3):
        bary = wp.vec3(1.0 / 6.0)
        bary[slot] = 2.0 / 3.0
        x = bary[0] * a + bary[1] * b + bary[2] * c
        _lower, phi, grad = eval_shape_sdf(
            shape_type[shape], shape_scale[shape], x, shape_sdf_index[shape], texture_sdf_table
        )
        grad_length = wp.length(grad)
        if not wp.isfinite(grad_length) or grad_length == 0.0 or not wp.isfinite(phi):
            wp.atomic_max(out_status, 0, int(MonolithicCollisionPipeline.Status.INVALID_SDF_GRADIENT))
            continue
        grad = grad / grad_length
        if phi - r_soft - shape_margin[shape] <= soft_contact_gap:
            body_pos = wp.transform_point(X_bs, x - phi * grad)
            world_normal = wp.transform_vector(X_ws, grad)
            normal_length = wp.length(world_normal)
            if not wp.isfinite(wp.length(body_pos)) or not wp.isfinite(normal_length) or normal_length == 0.0:
                wp.atomic_max(out_status, 0, int(MonolithicCollisionPipeline.Status.INVALID_SDF_GRADIENT))
                continue
            record = wp.atomic_add(soft_contact_count, 0, 1)
            if record >= soft_contact_max:
                wp.atomic_max(out_status, 0, int(MonolithicCollisionPipeline.Status.CONTACT_CAPACITY_OVERFLOW))
                continue
            soft_contact_tids[3 * pair_id + slot] = record
            out_particle[record] = -1
            out_indices[record] = wp.vec3i(ia, ib, ic)
            out_barycentric[record] = bary
            out_shape[record] = shape
            out_body_pos[record] = body_pos
            out_body_vel[record] = wp.vec3(0.0)
            out_normal[record] = world_normal / normal_length


class MonolithicCollisionPipeline:
    """Sample tet boundary faces against articulation SDFs with fixed P1Q3 quadrature.

    Args:
        model: Model containing the target tetrahedra and articulated rigid bodies.
        soft_contact_gap: Nonnegative detection distance [m], independent of physical margins.

    .. experimental::

        This pipeline is part of :mod:`newton.solvers.experimental.monolithic` and may
        change without notice. It produces only face records and supports a single
        participating world. Texture SDF queries require CUDA.
    """

    class Status(enum.IntEnum):
        """Identify configuration or collision failures."""

        NONE = 0
        INVALID_BOUNDARY = 1
        INVALID_WORLD = 2
        UNSUPPORTED_PARTICLE_RADIUS = 3
        UNSUPPORTED_SHAPE_SDF_ACCURACY = 4
        UNSUPPORTED_FINITE_PLANE = 5
        UNSUPPORTED_SDF_RUNTIME_SCALE = 6
        INVALID_SDF_DESCRIPTOR = 7
        INVALID_SHAPE = 8
        CONTACT_PROVENANCE = 9
        CONTACT_CAPACITY = 10
        INVALID_SDF_GRADIENT = 11
        CONTACT_CAPACITY_OVERFLOW = 12
        INVALID_MODEL = 13
        INVALID_STATE = 14
        UNSUPPORTED_COLLISION_DT = 15

    class Error(ValueError):
        """Report a structured collision failure with a specific subreason."""

        def __init__(self, status: MonolithicCollisionPipeline.Status, subreason: str):
            self.status = status
            self.subreason = subreason
            super().__init__(f"{status.name}: {subreason}")

    def __init__(self, model: Model, *, soft_contact_gap: float = 0.01):
        if not math.isfinite(soft_contact_gap) or soft_contact_gap < 0.0 or soft_contact_gap > np.finfo(np.float32).max:
            raise self.Error(self.Status.INVALID_SHAPE, "soft_contact_gap_must_be_finite_nonnegative")
        if model.requires_grad:
            raise self.Error(self.Status.INVALID_MODEL, "requires_grad")
        self.model = model
        self._device = model.device
        self._soft_contact_gap = float(np.float32(soft_contact_gap))
        faces, referenced = _validate_and_collect_boundary_faces(model)
        radii = model.particle_radius.numpy()[referenced]
        reason = ""
        if not np.isfinite(radii).all():
            reason = "nonfinite"
        elif np.any(radii < 0.0):
            reason = "negative"
        elif not np.all(radii == radii[0]):
            reason = "nonuniform"
        if reason:
            raise self.Error(self.Status.UNSUPPORTED_PARTICLE_RADIUS, reason)
        self._r_soft = float(radii[0])
        target_bodies = np.unique(model.joint_child.numpy())
        shapes = _validate_and_collect_shapes(model, target_bodies)
        pairs = _build_face_shape_pairs(model, faces, shapes)
        self._face_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)
        self._status = wp.zeros(1, dtype=int, device=model.device)
        self._token = object()
        self._contact_contracts = WeakKeyDictionary()
        self._requested_attributes = frozenset(model.get_requested_contact_attributes())
        self._model_arrays = {
            name: getattr(model, name)
            for name in (
                "tet_indices",
                "tri_indices",
                "particle_q",
                "particle_radius",
                "particle_world",
                "body_world",
                "joint_child",
                "shape_body",
                "shape_world",
                "shape_type",
                "shape_flags",
                "shape_transform",
                "shape_scale",
                "shape_margin",
                "_shape_sdf_index",
                "_texture_sdf_data",
            )
        }

    @property
    def rigid_contact_max(self) -> int:
        """Maximum rigid-rigid record count (always zero)."""
        return 0

    @property
    def soft_contact_pair_count(self) -> int:
        """Number of fixed face-shape pairs."""
        return self._face_pairs.shape[0]

    @property
    def soft_contact_max(self) -> int:
        """Maximum face record count, three per pair."""
        return 3 * self.soft_contact_pair_count

    @property
    def soft_contact_tids_size(self) -> int:
        """Number of stable quadrature replay slots."""
        return self.soft_contact_max

    @property
    def r_soft(self) -> float:
        """Common stored particle radius [m]."""
        return self._r_soft

    def _validate_model(self) -> None:
        if self.model.device != self._device or self.model.requires_grad:
            raise self.Error(self.Status.INVALID_MODEL, "changed_device_or_grad")
        if any(getattr(self.model, name) is not value for name, value in self._model_arrays.items()):
            raise self.Error(self.Status.INVALID_MODEL, "replaced_model_array")
        if frozenset(self.model.get_requested_contact_attributes()) != self._requested_attributes:
            raise self.Error(self.Status.CONTACT_PROVENANCE, "changed_requested_attributes")

    def contacts(self) -> Contacts:
        """Allocate a contacts buffer owned by this pipeline."""
        self._validate_model()
        contacts = Contacts(
            0,
            self.soft_contact_max,
            soft_contact_tids_size=self.soft_contact_tids_size,
            device=self.model.device,
            requested_attributes=set(self._requested_attributes),
        )
        self.model._add_custom_attributes(contacts, Model.AttributeAssignment.CONTACT, requires_grad=False)
        contacts._enable_rigid_soft_full_surface_contact = True
        contacts._monolithic_pipeline_token = self._token
        contacts._monolithic_p1q3 = True
        self._contact_contracts[contacts] = {
            name: (value, value.shape, value.dtype, value.ptr, value.strides, value.requires_grad)
            for name, value in vars(contacts).items()
            if isinstance(value, wp.array)
        }
        return contacts

    def validate_contacts(self, contacts: Contacts) -> None:
        """Reject foreign, stale or malformed buffers before mutation."""
        self._validate_model()
        if (
            contacts not in self._contact_contracts
            or getattr(contacts, "_monolithic_pipeline_token", None) is not self._token
            or getattr(contacts, "_monolithic_p1q3", False) is not True
            or not contacts._enable_rigid_soft_full_surface_contact
            or contacts.requires_grad
            or contacts.contact_matching
            or contacts.clear_buffers
            or (contacts.force is not None) != ("force" in self._requested_attributes)
        ):
            raise self.Error(self.Status.CONTACT_PROVENANCE, "foreign_contacts")
        if contacts.rigid_contact_max != 0 or contacts.soft_contact_max != self.soft_contact_max:
            raise self.Error(self.Status.CONTACT_CAPACITY, "capacity")
        arrays = self._contact_contracts[contacts]
        if {name for name, value in vars(contacts).items() if isinstance(value, wp.array)} != set(arrays):
            raise self.Error(self.Status.CONTACT_PROVENANCE, "changed_attributes")
        for name, (value, shape, dtype, ptr, strides, requires_grad) in arrays.items():
            if (
                getattr(contacts, name) is not value
                or value.device != self.model.device
                or value.shape != shape
                or value.dtype != dtype
                or value.ptr != ptr
                or value.strides != strides
                or value.requires_grad != requires_grad
            ):
                raise self.Error(self.Status.CONTACT_CAPACITY, name)

    def collide(self, state: State, contacts: Contacts, *, dt: float = 0.0) -> None:
        """Refresh face contacts and advance the buffer generation exactly once.

        Args:
            state: Candidate positions and body transforms.
            contacts: A buffer created by this pipeline.
            dt: Must be zero [s]; speculative collision is unsupported.

        Raises:
            Error: On invalid input, SDF gradient or contact capacity overflow.
        """
        self.validate_contacts(contacts)
        if not math.isfinite(dt) or dt != 0.0:
            raise self.Error(self.Status.UNSUPPORTED_COLLISION_DT, "dt_must_be_zero")
        for name, count, dtype in (
            ("particle_q", self.model.particle_count, wp.vec3),
            ("body_q", self.model.body_count, wp.transform),
        ):
            array = getattr(state, name, None)
            if array is None or array.device != self.model.device or array.dtype != dtype or array.shape != (count,):
                raise self.Error(self.Status.INVALID_STATE, name)
        contacts.clear(bump_generation=True)
        contacts.soft_contact_tids.fill_(-1)
        self._status.zero_()
        if self.soft_contact_pair_count:
            m = self.model
            wp.launch(
                create_monolithic_p1q3_face_contacts,
                dim=self.soft_contact_pair_count,
                inputs=[
                    self._face_pairs,
                    state.particle_q,
                    m.tri_indices,
                    m.shape_body,
                    m.shape_type,
                    m.shape_flags,
                    m.shape_transform,
                    m.shape_scale,
                    state.body_q,
                    m._shape_sdf_index,
                    m._texture_sdf_data,
                    m.shape_margin,
                    self.r_soft,
                    self._soft_contact_gap,
                    self.soft_contact_max,
                    contacts.soft_contact_count,
                    contacts.soft_contact_tids,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self._status,
                ],
                device=m.device,
            )
        status = int(self._status.numpy()[0])
        if status:
            # Invalidate the attempted stream without a second generation bump or clear call.
            contacts.soft_contact_count.zero_()
            contacts.soft_contact_tids.fill_(-1)
            raise self.Error(self.Status(status), "collision_query")
