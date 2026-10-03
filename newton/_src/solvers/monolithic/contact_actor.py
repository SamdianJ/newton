# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native directed actor geometry and force projection; local laws live in contact."""

from typing import Any

import numpy as np
import warp as wp

from ...geometry import GeoType
from ...geometry.sdf_texture import TextureSDFData
from .contact import _factors, _failure, _geometry_type, _law, _validate_start
from .contact_geometry import _shape64
from .smooth_sdf import smooth_query


@wp.func
def _body_pose(
    body: int,
    kinds: wp.array[int],
    indices: wp.array[int],
    body_com: wp.array[Any],
    poses: wp.array[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
):
    result = type(poses[0])(type(rigid_com[0])(0.0), type(rigid_rotation[0])(0.0, 0.0, 0.0, 1.0))
    if body >= 0:
        local = indices[body]
        if kinds[body] == 1:
            rotation = rigid_rotation[local]
            result = type(poses[0])(rigid_com[local] - wp.quat_rotate(rotation, body_com[body]), rotation)
        else:
            result = poses[local]
    return result


@wp.func
def _center(body: int, kinds: wp.array[int], indices: wp.array[int], com: wp.array[Any], rigid_com: wp.array[Any]):
    local = indices[body]
    result = com[0]
    if kinds[body] == 1:
        result = rigid_com[local]
    else:
        result = com[local]
    return result


@wp.func
def _point_column(
    body: int,
    column: int,
    point: Any,
    kinds: wp.array[int],
    indices: wp.array[int],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
):
    result = type(point)(0.0)
    local = indices[body]
    if kinds[body] == 0:
        result = jt[local, column] + wp.cross(jw[local, column], point - com[local])
    else:
        basis = type(point)(0.0)
        basis[column % 3] = type(point[0])(1.0)
        result = basis
        if column >= 3:
            result = wp.cross(basis, point - rigid_com[local])
    return result


@wp.kernel
def _shape_bounds(
    selected: wp.array[int],
    centers: wp.array[Any],
    extents: wp.array[Any],
    expansion: wp.array[Any],
    kinds: wp.array[int],
    indices: wp.array[int],
    body_com: wp.array[Any],
    poses: wp.array[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
    bodies: wp.array[int],
    transforms: wp.array[Any],
    lower: wp.array[Any],
    upper: wp.array[Any],
):
    i = wp.tid()
    shape = selected[i]
    pose = _body_pose(bodies[shape], kinds, indices, body_com, poses, rigid_com, rigid_rotation) * transforms[shape]
    center = wp.transform_point(pose, centers[i])
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose))
    extent = type(center)(expansion[i])
    for row in range(3):
        for col in range(3):
            extent[row] += wp.abs(rotation[row, col]) * extents[i][col]
    lower[i] = center - extent
    upper[i] = center + extent


@wp.kernel
def _tet_bounds(x: wp.array[Any], expansion: Any, bounds: wp.array[Any]):
    lo, hi = x[0] + type(x[0])(0.0), x[0] + type(x[0])(0.0)
    for i in range(x.shape[0]):
        for axis in range(3):
            lo[axis] = wp.min(lo[axis], x[i][axis])
            hi[axis] = wp.max(hi[axis], x[i][axis])
    bounds[0] = lo - type(lo)(expansion)
    bounds[1] = hi + type(hi)(expansion)


@wp.kernel
def _actor_geometry(
    x: wp.array[Any],
    poses: wp.array[Any],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
    rigid_rotation: wp.array[Any],
    body_com: wp.array[Any],
    kinds: wp.array[int],
    indices: wp.array[int],
    bodies: wp.array[int],
    transforms: wp.array[Any],
    source_shapes: wp.array[int],
    target_shapes: wp.array[int],
    faces: wp.array[int],
    slots: wp.array[int],
    local_points: wp.array[Any],
    local_normals: wp.array[Any],
    triangles: wp.array2d[int],
    types: wp.array[int],
    scales: wp.array[Any],
    sdf_indices: wp.array[int],
    table: wp.array[TextureSDFData],
    smooth_table: wp.array[Any],
    support_offsets: wp.array[int],
    selectors: wp.array[wp.vec2i],
    jacobians: wp.array[Any],
    key_offset: int,
    quaternion_tolerance: Any,
    out: Any,
    world_points: wp.array[Any],
    flags: wp.array[int],
):
    key = wp.tid()
    source, target = source_shapes[key], target_shapes[key]
    point, normal = type(x[0])(0.0), type(x[0])(0.0)
    if source < 0:
        face, quadrature = faces[key], slots[key]
        for node in range(3):
            bary = type(x[0][0])(1.0) / type(x[0][0])(6.0)
            if node == quadrature:
                bary = type(x[0][0])(2.0) / type(x[0][0])(3.0)
            point += bary * x[triangles[face, node]]
        normal = wp.cross(x[triangles[face, 1]] - x[triangles[face, 0]], x[triangles[face, 2]] - x[triangles[face, 0]])
    else:
        pose = (
            _body_pose(bodies[source], kinds, indices, body_com, poses, rigid_com, rigid_rotation) * transforms[source]
        )
        point = wp.transform_point(pose, local_points[key])
        normal = wp.transform_vector(pose, local_normals[key])
    length = wp.length(normal)
    if not wp.isfinite(length) or length <= type(x[0][0])(0.0):
        _failure(flags, key + key_offset, 2)
        return
    target_pose = (
        _body_pose(bodies[target], kinds, indices, body_com, poses, rigid_com, rigid_rotation) * transforms[target]
    )
    q = wp.transform_get_rotation(target_pose)
    if not wp.isfinite(wp.length(q)) or wp.abs(wp.length(q) - type(x[0][0])(1.0)) > quaternion_tolerance:
        _failure(flags, key + key_offset, 2)
        return
    rotation = wp.quat_to_matrix(q)
    y = wp.transpose(rotation) * (point - wp.transform_get_translation(target_pose))
    d, g = _shape64(types[target], scales[target], y, sdf_indices[target], table)
    out.uncertified[key] = 0
    if types[target] == GeoType.MESH:
        index = sdf_indices[target]
        field = smooth_table[index]
        query = y
        if not table[index].scale_baked:
            query /= scales[target][0]
        d, g, _unused_hessian = smooth_query(field, query)
        if not table[index].scale_baked:
            d *= scales[target][0]
        for axis in range(3):
            if query[axis] <= field.lower[axis] or query[axis] >= field.upper[axis]:
                out.uncertified[key] = 1
    if not wp.isfinite(d) or not wp.isfinite(wp.length(g)) or not wp.isfinite(wp.length(point)):
        _failure(flags, key + key_offset, 1)
        return
    world_points[key] = point
    out.y[key] = y
    out.gradient[key] = g
    out.distance[key] = d
    out.rotation[key] = rotation
    out.soft_normal[key] = wp.transpose(rotation) * normal / length
    for component in range(3):
        factor = 3 * (key + key_offset) + component
        for index in range(support_offsets[factor], support_offsets[factor + 1]):
            selector = selectors[index]
            j = selector[1]
            value = type(point)(0.0)
            if selector[0] == 0:
                if source < 0:
                    bary = type(point[0])(1.0) / type(point[0])(6.0)
                    if j / 3 == slots[key]:
                        bary = type(point[0])(2.0) / type(point[0])(3.0)
                    value[j % 3] = bary
                else:
                    value = _point_column(bodies[source], j, point, kinds, indices, com, jt, jw, rigid_com)
            else:
                value = -_point_column(bodies[target], j, point, kinds, indices, com, jt, jw, rigid_com)
            value = wp.transpose(rotation) * value
            if not wp.isfinite(wp.length(value)):
                _failure(flags, key + key_offset, 1)
            jacobians[index] = value


@wp.kernel
def _actor_forces(
    geometry: Any,
    points: wp.array[Any],
    local_r: wp.array[Any],
    key_offset: int,
    source_shapes: wp.array[int],
    target_shapes: wp.array[int],
    bodies: wp.array[int],
    kinds: wp.array[int],
    indices: wp.array[int],
    com: wp.array[Any],
    rigid_com: wp.array[Any],
    faces: wp.array[int],
    slots: wp.array[int],
    triangles: wp.array2d[int],
    support_offsets: wp.array[int],
    columns: wp.array[int],
    jacobians: wp.array[Any],
    model_dofs: wp.array[int],
    qforce: wp.array[Any],
    pforce: wp.array[Any],
    bforce: wp.array[Any],
    pairs: wp.array2d[Any],
):
    key = wp.tid()
    p = points[key]
    local = local_r[key + key_offset]
    world = geometry.rotation[key] * local
    for end in range(2):
        shape = int(source_shapes[key])
        force = -world
        if end == 1:
            shape = int(target_shapes[key])
            force = world
        torque = wp.cross(p, force)
        pairs[key, end] = type(bforce[0])(force[0], force[1], force[2], torque[0], torque[1], torque[2])
        if shape >= 0:
            body = bodies[shape]
            if body >= 0:
                torque = wp.cross(p - _center(body, kinds, indices, com, rigid_com), force)
                wp.atomic_add(
                    bforce, body, type(bforce[0])(force[0], force[1], force[2], torque[0], torque[1], torque[2])
                )
        else:
            for node in range(3):
                bary = type(p[0])(1.0) / type(p[0])(6.0)
                if node == slots[key]:
                    bary = type(p[0])(2.0) / type(p[0])(3.0)
                wp.atomic_add(pforce, triangles[faces[key], node], bary * force)
    for index in range(support_offsets[3 * (key + key_offset)], support_offsets[3 * (key + key_offset) + 1]):
        dof = model_dofs[columns[index]]
        if dof >= 0:
            wp.atomic_add(qforce, dof, -wp.dot(jacobians[index], local))


class ActorContact:
    """Own fixed new-query arrays and native scratch appended to the old contact batch."""

    def __init__(self, contact, samples):
        self.contact, self.samples = contact, samples
        self.count = samples.count
        self._bound = {}

        def upload(name, values, dtype):
            value = wp.array(values, dtype=dtype, device=contact.model.device)
            setattr(self, name, value)
            self._bound[name] = (value, value.numpy().copy())
            return value

        for name in ("source_shapes", "targets", "faces", "slots", "body_kind", "body_local"):
            upload(name, getattr(samples, name), int)
        for name in ("points", "normals"):
            upload(name, getattr(samples, name), contact.vector)
        for name in ("weights", "offsets"):
            upload(name, getattr(samples, name).astype(contact.numpy_dtype), contact.dtype)
        upload("body_com", contact.model.body_com.numpy().astype(contact.numpy_dtype), contact.vector)
        offsets, columns, _ = contact.factor_support
        upload("support_offsets", offsets, int)
        upload("columns", columns, int)
        selector_values = [(0, 0)] * int(offsets[3 * contact.legacy_count])
        for selectors in samples.selectors:
            for _ in range(3):
                selector_values.extend(selectors)
        upload("selectors", np.asarray(selector_values, dtype=np.int32).reshape(-1, 2), wp.vec2i)
        self.jacobians = wp.zeros(len(columns), dtype=contact.vector, device=contact.model.device)
        upload("key_ids", np.arange(self.count, dtype=np.int32), int)
        mapping = np.full(contact.layout.scalar_dof_count, -1, dtype=np.int32)
        for i, global_row in enumerate(samples.actors.maps["joint_dof_to_global"]):
            if global_row >= 0:
                mapping[global_row] = i
        upload("model_dofs", mapping, int)
        self.world_points = wp.zeros(self.count, dtype=contact.vector, device=contact.model.device)
        self.pairs = wp.zeros(
            (self.count, 2),
            dtype=wp.spatial_vectord if contact.dtype == wp.float64 else wp.spatial_vector,
            device=contact.model.device,
        )
        self.current, self.start = self.geometry(), self.geometry()
        # Bounds are diagnostic only: frozen friction still queries every key.
        selected, centers, extents, expansion = [], [], [], []
        used = (set(samples.source_shapes) | set(samples.targets)) - {-1}
        for shape in sorted(used):
            geo, scale = contact.values["shape_type"][shape], contact.values["shape_scale"][shape].astype(float)
            if geo == GeoType.PLANE:
                continue
            center, extent = np.zeros(3), scale.copy()
            if geo == GeoType.SPHERE:
                extent[:] = scale[0]
            elif geo == GeoType.MESH:
                vertices = np.asarray(samples.meshes[shape][1], dtype=float) * scale
                center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
                extent = (vertices.max(axis=0) - vertices.min(axis=0)) / 2
            selected.append(shape)
            centers.append(center)
            extents.append(extent)
            expansion.append(
                float(contact.values["shape_margin"][shape]) + contact.parameters[3] + contact.parameters[4]
            )
        upload("bounded_shapes", selected, int)
        upload("bound_centers", np.asarray(centers).reshape(-1, 3), contact.vector)
        upload("bound_extents", np.asarray(extents).reshape(-1, 3), contact.vector)
        upload("bound_expansion", expansion, contact.dtype)
        self.bound_lower = wp.empty(len(selected), dtype=contact.vector, device=contact.model.device)
        self.bound_upper = wp.empty_like(self.bound_lower)
        self.tet_bound = wp.empty(2, dtype=contact.vector, device=contact.model.device)
        for shape in samples.targets_used:
            if contact.values["shape_type"][shape] == GeoType.MESH:
                index = int(contact.values["_shape_sdf_index"][shape])
                if (
                    contact.model.device.is_cpu
                    or contact._descriptors is None
                    or index < 0
                    or index >= len(contact._descriptors)
                ):
                    raise ValueError("Actor mesh target requires CUDA texture SDF")
                if contact.smooth_sdf is None or contact.smooth_sdf.fields[index].representation != "c2-bspline":
                    raise ValueError("Actor mesh target requires bound C2 SDF")
                scale = contact.values["shape_scale"][shape]
                if not contact._descriptors["scale_baked"][index] and (scale[0] <= 0 or not np.all(scale == scale[0])):
                    raise ValueError("Unbaked actor mesh scale must be positive and uniform")

    def geometry(self):
        c = self.contact
        value = _geometry_type(c.dtype)()
        for name in ("y", "gradient", "soft_normal"):
            setattr(value, name, wp.zeros(self.count, dtype=c.vector, device=c.model.device))
        value.distance = wp.zeros(self.count, dtype=c.dtype, device=c.model.device)
        value.rotation = wp.zeros(self.count, dtype=c.matrix, device=c.model.device)
        value.uncertified = wp.zeros(self.count, dtype=int, device=c.model.device)
        value.jacobian_q = wp.empty((self.count, 0), dtype=c.vector, device=c.model.device)
        return value

    def buffers(self):
        return [
            value
            for owner in (self, self.start, self.current)
            for value in vars(owner).values()
            if isinstance(value, wp.array)
        ]

    def validate_identity(self):
        self.samples.validate_identity()
        for name, (value, expected) in self._bound.items():
            if getattr(self, name) is not value or not np.array_equal(value.numpy(), expected):
                raise ValueError(f"Actor contact map changed: {name}")

    def query(self, x, poses, com, jt, jw, rigid_com, rigid_rotation, output, flags):
        c = self.contact
        if rigid_com is None or rigid_rotation is None:
            raise ValueError("Actor contact requires authoritative native rigid state")
        for value, dtype, shape in (
            (rigid_com, c.vector, (len(self.samples.actors.actors) - 2,)),
            (rigid_rotation, wp.quatd if c.dtype == wp.float64 else wp.quat, (len(self.samples.actors.actors) - 2,)),
        ):
            if not isinstance(value, wp.array) or (value.dtype, value.shape, value.device) != (
                dtype,
                shape,
                c.model.device,
            ):
                raise ValueError("Invalid native rigid contact state")
        wp.launch(
            _shape_bounds,
            self.bounded_shapes.size,
            [
                self.bounded_shapes,
                self.bound_centers,
                self.bound_extents,
                self.bound_expansion,
                self.body_kind,
                self.body_local,
                self.body_com,
                poses,
                rigid_com,
                rigid_rotation,
                c.bodies,
                c.transforms,
                self.bound_lower,
                self.bound_upper,
            ],
            device=c.model.device,
        )
        if np.any(self.samples.source_shapes < 0):
            wp.launch(
                _tet_bounds, 1, [x, c.dtype(c.parameters[3] + c.parameters[4]), self.tet_bound], device=c.model.device
            )
        wp.launch(
            _actor_geometry,
            self.count,
            [
                x,
                poses,
                com,
                jt,
                jw,
                rigid_com,
                rigid_rotation,
                self.body_com,
                self.body_kind,
                self.body_local,
                c.bodies,
                c.transforms,
                self.source_shapes,
                self.targets,
                self.faces,
                self.slots,
                self.points,
                self.normals,
                c.triangles,
                c.types,
                c.scales,
                c.sdf_indices,
                c.table,
                c.smooth_table,
                self.support_offsets,
                self.selectors,
                self.jacobians,
                c.legacy_count,
                c.dtype(c.quaternion_norm_tolerance),
                output,
                self.world_points,
                flags,
            ],
            device=c.model.device,
        )

    def begin_step(self, *state, rigid_com, rigid_rotation):
        c = self.contact
        self.query(*state, rigid_com, rigid_rotation, self.start, c.start_flags)
        wp.launch(
            _validate_start,
            self.count,
            [
                self.start,
                self.key_ids,
                self.offsets,
                c.dtype(0),
                c.dtype(c.parameters[3]),
                int(c._alignment_policy == "mask"),
                c.start_flags,
            ],
            device=c.model.device,
        )

    def evaluate(self, *state, rigid_com, rigid_rotation, dt):
        c = self.contact
        self.query(*state, rigid_com, rigid_rotation, self.current, c.flags)
        wp.launch(
            _law,
            self.count,
            [
                self.current,
                self.start,
                self.targets,
                c.margins,
                c.triangles,
                self.weights,
                c.mapping,
                c.layout.q_scalar_offset,
                c.layout.x_scalar_offset,
                c.dtype(0),
                c.dtype(c.parameters[0]),
                c.dtype(c.parameters[3]),
                c.dtype(c.parameters[1]),
                c.dtype(c.parameters[2] * dt),
                ("C-EE", "C-IE", "C-II").index(c.contact_model),
                int(c._alignment_policy == "mask"),
                c.flags,
                c.energy,
                c.residual,
                c.normal,
                c.tangent,
                c.weights,
                c.local_residual,
                c.omitted,
                c.mask_seen,
                1,
                c.legacy_count,
                self.offsets,
                self.support_offsets,
                self.columns,
                self.jacobians,
            ],
            device=c.model.device,
        )

    def assemble(self, assembly):
        c = self.contact
        wp.launch(
            _factors,
            self.count,
            [
                self.current,
                self.targets,
                c.normal,
                c.tangent,
                c.weights,
                c.flags,
                assembly.contact_factors,
                assembly.global_scalar_triplets,
                1,
                c.legacy_count,
                self.jacobians,
            ],
            device=c.model.device,
        )

    def publish(self, state, kinematics):
        c, k = self.contact, kinematics
        history = state._monolithic_coupled
        wp.launch(
            _actor_forces,
            self.count,
            [
                self.current,
                self.world_points,
                c.local_residual,
                c.legacy_count,
                self.source_shapes,
                self.targets,
                c.bodies,
                self.body_kind,
                self.body_local,
                k.com,
                history.rigid_com,
                self.faces,
                self.slots,
                c.triangles,
                self.support_offsets,
                self.columns,
                self.jacobians,
                self.model_dofs,
                history.contact_joint_force,
                history.contact_particle_force,
                history.contact_body_force,
                self.pairs,
            ],
            device=c.model.device,
        )
        return self.diagnostics()

    def diagnostics(self):
        forces = self.pairs.numpy()
        result = {}
        for key, (source, target, _face, _slot) in enumerate(self.samples.keys):
            label = (source, target) if source < 0 else tuple(sorted((source, target)))
            reverse = source >= 0 and source != label[0]
            pair = result.setdefault(label, np.zeros((2, 6)))
            pair += forces[key, ::-1] if reverse else forces[key]
        bounds = {
            int(shape): [lo.tolist(), hi.tolist()]
            for shape, lo, hi in zip(
                self.bounded_shapes.numpy(), self.bound_lower.numpy(), self.bound_upper.numpy(), strict=True
            )
        }
        if np.any(self.samples.source_shapes < 0):
            bounds[-1] = self.tet_bound.numpy().tolist()
        return [
            {
                "source": source,
                "target": target,
                "kind": self.samples.pair_kinds[source, target],
                "world_wrench": value.tolist(),
                "bounds": {shape: bounds[shape] for shape in (source, target) if shape in bounds},
            }
            for (source, target), value in result.items()
        ]
