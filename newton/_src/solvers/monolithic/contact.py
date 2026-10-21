# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private full-pair contact components; no step or publication policy.

The C-IE law uses current and frozen start geometry, not historical spring/Huber
xi state. Contact factors are shared through linear; geometry and smooth SDF
queries do not depend on the historical contact module.
"""

import enum
import math
from functools import cache
from typing import Any

import numpy as np
import warp as wp

from ...geometry import GeoType, ShapeFlags
from ...geometry.sdf_texture import TextureSDFData
from .contact_sampling import ContactSamples, _checked_contact_capacity
from .linear_types import MonolithicLinearLayout
from .smooth_sdf import _sdf_type


class ContactStatus(enum.IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    INVALID_GEOMETRY = 2
    CAPACITY = 3
    INVALID_ASSEMBLY = 4
    UNCERTIFIED_BOUNDARY = 5


@cache
def _geometry_type(dtype):
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    matrix = wp.mat33d if dtype == wp.float64 else wp.mat33

    @wp.struct
    class Geometry:
        y: wp.array[vector]
        # Target-local slip y - y_n formed from the step increments.
        slip: wp.array[vector]
        # Length by which rounding the local point y can move the gap.
        rounding: wp.array[dtype]
        gradient: wp.array[vector]
        distance: wp.array[dtype]
        rotation: wp.array[matrix]
        jacobian_q: wp.array2d[vector]
        soft_normal: wp.array[vector]
        uncertified: wp.array[int]

    return Geometry


_Geometry = _geometry_type(wp.float64)


@wp.func
def _failure(flags: wp.array[int], key: int, status: int):
    wp.atomic_max(flags, 0, status)
    wp.atomic_min(flags, 1, key)


@wp.func
def _penalty64(gap: Any, stiffness: Any, smoothing: Any):
    u = -gap
    if smoothing == type(gap)(0.0) and u <= type(gap)(0.0):
        return type(gap)(0.0), type(gap)(0.0), type(gap)(0.0)
    p, dp, ddp = type(gap)(0.0), type(gap)(0.0), type(gap)(0.0)
    if u >= smoothing:
        p, dp = u, type(gap)(1.0)
    elif u > -smoothing:
        t = (u + smoothing) / (type(gap)(2.0) * smoothing)
        p = smoothing * t * t * t * (type(gap)(2.0) - t)
        dp = t * t * (type(gap)(3.0) - type(gap)(2.0) * t)
        ddp = type(gap)(3.0) * t * (type(gap)(1.0) - t) / smoothing
    return stiffness * p * p / type(gap)(2.0), stiffness * p * dp, stiffness * (dp * dp + p * ddp)


@wp.kernel
def _reset_flags(flags: wp.array[int]):
    flags[0] = 0
    flags[1] = 2147483647


@wp.kernel
def _validate_start(
    geometry: Any,
    shapes: wp.array[int],
    margins: wp.array[Any],
    radius: Any,
    smoothing: Any,
    alignment_mask: int,
    flags: wp.array[int],
    bounded: bool,
    batch_count: wp.array[int],
):
    key = wp.tid()
    if bounded and key >= batch_count[0]:
        return
    gap = geometry.distance[key] - radius - margins[shapes[key % shapes.shape[0]]]
    g = geometry.gradient[key]
    if gap < smoothing:
        if geometry.uncertified[key] != 0:
            _failure(flags, key, 5)
        if wp.length(g) == type(radius)(0.0) or (
            alignment_mask == 0 and gap < type(radius)(0.0) and wp.dot(geometry.soft_normal[key], g) > type(radius)(0.0)
        ):
            _failure(flags, key, 2)


@wp.kernel
def _law(
    current: Any,
    start: Any,
    shapes: wp.array[int],
    margins: wp.array[Any],
    area: wp.array[Any],
    radius: Any,
    stiffness: Any,
    smoothing: Any,
    mu: Any,
    falloff: Any,
    alignment_mask: int,
    flags: wp.array[int],
    energy: wp.array2d[Any],
    terms: wp.array[Any],
    normal: wp.array[Any],
    tangent: wp.array[Any],
    weights: wp.array2d[Any],
    local_r: wp.array[Any],
    omitted: wp.array2d[Any],
    mask_seen: wp.array[int],
    direct: int,
    key_offset: int,
    gap_offsets: wp.array[Any],
    support_offsets: wp.array[int],
    support_local: wp.array[int],
    jacobians: wp.array[Any],
    batch_count: wp.array[int],
):
    key = wp.tid()
    if direct >= 2 and key >= batch_count[0]:
        return
    if flags[0] != 0:
        return
    face = key / (3 * shapes.shape[0])
    slot = (key / shapes.shape[0]) % 3
    shape = shapes[key % shapes.shape[0]]
    k = stiffness * area[face] / type(radius)(3.0)
    gap = current.distance[key] - radius - margins[shape]
    if direct != 0:
        k = stiffness * area[key]
        gap = current.distance[key] - gap_offsets[key]
        if direct == 2:
            # Bounded legacy geometry keeps the original multiply/divide and
            # two subtractions; changing these would also change the law.
            k /= type(radius)(3.0)
            gap = current.distance[key] - radius - margins[shape]
    output_key = key + key_offset
    g = current.gradient[key]
    u = current.slip[key]
    en, load_normal, curvature = _penalty64(gap, k, smoothing)
    direction = g
    # C-IE: friction load from the gap linearized about the frozen start point.
    _unused, load, _unused_curvature = _penalty64(gap - wp.dot(g, u), k, smoothing)
    if mu == type(radius)(0.0):
        load = type(radius)(0.0)
    if load_normal > type(radius)(0.0) or load > type(radius)(0.0):
        if current.uncertified[key] != 0 or (load > type(radius)(0.0) and start.uncertified[key] != 0):
            _failure(flags, output_key, 5)
            return
        if wp.length(g) == type(radius)(0.0) or (
            load > type(radius)(0.0) and wp.length(direction) == type(radius)(0.0)
        ):
            _failure(flags, output_key, 2)
            return
        if alignment_mask == 0 and gap < type(radius)(0.0) and wp.dot(current.soft_normal[key], g) > type(radius)(0.0):
            _failure(flags, output_key, 2)
            return
    n = type(current.y[0])(0.0)
    if wp.length(direction) > type(radius)(0.0):
        n = wp.normalize(direction)
    v = u - wp.dot(n, u) * n
    rho = wp.length(v)
    psi, factor, radial = type(radius)(0.0), type(radius)(0.0), type(radius)(0.0)
    if load > type(radius)(0.0):
        if rho < falloff:
            psi = rho * rho / falloff - rho * rho * rho / (type(radius)(3.0) * falloff * falloff)
            factor = type(radius)(2.0) / falloff - rho / (falloff * falloff)
            # Evaluate the nonnegative radial curvature without cancellation.
            radial = type(radius)(2.0) * mu * load * (falloff - rho) / (falloff * falloff)
        else:
            psi = rho - falloff / type(radius)(3.0)
            factor = type(radius)(1.0) / rho
    beta = mu * load * factor
    r = -load_normal * g + beta * v
    dissipative = mu * load * psi
    if (
        not wp.isfinite(en)
        or not wp.isfinite(dissipative)
        or not wp.isfinite(wp.length(r))
        or not wp.isfinite(beta)
        or not wp.isfinite(radial)
        or not wp.isfinite(curvature)
    ):
        _failure(flags, output_key, 1)
        return
    if alignment_mask != 0:
        # Select the entire contact term with the law's explicit/implicit collider normal.
        masked = wp.dot(current.soft_normal[key], direction) > type(radius)(0.0)
        if load_normal > type(radius)(0.0) or load > type(radius)(0.0):
            bit = int(1)
            if masked:
                bit = 2
            wp.atomic_or(mask_seen, output_key, bit)
        if masked:
            omitted[output_key, 0] = en
            omitted[output_key, 1] = dissipative
            omitted[output_key, 2] = load_normal
            omitted[output_key, 3] = load
            normal[output_key] = type(current.y[0])(0.0)
            tangent[output_key] = type(current.y[0])(0.0)
            return
    energy[output_key, 0] = en
    energy[output_key, 1] = dissipative
    normal[output_key] = n
    tangent[output_key] = type(current.y[0])(0.0)
    # Reproject before normalization: near zero slip, cancellation can leave
    # a roundoff vector almost parallel to n instead of a tangent direction.
    transverse = wp.cross(n, v)
    transverse_length = wp.length(transverse)
    if transverse_length > type(radius)(0.0):
        tangent[output_key] = wp.cross(transverse / transverse_length, n)
    local_r[output_key] = r
    weights[output_key, 0] = curvature
    weights[output_key, 1] = beta
    weights[output_key, 2] = radial
    if direct >= 2:
        # Bounded producers project actual endpoint support after local law.
        return
    first = support_offsets[3 * output_key]
    world_r = current.rotation[key] * r
    for index in range(first, support_offsets[3 * output_key + 1]):
        value = type(radius)(0.0)
        if direct != 0:
            value = wp.dot(jacobians[index], r)
        elif support_local[index] >= 0:
            value = wp.dot(current.jacobian_q[key, support_local[index]], r)
        else:
            coordinate = -support_local[index] - 1
            weight = type(radius)(1.0) / type(radius)(6.0)
            if coordinate / 3 == slot:
                weight = type(radius)(2.0) / type(radius)(3.0)
            value = weight * world_r[coordinate % 3]
        # Three factor supports share one residual contribution per sample/DoF.
        terms[first / 3 + (index - first)] = value


class ContactObjective:
    """Validate complete contact inputs, optionally retaining only bounded-generator metadata.

    The default owns the original full evaluation storage. The private
    metadata-only construction skips expanded support and geometry tables;
    native evaluation then belongs to the bounded contact batch.
    """

    def __init__(
        self,
        model,
        particle_to_dynamic,
        *,
        q_dof_count,
        normal_stiffness,
        friction_coefficient,
        friction_velocity_scale,
        smoothing_width,
        particle_radius,
        smooth_sdf=None,
        alignment_policy="guard",
        dtype=wp.float64,
        layout=None,
        articulation_actor=None,
        base_triplet_count=0,
        actors=None,
        _materials=None,
    ):
        self._materials = _materials
        if dtype not in (wp.float64, wp.float32):
            raise ValueError("Contact requires float64 or float32")
        self.dtype = dtype
        self.numpy_dtype = np.float64 if dtype == wp.float64 else np.float32
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
        self.transform = wp.transformd if dtype == wp.float64 else wp.transform
        if alignment_policy not in ("guard", "mask"):
            raise ValueError("Unknown contact alignment policy")
        self.alignment_policy = alignment_policy
        self._alignment_policy = alignment_policy
        parameters = (normal_stiffness, friction_coefficient, friction_velocity_scale, smoothing_width, particle_radius)
        native_parameters = np.asarray(parameters, dtype=self.numpy_dtype)
        if not np.isfinite(native_parameters).all() or any(
            original != 0 and converted == 0 for original, converted in zip(parameters, native_parameters, strict=True)
        ):
            raise ValueError("Unrepresentable contact parameters")
        if (
            not all(math.isfinite(v) for v in parameters)
            or normal_stiffness <= 0
            or friction_coefficient < 0
            or smoothing_width < 0
            or particle_radius < 0
            or (friction_coefficient > 0 and friction_velocity_scale <= 0)
        ):
            raise ValueError("Invalid variational contact parameters")
        if type(q_dof_count) is not int or q_dof_count < 0:
            raise ValueError("Invalid joint layout")
        mapping = np.asarray(particle_to_dynamic)
        if (
            mapping.shape != (model.particle_count,)
            or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < -1)
        ):
            raise ValueError("Invalid particle map")
        dynamic = mapping[mapping >= 0]
        if not np.array_equal(np.sort(dynamic), np.arange(len(dynamic))):
            raise ValueError("Particle map must be compact and bijective")
        if not np.array_equal(mapping >= 0, model.particle_inv_mass.numpy() > 0):
            raise ValueError("Particle map must preserve fixed nodes")
        names = (
            "particle_inv_mass",
            "shape_flags",
            "shape_type",
            "shape_scale",
            "shape_body",
            "shape_transform",
            "shape_margin",
            "_shape_sdf_index",
            "tri_indices",
            "particle_q",
        )
        self.values = {
            name: getattr(model, name).numpy().copy()
            if getattr(model, name) is not None
            else np.empty((0, 3), dtype=np.int32)
            for name in names
        }
        v = self.values
        self.actor_samples = (
            None
            if actors is None
            else ContactSamples(
                model,
                actors,
                radius=particle_radius,
                base_triplet_count=base_triplet_count,
                _expand_support=False,
                _materials=_materials,
            )
        )
        shapes = np.flatnonzero(v["shape_flags"] & int(ShapeFlags.COLLIDE_PARTICLES)).astype(np.int32)
        if self.actor_samples is not None:
            shapes = shapes[~np.isin(shapes, self.actor_samples.rigid_shapes)]
        if not np.isin(
            v["shape_type"][shapes], [int(GeoType.PLANE), int(GeoType.SPHERE), int(GeoType.BOX), int(GeoType.MESH)]
        ).all():
            raise ValueError("Unsupported contact shape")
        if _materials is not None:
            shapes = np.empty(0, dtype=np.int32)
        table = model._texture_sdf_data
        descriptors = table.numpy() if table is not None else None
        for i in shapes:
            geo, scale = v["shape_type"][i], v["shape_scale"][i]
            if not np.isfinite(scale).all() or not np.isfinite(v["shape_margin"][i]) or v["shape_margin"][i] < 0:
                raise ValueError("Invalid shape scale or margin")
            if geo == GeoType.PLANE and np.any(scale[:2] != 0):
                raise ValueError("Only infinite planes are supported")
            if (geo == GeoType.SPHERE and scale[0] <= 0) or (geo == GeoType.BOX and np.any(scale <= 0)):
                raise ValueError("Analytic shape size must be positive")
            if geo == GeoType.MESH:
                index = int(v["_shape_sdf_index"][i])
                if model.device.is_cpu or descriptors is None or index < 0 or index >= len(descriptors):
                    raise ValueError("Mesh contact requires an existing CUDA texture SDF")
                if not descriptors["scale_baked"][index] and (scale[0] <= 0 or not np.all(scale == scale[0])):
                    raise ValueError("Unbaked mesh scale must be positive and uniform")
        transforms = v["shape_transform"].astype(self.numpy_dtype)
        norms = np.linalg.norm(transforms[:, 3:], axis=1)
        if not np.isfinite(transforms).all() or np.any(abs(norms - 1) > 1e-5):
            raise ValueError("Invalid static shape transform")
        transforms[:, 3:] /= norms[:, None]
        triangles = v["tri_indices"]
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or np.any(triangles < 0)
            or np.any(triangles >= model.particle_count)
        ):
            raise ValueError("Invalid surface topology")
        points = v["particle_q"].astype(self.numpy_dtype)[triangles]
        areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
        if not np.isfinite(areas).all() or np.any(areas <= 0):
            raise ValueError("Surface faces must have positive finite rest area")
        if np.any(v["shape_body"][shapes] < -1) or np.any(v["shape_body"][shapes] >= model.body_count):
            raise ValueError("Invalid shape body")
        count = len(triangles) * 3 * len(shapes)
        if 3 * count >= 2**31:
            raise ValueError("Contact query capacity exceeds int32")
        self.layout = layout or MonolithicLinearLayout(q_dof_count, len(dynamic))
        if self.layout.q_dof_count != q_dof_count or (
            _materials is None and self.layout.dynamic_particle_count != len(dynamic)
        ):
            raise ValueError("Contact layout does not match endpoint DoFs")
        ancestors = {}
        if articulation_actor is not None:
            parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
            starts = model.joint_qd_start.numpy()
            incoming = {int(children[j]): j for j in articulation_actor.joint_indices}
            local_dofs = {index: i for i, index in enumerate(articulation_actor.dof_indices)}
            for body in articulation_actor.body_indices:
                selected, cursor = [], body
                while cursor >= 0:
                    joint = incoming[cursor]
                    selected.extend(local_dofs[i] for i in range(int(starts[joint]), int(starts[joint + 1])))
                    cursor = int(parents[joint])
                ancestors[body] = sorted(selected)
        self._legacy_ancestors = {body: tuple(selected) for body, selected in ancestors.items()}
        if type(base_triplet_count) is not int or not 0 <= base_triplet_count < 2**31:
            raise ValueError("Invalid base triplet capacity")
        support_count, triplet_count = 0, 0
        for face in triangles:
            soft_count = 3 * int(np.count_nonzero(mapping[face] >= 0))
            for shape in shapes:
                body = int(v["shape_body"][shape])
                q_count = (
                    len(ancestors[body])
                    if articulation_actor is not None and body >= 0
                    else (q_dof_count if body >= 0 else 0)
                )
                k = q_count + soft_count
                # Three quadrature samples, each with three stable factors.
                support_count += 9 * k
                triplet_count += 9 * k * k
                if support_count >= 2**31 or base_triplet_count + triplet_count >= 2**31:
                    raise ValueError("Contact support/triplet capacity exceeds int32")
        self.legacy_count = count
        samples = self.actor_samples
        if _materials is not None:
            for name in ("weights", "offsets", "cable_coordinates"):
                original = getattr(samples, name)
                converted = original.astype(self.numpy_dtype)
                if (
                    not np.isfinite(converted).all()
                    or np.any((original != 0) & (converted == 0))
                    or (name == "weights" and np.any(converted <= 0))
                ):
                    raise ValueError("Unrepresentable material contact sampling")
        if samples is not None:
            _checked_contact_capacity(
                count + samples.count,
                support_count + samples.support_size,
                triplet_count + samples.triplet_capacity,
                base_triplet_count,
            )
            count += samples.count
            triplet_count += samples.triplet_capacity
        self.triplet_capacity = triplet_count
        self.model, self.contact_model = model, "C-IE"
        # Solver actor tables own articulation bodies; a standalone owner uses every body.
        self.body_count = (
            len(articulation_actor.body_indices)
            if articulation_actor is not None
            else model.body_count
            if actors is None and _materials is None
            else 0
        )
        self.parameters, self.nq, self.count = parameters, q_dof_count, count
        self._descriptors = descriptors.copy() if descriptors is not None else None
        # C-IE lags only the friction load at the frozen start. Without contact
        # samples, or with zero friction on guarded normals and no material face
        # selection, the remaining normal penalty is a true potential.
        self.conservative_energy = count == 0 or (
            parameters[1] == 0
            and alignment_policy == "guard"
            and (_materials is None or samples.material_start == samples.count)
        )
        self.table = table if table is not None else wp.empty(0, dtype=TextureSDFData, device=model.device)
        self.smooth_sdf = smooth_sdf
        if smooth_sdf is not None:
            if (
                smooth_sdf.dtype != dtype
                or table is None
                or smooth_sdf.table.size != table.size
                or smooth_sdf.table.device != model.device
            ):
                raise ValueError("Smooth SDF must match the complete original table and device")
            smooth_sdf.validate_identity()
        self.smooth_table = (
            smooth_sdf.table
            if smooth_sdf is not None
            else wp.empty(0, dtype=_sdf_type(self.dtype), device=model.device)
        )
        self.shapes = wp.array(shapes, dtype=int, device=model.device)
        self.mapping = wp.array(mapping.astype(np.int32), device=model.device)
        self.triangles = wp.array(triangles, dtype=int, device=model.device)
        self.types = wp.clone(model.shape_type)
        self.bodies = wp.clone(model.shape_body)
        body_local = np.arange(model.body_count, dtype=np.int32)
        if articulation_actor is not None:
            body_local.fill(-1)
            body_local[list(articulation_actor.body_indices)] = np.arange(len(articulation_actor.body_indices))
        self.body_local = wp.array(body_local, dtype=int, device=model.device)
        target_bodies = v["shape_body"].copy()
        dynamic_shapes = target_bodies >= 0
        target_bodies[dynamic_shapes] = body_local[target_bodies[dynamic_shapes]]
        self.target_bodies = wp.array(target_bodies, dtype=int, device=model.device)
        self.sdf_indices = wp.clone(model._shape_sdf_index)
        self.transforms = wp.array(transforms, dtype=self.transform, device=model.device)
        self.scales = wp.array(v["shape_scale"].astype(self.numpy_dtype), dtype=self.vector, device=model.device)
        self.margins = wp.array(v["shape_margin"].astype(self.numpy_dtype), dtype=self.dtype, device=model.device)
        self.area = wp.array(areas, dtype=self.dtype, device=model.device)
        self._started, self._evaluated = False, False
        if samples is not None and samples.count:
            from .contact_actor import _validate_actor_targets  # noqa: PLC0415

            _validate_actor_targets(self, samples)

    @property
    def quaternion_norm_tolerance(self):
        """Unit-quaternion guard selected by native arithmetic precision."""
        return 1e-6 if self.dtype == wp.float32 else 1e-10
