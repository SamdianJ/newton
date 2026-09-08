# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bilateral P1Q3 quadratic-hinge contact and final force publication."""

import enum
import hashlib
import json
import math
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from ...sim import Contacts, Model, State
from .articulation import MonolithicArticulationWorkspace, articulation_point_jacobian_column
from .collision import MonolithicCollisionPipeline
from .linear import (
    MonolithicContactFactorKind,
    MonolithicLinearAssembly,
    MonolithicLinearGeneration,
    MonolithicLinearWorkspace,
    _append_monolithic_contact_factor_triplets,
    _MonolithicContactFactors,
    _MonolithicScalarTriplets,
    _reserve_monolithic_contact_factor,
)


class _MonolithicReturnedStateRole(enum.IntEnum):
    STATE_OUT_CONVERGED = 0
    STATE_OUT_SOFT_STOP = 1
    STATE_IN_ROLLBACK = 2


@dataclass(frozen=True, slots=True)
class _MonolithicHistoryIdentity:
    """PR-6B identity; hashes bind immutable topology and configuration.

    Pending/committed storage belongs to one continuous trajectory.
    """

    topology_sha256: str
    config_sha256: str
    config_generation: int
    history_epoch: int

    def __post_init__(self):
        for value in (self.topology_sha256, self.config_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("History requires lowercase topology/config SHA-256 identities")
        for value in (self.config_generation, self.history_epoch):
            if type(value) is not int or value < 0:
                raise ValueError("History generations must be nonnegative integers")


@wp.struct
class _MonolithicHistoryBuffer:
    """Fixed candidate-tid storage; vectors use the rigid body local frame."""

    valid: wp.array[int]
    xi_local: wp.array[wp.vec3]
    normal_local: wp.array[wp.vec3]


@wp.struct
class _ContactStepContext:
    smoothing: float
    mu: float
    kt: float
    dt: float
    integrate: int
    prepare: int
    transport_cos: float
    start_x: wp.array[wp.vec3]
    committed: _MonolithicHistoryBuffer
    pending: _MonolithicHistoryBuffer
    samples: wp.array2d[float]


@dataclass(frozen=True, slots=True, eq=False)
class _MonolithicFinalForceGeneration:
    step_generation: int
    contacts: Contacts
    contact_generation: int
    returned_state: State
    returned_state_role: _MonolithicReturnedStateRole
    config_sha256: str = ""
    history_epoch: int = 0


@wp.func
def _normal_response(u: float, eps: float, stiffness: float):
    """Return sample energy, normal magnitude and positive GN curvature."""
    p, dp, ddp = wp.max(u, 0.0), float(0.0), float(0.0)
    if u > 0.0:
        dp = 1.0
    if eps > 0.0 and u > -eps and u < eps:
        t = (u + eps) / (2.0 * eps)
        p = eps * (2.0 * t * t * t - t * t * t * t)
        dp = 3.0 * t * t - 2.0 * t * t * t
        ddp = 3.0 * t * (1.0 - t) / eps
    return wp.vec3(0.5 * stiffness * p * p, stiffness * p * dp, stiffness * (dp * dp + p * ddp))


@wp.func
def _tangent_response(xi: wp.vec3, n: wp.vec3, stiffness: float, cap: float):
    """Return soft force, local Huber potential and PSD Hessian (sliding at equality)."""
    force, energy, hessian = wp.vec3(), float(0.0), wp.mat33()
    radius = wp.length(xi)
    projector = wp.identity(n=3, dtype=float) - wp.outer(n, n)
    if cap > 0.0 and stiffness > 0.0:
        if stiffness * radius < cap:
            force = -stiffness * xi
            energy = 0.5 * stiffness * radius * radius
            hessian = stiffness * projector
        elif radius > 0.0:
            direction = xi / radius
            force = -cap * direction
            energy = cap * radius - cap * cap / (2.0 * stiffness)
            hessian = (cap / radius) * (projector - wp.outer(direction, direction))
    return force, energy, hessian


@wp.kernel(enable_backward=False)
def evaluate_monolithic_p1q3_contacts(
    mode: int,
    soft_contact_count: wp.array[int],
    soft_contact_max: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q_rest: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_to_dynamic: wp.array[int],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_qd: wp.array[wp.spatial_vector],
    context: _ContactStepContext,
    body_to_articulation: wp.array[int],
    body_to_link_index: wp.array[int],
    articulation_dof_count: wp.array[int],
    spatial_jacobian: wp.array3d[float],
    q_dof_count: int,
    r_soft: float,
    contact_stiffness: float,
    global_residual: wp.array[float],
    triplets: _MonolithicScalarTriplets,
    factor_count: wp.array[int],
    factor_contact_record: wp.array[int],
    record_candidate_tid: wp.array[int],
    factor_kind: wp.array[int],
    factor_candidate_tid: wp.array[int],
    factor_gap: wp.array[float],
    contact_W: wp.array[float],
    contact_Gq: wp.array2d[float],
    contact_Gx_columns: wp.array2d[int],
    contact_Gx_values: wp.array2d[float],
    final_force_linear: wp.array[wp.vec3],
    final_force_moment: wp.array[wp.vec3],
    out_status: wp.array[int],
):
    record = wp.tid()
    count = soft_contact_count[0]
    if count < 0 or count > soft_contact_max:
        wp.atomic_max(out_status, 0, 6)
    if record >= wp.min(count, soft_contact_max) or out_status[0] != 0:
        return
    if mode == 0 and triplets.active_token[0] != triplets.token:
        wp.atomic_max(out_status, 0, 2)
        return
    ids = soft_contact_indices[record]
    bary = soft_contact_barycentric[record]
    shape = soft_contact_shape[record]
    if shape < 0 or shape >= shape_body.shape[0]:
        wp.atomic_max(out_status, 0, 1)
        return
    for a in range(3):
        if ids[a] < 0 or ids[a] >= particle_q.shape[0]:
            wp.atomic_max(out_status, 0, 1)
            return
    body = shape_body[shape]
    if body < 0 or body >= body_q.shape[0]:
        wp.atomic_max(out_status, 0, 1)
        return
    n = soft_contact_normal[record]
    xr = wp.transform_point(body_q[body], soft_contact_body_pos[record])
    xs = bary[0] * particle_q[ids[0]] + bary[1] * particle_q[ids[1]] + bary[2] * particle_q[ids[2]]
    area = 0.5 * wp.length(
        wp.cross(particle_q_rest[ids[1]] - particle_q_rest[ids[0]], particle_q_rest[ids[2]] - particle_q_rest[ids[0]])
    )
    gap = wp.dot(n, xs - xr) - r_soft - shape_margin[shape]
    weight = contact_stiffness * (area / 3.0)
    if not wp.isfinite(gap) or not wp.isfinite(weight) or not wp.isfinite(wp.length(n)):
        wp.atomic_max(out_status, 0, 7)
        return
    if weight <= 0.0 or wp.abs(wp.length(n) - 1.0) > 1.0e-5:
        wp.atomic_max(out_status, 0, 1)
        return
    response = _normal_response(-gap, context.smoothing, weight)
    normal_magnitude, normal_curvature = response[1], response[2]
    if normal_magnitude <= 0.0:
        return
    com = wp.transform_point(body_q[body], body_com[body])
    normal_force = -normal_magnitude * n
    tangent_soft = wp.vec3()
    xi = wp.vec3()
    cap = context.mu * normal_magnitude
    kt = context.kt * area / 3.0
    key = record_candidate_tid[record]
    context.samples[key, 0] = normal_magnitude
    if context.mu > 0.0:
        rotation = wp.transform_get_rotation(body_q[body])
        old = wp.vec3()
        original_energy = float(0.0)
        matched, reset = float(0.0), float(1.0)
        if context.committed.valid[key] != 0:
            old_world = wp.quat_rotate(rotation, context.committed.xi_local[key])
            original_energy = 0.5 * kt * wp.dot(old_world, old_world)
            old_normal = wp.quat_rotate(rotation, context.committed.normal_local[key])
            if wp.dot(old_normal, n) >= context.transport_cos:
                old = old_world - wp.dot(n, old_world) * n
                matched, reset = 1.0, 0.0
            else:
                reset = 2.0
        velocity = wp.vec3()
        if context.integrate != 0:
            vs = (
                bary[0] * (particle_q[ids[0]] - context.start_x[ids[0]])
                + bary[1] * (particle_q[ids[1]] - context.start_x[ids[1]])
                + bary[2] * (particle_q[ids[2]] - context.start_x[ids[2]])
            ) / context.dt
            twist = body_qd[body]
            vr = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), xs - com)
            relative = vs - vr
            velocity = relative - wp.dot(n, relative) * n
        xi = old + context.dt * velocity
        tangent_soft, _potential, _local_hessian = _tangent_response(xi, n, kt, cap)
        pending = -tangent_soft / kt
        elastic = 0.5 * kt * wp.dot(pending, pending)
        plastic = cap * wp.length(xi - pending)
        context.samples[key, 0] = normal_magnitude
        context.samples[key, 1] = wp.length(tangent_soft)
        context.samples[key, 2] = wp.length(velocity)
        context.samples[key, 3] = 1.0
        if kt * wp.length(xi) >= cap:
            context.samples[key, 3] = 2.0
        context.samples[key, 4] = matched
        context.samples[key, 5] = reset
        context.samples[key, 6] = elastic
        context.samples[key, 7] = wp.dot(tangent_soft, context.dt * velocity)
        context.samples[key, 8] = plastic
        context.samples[key, 9] = 0.5 * kt * wp.dot(old, old) - original_energy
        context.samples[key, 10] = elastic - original_energy
        if context.prepare != 0:
            context.pending.valid[key] = 1
            context.pending.xi_local[key] = wp.quat_rotate_inv(rotation, pending)
            context.pending.normal_local[key] = wp.quat_rotate_inv(rotation, n)
    force = normal_force - tangent_soft
    moment = wp.cross(xr - com, normal_force) - wp.cross(xs - com, tangent_soft)
    if not wp.isfinite(wp.length(force)) or not wp.isfinite(wp.length(moment)):
        wp.atomic_max(out_status, 0, 7)
        return
    # Current/trial diagnostics use the evaluated forces, not another contact law.
    final_force_linear[record] = force
    final_force_moment[record] = moment
    articulation = body_to_articulation[body]
    for d in range(articulation_dof_count[articulation]):
        jn = articulation_point_jacobian_column(
            spatial_jacobian, articulation, body_to_link_index[body], d, body_q[body], body_com[body], xr
        )
        jt = articulation_point_jacobian_column(
            spatial_jacobian, articulation, body_to_link_index[body], d, body_q[body], body_com[body], xs
        )
        contribution = -wp.dot(jn, normal_force) + wp.dot(jt, tangent_soft)
        if not wp.isfinite(contribution):
            wp.atomic_max(out_status, 0, 7)
            return
        wp.atomic_add(global_residual, d, contribution)
    for a in range(3):
        dynamic = particle_to_dynamic[ids[a]]
        if dynamic >= 0:
            for axis in range(3):
                wp.atomic_add(global_residual, q_dof_count + 3 * dynamic + axis, bary[a] * force[axis])
    if mode != 0:
        return
    factors = _MonolithicContactFactors()
    factors.gq = contact_Gq
    factors.gx_columns = contact_Gx_columns
    factors.gx_values = contact_Gx_values
    factors.weights = contact_W
    factors.count = factor_count
    factors.status = out_status
    factors.capacity = contact_W.shape[0]
    factors.q_dof_count = q_dof_count
    factors.dynamic_particle_count = (global_residual.shape[0] - q_dof_count) / 3
    factors.active_token = triplets.active_token
    factors.token = triplets.token
    # A tangent frame avoids a device eigensolver; sliding has only one positive direction.
    frame_axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.8:
        frame_axis = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(n, frame_axis))
    t2 = wp.cross(n, t1)
    w1, w2 = float(0.0), float(0.0)
    if cap > 0.0:
        radius = wp.length(xi)
        if kt * radius < cap:
            w1, w2 = kt, kt
        elif radius > 0.0:
            t1 = wp.normalize(wp.cross(n, xi / radius))
            w1 = cap / radius
    for component in range(3):
        direction, curvature, point = n, normal_curvature, xr
        kind = wp.static(int(MonolithicContactFactorKind.NORMAL))
        if component == 1:
            direction, curvature, point = t1, w1, xs
            kind = wp.static(int(MonolithicContactFactorKind.TANGENT))
        elif component == 2:
            direction, curvature, point = t2, w2, xs
            kind = wp.static(int(MonolithicContactFactorKind.TANGENT))
        if curvature > 0.0:
            factor = _reserve_monolithic_contact_factor(factors)
            if factor < 0:
                return
            factor_contact_record[factor] = record
            factor_kind[factor] = kind
            factor_candidate_tid[factor] = key
            factor_gap[factor] = gap
            contact_W[factor] = curvature
            for d in range(articulation_dof_count[articulation]):
                column = articulation_point_jacobian_column(
                    spatial_jacobian, articulation, body_to_link_index[body], d, body_q[body], body_com[body], point
                )
                contact_Gq[factor, d] = -wp.dot(direction, column)
            for a in range(3):
                dynamic = particle_to_dynamic[ids[a]]
                for axis_index in range(3):
                    local_column, value = int(-1), float(0.0)
                    if dynamic >= 0:
                        local_column = 3 * dynamic + axis_index
                        value = bary[a] * direction[axis_index]
                    contact_Gx_columns[factor, 3 * a + axis_index] = local_column
                    contact_Gx_values[factor, 3 * a + axis_index] = value
            _append_monolithic_contact_factor_triplets(factors, factor, triplets, q_dof_count)


@wp.kernel(enable_backward=False)
def publish_monolithic_contact_forces(
    soft_contact_count: wp.array[int],
    rigid_contact_max: int,
    final_force_linear: wp.array[wp.vec3],
    final_force_moment: wp.array[wp.vec3],
    out_contact_force: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    out_contact_force[i] = wp.spatial_vector()
    record = i - rigid_contact_max
    if record >= 0 and record < wp.min(soft_contact_count[0], final_force_linear.shape[0]):
        out_contact_force[i] = wp.spatial_vector(final_force_linear[record], final_force_moment[record])


@wp.kernel
def _validate_record_slots(
    pairs: wp.array[wp.vec2i],
    faces: wp.array2d[int],
    tids: wp.array[int],
    count: wp.array[int],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shapes: wp.array[int],
    seen: wp.array[int],
    record_candidate_tid: wp.array[int],
    status: wp.array[int],
):
    tid = wp.tid()
    record = tids[tid]
    if record == -1:
        return
    if record < 0 or record >= wp.min(count[0], indices.shape[0]):
        wp.atomic_max(status, 0, 1)
        return
    pair = pairs[tid / 3]
    ids = wp.vec3i(faces[pair[0], 0], faces[pair[0], 1], faces[pair[0], 2])
    bary = wp.vec3(1.0 / 6.0)
    bary[tid % 3] = 2.0 / 3.0
    actual_ids, actual_bary = indices[record], barycentric[record]
    for a in range(3):
        if actual_ids[a] != ids[a] or actual_bary[a] != bary[a]:
            wp.atomic_max(status, 0, 1)
    if shapes[record] != pair[1]:
        wp.atomic_max(status, 0, 1)
    wp.atomic_add(seen, record, 1)
    record_candidate_tid[record] = tid


@wp.kernel
def _validate_record_coverage(count: wp.array[int], seen: wp.array[int], status: wp.array[int]):
    record = wp.tid()
    if count[0] < 0 or count[0] > seen.shape[0]:
        wp.atomic_max(status, 0, 6)
    if record < wp.min(count[0], seen.shape[0]) and seen[record] != 1:
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _validate_cached_force(force: wp.array[wp.vec3], moment: wp.array[wp.vec3], status: wp.array[int]):
    i = wp.tid()
    if not wp.isfinite(wp.length(force[i])) or not wp.isfinite(wp.length(moment[i])):
        wp.atomic_max(status, 0, 7)


@wp.kernel
def _add_contact_residual(source: wp.array[float], destination: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    if not wp.isfinite(source[i]) or not wp.isfinite(destination[i] + source[i]):
        wp.atomic_max(status, 0, 7)
    destination[i] += source[i]


@wp.kernel
def _contact_diagnostics(
    count: wp.array[int],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shapes: wp.array[int],
    body_pos: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    shape_body: wp.array[int],
    margin: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_link: wp.array[int],
    jacobian: wp.array3d[float],
    mapping: wp.array[int],
    q_count: int,
    radius: float,
    smoothing: float,
    final_force: wp.array[wp.vec3],
    final_moment: wp.array[wp.vec3],
    diagnostics: wp.array[float],
    projection: wp.array[float],
    status: wp.array[int],
):
    record = wp.tid()
    if record >= wp.min(count[0], indices.shape[0]) or status[0] != 0:
        return
    ids, bary, n = indices[record], barycentric[record], normals[record]
    shape = shapes[record]
    body = shape_body[shape]
    xr = wp.transform_point(body_q[body], body_pos[record])
    xs = bary[0] * x[ids[0]] + bary[1] * x[ids[1]] + bary[2] * x[ids[2]]
    gap = wp.dot(n, xs - xr) - radius - margin[shape]
    wp.atomic_max(diagnostics, 1, wp.max(0.0, -gap))
    if gap >= smoothing:
        return
    wp.atomic_add(diagnostics, 0, 1.0)
    force = final_force[record]
    com = wp.transform_point(body_q[body], body_com[body])
    rigid = force
    moment = final_moment[record]
    balance = rigid
    moment_balance = wp.cross(com, rigid) + moment
    for a in range(3):
        soft = -bary[a] * force
        balance += soft
        moment_balance += wp.cross(x[ids[a]], soft)
        dynamic = mapping[ids[a]]
        if dynamic >= 0:
            for axis in range(3):
                wp.atomic_add(projection, q_count + 3 * dynamic + axis, soft[axis])
    wp.atomic_add(diagnostics, 2, wp.length(balance))
    wp.atomic_add(diagnostics, 3, wp.length(moment_balance))
    wp.atomic_add(diagnostics, 4, wp.length(force))
    wp.atomic_add(diagnostics, 5, wp.length(wp.cross(xr, force)) + wp.length(wp.cross(xs, force)))
    for d in range(q_count):
        row = 6 * body_to_link[body]
        generalized = float(0.0)
        for axis in range(3):
            generalized += jacobian[0, row + axis, d] * rigid[axis] + jacobian[0, row + 3 + axis, d] * moment[axis]
        wp.atomic_add(projection, d, generalized)


@wp.kernel
def _projection_error(residual: wp.array[float], projection: wp.array[float], sums: wp.array[float]):
    i = wp.tid()
    r, q = residual[i], projection[i]
    wp.atomic_add(sums, 0, (r + q) * (r + q))
    wp.atomic_add(sums, 1, r * r)
    wp.atomic_add(sums, 2, q * q)


@wp.kernel
def _lost_history_samples(
    committed: _MonolithicHistoryBuffer,
    pending: _MonolithicHistoryBuffer,
    pairs: wp.array[wp.vec2i],
    faces: wp.array2d[int],
    rest: wp.array[wp.vec3],
    kt: float,
    samples: wp.array2d[float],
):
    key = wp.tid()
    if committed.valid[key] != 0 and pending.valid[key] == 0:
        face = pairs[key // 3][0]
        a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
        area = 0.5 * wp.length(wp.cross(rest[b] - rest[a], rest[c] - rest[a]))
        old = committed.xi_local[key]
        samples[key, 5] = 3.0
        samples[key, 10] = -0.5 * kt * (area / 3.0) * wp.dot(old, old)


class MonolithicContactWorkspace:
    """Own mode-isolated scratch and the last returned-state physical forces."""

    def __init__(
        self,
        model: Model,
        pipeline: MonolithicCollisionPipeline,
        linear_workspace: MonolithicLinearWorkspace,
        *,
        contact_stiffness: float,
        normal_smoothing_width: float = 0.0,
        friction_coefficient: float = 0.0,
        tangential_stiffness: float | None = None,
        physics_identity: str = "",
    ) -> None:
        for value in (normal_smoothing_width, friction_coefficient, tangential_stiffness):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or value < 0
                or value > np.finfo(np.float32).max
                or (value > 0 and np.float32(value) <= 0)
            ):
                raise ValueError("Invalid contact configuration")
        if normal_smoothing_width > pipeline._soft_contact_gap:
            raise ValueError("Detection gap must cover smoothing")
        if friction_coefficient > 0 and (tangential_stiffness is None or tangential_stiffness <= 0):
            raise ValueError("Friction requires positive tangential stiffness")
        if pipeline.model is not model or linear_workspace.device != model.device:
            raise ValueError("Contact workspace model/device mismatch")
        if linear_workspace.capacities.contact_factor_count != pipeline.soft_contact_max * (
            3 if friction_coefficient > 0 else 1
        ):
            raise ValueError("Contact factor capacity must match the pipeline")
        if (
            linear_workspace.layout.q_dof_count != model.joint_dof_count
            or linear_workspace.particle_to_dynamic.shape != (model.particle_count,)
        ):
            raise ValueError("Contact workspace layout mismatch")
        if (
            not math.isfinite(contact_stiffness)
            or contact_stiffness <= 0
            or contact_stiffness > np.finfo(np.float32).max
        ):
            raise ValueError("Contact stiffness must be finite and positive")
        self.model, self.pipeline, self.linear_workspace = model, pipeline, linear_workspace
        self.contact_stiffness = float(contact_stiffness)
        self.normal_smoothing_width = float(normal_smoothing_width)
        self.friction_coefficient = float(friction_coefficient)
        self.tangential_stiffness = float(tangential_stiffness or 0)
        self.transport_cos = math.cos(math.pi / 4)
        self.history_epoch = 0
        self._step_epoch = 0
        self._step_open = False
        self._prepared = False
        self._dt = 0.0
        self._start_x = model.particle_q
        self._rollback = False
        identity_arrays = tuple(
            getattr(model, name)
            for name in (
                "tri_indices",
                "tet_indices",
                "particle_q",
                "shape_body",
                "shape_type",
                "shape_scale",
                "shape_transform",
                "joint_parent",
                "joint_child",
                "joint_axis",
                "joint_q_start",
                "body_com",
            )
        )
        identity_arrays = (*identity_arrays, pipeline._face_pairs)
        self.topology_sha256 = hashlib.sha256(b"".join(a.numpy().tobytes() for a in identity_arrays)).hexdigest()
        self._configuration = (
            self.contact_stiffness,
            self.normal_smoothing_width,
            self.friction_coefficient,
            self.tangential_stiffness,
            self.transport_cos,
        )
        self.config_sha256 = hashlib.sha256((json.dumps(self._configuration) + physics_identity).encode()).hexdigest()
        self._history_arrays = {
            name: getattr(model, name)
            for name in ("tri_indices", "tet_indices", "particle_q", "shape_body", "body_com")
        }

        self._mapping = linear_workspace.particle_to_dynamic
        self._final_generation = None
        self._final_contact_count = 0
        capacity, device = pipeline.soft_contact_max, model.device
        self.factor_contact_record = wp.full(
            linear_workspace.capacities.contact_factor_count, -1, dtype=int, device=device
        )
        self._record_candidate_tid = wp.full(capacity, -1, dtype=int, device=device)
        self.factor_gap = wp.zeros(linear_workspace.capacities.contact_factor_count, dtype=float, device=device)
        self.final_force_linear = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.final_force_moment = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._forces = [wp.zeros_like(self.final_force_linear) for _ in range(2)] + [self.final_force_linear]
        self._moments = [wp.zeros_like(self.final_force_moment) for _ in range(2)] + [self.final_force_moment]
        self._samples = [wp.zeros((capacity, 11), dtype=float, device=device) for _ in range(3)]
        history_capacity = capacity if friction_coefficient > 0 else 0
        self.committed, self.pending = _MonolithicHistoryBuffer(), _MonolithicHistoryBuffer()
        for buffer in (self.committed, self.pending):
            buffer.valid = wp.zeros(history_capacity, dtype=int, device=device)
            buffer.xi_local = wp.zeros(history_capacity, dtype=wp.vec3, device=device)
            buffer.normal_local = wp.zeros(history_capacity, dtype=wp.vec3, device=device)
        self._seen = wp.zeros(capacity, dtype=int, device=device)
        self._publication_status = wp.zeros(1, dtype=int, device=device)
        self._projection_sums = wp.zeros(3, dtype=float, device=device)
        self._dof_count = wp.array([model.joint_dof_count], dtype=int, device=device)
        self._residual = [
            wp.zeros(linear_workspace.layout.scalar_dof_count, dtype=float, device=device) for _ in range(3)
        ]
        self._projection = [wp.zeros_like(r) for r in self._residual]
        self._diagnostic_values = [wp.zeros(6, dtype=float, device=device) for _ in range(3)]
        # Trial/final launches never receive descriptors for a mutable current assembly.
        self._dummy_triplets = _MonolithicScalarTriplets()
        self._dummy_factor_count = wp.zeros(1, dtype=int, device=device)
        self._dummy_gq = wp.empty((0, model.joint_dof_count), dtype=float, device=device)
        self._dummy_gx = wp.empty((0, 9), dtype=float, device=device)
        self._dummy_columns = wp.empty((0, 9), dtype=int, device=device)
        self._dummy_scalar = wp.empty(0, dtype=float, device=device)
        self._dummy_int = wp.empty(0, dtype=int, device=device)

    @property
    def history_identity(self) -> _MonolithicHistoryIdentity:
        """Return the immutable configuration and current committed epoch."""
        return _MonolithicHistoryIdentity(self.topology_sha256, self.config_sha256, 0, self.history_epoch)

    def begin_step(self, original: State, dt: float) -> None:
        """Freeze the physical-step origin; nonlinear trials never change history."""
        if self._step_open or not math.isfinite(dt) or dt <= 0:
            raise ValueError("History step is already open or dt is invalid")
        self._start_x, self._dt = original.particle_q, float(dt)
        self._step_epoch = self.history_epoch
        self._step_open, self._prepared = True, False

    def commit_history(self) -> None:
        """Swap validated storage only after State and both force publications succeed."""
        if self.friction_coefficient > 0:
            if not self._step_open or not self._prepared or self.history_epoch != self._step_epoch:
                raise ValueError("History commit is stale or unprepared")
            self.committed, self.pending = self.pending, self.committed
            self.history_epoch += 1
        self._step_open = False

    def discard_history(self) -> None:
        """Undo a completed swap if a later transaction operation raised."""
        if self.history_epoch != self._step_epoch:
            self.committed, self.pending = self.pending, self.committed
            self.history_epoch = self._step_epoch
        self._prepared = False
        self._step_open = False

    def history_diagnostics(self) -> dict:
        """Read final-only per-key metrics for reports and examples."""
        samples = self._samples[2].numpy()
        return {
            "history_epoch": self.history_epoch,
            "normal_force_sum": float(samples[:, 0].sum()),
            "tangent_force_sum": float(samples[:, 1].sum()),
            "relative_slip_max": float(samples[:, 2].max(initial=0)),
            "stick_count": int(np.count_nonzero(samples[:, 3] == 1)),
            "slide_count": int(np.count_nonzero(samples[:, 3] == 2)),
            "history_match_count": int(samples[:, 4].sum()),
            "history_reset_count": int(np.count_nonzero(samples[:, 5])),
            "history_new_count": int(np.count_nonzero(samples[:, 5] == 1)),
            "history_normal_reset_count": int(np.count_nonzero(samples[:, 5] == 2)),
            "history_lost_count": int(np.count_nonzero(samples[:, 5] == 3)),
            "history_elastic_energy": float(samples[:, 6].sum()),
            "friction_work": float(samples[:, 7].sum()),
            "return_dissipation": float(samples[:, 8].sum()),
            "transport_energy_change": float(samples[:, 9].sum()),
            "elastic_energy_change": float(samples[:, 10].sum()),
        }

    @property
    def final_force_generation(self) -> _MonolithicFinalForceGeneration | None:
        return self._final_generation

    def bind_returned_state(self, state: State) -> None:
        """Associate FK scratch evaluation with the actual returned State."""
        if self._final_generation is None:
            raise ValueError("No validated final force generation")
        self._final_generation = replace(self._final_generation, returned_state=state)

    def invalidate_final_force(self) -> None:
        self._final_generation = None

    def validate_contract(self, model: Model, pipeline: MonolithicCollisionPipeline, contacts: Contacts) -> None:
        if (
            model is not self.model
            or pipeline is not self.pipeline
            or self.linear_workspace.particle_to_dynamic is not self._mapping
        ):
            raise ValueError("Contact workspace is stale")
        pipeline.validate_contacts(contacts)
        if any(getattr(model, name) is not array for name, array in self._history_arrays.items()):
            raise ValueError("History topology is stale")
        if (self.friction_coefficient > 0 or self.normal_smoothing_width > 0) and self._configuration != (
            self.contact_stiffness,
            self.normal_smoothing_width,
            self.friction_coefficient,
            self.tangential_stiffness,
            self.transport_cos,
        ):
            raise ValueError("History configuration is stale; rebuild the solver")

    def _diagnostics(self, mode: int) -> dict:
        values = self._diagnostic_values[mode].numpy()
        self._projection_sums.zero_()
        wp.launch(
            _projection_error,
            self._residual[mode].shape[0],
            [self._residual[mode], self._projection[mode], self._projection_sums],
            device=self.model.device,
        )
        sums = self._projection_sums.numpy()
        error = float(np.sqrt(sums[0]) / max(np.sqrt(sums[1]) + np.sqrt(sums[2]), 1e-30))
        return {
            "active_sample_count": int(values[0]),
            "max_penetration": float(values[1]),
            "force_imbalance": float(values[2] / max(float(values[4]), 1e-30)),
            "moment_imbalance": float(values[3] / max(float(values[5]), 1e-30)),
            "generalized_projection_error": error,
            "contact_sign_error": error,
        }

    def _publish_final_forces(
        self,
        state: State,
        contacts: Contacts,
        *,
        step_generation: int,
        returned_state_role: _MonolithicReturnedStateRole,
    ) -> None:
        self.validate_contract(self.model, self.pipeline, contacts)
        generation = self._final_generation
        if (
            generation is None
            or generation.config_sha256 != self.config_sha256
            or generation.history_epoch
            != self.history_epoch + int(self._step_open and self._prepared and self.friction_coefficient > 0)
            or generation.returned_state is not state
            or generation.contacts is not contacts
            or generation.step_generation != step_generation
            or generation.returned_state_role != returned_state_role
            or generation.contact_generation != int(contacts.contact_generation.numpy()[0])
            or self._final_contact_count != int(contacts.soft_contact_count.numpy()[0])
        ):
            raise ValueError("Final contact force cache is stale")
        if contacts.force is None:
            raise ValueError("Request contact attribute 'force' before constructing the pipeline")
        self._publication_status.zero_()
        self._seen.zero_()
        if self.pipeline.soft_contact_max:
            wp.launch(
                _validate_record_slots,
                self.pipeline.soft_contact_max,
                [
                    self.pipeline._face_pairs,
                    self.model.tri_indices,
                    contacts.soft_contact_tids,
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    self._seen,
                    self._record_candidate_tid,
                    self._publication_status,
                ],
                device=self.model.device,
            )
            wp.launch(
                _validate_record_coverage,
                self.pipeline.soft_contact_max,
                [contacts.soft_contact_count, self._seen, self._publication_status],
                device=self.model.device,
            )
            wp.launch(
                _validate_cached_force,
                self.pipeline.soft_contact_max,
                [self.final_force_linear, self.final_force_moment, self._publication_status],
                device=self.model.device,
            )
        if int(self._publication_status.numpy()[0]):
            self.invalidate_final_force()
            raise ValueError("Final contact force cache has invalid records or forces")
        wp.launch(
            publish_monolithic_contact_forces,
            contacts.force.shape[0],
            [
                contacts.soft_contact_count,
                contacts.rigid_contact_max,
                self.final_force_linear,
                self.final_force_moment,
                contacts.force,
            ],
            device=self.model.device,
        )


def _evaluate(mode, model, state, contacts, pipeline, articulation, workspace, out_residual, out_status, assembly=None):
    workspace.validate_contract(model, pipeline, contacts)
    articulation._validate_state(model, state)
    if (
        out_residual.shape != (workspace.linear_workspace.layout.scalar_dof_count,)
        or out_residual.dtype != wp.float32
        or out_residual.device != model.device
    ):
        raise ValueError("Invalid contact residual buffer")
    if out_status.shape != (1,) or out_status.dtype != wp.int32 or out_status.device != model.device:
        raise ValueError("Invalid contact status buffer")
    if (
        state.particle_q.shape != (model.particle_count,)
        or state.particle_q.dtype != wp.vec3
        or state.particle_q.device != model.device
    ):
        raise ValueError("Invalid contact particle state")
    residual = workspace._residual[mode]
    residual.zero_()
    workspace._projection[mode].zero_()
    workspace._diagnostic_values[mode].zero_()
    workspace._seen.zero_()
    capacity = pipeline.soft_contact_max
    if capacity:
        wp.launch(
            _validate_record_slots,
            capacity,
            [
                pipeline._face_pairs,
                model.tri_indices,
                contacts.soft_contact_tids,
                contacts.soft_contact_count,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                workspace._seen,
                workspace._record_candidate_tid,
                out_status,
            ],
            device=model.device,
        )
        wp.launch(
            _validate_record_coverage,
            capacity,
            [contacts.soft_contact_count, workspace._seen, out_status],
            device=model.device,
        )
    elif int(contacts.soft_contact_count.numpy()[0]) != 0:
        out_status.fill_(6)
    workspace._forces[mode].zero_()
    workspace._moments[mode].zero_()
    workspace._samples[mode].zero_()
    context = _ContactStepContext()
    context.smoothing, context.mu, context.kt = (
        workspace.normal_smoothing_width,
        workspace.friction_coefficient,
        workspace.tangential_stiffness,
    )
    context.dt = workspace._dt
    context.integrate = int(workspace._step_open and not workspace._rollback)
    context.prepare = int(mode == 2 and context.integrate)
    context.transport_cos = workspace.transport_cos
    context.start_x = workspace._start_x
    context.committed, context.pending = workspace.committed, workspace.pending
    context.samples = workspace._samples[mode]
    if context.mu > 0 and mode == 2:
        workspace.pending.valid.zero_()
        workspace.pending.xi_local.zero_()
        workspace.pending.normal_local.zero_()
    if context.mu > 0 and mode != 2 and not workspace._step_open:
        raise ValueError("Friction requires a frozen physical step")
    factors = assembly.contact_factors if assembly is not None else None
    wp.launch(
        evaluate_monolithic_p1q3_contacts,
        capacity,
        [
            mode,
            contacts.soft_contact_count,
            capacity,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_normal,
            model.particle_q,
            state.particle_q,
            workspace._mapping,
            model.shape_body,
            model.shape_margin,
            state.body_q,
            model.body_com,
            state.body_qd,
            context,
            articulation.body_to_articulation,
            articulation.body_to_link_index,
            workspace._dof_count,
            articulation.scratch.J,
            model.joint_dof_count,
            pipeline.r_soft,
            workspace.contact_stiffness,
            residual,
            assembly.global_scalar_triplets if assembly is not None else workspace._dummy_triplets,
            factors.count if factors is not None else workspace._dummy_factor_count,
            workspace.factor_contact_record if factors is not None else workspace._dummy_int,
            workspace._record_candidate_tid,
            factors.kind if factors is not None else workspace._dummy_int,
            factors.candidate_tid if factors is not None else workspace._dummy_int,
            workspace.factor_gap if factors is not None else workspace._dummy_scalar,
            factors.weights if factors is not None else workspace._dummy_scalar,
            factors.gq if factors is not None else workspace._dummy_gq,
            factors.gx_columns if factors is not None else workspace._dummy_columns,
            factors.gx_values if factors is not None else workspace._dummy_gx,
            workspace._forces[mode],
            workspace._moments[mode],
            out_status,
        ],
        device=model.device,
    )
    if context.prepare and context.mu > 0:
        wp.launch(
            _lost_history_samples,
            capacity,
            [
                workspace.committed,
                workspace.pending,
                pipeline._face_pairs,
                model.tri_indices,
                model.particle_q,
                workspace.tangential_stiffness,
                workspace._samples[mode],
            ],
            device=model.device,
        )
    wp.launch(_add_contact_residual, residual.shape[0], [residual, out_residual, out_status], device=model.device)
    wp.launch(
        _contact_diagnostics,
        capacity,
        [
            contacts.soft_contact_count,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_normal,
            state.particle_q,
            model.shape_body,
            model.shape_margin,
            state.body_q,
            model.body_com,
            articulation.body_to_link_index,
            articulation.scratch.J,
            workspace._mapping,
            model.joint_dof_count,
            pipeline.r_soft,
            workspace.normal_smoothing_width,
            workspace._forces[mode],
            workspace._moments[mode],
            workspace._diagnostic_values[mode],
            workspace._projection[mode],
            out_status,
        ],
        device=model.device,
    )


def assemble_current_contacts(
    model: Model,
    state: State,
    contacts: Contacts,
    pipeline: MonolithicCollisionPipeline,
    articulation: MonolithicArticulationWorkspace,
    workspace: MonolithicContactWorkspace,
    assembly: MonolithicLinearAssembly,
    out_residual: wp.array[float],
    *,
    generation: MonolithicLinearGeneration,
) -> None:
    linear = workspace.linear_workspace
    if (
        generation != assembly.generation
        or generation != linear._generation
        or linear._sealed
        or assembly.contact_factors is not linear._factors
        or assembly.global_scalar_triplets is not linear._global
        or generation.contact_generation != int(contacts.contact_generation.numpy()[0])
        or generation.history_epoch != workspace.history_epoch
    ):
        raise ValueError("Current contact assembly generation is stale")
    _evaluate(
        0,
        model,
        state,
        contacts,
        pipeline,
        articulation,
        workspace,
        out_residual,
        assembly.contact_factors.status,
        assembly,
    )


def evaluate_trial_contacts(
    model: Model,
    state: State,
    contacts: Contacts,
    pipeline: MonolithicCollisionPipeline,
    articulation: MonolithicArticulationWorkspace,
    workspace: MonolithicContactWorkspace,
    out_residual: wp.array[float],
    out_status: wp.array[int],
    *,
    owner_generation: MonolithicLinearGeneration,
    trial_generation: int,
) -> None:
    if (
        owner_generation != workspace.linear_workspace._generation
        or trial_generation < 0
        or owner_generation.history_epoch != workspace.history_epoch
        or owner_generation.config_generation != 0
    ):
        raise ValueError("Trial contact owner generation is stale")
    _evaluate(1, model, state, contacts, pipeline, articulation, workspace, out_residual, out_status)


def evaluate_final_contacts(
    model: Model,
    state: State,
    contacts: Contacts,
    pipeline: MonolithicCollisionPipeline,
    articulation: MonolithicArticulationWorkspace,
    workspace: MonolithicContactWorkspace,
    out_residual: wp.array[float],
    out_status: wp.array[int],
    *,
    step_generation: int,
    returned_state_role: _MonolithicReturnedStateRole,
) -> None:
    workspace.invalidate_final_force()
    if not isinstance(returned_state_role, _MonolithicReturnedStateRole) or step_generation < 0:
        raise ValueError("Invalid final returned-state role/generation")
    if (
        workspace.friction_coefficient > 0
        and not workspace._step_open
        and returned_state_role != _MonolithicReturnedStateRole.STATE_IN_ROLLBACK
    ):
        raise ValueError("Final history preparation requires an open physical step")
    workspace._rollback = returned_state_role == _MonolithicReturnedStateRole.STATE_IN_ROLLBACK
    try:
        _evaluate(2, model, state, contacts, pipeline, articulation, workspace, out_residual, out_status)
    finally:
        workspace._rollback = False
    if int(out_status.numpy()[0]) == 0:
        workspace._prepared = returned_state_role != _MonolithicReturnedStateRole.STATE_IN_ROLLBACK
        workspace._final_contact_count = int(contacts.soft_contact_count.numpy()[0])
        workspace._final_generation = _MonolithicFinalForceGeneration(
            step_generation,
            contacts,
            int(contacts.contact_generation.numpy()[0]),
            state,
            returned_state_role,
            workspace.config_sha256,
            workspace.history_epoch + int(workspace._prepared and workspace.friction_coefficient > 0),
        )
