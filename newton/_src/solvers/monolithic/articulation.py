# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision kinematics for the Monolithic articulation incremental objective."""

import hashlib
import json
import struct
from typing import Any

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import BodyFlags, JointType
from .joint_parameters import _validate_joint_coordinate_layout


@wp.kernel
def _forward(
    types: wp.array[int],
    parents: wp.array[int],
    children: wp.array[int],
    starts: wp.array[int],
    axes: wp.array[Any],
    xp: wp.array[Any],
    xc: wp.array[Any],
    local_com: wp.array[Any],
    q: wp.array[Any],
    poses: wp.array[Any],
    rotations: wp.array[Any],
    com: wp.array[Any],
    world_axes: wp.array[Any],
    pivots: wp.array[Any],
    joint_rotations: wp.array[Any],
    joint_translations: wp.array[Any],
):
    # One tree, in parent-before-child order; no cross-thread FK dependency.
    for j in range(types.shape[0]):
        parent = type(xp[0])(type(axes[0])(0.0), wp.quat_identity(dtype=q.dtype))
        if parents[j] >= 0:
            parent = poses[parents[j]]
        joint_frame = parent * xp[j]
        motion = type(xp[0])(type(axes[0])(0.0), wp.quat_identity(dtype=q.dtype))
        world_axes[j] = type(axes[0])(0.0)
        pivots[j] = wp.transform_get_translation(joint_frame)
        if types[j] != JointType.FIXED:
            i = starts[j]
            axis = axes[i]
            world_axes[j] = wp.transform_vector(joint_frame, axis)
            if types[j] == JointType.REVOLUTE:
                motion = type(xp[0])(type(axes[0])(0.0), wp.quat_from_axis_angle(axis, q[i]))
            else:
                motion = type(xp[0])(axis * q[i], wp.quat_identity(dtype=q.dtype))
        joint_rotations[j] = wp.quat_to_matrix(wp.transform_get_rotation(motion))
        joint_translations[j] = wp.transform_get_translation(motion)
        child = children[j]
        pose = joint_frame * motion * wp.transform_inverse(xc[j])
        # Maintain a proper rotation through long fixed/revolute chains.
        pose = type(xp[0])(wp.transform_get_translation(pose), wp.normalize(wp.transform_get_rotation(pose)))
        poses[child] = pose
        rotations[child] = wp.quat_to_matrix(wp.transform_get_rotation(pose))
        com[child] = wp.transform_point(pose, local_com[child])


@wp.kernel
def _jacobians(
    ancestors: wp.array2d[int],
    dof_joint: wp.array[int],
    types: wp.array[int],
    axes: wp.array[Any],
    pivots: wp.array[Any],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
):
    body, dof = wp.tid()
    translation = type(com[0])(0.0)
    angular = type(com[0])(0.0)
    if ancestors[body, dof] != 0:
        joint = dof_joint[dof]
        if types[joint] == JointType.REVOLUTE:
            angular = axes[joint]
            translation = wp.cross(angular, com[body] - pivots[joint])
        else:
            translation = axes[joint]
    jt[body, dof] = translation
    jw[body, dof] = angular


@wp.kernel
def _body_outputs(
    poses: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    qd: wp.array[Any],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    b = wp.tid()
    linear, angular = type(jt[0, 0])(0.0), type(jt[0, 0])(0.0)
    for i in range(qd.shape[0]):
        linear += jt[b, i] * qd[i]
        angular += jw[b, i] * qd[i]
    body_q[b] = wp.transform(
        wp.vec3(wp.transform_get_translation(poses[b])), wp.quat(wp.transform_get_rotation(poses[b]))
    )
    body_qd[b] = wp.spatial_vector(wp.vec3(linear), wp.vec3(angular))


def _configuration_digest(fields):
    digest = hashlib.sha256()
    for name, field in sorted(fields.items()):
        value = np.asarray(field)
        value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        metadata = json.dumps(
            {"schema_version": 1, "field_name": name, "shape": value.shape, "dtype": value.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for part in (metadata, value.tobytes()):
            digest.update(struct.pack("<Q", len(part)))
            digest.update(part)
    return digest.digest()


class _TreeKinematics:
    """Own the immutable geometry and native scratch for one supported tree."""

    _fields = (
        "articulation_start",
        "articulation_end",
        "joint_type",
        "joint_parent",
        "joint_child",
        "joint_q_start",
        "joint_qd_start",
        "joint_dof_dim",
        "joint_enabled",
        "joint_axis",
        "joint_X_p",
        "joint_X_c",
        "body_com",
        "body_mass",
        "body_inertia",
        "body_flags",
        "body_world",
        "joint_armature",
        "joint_damping",
        "joint_target_mode",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_friction",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_target_q_start",
        "shape_flags",
    )

    def __init__(self, model, *, extra_config=None, dtype=wp.float64):
        if dtype not in (wp.float64, wp.float32):
            raise ValueError("Articulation requires float64 or float32")
        self.dtype = dtype
        self.numpy_dtype = np.float64 if dtype == wp.float64 else np.float32
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
        self.transform = wp.transformd if dtype == wp.float64 else wp.transform
        _validate_joint_coordinate_layout(model)
        if model.articulation_count != 1 or not model.joint_dof_count:
            raise ValueError("Monolithic articulation requires one nonempty tree")
        if model.requires_grad or model.constraint_mimic_count or model.actuators:
            raise ValueError("Monolithic articulation does not support autograd, mimic constraints or model actuators")
        if model.muscle_count:
            raise ValueError("Monolithic articulation does not support muscle forces")
        self.model, self.device = model, model.device
        self.counts = {
            name: getattr(model, name)
            for name in (
                "articulation_count",
                "particle_count",
                "tet_count",
                "joint_dof_count",
                "joint_count",
                "body_count",
                "muscle_count",
                "constraint_mimic_count",
                "requires_grad",
            )
        }
        self.sources = {name: getattr(model, name) for name in self._fields}
        self.values = {name: array.numpy().copy() for name, array in self.sources.items()}
        v = self.values
        if v["articulation_start"].tolist() != [0, model.joint_count] or v["articulation_end"].tolist() != [
            model.joint_count
        ]:
            raise ValueError("Unowned joints are unsupported")
        if not v["joint_enabled"].all() or len(np.unique(v["body_world"])) != 1:
            raise ValueError("Monolithic requires enabled joints in one world")
        self.count, self.bodies, self.joints = model.joint_dof_count, model.body_count, model.joint_count
        seen, roots = set(), 0
        ancestors = np.zeros((self.bodies, self.count), dtype=np.int32)
        dof_joint = np.zeros(self.count, dtype=np.int32)
        for j, (parent, child) in enumerate(zip(v["joint_parent"], v["joint_child"], strict=True)):
            if child < 0 or child >= self.bodies or child in seen or (parent != -1 and parent not in seen):
                raise ValueError("Monolithic requires a parent-before-child world-anchored tree")
            roots += int(parent == -1)
            seen.add(int(child))
            if parent >= 0:
                ancestors[child] = ancestors[parent]
            if v["joint_type"][j] != JointType.FIXED:
                dof = v["joint_q_start"][j]
                ancestors[child, dof] = 1
                dof_joint[dof] = j
        if roots != 1 or len(seen) != self.bodies:
            raise ValueError("Monolithic tree must own every body")
        if not np.isin(v["body_flags"], [int(BodyFlags.DYNAMIC), int(BodyFlags.KINEMATIC)]).all():
            raise ValueError(
                "Monolithic articulation requires dynamic or kinematic body flags; proxies are unsupported"
            )
        if np.any((v["body_flags"] == int(BodyFlags.KINEMATIC)) & ancestors.any(axis=1)):
            raise ValueError("Monolithic articulation does not support kinematic bodies with movable ancestors")
        for name, value in v.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite articulation configuration: {name}")
        inertia = v["body_inertia"].astype(self.numpy_dtype)
        second = (
            self.numpy_dtype(0.5)
            * np.trace(inertia, axis1=1, axis2=2)[:, None, None]
            * np.eye(3, dtype=self.numpy_dtype)
            - inertia
        )
        if (
            np.any(v["body_mass"] < 0)
            or not np.allclose(inertia, inertia.transpose(0, 2, 1), atol=0, rtol=1e-6)
            or np.any(np.linalg.eigvalsh(second) < 0)
        ):
            raise ValueError("Invalid physical mass or second moment")
        for name in ("joint_armature", "joint_damping"):
            if np.any(v[name] < 0):
                raise ValueError(f"Negative {name}")
        fields = dict(v)
        fields.update({} if extra_config is None else extra_config)
        if dtype == wp.float32:
            fields["native_precision"] = np.array([32], dtype=np.int32)
        self.digest = _configuration_digest(fields)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.types = array(v["joint_type"], int)
        self.parents = array(v["joint_parent"], int)
        self.children = array(v["joint_child"], int)
        self.starts = array(v["joint_q_start"], int)
        self.dof_joint = array(dof_joint, int)
        self.ancestors = array(ancestors, int)
        axes = v["joint_axis"].astype(self.numpy_dtype)
        norms = np.linalg.norm(axes, axis=1)
        if np.any(abs(norms - 1) > 1e-5):
            raise ValueError("Joint axes must be unit directions")
        self.axes = array(axes / norms[:, None], self.vector)
        for name in ("joint_X_p", "joint_X_c"):
            transforms = v[name].astype(self.numpy_dtype)
            norms = np.linalg.norm(transforms[:, 3:], axis=1)
            if np.any(abs(norms - 1) > 1e-5):
                raise ValueError("Static joint rotations must be unit quaternions")
            transforms[:, 3:] /= norms[:, None]
            setattr(self, "xp" if name == "joint_X_p" else "xc", array(transforms, self.transform))
        self.local_com = array(v["body_com"].astype(self.numpy_dtype), self.vector)
        self.mass = array(v["body_mass"].astype(self.numpy_dtype), self.dtype)
        self.second = array(second, self.matrix)
        self.poses = wp.zeros(self.bodies, dtype=self.transform, device=self.device)
        self.rotations = wp.zeros(self.bodies, dtype=self.matrix, device=self.device)
        self.com = wp.zeros(self.bodies, dtype=self.vector, device=self.device)
        self.world_axes = wp.zeros(self.joints, dtype=self.vector, device=self.device)
        self.pivots = wp.zeros(self.joints, dtype=self.vector, device=self.device)
        self.joint_rotations = wp.zeros(self.joints, dtype=self.matrix, device=self.device)
        self.joint_translations = wp.zeros(self.joints, dtype=self.vector, device=self.device)
        self.jt = wp.zeros((self.bodies, self.count), dtype=self.vector, device=self.device)
        self.jw = wp.zeros_like(self.jt)
        self.evaluate(wp.array(model.joint_q.numpy(), dtype=self.dtype, device=self.device))
        self.validate_inertia()

    def timestep(self, value):
        """Validate a timestep in the arithmetic used by this component."""
        with np.errstate(over="ignore", under="ignore"):
            value = self.numpy_dtype(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Articulation requires a representable finite positive dt")
        return value

    def validate_inertia(self):
        """Require physical positive inertia at the currently evaluated private FK."""
        v = self.values
        jt, jw, rotation = self.jt.numpy(), self.jw.numpy(), self.rotations.numpy()
        world_inertia = rotation @ v["body_inertia"].astype(self.numpy_dtype) @ rotation.transpose(0, 2, 1)
        effective_inertia = np.einsum("bni,bmi,b->nm", jt, jt, v["body_mass"])
        effective_inertia += np.einsum("bni,bij,bmj->nm", jw, world_inertia, jw)
        effective_inertia += np.diag(v["joint_armature"].astype(self.numpy_dtype))
        if not np.isfinite(effective_inertia).all() or np.linalg.eigvalsh(effective_inertia).min() <= 0:
            raise ValueError("Articulation inertia must be positive definite on every free DOF")

    def validate_identity(self):
        """Reject storage replacement and in-place configuration edits."""
        if self.model.actuators or any(getattr(self.model, name) != value for name, value in self.counts.items()):
            raise ValueError("Articulation configuration changed: unsupported scope or counts")
        for name, original in self.sources.items():
            current = getattr(self.model, name)
            if current is not original or not np.array_equal(current.numpy(), self.values[name]):
                raise ValueError(f"Articulation configuration changed: {name}")

    def evaluate(self, q, *, state=None):
        """Evaluate FK and COM/angular Jacobians without using public FP32 FK."""
        if (q.shape, q.dtype, q.device) != ((self.count,), self.dtype, self.device):
            raise ValueError("Monolithic FK requires owned-shape native coordinates on the model device")
        wp.launch(
            _forward,
            1,
            [
                self.types,
                self.parents,
                self.children,
                self.starts,
                self.axes,
                self.xp,
                self.xc,
                self.local_com,
                q,
                self.poses,
                self.rotations,
                self.com,
                self.world_axes,
                self.pivots,
                self.joint_rotations,
                self.joint_translations,
            ],
            device=self.device,
        )
        wp.launch(
            _jacobians,
            (self.bodies, self.count),
            [self.ancestors, self.dof_joint, self.types, self.world_axes, self.pivots, self.com, self.jt, self.jw],
            device=self.device,
        )
        if state is not None:
            wp.launch(
                _body_outputs,
                self.bodies,
                [self.poses, self.jt, self.jw, state.joint_qd, state.body_q, state.body_qd],
                device=self.device,
            )


class ArticulationKinematics(_TreeKinematics):
    """Retain the q-only entrance contract while sharing supported-tree FK."""

    def __init__(self, model, *, extra_config=None, dtype=wp.float64):
        if model.particle_count or model.tet_count:
            raise ValueError("Monolithic articulation requires one nonempty tree and no particles/tets")
        if np.any(model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)):
            raise ValueError("Monolithic articulation requires collision-disabled shapes")
        super().__init__(model, extra_config=extra_config, dtype=dtype)


@wp.func
def _inner(a: Any, b: Any):
    result = type(a[0, 0])(0.0)
    for i in range(3):
        for j in range(3):
            result += a[i, j] * b[i, j]
    return result


@wp.func
def _limit_primitive(y: Any, dn: Any, width: Any):
    z = wp.min(wp.max(y, type(y)(0.0)), width)
    value = type(y)(0.75) * z * z * z * z / (width * width) - dn * z * z * z / (width * width)
    value += -type(y)(0.4) * z * z * z * z * z / (width * width * width) + dn * z * z * z * z / (
        type(y)(2.0) * width * width * width
    )
    if y > width:
        value += (y - width) * ((y + width) * type(y)(0.5) - dn)
    return value


@wp.func
def _limit_variation(d: Any, dn: Any, ke: Any, kd: Any, width: Any, h: Any):
    zero = type(d)(0.0)
    penetration = wp.max(d, zero)
    activation, derivative = zero, zero
    if d >= width:
        activation = type(d)(1.0)
    elif d > zero:
        t = d / width
        activation = t * t * (type(d)(3.0) - type(d)(2.0) * t)
        derivative = type(d)(6.0) * t * (type(d)(1.0) - t) / width
    outward = wp.max(d - dn, zero)
    energy = ke * penetration * penetration * type(d)(0.5)
    if d > wp.max(dn, zero):
        energy += kd / h * (_limit_primitive(d, dn, width) - _limit_primitive(wp.max(dn, zero), dn, width))
    residual = ke * penetration + kd / h * activation * outward
    tangent = kd / h * derivative * outward
    if d > zero:
        tangent += ke
    if d > dn:
        tangent += kd / h * activation
    return energy, residual, tangent


@wp.kernel
def _joint_variation(
    q: wp.array[Any],
    qn: wp.array[Any],
    vn: wp.array[Any],
    params: wp.array2d[Any],
    armature: wp.array[Any],
    damping: wp.array[Any],
    qt: wp.array[Any],
    vt: wp.array[Any],
    force: wp.array[Any],
    h: Any,
    energy: wp.array[Any],
    residual: wp.array[Any],
    tangent: wp.array[Any],
):
    i = wp.tid()
    delta = q[i] - qn[i]
    v = delta / h
    inertial = delta - h * vn[i]
    e = (
        armature[i] * inertial * inertial / (type(h)(2.0) * h * h)
        + damping[i] * delta * delta / (type(h)(2.0) * h)
        - force[i] * delta
    )
    r = armature[i] * inertial / (h * h) + damping[i] * v - force[i]
    b = armature[i] / (h * h) + damping[i] / h
    kp, kd, effort = params[i, 0], params[i, 1], params[i, 2]
    a = kp + kd / h
    if a > type(h)(0.0):
        u = kp * (q[i] - qt[i]) + kd * (v - vt[i])
        r += wp.clamp(u, -effort, effort)
        if wp.abs(u) < effort:
            e += u * u / (type(h)(2.0) * a)
            b += a
        else:
            e += effort * (wp.abs(u) - effort * type(h)(0.5)) / a
    for side in range(2):
        sign = type(h)(1.0)
        bound = params[i, 4]
        if side == 1:
            sign = type(h)(-1.0)
            bound = params[i, 3]
        el, rl, bl = _limit_variation(
            sign * (q[i] - bound), sign * (qn[i] - bound), params[i, 5], params[i, 6], params[i, 7], h
        )
        e += el
        r += sign * rl
        b += bl
    f, eps = params[i, 8], params[i, 9]
    # Scaled hypot avoids intermediate overflow for finite velocities.
    scale = wp.max(wp.abs(v), eps)
    denom = scale * wp.sqrt((v / scale) * (v / scale) + (eps / scale) * (eps / scale))
    e += h * f * v * (v / (denom + eps))
    r += f * (v / denom)
    b += f / h * (eps / denom) * (eps / denom) / denom
    energy[i] = e
    residual[i] = r
    tangent[i] = b


@wp.kernel
def _link_variation(
    t: wp.array[Any],
    r: wp.array[Any],
    tn: wp.array[Any],
    predicted_t: wp.array[Any],
    predicted_r: wp.array[Any],
    mass: wp.array[Any],
    second: wp.array[Any],
    force: wp.array[Any],
    h: Any,
    energy: wp.array[Any],
    translation_gradient: wp.array[Any],
    rotation_gradient: wp.array[Any],
    rotation_tangent: wp.array[Any],
    build_tangent: bool,
):
    b = wp.tid()
    dt, dr = t[b] - predicted_t[b], r[b] - predicted_r[b]
    rs = dr * second[b]
    energy[b] = (mass[b] * wp.dot(dt, dt) + _inner(rs, dr)) / (type(h)(2.0) * h * h) - wp.dot(force[b], t[b] - tn[b])
    translation_gradient[b] = mass[b] * dt / (h * h) - force[b]
    a = rs * wp.transpose(r[b]) / (h * h)
    rotation_gradient[b] = type(t[0])(a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1])
    if build_tangent:
        c = r[b] * second[b] * wp.transpose(predicted_r[b])
        local = (wp.identity(n=3, dtype=type(h)) * wp.trace(c) - (c + wp.transpose(c)) * type(h)(0.5)) / (h * h)
        vectors, values = wp.eig3(local)
        positive = type(t[0])(
            wp.max(values[0], type(h)(0.0)), wp.max(values[1], type(h)(0.0)), wp.max(values[2], type(h)(0.0))
        )
        rotation_tangent[b] = vectors * wp.diag(positive) * wp.transpose(vectors)


@wp.kernel
def _reduced_variation(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    mass: wp.array[Any],
    gt: wp.array[Any],
    gr: wp.array[Any],
    kr: wp.array[Any],
    joint_r: wp.array[Any],
    joint_b: wp.array[Any],
    h: Any,
    residual: wp.array[Any],
    tangent: wp.array2d[Any],
    tangent32: wp.array2d[float],
):
    i, j = wp.tid()
    b = type(h)(0.0)
    r = joint_r[i]
    if i == j:
        b = joint_b[i]
    for link in range(mass.shape[0]):
        b += mass[link] * wp.dot(jt[link, i], jt[link, j]) / (h * h)
        b += (wp.dot(jw[link, i], kr[link] * jw[link, j]) + wp.dot(jw[link, j], kr[link] * jw[link, i])) * type(h)(0.5)
        if i == j:
            r += wp.dot(jt[link, i], gt[link]) + wp.dot(jw[link, i], gr[link])
    tangent[i, j] = b
    tangent32[i, j] = float(b)
    if i == j:
        residual[i] = r


@wp.kernel
def _reduced_residual(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    gt: wp.array[Any],
    gr: wp.array[Any],
    joint_r: wp.array[Any],
    residual: wp.array[Any],
):
    i = wp.tid()
    r = joint_r[i]
    for link in range(gt.shape[0]):
        r += wp.dot(jt[link, i], gt[link]) + wp.dot(jw[link, i], gr[link])
    residual[i] = r


class ArticulationObjective:
    """Evaluate the frozen physical incremental objective on device in the kinematics dtype."""

    def __init__(self, kinematics, parameters):
        self.k = kinematics
        k = kinematics
        n, b, device = k.count, k.bodies, k.device
        self.params = wp.array(parameters, dtype=k.dtype, device=device)
        self.armature = wp.array(k.values["joint_armature"], dtype=k.dtype, device=device)
        self.damping = wp.array(k.values["joint_damping"], dtype=k.dtype, device=device)
        for name in ("qn", "vn", "qt", "vt", "joint_force", "joint_energy", "joint_r", "joint_b", "residual"):
            setattr(self, name, wp.zeros(n, dtype=k.dtype, device=device))
        for name in ("tn", "predicted_t", "force", "gt", "gr"):
            setattr(self, name, wp.zeros(b, dtype=k.vector, device=device))
        for name in ("rn", "predicted_r", "kr"):
            setattr(self, name, wp.zeros(b, dtype=k.matrix, device=device))
        self.link_energy = wp.zeros(b, dtype=k.dtype, device=device)
        self.tangent = wp.zeros((n, n), dtype=k.dtype, device=device)
        self.tangent32 = wp.zeros((n, n), dtype=wp.float32, device=device)

    def evaluate(self, q, h, *, build_tangent=True):
        """Evaluate without reading derived FP32 body outputs or changing history."""
        k = self.k
        h = k.timestep(h)
        k.evaluate(q)
        wp.launch(
            _joint_variation,
            k.count,
            [
                q,
                self.qn,
                self.vn,
                self.params,
                self.armature,
                self.damping,
                self.qt,
                self.vt,
                self.joint_force,
                k.dtype(h),
                self.joint_energy,
                self.joint_r,
                self.joint_b,
            ],
            device=k.device,
        )
        wp.launch(
            _link_variation,
            k.bodies,
            [
                k.com,
                k.rotations,
                self.tn,
                self.predicted_t,
                self.predicted_r,
                k.mass,
                k.second,
                self.force,
                k.dtype(h),
                self.link_energy,
                self.gt,
                self.gr,
                self.kr,
                build_tangent,
            ],
            device=k.device,
        )
        if not build_tangent:
            wp.launch(
                _reduced_residual, k.count, [k.jt, k.jw, self.gt, self.gr, self.joint_r, self.residual], device=k.device
            )
            return
        wp.launch(
            _reduced_variation,
            (k.count, k.count),
            [
                k.jt,
                k.jw,
                k.mass,
                self.gt,
                self.gr,
                self.kr,
                self.joint_r,
                self.joint_b,
                k.dtype(h),
                self.residual,
                self.tangent,
                self.tangent32,
            ],
            device=k.device,
        )
