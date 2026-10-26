# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision kinematics for the Monolithic articulation incremental objective."""

import hashlib
import itertools
import json
import struct
from typing import Any

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import BodyFlags, JointType
from .actors import articulation_view
from .convergence import abs3, axial_absolute
from .joint_parameters import _validate_joint_coordinate_layout


@wp.kernel
def _forward(
    tree_joints: wp.array[int],
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
    # Each thread owns one tree, in parent-before-child order.
    tree = wp.tid()
    for j in range(tree_joints[tree], tree_joints[tree + 1]):
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


@wp.func
def _axis_rotation_step(axis: Any, angle: Any):
    """Return ``Rot(axis, angle) - I`` without cancellation for small angles."""
    k = wp.skew(axis)
    half = wp.sin(angle * type(angle)(0.5))
    return wp.sin(angle) * k + type(angle)(2.0) * half * half * (k * k)


@wp.kernel
def _forward_step(
    tree_joints: wp.array[int],
    types: wp.array[int],
    parents: wp.array[int],
    children: wp.array[int],
    starts: wp.array[int],
    axes: wp.array[Any],
    xp: wp.array[Any],
    xc: wp.array[Any],
    local_com: wp.array[Any],
    q: wp.array[Any],
    step: wp.array[Any],
    rn: wp.array[Any],
    axes_n: wp.array[Any],
    rotation_steps: wp.array[Any],
    translation_steps: wp.array[Any],
    com_steps: wp.array[Any],
    joint_rotation_steps: wp.array[Any],
    joint_translation_steps: wp.array[Any],
):
    # Advance every link from its step-start pose by the joint increments, in
    # parent-before-child order: R = (I + E) R_n and t = t_n + d. Each update is
    # a sum of increment-size terms, so it keeps the increment's precision.
    zero = type(axes[0])(0.0)
    tree = wp.tid()
    for j in range(tree_joints[tree], tree_joints[tree + 1]):
        parent_e = wp.matrix(dtype=type(step[0]), shape=(3, 3))
        parent_d = zero
        parent_rn = wp.identity(n=3, dtype=type(step[0]))
        if parents[j] >= 0:
            parent_e = rotation_steps[parents[j]]
            parent_d = translation_steps[parents[j]]
            parent_rn = rn[parents[j]]
        world = wp.matrix(dtype=type(step[0]), shape=(3, 3))
        local = wp.matrix(dtype=type(step[0]), shape=(3, 3))
        slide, linear = zero, zero
        if types[j] != JointType.FIXED:
            i = starts[j]
            if types[j] == JointType.REVOLUTE:
                world = _axis_rotation_step(axes_n[j], step[i])
                local = _axis_rotation_step(axes[i], step[i])
            else:
                slide = axes[i] * step[i]
                linear = parent_e * (axes_n[j] * q[i]) + axes_n[j] * step[i]
        child = children[j]
        e = parent_e + world + parent_e * world
        d = parent_d + parent_e * (parent_rn * wp.transform_get_translation(xp[j])) + linear
        d -= e * (rn[child] * wp.transform_get_translation(xc[j]))
        rotation_steps[child] = e
        translation_steps[child] = d
        com_steps[child] = d + e * (rn[child] * local_com[child])
        joint_rotation_steps[j] = local
        joint_translation_steps[j] = slide


@wp.kernel
def _jacobians(
    ancestors: wp.array[int],
    jac_offsets: wp.array[int],
    body_dof_starts: wp.array[int],
    body_dof_counts: wp.array[int],
    dof_joint: wp.array[int],
    types: wp.array[int],
    axes: wp.array[Any],
    pivots: wp.array[Any],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
):
    body, local = wp.tid()
    if local >= body_dof_counts[body]:
        return
    dof = body_dof_starts[body] + local
    entry = jac_offsets[body] + local
    translation = type(com[0])(0.0)
    angular = type(com[0])(0.0)
    if ancestors[entry] != 0:
        joint = dof_joint[dof]
        if types[joint] == JointType.REVOLUTE:
            angular = axes[joint]
            translation = wp.cross(angular, com[body] - pivots[joint])
        else:
            translation = axes[joint]
    jt[entry / jt.shape[1], entry % jt.shape[1]] = translation
    jw[entry / jw.shape[1], entry % jw.shape[1]] = angular


@wp.func
def _packed_column(values: wp.array2d[Any], entry: int):
    """Read a scalar column of a packed tree-local rectangular block."""
    return values[entry / values.shape[1], entry % values.shape[1]]


@wp.kernel
def _body_outputs(
    poses: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    jac_offsets: wp.array[int],
    body_dof_starts: wp.array[int],
    body_dof_counts: wp.array[int],
    qd: wp.array[Any],
    body_map: wp.array[int],
    dof_map: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    b = wp.tid()
    linear, angular = type(jt[0, 0])(0.0), type(jt[0, 0])(0.0)
    for local in range(body_dof_counts[b]):
        i = body_dof_starts[b] + local
        rate = type(jt[0, 0][0])(qd[dof_map[i]])
        linear += _packed_column(jt, jac_offsets[b] + local) * rate
        angular += _packed_column(jw, jac_offsets[b] + local) * rate
    body_q[body_map[b]] = wp.transform(
        wp.vec3(wp.transform_get_translation(poses[b])), wp.quat(wp.transform_get_rotation(poses[b]))
    )
    body_qd[body_map[b]] = wp.spatial_vector(wp.vec3(linear), wp.vec3(angular))


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


@wp.kernel
def _difference(a: wp.array[Any], b: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = a[i] - b[i]


@wp.kernel
def _gather_native(source: wp.array[Any], mapping: wp.array[int], target: wp.array[Any]):
    i = wp.tid()
    target[i] = target.dtype(source[mapping[i]])


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

    def __init__(self, model, *, actor=None, actors=None, extra_config=None, dtype=wp.float64):
        if actors is not None:
            if actor is not None:
                raise ValueError("Specify one mechanical view or a collection of trees")
            actors = tuple(actors)
            actor = articulation_view(actors)
        if dtype not in (wp.float64, wp.float32):
            raise ValueError("Articulation requires float64 or float32")
        self.dtype = dtype
        self.numpy_dtype = np.float64 if dtype == wp.float64 else np.float32
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
        self.transform = wp.transformd if dtype == wp.float64 else wp.transform
        _validate_joint_coordinate_layout(model, actor)
        empty = actor is not None and not actor.joint_indices and not actor.body_indices and not actor.dof_indices
        if not empty and ((actor is None and model.articulation_count != 1) or not model.joint_dof_count):
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
                "joint_coord_count",
                "joint_count",
                "body_count",
                "muscle_count",
                "constraint_mimic_count",
                "requires_grad",
            )
        }
        self.sources = {name: getattr(model, name) for name in self._fields}
        self._source_values = {name: array.numpy().copy() for name, array in self.sources.items()}
        v = dict(self._source_values)
        if actor is None and (
            v["articulation_start"].tolist() != [0, model.joint_count]
            or v["articulation_end"].tolist() != [model.joint_count]
        ):
            raise ValueError("Unowned joints are unsupported")
        body_ids = list(range(model.body_count)) if actor is None else list(actor.body_indices)
        joint_ids = list(range(model.joint_count)) if actor is None else list(actor.joint_indices)
        dof_ids = list(range(model.joint_dof_count)) if actor is None else list(actor.dof_indices)
        if actor is not None:
            joint_fields = {
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_q_start",
                "joint_qd_start",
                "joint_dof_dim",
                "joint_enabled",
                "joint_X_p",
                "joint_X_c",
                "joint_target_q_start",
            }
            for name, value in v.items():
                if name.startswith("body_"):
                    v[name] = value[body_ids]
                elif name in joint_fields:
                    v[name] = value[joint_ids]
                elif name.startswith("joint_"):
                    v[name] = value[dof_ids]
            body_local = {body: i for i, body in enumerate(body_ids)}
            v["joint_parent"] = np.array([body_local[p] if p >= 0 else -1 for p in v["joint_parent"]], dtype=np.int32)
            v["joint_child"] = np.array([body_local[b] for b in v["joint_child"]], dtype=np.int32)
            prefix = np.r_[0, np.cumsum(v["joint_type"] != JointType.FIXED)].astype(np.int32)
            v["joint_q_start"] = v["joint_qd_start"] = prefix
        self.values = v
        if not v["joint_enabled"].all() or (not empty and len(np.unique(v["body_world"])) != 1):
            raise ValueError("Monolithic requires enabled joints in one world")
        self.count, self.bodies, self.joints = len(dof_ids), len(body_ids), len(joint_ids)
        if not self.count and not empty:
            raise ValueError("Monolithic articulation requires one nonempty tree")
        sizes = (
            [(self.bodies, self.joints, self.count)]
            if actors is None and self.count
            else [(len(a.body_indices), len(a.joint_indices), a.dof_count) for a in (actors or ())]
        )
        self.tree_count = len(sizes)
        tree_bodies = np.r_[0, np.cumsum([s[0] for s in sizes])].astype(np.int32)
        tree_joints = np.r_[0, np.cumsum([s[1] for s in sizes])].astype(np.int32)
        tree_dofs = np.r_[0, np.cumsum([s[2] for s in sizes])].astype(np.int32)
        body_dof_starts = np.repeat(tree_dofs[:-1], [s[0] for s in sizes]).astype(np.int32)
        body_dof_counts = np.repeat([s[2] for s in sizes], [s[0] for s in sizes]).astype(np.int32)
        self._tree_sizes = sizes
        self.max_dofs = max((s[2] for s in sizes), default=0)
        jac_offsets = np.r_[0, np.cumsum(body_dof_counts)].astype(np.int32)
        seen, roots = set(), 0
        ancestors = [np.zeros(count, dtype=np.int32) for count in body_dof_counts]
        dof_joint = np.zeros(self.count, dtype=np.int32)
        for j, (parent, child) in enumerate(zip(v["joint_parent"], v["joint_child"], strict=True)):
            if child < 0 or child >= self.bodies or child in seen or (parent != -1 and parent not in seen):
                raise ValueError("Monolithic requires a parent-before-child world-anchored tree")
            roots += int(parent == -1)
            seen.add(int(child))
            if parent >= 0:
                if body_dof_starts[parent] != body_dof_starts[child]:
                    raise ValueError("Mechanical parent must belong to the same articulation")
                ancestors[child][:] = ancestors[parent]
            if v["joint_type"][j] != JointType.FIXED:
                dof = v["joint_q_start"][j]
                ancestors[child][dof - body_dof_starts[child]] = 1
                dof_joint[dof] = j
        if roots != self.tree_count or len(seen) != self.bodies:
            raise ValueError("Monolithic tree must own every body")
        if not np.isin(v["body_flags"], [int(BodyFlags.DYNAMIC), int(BodyFlags.KINEMATIC)]).all():
            raise ValueError(
                "Monolithic articulation requires dynamic or kinematic body flags; proxies are unsupported"
            )
        if np.any((v["body_flags"] == int(BodyFlags.KINEMATIC)) & np.array([a.any() for a in ancestors], dtype=bool)):
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
        fields = dict(self._source_values)
        coord_ids = list(range(model.joint_coord_count)) if actor is None else list(actor.coord_indices)
        fields.update(
            mechanical_bodies=np.asarray(body_ids, dtype=np.int32),
            mechanical_joints=np.asarray(joint_ids, dtype=np.int32),
            mechanical_coordinates=np.asarray(coord_ids, dtype=np.int32),
            mechanical_dofs=np.asarray(dof_ids, dtype=np.int32),
            mechanical_trees=np.asarray(sizes, dtype=np.int32).reshape(-1, 3),
        )
        fields.update({} if extra_config is None else extra_config)
        if dtype == wp.float32:
            fields["native_precision"] = np.array([32], dtype=np.int32)
        self.digest = _configuration_digest(fields)

        def array(data, dtype):
            return wp.array(data, dtype=dtype, device=self.device)

        self.body_map = array(body_ids, int)
        self.dof_map = array(dof_ids, int)
        self.types = array(v["joint_type"], int)
        self.parents = array(v["joint_parent"], int)
        self.children = array(v["joint_child"], int)
        self.starts = array(v["joint_q_start"], int)
        self.dof_joint = array(dof_joint, int)
        self.ancestors = array(np.concatenate(ancestors) if ancestors else np.empty(0, dtype=np.int32), int)
        self.tree_joints = array(tree_joints, int)
        self.jac_offsets = array(jac_offsets, int)
        self.body_dof_starts = array(body_dof_starts, int)
        self.body_dof_counts = array(body_dof_counts, int)
        self.dof_body_starts = array(np.repeat(tree_bodies[:-1], [s[2] for s in sizes]), int)
        self.dof_body_ends = array(np.repeat(tree_bodies[1:], [s[2] for s in sizes]), int)
        self.matrix_rows = array(
            [i for start, end in itertools.pairwise(tree_dofs) for i in range(start, end) for _ in range(start, end)],
            int,
        )
        self.matrix_columns = array(
            [j for start, end in itertools.pairwise(tree_dofs) for _ in range(start, end) for j in range(start, end)],
            int,
        )
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
        # Link and joint motion increments from the step-start pose.
        self.rotation_steps = wp.zeros(self.bodies, dtype=self.matrix, device=self.device)
        self.translation_steps = wp.zeros(self.bodies, dtype=self.vector, device=self.device)
        self.com_steps = wp.zeros(self.bodies, dtype=self.vector, device=self.device)
        self.joint_rotation_steps = wp.zeros(self.joints, dtype=self.matrix, device=self.device)
        self.joint_translation_steps = wp.zeros(self.joints, dtype=self.vector, device=self.device)
        jac_shape = (self.bodies, self.count) if self.tree_count <= 1 else (int(jac_offsets[-1]), 1)
        self.jt = wp.zeros(jac_shape, dtype=self.vector, device=self.device)
        self.jw = wp.zeros_like(self.jt)
        self._direct_q = coord_ids == list(range(model.joint_coord_count))
        self._direct_qd = dof_ids == list(range(model.joint_dof_count))
        self.coord_map = array(coord_ids, int)
        self.local_q = wp.zeros(self.count, dtype=self.dtype, device=self.device)
        self.local_qd = wp.zeros_like(self.local_q)
        self.evaluate(wp.array(model.joint_q.numpy()[coord_ids], dtype=self.dtype, device=self.device))
        self.validate_inertia()

    def gather_q(self, state):
        if self._direct_q and state.joint_q.dtype == self.dtype:
            return state.joint_q
        wp.launch(_gather_native, self.count, [state.joint_q, self.coord_map, self.local_q], device=self.device)
        return self.local_q

    def gather_qd(self, state):
        if self._direct_qd and state.joint_qd.dtype == self.dtype:
            return state.joint_qd
        wp.launch(_gather_native, self.count, [state.joint_qd, self.dof_map, self.local_qd], device=self.device)
        return self.local_qd

    def timestep(self, value):
        """Validate a timestep in the arithmetic used by this component."""
        with np.errstate(over="ignore", under="ignore"):
            value = self.numpy_dtype(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Articulation requires a representable finite positive dt")
        return value

    def validate_inertia(self):
        """Require physical positive inertia at the currently evaluated private FK."""
        if not self.count:
            return
        v = self.values
        jt, jw, rotation = self.jt.numpy().reshape(-1, 3), self.jw.numpy().reshape(-1, 3), self.rotations.numpy()
        world_inertia = rotation @ v["body_inertia"].astype(self.numpy_dtype) @ rotation.transpose(0, 2, 1)
        body, dof, offset = 0, 0, 0
        for bodies, _, count in self._tree_sizes:
            t = jt[offset : offset + bodies * count].reshape(bodies, count, 3)
            w = jw[offset : offset + bodies * count].reshape(bodies, count, 3)
            effective_inertia = np.einsum("bni,bmi,b->nm", t, t, v["body_mass"][body : body + bodies])
            effective_inertia += np.einsum("bni,bij,bmj->nm", w, world_inertia[body : body + bodies], w)
            effective_inertia += np.diag(v["joint_armature"][dof : dof + count].astype(self.numpy_dtype))
            if not np.isfinite(effective_inertia).all() or np.linalg.eigvalsh(effective_inertia).min() <= 0:
                raise ValueError("Articulation inertia must be positive definite on every free DOF")
            body, dof, offset = body + bodies, dof + count, offset + bodies * count

    def validate_identity(self):
        """Reject storage replacement and in-place configuration edits."""
        if self.model.actuators or any(getattr(self.model, name) != value for name, value in self.counts.items()):
            raise ValueError("Articulation configuration changed: unsupported scope or counts")
        for name, original in self.sources.items():
            current = getattr(self.model, name)
            if current is not original or not np.array_equal(current.numpy(), self._source_values[name]):
                raise ValueError(f"Articulation configuration changed: {name}")

    def evaluate(self, q, *, state=None):
        """Evaluate FK and COM/angular Jacobians without using public FP32 FK."""
        if (q.shape, q.dtype, q.device) != ((self.count,), self.dtype, self.device):
            raise ValueError("Monolithic FK requires owned-shape native coordinates on the model device")
        if not self.joints:
            return
        wp.launch(
            _forward,
            self.tree_count,
            [
                self.tree_joints,
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
            (self.bodies, self.max_dofs),
            [
                self.ancestors,
                self.jac_offsets,
                self.body_dof_starts,
                self.body_dof_counts,
                self.dof_joint,
                self.types,
                self.world_axes,
                self.pivots,
                self.com,
                self.jt,
                self.jw,
            ],
            device=self.device,
        )
        if state is not None:
            wp.launch(
                _body_outputs,
                self.bodies,
                [
                    self.poses,
                    self.jt,
                    self.jw,
                    self.jac_offsets,
                    self.body_dof_starts,
                    self.body_dof_counts,
                    state.joint_qd,
                    self.body_map,
                    self.dof_map,
                    state.body_q,
                    state.body_qd,
                ],
                device=self.device,
            )

    def evaluate_step(self, q, step, rn, axes_n):
        """Evaluate link and joint motion increments from step-start rotations and world axes."""
        if not self.joints:
            return
        wp.launch(
            _forward_step,
            self.tree_count,
            [
                self.tree_joints,
                self.types,
                self.parents,
                self.children,
                self.starts,
                self.axes,
                self.xp,
                self.xc,
                self.local_com,
                q,
                step,
                rn,
                axes_n,
                self.rotation_steps,
                self.translation_steps,
                self.com_steps,
                self.joint_rotation_steps,
                self.joint_translation_steps,
            ],
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
    step: wp.array[Any],
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
    absolute: wp.array[Any],
    tangent: wp.array[Any],
):
    i = wp.tid()
    # Terms read the exact step; offsets from the step start are fixed in a step.
    delta = step[i]
    v = delta / h
    inertial = delta - h * vn[i]
    e = (
        armature[i] * inertial * inertial / (type(h)(2.0) * h * h)
        + damping[i] * delta * delta / (type(h)(2.0) * h)
        - force[i] * delta
    )
    r = armature[i] * inertial / (h * h) + damping[i] * v - force[i]
    magnitude = wp.abs(armature[i] * inertial / (h * h)) + wp.abs(damping[i] * v) + wp.abs(force[i])
    b = armature[i] / (h * h) + damping[i] / h
    kp, kd, effort = params[i, 0], params[i, 1], params[i, 2]
    a = kp + kd / h
    if a > type(h)(0.0):
        u = kp * ((qn[i] - qt[i]) + delta) + kd * (v - vt[i])
        r += wp.clamp(u, -effort, effort)
        magnitude += wp.abs(wp.clamp(u, -effort, effort))
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
        dn = sign * (qn[i] - bound)
        el, rl, bl = _limit_variation(dn + sign * delta, dn, params[i, 5], params[i, 6], params[i, 7], h)
        e += el
        r += sign * rl
        magnitude += wp.abs(rl)
        b += bl
    f, eps = params[i, 8], params[i, 9]
    # Scaled hypot avoids intermediate overflow for finite velocities.
    scale = wp.max(wp.abs(v), eps)
    denom = scale * wp.sqrt((v / scale) * (v / scale) + (eps / scale) * (eps / scale))
    e += h * f * v * (v / (denom + eps))
    r += f * (v / denom)
    magnitude += wp.abs(f * (v / denom))
    b += f / h * (eps / denom) * (eps / denom) / denom
    energy[i] = e
    residual[i] = r
    absolute[i] = magnitude
    tangent[i] = b


@wp.kernel
def _link_variation(
    com_step: wp.array[Any],
    rotation_step: wp.array[Any],
    r: wp.array[Any],
    rn: wp.array[Any],
    step_t: wp.array[Any],
    step_r: wp.array[Any],
    predicted_r: wp.array[Any],
    mass: wp.array[Any],
    second: wp.array[Any],
    force: wp.array[Any],
    h: Any,
    energy: wp.array[Any],
    translation_gradient: wp.array[Any],
    rotation_gradient: wp.array[Any],
    translation_absolute: wp.array[Any],
    rotation_absolute: wp.array[Any],
    rotation_tangent: wp.array[Any],
    build_tangent: bool,
):
    b = wp.tid()
    # Differences from the prediction are formed between increments.
    dt = com_step[b] - step_t[b]
    dr = (rotation_step[b] - step_r[b]) * rn[b]
    rs = dr * second[b]
    energy[b] = (mass[b] * wp.dot(dt, dt) + _inner(rs, dr)) / (type(h)(2.0) * h * h) - wp.dot(force[b], com_step[b])
    translation_gradient[b] = mass[b] * dt / (h * h) - force[b]
    translation_absolute[b] = abs3(mass[b] * dt / (h * h)) + abs3(force[b])
    a = rs * wp.transpose(r[b]) / (h * h)
    rotation_gradient[b] = type(dt)(a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1])
    rotation_absolute[b] = axial_absolute(a)
    if build_tangent:
        c = r[b] * second[b] * wp.transpose(predicted_r[b])
        local = (wp.identity(n=3, dtype=type(h)) * wp.trace(c) - (c + wp.transpose(c)) * type(h)(0.5)) / (h * h)
        vectors, values = wp.eig3(local)
        positive = type(dt)(
            wp.max(values[0], type(h)(0.0)), wp.max(values[1], type(h)(0.0)), wp.max(values[2], type(h)(0.0))
        )
        rotation_tangent[b] = vectors * wp.diag(positive) * wp.transpose(vectors)


@wp.kernel
def _reduced_variation(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    jac_offsets: wp.array[int],
    body_dof_starts: wp.array[int],
    dof_body_starts: wp.array[int],
    dof_body_ends: wp.array[int],
    rows: wp.array[int],
    columns: wp.array[int],
    mass: wp.array[Any],
    gt: wp.array[Any],
    gr: wp.array[Any],
    kr: wp.array[Any],
    joint_r: wp.array[Any],
    joint_b: wp.array[Any],
    gta: wp.array[Any],
    gra: wp.array[Any],
    joint_absolute: wp.array[Any],
    h: Any,
    residual: wp.array[Any],
    absolute: wp.array[Any],
    tangent: wp.array2d[Any],
):
    entry = wp.tid()
    i, j = rows[entry], columns[entry]
    b = type(h)(0.0)
    r = joint_r[i]
    magnitude = joint_absolute[i]
    if i == j:
        b = joint_b[i]
    for link in range(dof_body_starts[i], dof_body_ends[i]):
        base = jac_offsets[link] - body_dof_starts[link]
        ti, tj = _packed_column(jt, base + i), _packed_column(jt, base + j)
        wi, wj = _packed_column(jw, base + i), _packed_column(jw, base + j)
        b += mass[link] * wp.dot(ti, tj) / (h * h)
        b += (wp.dot(wi, kr[link] * wj) + wp.dot(wj, kr[link] * wi)) * type(h)(0.5)
        if i == j:
            r += wp.dot(ti, gt[link]) + wp.dot(wi, gr[link])
            magnitude += wp.dot(abs3(ti), gta[link]) + wp.dot(abs3(wi), gra[link])
    tangent[entry / tangent.shape[1], entry % tangent.shape[1]] = b
    if i == j:
        residual[i] = r
        absolute[i] = magnitude


@wp.kernel
def _reduced_residual(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    jac_offsets: wp.array[int],
    body_dof_starts: wp.array[int],
    dof_body_starts: wp.array[int],
    dof_body_ends: wp.array[int],
    gt: wp.array[Any],
    gr: wp.array[Any],
    joint_r: wp.array[Any],
    gta: wp.array[Any],
    gra: wp.array[Any],
    joint_absolute: wp.array[Any],
    residual: wp.array[Any],
    absolute: wp.array[Any],
):
    i = wp.tid()
    r = joint_r[i]
    magnitude = joint_absolute[i]
    for link in range(dof_body_starts[i], dof_body_ends[i]):
        entry = jac_offsets[link] + i - body_dof_starts[link]
        t, w = _packed_column(jt, entry), _packed_column(jw, entry)
        r += wp.dot(t, gt[link]) + wp.dot(w, gr[link])
        magnitude += wp.dot(abs3(t), gta[link]) + wp.dot(abs3(w), gra[link])
    residual[i] = r
    absolute[i] = magnitude


class ArticulationObjective:
    """Evaluate the frozen physical incremental objective on device in the kinematics dtype."""

    def __init__(self, kinematics, parameters):
        self.k = kinematics
        k = kinematics
        n, b, device = k.count, k.bodies, k.device
        self.params = wp.array(parameters, dtype=k.dtype, device=device)
        self.armature = wp.array(k.values["joint_armature"], dtype=k.dtype, device=device)
        self.damping = wp.array(k.values["joint_damping"], dtype=k.dtype, device=device)
        for name in (
            "qn",
            "vn",
            "qt",
            "vt",
            "joint_force",
            "joint_energy",
            "joint_r",
            "joint_absolute",
            "joint_b",
            "residual",
            "absolute",
        ):
            setattr(self, name, wp.zeros(n, dtype=k.dtype, device=device))
        # Step-start link poses and world joint axes, and the exact predicted
        # increments step_t = h v and step_r with R_p = (I + step_r) R_n.
        for name in ("tn", "predicted_t", "step_t", "force", "gt", "gr", "gta", "gra"):
            setattr(self, name, wp.zeros(b, dtype=k.vector, device=device))
        for name in ("rn", "predicted_r", "step_r", "kr"):
            setattr(self, name, wp.zeros(b, dtype=k.matrix, device=device))
        self.axes_n = wp.zeros(k.joints, dtype=k.vector, device=device)
        self.step = wp.zeros(n, dtype=k.dtype, device=device)
        self.link_energy = wp.zeros(b, dtype=k.dtype, device=device)
        shape = (n, n) if k.tree_count <= 1 else (k.matrix_rows.size, 1)
        self.tangent = wp.zeros(shape, dtype=k.dtype, device=device)

    def begin(self, qn):
        """Freeze the step-start coordinates, link poses and world joint axes at ``qn``."""
        k = self.k
        k.evaluate(qn)
        wp.copy(self.qn, qn)
        wp.copy(self.tn, k.com)
        wp.copy(self.rn, k.rotations)
        wp.copy(self.axes_n, k.world_axes)

    def evaluate(self, q, h, *, build_tangent=True, step=None):
        """Evaluate without reading derived FP32 body outputs or changing history.

        ``step`` is the exact increment ``q - qn``; without it the increment is
        derived from the rounded coordinates.
        """
        k = self.k
        h = k.timestep(h)
        if step is None:
            step = self.step
            wp.launch(_difference, k.count, [q, self.qn, step], device=k.device)
        k.evaluate(q)
        k.evaluate_step(q, step, self.rn, self.axes_n)
        wp.launch(
            _joint_variation,
            k.count,
            [
                step,
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
                self.joint_absolute,
                self.joint_b,
            ],
            device=k.device,
        )
        wp.launch(
            _link_variation,
            k.bodies,
            [
                k.com_steps,
                k.rotation_steps,
                k.rotations,
                self.rn,
                self.step_t,
                self.step_r,
                self.predicted_r,
                k.mass,
                k.second,
                self.force,
                k.dtype(h),
                self.link_energy,
                self.gt,
                self.gr,
                self.gta,
                self.gra,
                self.kr,
                build_tangent,
            ],
            device=k.device,
        )
        if not build_tangent:
            wp.launch(
                _reduced_residual,
                k.count,
                [
                    k.jt,
                    k.jw,
                    k.jac_offsets,
                    k.body_dof_starts,
                    k.dof_body_starts,
                    k.dof_body_ends,
                    self.gt,
                    self.gr,
                    self.joint_r,
                    self.gta,
                    self.gra,
                    self.joint_absolute,
                    self.residual,
                    self.absolute,
                ],
                device=k.device,
            )
            return
        wp.launch(
            _reduced_variation,
            k.matrix_rows.size,
            [
                k.jt,
                k.jw,
                k.jac_offsets,
                k.body_dof_starts,
                k.dof_body_starts,
                k.dof_body_ends,
                k.matrix_rows,
                k.matrix_columns,
                k.mass,
                self.gt,
                self.gr,
                self.kr,
                self.joint_r,
                self.joint_b,
                self.gta,
                self.gra,
                self.joint_absolute,
                k.dtype(h),
                self.residual,
                self.absolute,
                self.tangent,
            ],
            device=k.device,
        )
