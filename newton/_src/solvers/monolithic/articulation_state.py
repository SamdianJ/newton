# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private, explicitly owned native-precision articulation integration history."""

from typing import Any

import numpy as np
import warp as wp

from ...sim import JointType, Model


@wp.func
def _skew(v: Any):
    return wp.matrix(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0, shape=(3, 3), dtype=type(v[0]))


@wp.func
def _store_symmetric(a: Any, values: wp.array2d[Any], i: int):
    values[i, 0] = a[0, 0]
    values[i, 1] = a[1, 1]
    values[i, 2] = a[2, 2]
    values[i, 3] = (a[0, 1] + a[1, 0]) * type(a[0, 0])(0.5)
    values[i, 4] = (a[0, 2] + a[2, 0]) * type(a[0, 0])(0.5)
    values[i, 5] = (a[1, 2] + a[2, 1]) * type(a[0, 0])(0.5)


@wp.func
def _load_symmetric(values: wp.array2d[Any], i: int):
    return wp.matrix(
        values[i, 0],
        values[i, 3],
        values[i, 4],
        values[i, 3],
        values[i, 1],
        values[i, 5],
        values[i, 4],
        values[i, 5],
        values[i, 2],
        shape=(3, 3),
        dtype=type(values[0, 0]),
    )


@wp.kernel
def _initial_link_rates(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    qd: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
):
    b = wp.tid()
    linear, angular = type(v[0])(0.0), type(v[0])(0.0)
    for i in range(qd.shape[0]):
        linear += jt[b, i] * qd[i]
        angular += jw[b, i] * qd[i]
    v[b] = linear
    omega[b] = angular


@wp.kernel
def _initial_joint_rates(
    types: wp.array[int],
    starts: wp.array[int],
    axes: wp.array[Any],
    qd: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
):
    j = wp.tid()
    v[j] = type(v[0])(0.0)
    omega[j] = type(v[0])(0.0)
    if types[j] == JointType.REVOLUTE:
        omega[j] = axes[starts[j]] * qd[starts[j]]
    elif types[j] == JointType.PRISMATIC:
        v[j] = axes[starts[j]] * qd[starts[j]]


@wp.kernel
def _initial_symmetric(omega: wp.array[Any], h: Any, sym: wp.array2d[Any]):
    i = wp.tid()
    cross = _skew(omega[i])
    value = h * (cross * cross) / (type(h)(1.0) + wp.sqrt(type(h)(1.0) - h * h * wp.dot(omega[i], omega[i])))
    _store_symmetric(value, sym, i)


@wp.kernel
def _recover_history(
    old_t: wp.array[Any],
    old_r: wp.array[Any],
    new_t: wp.array[Any],
    new_r: wp.array[Any],
    h: Any,
    v: wp.array[Any],
    omega: wp.array[Any],
    sym: wp.array2d[Any],
):
    i = wp.tid()
    v[i] = (new_t[i] - old_t[i]) / h
    a = (new_r[i] - old_r[i]) * wp.transpose(new_r[i]) / h
    omega[i] = type(v[0])(a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]) * type(h)(0.5)
    _store_symmetric(a, sym, i)


@wp.kernel
def _predict(
    t: wp.array[Any],
    r: wp.array[Any],
    v: wp.array[Any],
    omega: wp.array[Any],
    sym: wp.array2d[Any],
    h: Any,
    predicted_t: wp.array[Any],
    predicted_r: wp.array[Any],
):
    i = wp.tid()
    predicted_t[i] = t[i] + h * v[i]
    predicted_r[i] = r[i] + h * (_skew(omega[i]) + _load_symmetric(sym, i)) * r[i]


class ArticulationState:
    """Allocate and validate the complete version-one private State schema."""

    namespace = "_monolithic_articulation"
    version = 1

    def __init__(self, kinematics):
        self.kinematics = kinematics
        self.dtype, self.vector = kinematics.dtype, kinematics.vector
        self.model, self.device = kinematics.model, kinematics.device
        n, j, b = kinematics.count, kinematics.joints, kinematics.bodies
        self.schema = {
            "joint_v": ((j,), self.vector),
            "joint_omega": ((j,), self.vector),
            "joint_vsym": ((j, 6), self.dtype),
            "link_v_com": ((b,), self.vector),
            "link_omega": ((b,), self.vector),
            "link_vsym": ((b, 6), self.dtype),
            "bound_q": ((n,), self.dtype),
            "bound_qd": ((n,), self.dtype),
            "last_dt": ((1,), self.dtype),
            "schema_version": ((1,), wp.int32),
            "valid": ((1,), wp.int32),
            "config_digest": ((32,), wp.uint8),
        }

    def create(self, *, dt, source=None):
        """Initialize from coordinates/rates, never silently inherit old history."""
        state = self.model.state()
        source = self.model if source is None else source
        self._copy_coordinates(state, source)
        if getattr(source, "body_f", None) is not None:
            wp.copy(state.body_f, source.body_f)
        history = Model.AttributeNamespace(self.namespace)
        for name, (shape, dtype) in self.schema.items():
            setattr(history, name, wp.zeros(shape, dtype=dtype, device=self.device))
        setattr(state, self.namespace, history)
        self.reset(state, dt=dt)
        return state

    def copy(self, source):
        """Allocate an exact continuation without applying initialization rules."""
        self.validate(source)
        state = self.model.state()
        self._copy_coordinates(state, source)
        history = Model.AttributeNamespace(self.namespace)
        for name in self.schema:
            setattr(history, name, wp.clone(getattr(getattr(source, self.namespace), name)))
        setattr(state, self.namespace, history)
        state.assign(source)
        return state

    def _copy_coordinates(self, state, source):
        for name in ("joint_q", "joint_qd"):
            setattr(state, name, wp.array(getattr(source, name).numpy(), dtype=self.dtype, device=self.device))

    def _particle_schema(self):
        return dict.fromkeys(("particle_q", "particle_qd", "particle_f"), ((0,), wp.vec3))

    def save_checkpoint(self, state, path):
        """Save standard arrays and the complete owned history without pickles."""
        history = self.validate(state)
        arrays = {name: value.numpy() for name, value in vars(state).items() if isinstance(value, wp.array)}
        arrays.update({"history." + name: getattr(history, name).numpy() for name in self.schema})
        np.savez(path, **arrays)

    def load_checkpoint(self, state, path):
        """Validate a complete checkpoint in scratch before replacing its target."""
        scratch = self.copy(state)
        arrays = {name: value for name, value in vars(scratch).items() if isinstance(value, wp.array)}
        arrays.update({"history." + name: getattr(getattr(scratch, self.namespace), name) for name in self.schema})
        with np.load(path, allow_pickle=False) as checkpoint:
            if set(checkpoint.files) != arrays.keys():
                raise ValueError("Checkpoint must contain complete articulation state and history")
            for name, array in arrays.items():
                value, expected = checkpoint[name], array.numpy()
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f"Invalid checkpoint field: {name}")
                array.assign(value)
        self.validate(scratch)
        state.assign(scratch)

    def validate(self, state, *, bound=True):
        """Reject incomplete, incompatible, nonfinite or externally edited states."""
        k = self.kinematics
        history = getattr(state, self.namespace, None)
        if not isinstance(history, Model.AttributeNamespace):
            raise ValueError("Monolithic requires a complete articulation state")
        actual = {name for name, value in vars(history).items() if isinstance(value, wp.array)}
        if actual != self.schema.keys():
            raise ValueError("Invalid articulation history schema")
        fields = {name: (getattr(history, name), shape, dtype) for name, (shape, dtype) in self.schema.items()}
        for name in ("joint_q", "joint_qd"):
            fields[name] = (getattr(state, name, None), (k.count,), self.dtype)
        standard = {"joint_q", "joint_qd", "body_q", "body_qd", "body_f", "particle_q", "particle_qd", "particle_f"}
        for name, dtype in (("body_q", wp.transform), ("body_qd", wp.spatial_vector), ("body_f", wp.spatial_vector)):
            fields[name] = (getattr(state, name, None), (k.bodies,), dtype)
        for name, (shape, dtype) in self._particle_schema().items():
            value = getattr(state, name, None)
            if value is not None:
                fields[name] = (value, shape, dtype)
        for name, value in vars(state).items():
            if isinstance(value, wp.array) and name not in standard:
                raise ValueError(f"Unsupported articulation state array: {name}")
            if isinstance(value, Model.AttributeNamespace) and name != self.namespace:
                raise ValueError(f"Unsupported articulation state namespace: {name}")
        for name, (array, shape, dtype) in fields.items():
            if not isinstance(array, wp.array) or (array.shape, array.dtype, array.device) != (
                shape,
                dtype,
                self.device,
            ):
                raise ValueError(f"Invalid articulation state array: {name}")
            if not np.isfinite(array.numpy()).all():
                raise ValueError(f"Nonfinite articulation state: {name}")
        if bound:
            if history.schema_version.numpy()[0] != self.version or history.valid.numpy()[0] != 1:
                raise ValueError("Invalid articulation state version/validity")
            if history.last_dt.numpy()[0] <= 0 or history.config_digest.numpy().tobytes() != k.digest:
                raise ValueError("Incompatible articulation state configuration")
            for name in ("q", "qd"):
                if not np.array_equal(
                    getattr(state, "joint_" + name).numpy(), getattr(history, "bound_" + name).numpy()
                ):
                    raise ValueError("Coordinates/rates changed; explicitly reset articulation history")
        k.evaluate(state.joint_q)
        k.validate_inertia()
        return history

    def reset(self, state, *, dt):
        """Explicitly initialize discrete velocities from physical q/qd and dt."""
        k = self.kinematics
        dt = k.timestep(dt)
        k.validate_identity()
        history = self.validate(state, bound=False)
        # Validate before modifying any published state, including on reset failure.
        omega = np.einsum("bni,n->bi", k.jw.numpy(), state.joint_qd.numpy())
        if (
            not np.isfinite(omega).all()
            or np.any(dt * np.linalg.norm(omega, axis=1) >= 1)
            or np.any(
                dt * abs(state.joint_qd.numpy()[k.values["joint_type"][k.dof_joint.numpy()] == JointType.REVOLUTE]) >= 1
            )
        ):
            raise ValueError("Initial angular velocity requires dt * |omega| < 1")
        wp.launch(
            _initial_link_rates,
            k.bodies,
            [k.jt, k.jw, state.joint_qd, history.link_v_com, history.link_omega],
            device=self.device,
        )
        wp.launch(
            _initial_joint_rates,
            k.joints,
            [k.types, k.starts, k.axes, state.joint_qd, history.joint_v, history.joint_omega],
            device=self.device,
        )
        for prefix, count in (("joint", k.joints), ("link", k.bodies)):
            wp.launch(
                _initial_symmetric,
                count,
                [getattr(history, prefix + "_omega"), self.dtype(dt), getattr(history, prefix + "_vsym")],
                device=self.device,
            )
        k.evaluate(state.joint_q, state=state)
        self.bind(state, dt=dt)

    def bind(self, state, *, dt):
        """Bind metadata after complete initialization or physical-step recovery."""
        history = getattr(state, self.namespace)
        wp.copy(history.bound_q, state.joint_q)
        wp.copy(history.bound_qd, state.joint_qd)
        history.last_dt.fill_(dt)
        history.schema_version.fill_(self.version)
        history.valid.fill_(1)
        history.config_digest.assign(np.frombuffer(self.kinematics.digest, dtype=np.uint8))
