# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device assembly and publication primitives for the private joint/particle path."""

from typing import Any

import warp as wp


@wp.kernel(module="unique", module_options={"fuse_fp": False})
def freeze_step_forces(
    body_f: wp.array[wp.spatial_vector],
    gravity: wp.array[wp.vec3],
    mass: wp.array[Any],
    worlds: wp.array[int],
    joint_f: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    force: wp.array[Any],
    joint_force: wp.array[Any],
    status: wp.array[int],
):
    i = wp.tid()
    if i < joint_f.shape[0]:
        if not wp.isfinite(joint_f[i]):
            wp.atomic_max(status, 0, 1)
        if not wp.isfinite(target_q[i]):
            wp.atomic_max(status, 1, 1)
        if not wp.isfinite(target_qd[i]):
            wp.atomic_max(status, 2, 1)
        joint_force[i] = type(joint_force[0])(joint_f[i])
    if i < gravity.shape[0]:
        if not wp.isfinite(gravity[i]):
            wp.atomic_max(status, 3, 1)
    if i < body_f.shape[0]:
        f = body_f[i]
        if not wp.isfinite(f) or f[3] != 0.0 or f[4] != 0.0 or f[5] != 0.0:
            wp.atomic_max(status, 3, 1)
        g = gravity[worlds[i]]
        external = type(force[0])(type(mass[0])(f[0]), type(mass[0])(f[1]), type(mass[0])(f[2]))
        acceleration = type(force[0])(type(mass[0])(g[0]), type(mass[0])(g[1]), type(mass[0])(g[2]))
        force[i] = external + mass[i] * acceleration


@wp.kernel
def combine(q: wp.array[Any], x: wp.array[Any], contact: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    value = contact[i]
    if i < q.shape[0]:
        value += q[i]
    else:
        j = i - q.shape[0]
        value += x[j / 3][j % 3]
    out[i] = value


@wp.kernel
def particle_trial(
    x: wp.array[Any],
    xn: wp.array[Any],
    mapping: wp.array[int],
    direction: wp.array[float],
    nq: int,
    alpha: Any,
    dt: Any,
    out: wp.array[Any],
    velocity: wp.array[Any],
):
    i = wp.tid()
    j = mapping[i]
    out[i] = xn[i]
    velocity[i] = type(x[0])(0.0)
    if j >= 0:
        delta = type(x[0])(
            type(dt)(direction[nq + 3 * j]),
            type(dt)(direction[nq + 3 * j + 1]),
            type(dt)(direction[nq + 3 * j + 2]),
        )
        out[i] = x[i] + alpha * delta
        velocity[i] = (out[i] - xn[i]) / dt


@wp.kernel
def particle_predict(
    xn: wp.array[Any],
    vn: wp.array[Any],
    mapping: wp.array[int],
    dt: Any,
    out: wp.array[Any],
    velocity: wp.array[Any],
):
    i = wp.tid()
    out[i] = xn[i]
    velocity[i] = type(xn[0])(0.0)
    if mapping[i] >= 0:
        out[i] = xn[i] + dt * vn[i]
        velocity[i] = (out[i] - xn[i]) / dt


@wp.kernel
def dynamic_diagonal(
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rotations: wp.array[Any],
    mass: wp.array[Any],
    second: wp.array[Any],
    armature: wp.array[Any],
    ids: wp.array[int],
    particle_mass: wp.array[float],
    dt: Any,
    out: wp.array[float],
):
    i = wp.tid()
    value = type(dt)(0.0)
    nq = armature.shape[0]
    if i < nq:
        value = armature[i]
        for body in range(mass.shape[0]):
            inertia = wp.trace(second[body]) * wp.identity(n=3, dtype=type(dt)) - second[body]
            world = rotations[body] * inertia * wp.transpose(rotations[body])
            value += mass[body] * wp.dot(jt[body, i], jt[body, i]) + wp.dot(jw[body, i], world * jw[body, i])
    else:
        value = type(dt)(particle_mass[ids[(i - nq) / 3]])
    out[i] = float(value / (dt * dt))


@wp.kernel
def sparse_posterior(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    direction: wp.array[float],
    residual: wp.array[Any],
    scale: wp.array[float],
    error: wp.array[Any],
):
    row = wp.tid()
    value = residual[row]
    for j in range(offsets[row], offsets[row + 1]):
        value += type(residual[0])(values[j]) * type(residual[0])(direction[columns[j]])
    error[row] = type(residual[0])(scale[row]) * value


@wp.kernel
def contact_forces(
    geometry: Any,
    local_residual: wp.array[Any],
    shapes: wp.array[int],
    bodies: wp.array[int],
    triangles: wp.array2d[int],
    x: wp.array[Any],
    com: wp.array[Any],
    qforce: wp.array[Any],
    pforce: wp.array[Any],
    bforce: wp.array[Any],
):
    key = wp.tid()
    shape = shapes[key % shapes.shape[0]]
    face = key / (3 * shapes.shape[0])
    slot = (key / shapes.shape[0]) % 3
    local = local_residual[key]
    world = geometry.rotation[key] * local
    point = type(x[0])(0.0)
    for a in range(3):
        weight = type(x[0][0])(1.0) / type(x[0][0])(6.0)
        if a == slot:
            weight = type(x[0][0])(2.0) / type(x[0][0])(3.0)
        particle = triangles[face, a]
        point += weight * x[particle]
        wp.atomic_add(pforce, particle, -weight * world)
    for j in range(qforce.shape[0]):
        wp.atomic_add(qforce, j, -wp.dot(geometry.jacobian_q[key, j], local))
    body = bodies[shape]
    if body >= 0:
        torque = wp.cross(point - com[body], world)
        wp.atomic_add(bforce, body, type(bforce[0])(world[0], world[1], world[2], torque[0], torque[1], torque[2]))


@wp.kernel
def _scatter(matrix: wp.array2d[float], actor: wp.array2d[float], values: wp.array[float], diagonal: wp.array[float]):
    i, j = wp.tid()
    actor[i, j] = matrix[i, j]
    values[i * matrix.shape[0] + j] = matrix[i, j]
    if i == j:
        diagonal[i] = matrix[i, i]


@wp.kernel
def _coordinate_trial(
    q: wp.array[Any],
    direction: wp.array[float],
    alpha: Any,
    qn: wp.array[Any],
    h: Any,
    out: wp.array[Any],
    vout: wp.array[Any],
):
    i = wp.tid()
    out[i] = q[i] + alpha * type(h)(direction[i])
    vout[i] = (out[i] - qn[i]) / h


@wp.kernel
def _coordinate_predictor(
    q: wp.array[Any],
    v: wp.array[Any],
    h: Any,
    out: wp.array[Any],
    vout: wp.array[Any],
):
    i = wp.tid()
    out[i] = q[i] + h * v[i]
    vout[i] = (out[i] - q[i]) / h
