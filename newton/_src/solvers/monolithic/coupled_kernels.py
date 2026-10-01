# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device assembly and publication primitives for the private joint/particle path."""

from typing import Any

import warp as wp


@wp.kernel
def combine(q: wp.array[Any], x: wp.array[Any], contact: wp.array[Any], q_start: int, x_start: int, out: wp.array[Any]):
    i = wp.tid()
    value = contact[i]
    if i >= q_start and i < q_start + q.shape[0]:
        value += q[i - q_start]
    else:
        j = i - x_start
        value += x[j / 3][j % 3]
    out[i] = value


@wp.kernel
def particle_trial(
    x: wp.array[Any],
    xn: wp.array[Any],
    mapping: wp.array[int],
    direction: wp.array[float],
    x_start: int,
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
            type(dt)(direction[x_start + 3 * j]),
            type(dt)(direction[x_start + 3 * j + 1]),
            type(dt)(direction[x_start + 3 * j + 2]),
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
    q_start: int,
    x_start: int,
    out: wp.array[float],
):
    i = wp.tid()
    value = type(dt)(0.0)
    nq = armature.shape[0]
    if i >= q_start and i < q_start + nq:
        local = i - q_start
        value = armature[local]
        for body in range(mass.shape[0]):
            inertia = wp.trace(second[body]) * wp.identity(n=3, dtype=type(dt)) - second[body]
            world = rotations[body] * inertia * wp.transpose(rotations[body])
            value += mass[body] * wp.dot(jt[body, local], jt[body, local]) + wp.dot(
                jw[body, local], world * jw[body, local]
            )
    else:
        value = type(dt)(particle_mass[ids[(i - x_start) / 3]])
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
    dof_map: wp.array[int],
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
    for j in range(dof_map.shape[0]):
        wp.atomic_add(qforce, dof_map[j], -wp.dot(geometry.jacobian_q[key, j], local))
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
    q_start: int,
    alpha: Any,
    qn: wp.array[Any],
    h: Any,
    out: wp.array[Any],
    vout: wp.array[Any],
):
    i = wp.tid()
    out[i] = q[i] + alpha * type(h)(direction[q_start + i])
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
