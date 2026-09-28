# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device assembly and publication primitives for the private joint/particle path."""

import warp as wp

from .contact_variational import _Geometry


@wp.kernel
def combine(q: wp.array[wp.float64], x: wp.array[wp.vec3d], contact: wp.array[wp.float64], out: wp.array[wp.float64]):
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
    x: wp.array[wp.vec3d],
    xn: wp.array[wp.vec3d],
    mapping: wp.array[int],
    direction: wp.array[float],
    nq: int,
    alpha: wp.float64,
    dt: wp.float64,
    out: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
):
    i = wp.tid()
    j = mapping[i]
    out[i] = xn[i]
    velocity[i] = wp.vec3d(0.0)
    if j >= 0:
        delta = wp.vec3d(
            wp.float64(direction[nq + 3 * j]),
            wp.float64(direction[nq + 3 * j + 1]),
            wp.float64(direction[nq + 3 * j + 2]),
        )
        out[i] = x[i] + alpha * delta
        velocity[i] = (out[i] - xn[i]) / dt


@wp.kernel
def particle_predict(
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    mapping: wp.array[int],
    dt: wp.float64,
    out: wp.array[wp.vec3d],
    velocity: wp.array[wp.vec3d],
):
    i = wp.tid()
    out[i] = xn[i]
    velocity[i] = wp.vec3d(0.0)
    if mapping[i] >= 0:
        out[i] = xn[i] + dt * vn[i]
        velocity[i] = (out[i] - xn[i]) / dt


@wp.kernel
def dynamic_diagonal(
    jt: wp.array2d[wp.vec3d],
    jw: wp.array2d[wp.vec3d],
    rotations: wp.array[wp.mat33d],
    mass: wp.array[wp.float64],
    second: wp.array[wp.mat33d],
    armature: wp.array[wp.float64],
    ids: wp.array[int],
    particle_mass: wp.array[float],
    dt: wp.float64,
    out: wp.array[float],
):
    i = wp.tid()
    value = wp.float64(0.0)
    nq = armature.shape[0]
    if i < nq:
        value = armature[i]
        for body in range(mass.shape[0]):
            inertia = wp.trace(second[body]) * wp.identity(n=3, dtype=wp.float64) - second[body]
            world = rotations[body] * inertia * wp.transpose(rotations[body])
            value += mass[body] * wp.dot(jt[body, i], jt[body, i]) + wp.dot(jw[body, i], world * jw[body, i])
    else:
        value = wp.float64(particle_mass[ids[(i - nq) / 3]])
    out[i] = float(value / (dt * dt))


@wp.kernel
def sparse_posterior(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    direction: wp.array[float],
    residual: wp.array[wp.float64],
    scale: wp.array[float],
    error: wp.array[wp.float64],
):
    row = wp.tid()
    value = residual[row]
    for j in range(offsets[row], offsets[row + 1]):
        value += wp.float64(values[j]) * wp.float64(direction[columns[j]])
    error[row] = wp.float64(scale[row]) * value


@wp.kernel
def contact_forces(
    geometry: _Geometry,
    local_residual: wp.array[wp.vec3d],
    shapes: wp.array[int],
    bodies: wp.array[int],
    triangles: wp.array2d[int],
    x: wp.array[wp.vec3d],
    com: wp.array[wp.vec3d],
    qforce: wp.array[wp.float64],
    pforce: wp.array[wp.vec3d],
    bforce: wp.array[wp.spatial_vectord],
):
    key = wp.tid()
    shape = shapes[key % shapes.shape[0]]
    face = key / (3 * shapes.shape[0])
    slot = (key / shapes.shape[0]) % 3
    local = local_residual[key]
    world = geometry.rotation[key] * local
    point = wp.vec3d(0.0)
    for a in range(3):
        weight = wp.float64(1.0) / wp.float64(6.0)
        if a == slot:
            weight = wp.float64(2.0) / wp.float64(3.0)
        particle = triangles[face, a]
        point += weight * x[particle]
        wp.atomic_add(pforce, particle, -weight * world)
    for j in range(qforce.shape[0]):
        wp.atomic_add(qforce, j, -wp.dot(geometry.jacobian_q[key, j], local))
    body = bodies[shape]
    if body >= 0:
        torque = wp.cross(point - com[body], world)
        wp.atomic_add(bforce, body, wp.spatial_vectord(world[0], world[1], world[2], torque[0], torque[1], torque[2]))


@wp.kernel
def _scatter(matrix: wp.array2d[float], actor: wp.array2d[float], values: wp.array[float], diagonal: wp.array[float]):
    i, j = wp.tid()
    actor[i, j] = matrix[i, j]
    values[i * matrix.shape[0] + j] = matrix[i, j]
    if i == j:
        diagonal[i] = matrix[i, i]


@wp.kernel
def _coordinate_trial(
    q: wp.array[wp.float64],
    direction: wp.array[float],
    alpha: wp.float64,
    qn: wp.array[wp.float64],
    h: wp.float64,
    out: wp.array[wp.float64],
    vout: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = q[i] + alpha * wp.float64(direction[i])
    vout[i] = (out[i] - qn[i]) / h


@wp.kernel
def _coordinate_predictor(
    q: wp.array[wp.float64],
    v: wp.array[wp.float64],
    h: wp.float64,
    out: wp.array[wp.float64],
    vout: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = q[i] + h * v[i]
    vout[i] = (out[i] - q[i]) / h
