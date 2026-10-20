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
    bodies: wp.array[int],
    dofs: wp.array[int],
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
    if i < target_q.shape[0]:
        if not wp.isfinite(target_q[i]):
            wp.atomic_max(status, 1, 1)
    if i < target_qd.shape[0]:
        if not wp.isfinite(target_qd[i]):
            wp.atomic_max(status, 2, 1)
    if i < dofs.shape[0]:
        joint_force[i] = type(joint_force[0])(joint_f[dofs[i]])
    if i < gravity.shape[0]:
        if not wp.isfinite(gravity[i]):
            wp.atomic_max(status, 3, 1)
    if i < bodies.shape[0]:
        body = bodies[i]
        f = body_f[body]
        if not wp.isfinite(f) or f[3] != 0.0 or f[4] != 0.0 or f[5] != 0.0:
            wp.atomic_max(status, 3, 1)
        g = gravity[worlds[body]]
        external = type(force[0])(type(mass[0])(f[0]), type(mass[0])(f[1]), type(mass[0])(f[2]))
        acceleration = type(force[0])(type(mass[0])(g[0]), type(mass[0])(g[1]), type(mass[0])(g[2]))
        force[i] = external + mass[i] * acceleration


@wp.kernel
def combine(q: wp.array[Any], x: wp.array[Any], contact: wp.array[Any], q_start: int, x_start: int, out: wp.array[Any]):
    i = wp.tid()
    value = contact[i]
    if i >= q_start and i < q_start + q.shape[0]:
        value += q[i - q_start]
    elif i >= x_start and i < x_start + 3 * x.shape[0]:
        j = i - x_start
        value += x[j / 3][j % 3]
    out[i] = value


@wp.kernel
def particle_trial(
    x: wp.array[Any],
    xn: wp.array[Any],
    mapping: wp.array[int],
    direction: wp.array[Any],
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
    out: wp.array[Any],
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
    elif i >= x_start and i < x_start + 3 * ids.shape[0]:
        value = type(dt)(particle_mass[ids[(i - x_start) / 3]])
    else:
        return
    out[i] = value / (dt * dt)


@wp.kernel
def sparse_posterior(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    direction: wp.array[Any],
    residual: wp.array[Any],
    scale: wp.array[Any],
    regularization: wp.array[Any],
    original: wp.array[Any],
    regularization_term: wp.array[Any],
    error: wp.array[Any],
):
    row = wp.tid()
    value = residual[row]
    for j in range(offsets[row], offsets[row + 1]):
        value += values[j] * direction[columns[j]]
    original[row] = scale[row] * value
    term = residual.dtype(0.0)
    if regularization[0] != term:
        term = regularization[0] * (direction[row] / scale[row])
    regularization_term[row] = term
    error[row] = original[row] + term


@wp.kernel
def _scatter(matrix: wp.array2d[Any], actor: wp.array2d[Any], values: wp.array[Any], diagonal: wp.array[Any]):
    i, j = wp.tid()
    actor[i, j] = matrix[i, j]
    values[i * matrix.shape[0] + j] = matrix[i, j]
    if i == j:
        diagonal[i] = matrix[i, i]


@wp.kernel
def _coordinate_trial(
    q: wp.array[Any],
    direction: wp.array[Any],
    q_start: int,
    alpha: Any,
    qn: wp.array[Any],
    h: Any,
    coords: wp.array[int],
    dofs: wp.array[int],
    out: wp.array[Any],
    vout: wp.array[Any],
):
    i = wp.tid()
    out[coords[i]] = q[coords[i]] + alpha * type(h)(direction[q_start + i])
    vout[dofs[i]] = (out[coords[i]] - qn[i]) / h


@wp.kernel
def _coordinate_predictor(
    q: wp.array[Any],
    v: wp.array[Any],
    h: Any,
    coords: wp.array[int],
    dofs: wp.array[int],
    out: wp.array[Any],
    vout: wp.array[Any],
):
    i = wp.tid()
    out[coords[i]] = q[i] + h * v[i]
    vout[dofs[i]] = (out[coords[i]] - q[i]) / h
