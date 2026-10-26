# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device assembly and publication primitives for the private joint/particle path."""

from typing import Any

import warp as wp

from .articulation import _packed_column


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
def combine(
    q: wp.array[Any], x: wp.array[Any], contact: wp.array[Any], q_local: wp.array[int], x_start: int, out: wp.array[Any]
):
    i = wp.tid()
    value = contact[i]
    if q_local[i] >= 0:
        value += q[q_local[i]]
    elif i >= x_start and i < x_start + 3 * x.shape[0]:
        j = i - x_start
        value += x[j / 3][j % 3]
    out[i] = value


@wp.kernel
def particle_trial(
    accepted: wp.array[Any],
    xn: wp.array[Any],
    mapping: wp.array[int],
    direction: wp.array[Any],
    x_start: int,
    alpha: Any,
    dt: Any,
    out: wp.array[Any],
    velocity: wp.array[Any],
    displacement: wp.array[Any],
):
    """Advance the step displacement of dynamic tet particles; positions are formed once from it."""
    i = wp.tid()
    j = mapping[i]
    out[i] = xn[i]
    velocity[i] = type(xn[0])(0.0)
    displacement[i] = type(xn[0])(0.0)
    if j >= 0:
        delta = type(xn[0])(
            type(dt)(direction[x_start + 3 * j]),
            type(dt)(direction[x_start + 3 * j + 1]),
            type(dt)(direction[x_start + 3 * j + 2]),
        )
        step = accepted[i] + alpha * delta
        displacement[i] = step
        out[i] = xn[i] + step
        velocity[i] = step / dt


@wp.kernel
def step_displacement(x: wp.array[Any], xn: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = x[i] - xn[i]


@wp.kernel
def particle_predict(
    xn: wp.array[Any],
    vn: wp.array[Any],
    mapping: wp.array[int],
    dt: Any,
    out: wp.array[Any],
    velocity: wp.array[Any],
    displacement: wp.array[Any],
):
    """Predict the step displacement dt v_n of dynamic tet particles."""
    i = wp.tid()
    out[i] = xn[i]
    velocity[i] = type(xn[0])(0.0)
    displacement[i] = type(xn[0])(0.0)
    if mapping[i] >= 0:
        step = dt * vn[i]
        displacement[i] = step
        out[i] = xn[i] + step
        velocity[i] = step / dt


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
    q_local: wp.array[int],
    jac_offsets: wp.array[int],
    body_dof_starts: wp.array[int],
    dof_body_starts: wp.array[int],
    dof_body_ends: wp.array[int],
    x_start: int,
    out: wp.array[Any],
):
    i = wp.tid()
    value = type(dt)(0.0)
    local = q_local[i]
    if local >= 0:
        value = armature[local]
        for body in range(dof_body_starts[local], dof_body_ends[local]):
            inertia = wp.trace(second[body]) * wp.identity(n=3, dtype=type(dt)) - second[body]
            world = rotations[body] * inertia * wp.transpose(rotations[body])
            entry = jac_offsets[body] + local - body_dof_starts[body]
            t, w = _packed_column(jt, entry), _packed_column(jw, entry)
            value += mass[body] * wp.dot(t, t) + wp.dot(w, world * w)
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
    magnitude: wp.array[Any],
    material: wp.array[Any],
    material_magnitude: wp.array[Any],
):
    row = wp.tid()
    # ``material`` carries the element tangents applied outside the assembly.
    value = residual[row] + material[row]
    for j in range(offsets[row], offsets[row + 1]):
        value += values[j] * direction[columns[j]]
    absolute = material_magnitude[row]
    for j in range(offsets[row], offsets[row + 1]):
        absolute += wp.abs(values[j] * direction[columns[j]])
    original[row] = scale[row] * value
    term = residual.dtype(0.0)
    if regularization[0] != term:
        term = regularization[0] * (direction[row] / scale[row])
    regularization_term[row] = term
    error[row] = original[row] + term
    # |A| |y| of the scaled operator, whose rounding bounds the attainable error.
    magnitude[row] = scale[row] * absolute + wp.abs(term)


@wp.kernel
def _scatter(
    matrix: wp.array[Any],
    actor: wp.array[Any],
    values: wp.array[Any],
    rows: wp.array[int],
    columns: wp.array[int],
    diagonal: wp.array[Any],
):
    entry = wp.tid()
    actor[entry] = matrix[entry]
    values[entry] = matrix[entry]
    if rows[entry] == columns[entry]:
        diagonal[rows[entry]] = matrix[entry]


@wp.kernel
def _coordinate_trial(
    accepted_step: wp.array[Any],
    direction: wp.array[Any],
    rows: wp.array[int],
    alpha: Any,
    qn: wp.array[Any],
    h: Any,
    coords: wp.array[int],
    dofs: wp.array[int],
    out: wp.array[Any],
    vout: wp.array[Any],
    step: wp.array[Any],
):
    i = wp.tid()
    # Advance the exact step; coordinates and rates are formed from it once.
    step[i] = accepted_step[i] + alpha * type(h)(direction[rows[i]])
    out[coords[i]] = qn[i] + step[i]
    vout[dofs[i]] = step[i] / h


@wp.kernel
def _coordinate_predictor(
    q: wp.array[Any],
    v: wp.array[Any],
    h: Any,
    coords: wp.array[int],
    dofs: wp.array[int],
    out: wp.array[Any],
    vout: wp.array[Any],
    step: wp.array[Any],
):
    i = wp.tid()
    step[i] = h * v[i]
    out[coords[i]] = q[i] + step[i]
    vout[dofs[i]] = step[i] / h
