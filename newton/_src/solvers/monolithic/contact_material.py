# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic circular-rod surface kinematics for the shared bounded contact producer."""

from typing import Any

import numpy as np
import warp as wp

from .cable import _transport, _transport_differential
from .contact import _failure, _reset_flags
from .contact_actor import _center, _point_column
from .contact_batch import _project_contact_column


@wp.kernel
def _surface(
    edges: wp.array[wp.vec3i],
    coordinates: wp.array[Any],
    normals: wp.array[Any],
    x: wp.array[Any],
    anchor_x: wp.array[Any],
    anchor_d: wp.array[Any],
    theta: wp.array[Any],
    points: wp.array[Any],
    directions: wp.array[Any],
    jacobian: wp.array2d[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    a, b, angle = edges[i][0], edges[i][1], edges[i][2]
    e = x[b] - x[a]
    length = wp.length(e)
    old = wp.normalize(anchor_x[b] - anchor_x[a])
    if not wp.isfinite(length) or length <= type(length)(0.0):
        _failure(flags, i, 2)
        return
    t = e / length
    if type(length)(1.0) + wp.dot(old, t) <= type(length)(1e-10):
        _failure(flags, i, 2)
        return
    d = _transport(old, t, anchor_d[angle])
    c, s = wp.cos(theta[angle]), wp.sin(theta[angle])
    u = coordinates[i][1] * c - coordinates[i][2] * s
    v = coordinates[i][1] * s + coordinates[i][2] * c
    perpendicular = wp.cross(t, d)
    axial = coordinates[i][0]
    zero = type(t)(0.0)
    points[i] = (type(length)(1.0) - axial) * x[a] + axial * x[b] + u * d + v * perpendicular
    directions[i] = (normals[i][1] * c - normals[i][2] * s) * d + (
        normals[i][1] * s + normals[i][2] * c
    ) * perpendicular
    if normals[i][0] != type(length)(0.0):
        directions[i] = normals[i][0] * t
    for j in range(6):
        de = type(t)(0.0)
        sign = type(length)(1.0)
        bary = axial
        if j < 3:
            sign = type(length)(-1.0)
            bary = type(length)(1.0) - axial
        de[j % 3] = sign
        dt = (de - wp.dot(t, de) * t) / length
        dd = _transport_differential(old, t, anchor_d[angle], zero, dt, zero)
        column = u * dd + v * (wp.cross(dt, d) + wp.cross(t, dd))
        column[j % 3] += bary
        jacobian[i, j] = column
    jacobian[i, 6] = -v * d + u * perpendicular


@wp.kernel
def _copy_surface(samples: Any, points: wp.array[Any], normals: wp.array[Any]):
    i = wp.tid()
    if samples.sources[i] == -3:
        samples.points[i] = points[samples.faces[i]]
        samples.normals[i] = normals[samples.faces[i]]


@wp.kernel
def _project_surface(
    samples: Any,
    support: Any,
    geometry: Any,
    points: wp.array[Any],
    bodies: wp.array[int],
    edges: wp.array[wp.vec3i],
    theta_rows: wp.array[int],
    jacobian: wp.array2d[Any],
    com: wp.array[Any],
    jt: wp.array2d[Any],
    jw: wp.array2d[Any],
    rigid_com: wp.array[Any],
    normal: wp.array[Any],
    tangent: wp.array[Any],
    weights: wp.array2d[Any],
    local_r: wp.array[Any],
    candidates: Any,
    flags: wp.array[int],
    particle_count: int,
    body_count: int,
):
    i = wp.tid()
    if i >= candidates.count[0] or samples.sources[i] != -3 or flags[0] != 0:
        return
    face, target = samples.faces[i], samples.targets[i]
    point, rotation, n = points[i], geometry.rotation[i], normal[i]
    axis = type(point)(1.0, 0.0, 0.0)
    if wp.abs(n[1]) < wp.abs(n[0]) and wp.abs(n[1]) <= wp.abs(n[2]):
        axis = type(point)(0.0, 1.0, 0.0)
    elif wp.abs(n[2]) < wp.abs(n[0]):
        axis = type(point)(0.0, 0.0, 1.0)
    t = tangent[i]
    if weights[i, 1] > type(weights[0, 0])(0.0) and wp.length(t) == type(weights[0, 0])(0.0):
        t = wp.normalize(wp.cross(n, axis))
    transverse = wp.cross(n, t)
    candidates.weights[i, 0] = float(weights[i, 0])
    candidates.weights[i, 1] = float(weights[i, 2])
    candidates.weights[i, 2] = float(weights[i, 1])
    r, k = local_r[i], int(0)
    world = rotation * r
    for j in range(7):
        row = theta_rows[edges[face][2]]
        force_row = 3 * particle_count + 6 * body_count + edges[face][2]
        if j < 6:
            particle = edges[face][j // 3]
            row = support.particles[particle]
            if row >= 0:
                row += j % 3
            force_row = 3 * particle + j % 3
        column = wp.transpose(rotation) * jacobian[face, j]
        if row >= 0:
            _project_contact_column(
                candidates,
                i,
                k,
                row,
                column,
                r,
                geometry.gradient[i],
                t,
                transverse,
                False,
                type(point[0])(0.0),
                0,
                rotation,
            )
            k += 1
        candidates.force_rows[i, j] = force_row
        candidates.forces[i, j] = -wp.dot(jacobian[face, j], world)
    body = bodies[target]
    if body >= 0:
        for index in range(support.offsets[body], support.offsets[body + 1]):
            column = -wp.transpose(rotation) * _point_column(
                body, support.local[index], point, support.kinds, support.indices, com, jt, jw, rigid_com
            )
            _project_contact_column(
                candidates,
                i,
                k,
                support.columns[index],
                column,
                r,
                geometry.gradient[i],
                t,
                transverse,
                False,
                type(point[0])(0.0),
                0,
                rotation,
            )
            k += 1
    candidates.support_counts[i] = k
    f = int(7)
    for end in range(2):
        force = -world
        if end == 1:
            force = world
        torque = wp.cross(point, force)
        for j in range(3):
            candidates.pairs[i, end, j] = force[j]
            candidates.pairs[i, end, j + 3] = torque[j]
    if body >= 0:
        torque = wp.cross(point - _center(body, support.kinds, support.indices, com, rigid_com), world)
        for j in range(6):
            candidates.force_rows[i, f] = 3 * particle_count + 6 * body + j
            value = world[j % 3]
            if j >= 3:
                value = torque[j % 3]
            candidates.forces[i, f] = value
            f += 1
    candidates.force_counts[i] = f


class _MaterialSurface:
    """Own cable surface points and exact chart derivatives shared by all target queries."""

    def __init__(self, batch):
        c, a = batch.contact, batch.contact.actor_samples
        self.batch = batch
        self.edges = wp.array(a.cable_edges, dtype=wp.vec3i, device=batch.device)
        self.coordinates = wp.array(a.cable_coordinates, dtype=batch.vector, device=batch.device)
        self.normals = wp.array(a.cable_normals, dtype=batch.vector, device=batch.device)
        self.theta_rows = wp.array(np.asarray(a.actors.maps["theta_to_global"]), dtype=int, device=batch.device)
        self.points = wp.zeros(len(a.cable_edges), dtype=batch.vector, device=batch.device)
        self.directions = wp.zeros_like(self.points)
        self.jacobian = wp.zeros((len(a.cable_edges), 7), dtype=batch.vector, device=batch.device)
        self.anchor_x = wp.zeros(c.model.particle_count, dtype=batch.vector, device=batch.device)
        self.anchor_d = wp.zeros(c._materials.theta_count, dtype=batch.vector, device=batch.device)
        self.flags = wp.zeros(2, dtype=int, device=batch.device)

    def update(self, state, *, begin=False):
        if not self.edges.size:
            return
        if begin:
            wp.copy(self.anchor_x, state.particle_q)
            wp.copy(self.anchor_d, state._monolithic_coupled.cable_ref_director)
        wp.launch(_reset_flags, 1, [self.flags], device=self.batch.device)
        wp.launch(
            _surface,
            self.edges.size,
            [
                self.edges,
                self.coordinates,
                self.normals,
                state.particle_q,
                self.anchor_x,
                self.anchor_d,
                state._monolithic_coupled.cable_theta,
                self.points,
                self.directions,
                self.jacobian,
                self.flags,
            ],
            device=self.batch.device,
        )
        if self.flags.numpy()[0]:
            raise ValueError("Invalid cable contact frame")
        wp.launch(
            _copy_surface,
            self.batch.contact.count,
            [self.batch.all_samples, self.points, self.directions],
            device=self.batch.device,
        )

    def project(self, state):
        b, c = self.batch, self.batch.contact
        wp.launch(
            _project_surface,
            b.batch_capacity,
            [
                b.samples,
                b.support,
                b.current,
                b.points,
                c.bodies,
                self.edges,
                self.theta_rows,
                self.jacobian,
                state[2],
                state[3],
                state[4],
                state[5],
                b.normal,
                b.tangent,
                b.weights,
                b.local_r,
                b.pool.candidates,
                b._flags,
                c.model.particle_count,
                c.model.body_count,
            ],
            device=b.device,
        )

    def buffers(self):
        return tuple(a for a in vars(self).values() if isinstance(a, wp.array))
