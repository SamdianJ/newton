# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble each cloth's shell terms in the shared scalar material solve."""

from typing import Any

import numpy as np
import warp as wp

from .cable_system import _gather
from .cloth import ClothObjective


@wp.kernel
def _residual(
    x: wp.array[Any],
    xn: wp.array[Any],
    vn: wp.array[Any],
    forces: wp.array[wp.vec3],
    gravity: wp.vec3,
    mass: wp.array[Any],
    fixed: wp.array[int],
    offsets: wp.array[int],
    entries: wp.array[int],
    local: wp.array2d[Any],
    rows: wp.array[int],
    dt: Any,
    raw: wp.array[Any],
    absolute: wp.array[Any],
    output: wp.array[Any],
    energy: wp.array[Any],
):
    row = wp.tid()
    vertex, axis = row // 3, row % 3
    value = type(dt)(0.0)
    magnitude = type(dt)(0.0)
    for i in range(offsets[row], offsets[row + 1]):
        entry = entries[i]
        value += local[entry // 18, entry % 18]
        magnitude += wp.abs(local[entry // 18, entry % 18])
    force = type(dt)(forces[vertex][axis]) + mass[vertex] * type(dt)(gravity[axis])
    displacement = x[vertex][axis] - xn[vertex][axis]
    difference = displacement - dt * vn[vertex][axis]
    value -= force
    magnitude += wp.abs(force)
    e = -force * displacement
    if fixed[vertex] == 0:
        value += mass[vertex] * difference / (dt * dt)
        magnitude += wp.abs(mass[vertex] * difference / (dt * dt))
        e += type(dt)(0.5) * mass[vertex] * difference * difference / (dt * dt)
    raw[row] = value
    absolute[row] = magnitude
    energy[row] = e
    if rows[row] >= 0:
        output[rows[row]] = value


@wp.kernel
def _tangent(
    entries: wp.array[int],
    mass: wp.array[Any],
    matrix: wp.array3d[Any],
    dt: Any,
    offset: int,
    output: wp.array[Any],
):
    i = wp.tid()
    entry = entries[i]
    value = output.dtype(0.0)
    if entry < 0:
        value = mass[-entry - 1] / (dt * dt)
    else:
        value = matrix[entry // 324, (entry % 324) // 18, entry % 18]
    output[offset + i] = value


@wp.kernel
def _candidate(
    ids: wp.array[int],
    rows: wp.array[int],
    accepted: wp.array[Any],
    original: wp.array[Any],
    velocity: wp.array[Any],
    delta: wp.array[Any],
    alpha: Any,
    dt: Any,
    predict: bool,
    trial: wp.array[Any],
    trial_velocity: wp.array[Any],
):
    i = wp.tid()
    p = ids[i]
    x = original[p]
    if rows[3 * i] >= 0:
        x = accepted[p]
        if predict:
            x = original[p] + dt * velocity[p]
        elif alpha != type(dt)(0.0):
            for axis in range(3):
                x[axis] += alpha * type(dt)(delta[rows[3 * i + axis]])
    trial[p] = x
    trial_velocity[p] = (x - original[p]) / dt


@wp.kernel
def _path(
    triangles: wp.array2d[int],
    ids: wp.array[int],
    original: wp.array[Any],
    trial: wp.array[Any],
    areas: wp.array2d[Any],
    status: wp.array[int],
):
    face = wp.tid()
    a, b, c = ids[triangles[face, 0]], ids[triangles[face, 1]], ids[triangles[face, 2]]
    e, f = original[b] - original[a], original[c] - original[a]
    de, df = trial[b] - trial[a] - e, trial[c] - trial[a] - f
    normal = wp.cross(e, f)
    length = wp.length(normal)
    n = normal / length
    linear = wp.dot(n, wp.cross(de, f) + wp.cross(e, df))
    quadratic = wp.dot(n, wp.cross(de, df))
    minimum = wp.min(length, length + linear + quadratic)
    if quadratic > type(length)(0.0):
        t = wp.clamp(-linear / (type(length)(2.0) * quadratic), type(length)(0.0), type(length)(1.0))
        minimum = wp.min(minimum, length + t * (linear + t * quadratic))
    # A positive projection bounds the area away from zero for the entire
    # segment. Conservatively shorten trials leaving this local normal chart.
    if not wp.isfinite(minimum) or minimum <= type(length)(1e-8) * areas[face, 0]:
        wp.atomic_max(status, 0, 1)


class ClothBlock:
    """Own a cloth's immutable maps and local buffers without an inner solve."""

    def __init__(self, obj, actor, *, dtype, device):
        self.reference, self.actor = obj, actor
        self.dtype, self.device = dtype, wp.get_device(device)
        vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        ids = obj["particle_ids"]
        lookup = {p: i for i, p in enumerate(ids)}
        stencils = np.array([[lookup[p] if p >= 0 else -1 for p in face] for face in obj["stencils"]], dtype=np.int32)
        material = np.c_[obj["membrane"][:, ::-1], np.tile(obj["bending"], (len(stencils), 1))]
        self.objective = ClothObjective(obj["positions"], stencils, material, dtype=dtype, device=device)
        rows = np.full(3 * len(ids), -1, dtype=np.int32)
        for i, p in enumerate(actor.dynamic_particle_indices):
            local = lookup[p]
            rows[3 * local : 3 * local + 3] = actor.dof_start + 3 * i + np.arange(3)
        self.rows_host = rows
        self.rows = wp.array(rows, dtype=int, device=device)
        self.ids = wp.array(ids, dtype=int, device=device)
        self.mass = wp.array(obj["mass"], dtype=dtype, device=device)
        self.fixed = wp.array(obj["fixed"], dtype=int, device=device)
        self.positions = wp.zeros(len(ids), dtype=vector, device=device)
        self.original_x = wp.zeros_like(self.positions)
        self.original_v = wp.zeros_like(self.positions)
        self.forces = wp.zeros(len(ids), dtype=wp.vec3, device=device)
        self.raw = wp.zeros(len(rows), dtype=dtype, device=device)
        self.absolute = wp.zeros(len(rows), dtype=dtype, device=device)
        self.external_energy = wp.zeros_like(self.raw)
        contributions = [[] for _ in rows]
        rr, cc, values = [], [], []
        for row, global_row in enumerate(rows):
            if global_row >= 0:
                rr.append(global_row)
                cc.append(global_row)
                values.append(-(row // 3) - 1)
        for face, stencil in enumerate(stencils):
            local_rows = np.array([3 * p + axis if p >= 0 else -1 for p in stencil for axis in range(3)])
            for a, row in enumerate(local_rows):
                if row >= 0:
                    contributions[row].append(18 * face + a)
                    for b, column in enumerate(local_rows):
                        if column >= 0 and rows[row] >= 0 and rows[column] >= 0:
                            rr.append(rows[row])
                            cc.append(rows[column])
                            values.append(face * 324 + a * 18 + b)
        self.offsets = wp.array(np.r_[0, np.cumsum([len(v) for v in contributions])], dtype=int, device=device)
        self.entries = wp.array([v for group in contributions for v in group], dtype=int, device=device)
        self.triplet_rows, self.triplet_columns = np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32)
        self.tangent_entries = wp.array(values, dtype=int, device=device)
        self.triplet_offset = 0
        self.path_status = wp.zeros(1, dtype=int, device=device)

    def begin_step(self, original, forces):
        """Freeze inertia and external loads for all trials in the step."""
        for source, target in (
            (original.particle_q, self.original_x),
            (original.particle_qd, self.original_v),
            (forces, self.forces),
        ):
            wp.launch(_gather, self.ids.size, [source, self.ids, target], device=self.device)

    def evaluate(self, state, gravity, dt, complete, *, build_tangent):
        """Combine exact internal forces, external work and positional inertia."""
        wp.launch(_gather, self.ids.size, [state.particle_q, self.ids, self.positions], device=self.device)
        self.objective.evaluate(self.positions, build_tangent=build_tangent)
        if self.objective.flags.numpy()[0]:
            raise ValueError("Invalid cloth geometry or membrane projection")
        wp.launch(
            _residual,
            self.raw.size,
            [
                self.positions,
                self.original_x,
                self.original_v,
                self.forces,
                gravity,
                self.mass,
                self.fixed,
                self.offsets,
                self.entries,
                self.objective.residual,
                self.rows,
                self.dtype(dt),
                self.raw,
                self.absolute,
                complete,
                self.external_energy,
            ],
            device=self.device,
        )
        parts = np.r_[self.objective.energy.numpy().ravel(), self.external_energy.numpy()]
        if not np.isfinite(parts).all() or not np.isfinite(self.raw.numpy()).all():
            raise ValueError("Nonfinite cloth energy or residual")
        return parts.sum(), abs(parts).sum()

    def assemble(self, assembly, dt):
        """Retain every free-coordinate shell block and inertia entry."""
        wp.launch(
            _tangent,
            self.tangent_entries.size,
            [
                self.tangent_entries,
                self.mass,
                self.objective.tangent,
                self.dtype(dt),
                self.triplet_offset,
                assembly.global_scalar_triplets.values,
            ],
            device=self.device,
        )

    def candidate(self, accepted, original, trial, delta, alpha, dt, *, predict=False):
        """Update free positions in the common trial and retain prescribed vertices."""
        wp.launch(
            _candidate,
            self.ids.size,
            [
                self.ids,
                self.rows,
                accepted.particle_q,
                original.particle_q,
                original.particle_qd,
                delta,
                self.dtype(alpha),
                self.dtype(dt),
                predict,
                trial.particle_q,
                trial.particle_qd,
            ],
            device=self.device,
        )

    def path_valid(self, accepted, trial):
        """Reject triangle collapse along the entire actual trial segment."""
        self.path_status.zero_()
        wp.launch(
            _path,
            self.objective.stencils.shape[0],
            [
                self.objective.stencils,
                self.ids,
                accepted.particle_q,
                trial.particle_q,
                self.objective.areas,
                self.path_status,
            ],
            device=self.device,
        )
        return not bool(self.path_status.numpy()[0])
