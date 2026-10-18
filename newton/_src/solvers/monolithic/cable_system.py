# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gather/scatter a cable's physical terms into the shared scalar solve."""

from typing import Any

import numpy as np
import warp as wp

from .cable import CableObjective


@wp.kernel
def _gather(source: wp.array[Any], ids: wp.array[int], target: wp.array[Any]):
    i = wp.tid()
    target[i] = source[ids[i]]


@wp.kernel
def _residual(
    x: wp.array[Any],
    xn: wp.array[Any],
    vn: wp.array[Any],
    theta: wp.array[Any],
    torque: wp.array[Any],
    forces: wp.array[wp.vec3],
    gravity: wp.vec3,
    mass: wp.array[Any],
    fixed: wp.array[int],
    stretch: wp.array2d[Any],
    bend: wp.array2d[Any],
    rows: wp.array[int],
    dt: Any,
    raw: wp.array[Any],
    absolute: wp.array[Any],
    complete: wp.array[Any],
    external_energy: wp.array[Any],
):
    row = wp.tid()
    nv = x.shape[0]
    value = type(dt)(0.0)
    magnitude = type(dt)(0.0)
    energy = type(dt)(0.0)
    if row < 3 * nv:
        vertex, axis = row // 3, row % 3
        for edge in range(wp.max(0, vertex - 1), wp.min(nv - 1, vertex + 1)):
            value += stretch[edge, 3 * (vertex - edge) + axis]
            magnitude += wp.abs(stretch[edge, 3 * (vertex - edge) + axis])
        for stencil in range(wp.max(0, vertex - 2), wp.min(nv - 2, vertex + 1)):
            value += bend[stencil, 3 * (vertex - stencil) + axis]
            magnitude += wp.abs(bend[stencil, 3 * (vertex - stencil) + axis])
        applied = type(dt)(forces[vertex][axis]) + mass[vertex] * type(dt)(gravity[axis])
        displacement = x[vertex][axis] - xn[vertex][axis]
        prediction_error = displacement - dt * vn[vertex][axis]
        value -= applied
        magnitude += wp.abs(applied)
        energy -= applied * displacement
        if fixed[vertex] == 0:
            value += mass[vertex] * prediction_error / (dt * dt)
            magnitude += wp.abs(mass[vertex] * prediction_error / (dt * dt))
            energy += type(dt)(0.5) * mass[vertex] * prediction_error * prediction_error / (dt * dt)
    else:
        edge = row - 3 * nv
        for stencil in range(wp.max(0, edge - 1), wp.min(nv - 2, edge + 1)):
            value += bend[stencil, 9 + edge - stencil]
            magnitude += wp.abs(bend[stencil, 9 + edge - stencil])
        value -= torque[edge]
        magnitude += wp.abs(torque[edge])
        energy = -torque[edge] * theta[edge]
    raw[row] = value
    absolute[row] = magnitude
    external_energy[row] = energy
    if rows[row] >= 0:
        complete[rows[row]] = value


@wp.kernel
def _scatter_tangent(
    kind: wp.array[int],
    element: wp.array[int],
    a: wp.array[int],
    b: wp.array[int],
    mass: wp.array[Any],
    dt: Any,
    stretch: wp.array3d[float],
    bend: wp.array3d[float],
    offset: int,
    output: wp.array[float],
):
    i = wp.tid()
    value = float(0.0)
    if kind[i] == 0:
        value = float(mass[element[i]] / (dt * dt))
    elif kind[i] == 1:
        value = stretch[element[i], a[i], b[i]]
    else:
        value = bend[element[i], a[i], b[i]]
    output[offset + i] = value


@wp.kernel
def _candidate(
    ids: wp.array[int],
    rows: wp.array[int],
    fixed: wp.array[int],
    theta_fixed: wp.array[int],
    theta_offset: int,
    target_x: wp.array[Any],
    target_theta: wp.array[Any],
    accepted_x: wp.array[Any],
    original_x: wp.array[Any],
    original_v: wp.array[Any],
    accepted_theta: wp.array[Any],
    delta: wp.array[Any],
    alpha: Any,
    dt: Any,
    predict: bool,
    trial_x: wp.array[Any],
    trial_v: wp.array[Any],
    trial_theta: wp.array[Any],
):
    i = wp.tid()
    nv = ids.shape[0]
    if i < nv:
        p = ids[i]
        x = target_x[p]
        if fixed[i] == 0:
            x = accepted_x[p]
            if predict:
                x = original_x[p] + dt * original_v[p]
            else:
                for axis in range(3):
                    x[axis] += alpha * type(dt)(delta[rows[3 * i + axis]])
        trial_x[p] = x
        trial_v[p] = (x - original_x[p]) / dt
    if i < nv - 1:
        theta = accepted_theta[theta_offset + i]
        if theta_fixed[i] != 0:
            theta = target_theta[theta_offset + i]
        elif not predict and rows[3 * nv + i] >= 0:
            theta += alpha * type(dt)(delta[rows[3 * nv + i]])
        trial_theta[theta_offset + i] = theta


@wp.kernel
def _publish_frame(
    director: wp.array[Any],
    phi: wp.array[Any],
    theta_offset: int,
    phi_offset: int,
    history_director: wp.array[Any],
    history_phi: wp.array[Any],
):
    i = wp.tid()
    history_director[theta_offset + i] = director[i]
    if i < phi.shape[0]:
        history_phi[phi_offset + i] = phi[i]


@wp.kernel
def _path_domain(
    ids: wp.array[int], x: wp.array[Any], trial: wp.array[Any], lengths0: wp.array[Any], status: wp.array[int]
):
    i = wp.tid()
    edge = x[ids[i + 1]] - x[ids[i]]
    change = trial[ids[i + 1]] - trial[ids[i]] - edge
    norm_sq = wp.dot(change, change)
    alpha = type(lengths0[0])(0.0)
    if norm_sq > type(alpha)(0.0):
        alpha = wp.clamp(-wp.dot(edge, change) / norm_sq, type(alpha)(0.0), type(alpha)(1.0))
    minimum = wp.length(edge + alpha * change)
    if not wp.isfinite(minimum) or minimum <= type(alpha)(1e-8) * lengths0[i]:
        wp.atomic_max(status, 0, 1)


class CableBlock:
    """Own immutable topology, local device terms and exact global scalar maps."""

    def __init__(self, obj, actor, *, dtype, device, phi_offset):
        self.reference, self.actor = obj, actor
        self.dtype, self.device = dtype, wp.get_device(device)
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        self.theta_offset = int(obj["theta_offset"])
        self.phi_offset = phi_offset
        self.objective = CableObjective(
            obj["positions"],
            obj["rest_theta"],
            obj["rest_director"],
            (obj["EA"], obj["bending"], obj["GJ"]),
            dtype=dtype,
            device=device,
        )
        nv = len(obj["particle_ids"])
        rows = np.full(3 * nv + nv - 1, -1, dtype=np.int32)
        local = {p: i for i, p in enumerate(actor.particle_indices)}
        for i, p in enumerate(actor.dynamic_particle_indices):
            rows[3 * local[p] : 3 * local[p] + 3] = actor.dof_start + 3 * i + np.arange(3)
        theta_start = actor.dof_start + 3 * len(actor.dynamic_particle_indices)
        for i, edge in enumerate(actor.free_theta_indices):
            rows[3 * nv + edge - self.theta_offset] = theta_start + i
        self.rows_host = rows
        order = []
        for i in range(nv):
            order.extend(rows[3 * i : 3 * i + 3][rows[3 * i : 3 * i + 3] >= 0])
            if i < nv - 1 and rows[3 * nv + i] >= 0:
                order.append(rows[3 * nv + i])
        self.order = np.asarray(order, dtype=np.int32)
        self.rows = wp.array(rows, dtype=int, device=device)
        self.ids = wp.array(obj["particle_ids"], dtype=int, device=device)
        self.fixed = wp.array(obj["fixed"], dtype=int, device=device)
        self.theta_fixed = wp.array(obj["fixed_theta"], dtype=int, device=device)
        self.mass = wp.array(obj["mass"], dtype=dtype, device=device)
        self.positions = wp.zeros(nv, dtype=self.vector, device=device)
        self.original_x = wp.zeros_like(self.positions)
        self.original_v = wp.zeros_like(self.positions)
        self.forces = wp.zeros(nv, dtype=wp.vec3, device=device)
        self.raw = wp.zeros(len(rows), dtype=dtype, device=device)
        self.absolute = wp.zeros(len(rows), dtype=dtype, device=device)
        self.external_energy = wp.zeros_like(self.raw)
        global_rows, global_columns, kinds, elements, aa, bb = [], [], [], [], [], []

        def entry(kind, element, a, b, row, column):
            if row >= 0 and column >= 0:
                global_rows.append(row)
                global_columns.append(column)
                kinds.append(kind)
                elements.append(element)
                aa.append(a)
                bb.append(b)

        for i in range(nv):
            for axis in range(3):
                row = rows[3 * i + axis]
                entry(0, i, axis, axis, row, row)
        for kind, count, width in ((1, nv - 1, 6), (2, nv - 2, 11)):
            for i in range(count):
                local_rows = (
                    rows[3 * i : 3 * i + 6]
                    if kind == 1
                    else np.r_[rows[3 * i : 3 * i + 9], rows[3 * nv + i : 3 * nv + i + 2]]
                )
                for a in range(width):
                    for b in range(width):
                        entry(kind, i, a, b, local_rows[a], local_rows[b])
        self.triplet_rows, self.triplet_columns = (
            np.asarray(global_rows, dtype=np.int32),
            np.asarray(global_columns, dtype=np.int32),
        )
        self.kinds, self.elements, self.aa, self.bb = [
            wp.array(v, dtype=int, device=device) for v in (kinds, elements, aa, bb)
        ]
        self.triplet_offset = 0
        self.path_status = wp.zeros(1, dtype=int, device=device)

    def begin_step(self, original, forces):
        """Snapshot position/velocity and applied forces once for the step."""
        for source, output in (
            (original.particle_q, self.original_x),
            (original.particle_qd, self.original_v),
            (forces, self.forces),
        ):
            wp.launch(_gather, self.ids.size, [source, self.ids, output], device=self.device)

    def evaluate(self, state, original, torque, gravity, dt, complete):
        """Evaluate the same frozen frame chart for every current and trial state."""
        objective = self.objective
        ne, ni = objective.lengths.size, objective.phi.size
        history, anchor = state._monolithic_coupled, original._monolithic_coupled
        theta = history.cable_theta[self.theta_offset : self.theta_offset + ne]
        wp.launch(_gather, self.ids.size, [state.particle_q, self.ids, self.positions], device=self.device)
        objective.evaluate(
            self.positions,
            theta,
            self.original_x,
            anchor.cable_ref_director[self.theta_offset : self.theta_offset + ne],
            anchor.cable_ref_twist[self.phi_offset : self.phi_offset + ni] if ni else objective.phi,
        )
        if objective.flags.numpy()[0]:
            raise ValueError("Cable candidate left the committed geometry/frame chart")
        wp.launch(
            _residual,
            self.raw.size,
            [
                self.positions,
                self.original_x,
                self.original_v,
                theta,
                torque[self.theta_offset : self.theta_offset + ne],
                self.forces,
                gravity,
                self.mass,
                self.fixed,
                objective.stretch_r,
                objective.bend_twist_r,
                self.rows,
                self.dtype(dt),
                self.raw,
                self.absolute,
                complete,
                self.external_energy,
            ],
            device=self.device,
        )
        parts = np.r_[
            objective.stretch_energy.numpy(), objective.bend_twist_energy.numpy().ravel(), self.external_energy.numpy()
        ]
        if not np.isfinite(parts).all() or not np.isfinite(self.raw.numpy()).all():
            raise ValueError("Nonfinite cable energy or residual")
        if state is not original:
            wp.launch(
                _publish_frame,
                ne,
                [
                    objective.director,
                    objective.phi,
                    self.theta_offset,
                    self.phi_offset,
                    history.cable_ref_director,
                    history.cable_ref_twist,
                ],
                device=self.device,
            )
        return parts.sum(), abs(parts).sum()

    def assemble(self, assembly, dt):
        """Write every free-coordinate mass/DER entry into reserved global triplets."""
        wp.launch(
            _scatter_tangent,
            self.kinds.size,
            [
                self.kinds,
                self.elements,
                self.aa,
                self.bb,
                self.mass,
                self.dtype(dt),
                self.objective.stretch_b,
                self.objective.bend_twist_b,
                self.triplet_offset,
                assembly.global_scalar_triplets.values,
            ],
            device=self.device,
        )

    def candidate(self, accepted, original, trial, delta, target_x, target_theta, alpha, dt, *, predict=False):
        """Apply xyz/theta in the same trial, including prescribed boundary values."""
        wp.launch(
            _candidate,
            self.ids.size,
            [
                self.ids,
                self.rows,
                self.fixed,
                self.theta_fixed,
                self.theta_offset,
                target_x,
                target_theta,
                accepted.particle_q,
                original.particle_q,
                original.particle_qd,
                accepted._monolithic_coupled.cable_theta,
                delta,
                self.dtype(alpha),
                self.dtype(dt),
                predict,
                trial.particle_q,
                trial.particle_qd,
                trial._monolithic_coupled.cable_theta,
            ],
            device=self.device,
        )

    def path_valid(self, accepted, trial):
        """Reject an edge collapse anywhere along the actual line-search segment."""
        self.path_status.zero_()
        wp.launch(
            _path_domain,
            self.ids.size - 1,
            [self.ids, accepted.particle_q, trial.particle_q, self.objective.lengths0, self.path_status],
            device=self.device,
        )
        return not bool(self.path_status.numpy()[0])
