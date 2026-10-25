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
    step: wp.array[Any],
    vn: wp.array[Any],
    theta_step: wp.array[Any],
    torque: wp.array[Any],
    forces: wp.array[wp.vec3],
    gravity: wp.vec3,
    mass: wp.array[Any],
    fixed: wp.array[int],
    stretch: wp.array2d[Any],
    bend: wp.array2d[Any],
    bend_rounding: wp.array2d[Any],
    rows: wp.array[int],
    dt: Any,
    raw: wp.array[Any],
    absolute: wp.array[Any],
    complete: wp.array[Any],
    external_energy: wp.array[Any],
):
    row = wp.tid()
    nv = step.shape[0]
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
            magnitude += bend_rounding[stencil, 3 * (vertex - stencil) + axis]
        applied = type(dt)(forces[vertex][axis]) + mass[vertex] * type(dt)(gravity[axis])
        displacement = step[vertex][axis]
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
            magnitude += bend_rounding[stencil, 9 + edge - stencil]
        value -= torque[edge]
        magnitude += wp.abs(torque[edge])
        energy = -torque[edge] * theta_step[edge]
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
    stretch: wp.array3d[Any],
    bend: wp.array3d[Any],
    offset: int,
    output: wp.array[Any],
):
    i = wp.tid()
    value = output.dtype(0.0)
    if kind[i] == 0:
        value = mass[element[i]] / (dt * dt)
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
    accepted_step: wp.array[Any],
    original_x: wp.array[Any],
    original_v: wp.array[Any],
    accepted_theta_step: wp.array[Any],
    original_theta: wp.array[Any],
    delta: wp.array[Any],
    alpha: Any,
    dt: Any,
    predict: bool,
    trial_x: wp.array[Any],
    trial_v: wp.array[Any],
    trial_step: wp.array[Any],
    trial_theta: wp.array[Any],
    trial_theta_step: wp.array[Any],
):
    """Advance the exact position and angle increments; trial values are formed once from them."""
    i = wp.tid()
    nv = ids.shape[0]
    if i < nv:
        p = ids[i]
        x = target_x[p]
        step = x - original_x[p]
        if fixed[i] == 0:
            step = accepted_step[p]
            if predict:
                step = dt * original_v[p]
            else:
                for axis in range(3):
                    step[axis] += alpha * type(dt)(delta[rows[3 * i + axis]])
            x = original_x[p] + step
        trial_x[p] = x
        trial_step[p] = step
        trial_v[p] = step / dt
    if i < nv - 1:
        j = theta_offset + i
        theta = target_theta[j]
        increment = theta - original_theta[j]
        if theta_fixed[i] == 0:
            increment = accepted_theta_step[j]
            if not predict and rows[3 * nv + i] >= 0:
                increment += alpha * type(dt)(delta[rows[3 * nv + i]])
            theta = original_theta[j] + increment
        trial_theta[j] = theta
        trial_theta_step[j] = increment


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
    # Parallel transport keeps unit length only to rounding; the published frame
    # anchors the next step, so normalize it rather than let the error accumulate.
    history_director[theta_offset + i] = wp.normalize(director[i])
    if i < phi.shape[0]:
        history_phi[phi_offset + i] = phi[i]


@wp.kernel
def _path_domain(
    ids: wp.array[int],
    original: wp.array[Any],
    accepted: wp.array[Any],
    trial: wp.array[Any],
    lengths0: wp.array[Any],
    status: wp.array[int],
):
    i = wp.tid()
    a, b = ids[i], ids[i + 1]
    edge = (original[b] - original[a]) + (accepted[b] - accepted[a])
    change = (trial[b] - trial[a]) - (accepted[b] - accepted[a])
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
        self.step = wp.zeros(nv, dtype=self.vector, device=device)
        self.original_x = wp.zeros_like(self.step)
        self.original_v = wp.zeros_like(self.step)
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

    def evaluate(self, state, step, theta_step, original, torque, gravity, dt, complete):
        """Evaluate the same frozen frame chart for every current and trial state.

        ``step`` and ``theta_step`` are the state's exact particle and angle
        increments from the step start ``original``.
        """
        objective = self.objective
        ne, ni = objective.lengths.size, objective.phi.size
        history, anchor = state._monolithic_coupled, original._monolithic_coupled
        angles = slice(self.theta_offset, self.theta_offset + ne)
        theta_step = theta_step[angles]
        wp.launch(_gather, self.ids.size, [step, self.ids, self.step], device=self.device)
        objective.evaluate_step(
            self.step,
            anchor.cable_theta[angles],
            theta_step,
            self.original_x,
            anchor.cable_ref_director[angles],
            anchor.cable_ref_twist[self.phi_offset : self.phi_offset + ni] if ni else objective.phi,
        )
        if objective.flags.numpy()[0]:
            raise ValueError("Cable candidate left the committed geometry/frame chart")
        wp.launch(
            _residual,
            self.raw.size,
            [
                self.step,
                self.original_v,
                theta_step,
                torque[self.theta_offset : self.theta_offset + ne],
                self.forces,
                gravity,
                self.mass,
                self.fixed,
                objective.stretch_r,
                objective.bend_twist_r,
                objective.bend_twist_rounding,
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

    def candidate(
        self,
        accepted_steps,
        original,
        trial,
        trial_steps,
        delta,
        target_x,
        target_theta,
        alpha,
        dt,
        *,
        predict=False,
    ):
        """Apply xyz/theta in the same trial, including prescribed boundary values.

        ``accepted_steps`` and ``trial_steps`` are the ``(particle, theta)``
        increments from ``original``; the trial values are formed from them.
        """
        (accepted_step, accepted_theta_step), (trial_step, trial_theta_step) = accepted_steps, trial_steps
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
                accepted_step,
                original.particle_q,
                original.particle_qd,
                accepted_theta_step,
                original._monolithic_coupled.cable_theta,
                delta,
                self.dtype(alpha),
                self.dtype(dt),
                predict,
                trial.particle_q,
                trial.particle_qd,
                trial_step,
                trial._monolithic_coupled.cable_theta,
                trial_theta_step,
            ],
            device=self.device,
        )

    def path_valid(self, original, accepted_step, trial_step):
        """Reject an edge collapse anywhere along the actual line-search segment."""
        self.path_status.zero_()
        wp.launch(
            _path_domain,
            self.ids.size - 1,
            [self.ids, original.particle_q, accepted_step, trial_step, self.objective.lengths0, self.path_status],
            device=self.device,
        )
        return not bool(self.path_status.numpy()[0])
