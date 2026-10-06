# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision device reductions for the synchronous outer Newton loop."""

from typing import Any

import numpy as np
import warp as wp

from . import state_checks

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def measure_terms(
    r: wp.array[Any],
    reference: wp.array[Any],
    scale: wp.array[float],
    nq: int,
    terms: wp.array2d[Any],
    rhs: wp.array[float],
):
    i = wp.tid()
    s = type(r[0])(scale[i])
    a, b = s * r[i], s * reference[i]
    rhs[i] = float(-a)
    for block in range(3):
        active = block == 0 or (block == 1 and i < nq) or (block == 2 and i >= nq)
        aa, bb, rr = type(r[0])(0.0), type(r[0])(0.0), type(r[0])(0.0)
        if active:
            aa, bb, rr = a * a, b * b, r[i] * r[i]
        terms[3 * block, i] = aa
        terms[3 * block + 1, i] = bb
        terms[3 * block + 2, i] = rr


@wp.kernel
def norm_terms(r: wp.array[Any], nq: int, terms: wp.array2d[Any]):
    i = wp.tid()
    for block in range(3):
        value = type(r[0])(0.0)
        if block == 0 or (block == 1 and i < nq) or (block == 2 and i >= nq):
            value = r[i] * r[i]
        terms[block, i] = value


@wp.kernel
def effective_displacement(
    q: wp.array[Any],
    accepted_q: wp.array[Any],
    x: wp.array[Any],
    accepted_x: wp.array[Any],
    ids: wp.array[int],
    out: wp.array[Any],
):
    i = wp.tid()
    nq = q.shape[0]
    if i < nq:
        out[i] = q[i] - accepted_q[i]
    else:
        j = (i - nq) / 3
        axis = (i - nq) % 3
        out[i] = x[ids[j]][axis] - accepted_x[ids[j]][axis]


@wp.kernel
def trial_terms(
    r: wp.array[Any],
    trial: wp.array[Any],
    scale: wp.array[float],
    direction: wp.array[float],
    effective: wp.array[Any],
    nq: int,
    terms: wp.array2d[Any],
):
    i = wp.tid()
    s, d, e = type(r[0])(scale[i]), type(r[0])(direction[i]), effective[i]
    scaled = s * trial[i]
    step = e / s
    terms[0, i] = r[i] * e
    terms[1, i] = scaled * scaled
    terms[2, i] = s * s
    terms[3, i] = d * d
    terms[4, i] = wp.abs(d)
    terms[5, i] = wp.abs(r[i] * d)
    for block in range(3):
        value = type(r[0])(0.0)
        if block == 0 or (block == 1 and i < nq) or (block == 2 and i >= nq):
            value = step * step
        terms[6 + block, i] = value


@wp.kernel
def energy_terms(
    link: wp.array[Any],
    joint: wp.array[Any],
    tet: wp.array[Any],
    contact: wp.array[Any],
    terms: wp.array2d[Any],
):
    i = wp.tid()
    j = i
    value = type(link[0])(0.0)
    if j < link.shape[0]:
        value = link[j]
    else:
        j -= link.shape[0]
        if j < joint.shape[0]:
            value = joint[j]
        else:
            j -= joint.shape[0]
            if j < tet.shape[0]:
                value = tet[j]
            else:
                value = contact[j - tet.shape[0]]
    terms[0, i] = value
    terms[1, i] = wp.abs(value)


@wp.kernel
def evaluation_packet(
    tet: wp.array[int],
    contact: wp.array[int],
    finite: wp.array[int],
    energy: wp.array2d[Any],
    det: wp.array[Any],
    packet: wp.array[Any],
):
    packet[0] = type(packet[0])(tet[0])
    packet[1] = type(packet[0])(contact[0])
    packet[2] = type(packet[0])(finite[0])
    packet[3] = energy[0, 0]
    packet[4] = energy[1, 0]
    packet[5] = det[0]


def initialize(solver):
    """Allocate overwrite-only scratch once, without changing physical owners."""
    self = solver
    n, device, dtype = self._complete_r.size, self.model.device, self._dtype
    for name in ("_accepted_r", "_reference_device", "_effective_device", "_stats_residual"):
        setattr(self, name, wp.zeros(n, dtype=dtype, device=device))
    self._nonlinear_terms = wp.zeros((9, n), dtype=dtype, device=device)
    self._nonlinear_sums = wp.zeros((9, 1), dtype=dtype, device=device)
    self._norm_terms = wp.zeros((3, n), dtype=dtype, device=device)
    self._norm_sums = wp.zeros((3, 1), dtype=dtype, device=device)
    energy_size = sum(
        a.size
        for a in (self._objective.link_energy, self._objective.joint_energy, self._tet.parts, self._contact.energy)
    )
    self._energy_terms = wp.zeros((2, energy_size), dtype=dtype, device=device)
    self._energy_sums = wp.zeros((2, 1), dtype=dtype, device=device)
    self._evaluation_status = wp.zeros(1, dtype=int, device=device)
    self._evaluation_packet = wp.zeros(6, dtype=dtype, device=device)


def evaluate(solver):
    """Append energy reduction and numerical guards to the objective graph."""
    self = solver
    wp.launch(
        energy_terms,
        self._energy_terms.shape[1],
        [
            self._objective.link_energy,
            self._objective.joint_energy,
            self._tet.parts,
            self._contact.energy.flatten(),
            self._energy_terms,
        ],
        device=self.model.device,
    )
    wp.utils.array_sum(self._energy_terms, out=self._energy_sums, axis=1)
    self._evaluation_status.zero_()
    wp.launch(
        state_checks.finite,
        self._complete_r.size,
        [self._complete_r, self._evaluation_status, 0],
        device=self.model.device,
    )
    wp.launch(
        evaluation_packet,
        1,
        [
            self._tet.flags,
            self._contact.flags,
            self._evaluation_status,
            self._energy_sums,
            self._tet.min_det_f,
            self._evaluation_packet,
        ],
        device=self.model.device,
    )


def measure(solver, residual, reference):
    """Return native norms for global/q/x and build the device RHS."""
    self = solver
    if not isinstance(reference, wp.array):
        reference = wp.array(reference, dtype=self._dtype, device=self.model.device)
    buffers = [residual, reference, self._linear.scale, self._nonlinear_terms, self._nonlinear_sums, self._rhs]

    def execute():
        wp.launch(
            measure_terms,
            residual.size,
            [residual, reference, self._linear.scale, self._kinematics.count, self._nonlinear_terms, self._rhs],
            device=self.model.device,
        )
        wp.utils.array_sum(self._nonlinear_terms, out=self._nonlinear_sums, axis=1)

    self._graphs.run("nonlinear_measure", (), buffers, execute)
    norms = np.sqrt(self._nonlinear_sums.numpy().reshape(3, 3))
    self._rhs_norms_host = norms[:, 0].copy()
    return norms


def norms(solver, error):
    """Read only the three physical posterior error norms."""
    self = solver

    def execute():
        wp.launch(norm_terms, error.size, [error, self._kinematics.count, self._norm_terms], device=self.model.device)
        wp.utils.array_sum(self._norm_terms, out=self._norm_sums, axis=1)

    self._graphs.run("nonlinear_posterior_norms", (), [error, self._norm_terms, self._norm_sums], execute)
    return np.sqrt(self._norm_sums.numpy().ravel())


def trial(solver, residual, candidate, effective):
    """Read slope, noise scales and block displacement norms together."""
    self = solver
    buffers = [
        residual,
        candidate,
        self._linear.scale,
        self._delta,
        effective,
        self._nonlinear_terms,
        self._nonlinear_sums,
    ]

    def execute():
        wp.launch(
            trial_terms,
            residual.size,
            [*buffers[:5], self._kinematics.count, self._nonlinear_terms],
            device=self.model.device,
        )
        wp.utils.array_sum(self._nonlinear_terms, out=self._nonlinear_sums, axis=1)

    self._graphs.run("nonlinear_trial", (), buffers, execute)
    return self._nonlinear_sums.numpy().ravel()


@wp.kernel
def domain_check(
    q: wp.array[Any],
    qn: wp.array[Any],
    dof_joint: wp.array[int],
    types: wp.array[int],
    rotations: wp.array[Any],
    rn: wp.array[Any],
    revolute: int,
    half_pi: Any,
    flags: wp.array[int],
):
    i = wp.tid()
    if i < q.shape[0] and types[dof_joint[i]] == revolute:
        if wp.abs(q[i] - qn[i]) >= half_pi:
            wp.atomic_max(flags, 0, 1)
    if i < rotations.shape[0]:
        value = type(q[0])(0.0)
        for a in range(3):
            for b in range(3):
                value += rotations[i][a, b] * rn[i][a, b]
        if not (value > type(q[0])(1.0)):
            wp.atomic_max(flags, 0, 1)


def domain(solver, state, revolute):
    """Check the local coordinate and rotation domain without host matrices."""
    self, k, obj = solver, solver._kinematics, solver._objective
    flags = self._evaluation_status
    buffers = [k.gather_q(state), obj.qn, k.dof_joint, k.types, k.rotations, obj.rn, flags]

    def execute():
        flags.zero_()
        wp.launch(
            domain_check,
            max(k.count, k.bodies),
            [*buffers[:6], revolute, self._dtype(np.pi / 2), flags],
            device=self.model.device,
        )

    self._graphs.run("nonlinear_domain", (), buffers, execute)
    return not bool(flags.numpy()[0])
