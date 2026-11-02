# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision device reductions for the synchronous outer Newton loop."""

from typing import Any

import numpy as np
import warp as wp

from . import state_checks
from .reductions import SegmentedReduction

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def measure_rows(
    r: wp.array[Any],
    reference: wp.array[Any],
    scale: wp.array[Any],
    terms: wp.array2d[Any],
    rhs: wp.array[Any],
):
    i = wp.tid()
    a, b = scale[i] * r[i], scale[i] * reference[i]
    rhs[i] = -a
    terms[0, i] = a * a
    terms[1, i] = b * b
    terms[2, i] = r[i] * r[i]


@wp.kernel
def assessment_rows(
    r: wp.array[Any],
    reference: wp.array[Any],
    tau: wp.array[Any],
    scale: wp.array[Any],
    terms: wp.array2d[Any],
):
    i = wp.tid()
    a, b = r[i] / tau[i], reference[i] / tau[i]
    terms[0, i] = a * a
    terms[1, i] = b * b
    terms[2, i] = scale[i] * tau[i]


@wp.kernel
def unrepresentable_rows(effective: wp.array[Any], coordinates: wp.array[Any], eps: Any, count: wp.array[int]):
    i = wp.tid()
    if wp.abs(effective[i]) > eps * coordinates[i]:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def norm_rows(r: wp.array[Any], terms: wp.array2d[Any]):
    i = wp.tid()
    terms[0, i] = r[i] * r[i]
    terms[1, i] = terms.dtype(0)
    terms[2, i] = terms.dtype(0)


@wp.kernel
def effective_displacement(
    q_step: wp.array[Any],
    accepted_q_step: wp.array[Any],
    x: wp.array[Any],
    accepted_x: wp.array[Any],
    ids: wp.array[int],
    out: wp.array[Any],
):
    i = wp.tid()
    nq = q_step.shape[0]
    if i < nq:
        out[i] = q_step[i] - accepted_q_step[i]
    else:
        j = (i - nq) / 3
        axis = (i - nq) % 3
        out[i] = x[ids[j]][axis] - accepted_x[ids[j]][axis]


@wp.kernel
def trial_rows(
    r: wp.array[Any],
    scale: wp.array[Any],
    direction: wp.array[Any],
    effective: wp.array[Any],
    unscaled: wp.array[int],
    terms: wp.array2d[Any],
):
    i = wp.tid()
    step = effective[i]
    if unscaled[i] == 0:
        step /= scale[i]
    terms[0, i] = r[i] * effective[i]
    terms[1, i] = wp.abs(r[i] * direction[i])
    terms[2, i] = step * step


@wp.kernel
def energy_terms(
    link: wp.array[Any],
    joint: wp.array[Any],
    tet: wp.array[Any],
    contact: wp.array[Any],
    contact_absolute: wp.array[Any],
    pooled: bool,
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
    if pooled and i >= link.shape[0] + joint.shape[0] + tet.shape[0]:
        terms[1, i] = contact_absolute[j - tet.shape[0]]


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


@wp.kernel
def host_evaluation_packet(
    rigid: wp.array[Any],
    link: wp.array[Any],
    joint: wp.array[Any],
    tet: wp.array[Any],
    contact: wp.array[Any],
    absolute: wp.array[Any],
    tet_flags: wp.array[int],
    contact_flags: wp.array[int],
    det: wp.array[Any],
    pooled: bool,
    packet: wp.array[Any],
):
    i = wp.tid()
    value = packet.dtype(0)
    if i == 0:
        value = packet.dtype(tet_flags[0])
    elif i == 1:
        value = packet.dtype(contact_flags[0])
    elif i == 2:
        value = det[0]
    elif i < 5:
        if pooled:
            value = absolute[i - 3]
    else:
        j = i - 5
        if j < rigid.shape[0]:
            value = rigid[j]
        else:
            j -= rigid.shape[0]
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
    packet[i] = value


def initialize(solver):
    """Allocate overwrite-only scratch once, without changing physical owners."""
    self = solver
    n, device, dtype = self._complete_r.size, self.model.device, self._dtype
    for name in ("_accepted_r", "_reference_device", "_effective_device", "_stats_residual"):
        setattr(self, name, wp.zeros(n, dtype=dtype, device=device))
    self._row_terms = wp.zeros((3, n), dtype=dtype, device=device)
    self._group_reduction = SegmentedReduction(
        n,
        [np.arange(start, start + count) for _, start, count in self._layout.groups],
        3,
        dtype=dtype,
        device=device,
    )
    unscaled = np.zeros(n, dtype=np.int32)
    for _, start in self._layout.rigid_ranges:
        unscaled[start : start + 6] = 1
    self._step_unscaled = wp.array(unscaled, dtype=int, device=device)
    self._unrepresentable = wp.zeros(1, dtype=int, device=device)
    energy_size = sum(
        a.size
        for a in (self._objective.link_energy, self._objective.joint_energy, self._tet.parts, self._contact.energy)
    )
    self._energy_terms = wp.zeros((2, energy_size), dtype=dtype, device=device)
    self._energy_sums = wp.zeros((2, 1), dtype=dtype, device=device)
    self._evaluation_status = wp.zeros(1, dtype=int, device=device)
    self._evaluation_packet = wp.zeros(6, dtype=dtype, device=device)
    # Copy terms without reducing them: the host path retains NumPy's sum order.
    self._host_evaluation_packet = wp.zeros(5 + self._rigid.energy.size + energy_size, dtype=dtype, device=device)


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
            self._contact.absolute_energy if self._contact_pool else self._contact.energy.flatten(),
            self._contact_pool,
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
    """Build the RHS and read norms in LinearLayout group order."""
    self = solver
    reduction = self._group_reduction
    buffers = [residual, reference, self._linear.scale, self._row_terms, self._rhs, *reduction.buffers()]

    def execute():
        wp.launch(measure_rows, residual.size, buffers[:5], device=self.model.device)
        reduction.reduce(self._row_terms)

    self._graphs.run("nonlinear_measure", (), buffers, execute)
    values = np.sqrt(reduction.result.numpy())
    self._rhs_norms_host = values[:, 0].copy()
    return values


def norms(solver, error):
    """Read posterior error norms for every active layout group."""
    self = solver
    reduction = self._group_reduction

    def execute():
        wp.launch(norm_rows, error.size, [error, self._row_terms], device=self.model.device)
        reduction.reduce(self._row_terms)

    self._graphs.run("nonlinear_posterior_norms", (), [error, self._row_terms, *reduction.buffers()], execute)
    return np.sqrt(reduction.result.numpy()[:, 0])


def trial(solver, residual, candidate, effective):
    """Read slope, noise scale and actual per-group step norms together."""
    self = solver
    reduction = self._group_reduction
    buffers = [residual, self._linear.scale, self._delta, effective, self._step_unscaled, self._row_terms]

    def execute():
        wp.launch(trial_rows, residual.size, buffers, device=self.model.device)
        reduction.reduce(self._row_terms)

    self._graphs.run("nonlinear_trial", (), [*buffers, *reduction.buffers()], execute)
    values = reduction.result.numpy()
    # Preserve the strategy packet until it moves into the problem interface.
    return np.r_[values[0, 0], np.zeros(4, dtype=self._numpy_dtype), values[0, 1], values[:, 2]]


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


def assessment(solver, residual, reference):
    """Read per-group r/tau squares and the smallest scaled row tolerance."""
    self = solver
    reduction = self._group_reduction
    wp.launch(
        assessment_rows,
        residual.size,
        [residual, reference, self._criterion.tau, self._linear.scale, self._row_terms],
        device=self.model.device,
    )
    reduction.reduce(self._row_terms, minimum=2)
    values = reduction.result.numpy()
    return values[:, :2], values[:, 2]


def representable(solver, effective, eps):
    """Return whether no accepted row moved by more than its coordinate rounding."""
    self = solver
    self._unrepresentable.zero_()
    wp.launch(
        unrepresentable_rows,
        effective.size,
        [effective, self._coordinates, self._dtype(eps), self._unrepresentable],
        device=self.model.device,
    )
    return not bool(self._unrepresentable.numpy()[0])
