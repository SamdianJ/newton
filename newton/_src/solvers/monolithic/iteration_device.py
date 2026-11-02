# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision device reductions for the synchronous outer Newton loop."""

from typing import Any

import numpy as np
import warp as wp

from .reductions import SegmentedReduction

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def assessment_rows(
    r: wp.array[Any],
    reference: wp.array[Any],
    tau: wp.array[Any],
    scale: wp.array[Any],
    terms: wp.array2d[Any],
    rhs: wp.array[Any],
):
    i = wp.tid()
    scaled = scale[i] * r[i]
    a, b = r[i] / tau[i], reference[i] / tau[i]
    rhs[i] = -scaled
    terms[0, i] = scaled * scaled
    terms[1, i] = r[i] * r[i]
    terms[2, i] = a * a
    terms[3, i] = b * b
    terms[4, i] = scale[i] * tau[i]


@wp.struct
class _AssessmentPacket:
    # Transport both native precisions losslessly; counts remain integers.
    rhs_square: wp.float64
    raw_square: wp.float64
    residual_square: wp.float64
    reference_square: wp.float64
    minimum: wp.float64
    worst: wp.float64
    limited: int
    merit_square: wp.float64


@wp.kernel
def _pack_assessment(
    sums: wp.array2d[Any],
    worst: wp.array[Any],
    limited: wp.array[int],
    square_sum: wp.array[Any],
    packet: wp.array[_AssessmentPacket],
):
    i = wp.tid()
    value = _AssessmentPacket()
    if i < sums.shape[0]:
        value.rhs_square = wp.float64(sums[i, 0])
        value.raw_square = wp.float64(sums[i, 1])
        value.residual_square = wp.float64(sums[i, 2])
        value.reference_square = wp.float64(sums[i, 3])
        value.minimum = wp.float64(sums[i, 4])
    if i < worst.shape[0]:
        value.worst = wp.float64(worst[i])
        value.limited = limited[i]
    if i == 0:
        value.merit_square = wp.float64(square_sum[0])
    packet[i] = value


@wp.kernel
def unrepresentable_rows(effective: wp.array[Any], coordinates: wp.array[Any], eps: Any, count: wp.array[int]):
    i = wp.tid()
    if wp.abs(effective[i]) > eps * coordinates[i]:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def _posterior_chunks(
    error: wp.array[Any],
    rhs: wp.array[Any],
    magnitude: wp.array[Any],
    original: wp.array[Any],
    regularization: wp.array[Any],
    coefficient: wp.array[Any],
    rows: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    partials: wp.array2d[Any],
):
    chunk = wp.tid()
    e, b, m, o, t = error.dtype(0), error.dtype(0), error.dtype(0), error.dtype(0), error.dtype(0)
    for j in range(starts[chunk], ends[chunk]):
        row = rows[j]
        e += error[row] * error[row]
        b += rhs[row] * rhs[row]
        m += magnitude[row] * magnitude[row]
        if coefficient[0] != coefficient.dtype(0):
            o += original[row] * original[row]
            t += regularization[row] * regularization[row]
    if coefficient[0] == coefficient.dtype(0):
        o = e
    partials[chunk, 0] = e
    partials[chunk, 1] = b
    partials[chunk, 2] = m
    partials[chunk, 3] = o
    partials[chunk, 4] = t


@wp.kernel
def _posterior_groups(
    partials: wp.array2d[Any], offsets: wp.array[int], internal: wp.array[Any], packet: wp.array2d[Any]
):
    group, metric = wp.tid()
    # Five squared norms followed by the already computed PCG internal norm.
    if metric == 5:
        packet[group, metric] = internal[group]
    else:
        value = partials.dtype(0)
        for j in range(offsets[group], offsets[group + 1]):
            value += partials[j, metric]
        packet[group, metric] = value


@wp.kernel
def trial_rows(
    r: wp.array[Any],
    scale: wp.array[Any],
    direction: wp.array[Any],
    effective: wp.array[Any],
    terms: wp.array2d[Any],
):
    i = wp.tid()
    step = effective[i] / scale[i]
    terms[0, i] = r[i] * effective[i]
    terms[1, i] = wp.abs(r[i] * direction[i])
    terms[2, i] = step * step
    terms[3, i] = effective[i] * effective[i]
    terms[4, i] = terms.dtype(0)


def initialize(solver):
    """Allocate overwrite-only scratch once, without changing physical owners."""
    self = solver
    n, device, dtype = self._complete_r.size, self.model.device, self._dtype
    for name in ("_accepted_r", "_reference_device", "_effective_device", "_stats_residual"):
        setattr(self, name, wp.zeros(n, dtype=dtype, device=device))
    self._row_terms = wp.zeros((5, n), dtype=dtype, device=device)
    self._group_reduction = SegmentedReduction(
        n,
        [np.arange(start, start + count) for _, start, count in self._layout.groups],
        5,
        dtype=dtype,
        device=device,
    )
    group_count = len(self._layout.groups)
    self._posterior_packet = wp.zeros((group_count, 6), dtype=dtype, device=device)
    self._assessment_packet = wp.zeros(
        max(1, group_count, len(self._criterion.rows.names)), dtype=_AssessmentPacket, device=device
    )
    self._unrepresentable = wp.zeros(1, dtype=int, device=device)


def posterior(solver, error):
    """Read external and PCG-internal norms together in fixed column order."""
    self = solver
    reduction = self._group_reduction
    internal = self._linear._masked_pcg_plan.norms
    inputs = [
        error,
        self._rhs,
        self._post_magnitude,
        self._post_original,
        self._post_regularization,
        self._post_lambda,
        reduction.rows,
        reduction.starts,
        reduction.ends,
        reduction.partials,
    ]

    def execute():
        wp.launch(_posterior_chunks, reduction.starts.size, inputs, device=self.model.device)
        wp.launch(
            _posterior_groups,
            self._posterior_packet.shape,
            [reduction.partials, reduction.offsets, internal, self._posterior_packet],
            device=self.model.device,
        )

    self._graphs.run(
        "nonlinear_posterior_norms", (), [*inputs, reduction.offsets, internal, self._posterior_packet], execute
    )
    values = self._posterior_packet.numpy()
    values[:, :5] = np.sqrt(values[:, :5])
    return values


def trial(solver, residual, effective):
    """Read slope, noise scale and actual per-group step norms together."""
    self = solver
    reduction = self._group_reduction
    buffers = [residual, self._linear.scale, self._delta, effective, self._row_terms]

    def execute():
        wp.launch(trial_rows, residual.size, buffers, device=self.model.device)
        reduction.reduce(self._row_terms)

    self._graphs.run("nonlinear_trial", (), [*buffers, *reduction.buffers()], execute)
    return reduction.result.numpy()


def assessment(solver, residual, reference):
    """Build the RHS and collect linear and physical summaries at one boundary."""
    self = solver
    reduction, criterion = self._group_reduction, self._criterion
    inputs = [residual, reference, criterion.tau, self._linear.scale, self._row_terms, self._rhs]

    def execute():
        wp.launch(assessment_rows, residual.size, inputs, device=self.model.device)
        reduction.reduce(self._row_terms, minimum=4)
        criterion.assess(residual)
        wp.launch(
            _pack_assessment,
            self._assessment_packet.size,
            [reduction.result, criterion.worst, criterion.limited, criterion.square_sum, self._assessment_packet],
            device=self.model.device,
        )

    self._graphs.run(
        "nonlinear_assessment",
        (),
        [*inputs, *reduction.buffers(), *criterion.buffers(), self._assessment_packet],
        execute,
    )
    return self._assessment_packet.numpy()


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
