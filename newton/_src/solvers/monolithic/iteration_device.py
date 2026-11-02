# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native-precision device reductions for the synchronous outer Newton loop."""

from typing import Any

import numpy as np
import warp as wp

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
    terms[3, i] = terms.dtype(0)


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
    terms[3, i] = terms.dtype(0)


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
    terms[3, i] = terms.dtype(0)


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


def initialize(solver):
    """Allocate overwrite-only scratch once, without changing physical owners."""
    self = solver
    n, device, dtype = self._complete_r.size, self.model.device, self._dtype
    for name in ("_accepted_r", "_reference_device", "_effective_device", "_stats_residual"):
        setattr(self, name, wp.zeros(n, dtype=dtype, device=device))
    self._row_terms = wp.zeros((4, n), dtype=dtype, device=device)
    self._group_reduction = SegmentedReduction(
        n,
        [np.arange(start, start + count) for _, start, count in self._layout.groups],
        4,
        dtype=dtype,
        device=device,
    )
    self._unrepresentable = wp.zeros(1, dtype=int, device=device)


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
