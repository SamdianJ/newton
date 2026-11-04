# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device qualification and projection of cable common-angle null modes."""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .convergence import Unit
from .material_batch import _DeviceRecords
from .reductions import SegmentedReduction
from .solver_types import _StepFailure

wp.set_module_options({"fuse_fp": False})


@cache
def _record_type(dtype):
    @wp.struct
    class Cable:
        raw: wp.array[dtype]
        fixed: wp.array[int]
        jacobian: wp.array3d[dtype]
        start: int
        offset: int

    return Cable


@wp.kernel
def _gauge_terms(cables: wp.array[Any], jobs: wp.array[wp.vec2i], terms: wp.array2d[Any], blocked: wp.array[int]):
    job = jobs[wp.tid()]
    actor, index = job[0], job[1]
    cable = cables[actor]
    if index < cable.fixed.shape[0]:
        value = cable.raw[cable.start + index]
        terms[0, cable.offset + index] = value
        square = value * value
        if cable.fixed[index] != 0:
            square = terms.dtype(0)
        terms[1, cable.offset + index] = square
    width = cable.jacobian.shape[1]
    if index < cable.jacobian.shape[0] * width:
        i, j = index // width, index % width
        if cable.jacobian[i, j, 9] != -cable.jacobian[i, j, 10]:
            wp.atomic_max(blocked, actor, 1)


@wp.kernel
def _factor_modes(
    count: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    weights: wp.array[Any],
    row_actor: wp.array[int],
    blocked: wp.array[int],
):
    factor = wp.tid()
    if factor >= count[0] or not (weights[factor] > weights.dtype(0)):
        return
    begin, end = offsets[factor], offsets[factor + 1]
    # A current material contact reads one edge angle. Group by row ownership
    # explicitly so empty factors and arbitrary support ordering remain valid.
    for j in range(begin, end):
        actor = row_actor[columns[j]]
        if actor < 0:
            continue
        first = bool(True)
        for k in range(begin, j):
            if row_actor[columns[k]] == actor:
                first = False
        if first:
            total = values.dtype(0)
            for k in range(j, end):
                if row_actor[columns[k]] == actor:
                    total += values[k]
            if total != values.dtype(0):
                wp.atomic_max(blocked, actor, 1)


@wp.kernel
def _qualify(
    eligible: wp.array[int],
    permanent: wp.array[int],
    first_rows: wp.array[int],
    blocked: wp.array[int],
    sums: wp.array2d[Any],
    reference: bool,
    tolerance: Any,
    pins: wp.array[int],
    summary: wp.array2d[Any],
):
    i = wp.tid()
    pins[i] = -1
    summary[i, 0] = wp.sqrt(sums[i, 1])
    summary[i, 1] = summary.dtype(0)
    if eligible[i] != 0 and not reference:
        null = blocked[i] == 0
        if permanent[i] != 0 or null:
            if wp.abs(sums[i, 0]) > tolerance:
                summary[i, 1] = summary.dtype(1)
        if null and permanent[i] == 0:
            pins[i] = first_rows[i]


@wp.kernel
def _direction_terms(delta: wp.array[Any], terms: wp.array2d[wp.float64]):
    i = wp.tid()
    terms[0, i] = wp.float64(delta[i])


@wp.kernel
def _project(
    rows: wp.array[int],
    counts: wp.array[int],
    pins: wp.array[int],
    sums: wp.array2d[wp.float64],
    delta: wp.array[Any],
):
    i = wp.tid()
    actor = rows[i]
    if actor >= 0:
        if pins[actor] >= 0:
            mean = sums[actor, 0] / wp.float64(counts[actor])
            delta[i] = delta.dtype(wp.float64(delta[i]) - mean)


class CableGauges:
    """Keep trial qualification separate from the accepted operator's pins."""

    def __init__(self, solver):
        self.cables = tuple(solver._cables)
        if not self.cables:
            return
        self.device, self.dtype = solver.model.device, solver._dtype
        n, count = solver._layout.scalar_dof_count, len(self.cables)
        groups, theta_rows, eligible, permanent, first_rows = [], [], [], [], []
        row_actor = np.full(n, -1, dtype=np.int32)
        offset = 0
        self.offsets = []
        for index, cable in enumerate(self.cables):
            self.offsets.append(offset)
            size = cable.theta_fixed.size
            groups.append(np.arange(offset, offset + size))
            offset += size
            rows = cable.rows_host[3 * cable.ids.size :]
            free = rows[rows >= 0]
            theta_rows.append(free)
            row_actor[free] = index
            eligible.append(not cable.reference["fixed_theta"].any())
            permanent.append(cable.reference["gauge"])
            first_rows.append(int(free[0]) if free.size else -1)
        self.row_actor = wp.array(row_actor, dtype=int, device=self.device)
        self.eligible, self.permanent, self.first_rows, self.counts = (
            wp.array(np.asarray(values, dtype=np.int32), dtype=int, device=self.device)
            for values in (eligible, permanent, first_rows, [len(rows) for rows in theta_rows])
        )
        self.blocked = wp.zeros(count, dtype=int, device=self.device)
        self.pins = wp.full(count, -1, dtype=int, device=self.device)
        self.accepted_pins = wp.full(count, -1, dtype=int, device=self.device)
        self.terms = wp.zeros((2, offset), dtype=self.dtype, device=self.device)
        self.raw_reduction = SegmentedReduction(offset, groups, 2, dtype=self.dtype, device=self.device)
        self.summary = wp.zeros((count, 2), dtype=self.dtype, device=self.device)
        # Preserve the existing float64 mean, including for an FP32 direction.
        self.direction_terms = wp.zeros((1, n), dtype=wp.float64, device=self.device)
        self.direction_reduction = SegmentedReduction(n, theta_rows, 1, dtype=wp.float64, device=self.device)
        counts = [max(c.theta_fixed.size, np.prod(c.objective.jacobian.shape[:2])) for c in self.cables]
        self._records = _DeviceRecords(_record_type(self.dtype), self._views(), counts, device=self.device)

    def _views(self):
        return [
            (c.raw, c.theta_fixed, c.objective.jacobian, 3 * c.ids.size, offset)
            for c, offset in zip(self.cables, self.offsets, strict=True)
        ]

    def qualify(self, solver):
        if not self.cables:
            solver._cable_raw_theta_norms = {}
            return
        records = solver._contact.pool.records
        reference = solver._contact_spatial and solver._contact._reference_mode
        factors = [
            records.factor_count,
            records.factor_offsets,
            records.factor_columns,
            records.factor_values,
            records.factor_weights,
            self.row_actor,
            self.blocked,
        ]
        bound = [self.eligible, self.permanent, self.first_rows, self.blocked, self.raw_reduction.result]
        buffers = [*factors, *bound, self.terms, self.pins, self.summary, *self.raw_reduction.buffers()]
        records_view = self._records.bind(self._views())
        buffers.extend(self._records.buffers())

        def execute():
            self.blocked.zero_()
            wp.launch(
                _gauge_terms,
                self._records.jobs.size,
                [records_view, self._records.jobs, self.terms, self.blocked],
                device=self.device,
            )
            self.raw_reduction.reduce(self.terms)
            wp.launch(_factor_modes, records.factor_weights.size, factors, device=self.device)
            wp.launch(
                _qualify,
                len(self.cables),
                [
                    *bound,
                    reference,
                    self.dtype(solver._config.convergence.absolute[Unit.TORQUE]),
                    self.pins,
                    self.summary,
                ],
                device=self.device,
            )

        solver._graphs.run("cable_gauge_qualify", (reference,), buffers, execute)
        summary = self.summary.numpy()
        solver._cable_raw_theta_norms = {
            f"cable_{cable.actor.source_index}_theta": float(summary[index, 0])
            for index, cable in enumerate(self.cables)
        }
        if np.any(summary[:, 1]):
            raise _StepFailure(
                solver.Status.LINEAR_NON_POSITIVE_CURVATURE, "Cable common-angle mode has incompatible net torque"
            )

    def commit(self, preconditioner):
        """Freeze pins for the operator that PCG is about to solve."""
        if not self.cables:
            return
        wp.copy(self.accepted_pins, self.pins)
        if preconditioner is not None:
            wp.copy(preconditioner.gauge_rows, self.pins)

    def project(self, solver):
        if not self.cables:
            return
        reduction = self.direction_reduction
        buffers = [self.row_actor, self.counts, self.accepted_pins, reduction.result, solver._delta]

        def execute():
            wp.launch(_direction_terms, solver._delta.size, [solver._delta, self.direction_terms], device=self.device)
            reduction.reduce(self.direction_terms)
            wp.launch(_project, solver._delta.size, buffers, device=self.device)

        # Projection changes delta; a warmup would apply it an extra time.
        solver._graphs.run(
            "cable_gauge_project", (), [*buffers, self.direction_terms, *reduction.buffers()], execute, warmup=False
        )
