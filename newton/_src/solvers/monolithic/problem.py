# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bound device computation for the composed monolithic nonlinear problem."""

from typing import Any

import numpy as np
import warp as wp

from . import state_checks
from .producers import validate_row_partition
from .reductions import SegmentedReduction
from .solver_types import _StepFailure

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _energy_terms(
    values: wp.array[Any], absolute: wp.array[Any], supplied_absolute: bool, offset: int, terms: wp.array2d[Any]
):
    i = wp.tid()
    terms[0, offset + i] = values[i]
    magnitude = wp.abs(values[i])
    if supplied_absolute:
        magnitude = absolute[i]
    terms[1, offset + i] = magnitude


@wp.kernel
def _copy_flag(source: wp.array[int], index: int, flags: wp.array[int]):
    flags[index] = source[0]


class _MonolithicProblem:
    """Bind owners, interactions and their required outputs once per solver.

    Producers submit the same kernels on CPU, CUDA and graph replay. Only
    overwrite-only owner work is captured here; interactions retain their
    once-per-evaluation reservation and generation lifecycle.
    """

    def __init__(self, solver):
        self.solver = solver
        self.actors, self.layout = solver._actors, solver._layout
        self.producers = tuple(solver._producers)
        validate_row_partition(self.producers, self.layout.scalar_dof_count)
        names = [producer.name for producer in self.producers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate nonlinear contribution")
        for producer in self.producers:
            if not producer.energies:
                raise ValueError(f"Missing energy contribution: {producer.name}")
            for name in (
                "begin_step",
                "candidate",
                "evaluate",
                "magnitude",
                "coordinates",
                "assemble",
                "domain",
                "publish",
            ):
                if not callable(getattr(producer, name, None)):
                    raise ValueError(f"Missing nonlinear phase {name}: {producer.name}")
        self._energy_bindings = tuple(binding for producer in self.producers for binding in producer.energies)
        self._check_bindings = tuple(binding for producer in self.producers for binding in producer.checks)
        self._finite_bindings = (
            (solver, "_complete_r"),
            (solver, "_complete_a"),
            *(binding for producer in self.producers for binding in producer.finite_values),
        )
        self._bind_evaluation()
        self._energy_shapes = tuple(energy.shape for energy, _ in self.energies)
        self.owners = tuple(p for p in self.producers if not p.interaction)
        self.interactions = tuple(p for p in self.producers if p.interaction)
        count = sum(energy.size for energy, _ in self.energies)
        self.energy_terms = wp.zeros((2, count), dtype=solver._dtype, device=solver.model.device)
        self.energy_reduction = SegmentedReduction(
            count, [np.arange(count)], 2, dtype=solver._dtype, device=solver.model.device
        )
        self.flags = wp.zeros(1 + len(self.checks), dtype=int, device=solver.model.device)
        self.domain_flags = wp.zeros(1, dtype=int, device=solver.model.device)

    def _bind_evaluation(self):
        """Resolve registered storage; numerical generations may reuse or replace it."""
        solver = self.solver
        self.energies = []
        for owner, name, absolute_name in self._energy_bindings:
            energy = state_checks.scalar_view(getattr(owner, name))
            absolute = None if absolute_name is None else state_checks.scalar_view(getattr(owner, absolute_name))
            if energy.dtype != solver._dtype or energy.device != solver.model.device:
                raise ValueError(f"Invalid energy binding: {name}")
            if absolute is not None and (
                absolute.shape != energy.shape or absolute.dtype != energy.dtype or absolute.device != energy.device
            ):
                raise ValueError(f"Invalid absolute energy binding: {name}")
            self.energies.append((energy, absolute))
        shapes = tuple(energy.shape for energy, _ in self.energies)
        if shapes != getattr(self, "_energy_shapes", shapes):
            raise ValueError("Energy replacement changed the contribution layout")
        self.checks = [
            (getattr(owner, name), statuses, message) for owner, name, statuses, message in self._check_bindings
        ]
        self.finite_values = [state_checks.scalar_view(getattr(owner, name)) for owner, name in self._finite_bindings]

    def difference(self, accepted, trial):
        """Write actual tangent displacement through each owner's state geometry."""
        solver = self.solver

        def execute():
            for producer in self.owners:
                producer.difference(solver, accepted, trial, solver._effective_device)

        solver._graphs.run(
            "problem_difference",
            (),
            [*self._buffers(accepted), *self._buffers(trial), solver._effective_device],
            execute,
        )
        return solver._effective_device

    def domain(self, accepted, candidate):
        """Check all local charts and trial paths with one status read."""
        solver = self.solver

        def execute():
            self.domain_flags.zero_()
            for producer in self.producers:
                producer.domain(solver, accepted, candidate, self.domain_flags)

        solver._graphs.run(
            "problem_domain",
            (),
            [*self._buffers(accepted), *self._buffers(candidate), self.domain_flags],
            execute,
        )
        return not bool(self.domain_flags.numpy()[0])

    def _buffers(self, state):
        solver = self.solver
        buffers = [
            solver._complete_r,
            solver._complete_a,
            solver._linear.q_rows,
            solver._frozen_force,
            solver._frozen_gravity,
            solver._cable_torque,
            solver._displacement(state),
            solver._theta_step(state),
            solver._joint_step(state),
            *solver._rigid_steps(state),
        ]
        for owner in (state, state._monolithic_coupled, solver._original, solver._original._monolithic_coupled):
            buffers.extend(value for value in vars(owner).values() if isinstance(value, wp.array))
        for producer in self.owners:
            for owner in producer.owners:
                buffers.extend(value for value in vars(owner).values() if isinstance(value, wp.array))
        return buffers

    def evaluate(self, state, dt, *, build_tangent):
        solver = self.solver
        self._bind_evaluation()
        buffers = self._buffers(state)

        def owners():
            solver._complete_r.zero_()
            for producer in self.owners:
                producer.evaluate(solver, state, dt, solver._complete_r, build_tangent=build_tangent)

        reference = solver._contact_spatial and solver._contact._reference_mode
        parameters = (self.layout.identity, solver.precision.value, dt, build_tangent, reference)
        solver._graphs.run("problem_owners", parameters, buffers, owners)
        for producer in self.interactions:
            producer.evaluate(solver, state, dt, solver._complete_r, build_tangent=build_tangent)

        def summarize():
            solver._complete_a.zero_()
            for producer in self.producers:
                producer.magnitude(solver, solver._complete_a)
            offset = 0
            for energy, absolute in self.energies:
                wp.launch(
                    _energy_terms,
                    energy.size,
                    [energy, energy if absolute is None else absolute, absolute is not None, offset, self.energy_terms],
                    device=solver.model.device,
                )
                offset += energy.size
            self.energy_reduction.reduce(self.energy_terms)
            self.flags.zero_()
            for values in (*self.finite_values, self.energy_terms.flatten()):
                wp.launch(state_checks.finite, values.size, [values, self.flags, 0], device=solver.model.device)
            for index, (flags, _, _) in enumerate(self.checks, 1):
                wp.launch(_copy_flag, 1, [flags, index, self.flags], device=solver.model.device)

        result_buffers = [self.flags, self.energy_terms, *self.energy_reduction.buffers(), *self.finite_values]
        result_buffers.extend(values for pair in self.energies for values in pair if values is not None)
        result_buffers.extend(flags for flags, _, _ in self.checks)
        solver._graphs.run(
            "problem_summary", parameters, [*buffers, *result_buffers, *solver._contact.buffers()], summarize
        )
        flags = self.flags.numpy()
        for index, (_, statuses, message) in enumerate(self.checks, 1):
            if flags[index]:
                raise _StepFailure(statuses.get(int(flags[index]), solver.Status.NONFINITE), message)
        energy, absolute = self.energy_reduction.result.numpy()[0]
        if flags[0] or not np.isfinite(energy) or not np.isfinite(absolute):
            raise _StepFailure(solver.Status.NONFINITE, "Nonfinite coupled E/r")
        solver._metrics["min_det_f"] = float(solver._tet.min_det_f.numpy()[0])
        solver._current_state = state
        solver._settle_cable_gauges()
        return energy, solver._complete_r, absolute
