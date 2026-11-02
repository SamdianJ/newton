# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bound device computation for the composed monolithic nonlinear problem."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import numpy as np
import warp as wp

from . import iteration_device, state_checks
from .convergence import Unit
from .linear_types import MonolithicPcgConfig
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


@dataclass(frozen=True)
class Assessment:
    """Physical row convergence, normalized merit and sufficient linear accuracy."""

    converged: bool
    merit: float
    caps: dict


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
        energy_bindings = set()
        for producer in self.producers:
            for owner, name, _ in producer.energies:
                binding = (id(owner), name)
                if binding in energy_bindings:
                    raise ValueError(f"Duplicate energy contribution: {producer.name}/{name}")
                energy_bindings.add(binding)
            if not producer.energies or tuple(binding[1] for binding in producer.energies) != producer.energy_fields:
                raise ValueError(f"Missing energy contribution: {producer.name}")
            for name in (
                "begin_step",
                "candidate",
                "difference",
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
        self._assembly_bindings = tuple(binding for producer in self.producers for binding in producer.assembly_checks)
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
        self.assembly_flags = wp.zeros(len(self._assembly_bindings), dtype=int, device=solver.model.device)
        self._predictors = tuple(sorted(self.producers, key=lambda producer: producer.begin_rank))
        self._bind_groups()

    def predict(self, state_in):
        solver = self.solver
        for producer in self.interactions:
            producer.begin_step(self.solver, state_in)

        def execute():
            for producer in self._predictors:
                if not producer.interaction:
                    producer.begin_step(solver, state_in)

        solver._graphs.run(
            "problem_predict",
            (solver._dt,),
            [*self._buffers(state_in), *self._buffers(solver._accepted), solver._delta],
            execute,
        )

    def publish(self, state):
        solver = self.solver

        def execute():
            for producer in self.owners:
                producer.publish(solver, state)

        solver._graphs.run("problem_publish", (solver._dt,), self._buffers(state), execute)
        for producer in self.interactions:
            producer.publish(solver, state)

    def assemble(self, assembly, dt):
        solver = self.solver

        def execute():
            for producer in self.owners:
                producer.assemble(solver, assembly, dt)

        buffers = [
            *self._buffers(solver._current_state),
            solver._diagonal,
            solver._tet_assembly_residual,
            solver._linear.q_pair_rows,
            solver._linear.q_pair_columns,
            assembly.aq_actor_dense,
            assembly.global_scalar_triplets.values,
            assembly.ax_internal_triplets.values,
        ]
        solver._graphs.run("problem_assemble", (dt,), buffers, execute)
        for producer in self.interactions:
            producer.assemble(solver, assembly, dt)
        for index, (owner, field, _, _) in enumerate(self._assembly_bindings):
            wp.launch(_copy_flag, 1, [getattr(owner, field), index, self.assembly_flags], device=solver.model.device)
        flags = self.assembly_flags.numpy()
        for index, (_, _, statuses, message) in enumerate(self._assembly_bindings):
            if flags[index]:
                raise _StepFailure(
                    statuses.get(int(flags[index]), statuses.get(None, solver.Status.NONFINITE)), message
                )

    def _bind_groups(self):
        """Derive reporting and linear policy independently of physical row groups."""
        solver, rows = self.solver, self.solver._criterion.rows
        aliases = {"articulation": "q", "tet": "x"}
        self.blocks = tuple(
            (aliases.get(kind, kind), slice(start, start + count)) for kind, start, count in self.layout.groups
        )
        self.aggregate_name = self.blocks[0][0]
        owner_rows = np.empty(self.layout.scalar_dof_count, dtype=np.int32)
        for index, actor in enumerate(self.actors.actors):
            owner_rows[actor.dof_start : actor.dof_start + actor.dof_count] = index
        self._categories, self._physical_groups = {}, []
        physical_steps, floors, sizes = [], [], []
        for (kind, start, count), (name, part) in zip(self.layout.groups, self.blocks, strict=True):
            sizes.append(count)
            self._physical_groups.append(np.unique(rows.groups[part]))
            actor = None if kind == "global" or not count else self.actors.actors[owner_rows[start]]
            physical_steps.append(actor is not None and actor.kind == "rigid")
            if kind in ("global", "articulation", "tet"):
                self._categories[name] = None
            else:
                self._categories[name] = {
                    "articulation": "articulation_blocks",
                    "rigid": "rigid_blocks",
                    "cloth": "material_blocks",
                    "cable": "material_blocks",
                }[actor.kind]
            if actor is None:
                floor = solver._config.residual_floor_global
            elif actor.kind == "cable":
                floor = 1e-12
            elif actor.kind == "articulation" or np.any(rows.units[part] == Unit.TORQUE):
                floor = solver._config.residual_floor_q
            else:
                floor = solver._config.residual_floor_x
            floors.append(floor)
        self._step_physical = np.asarray(physical_steps, dtype=bool)
        self._group_sizes = np.asarray(sizes, dtype=solver._numpy_dtype)
        self._linear_floors = tuple(floors)
        self._true_residual_interval = 1 if self.layout.rigid_ranges else 10

    def record(self, name, key, value):
        category = self._categories[name]
        if category is None:
            self.solver._metrics[key] = value
        else:
            self.solver._variational.setdefault(category, {}).setdefault(name, {})[key] = value

    def read(self, name, key):
        category = self._categories[name]
        if category is None:
            return self.solver._metrics[key]
        return self.solver._variational[category][name][key]

    def record_initial_merit(self):
        for name, _ in self.blocks:
            key = "merit" + ("" if name == self.aggregate_name else "_" + name)
            self.record(name, key + "_initial", self.read(name, key + "_final"))

    def record_step(self, moved):
        for index, (name, _) in enumerate(self.blocks):
            self.record(name, "scaled_step" + ("" if name == self.aggregate_name else "_" + name), float(moved[index]))

    def pcg_config(self, limits):
        solver = self.solver
        return MonolithicPcgConfig(
            maximum_iterations=solver.linear_max_iterations,
            true_residual_interval=self._true_residual_interval,
            linear_tolerance=solver.linear_tolerance,
            linear_absolute_tolerance=float(max(limits.values())),
            group_norm_limits=tuple(limits[name] for name, _ in self.blocks),
            group_residual_floors=self._linear_floors,
            residual_floor_global=solver._config.residual_floor_global,
            residual_floor_q=solver._config.residual_floor_q,
            residual_floor_x=solver._config.residual_floor_x,
        )

    def evaluate_accepted(self, dt):
        solver = self.solver
        energy, residual, absolute = self.evaluate(solver._accepted, dt, build_tangent=True)
        wp.copy(solver._accepted_r, residual)
        return energy, solver._accepted_r, absolute

    def freeze_reference(self, residual):
        solver = self.solver
        solver._criterion.freeze_reference(residual)
        wp.copy(solver._reference_device, residual)
        return solver._reference_device

    def record_residual(self, energy, residual):
        solver = self.solver
        wp.copy(solver._stats_residual, residual)
        solver._variational.update(objective_total=energy, residual=solver._stats_residual)

    def linearize(self, iteration):
        return self.solver._assemble(iteration, device_result=True)[1]

    def coordinates(self):
        solver = self.solver

        def execute():
            for producer in self.producers:
                producer.coordinates(solver, solver._accepted, solver._coordinates)

        solver._graphs.run(
            "problem_coordinates",
            (),
            [*self._buffers(solver._accepted), solver._coordinates, *solver._contact.buffers()],
            execute,
        )

    def assess(self, residual, reference):
        solver, criterion = self.solver, self.solver._criterion
        native = solver._numpy_dtype
        self.coordinates()
        criterion.round(
            solver._linear.k_global_scalar_bsr,
            solver._coordinates,
            solver._complete_a,
            solver._linear._material_operator,
        )
        packet = iteration_device.assessment(solver, residual, reference)
        linear = packet[: len(self.blocks)]
        worst = packet["worst"][: len(criterion.rows.names)].astype(native, copy=False)
        limited = packet["limited"][: len(criterion.rows.names)]
        merit_square = native(packet["merit_square"][0])
        if not np.all(np.isfinite(worst)) or not np.isfinite(merit_square):
            raise _StepFailure(solver.Status.NONFINITE, "Nonfinite convergence summary")
        solver._rhs_norms_host = np.sqrt(linear["rhs_square"].astype(native, copy=False))
        raw_norms = np.sqrt(linear["raw_square"].astype(native, copy=False))
        caps = {}
        ratios = [native(worst[groups].max(initial=0)) for groups in self._physical_groups]
        for index, (name, _) in enumerate(self.blocks):
            squares = native(linear["residual_square"][index])
            reference_squares = native(linear["reference_square"][index])
            size = max(native(1), self._group_sizes[index])
            caps[name] = native(0.5) * native(linear["minimum"][index])
            suffix = "" if name == "global" else "_" + name
            self.record(name, "convergence_ratio" + suffix, float(ratios[index]))
            self.record(name, "merit" + suffix + "_final", float(np.sqrt(squares / size)))
            self.record(name, "merit" + suffix + "_reference", float(np.sqrt(reference_squares / size)))
            if name != "global":
                self.record(name, "raw_residual_" + name + "_norm", float(raw_norms[index]))
        solver._variational["convergence_groups"] = {
            group: {"ratio": float(value), "rounding_limited": int(count)}
            for group, value, count in zip(criterion.rows.names, worst, limited, strict=True)
        }
        merit = native(np.sqrt(merit_square / native(max(1, criterion.tau.size))))
        return Assessment(bool(ratios[0] <= 1), merit, caps)

    def merit(self):
        solver, criterion = self.solver, self.solver._criterion
        criterion.assess(solver._complete_r)
        native = solver._numpy_dtype
        return native(np.sqrt(criterion.square_sum.numpy()[0] / native(max(1, criterion.tau.size))))

    def posterior_norms(self, error):
        return iteration_device.posterior(self.solver, error)

    def trial_summary(self, residual, effective):
        values = iteration_device.trial(self.solver, residual, effective)
        # Global statistics retain scaled motion; rigid groups report physical
        # COM/rotation increments, as in the existing publication contract.
        squares = np.where(self._step_physical, values[:, 3], values[:, 2])
        moved = np.sqrt(squares / np.maximum(self._group_sizes, 1))
        return values[0, 0], values[0, 1], moved

    def unrepresentable(self, effective, eps):
        return iteration_device.representable(self.solver, effective, eps)

    def candidate(self, alpha, dt):
        solver = self.solver

        def execute():
            for producer in self.producers:
                producer.candidate(solver, solver._accepted, solver._trial, solver._delta, alpha, dt)

        buffers = [
            *self._buffers(solver._accepted),
            *self._buffers(solver._trial),
            solver._delta,
            solver._cable_target_x,
            solver._cable_target_theta,
        ]
        solver._graphs.run("problem_candidate", (float(alpha), dt), buffers, execute)

    def evaluate_trial(self, dt):
        solver = self.solver
        try:
            return self.evaluate(solver._trial, dt, build_tangent=False)
        except _StepFailure as error:
            if error.status == solver.Status.CONTACT_OVERFLOW:
                raise
            solver._variational["trials"].append({"accepted": False, "domain_failure": str(error)})
            return None

    def accept(self):
        solver = self.solver
        solver._accepted, solver._trial = solver._trial, solver._accepted

    def refresh(self):
        if self.solver._contact_spatial:
            self.solver._refresh_candidates()

    def conservative_energy(self):
        return self.solver._conservative_energy()

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
            solver._cable_target_x,
            solver._cable_target_theta,
            solver._displacement(state),
            solver._theta_step(state),
            solver._joint_step(state),
            *solver._rigid_steps(state),
        ]
        for owner in (state, state._monolithic_coupled, solver._original, solver._original._monolithic_coupled):
            buffers.extend(value for value in vars(owner).values() if isinstance(value, wp.array))
        for producer in self.owners:
            for owner in producer.owners:
                names = (field.name for field in fields(owner)) if is_dataclass(owner) else vars(owner)
                buffers.extend(getattr(owner, name) for name in names if isinstance(getattr(owner, name), wp.array))
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
        for index, (source, statuses, message) in enumerate(self.checks, 1):
            if flags[index]:
                raise _StepFailure(
                    statuses.get(int(flags[index]), solver.Status.NONFINITE),
                    f"{message}: {source.numpy().tolist()}",
                )
        energy, absolute = self.energy_reduction.result.numpy()[0]
        if flags[0] or not np.isfinite(energy) or not np.isfinite(absolute):
            raise _StepFailure(solver.Status.NONFINITE, "Nonfinite coupled E/r")
        solver._metrics["min_det_f"] = float(solver._tet.min_det_f.numpy()[0])
        solver._current_state = state
        solver._settle_cable_gauges()
        return energy, solver._complete_r, absolute
