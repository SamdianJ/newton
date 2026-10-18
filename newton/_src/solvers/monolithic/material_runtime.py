# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material producers for the existing monolithic iteration and publication path."""

from typing import Any

import numpy as np
import warp as wp

from . import coupled_kernels as kernels
from .linear_types import MonolithicLinearGeneration
from .rigid import _scatter_residual, _scatter_tangent
from .solver_types import _StepFailure
from .state_checks import scalar_view
from .tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
)


@wp.kernel
def _scatter(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = values[i]


@wp.kernel
def _add_contact(source: wp.array[Any], target: wp.array[Any]):
    i = wp.tid()
    target[i] += source[i]


def evaluate(self, state, dt, *, build_tangent=True):
    """Combine A/R/T and cloth/cable E/r without splitting any material angle solve."""
    k, obj = self._kinematics, self._objective
    self._complete_r.zero_()
    obj.evaluate(k.gather_q(state), dt, build_tangent=build_tangent)
    self._rigid.evaluate(state, dt, build_tangent)
    wp.launch(
        _scatter, k.count, [obj.residual, self._layout.q_scalar_offset, self._complete_r], device=self.model.device
    )
    wp.launch(
        _scatter_residual,
        (self._rigid.count, 6),
        [self._rigid.rows, self._rigid.residual, self._complete_r],
        device=self.model.device,
    )
    parts = np.r_[obj.link_energy.numpy(), obj.joint_energy.numpy(), self._rigid.energy.numpy()]
    total, absolute = parts.sum(), abs(parts).sum()
    if self._has_tet:
        self._tet.evaluate(
            state.particle_q,
            self._original.particle_q,
            self._original.particle_qd,
            self._frozen_force,
            self._frozen_gravity,
            dt=dt,
            min_det_f_guard=self._guard,
        )
        if self._tet.flags.numpy()[0]:
            raise _StepFailure(self.Status.TET_INVERSION, "Invalid tet candidate")
        wp.launch(
            _scatter,
            3 * self._tet.ids.size,
            [scalar_view(self._tet.residual), self._layout.x_scalar_offset, self._complete_r],
            device=self.model.device,
        )
        parts = self._tet.parts.numpy()
        total += parts.sum()
        absolute += abs(parts).sum()
    self._temporary_gauges = []
    self._cable_raw_theta_norms = {}
    gravity = self._frozen_gravity.numpy()
    for cloth in self._cloths:
        try:
            e, a = cloth.evaluate(
                state,
                wp.vec3(gravity[int(cloth.reference["world"])]),
                dt,
                self._complete_r,
                build_tangent=build_tangent,
            )
        except ValueError as error:
            raise _StepFailure(self.Status.NONFINITE, str(error)) from error
        total += e
        absolute += a
    for cable in self._cables:
        try:
            e, a = cable.evaluate(
                state,
                self._original,
                self._cable_torque,
                wp.vec3(gravity[int(cable.reference["world"])]),
                dt,
                self._complete_r,
            )
        except ValueError as error:
            raise _StepFailure(self.Status.NONFINITE, str(error)) from error
        total += e
        absolute += a
    reference_mode = self._contact_spatial and self._contact._reference_mode
    if self._contact_pool:
        h = state._monolithic_coupled
        try:
            self._contact._batch.material_surface.update(state)
        except ValueError as error:
            raise _StepFailure(self.Status.NONFINITE, str(error)) from error
        self._contact.evaluate(
            state.particle_q, k.poses, k.com, k.jt, k.jw, dt=dt, rigid_com=h.rigid_com, rigid_rotation=h.rigid_rotation
        )
        diagnostics = {} if reference_mode else self._contact.evaluation_diagnostics()
        if not reference_mode:
            self._variational["contact_pool"] = diagnostics
        if (
            self._contact.flags.numpy()[0]
            or diagnostics.get("dropped", 0)
            or (self._contact_spatial and self._contact._query_dropped)
        ):
            raise _StepFailure(self.Status.NONFINITE, "Invalid or incomplete material contact evaluation")
        wp.launch(
            _add_contact, self._complete_r.size, [self._contact.residual, self._complete_r], device=self.model.device
        )
        total += self._contact.energy.numpy().sum()
        absolute += self._contact.absolute_energy.numpy().sum()
        forces = self._contact.pool.force.result.numpy()
        for block in self._cloths + self._cables:
            raw = block.raw.numpy()
            if reference_mode:
                rows = block.rows_host
                free = rows >= 0
                raw[free] += self._contact.residual.numpy()[rows[free]]
                block.raw.assign(raw)
                continue
            ids = block.ids.numpy()
            raw[: 3 * len(ids)] -= forces[: 3 * self.model.particle_count].reshape(-1, 3)[ids].ravel()
            if block in self._cables:
                offset = (
                    3 * self.model.particle_count + 6 * self.model.body_count + int(block.reference["theta_offset"])
                )
                raw[3 * len(ids) :] -= forces[offset : offset + len(ids) - 1]
            block.raw.assign(raw)
    for cable in self._cables:
        raw = cable.raw.numpy()[3 * cable.ids.size :]
        self._cable_raw_theta_norms[f"cable_{cable.actor.source_index}_theta"] = float(
            np.linalg.norm(raw[~cable.reference["fixed_theta"]])
        )
        if not reference_mode and not cable.reference["fixed_theta"].any():
            jacobian = cable.objective.jacobian.numpy()
            null = np.array_equal(jacobian[:, :, 9], -jacobian[:, :, 10])
            if self._contact_pool and null:
                rows = cable.rows_host[3 * cable.ids.size :]
                records = self._contact.pool.records
                count = int(records.factor_count.numpy()[0])
                offsets = records.factor_offsets.numpy()
                columns = records.factor_columns.numpy()
                values = records.factor_values.numpy()
                weights = records.factor_weights.numpy()
                for factor in range(count):
                    first, end = offsets[factor : factor + 2]
                    if weights[factor] > 0 and values[first:end][np.isin(columns[first:end], rows)].sum() != 0:
                        null = False
                        break
            if bool(cable.reference["gauge"]) or null:
                if abs(raw.sum()) > self._config.merit_absolute_cable_theta:
                    raise _StepFailure(
                        self.Status.LINEAR_NON_POSITIVE_CURVATURE, "Cable common-angle mode has incompatible net torque"
                    )
            if null and not bool(cable.reference["gauge"]):
                self._temporary_gauges.append(cable)
    residual = self._complete_r.numpy().copy()
    if not np.isfinite(residual).all() or not np.isfinite(total):
        raise _StepFailure(self.Status.NONFINITE, "Nonfinite material-scene E/r")
    self._current_state = state
    return total, residual, absolute


def assemble(self, iteration):
    """Publish the complete scalar operator and explicit position/theta unit scaling."""
    self._assembly_sequence += 1
    self._generation = MonolithicLinearGeneration(
        self._step_generation,
        iteration,
        self._contact.evaluation_generation if self._contact_pool else self._step_generation,
        self._assembly_sequence,
        history_epoch=self._step_generation,
    )
    assembly = self._linear.begin_assembly(self._generation)
    nq = self._kinematics.count
    wp.launch(
        kernels._scatter,
        (nq, nq),
        [self._objective.tangent32, assembly.aq_actor_dense, assembly.global_scalar_triplets.values, self._diagonal],
        device=self.model.device,
    )
    if self._has_tet:
        scatter = TetScatterBuffers(
            assembly.ax_internal_triplets.values,
            assembly.global_scalar_triplets.values[nq * nq : nq * nq + 9 * assembly.ax_internal_triplets.capacity],
        )
        assemble_tet_residual_tangent(
            self.model,
            candidate_particle_q=self._current_state.particle_q,
            particle_q_n=self._original.particle_q,
            particle_qd_n=self._original.particle_qd,
            frozen_particle_f=self._frozen_force,
            dynamic_particle_ids=self._tet.ids,
            particle_to_dynamic=self._tet.mapping,
            residual_x=self._tet_r32,
            dt=float(self._dt),
            min_det_f_guard=self._guard,
            workspace=self._tet_assembly,
            scatter=scatter,
            frozen_gravity=self._frozen_gravity,
        )
        if self._tet_assembly.failure_flags.numpy()[0]:
            raise _StepFailure(self.Status.TET_PROJECTION_FAILURE, "Invalid tet tangent")
    wp.launch(
        _scatter_tangent,
        (self._rigid.count, 6, 6),
        [
            self._rigid.tangent,
            self._rigid.rows,
            self._rigid.diagonal,
            self._rigid_triplet_offset,
            assembly.global_scalar_triplets.values,
            self._diagonal,
        ],
        device=self.model.device,
    )
    for block in self._cloths + self._cables:
        block.assemble(assembly, self._dt)
    if self._contact_pool:
        self._contact.assemble_tangent(assembly)
    self._require_linear(self._linear.finalize_assembly(generation=self._generation).status)
    self._require_linear(self._linear.build_coordinate_scaling(self._material_scale, generation=self._generation))
    self._assembly_gauges = tuple(self._temporary_gauges)
    pins = np.full(len(self._cables), -1, dtype=np.int32)
    for i, cable in enumerate(self._cables):
        if cable in self._temporary_gauges:
            pins[i] = cable.rows_host[3 * cable.ids.size]
    if self._linear._cable_preconditioner is not None:
        self._linear._cable_preconditioner.gauge_rows.assign(pins)
    self._metrics["matrix_assembly_count"] += 1
    return None, self._material_scale.astype(self._numpy_dtype)


def accumulate_direction(self, correction):
    """Accumulate FP32 corrections in native precision, then select the null representative."""
    before = self._native_delta.numpy().copy()
    delta = (before + self._correction_delta.numpy()) if correction else self._delta.numpy().astype(self._numpy_dtype)
    for cable in self._assembly_gauges:
        rows = cable.rows_host[3 * cable.ids.size :]
        delta[rows] -= delta[rows].mean(dtype=np.float64)
    if not np.isfinite(delta).all():
        raise _StepFailure(self.Status.NONFINITE, "Nonfinite corrected material direction")
    self._native_delta.assign(delta)
    return not np.array_equal(before, delta) if correction else True
