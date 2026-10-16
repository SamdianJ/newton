# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material producers for the existing monolithic iteration and publication path."""

from dataclasses import replace
from typing import Any

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from . import coupled_kernels as kernels
from .articulation import ArticulationObjective
from .cable_system import CableBlock
from .cloth_system import ClothBlock
from .linear import MonolithicLinearWorkspace
from .linear_types import MonolithicLinearCapacities, MonolithicLinearGeneration
from .rigid import _scatter_residual, _scatter_tangent
from .segment_graph import _SegmentGraphCache
from .solver_types import _SolverMonolithicInternalConfig, _StepFailure
from .state_checks import scalar_view
from .tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from .tet_precision import TetObjective


@wp.kernel
def _scatter(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = values[i]


def initialize(self):
    """Allocate material producers once; retain the existing Newton/PCG driver."""
    config, model, layout = self._material_config, self.model, self._layout
    if np.any(model.shape_flags.numpy() & (int(ShapeFlags.COLLIDE_PARTICLES) | int(ShapeFlags.COLLIDE_SHAPES))):
        raise NotImplementedError("Material-scene contact assembly belongs to the contact integration")
    device, dtype = model.device, self._dtype
    self._cloths = []
    self._cables = []
    for actor in self._actors.actors:
        if actor.kind == "cloth":
            obj = self._materials.cloth[actor.source_index]
            self._cloths.append(ClothBlock(obj, actor, dtype=dtype, device=device))
        if actor.kind == "cable":
            obj = self._materials.cable[actor.source_index]
            phi_offset = sum(len(c["ref_twist"]) for c in self._materials.cable[: actor.source_index])
            self._cables.append(CableBlock(obj, actor, dtype=dtype, device=device, phi_offset=phi_offset))
    parameters = self._joint_terms.params.numpy()
    self._objective = ArticulationObjective(self._kinematics, parameters)
    self._joint_parameters_host = parameters.astype(self._numpy_dtype)
    self._globalization = config["globalization"]
    self._guard = config["min_det_f_guard"]
    self._config = replace(
        _SolverMonolithicInternalConfig(),
        regularization_values=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0),
        merit_absolute_cable_x=config["cable_position_tolerance"],
        merit_absolute_cable_theta=config["cable_theta_tolerance"],
        merit_absolute_global=1e-7,
        merit_absolute_q=2.5e-8,
        merit_absolute_x=1e-7,
        merit_relative_global=1e-6,
        merit_relative_q=1e-5,
        merit_relative_x=1e-6,
        step_tolerance_global=1e-10,
        step_tolerance_x=1e-10,
    )
    if config["fp32_absolute_tolerance"] != "default":
        tolerance = config["fp32_absolute_tolerance"]
        self._config = replace(
            self._config,
            merit_absolute_global=tolerance,
            merit_absolute_q=tolerance,
            merit_absolute_x=tolerance,
        )
    self._graphs = _SegmentGraphCache(device, enabled=self._use_cuda_graph)
    self._states._graphs = self._graphs
    self._contact = None
    self._contact_pool = self._contact_spatial = self._device_iteration = False
    self._original = self._accepted = self._trial = self._out_backup = None
    self._step_generation = self._assembly_sequence = 0
    self._dt = None
    self._old_joint_t = wp.empty_like(self._kinematics.joint_translations)
    self._old_joint_r = wp.empty_like(self._kinematics.joint_rotations)
    self._input_status = wp.zeros(4, dtype=int, device=device)
    self._force_status = wp.zeros(3, dtype=int, device=device)
    self._evaluation_status = wp.zeros(1, dtype=int, device=device)
    self._tet = None
    nq = layout.q_dof_count
    global_rows = list(np.repeat(layout.indices("articulation"), nq))
    global_columns = list(np.tile(layout.indices("articulation"), nq))
    internal_rows = internal_columns = np.empty(0, dtype=np.int32)
    tet_actor = next((a for a in self._actors.actors if a.kind == "tet"), None)
    mapping = np.full(model.particle_count, -1, dtype=np.int32)
    self._dynamic_ids_host = np.empty(0, dtype=np.int32)
    if tet_actor is not None:
        density = config["tet_rest_density"]
        self._tet = TetObjective(
            model,
            material_model=config["material_model"],
            mass_mode=config["mass_mode"],
            density=None if isinstance(density, str) else wp.array(density, dtype=float, device=device),
            dtype=dtype,
            actor=tet_actor,
            allow_cloth=bool(self._materials.cloth),
        )
        mapping = self._tet.mapping.numpy()
        self._dynamic_ids_host = self._tet.ids.numpy()
        pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), mapping, x_dof_start=layout.x_scalar_offset)
        global_rows.extend(pattern.global_rows)
        global_columns.extend(pattern.global_columns)
        internal_rows, internal_columns = pattern.ax_rows, pattern.ax_columns
        self._tet_assembly = create_tet_assembly_workspace(
            pattern, tet_count=model.tet_count, device=device, physics=self._tet.physics
        )
        self._tet_r32 = wp.zeros(layout.dynamic_particle_count, dtype=wp.vec3, device=device)
    self._particle_mapping = wp.array(mapping, dtype=int, device=device)
    self._rigid_triplet_offset = len(global_rows)
    global_rows.extend(np.repeat(self._rigid.rows_numpy, 6, axis=1).ravel())
    global_columns.extend(np.tile(self._rigid.rows_numpy, (1, 6)).ravel())
    for block in self._cloths + self._cables:
        block.triplet_offset = len(global_rows)
        global_rows.extend(block.triplet_rows)
        global_columns.extend(block.triplet_columns)
    self._linear = MonolithicLinearWorkspace(
        layout,
        MonolithicLinearCapacities(
            len(global_rows), len(internal_rows), 0, self.linear_max_iterations, self.linear_max_iterations + 2
        ),
        self._particle_mapping,
        device=device,
        pcg_mode=self.pcg_mode.value,
        use_cuda_graph=self._use_cuda_graph,
        cable_row_order=np.concatenate([c.order for c in self._cables])
        if self._cables
        else np.empty(0, dtype=np.int32),
    )
    self._linear._set_fixed_triplet_pattern(
        global_rows=np.asarray(global_rows, dtype=np.int32),
        global_columns=np.asarray(global_columns, dtype=np.int32),
        internal_rows=internal_rows,
        internal_columns=internal_columns,
    )
    for name in ("_rhs", "_y", "_delta", "_diagonal", "_correction_rhs", "_correction_delta", "_direction_base"):
        setattr(self, name, wp.zeros(layout.scalar_dof_count, dtype=float, device=device))
    for name in ("_native_delta", "_complete_r", "_post_r", "_post_error", "_post_original", "_post_regularization"):
        setattr(self, name, wp.zeros(layout.scalar_dof_count, dtype=dtype, device=device))
    self._post_lambda = wp.zeros(1, dtype=float, device=device)
    self._direction_status = wp.zeros(1, dtype=int, device=device)
    self._frozen_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
    self._frozen_gravity = wp.zeros_like(model.gravity)
    self._cable_target_x = wp.array(model.particle_q.numpy(), dtype=self._states.vector, device=device)
    self._cable_target_theta = wp.zeros(self._materials.theta_count, dtype=dtype, device=device)
    self._cable_torque = wp.zeros_like(self._cable_target_theta)
    self._material_scale = np.ones(layout.scalar_dof_count, dtype=np.float32)
    for cable in self._cables:
        rows = cable.rows_host[3 * cable.ids.size :]
        self._material_scale[rows[rows >= 0]] = 1 / float(cable.reference["length_scale"])
    self._temporary_gauges = []


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
    if self._tet is not None:
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
        raw = cable.raw.numpy()[3 * cable.ids.size :]
        self._cable_raw_theta_norms[f"cable_{cable.actor.source_index}_theta"] = float(
            np.linalg.norm(raw[~cable.reference["fixed_theta"]])
        )
        if not cable.reference["fixed_theta"].any():
            jacobian = cable.objective.jacobian.numpy()
            null = np.array_equal(jacobian[:, :, 9], -jacobian[:, :, 10])
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
        self._step_generation,
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
    if self._tet is not None:
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
