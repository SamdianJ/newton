# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared production tet tangent assembly, scatter and required residual scratch."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ...sim.model import Model
from .residual import _accumulate_rows, _row_gather_map
from .tet_common import (
    _validate_candidate_particles,
    _validate_evaluation,
    _validate_mapping,
)
from .tet_material import (
    _elastic_state,
    _nodal_hessian,
    _project_element_psd,
    _shape_gradient,
)


@dataclass(frozen=True, slots=True)
class TetTripletPattern:
    ax_rows: np.ndarray
    ax_columns: np.ndarray
    global_rows: np.ndarray
    global_columns: np.ndarray
    elastic_block_slots: np.ndarray
    residual_rows: np.ndarray
    """Compact residual row of each (tet, vertex) term, ``-1`` for fixed particles."""
    dynamic_count: int


@dataclass(slots=True)
class TetAssemblyWorkspace:
    elastic_block_slots: wp.array2d[int]
    residual_offsets: wp.array[int]
    residual_slots: wp.array[int]
    residual_terms: wp.array
    """Per (tet, vertex) residual terms, gathered by row in a fixed order."""
    tet_energy: wp.array
    """Per-element energy in the assembly precision."""
    min_det_f: wp.array
    """Smallest determinant of F in the assembly precision."""
    failure_flags: wp.array[int]
    physics: object = None


@dataclass(frozen=True, slots=True)
class TetScatterBuffers:
    ax_values: wp.array
    """Owner 3x3 blocks, ``mat33`` or ``mat33d`` matching ``global_values``."""
    global_values: wp.array
    """Global scalar triplet values in the linear workspace precision."""


def build_tet_triplet_pattern(
    tet_indices: np.ndarray, particle_to_dynamic: np.ndarray, *, x_dof_start: int
) -> TetTripletPattern:
    """Allocate fixed inertia then row-major elastic slots with no runtime append."""
    count = _validate_mapping(tet_indices, particle_to_dynamic)
    if not isinstance(x_dof_start, (int, np.integer)) or x_dof_start < 0:
        raise ValueError("x_dof_start must be a nonnegative integer")
    rows, columns = list(range(count)), list(range(count))
    slots = np.full((len(tet_indices), 16), -1, dtype=np.int32)
    for tet, vertices in enumerate(tet_indices):
        for a, i in enumerate(particle_to_dynamic[vertices]):
            for b, j in enumerate(particle_to_dynamic[vertices]):
                if i >= 0 and j >= 0:
                    slots[tet, 4 * a + b] = len(rows)
                    rows.append(i)
                    columns.append(j)
    rows, columns = np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)
    global_rows = (x_dof_start + 3 * rows[:, None] + np.repeat(np.arange(3), 3)).reshape(-1).astype(np.int32)
    global_columns = (x_dof_start + 3 * columns[:, None] + np.tile(np.arange(3), 3)).reshape(-1).astype(np.int32)
    residual_rows = particle_to_dynamic[tet_indices].reshape(-1).astype(np.int32)
    for array in (rows, columns, global_rows, global_columns, slots, residual_rows):
        array.setflags(write=False)
    return TetTripletPattern(rows, columns, global_rows, global_columns, slots, residual_rows, count)


def create_tet_assembly_workspace(
    pattern: TetTripletPattern, *, tet_count: int, device: wp.DeviceLike, dtype, physics=None
) -> TetAssemblyWorkspace:
    """Upload the fixed slot map and allocate reusable evaluation diagnostics in ``dtype``."""
    if pattern.elastic_block_slots.shape != (tet_count, 16):
        raise ValueError("tet_count disagrees with elastic slot map")
    if dtype not in (wp.float32, wp.float64):
        raise ValueError("Tet assembly requires float32 or float64")
    offsets, slots = _row_gather_map(pattern.residual_rows, pattern.dynamic_count, device)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    return TetAssemblyWorkspace(
        wp.array(pattern.elastic_block_slots, dtype=int, device=device),
        offsets,
        slots,
        wp.zeros(4 * tet_count, dtype=vector, device=device),
        wp.zeros(tet_count, dtype=dtype, device=device),
        wp.zeros(1, dtype=dtype, device=device),
        wp.zeros(1, dtype=int, device=device),
        physics,
    )


@wp.func
def _scatter_internal_block(slot: int, block: Any, ax_values: wp.array[Any], global_values: wp.array[Any]):
    ax_values[slot] = block
    for row in range(3):
        for col in range(3):
            global_values[9 * slot + 3 * row + col] = block[row, col]


@wp.kernel
def _assemble_inertia_tangent(
    inv_dt_sq: Any,
    particle_mass: wp.array[float],
    dynamic_particle_ids: wp.array[int],
    ax_values: wp.array[Any],
    global_values: wp.array[Any],
):
    index = wp.tid()
    mass = type(inv_dt_sq)(particle_mass[dynamic_particle_ids[index]])
    _scatter_internal_block(
        index, wp.identity(n=3, dtype=type(inv_dt_sq)) * (mass * inv_dt_sq), ax_values, global_values
    )


@wp.kernel
def _initialize_consistent_residual(
    gravity: wp.array[wp.vec3],
    worlds: wp.array[int],
    row_mass: wp.array[float],
    forces: wp.array[wp.vec3],
    dynamic_ids: wp.array[int],
    residual: wp.array[Any],
):
    i = wp.tid()
    particle = dynamic_ids[i]
    world = worlds[particle]
    if world < 0:
        world = gravity.shape[0] - 1
    mass = type(residual[0][0])(row_mass[particle])
    residual[i] = -mass * residual.dtype(gravity[world]) - residual.dtype(forces[particle])


@wp.kernel
def _add_consistent_inertia(
    dt: Any,
    indices: wp.array2d[int],
    mapping: wp.array[int],
    coefficients: wp.array[float],
    x: wp.array[Any],
    xn: wp.array[Any],
    vn: wp.array[Any],
    terms: wp.array[Any],
):
    tet, a = wp.tid()
    if mapping[indices[tet, a]] >= 0:
        r = terms.dtype()
        for b in range(4):
            j = indices[tet, b]
            weight = type(dt)(coefficients[tet])
            if a == b:
                weight *= type(dt)(2.0)
            r += weight * (x[j] - xn[j] - dt * vn[j]) / (dt * dt)
        terms[4 * tet + a] = r


@wp.kernel
def _evaluate_inertia_residual(
    dt: Any,
    gravity: wp.array[wp.vec3],
    worlds: wp.array[int],
    xn: wp.array[Any],
    vn: wp.array[Any],
    x: wp.array[Any],
    mass: wp.array[float],
    forces: wp.array[wp.vec3],
    ids: wp.array[int],
    residual: wp.array[Any],
):
    i = wp.tid()
    p = ids[i]
    world = worlds[p]
    if world < 0:
        world = gravity.shape[0] - 1
    residual[i] = type(dt)(mass[p]) * (
        (x[p] - xn[p] - dt * vn[p]) / (dt * dt) - residual.dtype(gravity[world])
    ) - residual.dtype(forces[p])


@wp.kernel
def _assemble_elastic_residual_tangent(
    material_model: int,
    consistent: int,
    inv_dt_sq: Any,
    epsilon: Any,
    mass_coefficients: wp.array[float],
    min_det_f_guard: Any,
    candidate_particle_q: wp.array[Any],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_to_dynamic: wp.array[int],
    elastic_block_slots: wp.array2d[int],
    residual_terms: wp.array[Any],
    ax_values: wp.array[Any],
    global_values: wp.array[Any],
    tet_energy: wp.array[Any],
    min_det_f: wp.array[Any],
    failure_flags: wp.array[int],
):
    tet = wp.tid()
    valid, stress, projected = _elastic_state(
        tet,
        material_model,
        min_det_f_guard,
        candidate_particle_q,
        tet_indices,
        tet_poses,
        tet_materials,
        tet_energy,
        min_det_f,
        failure_flags,
        True,
    )
    if valid:
        pose = wp.matrix(tet_poses[tet], shape=(3, 3), dtype=type(inv_dt_sq))
        nodal = wp.matrix(shape=(12, 12), dtype=type(inv_dt_sq))
        if material_model == 1:
            nodal, converged = _project_element_psd(_nodal_hessian(projected, pose), epsilon)
            if not converged:
                wp.atomic_max(failure_flags, 0, 5)
                return
        for a in range(4):
            ga = _shape_gradient(pose, a)
            index = particle_to_dynamic[tet_indices[tet, a]]
            if index >= 0:
                residual_terms[4 * tet + a] += stress * ga
            for b in range(a, 4):
                slot = elastic_block_slots[tet, 4 * a + b]
                if slot >= 0:
                    transpose_slot = elastic_block_slots[tet, 4 * b + a]
                    if slot >= ax_values.shape[0] or transpose_slot < 0 or transpose_slot >= ax_values.shape[0]:
                        wp.atomic_max(failure_flags, 0, 4)
                        continue
                    gb = _shape_gradient(pose, b)
                    block = wp.matrix(shape=(3, 3), dtype=type(inv_dt_sq))
                    if material_model == 1:
                        for row in range(3):
                            for col in range(3):
                                block[row, col] = nodal[3 * a + row, 3 * b + col]
                    else:
                        for row in range(3):
                            for col in range(3):
                                for i in range(3):
                                    for j in range(3):
                                        block[row, col] += ga[i] * projected[3 * i + row, 3 * j + col] * gb[j]
                    if consistent == 1:
                        coefficient = type(inv_dt_sq)(mass_coefficients[tet]) * inv_dt_sq
                        if a == b:
                            coefficient *= type(inv_dt_sq)(2.0)
                        block += coefficient * wp.identity(n=3, dtype=type(inv_dt_sq))
                    _scatter_internal_block(slot, block, ax_values, global_values)
                    if a != b:
                        _scatter_internal_block(transpose_slot, wp.transpose(block), ax_values, global_values)


@wp.kernel
def _check_residual_finite(residual: wp.array[Any], failure_flags: wp.array[int]):
    index = wp.tid()
    for axis in range(3):
        if not wp.isfinite(residual[index][axis]):
            wp.atomic_max(failure_flags, 0, 2)


@wp.kernel
def _check_scatter_finite(values: wp.array[Any], failure_flags: wp.array[int]):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_max(failure_flags, 0, 2)


def _prepare_evaluation(model, args):
    native = _validate_evaluation(model, args)
    scalar = wp.float64 if native == np.float64 else wp.float32
    dt, workspace = scalar(native(args["dt"])), args["workspace"]
    workspace.failure_flags.zero_()
    workspace.min_det_f.fill_(float("inf"))
    workspace.tet_energy.zero_()
    workspace.residual_terms.zero_()
    physics = workspace.physics
    wp.launch(
        _validate_candidate_particles,
        (model.tet_count, 4),
        [
            args["candidate_particle_q"],
            args["particle_q_n"],
            args["particle_qd_n"],
            args["frozen_particle_f"],
            physics.indices if physics is not None else model.tet_indices,
            args["particle_to_dynamic"],
            workspace.failure_flags,
        ],
        device=model.device,
    )
    gravity = args.get("frozen_gravity")
    gravity = model.gravity if gravity is None else gravity
    if physics is not None and physics.mass_mode == "consistent":
        wp.launch(
            _initialize_consistent_residual,
            args["dynamic_particle_ids"].size,
            [
                gravity,
                physics.worlds if physics is not None else model.particle_world,
                physics.row_mass,
                args["frozen_particle_f"],
                args["dynamic_particle_ids"],
                args["residual_x"],
            ],
            device=model.device,
        )
        wp.launch(
            _add_consistent_inertia,
            (model.tet_count, 4),
            [
                dt,
                physics.indices if physics is not None else model.tet_indices,
                args["particle_to_dynamic"],
                physics.coefficients,
                args["candidate_particle_q"],
                args["particle_q_n"],
                args["particle_qd_n"],
                workspace.residual_terms,
            ],
            device=model.device,
        )
    else:
        wp.launch(
            _evaluate_inertia_residual,
            args["dynamic_particle_ids"].size,
            [
                dt,
                gravity,
                physics.worlds if physics is not None else model.particle_world,
                args["particle_q_n"],
                args["particle_qd_n"],
                args["candidate_particle_q"],
                physics.mass if physics is not None else model.particle_mass,
                args["frozen_particle_f"],
                args["dynamic_particle_ids"],
                args["residual_x"],
            ],
            device=model.device,
        )
    return native


def assemble_tet_residual_tangent(
    model: Model,
    *,
    candidate_particle_q: wp.array[Any],
    particle_q_n: wp.array[Any],
    particle_qd_n: wp.array[Any],
    frozen_particle_f: wp.array[wp.vec3],
    dynamic_particle_ids: wp.array[int],
    particle_to_dynamic: wp.array[int],
    residual_x: wp.array[Any],
    dt: float,
    min_det_f_guard: float,
    workspace: TetAssemblyWorkspace,
    scatter: TetScatterBuffers,
    frozen_gravity: wp.array[wp.vec3] | None = None,
) -> None:
    """Write exact residual and identical projected/inertial owner and global blocks.

    Positions, residual, workspace diagnostics and scatter values share one
    precision, in which every element quantity is evaluated.
    """
    scalar = wp.float64 if candidate_particle_q.dtype == wp.vec3d else wp.float32
    matrix = wp.mat33d if scalar == wp.float64 else wp.mat33
    if (
        (scatter.ax_values.dtype, scatter.global_values.dtype) != (matrix, scalar)
        or scatter.ax_values.ndim != 1
        or scatter.global_values.ndim != 1
        or scatter.ax_values.device != model.device
        or scatter.global_values.device != model.device
        or scatter.global_values.size != 9 * scatter.ax_values.size
        or scatter.ax_values.size < dynamic_particle_ids.size
    ):
        raise ValueError("tet scatter requires matching block/scalar slices of one precision on the model device")
    if (
        workspace.residual_terms.dtype != residual_x.dtype
        or workspace.residual_terms.size != 4 * model.tet_count
        or workspace.residual_offsets.size != residual_x.size + 1
    ):
        raise ValueError("tet workspace residual terms must match the residual precision and rows")
    physics = workspace.physics
    mode = int(physics is not None and physics.material_model == "smith_log_stabilized")
    poses = physics.poses if physics is not None else model.tet_poses
    materials = physics.materials if physics is not None else model.tet_materials
    native = _prepare_evaluation(model, locals())
    inv_dt_sq = native(1.0 / np.float64(dt) ** 2)
    scatter.ax_values.zero_()
    scatter.global_values.zero_()
    consistent = int(physics is not None and physics.mass_mode == "consistent")
    if not consistent:
        wp.launch(
            _assemble_inertia_tangent,
            dynamic_particle_ids.size,
            [
                scalar(inv_dt_sq),
                physics.mass if physics is not None else model.particle_mass,
                dynamic_particle_ids,
                scatter.ax_values,
                scatter.global_values,
            ],
            device=model.device,
        )
    wp.launch(
        _assemble_elastic_residual_tangent,
        model.tet_count,
        [
            mode,
            consistent,
            scalar(inv_dt_sq),
            scalar(np.finfo(native).eps),
            physics.coefficients if physics is not None else model.particle_mass,
            scalar(native(min_det_f_guard)),
            candidate_particle_q,
            physics.indices if physics is not None else model.tet_indices,
            poses,
            materials,
            particle_to_dynamic,
            workspace.elastic_block_slots,
            workspace.residual_terms,
            scatter.ax_values,
            scatter.global_values,
            workspace.tet_energy,
            workspace.min_det_f,
            workspace.failure_flags,
        ],
        device=model.device,
    )
    # Element terms reach each row in a fixed order, independent of thread scheduling.
    wp.launch(
        _accumulate_rows,
        residual_x.size,
        [workspace.residual_offsets, workspace.residual_slots, workspace.residual_terms, residual_x],
        device=model.device,
    )
    wp.launch(_check_residual_finite, residual_x.size, [residual_x, workspace.failure_flags], device=model.device)
    wp.launch(
        _check_scatter_finite,
        scatter.global_values.size,
        [scatter.global_values, workspace.failure_flags],
        device=model.device,
    )
