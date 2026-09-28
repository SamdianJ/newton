# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Historical residual-only tet entry; shared P3 components live in tet_common/material/assembly."""

# Transitional imports remain until the historical parent is retired.
# ruff: noqa: F401

from typing import Any

import warp as wp

from ...sim.model import Model
from .tet_assembly import (
    TetAssemblyWorkspace,
    TetScatterBuffers,
    TetTripletPattern,
    _add_consistent_inertia,
    _add_consistent_inertia64,
    _assemble_elastic_residual_tangent,
    _assemble_inertia_tangent,
    _check_residual_finite,
    _check_scatter_finite,
    _evaluate_inertia_residual,
    _evaluate_inertia_residual64,
    _initialize_consistent_residual,
    _prepare_evaluation,
    _scatter_internal_block,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from .tet_common import (
    TetEvaluationStatus,
    TetPhysicsWorkspace,
    _validate_candidate_particles,
    _validate_evaluation,
    _validate_mapping,
    validate_tet_scope,
)
from .tet_material import (
    _compute_cofactor,
    _compute_cofactor_derivative,
    _compute_rest_volume,
    _elastic_state,
    _evaluate_energy_stress64,
    _evaluate_rest_volumes,
    _evaluate_smith,
    _evaluate_stable_neo_hookean,
    _nodal_hessian,
    _project_element_psd,
    _shape_gradient,
    mat99,
    mat1212,
)


@wp.kernel
def _evaluate_elastic_residual(
    material_model: int,
    high_precision: bool,
    min_det_f_guard: float,
    candidate_particle_q: wp.array[Any],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_to_dynamic: wp.array[int],
    residual_x: wp.array[wp.vec3],
    tet_energy: wp.array[float],
    min_det_f: wp.array[float],
    failure_flags: wp.array[int],
):
    tet = wp.tid()
    valid, stress, _projected = _elastic_state(
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
        False,
        high_precision,
    )
    if valid:
        for a in range(4):
            index = particle_to_dynamic[tet_indices[tet, a]]
            if index >= 0:
                wp.atomic_add(residual_x, index, stress * _shape_gradient(tet_poses[tet], a))


def evaluate_tet_residual(
    model: Model,
    *,
    candidate_particle_q: wp.array[Any],
    particle_q_n: wp.array[Any],
    particle_qd_n: wp.array[Any],
    frozen_particle_f: wp.array[wp.vec3],
    dynamic_particle_ids: wp.array[int],
    particle_to_dynamic: wp.array[int],
    residual_x: wp.array[wp.vec3],
    dt: float,
    min_det_f_guard: float,
    workspace: TetAssemblyWorkspace,
    frozen_gravity: wp.array[wp.vec3] | None = None,
) -> None:
    """Evaluate a trial without accessing tangent buffers; inspect failure_flags."""
    physics = workspace.physics
    mode = int(physics is not None and physics.material_model == "smith_log_stabilized")
    poses = physics.poses if physics is not None else model.tet_poses
    materials = physics.materials if physics is not None else model.tet_materials
    _prepare_evaluation(model, locals())
    wp.launch(
        _evaluate_elastic_residual,
        model.tet_count,
        [
            mode,
            candidate_particle_q.dtype == wp.vec3d,
            min_det_f_guard,
            candidate_particle_q,
            physics.indices if physics is not None else model.tet_indices,
            poses,
            materials,
            particle_to_dynamic,
            residual_x,
            workspace.tet_energy,
            workspace.min_det_f,
            workspace.failure_flags,
        ],
        device=model.device,
    )
    wp.launch(_check_residual_finite, residual_x.size, [residual_x, workspace.failure_flags], device=model.device)
