# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P1 tet exact residual and projected Gauss-Newton tangent.

The constitutive mapping is the Kim stable Neo-Hookean variant in SuperDex
54ae749a, ``mochi_core/materials/batched_kim_neo_hookean.h``: standard Lamé
parameters become ``mu`` and ``lambda + mu``. This is the variant without the
Smith logarithmic term. Its energy is shifted to zero at the rest state.
The production tangent drops the cofactor geometric derivative; it is not the
exact residual derivative or a spectral projection of the raw Hessian.
"""

import enum
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim.model import Model

mat99 = wp.types.matrix(shape=(9, 9), dtype=wp.float32)


class TetEvaluationStatus(enum.IntEnum):
    SUCCESS = 0
    DET_F_GUARD = 1
    NONFINITE = 2
    INVALID_FIXED_PARTICLE = 3
    INVALID_SCATTER = 4


@dataclass(frozen=True, slots=True)
class TetTripletPattern:
    ax_rows: np.ndarray
    ax_columns: np.ndarray
    global_rows: np.ndarray
    global_columns: np.ndarray
    elastic_block_slots: np.ndarray


@dataclass(slots=True)
class TetAssemblyWorkspace:
    elastic_block_slots: wp.array2d[int]
    tet_energy: wp.array[float]
    min_det_f: wp.array[float]
    failure_flags: wp.array[int]


@dataclass(frozen=True, slots=True)
class TetScatterBuffers:
    ax_values: wp.array[wp.mat33]
    global_values: wp.array[float]


def _validate_mapping(tet_indices: np.ndarray, particle_to_dynamic: np.ndarray) -> int:
    if tet_indices.ndim != 2 or tet_indices.shape[1] != 4 or not np.issubdtype(tet_indices.dtype, np.integer):
        raise ValueError("tet_indices must have integer shape [tet_count, 4]")
    if particle_to_dynamic.ndim != 1 or not np.issubdtype(particle_to_dynamic.dtype, np.integer):
        raise ValueError("particle_to_dynamic must be a one-dimensional integer array")
    if np.any(tet_indices < 0) or np.any(tet_indices >= len(particle_to_dynamic)):
        raise ValueError("tet particle index out of range")
    if np.any(np.diff(np.sort(tet_indices, axis=1), axis=1) == 0):
        raise ValueError("tet contains repeated particle indices")
    dynamic = particle_to_dynamic[particle_to_dynamic >= 0]
    if np.any(particle_to_dynamic < -1) or not np.array_equal(np.sort(dynamic), np.arange(len(dynamic))):
        raise ValueError("particle_to_dynamic must define a bijective compact map with fixed sentinel -1")
    if np.any(particle_to_dynamic[np.setdiff1d(np.arange(len(particle_to_dynamic)), tet_indices)] >= 0):
        raise ValueError("dynamic particle is not referenced by the target tets")
    return len(dynamic)


def validate_tet_scope(model: Model, *, dynamic_particle_ids: np.ndarray, particle_to_dynamic: np.ndarray) -> None:
    """Validate material, rest data and lumped mass; collision owns the boundary."""
    indices = model.tet_indices.numpy()
    count = _validate_mapping(indices, particle_to_dynamic)
    if not len(indices) or particle_to_dynamic.shape != (model.particle_count,):
        raise ValueError("tet scope must contain tetrahedra and a full particle map")
    if dynamic_particle_ids.shape != (count,) or not np.issubdtype(dynamic_particle_ids.dtype, np.integer):
        raise ValueError("dynamic_particle_ids must contain every compact dynamic particle")
    if np.any(dynamic_particle_ids < 0) or np.any(dynamic_particle_ids >= model.particle_count):
        raise ValueError("dynamic particle index out of range")
    if not np.array_equal(particle_to_dynamic[dynamic_particle_ids], np.arange(count)):
        raise ValueError("dynamic_particle_ids and particle_to_dynamic disagree")
    referenced = np.unique(indices)
    mass, inv_mass = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
    if not np.all(np.isfinite(mass[referenced])) or not np.all(np.isfinite(inv_mass[referenced])):
        raise ValueError("tet particle mass must be finite")
    if np.any(mass[referenced] < 0) or np.any(inv_mass[referenced] < 0):
        raise ValueError("tet particle mass must be nonnegative")
    expected = referenced[inv_mass[referenced] > 0]
    if not np.array_equal(np.sort(dynamic_particle_ids), expected):
        raise ValueError("dynamic mapping disagrees with particle_inv_mass")
    if np.any(mass[expected] <= 0) or not np.allclose(mass[expected] * inv_mass[expected], 1, rtol=2e-6, atol=0):
        raise ValueError("dynamic lumped mass and inverse mass must be reciprocal")
    if np.any((model.particle_flags.numpy()[expected] & ParticleFlags.ACTIVE) == 0):
        raise ValueError("dynamic tet particles must be active")
    positions, velocities = model.particle_q.numpy(), model.particle_qd.numpy()
    if not np.all(np.isfinite(positions[referenced])) or not np.all(np.isfinite(velocities[referenced])):
        raise ValueError("tet initial positions and velocities must be finite")
    fixed = referenced[inv_mass[referenced] == 0]
    if np.any(velocities[fixed] != 0):
        raise ValueError("fixed referenced particle velocity must be exactly zero")
    poses, materials = model.tet_poses.numpy(), model.tet_materials.numpy()
    if poses.shape != (model.tet_count, 3, 3) or not np.all(np.isfinite(poses)):
        raise ValueError("tet rest poses must be finite inverse rest matrices")
    if np.any(np.linalg.det(poses.astype(np.float64)) <= 0):
        raise ValueError("tet rest volume must be positive before runtime rounding")
    # Validate the actual device expression: a float64 determinant can hide
    # float32 intermediate overflow and incorrectly admit a zero runtime volume.
    device_volumes = wp.empty(model.tet_count, dtype=float, device=model.device)
    wp.launch(_evaluate_rest_volumes, model.tet_count, [model.tet_poses, device_volumes], device=model.device)
    volumes = device_volumes.numpy()
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("tet rest volume must be finite and positive in runtime float32 arithmetic")
    if materials.shape != (model.tet_count, 3) or not np.all(np.isfinite(materials)):
        raise ValueError("tet materials must contain finite mu, lambda, damping")
    lambda_nh = materials[:, 0].astype(np.float64) + materials[:, 1]
    if np.any(materials[:, 0] < 0) or np.any(lambda_nh <= 0) or np.any(lambda_nh > np.finfo(np.float32).max):
        raise ValueError("stable Neo-Hookean requires mu >= 0 and lambda + mu > 0")
    if np.any(materials[:, 2] != 0) or np.any(model.tet_activations.numpy() != 0):
        raise ValueError("tet damping and activation are unsupported")
    for name in ("tri_materials", "edge_bending_properties"):
        values = getattr(model, name)
        if values is not None and np.any(values.numpy() != 0):
            raise ValueError(f"{name} must be zero for the tet-only constitutive scope")
    if model.spring_count:
        raise ValueError("particle springs are unsupported")
    worlds = model.particle_world.numpy()[referenced]
    if len(np.unique(worlds)) != 1 or np.any(worlds < -1) or np.any(worlds >= model.world_count):
        raise ValueError("tet particles must share one valid effective world label")
    if not np.all(np.isfinite(model.gravity.numpy()[worlds])):
        raise ValueError("tet gravity must be finite")


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
    for array in (rows, columns, global_rows, global_columns, slots):
        array.setflags(write=False)
    return TetTripletPattern(rows, columns, global_rows, global_columns, slots)


def create_tet_assembly_workspace(
    pattern: TetTripletPattern, *, tet_count: int, device: wp.DeviceLike
) -> TetAssemblyWorkspace:
    """Upload the fixed slot map and allocate reusable evaluation diagnostics."""
    if pattern.elastic_block_slots.shape != (tet_count, 16):
        raise ValueError("tet_count disagrees with elastic slot map")
    return TetAssemblyWorkspace(
        wp.array(pattern.elastic_block_slots, dtype=int, device=device),
        wp.zeros(tet_count, dtype=float, device=device),
        wp.zeros(1, dtype=float, device=device),
        wp.zeros(1, dtype=int, device=device),
    )


@wp.func
def _compute_rest_volume(inverse_rest: wp.mat33) -> float:
    return 1.0 / (6.0 * wp.determinant(inverse_rest))


@wp.kernel
def _evaluate_rest_volumes(poses: wp.array[wp.mat33], volumes: wp.array[float]):
    tet = wp.tid()
    volumes[tet] = _compute_rest_volume(poses[tet])


@wp.func
def _compute_cofactor(F: wp.mat33) -> wp.mat33:
    return wp.matrix_from_cols(wp.cross(F[:, 1], F[:, 2]), wp.cross(F[:, 2], F[:, 0]), wp.cross(F[:, 0], F[:, 1]))


@wp.func
def _compute_cofactor_derivative(F: wp.mat33) -> mat99:
    # Column-major vec(F); this exact geometric derivative is used only by tests.
    result = mat99(0.0)
    for j in range(9):
        delta = wp.mat33(0.0)
        delta[j % 3, j // 3] = 1.0
        derivative = wp.matrix_from_cols(
            wp.cross(delta[:, 1], F[:, 2]) + wp.cross(F[:, 1], delta[:, 2]),
            wp.cross(delta[:, 2], F[:, 0]) + wp.cross(F[:, 2], delta[:, 0]),
            wp.cross(delta[:, 0], F[:, 1]) + wp.cross(F[:, 0], delta[:, 1]),
        )
        for i in range(9):
            result[i, j] = derivative[i % 3, i // 3]
    return result


@wp.func
def _evaluate_stable_neo_hookean(F: wp.mat33, rest_volume: float, mu_lame: float, lambda_lame: float):
    j_minus_one = wp.determinant(F) - 1.0
    lambda_nh = lambda_lame + mu_lame
    cofactor = _compute_cofactor(F)
    squared_norm = float(0.0)
    projected = mat99(0.0)
    for i in range(9):
        squared_norm += F[i % 3, i // 3] * F[i % 3, i // 3]
        for j in range(9):
            projected[i, j] = rest_volume * lambda_nh * cofactor[i % 3, i // 3] * cofactor[j % 3, j // 3]
        projected[i, i] += rest_volume * mu_lame
    energy = rest_volume * (
        0.5 * mu_lame * (squared_norm - 3.0) - mu_lame * j_minus_one + 0.5 * lambda_nh * j_minus_one * j_minus_one
    )
    stress = rest_volume * (mu_lame * F + (lambda_nh * j_minus_one - mu_lame) * cofactor)
    return energy, stress, projected


@wp.func
def _shape_gradient(inverse_rest: wp.mat33, node: int) -> wp.vec3:
    gradient = wp.vec3(0.0)
    if node == 0:
        gradient = -(inverse_rest[0, :] + inverse_rest[1, :] + inverse_rest[2, :])
    else:
        gradient = inverse_rest[node - 1, :]
    return gradient


@wp.func
def _scatter_internal_block(slot: int, block: wp.mat33, ax_values: wp.array[wp.mat33], global_values: wp.array[float]):
    ax_values[slot] = block
    for row in range(3):
        for col in range(3):
            global_values[9 * slot + 3 * row + col] = block[row, col]


@wp.kernel
def _validate_candidate_particles(
    candidate: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    indices: wp.array2d[int],
    mapping: wp.array[int],
    failure_flags: wp.array[int],
):
    tet, node = wp.tid()
    particle = indices[tet, node]
    for axis in range(3):
        if (
            not wp.isfinite(candidate[particle][axis])
            or not wp.isfinite(positions[particle][axis])
            or not wp.isfinite(velocities[particle][axis])
            or not wp.isfinite(forces[particle][axis])
        ):
            wp.atomic_max(failure_flags, 0, 2)
        if mapping[particle] == -1:
            if velocities[particle][axis] != 0.0 or candidate[particle][axis] != positions[particle][axis]:
                wp.atomic_max(failure_flags, 0, 3)


@wp.kernel
def _evaluate_inertia_residual(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[int],
    particle_q_n: wp.array[wp.vec3],
    particle_qd_n: wp.array[wp.vec3],
    candidate_particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_f: wp.array[wp.vec3],
    dynamic_particle_ids: wp.array[int],
    residual_x: wp.array[wp.vec3],
):
    index = wp.tid()
    particle = dynamic_particle_ids[index]
    world = particle_world[particle]
    if world < 0:
        world = gravity.shape[0] - 1
    residual_x[index] = (
        particle_mass[particle]
        * (
            (candidate_particle_q[particle] - particle_q_n[particle]) / (dt * dt)
            - particle_qd_n[particle] / dt
            - gravity[world]
        )
        - particle_f[particle]
    )


@wp.kernel
def _assemble_inertia_tangent(
    inv_dt_sq: float,
    particle_mass: wp.array[float],
    dynamic_particle_ids: wp.array[int],
    ax_values: wp.array[wp.mat33],
    global_values: wp.array[float],
):
    index = wp.tid()
    _scatter_internal_block(
        index,
        wp.identity(n=3, dtype=float) * (particle_mass[dynamic_particle_ids[index]] * inv_dt_sq),
        ax_values,
        global_values,
    )


@wp.func
def _elastic_state(
    tet: int,
    min_det_f_guard: float,
    candidate_particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    tet_energy: wp.array[float],
    min_det_f: wp.array[float],
    failure_flags: wp.array[int],
):
    origin = candidate_particle_q[tet_indices[tet, 0]]
    ds = wp.matrix_from_cols(
        candidate_particle_q[tet_indices[tet, 1]] - origin,
        candidate_particle_q[tet_indices[tet, 2]] - origin,
        candidate_particle_q[tet_indices[tet, 3]] - origin,
    )
    f = ds * tet_poses[tet]
    determinant = wp.determinant(f)
    valid = True
    if not wp.isfinite(determinant):
        wp.atomic_max(failure_flags, 0, 2)
        valid = False
    else:
        wp.atomic_min(min_det_f, 0, determinant)
        if determinant < min_det_f_guard:
            wp.atomic_max(failure_flags, 0, 1)
            valid = False
    rest_volume = _compute_rest_volume(tet_poses[tet])
    if not wp.isfinite(rest_volume) or rest_volume <= 0.0:
        wp.atomic_max(failure_flags, 0, 2)
        valid = False
    energy, stress, projected = _evaluate_stable_neo_hookean(
        f, rest_volume, tet_materials[tet, 0], tet_materials[tet, 1]
    )
    if not wp.isfinite(energy):
        wp.atomic_max(failure_flags, 0, 2)
        valid = False
    for i in range(9):
        if not wp.isfinite(stress[i % 3, i // 3]):
            wp.atomic_max(failure_flags, 0, 2)
            valid = False
        for j in range(9):
            if not wp.isfinite(projected[i, j]):
                wp.atomic_max(failure_flags, 0, 2)
                valid = False
    tet_energy[tet] = energy
    return valid, stress, projected


@wp.kernel
def _evaluate_elastic_residual(
    min_det_f_guard: float,
    candidate_particle_q: wp.array[wp.vec3],
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
        min_det_f_guard,
        candidate_particle_q,
        tet_indices,
        tet_poses,
        tet_materials,
        tet_energy,
        min_det_f,
        failure_flags,
    )
    if valid:
        for a in range(4):
            index = particle_to_dynamic[tet_indices[tet, a]]
            if index >= 0:
                wp.atomic_add(residual_x, index, stress * _shape_gradient(tet_poses[tet], a))


@wp.kernel
def _assemble_elastic_residual_tangent(
    min_det_f_guard: float,
    candidate_particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    particle_to_dynamic: wp.array[int],
    elastic_block_slots: wp.array2d[int],
    residual_x: wp.array[wp.vec3],
    ax_values: wp.array[wp.mat33],
    global_values: wp.array[float],
    tet_energy: wp.array[float],
    min_det_f: wp.array[float],
    failure_flags: wp.array[int],
):
    tet = wp.tid()
    valid, stress, projected = _elastic_state(
        tet,
        min_det_f_guard,
        candidate_particle_q,
        tet_indices,
        tet_poses,
        tet_materials,
        tet_energy,
        min_det_f,
        failure_flags,
    )
    if valid:
        for a in range(4):
            ga = _shape_gradient(tet_poses[tet], a)
            index = particle_to_dynamic[tet_indices[tet, a]]
            if index >= 0:
                wp.atomic_add(residual_x, index, stress * ga)
            for b in range(a, 4):
                slot = elastic_block_slots[tet, 4 * a + b]
                if slot >= 0:
                    transpose_slot = elastic_block_slots[tet, 4 * b + a]
                    if slot >= ax_values.shape[0] or transpose_slot < 0 or transpose_slot >= ax_values.shape[0]:
                        wp.atomic_max(failure_flags, 0, 4)
                        continue
                    gb = _shape_gradient(tet_poses[tet], b)
                    block = wp.mat33(0.0)
                    for row in range(3):
                        for col in range(3):
                            for i in range(3):
                                for j in range(3):
                                    block[row, col] += ga[i] * projected[3 * i + row, 3 * j + col] * gb[j]
                    _scatter_internal_block(slot, block, ax_values, global_values)
                    if a != b:
                        _scatter_internal_block(transpose_slot, wp.transpose(block), ax_values, global_values)


@wp.kernel
def _check_residual_finite(residual: wp.array[wp.vec3], failure_flags: wp.array[int]):
    index = wp.tid()
    for axis in range(3):
        if not wp.isfinite(residual[index][axis]):
            wp.atomic_max(failure_flags, 0, 2)


@wp.kernel
def _check_scatter_finite(values: wp.array[float], failure_flags: wp.array[int]):
    if not wp.isfinite(values[wp.tid()]):
        wp.atomic_max(failure_flags, 0, 2)


def _prepare_evaluation(model, args):
    dt, guard, workspace = args["dt"], args["min_det_f_guard"], args["workspace"]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        dt32 = np.float32(dt)
        inv_dt_sq = np.float32((1.0 / np.float64(dt)) ** 2)
        guard32 = np.float32(guard)
    if not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(inv_dt_sq) or inv_dt_sq <= 0:
        raise ValueError("dt must be finite, positive and representable in float32 inertia")
    if not np.isfinite(guard32) or guard32 <= 0:
        raise ValueError("min_det_f_guard must be finite and positive in runtime float32 arithmetic")
    specifications = (
        ("candidate_particle_q", wp.vec3, (model.particle_count,)),
        ("particle_q_n", wp.vec3, (model.particle_count,)),
        ("particle_qd_n", wp.vec3, (model.particle_count,)),
        ("frozen_particle_f", wp.vec3, (model.particle_count,)),
        ("particle_to_dynamic", wp.int32, (model.particle_count,)),
        ("dynamic_particle_ids", wp.int32, (args["dynamic_particle_ids"].size,)),
        ("residual_x", wp.vec3, (args["dynamic_particle_ids"].size,)),
    )
    for name, dtype, shape in specifications:
        array = args[name]
        if array.dtype != dtype or array.shape != shape or array.device != model.device:
            raise ValueError(f"{name} has incompatible dtype, shape or device")
    for array, dtype, shape in (
        (workspace.elastic_block_slots, wp.int32, (model.tet_count, 16)),
        (workspace.tet_energy, wp.float32, (model.tet_count,)),
        (workspace.min_det_f, wp.float32, (1,)),
        (workspace.failure_flags, wp.int32, (1,)),
    ):
        if array.dtype != dtype or array.shape != shape or array.device != model.device:
            raise ValueError("tet workspace has incompatible dtype, shape or device")
    workspace.failure_flags.zero_()
    workspace.min_det_f.fill_(float("inf"))
    workspace.tet_energy.zero_()
    wp.launch(
        _validate_candidate_particles,
        (model.tet_count, 4),
        [
            args["candidate_particle_q"],
            args["particle_q_n"],
            args["particle_qd_n"],
            args["frozen_particle_f"],
            model.tet_indices,
            args["particle_to_dynamic"],
            workspace.failure_flags,
        ],
        device=model.device,
    )
    wp.launch(
        _evaluate_inertia_residual,
        args["dynamic_particle_ids"].size,
        [
            dt,
            model.gravity,
            model.particle_world,
            args["particle_q_n"],
            args["particle_qd_n"],
            args["candidate_particle_q"],
            model.particle_mass,
            args["frozen_particle_f"],
            args["dynamic_particle_ids"],
            args["residual_x"],
        ],
        device=model.device,
    )


def evaluate_tet_residual(
    model: Model,
    *,
    candidate_particle_q: wp.array[wp.vec3],
    particle_q_n: wp.array[wp.vec3],
    particle_qd_n: wp.array[wp.vec3],
    frozen_particle_f: wp.array[wp.vec3],
    dynamic_particle_ids: wp.array[int],
    particle_to_dynamic: wp.array[int],
    residual_x: wp.array[wp.vec3],
    dt: float,
    min_det_f_guard: float,
    workspace: TetAssemblyWorkspace,
) -> None:
    """Evaluate a trial without accessing tangent buffers; inspect failure_flags."""
    _prepare_evaluation(model, locals())
    wp.launch(
        _evaluate_elastic_residual,
        model.tet_count,
        [
            min_det_f_guard,
            candidate_particle_q,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            particle_to_dynamic,
            residual_x,
            workspace.tet_energy,
            workspace.min_det_f,
            workspace.failure_flags,
        ],
        device=model.device,
    )
    wp.launch(_check_residual_finite, residual_x.size, [residual_x, workspace.failure_flags], device=model.device)


def assemble_tet_residual_tangent(
    model: Model,
    *,
    candidate_particle_q: wp.array[wp.vec3],
    particle_q_n: wp.array[wp.vec3],
    particle_qd_n: wp.array[wp.vec3],
    frozen_particle_f: wp.array[wp.vec3],
    dynamic_particle_ids: wp.array[int],
    particle_to_dynamic: wp.array[int],
    residual_x: wp.array[wp.vec3],
    dt: float,
    min_det_f_guard: float,
    workspace: TetAssemblyWorkspace,
    scatter: TetScatterBuffers,
) -> None:
    """Write exact residual and the same GN values into owner and global slices."""
    if (
        scatter.ax_values.dtype != wp.mat33
        or scatter.global_values.dtype != wp.float32
        or scatter.ax_values.ndim != 1
        or scatter.global_values.ndim != 1
        or scatter.ax_values.device != model.device
        or scatter.global_values.device != model.device
        or scatter.global_values.size != 9 * scatter.ax_values.size
        or scatter.ax_values.size < dynamic_particle_ids.size
    ):
        raise ValueError("tet scatter requires matching mat33/scalar slices on the model device")
    _prepare_evaluation(model, locals())
    scatter.ax_values.zero_()
    scatter.global_values.zero_()
    wp.launch(
        _assemble_inertia_tangent,
        dynamic_particle_ids.size,
        [1.0 / dt**2, model.particle_mass, dynamic_particle_ids, scatter.ax_values, scatter.global_values],
        device=model.device,
    )
    wp.launch(
        _assemble_elastic_residual_tangent,
        model.tet_count,
        [
            min_det_f_guard,
            candidate_particle_q,
            model.tet_indices,
            model.tet_poses,
            model.tet_materials,
            particle_to_dynamic,
            workspace.elastic_block_slots,
            residual_x,
            scatter.ax_values,
            scatter.global_values,
            workspace.tet_energy,
            workspace.min_det_f,
            workspace.failure_flags,
        ],
        device=model.device,
    )
    wp.launch(_check_residual_finite, residual_x.size, [residual_x, workspace.failure_flags], device=model.device)
    wp.launch(
        _check_scatter_finite,
        scatter.global_values.size,
        [scatter.global_values, workspace.failure_flags],
        device=model.device,
    )
