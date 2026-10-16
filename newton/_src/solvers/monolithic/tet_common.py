# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared tet configuration, mass snapshots, scope and candidate validation."""

import enum
import hashlib
import json
from typing import Any

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim.model import Model
from .tet_material import (
    _evaluate_rest_volumes,
)


class TetEvaluationStatus(enum.IntEnum):
    SUCCESS = 0
    DET_F_GUARD = 1
    NONFINITE = 2
    INVALID_FIXED_PARTICLE = 3
    INVALID_SCATTER = 4
    PSD_PROJECTION_FAILURE = 5


class TetPhysicsWorkspace:
    """Construction snapshots for P1 material/mass; no candidate host readback."""

    def __init__(self, model, *, material_model, mass_mode, density=None, particle_ids=None):
        if material_model not in ("kim_stable_no_log", "smith_log_stabilized") or mass_mode not in (
            "lumped",
            "consistent",
        ):
            raise ValueError("Unknown tet material/mass mode")
        self.model, self.material_model, self.mass_mode = model, material_model, mass_mode
        names = ("tet_indices", "tet_poses", "tet_materials", "particle_mass", "particle_inv_mass", "particle_world")
        self.sources = {name: getattr(model, name) for name in names}
        host = {name: value.numpy().copy() for name, value in self.sources.items()}
        materials = host["tet_materials"]
        if material_model == "smith_log_stabilized":
            mapped = np.column_stack(
                (
                    materials[:, 0].astype(float) * 4 / 3,
                    materials[:, 1].astype(float) + materials[:, 0].astype(float) * 5 / 6,
                )
            )
            if not np.isfinite(mapped).all() or np.any(mapped[:, 1] <= 0) or np.any(mapped > np.finfo(np.float32).max):
                raise ValueError("Smith mapped parameters must be finite with lambda_hat > 0")
        self.indices = wp.clone(model.tet_indices)
        self.worlds = wp.clone(model.particle_world)
        self.materials = wp.clone(model.tet_materials)
        self.poses = wp.clone(model.tet_poses)
        self.mass = wp.clone(model.particle_mass)
        coefficients = np.zeros(model.tet_count, dtype=np.float32)
        row_mass = host["particle_mass"].copy()
        diagonal = row_mass.copy()
        self.density = None
        if mass_mode == "consistent":
            if not isinstance(density, wp.array) or (density.shape, density.dtype, density.device) != (
                (model.tet_count,),
                wp.float32,
                model.device,
            ):
                raise ValueError("consistent mass requires same-device float32 per-tet density")
            rho = density.numpy()
            if not np.isfinite(rho).all() or np.any(rho <= 0):
                raise ValueError("tet density must be finite and positive")
            volume = 1 / (6 * np.linalg.det(host["tet_poses"].astype(np.float64)))
            coefficients = (rho.astype(float) * volume / 20).astype(np.float32)
            if not np.isfinite(coefficients).all() or np.any(coefficients <= 0):
                raise ValueError("consistent element mass must be positive finite float32")
            row_mass = np.zeros(model.particle_count, dtype=np.float64)
            diagonal = np.zeros_like(row_mass)
            for ids, coefficient in zip(host["tet_indices"], coefficients, strict=True):
                np.add.at(row_mass, ids, 5 * float(coefficient))
                np.add.at(diagonal, ids, 2 * float(coefficient))
            dynamic = host["particle_inv_mass"] > 0
            check = dynamic | (host["particle_mass"] > 0)
            if particle_ids is not None:
                check &= np.isin(np.arange(model.particle_count), particle_ids)
            if not np.allclose(host["particle_mass"][check], row_mass[check], rtol=2e-5, atol=0):
                raise ValueError("Builder mass disagrees with explicit density row sums")
            if not np.isfinite(row_mass).all() or np.any(row_mass > np.finfo(np.float32).max):
                raise ValueError("consistent row mass overflows float32")
            self.density = wp.clone(density)
            host["density"] = rho
        elif density is not None:
            raise ValueError("tet_rest_density requires consistent mass")
        self.coefficients = wp.array(coefficients, dtype=float, device=model.device)
        self.row_mass = wp.array(row_mass, dtype=float, device=model.device)
        self.dynamic_mass_diagonal = wp.array(diagonal, dtype=float, device=model.device)
        digest = hashlib.sha256(json.dumps([material_model, mass_mode]).encode())
        for name, values in sorted(host.items()):
            digest.update(name.encode())
            digest.update(str((values.shape, values.dtype.str)).encode())
            digest.update(values.tobytes())
        self.identity = digest.hexdigest()


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


@wp.kernel
def _validate_candidate_particles(
    candidate: wp.array[Any],
    positions: wp.array[Any],
    velocities: wp.array[Any],
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


def _validate_evaluation(model, args):
    dt, guard, workspace = args["dt"], args["min_det_f_guard"], args["workspace"]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        dt32 = np.float32(dt)
        inv_dt_sq = np.float32((1.0 / np.float64(dt)) ** 2)
        guard32 = np.float32(guard)
    if not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(inv_dt_sq) or inv_dt_sq <= 0:
        raise ValueError("dt must be finite, positive and representable in float32 inertia")
    if not np.isfinite(guard32) or guard32 <= 0:
        raise ValueError("min_det_f_guard must be finite and positive in runtime float32 arithmetic")
    position_dtype = args["candidate_particle_q"].dtype
    if position_dtype not in (wp.vec3, wp.vec3d):
        raise ValueError("Unsupported tet position dtype")
    specifications = (
        ("candidate_particle_q", position_dtype, (model.particle_count,)),
        ("particle_q_n", position_dtype, (model.particle_count,)),
        ("particle_qd_n", position_dtype, (model.particle_count,)),
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
