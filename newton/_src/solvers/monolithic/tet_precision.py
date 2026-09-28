# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Direct FP64 tet E/r for the private coupled path; legacy entry points are unchanged."""

import math

import numpy as np
import warp as wp

from .tet_common import TetPhysicsWorkspace, _validate_candidate_particles, validate_tet_scope
from .tet_material import _evaluate_energy_stress64


@wp.kernel
def _particles(
    x: wp.array[wp.vec3d],
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    forces: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    worlds: wp.array[int],
    ids: wp.array[int],
    mass: wp.array[float],
    row_mass: wp.array[float],
    consistent: bool,
    dt: wp.float64,
    residual: wp.array[wp.vec3d],
    parts: wp.array[wp.float64],
    flags: wp.array[int],
):
    i = wp.tid()
    p = ids[i]
    world = worlds[p]
    if world < 0:
        world = gravity.shape[0] - 1
    displacement = x[p] - xn[p]
    delta = displacement - dt * vn[p]
    applied = wp.float64(row_mass[p]) * wp.vec3d(gravity[world]) + wp.vec3d(forces[p])
    r = -applied
    inertia = wp.float64(0.0)
    if not consistent:
        r += wp.float64(mass[p]) * delta / (dt * dt)
        inertia = wp.float64(0.5) * wp.float64(mass[p]) * wp.dot(delta, delta) / (dt * dt)
    external = -wp.dot(applied, displacement)
    residual[i] = r
    if not wp.isfinite(inertia) or not wp.isfinite(external) or not wp.isfinite(wp.length(r)):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 2, external)


@wp.kernel
def _elements(
    x: wp.array[wp.vec3d],
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    mapping: wp.array[int],
    coefficients: wp.array[float],
    material: int,
    consistent: bool,
    dt: wp.float64,
    guard: wp.float64,
    residual: wp.array[wp.vec3d],
    parts: wp.array[wp.float64],
    min_det: wp.array[wp.float64],
    flags: wp.array[int],
):
    e = wp.tid()
    pose = wp.mat33d(poses[e])
    origin = x[indices[e, 0]]
    f = wp.matrix_from_cols(x[indices[e, 1]] - origin, x[indices[e, 2]] - origin, x[indices[e, 3]] - origin) * pose
    det = wp.determinant(f)
    if not wp.isfinite(det):
        wp.atomic_max(flags, 0, 2)
        return
    wp.atomic_min(min_det, 0, det)
    if det < guard:
        wp.atomic_max(flags, 0, 1)
        return
    volume = wp.float64(1.0) / (wp.float64(6.0) * wp.determinant(pose))
    energy, stress = _evaluate_energy_stress64(
        f, volume, wp.float64(materials[e, 0]), wp.float64(materials[e, 1]), material
    )
    if not wp.isfinite(energy) or not wp.isfinite(wp.ddot(stress, stress)):
        wp.atomic_max(flags, 0, 2)
        return
    inertia = wp.float64(0.0)
    for a in range(4):
        ia = indices[e, a]
        gradient = wp.vec3d(0.0)
        if a == 0:
            gradient = -wp.vec3d(
                pose[0, 0] + pose[1, 0] + pose[2, 0],
                pose[0, 1] + pose[1, 1] + pose[2, 1],
                pose[0, 2] + pose[1, 2] + pose[2, 2],
            )
        else:
            gradient = wp.vec3d(pose[a - 1, 0], pose[a - 1, 1], pose[a - 1, 2])
        r = stress * gradient
        da = x[ia] - xn[ia] - dt * vn[ia]
        if consistent:
            for b in range(4):
                ib = indices[e, b]
                db = x[ib] - xn[ib] - dt * vn[ib]
                weight = wp.float64(coefficients[e])
                if a == b:
                    weight *= wp.float64(2.0)
                r += weight * db / (dt * dt)
                inertia += weight * wp.dot(da, db) / (wp.float64(2.0) * dt * dt)
        if mapping[ia] >= 0:
            wp.atomic_add(residual, mapping[ia], r)
    if not wp.isfinite(inertia):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 1, energy)


@wp.kernel
def _finite(residual: wp.array[wp.vec3d], flags: wp.array[int]):
    if not wp.isfinite(wp.length(residual[wp.tid()])):
        wp.atomic_max(flags, 0, 2)


class TetObjective64:
    """Own preallocated FP64 E/r scratch; input coordinates and step history remain read-only."""

    def __init__(self, model, *, material_model, mass_mode, density=None):
        ids = np.flatnonzero(model.particle_inv_mass.numpy() > 0).astype(np.int32)
        mapping = np.full(model.particle_count, -1, dtype=np.int32)
        mapping[ids] = np.arange(len(ids))
        validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=mapping)
        self.model = model
        self.physics = TetPhysicsWorkspace(model, material_model=material_model, mass_mode=mass_mode, density=density)
        self.ids = wp.array(ids, dtype=int, device=model.device)
        self.mapping = wp.array(mapping, dtype=int, device=model.device)
        self.residual = wp.zeros(len(ids), dtype=wp.vec3d, device=model.device)
        self.parts = wp.zeros(3, dtype=wp.float64, device=model.device)
        self.min_det_f = wp.zeros(1, dtype=wp.float64, device=model.device)
        self.flags = wp.zeros(1, dtype=int, device=model.device)

    def evaluate(self, x, xn, vn, forces, gravity, *, dt, min_det_f_guard=0.2):
        """Overwrite E/r and status, without reading or rounding an existing FP32 residual."""
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(min_det_f_guard) or min_det_f_guard <= 0:
            raise ValueError("dt and detF guard must be finite and positive")
        for value, dtype, shape in (
            (x, wp.vec3d, (self.model.particle_count,)),
            (xn, wp.vec3d, (self.model.particle_count,)),
            (vn, wp.vec3d, (self.model.particle_count,)),
            (forces, wp.vec3, (self.model.particle_count,)),
            (gravity, wp.vec3, self.model.gravity.shape),
        ):
            if not isinstance(value, wp.array) or (value.dtype, value.shape, value.device) != (
                dtype,
                shape,
                self.model.device,
            ):
                raise ValueError("Incompatible FP64 tet inputs")
        physics = self.physics
        physics.validate_identity()
        self.flags.zero_()
        self.parts.zero_()
        self.min_det_f.fill_(float("inf"))
        wp.launch(
            _validate_candidate_particles,
            (self.model.tet_count, 4),
            [x, xn, vn, forces, physics.indices, self.mapping, self.flags],
            device=self.model.device,
        )
        wp.launch(
            _particles,
            self.ids.size,
            [
                x,
                xn,
                vn,
                forces,
                gravity,
                physics.worlds,
                self.ids,
                physics.mass,
                physics.row_mass,
                physics.mass_mode == "consistent",
                dt,
                self.residual,
                self.parts,
                self.flags,
            ],
            device=self.model.device,
        )
        wp.launch(
            _elements,
            self.model.tet_count,
            [
                x,
                xn,
                vn,
                physics.indices,
                physics.poses,
                physics.materials,
                self.mapping,
                physics.coefficients,
                int(physics.material_model == "smith_log_stabilized"),
                physics.mass_mode == "consistent",
                dt,
                min_det_f_guard,
                self.residual,
                self.parts,
                self.min_det_f,
                self.flags,
            ],
            device=self.model.device,
        )
        wp.launch(_finite, self.residual.size, [self.residual, self.flags], device=self.model.device)
