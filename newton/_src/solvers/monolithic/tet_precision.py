# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native FP64/FP32 tet E/r from one formula source; B/PCG remain FP32."""

import math
from typing import Any

import numpy as np
import warp as wp

from .residual import _accumulate_rows, _row_gather_map
from .tet_common import TetPhysicsWorkspace, _validate_candidate_particles, validate_tet_scope
from .tet_material import _evaluate_energy_stress


@wp.kernel
def _particles(
    x: wp.array[Any],
    xn: wp.array[Any],
    vn: wp.array[Any],
    forces: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    worlds: wp.array[int],
    ids: wp.array[int],
    mass: wp.array[float],
    row_mass: wp.array[float],
    consistent: bool,
    dt: Any,
    residual: wp.array[Any],
    parts: wp.array[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    p = ids[i]
    world = worlds[p]
    if world < 0:
        world = gravity.shape[0] - 1
    displacement = x[p] - xn[p]
    delta = displacement - dt * vn[p]
    applied = type(dt)(row_mass[p]) * type(x[0])(gravity[world]) + type(x[0])(forces[p])
    r = -applied
    inertia = type(dt)(0.0)
    if not consistent:
        r += type(dt)(mass[p]) * delta / (dt * dt)
        inertia = type(dt)(0.5) * type(dt)(mass[p]) * wp.dot(delta, delta) / (dt * dt)
    external = -wp.dot(applied, displacement)
    residual[i] = r
    if not wp.isfinite(inertia) or not wp.isfinite(external) or not wp.isfinite(wp.length(r)):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 2, external)


@wp.kernel
def _elements(
    x: wp.array[Any],
    xn: wp.array[Any],
    vn: wp.array[Any],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    mapping: wp.array[int],
    coefficients: wp.array[float],
    material: int,
    consistent: bool,
    dt: Any,
    guard: Any,
    terms: wp.array[Any],
    parts: wp.array[Any],
    min_det: wp.array[Any],
    flags: wp.array[int],
):
    e = wp.tid()
    pose = wp.matrix(poses[e], shape=(3, 3), dtype=type(dt))
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
    volume = type(dt)(1.0) / (type(dt)(6.0) * wp.determinant(pose))
    energy, stress = _evaluate_energy_stress(f, volume, type(dt)(materials[e, 0]), type(dt)(materials[e, 1]), material)
    if not wp.isfinite(energy) or not wp.isfinite(wp.ddot(stress, stress)):
        wp.atomic_max(flags, 0, 2)
        return
    inertia = type(dt)(0.0)
    for a in range(4):
        ia = indices[e, a]
        gradient = type(x[0])(0.0)
        if a == 0:
            gradient = -type(x[0])(
                pose[0, 0] + pose[1, 0] + pose[2, 0],
                pose[0, 1] + pose[1, 1] + pose[2, 1],
                pose[0, 2] + pose[1, 2] + pose[2, 2],
            )
        else:
            gradient = type(x[0])(pose[a - 1, 0], pose[a - 1, 1], pose[a - 1, 2])
        r = stress * gradient
        da = x[ia] - xn[ia] - dt * vn[ia]
        if consistent:
            for b in range(4):
                ib = indices[e, b]
                db = x[ib] - xn[ib] - dt * vn[ib]
                weight = type(dt)(coefficients[e])
                if a == b:
                    weight *= type(dt)(2.0)
                r += weight * db / (dt * dt)
                inertia += weight * wp.dot(da, db) / (type(dt)(2.0) * dt * dt)
        if mapping[ia] >= 0:
            terms[4 * e + a] = r
    if not wp.isfinite(inertia):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 1, energy)


@wp.kernel
def _finite(residual: wp.array[Any], flags: wp.array[int]):
    if not wp.isfinite(wp.length(residual[wp.tid()])):
        wp.atomic_max(flags, 0, 2)


class TetObjective:
    """Own native E/r scratch; input coordinates and step history remain read-only."""

    def __init__(self, model, *, material_model, mass_mode, density=None, dtype=wp.float64):
        if dtype not in (wp.float64, wp.float32):
            raise ValueError("Tet E/r requires float64 or float32")
        self.dtype = dtype
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        ids = np.flatnonzero(model.particle_inv_mass.numpy() > 0).astype(np.int32)
        mapping = np.full(model.particle_count, -1, dtype=np.int32)
        mapping[ids] = np.arange(len(ids))
        validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=mapping)
        if 4 * model.tet_count >= 2**31:
            raise ValueError("Tet residual contribution capacity exceeds int32")
        self.model = model
        self.physics = TetPhysicsWorkspace(model, material_model=material_model, mass_mode=mass_mode, density=density)
        self.ids = wp.array(ids, dtype=int, device=model.device)
        self.mapping = wp.array(mapping, dtype=int, device=model.device)
        self._residual_offsets, self._residual_slots = _row_gather_map(
            mapping[self.physics.indices.numpy().ravel()], len(ids), model.device
        )
        self._residual_terms = wp.zeros(4 * model.tet_count, dtype=self.vector, device=model.device)
        self.residual = wp.zeros(len(ids), dtype=self.vector, device=model.device)
        self.parts = wp.zeros(3, dtype=self.dtype, device=model.device)
        self.min_det_f = wp.zeros(1, dtype=self.dtype, device=model.device)
        self.flags = wp.zeros(1, dtype=int, device=model.device)

    def evaluate(self, x, xn, vn, forces, gravity, *, dt, min_det_f_guard=0.2):
        """Evaluate directly in the selected dtype, without promoting a rounded residual."""
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(min_det_f_guard) or min_det_f_guard <= 0:
            raise ValueError("dt and detF guard must be finite and positive")
        native = np.float64 if self.dtype == wp.float64 else np.float32
        with np.errstate(over="ignore", under="ignore"):
            dt, min_det_f_guard = native(dt), native(min_det_f_guard)
        if not np.isfinite([dt, min_det_f_guard]).all() or dt <= 0 or min_det_f_guard <= 0:
            raise ValueError("dt and detF guard must be representable and positive in the selected dtype")
        for value, dtype, shape in (
            (x, self.vector, (self.model.particle_count,)),
            (xn, self.vector, (self.model.particle_count,)),
            (vn, self.vector, (self.model.particle_count,)),
            (forces, wp.vec3, (self.model.particle_count,)),
            (gravity, wp.vec3, self.model.gravity.shape),
        ):
            if not isinstance(value, wp.array) or (value.dtype, value.shape, value.device) != (
                dtype,
                shape,
                self.model.device,
            ):
                raise ValueError("Incompatible native tet inputs")
        physics = self.physics
        self.flags.zero_()
        self.parts.zero_()
        self._residual_terms.zero_()
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
                self.dtype(dt),
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
                self.dtype(dt),
                self.dtype(min_det_f_guard),
                self._residual_terms,
                self.parts,
                self.min_det_f,
                self.flags,
            ],
            device=self.model.device,
        )
        wp.launch(
            _accumulate_rows,
            self.residual.size,
            [self._residual_offsets, self._residual_slots, self._residual_terms, self.residual],
            device=self.model.device,
        )
        wp.launch(_finite, self.residual.size, [self.residual, self.flags], device=self.model.device)
