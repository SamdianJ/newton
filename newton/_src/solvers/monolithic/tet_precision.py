# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native FP64/FP32 tet E/r from the formula source shared with the tangent assembly."""

import math
from typing import Any

import numpy as np
import warp as wp

from .convergence import abs3
from .residual import _accumulate_rows, _row_gather_map
from .tet_common import TetPhysicsWorkspace, _validate_candidate_particles, validate_tet_scope
from .tet_material import _cofactor_offset, _energy_stress_from, _volume_change, _volumetric_modulus


@wp.kernel
def _difference(x: wp.array[Any], xn: wp.array[Any], displacement: wp.array[Any]):
    i = wp.tid()
    displacement[i] = x[i] - xn[i]


@wp.kernel
def _particles(
    dx: wp.array[Any],
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
    absolute: wp.array[Any],
    parts: wp.array2d[Any],
    flags: wp.array[int],
):
    i = wp.tid()
    p = ids[i]
    world = worlds[p]
    if world < 0:
        world = gravity.shape[0] - 1
    displacement = dx[p]
    delta = displacement - dt * vn[p]
    applied = type(dt)(row_mass[p]) * type(dx[0])(gravity[world]) + type(dx[0])(forces[p])
    r = -applied
    magnitude = abs3(applied)
    inertia = type(dt)(0.0)
    if not consistent:
        r += type(dt)(mass[p]) * delta / (dt * dt)
        magnitude += abs3(type(dt)(mass[p]) * delta / (dt * dt))
        inertia = type(dt)(0.5) * type(dt)(mass[p]) * wp.dot(delta, delta) / (dt * dt)
    external = -wp.dot(applied, displacement)
    residual[i] = r
    absolute[i] = magnitude
    if not wp.isfinite(inertia) or not wp.isfinite(external) or not wp.isfinite(wp.length(r)):
        wp.atomic_max(flags, 0, 2)
    parts[0, i] = inertia
    parts[1, i] = external


@wp.kernel
def _elements(
    xn: wp.array[Any],
    dx: wp.array[Any],
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
    absolute_terms: wp.array[Any],
    parts: wp.array2d[Any],
    min_det: wp.array[Any],
    flags: wp.array[int],
):
    e = wp.tid()
    pose = wp.matrix(poses[e], shape=(3, 3), dtype=type(dt))
    # Displacement gradient h = F - I from the step-start edges plus the step
    # displacement: within a step it carries neither the rounding of absolute
    # candidate positions nor a cancellation against the identity.
    n0, n1, n2, n3 = indices[e, 0], indices[e, 1], indices[e, 2], indices[e, 3]
    start = wp.matrix_from_cols(xn[n1] - xn[n0], xn[n2] - xn[n0], xn[n3] - xn[n0]) * pose
    step = wp.matrix_from_cols(dx[n1] - dx[n0], dx[n2] - dx[n0], dx[n3] - dx[n0]) * pose
    h = (start - wp.identity(n=3, dtype=type(dt))) + step
    f = start + step
    det = wp.determinant(f)
    if not wp.isfinite(det):
        wp.atomic_max(flags, 0, 2)
        return
    wp.atomic_min(min_det, 0, det)
    if det < guard:
        wp.atomic_max(flags, 0, 1)
        return
    volume = type(dt)(1.0) / (type(dt)(6.0) * wp.determinant(pose))
    mu, lam = type(dt)(materials[e, 0]), type(dt)(materials[e, 1])
    # det F - 1 from its step-start value plus the step's exact change: a nearly
    # incompressible element's volumetric stress then rounds with the step, not
    # with the strain, and its bound stays below small loads.
    d, d_size = _volume_change(start - wp.identity(n=3, dtype=type(dt)), step)
    energy, stress = _energy_stress_from(h, d, volume, mu, lam, material)
    cofactor = wp.identity(n=3, dtype=type(dt)) + _cofactor_offset(h)
    stress_bound = wp.matrix(shape=(3, 3), dtype=type(dt))
    scale = volume * _volumetric_modulus(mu, lam, material) * d_size
    for i in range(3):
        for j in range(3):
            stress_bound[i, j] = scale * wp.abs(cofactor[i, j])
    if not wp.isfinite(energy) or not wp.isfinite(wp.ddot(stress, stress)):
        wp.atomic_max(flags, 0, 2)
        return
    inertia = type(dt)(0.0)
    for a in range(4):
        ia = indices[e, a]
        gradient = type(dx[0])(0.0)
        if a == 0:
            gradient = -type(dx[0])(
                pose[0, 0] + pose[1, 0] + pose[2, 0],
                pose[0, 1] + pose[1, 1] + pose[2, 1],
                pose[0, 2] + pose[1, 2] + pose[2, 2],
            )
        else:
            gradient = type(dx[0])(pose[a - 1, 0], pose[a - 1, 1], pose[a - 1, 2])
        r = stress * gradient
        magnitude = abs3(r) + stress_bound * abs3(gradient)
        da = dx[ia] - dt * vn[ia]
        if consistent:
            for b in range(4):
                ib = indices[e, b]
                db = dx[ib] - dt * vn[ib]
                weight = type(dt)(coefficients[e])
                if a == b:
                    weight *= type(dt)(2.0)
                r += weight * db / (dt * dt)
                magnitude += abs3(weight * db / (dt * dt))
                inertia += weight * wp.dot(da, db) / (type(dt)(2.0) * dt * dt)
        if mapping[ia] >= 0:
            terms[4 * e + a] = r
            absolute_terms[4 * e + a] = magnitude
    if not wp.isfinite(inertia):
        wp.atomic_max(flags, 0, 2)
    parts[0, e] = inertia
    parts[1, e] = energy


@wp.kernel
def _combine_parts(particle: wp.array2d[Any], element: wp.array2d[Any], parts: wp.array[Any]):
    parts[0] = particle[0, 0] + element[0, 0]
    parts[1] = element[1, 0]
    parts[2] = particle[1, 0]


@wp.kernel
def _finite(residual: wp.array[Any], flags: wp.array[int]):
    if not wp.isfinite(wp.length(residual[wp.tid()])):
        wp.atomic_max(flags, 0, 2)


class TetObjective:
    """Own native E/r scratch; input coordinates and step history remain read-only."""

    def __init__(
        self, model, *, material_model, mass_mode, density=None, dtype=wp.float64, actor=None, allow_cloth=False
    ):
        if dtype not in (wp.float64, wp.float32):
            raise ValueError("Tet E/r requires float64 or float32")
        self.dtype = dtype
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        ids = (
            np.flatnonzero(model.particle_inv_mass.numpy() > 0)
            if actor is None
            else np.asarray(actor.dynamic_particle_indices)
        ).astype(np.int32)
        mapping = np.full(model.particle_count, -1, dtype=np.int32)
        mapping[ids] = np.arange(len(ids))
        validate_tet_scope(model, dynamic_particle_ids=ids, particle_to_dynamic=mapping, allow_cloth=allow_cloth)
        if 4 * model.tet_count >= 2**31:
            raise ValueError("Tet residual contribution capacity exceeds int32")
        self.model = model
        self.physics = TetPhysicsWorkspace(
            model,
            material_model=material_model,
            mass_mode=mass_mode,
            density=density,
            particle_ids=None if actor is None else actor.particle_indices,
        )
        self.ids = wp.array(ids, dtype=int, device=model.device)
        self.mapping = wp.array(mapping, dtype=int, device=model.device)
        self._residual_offsets, self._residual_slots = _row_gather_map(
            mapping[self.physics.indices.numpy().ravel()], len(ids), model.device
        )
        self._residual_terms = wp.zeros(4 * model.tet_count, dtype=self.vector, device=model.device)
        self._absolute_terms = wp.zeros(4 * model.tet_count, dtype=self.vector, device=model.device)
        self.residual = wp.zeros(len(ids), dtype=self.vector, device=model.device)
        self.absolute = wp.zeros(len(ids), dtype=self.vector, device=model.device)
        self.parts = wp.zeros(3, dtype=self.dtype, device=model.device)
        # Per-particle (inertia, external) and per-element (inertia, elastic)
        # energies reduce in a fixed order rather than by atomic accumulation.
        self._particle_parts = wp.zeros((2, len(ids)), dtype=self.dtype, device=model.device)
        self._element_parts = wp.zeros((2, model.tet_count), dtype=self.dtype, device=model.device)
        self._particle_sums = wp.zeros((2, 1), dtype=self.dtype, device=model.device)
        self._element_sums = wp.zeros((2, 1), dtype=self.dtype, device=model.device)
        self.min_det_f = wp.zeros(1, dtype=self.dtype, device=model.device)
        self.flags = wp.zeros(1, dtype=int, device=model.device)
        self._difference = wp.zeros(model.particle_count, dtype=self.vector, device=model.device)

    def evaluate(self, x, xn, vn, forces, gravity, *, dt, min_det_f_guard=0.2, displacement=None):
        """Evaluate directly in the selected dtype, without promoting a rounded residual.

        Inertia and element edges use the step displacement ``x - xn``. A caller
        that tracks it exactly passes ``displacement``; otherwise it is formed
        from the rounded candidate positions ``x``.
        """
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(min_det_f_guard) or min_det_f_guard <= 0:
            raise ValueError("dt and detF guard must be finite and positive")
        native = np.float64 if self.dtype == wp.float64 else np.float32
        with np.errstate(over="ignore", under="ignore"):
            dt, min_det_f_guard = native(dt), native(min_det_f_guard)
        if not np.isfinite([dt, min_det_f_guard]).all() or dt <= 0 or min_det_f_guard <= 0:
            raise ValueError("dt and detF guard must be representable and positive in the selected dtype")
        derived = displacement is None
        for value, dtype, shape in (
            (x, self.vector, (self.model.particle_count,)),
            (xn, self.vector, (self.model.particle_count,)),
            (self._difference if derived else displacement, self.vector, (self.model.particle_count,)),
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
        if derived:
            displacement = self._difference
            wp.launch(_difference, x.size, [x, xn, displacement], device=self.model.device)
        physics = self.physics
        self.flags.zero_()
        self._element_parts.zero_()
        self._residual_terms.zero_()
        self._absolute_terms.zero_()
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
                displacement,
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
                self.absolute,
                self._particle_parts,
                self.flags,
            ],
            device=self.model.device,
        )
        wp.launch(
            _elements,
            self.model.tet_count,
            [
                xn,
                displacement,
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
                self._absolute_terms,
                self._element_parts,
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
        wp.launch(
            _accumulate_rows,
            self.absolute.size,
            [self._residual_offsets, self._residual_slots, self._absolute_terms, self.absolute],
            device=self.model.device,
        )
        wp.utils.array_sum(self._particle_parts, out=self._particle_sums, axis=1)
        wp.utils.array_sum(self._element_parts, out=self._element_sums, axis=1)
        wp.launch(_combine_parts, 1, [self._particle_sums, self._element_sums, self.parts], device=self.model.device)
        wp.launch(_finite, self.residual.size, [self.residual, self.flags], device=self.model.device)
