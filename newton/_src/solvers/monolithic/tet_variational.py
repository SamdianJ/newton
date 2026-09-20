# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete x-only incremental objective; no production Hessian assembly."""

import math

import numpy as np
import warp as wp

from .tet import _compute_rest_volume, _evaluate_energy_stress64, _validate_candidate_particles


@wp.kernel
def _element_objective(
    x: wp.array[wp.vec3d],
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    coefficients: wp.array[float],
    material: int,
    consistent: bool,
    dt: wp.float64,
    guard: float,
    parts: wp.array[wp.float64],
    flags: wp.array[int],
):
    e = wp.tid()
    origin = x[indices[e, 0]]
    ds32 = wp.matrix_from_cols(x[indices[e, 1]] - origin, x[indices[e, 2]] - origin, x[indices[e, 3]] - origin)
    j32 = wp.determinant(wp.mat33(ds32) * poses[e])
    if not wp.isfinite(j32):
        wp.atomic_max(flags, 0, 2)
    elif j32 < guard:
        wp.atomic_max(flags, 0, 1)
    ds = wp.matrix_from_cols(
        wp.vec3d(x[indices[e, 1]]) - wp.vec3d(origin),
        wp.vec3d(x[indices[e, 2]]) - wp.vec3d(origin),
        wp.vec3d(x[indices[e, 3]]) - wp.vec3d(origin),
    )
    f = ds * wp.mat33d(poses[e])
    volume = wp.float64(_compute_rest_volume(poses[e]))
    mu = wp.float64(materials[e, 0])
    lam = wp.float64(materials[e, 1])
    energy, _stress = _evaluate_energy_stress64(f, volume, mu, lam, material)
    inertia = wp.float64(0.0)
    if consistent:
        for a in range(4):
            ia = indices[e, a]
            da = wp.vec3d(x[ia]) - wp.vec3d(xn[ia]) - dt * wp.vec3d(vn[ia])
            for b in range(4):
                ib = indices[e, b]
                db = wp.vec3d(x[ib]) - wp.vec3d(xn[ib]) - dt * wp.vec3d(vn[ib])
                weight = wp.float64(coefficients[e])
                if a == b:
                    weight *= wp.float64(2.0)
                inertia += weight * wp.dot(da, db) / (wp.float64(2.0) * dt * dt)
    if not wp.isfinite(energy) or not wp.isfinite(inertia):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 1, energy)
    wp.atomic_add(parts, 4, wp.abs(energy) + wp.abs(inertia))


@wp.kernel
def _particle_objective(
    x: wp.array[wp.vec3d],
    xn: wp.array[wp.vec3d],
    vn: wp.array[wp.vec3d],
    ids: wp.array[int],
    mass: wp.array[float],
    row_mass: wp.array[float],
    gravity: wp.array[wp.vec3],
    worlds: wp.array[int],
    forces: wp.array[wp.vec3],
    consistent: bool,
    dt: wp.float64,
    parts: wp.array[wp.float64],
    flags: wp.array[int],
):
    i = ids[wp.tid()]
    delta = wp.vec3d(x[i]) - wp.vec3d(xn[i])
    d = delta - dt * wp.vec3d(vn[i])
    world = worlds[i]
    if world < 0:
        world = gravity.shape[0] - 1
    inertia = wp.float64(0.0)
    if not consistent:
        inertia = wp.float64(mass[i]) * wp.dot(d, d) / (wp.float64(2.0) * dt * dt)
    g = -wp.float64(row_mass[i]) * wp.dot(wp.vec3d(gravity[world]), delta)
    external = -wp.dot(wp.vec3d(forces[i]), delta)
    if not wp.isfinite(inertia) or not wp.isfinite(g) or not wp.isfinite(external):
        wp.atomic_max(flags, 0, 2)
    wp.atomic_add(parts, 0, inertia)
    wp.atomic_add(parts, 2, g)
    wp.atomic_add(parts, 3, external)
    wp.atomic_add(parts, 4, wp.abs(inertia) + wp.abs(g) + wp.abs(external))


@wp.kernel
def _directional_product(r: wp.array[float], p: wp.array[float], products: wp.array[wp.float64]):
    i = wp.tid()
    value = wp.float64(r[i]) * wp.float64(p[i])
    wp.atomic_add(products, 0, value)
    wp.atomic_add(products, 1, wp.abs(value))


class TetObjectiveWorkspace:
    """Overwrite-only FP64 energy scratch; frozen physical inputs belong to the step."""

    def __init__(self, model, layout, physics):
        self.model, self.layout, self.physics = model, layout, physics
        self.parts = wp.zeros(5, dtype=wp.float64, device=model.device)
        self.flags = wp.zeros(1, dtype=int, device=model.device)
        self.products = wp.zeros(2, dtype=wp.float64, device=model.device)

    def evaluate(self, candidate, original, inputs, dt, guard, graphs):
        model, layout, physics = self.model, self.layout, self.physics
        consistent = physics.mass_mode == "consistent"
        material = int(physics.material_model == "smith_log_stabilized")
        buffers = (
            candidate.state.particle_q,
            original.particle_q,
            original.particle_qd,
            inputs.particle_f,
            inputs.gravity,
            physics.indices,
            physics.worlds,
            physics.poses,
            physics.materials,
            physics.coefficients,
            physics.mass,
            physics.row_mass,
            layout.dynamic_particle_ids,
            layout.particle_to_dynamic,
            self.parts,
            self.flags,
        )

        def execute():
            self.parts.zero_()
            self.flags.zero_()
            wp.launch(
                _validate_candidate_particles,
                (model.tet_count, 4),
                [
                    candidate.state.particle_q,
                    original.particle_q,
                    original.particle_qd,
                    inputs.particle_f,
                    physics.indices,
                    layout.particle_to_dynamic,
                    self.flags,
                ],
                device=model.device,
            )
            wp.launch(
                _element_objective,
                model.tet_count,
                [
                    candidate.state.particle_q,
                    original.particle_q,
                    original.particle_qd,
                    physics.indices,
                    physics.poses,
                    physics.materials,
                    physics.coefficients,
                    material,
                    consistent,
                    dt,
                    guard,
                    self.parts,
                    self.flags,
                ],
                device=model.device,
            )
            wp.launch(
                _particle_objective,
                layout.dynamic_particle_count,
                [
                    candidate.state.particle_q,
                    original.particle_q,
                    original.particle_qd,
                    layout.dynamic_particle_ids,
                    physics.mass,
                    physics.row_mass,
                    inputs.gravity,
                    physics.worlds,
                    inputs.particle_f,
                    consistent,
                    dt,
                    self.parts,
                    self.flags,
                ],
                device=model.device,
            )

        graphs.run("tet_objective", (dt, guard, material, consistent), buffers, execute)
        values = self.parts.numpy()
        status = int(self.flags.numpy()[0])
        if not np.isfinite(values).all():
            status = 2
        return status, tuple(map(float, values[:4])), float(values[4])

    def slope(self, residual, delta):
        self.products.zero_()
        wp.launch(_directional_product, residual.size, [residual, delta, self.products], device=self.model.device)
        values = self.products.numpy()
        return float(values[0]), float(values[1])


def acceptance(mode, current, trial, slope, alpha, merit, trial_merit, *, energy_noise, residual_noise, slope_noise):
    """Apply independent A/R predicates after the caller's geometry/context checks."""
    if mode not in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
        raise ValueError("Unknown variational acceptance mode")
    if not all(math.isfinite(v) for v in (current, trial, slope, merit, trial_merit)):
        raise ValueError("Nonfinite variational trial")
    if not 0 < alpha <= 1 or not all(math.isfinite(v) and v >= 0 for v in (energy_noise, residual_noise, slope_noise)):
        raise ValueError("Invalid variational acceptance parameters")
    a = slope < -slope_noise and trial <= current + 1.0e-4 * alpha * slope + energy_noise
    r = trial_merit <= merit + residual_noise
    accepted = a if mode == "Armijo" else r if mode == "ResNorm" else a or r
    improved = accepted and (
        (mode != "ResNorm" and a and trial < current - energy_noise)
        or (mode != "Armijo" and r and trial_merit < merit - residual_noise)
    )
    return bool(accepted), bool(a), bool(r), bool(improved)
