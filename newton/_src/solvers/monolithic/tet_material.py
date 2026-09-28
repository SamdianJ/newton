# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P1 tet exact residual and material-specific projected tangent.

The constitutive mapping is the Kim stable Neo-Hookean variant in SuperDex
54ae749a, ``mochi_core/materials/batched_kim_neo_hookean.h``: standard Lamé
parameters become ``mu`` and ``lambda + mu``. This is the variant without the
Smith logarithmic term. Its energy is shifted to zero at the rest state.
The production tangent drops the cofactor geometric derivative; it is not the
exact residual derivative or a spectral projection of the raw Hessian.
Smith instead uses log(I_C+1) and spectral projection of the full nodal Hessian."""

from typing import Any

import warp as wp

mat99 = wp.types.matrix(shape=(9, 9), dtype=wp.float32)


mat1212 = wp.types.matrix(shape=(12, 12), dtype=wp.float32)


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
def _evaluate_stable_neo_hookean(
    F: wp.mat33, rest_volume: float, mu_lame: float, lambda_lame: float, tangent: bool = True
):
    j_minus_one = wp.determinant(F) - 1.0
    lambda_nh = lambda_lame + mu_lame
    cofactor = _compute_cofactor(F)
    squared_norm = float(0.0)
    projected = mat99(0.0)
    for i in range(9):
        squared_norm += F[i % 3, i // 3] * F[i % 3, i // 3]
        if tangent:
            for j in range(9):
                projected[i, j] = rest_volume * lambda_nh * cofactor[i % 3, i // 3] * cofactor[j % 3, j // 3]
            projected[i, i] += rest_volume * mu_lame
    energy = rest_volume * (
        0.5 * mu_lame * (squared_norm - 3.0) - mu_lame * j_minus_one + 0.5 * lambda_nh * j_minus_one * j_minus_one
    )
    stress = rest_volume * (mu_lame * F + (lambda_nh * j_minus_one - mu_lame) * cofactor)
    return energy, stress, projected


@wp.func
def _evaluate_smith(F: wp.mat33, rest_volume: float, mu: float, lam: float, tangent: bool = True):
    mu_hat = (4.0 / 3.0) * mu
    lam_hat = lam + (5.0 / 6.0) * mu
    ic = wp.ddot(F, F)
    d = wp.determinant(F) - 1.0
    c = _compute_cofactor(F)
    s = lam_hat * d - mu
    a = mu_hat * (1.0 - 1.0 / (ic + 1.0))
    # Rest-shifted energy avoids subtracting the large alpha-dependent constant.
    energy = rest_volume * (0.5 * mu_hat * (ic - 3.0 - wp.log((ic + 1.0) / 4.0)) + 0.5 * lam_hat * d * d - mu * d)
    stress = rest_volume * (a * F + s * c)
    raw = mat99(0.0)
    if tangent:
        dc = _compute_cofactor_derivative(F)
        for i in range(9):
            for j in range(9):
                raw[i, j] = rest_volume * (
                    2.0 * mu_hat / ((ic + 1.0) * (ic + 1.0)) * F[i % 3, i // 3] * F[j % 3, j // 3]
                    + lam_hat * c[i % 3, i // 3] * c[j % 3, j // 3]
                    + s * dc[i, j]
                )
            raw[i, i] += rest_volume * a
    return energy, stress, raw


@wp.func
def _project_element_psd(raw: mat1212):
    # Cyclic Jacobi on the full nodal Hessian, before fixed-node elimination.
    h = raw
    vectors = wp.identity(n=12, dtype=float)
    for _sweep in range(20):
        largest = float(0.0)
        scale = float(0.0)
        for i in range(12):
            scale = wp.max(scale, wp.abs(h[i, i]))
            for j in range(i + 1, 12):
                largest = wp.max(largest, wp.abs(h[i, j]))
        if largest <= 1.0e-7 * wp.max(scale, 1.0e-20):
            break
        for p in range(11):
            for q in range(p + 1, 12):
                off = h[p, q]
                if wp.abs(off) <= 1.0e-8 * wp.max(scale, 1.0e-20):
                    continue
                theta = 0.5 * wp.atan2(2.0 * off, h[q, q] - h[p, p])
                c, s = wp.cos(theta), wp.sin(theta)
                hp, hq = h[p, p], h[q, q]
                h[p, p] = c * c * hp - 2.0 * c * s * off + s * s * hq
                h[q, q] = s * s * hp + 2.0 * c * s * off + c * c * hq
                h[p, q] = 0.0
                h[q, p] = 0.0
                for k in range(12):
                    if k != p and k != q:
                        a, b = h[k, p], h[k, q]
                        h[k, p] = c * a - s * b
                        h[p, k] = h[k, p]
                        h[k, q] = s * a + c * b
                        h[q, k] = h[k, q]
                    a, b = vectors[k, p], vectors[k, q]
                    vectors[k, p] = c * a - s * b
                    vectors[k, q] = s * a + c * b
    result = mat1212(0.0)
    for i in range(12):
        for j in range(i, 12):
            value = float(0.0)
            for k in range(12):
                value += vectors[i, k] * wp.max(h[k, k], 0.0) * vectors[j, k]
            result[i, j] = value
            result[j, i] = value
    remaining = float(0.0)
    scale = float(0.0)
    for i in range(12):
        scale = wp.max(scale, wp.abs(h[i, i]))
        for j in range(i + 1, 12):
            remaining = wp.max(remaining, wp.abs(h[i, j]))
    return result, remaining <= 1.0e-6 * wp.max(scale, 1.0e-20)


@wp.func
def _nodal_hessian(raw: mat99, pose: wp.mat33):
    h = mat1212(0.0)
    for a in range(4):
        ga = _shape_gradient(pose, a)
        for b in range(4):
            gb = _shape_gradient(pose, b)
            for row in range(3):
                for col in range(3):
                    value = float(0.0)
                    for i in range(3):
                        for j in range(3):
                            value += ga[i] * raw[3 * i + row, 3 * j + col] * gb[j]
                    h[3 * a + row, 3 * b + col] = value
    return 0.5 * (h + wp.transpose(h))


@wp.func
def _shape_gradient(inverse_rest: wp.mat33, node: int) -> wp.vec3:
    gradient = wp.vec3(0.0)
    if node == 0:
        gradient = -(inverse_rest[0, :] + inverse_rest[1, :] + inverse_rest[2, :])
    else:
        gradient = inverse_rest[node - 1, :]
    return gradient


@wp.func
def _evaluate_energy_stress64(f: wp.mat33d, volume: wp.float64, mu: wp.float64, lam: wp.float64, material: int):
    ic = wp.ddot(f, f)
    d = wp.determinant(f) - wp.float64(1.0)
    cof = wp.matrix_from_cols(wp.cross(f[:, 1], f[:, 2]), wp.cross(f[:, 2], f[:, 0]), wp.cross(f[:, 0], f[:, 1]))
    a = mu
    s = (lam + mu) * d - mu
    energy = wp.float64(0.5) * mu * (ic - wp.float64(3.0)) - mu * d + wp.float64(0.5) * (lam + mu) * d * d
    if material == 1:
        mh = wp.float64(4.0) * mu / wp.float64(3.0)
        lh = lam + wp.float64(5.0) * mu / wp.float64(6.0)
        a = mh * (wp.float64(1.0) - wp.float64(1.0) / (ic + wp.float64(1.0)))
        s = lh * d - mu
        energy = (
            wp.float64(0.5) * mh * (ic - wp.float64(3.0) - wp.log((ic + wp.float64(1.0)) / wp.float64(4.0)))
            + wp.float64(0.5) * lh * d * d
            - mu * d
        )
    return volume * energy, volume * (a * f + s * cof)


@wp.func
def _elastic_state(
    tet: int,
    material_model: int,
    min_det_f_guard: float,
    candidate_particle_q: wp.array[Any],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    tet_energy: wp.array[float],
    min_det_f: wp.array[float],
    failure_flags: wp.array[int],
    tangent: bool,
    high_precision: bool,
):
    origin = candidate_particle_q[tet_indices[tet, 0]]
    ds = wp.matrix_from_cols(
        candidate_particle_q[tet_indices[tet, 1]] - origin,
        candidate_particle_q[tet_indices[tet, 2]] - origin,
        candidate_particle_q[tet_indices[tet, 3]] - origin,
    )
    f = wp.mat33(ds) * tet_poses[tet]
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
        f, rest_volume, tet_materials[tet, 0], tet_materials[tet, 1], tangent
    )
    if material_model == 1:
        energy, stress, projected = _evaluate_smith(
            f, rest_volume, tet_materials[tet, 0], tet_materials[tet, 1], tangent
        )
    if high_precision:
        exact_energy, exact_stress = _evaluate_energy_stress64(
            wp.mat33d(ds) * wp.mat33d(tet_poses[tet]),
            wp.float64(rest_volume),
            wp.float64(tet_materials[tet, 0]),
            wp.float64(tet_materials[tet, 1]),
            material_model,
        )
        energy = float(exact_energy)
        stress = wp.mat33(exact_stress)
    if not wp.isfinite(energy):
        wp.atomic_max(failure_flags, 0, 2)
        valid = False
    for i in range(9):
        if not wp.isfinite(stress[i % 3, i // 3]):
            wp.atomic_max(failure_flags, 0, 2)
            valid = False
        if tangent:
            for j in range(9):
                if not wp.isfinite(projected[i, j]):
                    wp.atomic_max(failure_flags, 0, 2)
                    valid = False
    tet_energy[tet] = energy
    return valid, stress, projected
