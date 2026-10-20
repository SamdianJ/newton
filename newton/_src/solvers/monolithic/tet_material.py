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

# Single-precision aliases of the generic 9x9 and 12x12 element matrices; the
# functions below construct their matrices in the precision of their inputs.
mat99 = wp.types.matrix(shape=(9, 9), dtype=wp.float32)
mat99d = wp.types.matrix(shape=(9, 9), dtype=wp.float64)
mat1212 = wp.types.matrix(shape=(12, 12), dtype=wp.float32)
mat1212d = wp.types.matrix(shape=(12, 12), dtype=wp.float64)


@wp.func
def _compute_rest_volume(inverse_rest: Any):
    return type(inverse_rest[0, 0])(1.0) / (type(inverse_rest[0, 0])(6.0) * wp.determinant(inverse_rest))


@wp.kernel
def _evaluate_rest_volumes(poses: wp.array[wp.mat33], volumes: wp.array[float]):
    tet = wp.tid()
    volumes[tet] = _compute_rest_volume(poses[tet])


@wp.func
def _compute_cofactor(F: Any):
    return wp.matrix_from_cols(wp.cross(F[:, 1], F[:, 2]), wp.cross(F[:, 2], F[:, 0]), wp.cross(F[:, 0], F[:, 1]))


@wp.func
def _compute_cofactor_derivative(F: Any):
    # Column-major vec(F) derivative of the cofactor, used by the Smith tangent.
    result = wp.matrix(shape=(9, 9), dtype=type(F[0, 0]))
    for j in range(9):
        delta = wp.matrix(shape=(3, 3), dtype=type(F[0, 0]))
        delta[j % 3, j // 3] = type(F[0, 0])(1.0)
        derivative = wp.matrix_from_cols(
            wp.cross(delta[:, 1], F[:, 2]) + wp.cross(F[:, 1], delta[:, 2]),
            wp.cross(delta[:, 2], F[:, 0]) + wp.cross(F[:, 2], delta[:, 0]),
            wp.cross(delta[:, 0], F[:, 1]) + wp.cross(F[:, 0], delta[:, 1]),
        )
        for i in range(9):
            result[i, j] = derivative[i % 3, i // 3]
    return result


@wp.func
def _evaluate_stable_neo_hookean(F: Any, rest_volume: Any, mu_lame: Any, lambda_lame: Any, tangent: bool = True):
    one = type(rest_volume)(1.0)
    half = type(rest_volume)(0.5)
    j_minus_one = wp.determinant(F) - one
    lambda_nh = lambda_lame + mu_lame
    cofactor = _compute_cofactor(F)
    squared_norm = type(rest_volume)(0.0)
    projected = wp.matrix(shape=(9, 9), dtype=type(rest_volume))
    for i in range(9):
        squared_norm += F[i % 3, i // 3] * F[i % 3, i // 3]
        if tangent:
            for j in range(9):
                projected[i, j] = rest_volume * lambda_nh * cofactor[i % 3, i // 3] * cofactor[j % 3, j // 3]
            projected[i, i] += rest_volume * mu_lame
    energy = rest_volume * (
        half * mu_lame * (squared_norm - type(rest_volume)(3.0))
        - mu_lame * j_minus_one
        + half * lambda_nh * j_minus_one * j_minus_one
    )
    stress = rest_volume * (mu_lame * F + (lambda_nh * j_minus_one - mu_lame) * cofactor)
    return energy, stress, projected


@wp.func
def _evaluate_smith(F: Any, rest_volume: Any, mu: Any, lam: Any, tangent: bool = True):
    one = type(rest_volume)(1.0)
    half = type(rest_volume)(0.5)
    mu_hat = type(rest_volume)(4.0) / type(rest_volume)(3.0) * mu
    lam_hat = lam + type(rest_volume)(5.0) / type(rest_volume)(6.0) * mu
    ic = wp.ddot(F, F)
    d = wp.determinant(F) - one
    c = _compute_cofactor(F)
    s = lam_hat * d - mu
    a = mu_hat * (one - one / (ic + one))
    # Rest-shifted energy avoids subtracting the large alpha-dependent constant.
    energy = rest_volume * (
        half * mu_hat * (ic - type(rest_volume)(3.0) - wp.log((ic + one) / type(rest_volume)(4.0)))
        + half * lam_hat * d * d
        - mu * d
    )
    stress = rest_volume * (a * F + s * c)
    raw = wp.matrix(shape=(9, 9), dtype=type(rest_volume))
    if tangent:
        dc = _compute_cofactor_derivative(F)
        for i in range(9):
            for j in range(9):
                raw[i, j] = rest_volume * (
                    type(rest_volume)(2.0) * mu_hat / ((ic + one) * (ic + one)) * F[i % 3, i // 3] * F[j % 3, j // 3]
                    + lam_hat * c[i % 3, i // 3] * c[j % 3, j // 3]
                    + s * dc[i, j]
                )
            raw[i, i] += rest_volume * a
    return energy, stress, raw


@wp.func
def _project_element_psd(raw: Any, epsilon: Any):
    """Clamp the spectrum of a symmetric 12x12 nodal Hessian at zero by cyclic Jacobi.

    Sweeps stop once every off-diagonal entry is within ``epsilon`` (the
    precision's unit roundoff) of the largest diagonal magnitude; entries ten
    times smaller are not rotated, and convergence is reported at eight
    roundoffs. These relative tolerances are the same in either precision.
    """
    h = raw
    vectors = wp.identity(n=12, dtype=type(epsilon))
    tiny = type(epsilon)(1.0e-20)
    for _sweep in range(20):
        largest = type(epsilon)(0.0)
        scale = type(epsilon)(0.0)
        for i in range(12):
            scale = wp.max(scale, wp.abs(h[i, i]))
            for j in range(i + 1, 12):
                largest = wp.max(largest, wp.abs(h[i, j]))
        if largest <= epsilon * wp.max(scale, tiny):
            break
        for p in range(11):
            for q in range(p + 1, 12):
                off = h[p, q]
                if wp.abs(off) <= type(epsilon)(0.1) * epsilon * wp.max(scale, tiny):
                    continue
                theta = type(epsilon)(0.5) * wp.atan2(type(epsilon)(2.0) * off, h[q, q] - h[p, p])
                c, s = wp.cos(theta), wp.sin(theta)
                hp, hq = h[p, p], h[q, q]
                h[p, p] = c * c * hp - type(epsilon)(2.0) * c * s * off + s * s * hq
                h[q, q] = s * s * hp + type(epsilon)(2.0) * c * s * off + c * c * hq
                h[p, q] = type(epsilon)(0.0)
                h[q, p] = type(epsilon)(0.0)
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
    result = wp.matrix(shape=(12, 12), dtype=type(epsilon))
    for i in range(12):
        for j in range(i, 12):
            value = type(epsilon)(0.0)
            for k in range(12):
                value += vectors[i, k] * wp.max(h[k, k], type(epsilon)(0.0)) * vectors[j, k]
            result[i, j] = value
            result[j, i] = value
    remaining = type(epsilon)(0.0)
    scale = type(epsilon)(0.0)
    for i in range(12):
        scale = wp.max(scale, wp.abs(h[i, i]))
        for j in range(i + 1, 12):
            remaining = wp.max(remaining, wp.abs(h[i, j]))
    return result, remaining <= type(epsilon)(8.0) * epsilon * wp.max(scale, tiny)


@wp.func
def _nodal_hessian(raw: Any, pose: Any):
    h = wp.matrix(shape=(12, 12), dtype=type(pose[0, 0]))
    for a in range(4):
        ga = _shape_gradient(pose, a)
        for b in range(4):
            gb = _shape_gradient(pose, b)
            for row in range(3):
                for col in range(3):
                    value = type(pose[0, 0])(0.0)
                    for i in range(3):
                        for j in range(3):
                            value += ga[i] * raw[3 * i + row, 3 * j + col] * gb[j]
                    h[3 * a + row, 3 * b + col] = value
    return type(pose[0, 0])(0.5) * (h + wp.transpose(h))


@wp.func
def _shape_gradient(inverse_rest: Any, node: int):
    if node == 0:
        return -(inverse_rest[0, :] + inverse_rest[1, :] + inverse_rest[2, :])
    return inverse_rest[node - 1, :]


@wp.func
def _evaluate_energy_stress(f: Any, volume: Any, mu: Any, lam: Any, material: int):
    ic = wp.ddot(f, f)
    d = wp.determinant(f) - type(volume)(1.0)
    cof = wp.matrix_from_cols(wp.cross(f[:, 1], f[:, 2]), wp.cross(f[:, 2], f[:, 0]), wp.cross(f[:, 0], f[:, 1]))
    a = mu
    s = (lam + mu) * d - mu
    energy = type(volume)(0.5) * mu * (ic - type(volume)(3.0)) - mu * d + type(volume)(0.5) * (lam + mu) * d * d
    if material == 1:
        mh = type(volume)(4.0) * mu / type(volume)(3.0)
        lh = lam + type(volume)(5.0) * mu / type(volume)(6.0)
        a = mh * (type(volume)(1.0) - type(volume)(1.0) / (ic + type(volume)(1.0)))
        s = lh * d - mu
        energy = (
            type(volume)(0.5) * mh * (ic - type(volume)(3.0) - wp.log((ic + type(volume)(1.0)) / type(volume)(4.0)))
            + type(volume)(0.5) * lh * d * d
            - mu * d
        )
    return volume * energy, volume * (a * f + s * cof)


@wp.func
def _elastic_state(
    tet: int,
    material_model: int,
    min_det_f_guard: Any,
    candidate_particle_q: wp.array[Any],
    tet_indices: wp.array2d[int],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    tet_energy: wp.array[Any],
    min_det_f: wp.array[Any],
    failure_flags: wp.array[int],
    tangent: bool,
):
    """Evaluate one element in the precision of its candidate positions.

    Energy and stress use the shared formula of the native objective; the
    material functions supply only the tangent.
    """
    origin = candidate_particle_q[tet_indices[tet, 0]]
    ds = wp.matrix_from_cols(
        candidate_particle_q[tet_indices[tet, 1]] - origin,
        candidate_particle_q[tet_indices[tet, 2]] - origin,
        candidate_particle_q[tet_indices[tet, 3]] - origin,
    )
    pose = wp.matrix(tet_poses[tet], shape=(3, 3), dtype=type(min_det_f_guard))
    f = ds * pose
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
    rest_volume = _compute_rest_volume(pose)
    if not wp.isfinite(rest_volume) or rest_volume <= type(min_det_f_guard)(0.0):
        wp.atomic_max(failure_flags, 0, 2)
        valid = False
    mu = type(min_det_f_guard)(tet_materials[tet, 0])
    lam = type(min_det_f_guard)(tet_materials[tet, 1])
    energy, stress = _evaluate_energy_stress(f, rest_volume, mu, lam, material_model)
    _e, _s, projected = _evaluate_stable_neo_hookean(f, rest_volume, mu, lam, tangent)
    if material_model == 1:
        _e, _s, projected = _evaluate_smith(f, rest_volume, mu, lam, tangent)
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
