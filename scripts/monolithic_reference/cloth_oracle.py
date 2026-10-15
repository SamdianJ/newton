# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference for stable Neo-Hookean membrane and Mochi six-node curvature energy.

The curvature convention follows mochi_core/element_operations/fem_shell.h
and fem_shell.cpp in the workspace's superdex-reference-54ae749a checkout.
This small oracle imports no Newton/Warp code. Complex-step differentiation
provides an independent residual reference, not a production solver kernel.
PSD tangents below are independent references for the production approximation;
inertia, damping, contact and Warp assembly are outside this module.
"""

import numpy as np


def _extrapolate(points, boundary):
    result = points.copy()
    for opposite, missing in enumerate(boundary):
        if missing:
            result[opposite + 3] = points[(opposite + 1) % 3] + points[(opposite + 2) % 3] - points[opposite]
    return result


def _normals(points):
    triangles = points[np.array([[0, 1, 2], [3, 2, 1], [4, 0, 2], [5, 1, 0]])]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normals = np.cross(a, b)
    # No conjugation: preserve the analytic extension used by complex-step differentiation.
    return normals / np.sqrt(np.sum(normals * normals, axis=1))[:, None]


def _curvature(points):
    normals = _normals(points)
    means = (normals[0] + normals[1:]) / 2
    averaged = means / np.sum(means * means, axis=1)[:, None]
    edge0, edge2 = points[1] - points[0], points[0] - points[2]
    off_diagonal = 2 * averaged[0] @ edge2
    return np.array(
        [
            [2 * (averaged[1] - averaged[0]) @ edge0, off_diagonal],
            [off_diagonal, 2 * (averaged[0] - averaged[2]) @ edge2],
        ]
    )


def _check_geometry(points):
    triangles = points[np.array([[0, 1, 2], [3, 2, 1], [4, 0, 2], [5, 1, 0]])]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    relative_floor = 64 * np.finfo(np.float64).eps
    area = np.linalg.norm(np.cross(a, b), axis=1)
    if np.any(area <= relative_floor * np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)):
        raise ValueError("Degenerate shell stencil triangle")
    normals = _normals(points)
    sums = normals[0] + normals[1:]
    if np.any(np.sum(sums * sums, axis=1) <= relative_floor):
        raise ValueError("Opposing shell normals make curvature singular")


def _prepare(reference, positions, material, boundary):
    for points in (reference, positions):
        if np.iscomplexobj(points):
            raise ValueError("Shell coordinates must be real")
    reference, positions = np.asarray(reference, dtype=np.float64), np.asarray(positions, dtype=np.float64)
    if any(points.shape != (6, 3) or not np.isfinite(points).all() for points in (reference, positions)):
        raise ValueError("Shell coordinates require six finite 3D vertices")
    boundary = np.asarray(boundary)
    if boundary.shape != (3,) or boundary.dtype != np.bool_:
        raise ValueError("Shell boundary requires three boolean flags")
    material = np.asarray(material, dtype=np.float64)
    if material.shape != (4,) or not np.isfinite(material).all():
        raise ValueError("Shell material requires four finite coefficients")
    lam, mu, alpha, beta = material
    if mu <= 0 or lam <= -mu or beta <= 0 or alpha <= -beta / 2:
        raise ValueError("Shell material must have positive membrane and bending response")
    reference = _extrapolate(reference, boundary)
    _check_geometry(reference)
    _check_geometry(_extrapolate(positions, boundary))
    edges = (reference[1:3] - reference[0]).T
    metric = edges.T @ edges
    area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1])) / 2
    return reference, positions, material, boundary, np.linalg.inv(metric), area


def _energy(reference, positions, material, boundary, inverse, area):
    points = _extrapolate(positions, boundary)
    edges = (points[1:3] - points[0]).T
    gradient = edges @ np.linalg.cholesky(inverse)
    current_metric = gradient.T @ gradient
    normal = np.cross(edges[:, 0], edges[:, 1])
    jacobian = np.sqrt(normal @ normal) / (2 * area)
    # tr(C)-2J = ((C00-C11)^2+4*C01^2)/(tr(C)+2J). Avoid cancellation
    # near rest and in small simple shear; do not clamp the area or its derivatives.
    distortion = ((current_metric[0, 0] - current_metric[1, 1]) ** 2 + 4 * current_metric[0, 1] ** 2) / (
        np.trace(current_metric) + 2 * jacobian
    )
    curvature = inverse @ (_curvature(reference) - _curvature(points))
    lam, mu, alpha, beta = material
    membrane = area / 2 * (mu * distortion + (lam + mu) * (jacobian - 1) ** 2)
    bending = area / 2 * (alpha * np.trace(curvature) ** 2 + beta * np.trace(curvature @ curvature))
    return np.array([membrane, bending])


def shell_energy(reference, positions, material, *, boundary=(False, False, False)):
    """Return membrane and bending energies [J] for a single reference triangle.

    Args:
        reference: Six rest positions [m]; nodes 3+i lie opposite node i across its opposite edge.
        positions: Six current positions [m], in the same stencil order.
        material: Two-dimensional Lame lambda/mu [N/m], followed by bending alpha/beta [N m].
        boundary: Missing-neighbor flags; missing coordinates are replaced by boundary extrapolation.

    Returns:
        Two energies for the central triangle only; adjacent triangles own their own terms.
    """
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = _energy(*_prepare(reference, positions, material, boundary))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite shell energy")
    return result


def shell_residual(reference, positions, material, *, boundary=(False, False, False)):
    """Differentiate the energy in all 18 coordinates by complex steps; return dE/dx [N]."""
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        reference, positions, material, boundary, inverse, area = _prepare(reference, positions, material, boundary)
        result = np.empty((6, 3))
        for index in range(18):
            trial = positions.astype(np.complex128)
            trial.flat[index] += 1e-30j
            result.flat[index] = _energy(reference, trial, material, boundary, inverse, area).sum().imag / 1e-30
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite shell residual")
    return result


def shell_membrane_derivatives(reference, positions, material):
    """Return analytic membrane dE/dx [N] and the exact 9x9 Hessian [N/m].

    Use the same six-node inputs as :func:`shell_energy`, but only the central
    triangle contributes. Neighbor slots are extrapolated and ignored. Return
    a (3, 3) residual and a (9, 9) matrix in node-major xyz order. The Hessian
    includes geometric stiffness and can be indefinite under compression;
    it is not the production PSD tangent or a curvature-bending derivative.
    """
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        _, positions, material, _, inverse, area = _prepare(reference, positions, material, (True, True, True))
        edges = (positions[1:3] - positions[0]).T
        normal = np.cross(edges[:, 0], edges[:, 1])
        norm = np.linalg.norm(normal)
        unit = normal / norm
        jacobian = norm / (2 * area)
        incidence = np.array([[-1, -1], [1, 0], [0, 1]])
        edge_derivatives = np.kron(incidence.T, np.eye(3))
        de, df = edge_derivatives[:3].T, edge_derivatives[3:].T
        dn = np.cross(de, edges[:, 1]) + np.cross(edges[:, 0], df)
        dj = dn @ unit / (2 * area)
        d2n = np.cross(de[:, None, :], df[None, :, :]) + np.cross(de[None, :, :], df[:, None, :])
        d2j = (dn @ (np.eye(3) - np.outer(unit, unit)) @ dn.T / norm + d2n @ unit) / (2 * area)
        lam, mu = material[:2]
        pressure = (lam + mu) * (jacobian - 1) - mu
        residual = area * (mu * incidence @ inverse @ edges.T + pressure * dj.reshape(3, 3))
        hessian = area * (
            mu * np.kron(incidence @ inverse @ incidence.T, np.eye(3)) + (lam + mu) * np.outer(dj, dj) + pressure * d2j
        )
    if not np.isfinite(residual).all() or not np.isfinite(hessian).all():
        raise ValueError("Nonfinite membrane derivatives")
    return residual, hessian


def shell_membrane_tangent(reference, positions, material):
    """Project the complete 6x6 material Hessian, then map to all nine vertex DoFs."""
    _, positions, material, _, inverse, area = _prepare(reference, positions, material, (True, True, True))
    transform = np.linalg.cholesky(inverse)
    gradient = (positions[1:3] - positions[0]).T @ transform
    f, g = gradient.T
    normal = np.cross(f, g)
    jacobian = np.linalg.norm(normal)
    unit = normal / jacobian
    df, dg = np.eye(6)[:, :3], np.eye(6)[:, 3:]
    dn = np.cross(df, g) + np.cross(f, dg)
    dj = dn @ unit
    d2n = np.cross(df[:, None, :], dg[None, :, :]) + np.cross(df[None, :, :], dg[:, None, :])
    d2j = dn @ (np.eye(3) - np.outer(unit, unit)) @ dn.T / jacobian + d2n @ unit
    lam, mu = material[:2]
    density_hessian = mu * np.eye(6) + (lam + mu) * np.outer(dj, dj) + ((lam + mu) * (jacobian - 1) - mu) * d2j
    values, vectors = np.linalg.eigh(density_hessian)
    projected = (vectors * np.maximum(values, 0)) @ vectors.T
    incidence = np.array([[-1, -1], [1, 0], [0, 1]])
    mapping = np.kron((incidence @ transform).T, np.eye(3))
    return area * mapping.T @ projected @ mapping


def shell_bending_tangent(reference, positions, material, *, boundary=(False, False, False)):
    """Return the full 18x18 curvature material tangent; omit geometric stiffness."""
    reference, positions, material, boundary, inverse, area = _prepare(reference, positions, material, boundary)
    transform = np.linalg.cholesky(inverse)
    jacobian = np.empty((3, 18))
    for index in range(18):
        trial = positions.astype(np.complex128)
        trial.flat[index] += 1e-30j
        curvature = transform.T @ _curvature(_extrapolate(trial, boundary)) @ transform
        jacobian[:, index] = np.array([curvature[0, 0], curvature[1, 1], np.sqrt(2) * curvature[0, 1]]).imag / 1e-30
    alpha, beta = material[2:]
    constitutive = beta * np.eye(3) + alpha * np.outer([1, 1, 0], [1, 1, 0])
    return area * jacobian.T @ constitutive @ jacobian
