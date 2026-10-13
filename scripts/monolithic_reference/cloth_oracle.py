# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference for Mochi's StVK membrane and six-node curvature energy.

The curvature convention follows mochi_core/element_operations/fem_shell.h
and fem_shell.cpp in the workspace's superdex-reference-54ae749a checkout.
This small oracle imports no Newton/Warp code. Complex-step differentiation
provides an independent residual reference, not a production solver kernel.
Inertia, damping, contact and a production PSD tangent are outside this module.
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
    return reference, positions, material, boundary, metric, np.linalg.inv(metric), area


def _energy(reference, positions, material, boundary, metric, inverse, area):
    points = _extrapolate(positions, boundary)
    edges = (points[1:3] - points[0]).T
    strain = inverse @ ((edges.T @ edges - metric) / 2)
    curvature = inverse @ (_curvature(reference) - _curvature(points))
    lam, mu, alpha, beta = material
    membrane = area * (lam / 2 * np.trace(strain) ** 2 + mu * np.trace(strain @ strain))
    bending = area / 2 * (alpha * np.trace(curvature) ** 2 + beta * np.trace(curvature @ curvature))
    return np.array([membrane, bending])


def shell_energy(reference, positions, material, *, boundary=(False, False, False)):
    """Return membrane and bending energies [J] for a single reference triangle.

    Args:
        reference: Six rest positions [m]; nodes 3+i lie opposite node i across its opposite edge.
        positions: Six current positions [m], in the same stencil order.
        material: Membrane lambda/mu [N/m], followed by bending alpha/beta [N m].
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
        reference, positions, material, boundary, metric, inverse, area = _prepare(
            reference, positions, material, boundary
        )
        result = np.empty((6, 3))
        for index in range(18):
            trial = positions.astype(np.complex128)
            trial.flat[index] += 1e-30j
            result.flat[index] = _energy(reference, trial, material, boundary, metric, inverse, area).sum().imag / 1e-30
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite shell residual")
    return result
