# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent float64 surface-contact integration for P1Q3 calibration.

This module deliberately has no Newton or Warp dependency.  It evaluates analytic
SDFs and integrates the quadratic-hinge traction over the *same triangles* as a
production measurement.  A degree-five Dunavant rule is adaptively subdivided;
the production three-point records are never consumed by this oracle.
"""

from dataclasses import dataclass

import numpy as np

_BARYCENTRIC = np.asarray(
    (
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.059715871789770, 0.470142064105115, 0.470142064105115),
        (0.470142064105115, 0.059715871789770, 0.470142064105115),
        (0.470142064105115, 0.470142064105115, 0.059715871789770),
        (0.797426985353087, 0.101286507323456, 0.101286507323456),
        (0.101286507323456, 0.797426985353087, 0.101286507323456),
        (0.101286507323456, 0.101286507323456, 0.797426985353087),
    ),
    dtype=np.float64,
)
_WEIGHTS = np.asarray(
    (
        0.225,
        0.132394152788506,
        0.132394152788506,
        0.132394152788506,
        0.125939180544827,
        0.125939180544827,
        0.125939180544827,
    ),
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class ContactIntegral:
    """Reference-area integral of a normal quadratic hinge."""

    force_magnitude: float
    force_resultant: np.ndarray
    moment_resultant: np.ndarray
    consistent_nodal_forces: np.ndarray
    energy: float
    active_area: float
    maximum_penetration: float
    force_magnitude_absolute_error: float
    force_resultant_absolute_error: np.ndarray
    moment_resultant_absolute_error: np.ndarray
    energy_absolute_error: float
    active_area_absolute_error: float
    leaf_count: int


def _sdf(points: np.ndarray, shape: str, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = points - center
    if shape == "plane":
        return local[:, 2], np.broadcast_to(np.array((0.0, 0.0, 1.0)), points.shape).copy()
    if shape == "sphere":
        lengths = np.linalg.norm(local, axis=1)
        normals = np.zeros_like(local)
        nonzero = lengths > 0.0
        normals[nonzero] = local[nonzero] / lengths[nonzero, None]
        normals[~nonzero, 0] = 1.0
        return lengths - scale[0], normals
    if shape == "box":
        q = np.abs(local) - scale
        outside = np.maximum(q, 0.0)
        outside_length = np.linalg.norm(outside, axis=1)
        phi = outside_length + np.minimum(np.max(q, axis=1), 0.0)
        normals = np.zeros_like(local)
        outside_mask = outside_length > 0.0
        normals[outside_mask] = (
            np.sign(local[outside_mask]) * outside[outside_mask] / outside_length[outside_mask, None]
        )
        for index in np.flatnonzero(~outside_mask):
            axis = int(np.argmax(q[index]))
            normals[index, axis] = 1.0 if local[index, axis] >= 0.0 else -1.0
        return phi, normals
    raise ValueError(f"Unsupported analytic oracle shape: {shape}")


def _children(triangle: np.ndarray) -> tuple[np.ndarray, ...]:
    a, b, c = triangle
    ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
    return (
        np.asarray((a, ab, ca)),
        np.asarray((ab, b, bc)),
        np.asarray((ca, bc, c)),
        np.asarray((ab, bc, ca)),
    )


def _quadrature(
    triangle: np.ndarray,
    parent_barycentric: np.ndarray,
    shape: str,
    center: np.ndarray,
    scale: np.ndarray,
    particle_radius: float,
) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    points = _BARYCENTRIC @ triangle
    barycentric = _BARYCENTRIC @ parent_barycentric
    phi, normals = _sdf(points, shape, center, scale)
    penetration = np.maximum(particle_radius - phi, 0.0)
    force_density = -penetration[:, None] * normals
    nodal_force_density = (barycentric[:, :, None] * force_density[:, None, :]).reshape((-1, 9))
    moment_density = np.cross(points, force_density)
    # Scalar force, resultant, energy, origin moment, 3 consistent nodal
    # forces, and active area.  Each output keeps an independent error estimate.
    values = np.column_stack(
        (
            penetration,
            force_density,
            0.5 * penetration**2,
            moment_density,
            nodal_force_density,
            (penetration > 0.0).astype(np.float64),
        )
    )
    return area * (_WEIGHTS @ values)


def _integrate_triangle(
    triangle: np.ndarray,
    parent_barycentric: np.ndarray,
    shape: str,
    center: np.ndarray,
    scale: np.ndarray,
    particle_radius: float,
    depth: int,
    max_depth: int,
    relative_tolerance: float,
    force_absolute_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    centroid = triangle.mean(axis=0, keepdims=True)
    phi, _ = _sdf(centroid, shape, center, scale)
    reach = np.max(np.linalg.norm(triangle - centroid[0], axis=1))
    # Analytic SDFs are 1-Lipschitz.  This is a conservative zero-contact proof,
    # unlike deciding inactivity from quadrature samples alone.
    if phi[0] - reach >= particle_radius:
        return np.zeros(18), np.zeros(18), 1, 0.0

    coarse = _quadrature(triangle, parent_barycentric, shape, center, scale, particle_radius)
    children = _children(triangle)
    barycentric_children = _children(parent_barycentric)
    fine = sum(
        (
            _quadrature(child, barycentric, shape, center, scale, particle_radius)
            for child, barycentric in zip(children, barycentric_children, strict=True)
        ),
        np.zeros(18),
    )
    error = np.abs(fine - coarse)
    uncertain_zero = fine[0] == 0.0
    sample_phi, _ = _sdf(_BARYCENTRIC @ triangle, shape, center, scale)
    maximum_penetration = float(np.maximum(particle_radius - sample_phi, 0.0).max())
    if depth >= max_depth or (
        depth >= 2 and not uncertain_zero and error[0] <= force_absolute_tolerance + relative_tolerance * abs(fine[0])
    ):
        return fine, error, 4, maximum_penetration

    total = np.zeros(18)
    total_error = np.zeros(18)
    leaves = 0
    for child, barycentric in zip(children, barycentric_children, strict=True):
        value, child_error, child_leaves, child_maximum = _integrate_triangle(
            child,
            barycentric,
            shape,
            center,
            scale,
            particle_radius,
            depth + 1,
            max_depth,
            relative_tolerance,
            force_absolute_tolerance,
        )
        total += value
        total_error += child_error
        leaves += child_leaves
        maximum_penetration = max(maximum_penetration, child_maximum)
    return total, total_error, leaves, maximum_penetration


def integrate_contact_over_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    shape: str,
    center: np.ndarray,
    scale: np.ndarray,
    particle_radius: float,
    stiffness: float,
    max_depth: int = 7,
    relative_tolerance: float = 1.0e-6,
) -> ContactIntegral:
    """Integrate contact traction over triangles using only float64 geometry."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("points and faces must have shapes (n,3) and (m,3)")
    if max_depth < 2 or relative_tolerance <= 0.0 or particle_radius < 0.0 or stiffness <= 0.0:
        raise ValueError("invalid integration controls or contact parameters")

    total = np.zeros(18)
    error = np.zeros(18)
    nodal_forces = np.zeros((len(points), 3), dtype=np.float64)
    leaves = 0
    maximum_penetration = 0.0
    for face in faces:
        triangle = points[face]
        face_area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        value, face_error, face_leaves, face_maximum = _integrate_triangle(
            triangle,
            np.eye(3),
            shape,
            center,
            scale,
            particle_radius,
            0,
            max_depth,
            relative_tolerance,
            face_area * max(particle_radius, 1.0e-6) * 1.0e-6,
        )
        total += value
        error += face_error
        leaves += face_leaves
        nodal_forces[face] += value[8:17].reshape((3, 3))
        maximum_penetration = max(maximum_penetration, face_maximum)
    return ContactIntegral(
        force_magnitude=float(stiffness * total[0]),
        force_resultant=stiffness * total[1:4],
        energy=float(stiffness * total[4]),
        moment_resultant=stiffness * total[5:8],
        consistent_nodal_forces=stiffness * nodal_forces,
        active_area=float(total[17]),
        maximum_penetration=maximum_penetration,
        force_magnitude_absolute_error=float(stiffness * error[0]),
        force_resultant_absolute_error=stiffness * error[1:4],
        energy_absolute_error=float(stiffness * error[4]),
        moment_resultant_absolute_error=stiffness * error[5:8],
        active_area_absolute_error=float(error[17]),
        leaf_count=leaves,
    )
