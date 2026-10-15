# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy DER reference with time-parallel frames and quasistatic theta.

Equations follow Bergou et al., Discrete Viscous Threads (2010), sections 3-4:
https://www.cs.columbia.edu/cg/pdfs/171-threads.pdf . No external implementation
is imported or copied. This is a small mathematical oracle, not a stepper.
Residuals use complex steps; the PSD reference is the full strain Jacobian
Gauss-Newton matrix, including position/theta blocks and no theta inertia.
"""

import numpy as np


def _real_array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name} shape or nonfinite values")
    return value


def _geometry(positions):
    edges = np.diff(positions, axis=0)
    lengths = np.sqrt(np.sum(edges * edges, axis=1))
    if np.any(lengths.real <= 0):
        raise ValueError("Cable edges must have positive length")
    tangents = edges / lengths[:, None]
    if np.any(1 + np.sum(tangents[:-1] * tangents[1:], axis=1).real <= 1e-10):
        raise ValueError("Antiparallel adjacent cable tangents")
    return lengths, tangents


def _transport(source, target, directors):
    cosine = np.sum(source * target, axis=1)
    if np.any(1 + cosine.real <= 1e-10):
        raise ValueError("Antiparallel frame transport")
    axis = np.cross(source, target)
    first = np.cross(axis, directors)
    return directors + first + np.cross(axis, first) / (1 + cosine[:, None])


def _reference_twist(tangents, directors, anchor):
    transported = _transport(tangents[:-1], tangents[1:], directors[:-1])
    cosine = np.sum(transported * directors[1:], axis=1)
    sine = np.sum(np.cross(transported, directors[1:]) * tangents[1:], axis=1)
    # Rotate atan into a well-conditioned local chart for complex differentiation.
    angle = np.arctan2(sine.real, cosine.real)
    local = np.arctan((sine * np.cos(angle) - cosine * np.sin(angle)) / (cosine * np.cos(angle) + sine * np.sin(angle)))
    result = angle + local
    if anchor is not None:
        result += 2 * np.pi * np.rint((anchor - result.real) / (2 * np.pi))
        if np.any(np.abs(result.real - anchor) >= np.pi / 2):
            raise ValueError("Reference twist leaves the committed branch chart")
    return result


class CableFrame:
    """Own a copied time-transport anchor and unwrapped reference-twist history."""

    def __init__(self, positions, directors, *, reference_twist=None):
        count = len(positions)
        if count < 2:
            raise ValueError("Cable requires at least two vertices")
        self.positions = _real_array(positions, (count, 3), "frame positions")
        _, tangents = _geometry(self.positions)
        self.directors = _real_array(directors, (count - 1, 3), "frame directors")
        if not np.allclose(np.sum(self.directors * tangents, axis=1), 0, atol=1e-12, rtol=0) or not np.allclose(
            np.linalg.norm(self.directors, axis=1), 1, atol=1e-12, rtol=0
        ):
            raise ValueError("Cable directors must be unit and perpendicular to their tangents")
        anchor = None if reference_twist is None else _real_array(reference_twist, (count - 2,), "reference twist")
        self.reference_twist = _reference_twist(tangents, self.directors, anchor)
        if anchor is not None and not np.allclose(self.reference_twist, anchor, atol=1e-12, rtol=0):
            raise ValueError("Reference twist disagrees with frame directors")
        for value in (self.positions, self.directors, self.reference_twist):
            value.setflags(write=False)


def _trial_frame(frame, positions):
    _, old_tangents = _geometry(frame.positions)
    lengths, tangents = _geometry(positions)
    directors = _transport(old_tangents, tangents, frame.directors)
    twist = _reference_twist(tangents, directors, frame.reference_twist)
    return lengths, tangents, directors, twist


def advance_frame(frame, positions):
    """Return new frame history without changing the anchor, including on invalid trials."""
    positions = _real_array(positions, frame.positions.shape, "trial positions")
    _, _, directors, twist = _trial_frame(frame, positions)
    return CableFrame(positions, directors, reference_twist=twist)


def _strains(positions, theta, frame):
    lengths, tangents, directors, reference_twist = _trial_frame(frame, positions)
    perpendicular = np.cross(tangents, directors)
    m1 = np.cos(theta[:, None]) * directors + np.sin(theta[:, None]) * perpendicular
    m2 = -np.sin(theta[:, None]) * directors + np.cos(theta[:, None]) * perpendicular
    kb = 2 * np.cross(tangents[:-1], tangents[1:]) / (1 + np.sum(tangents[:-1] * tangents[1:], axis=1))[:, None]
    curvature = np.column_stack((np.sum(kb * (m2[:-1] + m2[1:]), axis=1), -np.sum(kb * (m1[:-1] + m1[1:]), axis=1))) / 2
    return lengths, curvature, np.diff(theta) + reference_twist


class CableReference:
    """Own rest strains and (EA, bending tensor, GJ) [N, N m², N m²].

    The bending tensor is a symmetric positive-definite 2x2 material tensor.
    A positive scalar is shorthand for isotropic EI times the identity.
    """

    def __init__(self, positions, theta, directors, stiffness):
        frame = CableFrame(positions, directors)
        theta = _real_array(theta, (len(positions) - 1,), "rest theta")
        if len(stiffness) != 3:
            raise ValueError("Cable stiffness requires EA, a bending tensor and GJ")
        axial_twist = _real_array([stiffness[0], stiffness[2]], (2,), "EA/GJ")
        bending = stiffness[1]
        if np.ndim(bending) == 0:
            bending = np.eye(2) * bending
        bending = _real_array(bending, (2, 2), "bending tensor")
        if np.any(axial_twist <= 0) or not np.array_equal(bending, bending.T) or np.linalg.eigvalsh(bending).min() <= 0:
            raise ValueError("EA/GJ must be positive and bending tensor must be symmetric positive definite")
        self.stiffness = (axial_twist[0], bending, axial_twist[1])
        self.lengths, self.curvature, self.twist = _strains(frame.positions, theta, frame)
        self.dual_lengths = (self.lengths[:-1] + self.lengths[1:]) / 2
        for value in (bending, self.lengths, self.curvature, self.twist, self.dual_lengths):
            value.setflags(write=False)


def _prepare(rest, positions, theta, frame):
    count = len(rest.lengths) + 1
    positions = _real_array(positions, (count, 3), "positions")
    theta = _real_array(theta, (count - 1,), "theta")
    if frame.positions.shape != positions.shape:
        raise ValueError("Frame and cable topology differ")
    lengths, _ = _geometry(positions)
    if np.any(lengths / rest.lengths <= 1e-8):
        raise ValueError("Cable trial edge collapses relative to rest length")
    _trial_frame(frame, positions)
    return positions, theta


def _weighted_strains(rest, positions, theta, frame):
    lengths, curvature, twist = _strains(positions, theta, frame)
    ea, bending, gj = rest.stiffness
    return (
        np.sqrt(ea / rest.lengths) * (lengths - rest.lengths),
        ((curvature - rest.curvature) @ np.linalg.cholesky(bending) / np.sqrt(rest.dual_lengths)[:, None]).ravel(),
        np.sqrt(gj / rest.dual_lengths) * (twist - rest.twist),
    )


def cable_energy(rest, positions, theta, frame):
    """Return stretch, bending and twisting energies [J] in a fixed temporal frame chart."""
    positions, theta = _prepare(rest, positions, theta, frame)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        strains = _weighted_strains(rest, positions, theta, frame)
        result = np.array([0.5 * np.dot(s, s) for s in strains])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cable energy")
    return result


def cable_derivatives(rest, positions, theta, frame):
    """Return dE/d(x,theta) [N, N m] and the coupled PSD strain-Jacobian tangent.

    Ordering is all node-major xyz coordinates followed by one unwrapped theta
    per edge. No gauge or physical boundary is eliminated by this oracle.
    The approximation omits strain-weighted second derivatives and is exact
    only at zero strain. Never add artificial theta inertia to this matrix.
    """
    positions, theta = _prepare(rest, positions, theta, frame)
    state = np.r_[positions.ravel(), theta]
    n = positions.size
    size = sum(s.size for s in _weighted_strains(rest, positions, theta, frame))
    jacobian = np.empty((size, len(state)))
    residual = np.empty(len(state))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for index in range(len(state)):
            trial = state.astype(np.complex128)
            trial[index] += 1e-30j
            strains = _weighted_strains(rest, trial[:n].reshape(-1, 3), trial[n:], frame)
            jacobian[:, index] = np.concatenate(strains).imag / 1e-30
            residual[index] = sum(0.5 * np.dot(s, s) for s in strains).imag / 1e-30
    tangent = jacobian.T @ jacobian
    if not np.isfinite(residual).all() or not np.isfinite(tangent).all():
        raise ValueError("Nonfinite cable derivatives")
    return residual, tangent
