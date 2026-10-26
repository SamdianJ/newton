# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed-form references for the tet patch, cantilever and axial-vibration benchmarks.

At small strain both Kim and Smith stable Neo-Hookean energies reduce to
mu * eps:eps + lambda / 2 * tr(eps)**2, so the cantilever and vibration
references apply to either model. Large-strain patch stresses use the
independent NumPy material oracle.
"""

import math
from itertools import pairwise

import numpy as np

from .tet_oracle import material


def young_poisson(mu, lam):
    """Return the small-strain Young's modulus [Pa] and Poisson ratio of Lamé parameters [Pa]."""
    if not mu > 0 or not lam >= 0:
        raise ValueError("Lamé parameters require mu > 0 and lambda >= 0")
    return mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))


def homogeneous_traction(f, mu, lam, *, smith):
    """Return the first Piola stress [Pa] of a homogeneous deformation gradient."""
    f = np.asarray(f, dtype=float)
    if f.shape != (3, 3) or not np.linalg.det(f) > 0:
        raise ValueError("Deformation gradient must be 3x3 with a positive determinant")
    return material(f, mu, lam, smith)[1]


def timoshenko_tip_deflection(force, length, width, height, mu, lam):
    """Return the small-deflection tip displacement [m] of a rectangular end-loaded cantilever.

    Uses Cowper's shear coefficient for a rectangular section.
    """
    young, nu = young_poisson(mu, lam)
    inertia = width * height**3 / 12
    shear = 10 * (1 + nu) / (12 + 11 * nu)
    return force * length**3 / (3 * young * inertia) + force * length / (shear * mu * width * height)


def fixed_free_frequency(length, mu, lam, density):
    """Return the fundamental axial angular frequency [rad/s] of a fixed-free zero-Poisson bar."""
    young, nu = young_poisson(mu, lam)
    if nu != 0:
        raise ValueError("The rod frequency is exact only for zero Poisson ratio")
    return 0.5 * math.pi * math.sqrt(young / density) / length


def implicit_euler_oscillation(omega, dt):
    """Return the implicit-Euler period [s] and amplitude ratio per period of an undamped mode."""
    if not omega > 0 or not dt > 0:
        raise ValueError("Frequency and time step must be positive")
    steps = 2 * math.pi / math.atan(omega * dt)
    return steps * dt, (1 + (omega * dt) ** 2) ** (-steps / 2)


def oscillation_period_and_decay(signal, dt):
    """Measure the mean period [s] and amplitude ratio per period from a sampled oscillation."""
    x = np.asarray(signal, dtype=float)
    up = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    if len(up) < 3:
        raise ValueError("Need at least two complete periods")
    crossings = (up + x[up] / (x[up] - x[up + 1])) * dt
    peaks = np.array([x[a:b].max() for a, b in pairwise(up)])
    return float(np.mean(np.diff(crossings))), float(np.exp(np.mean(np.diff(np.log(peaks)))))
