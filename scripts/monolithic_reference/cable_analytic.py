# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continuum benchmarks independent of the production DER discretization.

Localized-helix relations follow van der Heijden & Thompson (2000),
section 7.3, equations 69-73, expressed in dimensional arc length.
https://doi.org/10.1023/A:1008310425967
The boundary-controlled long-rod example is DER (2008), section 9.1:
https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf
"""

import numpy as np


def cantilever_deflection(s, *, length, force, bending):
    """Return small-deflection Euler-Bernoulli tip-load displacement [m]."""
    s = np.asarray(s, dtype=float)
    if not np.isfinite([length, force, bending]).all() or length <= 0 or bending <= 0:
        raise ValueError("Cantilever requires finite length/load and positive length/rigidity")
    if not np.isfinite(s).all() or np.any(s < 0) or np.any(s > length):
        raise ValueError("Arc length must lie inside the beam")
    return force * s**2 * (3 * length - s) / (6 * bending)


def localized_helix(s, *, length, bending, twisting, rotation, shortening):
    """Return the positive-torque localized branch with maximum tilt below pi/2.

    Parameters come solely from prescribed end rotation/shortening and material
    constants. No fitted simulation angle or torque enters the reference.
    The long-rod approximation's finite-tail errors are returned explicitly.
    Positions are centered at s=0; the axis is +x and the transverse phase is 0.
    """
    values = np.asarray([length, bending, twisting, rotation, shortening], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0) or shortening >= length:
        raise ValueError("Localized helix requires positive finite parameters and 0 < shortening < length")
    s = np.asarray(s, dtype=float)
    if not np.isfinite(s).all() or np.any(abs(s) > length / 2):
        raise ValueError("Centered arc length must lie inside the rod")

    def equation(angle):
        return shortening * twisting * (rotation - 2 * angle) - 4 * bending * length * np.sin(angle)

    low, high = 0.0, np.pi / 2
    if equation(low) <= 0 or equation(high) >= 0:
        raise ValueError("Boundary data do not select the specified localized branch")
    for _ in range(64):
        middle = (low + high) / 2
        if equation(middle) > 0:
            low = middle
        else:
            high = middle
    angle = (low + high) / 2
    torque = twisting * (rotation - 2 * angle) / length
    if torque <= 0:
        raise ValueError("Selected branch must have positive torque")
    b = torque / (2 * bending)
    a = b * np.tan(angle / 2)
    amplitude = 2 * a / (a * a + b * b)
    sech, tanh = 1 / np.cosh(a * s), np.tanh(a * s)
    c, sn = np.cos(b * s), np.sin(b * s)
    positions = np.stack((s - amplitude * tanh, amplitude * sech * c, amplitude * sech * sn), axis=-1)
    tangents = np.stack(
        (
            1 - amplitude * a * sech**2,
            amplitude * sech * (-a * tanh * c - b * sn),
            amplitude * sech * (-a * tanh * sn + b * c),
        ),
        axis=-1,
    )
    tail = np.tanh(a * length / 2)
    writhe_angle = 4 * np.arctan(np.tan(angle / 2) * tail)
    return {
        "positions": positions,
        "tangents": tangents,
        "max_tilt": angle,
        "torque": torque,
        "tension": bending * (a * a + b * b),
        "twist_density": torque / twisting,
        "localization_length": 1 / a,
        "envelope": tanh**2,
        "writhe_turns": writhe_angle / (2 * np.pi),
        "finite_shortening": 2 * amplitude * tail,
        "finite_end_rotation": torque * length / twisting + writhe_angle,
        "end_transverse_radius": amplitude / np.cosh(a * length / 2),
    }
