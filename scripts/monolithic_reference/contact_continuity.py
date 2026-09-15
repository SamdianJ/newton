# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-input contact continuity prototypes; not a solver policy.

The patch sampler deliberately rejects queries without a complete cubic stencil.
No extrapolation, normal fallback, or production configuration is introduced.
"""

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact_continuity import sample_patch


@wp.kernel(enable_backward=False)
def query_patch(
    values: wp.array3d[float],
    coordinates: wp.array[wp.vec3],
    inverse_dx: wp.vec3,
    result: wp.array[wp.vec4],
    valid: wp.array[int],
):
    i = wp.tid()
    phi, gradient, ok = sample_patch(values, coordinates[i], inverse_dx)
    result[i] = wp.vec4(phi, gradient[0], gradient[1], gradient[2])
    valid[i] = int(ok)


def patch_oracle(values, coordinate, inverse_dx):
    """Evaluate FP64 tensor Hermite interpolation with centered node slopes.

    This uses nested Hermite polynomials rather than the Warp weight sum.
    """
    p = np.asarray(coordinate, dtype=np.float64)
    cell = np.floor(p).astype(int)
    if np.any(cell < 1) or np.any(cell + 2 >= values.shape):
        raise ValueError("Incomplete cubic stencil")
    block = np.asarray(values, dtype=np.float64)[tuple(slice(c - 1, c + 3) for c in cell)]

    def hermite(a, t, derivative=False):
        a0, a1, a2, a3 = a
        m1, m2 = (a2 - a0) / 2, (a3 - a1) / 2
        if derivative:
            return (
                (6 * t * t - 6 * t) * a1
                + (3 * t * t - 4 * t + 1) * m1
                + (-6 * t * t + 6 * t) * a2
                + (3 * t * t - 2 * t) * m2
            )
        return (
            (2 * t**3 - 3 * t * t + 1) * a1
            + (t**3 - 2 * t * t + t) * m1
            + (-2 * t**3 + 3 * t * t) * a2
            + (t**3 - t * t) * m2
        )

    result = []
    for derivative in (-1, 0, 1, 2):
        a = block
        for axis in range(3):
            a = hermite(a, p[axis] - cell[axis], derivative == axis)
        result.append(float(a))
    result[1:] = np.asarray(result[1:]) * inverse_dx
    return np.asarray(result)
