# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Precision-aware numerical residual gates.

A frozen residual gate asks for an absolute force or torque. Each gate admits a
row up to the larger of its frozen value and a precision floor. In FP64 the floor
is the solver's rounding term ``c_u eps_T a_i``, which lies far below every frozen
gate, so FP64 checks stay unchanged. In FP32 it is the solver's complete row
tolerance: FP32 keeps a 1e-4 relative tolerance because its rounding term does not
yet bound near-rest tet elasticity, so its residual gates certify the solver's own
convergence (approved 2026-10-25). Physical gates (displacement, energy, analytic
error, penetration) never use the floor.
"""

import numpy as np

from newton._src.solvers.monolithic.iteration import blocks


def gate_floor(solver):
    """Return the per-row precision floor [N or N m] of the last assessed iterate.

    FP64 uses the rounding term ``c_u eps_T a_i``; FP32 the complete tolerance ``tau_i``.
    """
    criterion = solver._criterion
    if solver.precision == solver.Precision.FP32:
        return criterion.tau.numpy().astype(float)
    return criterion.noise.numpy().astype(float) * criterion.magnitude.numpy().astype(float)


def balance_rounding(solver):
    """Return ``c eps_T``, the relative rounding of contact wrenches summed in the solver precision."""
    return float(solver._criterion.noise.numpy().max())


def row_excess(residual, floor, gate):
    """Return ``max_i |r_i| / max(gate, floor_i)``; at most 1 passes."""
    residual = np.abs(np.asarray(residual, dtype=float))
    limit = np.maximum(gate, np.asarray(floor, dtype=float))
    return float(np.max(residual / limit, initial=0.0))


def block_norms(solver, residual, floor, name):
    """Return the L2 norms of the residual and of the rounding floor over one convergence block."""
    part = dict(blocks(solver))[name]
    return float(np.linalg.norm(np.asarray(residual, dtype=float)[part])), float(np.linalg.norm(floor[part]))
