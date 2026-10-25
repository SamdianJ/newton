# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Precision-aware numerical residual gates.

A frozen residual gate asks for an absolute force or torque. Below the solver's
rounding floor ``c_u eps_T a_i`` no precision can resolve a row, so each gate
admits a row up to the larger of its frozen value and that floor. In FP64 the
floor lies far below every frozen gate, which therefore stays unchanged; physical
gates (displacement, energy, analytic error, penetration) never use the floor.
"""

import numpy as np

from newton._src.solvers.monolithic.iteration import blocks


def rounding_floor(solver):
    """Return the per-row rounding floor ``c_u eps_T a_i`` [N or N m] of the last assessed iterate."""
    criterion = solver._criterion
    return criterion.noise.numpy().astype(float) * criterion.magnitude.numpy().astype(float)


def row_excess(residual, floor, gate):
    """Return ``max_i |r_i| / max(gate, floor_i)``; at most 1 passes."""
    residual = np.abs(np.asarray(residual, dtype=float))
    limit = np.maximum(gate, np.asarray(floor, dtype=float))
    return float(np.max(residual / limit, initial=0.0))


def block_norms(solver, residual, floor, name):
    """Return the L2 norms of the residual and of the rounding floor over one convergence block."""
    part = dict(blocks(solver))[name]
    return float(np.linalg.norm(np.asarray(residual, dtype=float)[part])), float(np.linalg.norm(floor[part]))
