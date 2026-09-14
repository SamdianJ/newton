# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tiny FP64 oracle for SCHUR-SPD-01; never used by the production solver."""

import itertools

import numpy as np


def _inverse_spd(matrix):
    factor = np.linalg.cholesky(matrix)
    return np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(len(matrix))))


def patch_oracle(base, rows, weights, block_sizes, patches):
    """Compare exact Woodbury with a Schur upper bound built from support overlap.

    Args:
        base: Contact-free, scaled block diagonal SPD matrix, including lambda once.
        rows: Scaled normal rows, with all fixed degrees of freedom removed.
        weights: Nonnegative normal weights; zero rows have no effect.
        block_sizes: Contiguous base block sizes (dense q, then dynamic node blocks).
        patches: Disjoint normal-row index lists covering every row exactly once.

    Returns:
        Tiny dense matrices for independent SPD and projection comparisons.
    """
    base, rows, weights = (np.asarray(value, dtype=np.float64) for value in (base, rows, weights))
    n = len(base)
    m = len(rows)
    if not 0 < n <= 64 or m > 128:
        raise ValueError("Dense oracle is restricted to at most 64 DoFs and 128 rows")
    if base.shape != (n, n) or rows.shape != (m, n) or weights.shape != (m,):
        raise ValueError("Inconsistent base/normal row/weight shapes")
    if not all(np.isfinite(value).all() for value in (base, rows, weights)) or np.any(weights < 0):
        raise ValueError("Require finite data and nonnegative normal weights")
    if not np.array_equal(base, base.T):
        raise ValueError("Base must be symmetric")
    if any(type(size) is not int or size <= 0 for size in block_sizes) or sum(block_sizes) != n:
        raise ValueError("Base blocks must cover all dynamic DoFs")
    flat = [index for patch in patches for index in patch]
    if any(type(index) is not int for index in flat) or sorted(flat) != list(range(m)) or any(not p for p in patches):
        raise ValueError("Patches must partition normal rows without duplicates or omissions")
    edges = np.cumsum([0, *block_sizes])
    blocks = [slice(a, b) for a, b in itertools.pairwise(edges)]
    off_diagonal = base.copy()
    for block in blocks:
        off_diagonal[block, block] = 0
    if np.any(off_diagonal):
        raise ValueError("Base must be block diagonal in the declared partition")
    inverse_base = _inverse_spd(base)
    u = np.sqrt(weights)[:, None] * rows
    # Each block counts distinct incident patches, not rows or individual q coordinates.
    support = np.asarray([[np.any(u[patch, block]) for block in blocks] for patch in patches], dtype=bool)
    degrees = support.sum(axis=0) if patches else np.zeros(len(blocks), dtype=int)
    schur = np.eye(m) + u @ inverse_base @ u.T
    bound = np.zeros((m, m))
    approximate = inverse_base.copy()
    naive = inverse_base.copy()
    for patch in patches:
        local_u = u[patch]
        local = np.eye(len(patch))
        for degree, block in zip(degrees, blocks, strict=True):
            local += degree * local_u[:, block] @ inverse_base[block, block] @ local_u[:, block].T
        bound[np.ix_(patch, patch)] = local
        lifted = inverse_base @ local_u.T
        approximate -= lifted @ _inverse_spd(local) @ lifted.T
        naive -= lifted @ _inverse_spd(schur[np.ix_(patch, patch)]) @ lifted.T
    return {
        "inverse": approximate,
        "exact_inverse": _inverse_spd(base + u.T @ u),
        "base_inverse": inverse_base,
        "naive_inverse": naive,
        "schur": schur,
        "schur_bound": bound,
        "degrees": degrees,
    }
