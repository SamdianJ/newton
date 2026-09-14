# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pinned, optional CPU SA-AMG oracle; never imported by the Newton runtime."""

import numpy as np


def elasticity_candidates(arrays, nq):
    """Restrict physical translations/rotations to dynamic nodes, scale and remove dependent columns."""
    xyz = np.asarray(arrays["rest_x"], dtype=np.float64)[arrays["dynamic_particle_ids"]]
    if not len(xyz):
        return np.zeros((0, 0))
    xyz = xyz - xyz.mean(axis=0)
    modes = np.zeros((len(xyz), 3, 6))
    for axis in range(3):
        modes[:, axis, axis] = 1
        modes[:, :, axis + 3] = np.cross(np.eye(3)[axis], xyz)
    modes = modes.reshape(-1, 6) / arrays["S"][nq:, None]
    u, singular, _ = np.linalg.svd(modes, full_matrices=False)
    rank = np.count_nonzero(singular > max(modes.shape) * np.finfo(float).eps * singular[0])
    return u[:, :rank]


class AmgOracle:
    """One fixed symmetric V-cycle, zero correction on every call, dense q Cholesky."""

    def __init__(self, reference, arrays):
        import pyamg  # noqa: PLC0415 - isolated optional dependency
        import scipy.linalg as la

        if pyamg.__version__ != "5.3.0":
            raise ValueError("Research requires pyamg==5.3.0 in its isolated environment")
        self.nq = reference.nq
        self.scale = reference.scale.copy()
        self.q_factor = la.cho_factor(reference.matrix[: self.nq, : self.nq].toarray()) if self.nq else None
        self.candidates = elasticity_candidates(arrays, self.nq)
        soft = reference.matrix[self.nq :, self.nq :].tobsr(blocksize=(3, 3))
        self.hierarchy = None
        if soft.shape[0]:
            self.hierarchy = pyamg.smoothed_aggregation_solver(
                soft,
                B=self.candidates,
                symmetry="symmetric",
                strength="symmetric",
                aggregate="standard",
                smooth=("jacobi", {"omega": 4.0 / 3.0}),
                improve_candidates=None,
                presmoother=None,
                postsmoother=None,
                coarse_solver="cholesky",
                max_levels=10,
                max_coarse=24,
            )
            # Force lazy Cholesky into build cost, not the first apply measurement.
            last = self.hierarchy.levels[-1].A
            self.hierarchy.coarse_solver(last, np.zeros(last.shape[0]))
            self._configure_smoothers()

    def _configure_smoothers(self):
        import scipy.linalg as la
        from pyamg.relaxation.smoothing import setup_block_gauss_seidel  # noqa: PLC0415 - isolated optional dependency
        from pyamg.util.utils import get_block_diag  # noqa: PLC0415 - isolated optional dependency

        for level in self.hierarchy.levels[:-1]:
            size = level.A.blocksize[0]
            blocks = get_block_diag(level.A, size, inv_flag=False)
            inverse = np.array([la.cho_solve(la.cho_factor(block), np.eye(size)) for block in blocks])
            # Explicit Cholesky also rejects invalid pivots instead of using a pseudoinverse.
            smoother = setup_block_gauss_seidel(level, iterations=1, sweep="symmetric", Dinv=inverse, blocksize=size)
            level.presmoother = smoother
            level.postsmoother = smoother

    def update(self, reference):
        """Reuse physical interpolation; rebuild Galerkin matrices, smoother blocks and coarse factor."""
        import pyamg  # noqa: PLC0415 - isolated optional dependency
        import scipy.linalg as la
        import scipy.sparse as sp

        if reference.nq != self.nq or len(reference.scale) != len(self.scale):
            raise ValueError("AMG layout changed; rebuild hierarchy")
        self.q_factor = la.cho_factor(reference.matrix[: self.nq, : self.nq].toarray()) if self.nq else None
        if self.hierarchy is not None:
            levels = self.hierarchy.levels
            if len(levels) > 1:
                levels[0].P = (sp.diags(self.scale[self.nq :] / reference.scale[self.nq :]) @ levels[0].P).tobsr(
                    blocksize=levels[0].P.blocksize
                )
                levels[0].R = levels[0].P.T.tobsr()
            levels[0].A = reference.matrix[self.nq :, self.nq :].tobsr(blocksize=(3, 3))
            for i in range(len(levels) - 1):
                levels[i + 1].A = (levels[i].R @ levels[i].A @ levels[i].P).tobsr()
            # PyAMG caches inverse diagonal blocks on A and Cholesky on coarse_solver.
            # Fresh A objects plus a fresh solver prevent stale numeric factors.
            self._configure_smoothers()
            self.hierarchy.coarse_solver = pyamg.coarse_grid_solver("cholesky")
            last = levels[-1].A
            self.hierarchy.coarse_solver(last, np.zeros(last.shape[0]))
        self.scale = reference.scale.copy()

    def apply(self, residual):
        import scipy.linalg as la

        result = np.zeros_like(residual, dtype=np.float64)
        if self.nq:
            result[: self.nq] = la.cho_solve(self.q_factor, residual[: self.nq])
        if self.hierarchy is not None:
            result[self.nq :] = self.hierarchy.solve(
                residual[self.nq :], x0=np.zeros(len(residual) - self.nq), tol=0, maxiter=1, cycle="V", accel=None
            )
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite AMG apply")
        return result
