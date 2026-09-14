# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Private PR8F fixed-hierarchy prototype, selected only by the research entry point."""

from functools import cache

import numpy as np
import warp as wp


@wp.kernel
def _project(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    scale: wp.array[float],
    blocks: wp.array2d[int],
    local: wp.array2d[int],
    nq: int,
    width: int,
    regularization: float,
    matrices: wp.array2d[wp.float64],
):
    i, level = wp.tid()
    row = nq + i
    block = blocks[level, i // 3]
    a = 3 * local[level, i // 3] + i % 3
    wi = wp.float64(1.0)
    if level > 0:
        wi = wp.float64(1.0) / wp.float64(scale[row])
    wp.atomic_add(matrices, block * width + a, a, wp.float64(regularization) * wi * wi)
    for entry in range(offsets[row], offsets[row + 1]):
        col = columns[entry]
        j = col - nq
        if j >= 0:
            if blocks[level, j // 3] == block:
                b = 3 * local[level, j // 3] + j % 3
                wj = wp.float64(1.0)
                if level > 0:
                    wj = wp.float64(1.0) / wp.float64(scale[col])
                wp.atomic_add(
                    matrices,
                    block * width + a,
                    b,
                    wp.float64(values[entry]) * wp.float64(scale[row]) * wp.float64(scale[col]) * wi * wj,
                )


@wp.kernel
def _pad(matrices: wp.array2d[wp.float64], sizes: wp.array[int], width: int):
    block, i = wp.tid()
    if i >= 3 * sizes[block]:
        matrices[block * width + i, i] = wp.float64(1.0)


@wp.kernel
def _check_factor(
    factors: wp.array2d[wp.float64], sizes: wp.array[int], width: int, tolerance: float, error: wp.array[int]
):
    block, i = wp.tid()
    value = factors[block * width + i, i]
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, 16)
    elif i < 3 * sizes[block] and (value <= wp.float64(0.0) or value * value <= wp.float64(tolerance)):
        wp.atomic_max(error, 0, 12)


@wp.kernel
def _clear_rhs(rhs: wp.array2d[float], active: wp.array[int]):
    i, j = wp.tid()
    if active[0] != 0:
        rhs[i, j] = 0.0


@wp.kernel
def _check_inverse(inverses: wp.array2d[float], error: wp.array[int]):
    i, j = wp.tid()
    if not wp.isfinite(inverses[i, j]):
        wp.atomic_max(error, 0, 16)


@wp.kernel
def _restrict(
    r: wp.array[float],
    scale: wp.array[float],
    blocks: wp.array2d[int],
    local: wp.array2d[int],
    nq: int,
    width: int,
    rhs: wp.array2d[float],
    active: wp.array[int],
):
    i, level = wp.tid()
    if active[0] != 0:
        value = r[nq + i]
        if level > 0:
            value /= scale[nq + i]
        row = width * blocks[level, i // 3] + 3 * local[level, i // 3] + i % 3
        wp.atomic_add(rhs, row, 0, value)


@wp.kernel
def _prolong(
    solution: wp.array2d[float],
    scale: wp.array[float],
    blocks: wp.array2d[int],
    local: wp.array2d[int],
    nq: int,
    width: int,
    out: wp.array[float],
    active: wp.array[int],
    error: wp.array[int],
):
    i = wp.tid()
    if active[0] != 0:
        value = float(0.0)
        for level in range(blocks.shape[0]):
            row = width * blocks[level, i // 3] + 3 * local[level, i // 3] + i % 3
            term = solution[row, 0]
            if level > 0:
                term /= scale[nq + i]
            value += term
        if not wp.isfinite(value):
            wp.atomic_max(error, 0, 16)
        out[nq + i] = value


@wp.func
def _to_float(value: wp.float64):
    return wp.float32(value)


@wp.func
def _to_double(value: wp.float32):
    return wp.float64(value)


@cache
def _kernels(width, explicit_inverse):
    apply_dtype = wp.float32 if explicit_inverse else wp.float64
    inverse_columns = 8 if width % 8 == 0 else 4

    @wp.kernel(enable_backward=False)
    def factor(matrix: wp.array2d[wp.float64], factors: wp.array2d[wp.float64]):
        block = wp.tid()
        tile = wp.tile_load(matrix, shape=(width, width), offset=(block * width, 0))
        wp.tile_cholesky_inplace(tile)
        wp.tile_store(factors, tile, offset=(block * width, 0))

    @wp.kernel(enable_backward=False)
    def inverse(factors: wp.array2d[wp.float64], identity: wp.array2d[wp.float64], inverses: wp.array2d[float]):
        block = wp.tid()
        tile = wp.tile_load(factors, shape=(width, width), offset=(block * width, 0))
        for column in range(width // inverse_columns):
            result = wp.tile_load(identity, shape=(width, inverse_columns), offset=(0, column * inverse_columns))
            wp.tile_cholesky_solve_inplace(tile, result)
            rounded = wp.tile_map(_to_float, result)
            wp.tile_store(inverses, rounded, offset=(block * width, column * inverse_columns))

    @wp.kernel(enable_backward=False)
    def apply(
        matrix: wp.array2d[apply_dtype], rhs: wp.array2d[float], solution: wp.array2d[float], active: wp.array[int]
    ):
        block = wp.tid()
        if active[0] != 0:
            tile = wp.tile_load(matrix, shape=(width, width), offset=(block * width, 0))
            b = wp.tile_load(rhs, shape=(width, 1), offset=(block * width, 0))
            if wp.static(explicit_inverse):
                result = wp.tile_matmul(tile, b)
                wp.tile_store(solution, result, offset=(block * width, 0))
            else:
                precise = wp.tile_map(_to_double, b)
                wp.tile_cholesky_solve_inplace(tile, precise)
                rounded = wp.tile_map(_to_float, precise)
                wp.tile_store(solution, rounded, offset=(block * width, 0))

    return factor, inverse, apply


class _BlockPreconditioner:
    """Sum of projected SPD block inverses with fixed maps and preallocated storage.

    Fine domains partition all soft degrees of freedom, so their inverse sum is
    positive definite. Coarse terms R.T B^-1 R are positive semidefinite. Each B
    projects the full scaled, regularized soft operator, including contact.
    No q rows or q-x coupling are changed. Numerical updates occur between PCGs.
    """

    def __init__(self, workspace, hierarchy):
        self.device = workspace.device
        self.nq = workspace.layout.q_dof_count
        self.nx = workspace.layout.dynamic_particle_count
        if hierarchy.block_ids.shape[1] != self.nx:
            raise ValueError("Hierarchy and workspace dimensions differ")
        self.width = 3 * hierarchy.block_size
        self.count = len(hierarchy.domains)
        self.levels = len(hierarchy.level_counts)
        self.fingerprint = hierarchy.fingerprint
        self.explicit_inverse = hierarchy.candidate.startswith("mas")
        self.blocks = wp.array(hierarchy.block_ids, dtype=int, device=self.device)
        self.local = wp.array(hierarchy.local_ids, dtype=int, device=self.device)
        self.sizes = wp.array([len(d) for d in hierarchy.domains], dtype=int, device=self.device)
        self.matrices = wp.zeros((self.count * self.width, self.width), dtype=wp.float64, device=self.device)
        self.factors = wp.zeros_like(self.matrices)
        self.inverses = (
            wp.zeros(self.matrices.shape, dtype=float, device=self.device) if self.explicit_inverse else self.factors
        )
        self.identity = wp.array(np.eye(self.width, dtype=np.float64), device=self.device)
        self.rhs = wp.zeros((self.count * self.width, 1), dtype=float, device=self.device)
        self.solution = wp.zeros_like(self.rhs)
        self.active = wp.ones(1, dtype=int, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)
        self.scale = workspace.scale
        self.kernels = _kernels(self.width, self.explicit_inverse)
        self._signature = self.signature()

    def signature(self):
        return (
            self.fingerprint,
            self.width,
            self.count,
            self.levels,
            self.explicit_inverse,
            tuple(
                (name, a.ptr, a.shape, a.dtype, a.device) for name, a in vars(self).items() if isinstance(a, wp.array)
            ),
        )

    def validate(self):
        if self.signature() != self._signature:
            raise ValueError("Research hierarchy storage changed; construct a new preconditioner")

    def factor(self, workspace, tolerance):
        self.validate()
        if self.scale is not workspace.scale:
            raise ValueError("Research scaling binding changed; construct a new preconditioner")
        if not self.count:
            return
        self.matrices.zero_()
        matrix = workspace.k_global_scalar_bsr
        wp.launch(
            _project,
            (3 * self.nx, self.levels),
            [
                matrix.offsets,
                matrix.columns,
                matrix.values,
                self.scale,
                self.blocks,
                self.local,
                self.nq,
                self.width,
                workspace._lambda,
                self.matrices,
            ],
            device=self.device,
        )
        wp.launch(_pad, (self.count, self.width), [self.matrices, self.sizes, self.width], device=self.device)
        wp.launch_tiled(
            self.kernels[0], dim=self.count, inputs=[self.matrices, self.factors], block_dim=128, device=self.device
        )
        wp.launch(
            _check_factor,
            (self.count, self.width),
            [self.factors, self.sizes, self.width, tolerance, workspace._status],
            device=self.device,
        )
        if self.explicit_inverse:
            wp.launch_tiled(
                self.kernels[1],
                dim=self.count,
                inputs=[self.factors, self.identity, self.inverses],
                block_dim=128,
                device=self.device,
            )
            wp.launch(_check_inverse, self.inverses.shape, [self.inverses, workspace._status], device=self.device)

    def apply(self, residual, output, *, active=None, error=None):
        self.validate()
        if not self.count:
            return
        mask = self.active if active is None else active
        status = self.error if error is None else error
        wp.launch(_clear_rhs, self.rhs.shape, [self.rhs, mask], device=self.device)
        wp.launch(
            _restrict,
            (3 * self.nx, self.levels),
            [residual, self.scale, self.blocks, self.local, self.nq, self.width, self.rhs, mask],
            device=self.device,
        )
        wp.launch_tiled(
            self.kernels[2],
            dim=self.count,
            inputs=[self.inverses, self.rhs, self.solution, mask],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            _prolong,
            3 * self.nx,
            [self.solution, self.scale, self.blocks, self.local, self.nq, self.width, output, mask, status],
            device=self.device,
        )
