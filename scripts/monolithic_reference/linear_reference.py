# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline diagonal inverse for independent linear-system checks."""

import warp as wp
from warp.optim.linear import LinearOperator


@wp.kernel
def _build_diagonal_inverse(
    diagonal: wp.array[float],
    scale: wp.array[float],
    regularization: float,
    inverse: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    value = scale[i] * diagonal[i] * scale[i] + regularization
    if value <= 0.0 or not wp.isfinite(value):
        wp.atomic_max(status, 0, 1)
        return
    inverse[i] = 1.0 / value
    if not wp.isfinite(inverse[i]):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _apply_diagonal_inverse(
    inverse: wp.array[float],
    x: wp.array[float],
    y: wp.array[float],
    z: wp.array[float],
    alpha: float,
    beta: float,
):
    i = wp.tid()
    value = alpha * inverse[i] * x[i]
    if beta != 0.0:
        value += beta * y[i]
    z[i] = value


class _DiagonalInverse:
    """Offline inverse of the same scaled production BSR diagonal."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.values = wp.empty(workspace.layout.scalar_dof_count, dtype=float, device=workspace.device)
        self.status = wp.zeros(1, dtype=int, device=workspace.device)
        self.operator = LinearOperator(workspace.operator.shape, wp.float32, workspace.device, self.matvec)

    def setup(self):
        self.status.zero_()
        w = self.workspace
        wp.launch(
            _build_diagonal_inverse,
            self.values.size,
            [w._matrix_diagonal, w.scale, w._lambda, self.values, self.status],
            device=w.device,
        )
        if self.status.numpy()[0]:
            raise ValueError("Offline diagonal inverse is not finite and positive")

    def matvec(self, x, y, z, alpha, beta):
        wp.launch(_apply_diagonal_inverse, x.size, [self.values, x, y, z, alpha, beta], device=self.workspace.device)
