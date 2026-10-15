# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise reference q/x/theta matrix assembly without a production solver or GPU."""

import unittest

import numpy as np

from scripts.monolithic_reference.cable_oracle import CableFrame, CableReference, cable_derivatives
from scripts.monolithic_reference.cloth_oracle import shell_bending_tangent, shell_membrane_tangent


class TestMonolithicMaterialCoupling(unittest.TestCase):
    def test_joint_cloth_cable_assembly(self):
        """Assemble complete mixed blocks, eliminate a physical theta boundary and solve each residual group."""
        # Scalar q + six cloth xyz nodes + three cable xyz nodes + two edge theta.
        cloth = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0]])
        cable = np.array([[0.0, 0, 0], [1, 0, 0], [1.8, 0.6, 0]])
        theta = np.array([0.0, 0.1])
        axes = np.tile([0.0, 0, 1.0], (2, 1))
        rest = CableReference(cable, theta, axes, (100.0, 2.0, 3.0))
        _, cable_b = cable_derivatives(rest, cable, theta, CableFrame(cable, axes))
        cloth_b = shell_bending_tangent(cloth, cloth, (30.0, 40.0, 0.02, 0.07))
        cloth_b[:9, :9] += shell_membrane_tangent(cloth, cloth, (30.0, 40.0, 0.02, 0.07))
        matrix = np.zeros((30, 30))
        matrix[0, 0] = 10
        matrix[1:19, 1:19] = cloth_b + 4 * np.eye(18)
        matrix[19:, 19:] = cable_b
        matrix[19:28, 19:28] += 5 * np.eye(9)
        # Two conservative q-to-material springs exercise both force endpoints.
        for node_dof in (3, 27):
            direction = np.zeros(30)
            direction[[0, node_dof]] = [1, -1]
            matrix += 7 * np.outer(direction, direction)
        active = np.delete(np.arange(30), 28)  # physical theta Dirichlet value
        rhs = np.zeros(30)
        rhs[[0, 3, 27, 29]] = [0.2, -0.1, 0.3, 0.05]
        reduced = matrix[np.ix_(active, active)]
        self.assertGreater(np.linalg.eigvalsh(reduced).min(), 0)
        update = np.zeros(30)
        update[active] = np.linalg.solve(reduced, rhs[active])
        residual = matrix @ update - rhs
        for group in ([0], range(1, 19), range(19, 28), [29]):
            np.testing.assert_allclose(residual[list(group)], 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(matrix[19:28, 28:]), 0.01)
        self.assertNotEqual(matrix[0, 3], 0)
        self.assertNotEqual(matrix[0, 27], 0)
        # Reordering actor blocks must preserve the physical solution.
        order = np.r_[np.arange(19, 30), np.arange(1, 19), 0]
        reordered_active = np.flatnonzero(order != 28)
        permuted = matrix[np.ix_(order, order)]
        reordered = np.zeros(30)
        reordered[reordered_active] = np.linalg.solve(
            permuted[np.ix_(reordered_active, reordered_active)], rhs[order][reordered_active]
        )
        np.testing.assert_allclose(reordered, update[order], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
