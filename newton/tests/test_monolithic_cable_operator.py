# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the element-wise cable operator against its dense assembly in both precisions."""

import unittest

import numpy as np
import warp as wp

from scripts.monolithic_reference.cable_benchmarks import CableBenchmark


def loaded_case(precision, *, segments, angle, steps, device):
    """Return a rotated cantilever after ``steps`` loaded steps, with its operator and free rows."""
    case = CableBenchmark("cantilever", segments=segments, angle=angle, precision=precision, device=device)
    force = np.zeros((segments + 1, 3), dtype=np.float32)
    force[-1] = 3e-6 * case.rotation[:, case.axis]
    for _ in range(steps):
        case.state.particle_f.assign(force)
        case.solver.step(case.state, case.state, case.control, dt=case.dt)
    operator = case.solver._linear._material_operator
    return case, operator, np.asarray(operator.rows, dtype=int)


class TestMonolithicCableOperator(unittest.TestCase):
    device = "cpu"

    def test_relative_product_equals_dense_elements(self):
        """Apply S B S x from element-relative displacements exactly as the dense element sum, in FP64."""
        _case, operator, rows = loaded_case("fp64", segments=12, angle=30.0, steps=3, device=self.device)
        rng = np.random.default_rng(5)
        n = operator.size
        x, scale = rng.standard_normal(n), rng.uniform(0.5, 2.0, n)
        dense = operator.dense()
        expected = scale * (dense @ (scale * x))
        out = wp.zeros(n, dtype=wp.float64, device=self.device)
        magnitude = wp.zeros_like(out)
        operator.add_product(
            wp.array(x, dtype=wp.float64, device=self.device),
            wp.array(scale, dtype=wp.float64, device=self.device),
            out,
            magnitude=magnitude,
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())
        self.assertTrue(np.all(magnitude.numpy()[rows] >= np.abs(out.numpy()[rows]) * (1 - 1e-12)))
        outside = np.setdiff1d(np.arange(n), rows)
        self.assertTrue(np.all(out.numpy()[outside] == 0))
        # Each element tangent annihilates a rigid translation of its own vertices,
        # which is what lets it act on displacements relative to its first vertex.
        cable = operator.cables[0]
        shift = np.array([0.3, -0.2, 0.7])
        for matrix, width in ((cable.stretch_matrix.numpy(), 6), (cable.bend_matrix.numpy(), 11)):
            translation = np.zeros(width)
            translation[: 3 * (width // 3)] = np.tile(shift, width // 3)
            np.testing.assert_allclose(matrix @ translation, 0, atol=1e-12 * np.abs(matrix).max())

    def test_float32_soft_mode_is_resolved(self):
        """Keep the FP32 product on a rotated rod's soft bending mode accurate where the assembly is not."""
        case, operator, rows = loaded_case("fp32", segments=128, angle=45.0, steps=1, device=self.device)
        dense = operator.dense()
        mass = case.solver._cables[0].mass.numpy().astype(float) / case.dt**2
        mapping = case.solver._actors.maps["particle_to_global"]
        for particle in range(case.segments + 1):
            if mapping[particle] >= 0:
                dense[mapping[particle] + np.arange(3), mapping[particle] + np.arange(3)] += mass[particle]
        free = rows
        load = np.zeros(operator.size)
        load[mapping[case.segments] : mapping[case.segments] + 3] = 3e-6 * case.rotation[:, case.axis]
        u = np.zeros(operator.size)
        u[free] = np.linalg.solve(dense[np.ix_(free, free)], load[free])
        out = wp.zeros(operator.size, dtype=wp.float32, device=self.device)
        operator.add_product(wp.array(u, dtype=wp.float32, device=self.device), operator.ones, out)
        element = out.numpy().astype(float)
        for particle in range(case.segments + 1):
            if mapping[particle] >= 0:
                index = mapping[particle] + np.arange(3)
                element[index] += np.float32(mass[particle]) * u[index].astype(np.float32).astype(float)
        # Judge the operator, not the rounding of its FP32 input: compare with the
        # exact product of the represented vector.
        represented = u.astype(np.float32).astype(float)
        exact = dense @ represented
        curvature = float(represented @ exact)
        assembled = (dense.astype(np.float32) @ u.astype(np.float32)).astype(float)
        element_error = np.abs(element - exact)[free].max()
        assembled_error = np.abs(assembled - exact)[free].max()
        self.assertLess(element_error, 0.1 * 3e-6)
        self.assertLess(element_error, 0.1 * assembled_error)
        self.assertLess(abs(float(represented @ element) / curvature - 1), 0.05)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMonolithicCableOperatorCUDA(TestMonolithicCableOperator):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
