# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare nonlinear device reductions with independent host arithmetic."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import iteration_device as kernels
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_reductions(test, device):
    """Check block norms, RHS, slope and actual displacement in both precisions."""
    rng = np.random.default_rng(104)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        n = 1025
        r, reference, trial, effective = [rng.normal(size=n).astype(native) for _ in range(4)]
        scale = np.geomspace(0.01, 100, n).astype(np.float32)
        direction = rng.normal(size=n).astype(np.float32)
        arrays = [wp.array(v, dtype=dtype, device=device) for v in (r, reference, trial, effective)]
        ds, dd = [wp.array(v, dtype=float, device=device) for v in (scale, direction)]
        terms = wp.empty((9, n), dtype=dtype, device=device)
        sums = wp.empty((9, 1), dtype=dtype, device=device)
        rhs = wp.empty(n, dtype=float, device=device)
        tolerance = 32 * np.finfo(native).eps
        for nq in (0, 3, n):
            parts = (slice(0, n), slice(0, nq), slice(nq, n))
            wp.launch(kernels.measure_terms, n, [arrays[0], arrays[1], ds, nq, terms, rhs], device=device)
            wp.utils.array_sum(terms, out=sums, axis=1)
            expected = [
                [np.sum((scale[p].astype(native) * v[p]) ** 2) for v in (r, reference)] + [np.sum(r[p] ** 2)]
                for p in parts
            ]
            np.testing.assert_allclose(sums.numpy().reshape(3, 3), expected, rtol=tolerance, atol=0)
            np.testing.assert_array_equal(rhs.numpy(), (-scale.astype(native) * r).astype(np.float32))
            wp.launch(kernels.trial_terms, n, [arrays[0], arrays[2], ds, dd, arrays[3], nq, terms], device=device)
            wp.utils.array_sum(terms, out=sums, axis=1)
            d, s = direction.astype(native), scale.astype(native)
            expected = [
                r @ effective,
                np.sum((s * trial) ** 2),
                np.sum(s * s),
                np.sum(d * d),
                np.sum(abs(d)),
                np.sum(abs(r * d)),
                *[np.sum((effective[p] / s[p]) ** 2) for p in parts],
            ]
            np.testing.assert_allclose(sums.numpy().ravel(), expected, rtol=tolerance, atol=tolerance)
        arrays[0].fill_(native(float("nan")))
        wp.launch(kernels.measure_terms, n, [arrays[0], arrays[1], ds, 3, terms, rhs], device=device)
        wp.utils.array_sum(terms, out=sums, axis=1)
        test.assertTrue(np.isnan(sums.numpy()[0, 0]))


class TestReductions(unittest.TestCase):
    pass


add_function_test(TestReductions, "test_reductions", test_reductions, devices=get_test_devices())


def test_domain_boundary(test, device):
    """Preserve inclusive joint and rotation rejection boundaries on device."""
    for dtype, native, matrix in ((wp.float32, np.float32, wp.mat33), (wp.float64, np.float64, wp.mat33d)):
        q = wp.zeros(2, dtype=dtype, device=device)
        qn = wp.zeros_like(q)
        ids = wp.array([0, 1], dtype=int, device=device)
        types = wp.array([1, 0], dtype=int, device=device)
        rn = wp.array(np.array([np.eye(3), np.eye(3)], dtype=native), dtype=matrix, device=device)
        rotations = wp.clone(rn)
        flag = wp.zeros(1, dtype=int, device=device)
        for case in range(5):
            values = np.zeros(2, dtype=native)
            mats = np.array([np.eye(3), np.eye(3)], dtype=native)
            if case == 1:
                values[0] = native(np.pi / 2)
            if case == 2:
                values[1] = 100
            if case == 3:
                mats[0, 0, 0] = -1
            if case == 4:
                mats[0, 1, 1] = np.nan
            q.assign(values)
            rotations.assign(mats)
            flag.zero_()
            wp.launch(
                kernels.domain_check, 2, [q, qn, ids, types, rotations, rn, 1, dtype(np.pi / 2), flag], device=device
            )
            test.assertEqual(bool(flag.numpy()[0]), case in (1, 3, 4))


add_function_test(TestReductions, "test_domain_boundary", test_domain_boundary, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
