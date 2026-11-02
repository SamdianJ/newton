# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare nonlinear device reductions with independent host arithmetic."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import iteration_device as kernels
from newton._src.solvers.monolithic.reductions import SegmentedReduction
from newton.tests.test_monolithic_material_state import scene as material_scene
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_reductions(test, device):
    """Check block norms, RHS, slope and actual displacement in both precisions."""
    rng = np.random.default_rng(104)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        n = 1025
        r, reference, trial, effective = [rng.normal(size=n).astype(native) for _ in range(4)]
        scale = np.geomspace(0.01, 100, n).astype(native)
        direction = rng.normal(size=n).astype(native)
        arrays = [wp.array(v, dtype=dtype, device=device) for v in (r, reference, trial, effective)]
        ds, dd = [wp.array(v, dtype=dtype, device=device) for v in (scale, direction)]
        terms = wp.empty((3, n), dtype=dtype, device=device)
        rhs = wp.empty(n, dtype=dtype, device=device)
        tau = np.geomspace(1e-4, 1e2, n).astype(native)
        dtau = wp.array(tau, dtype=dtype, device=device)
        # Rigid tangent rows use the physical displacement without scaling.
        unscaled = (np.arange(n) % 7 == 0).astype(np.int32)
        mask = wp.array(unscaled, dtype=int, device=device)
        tolerance = 64 * np.finfo(native).eps
        for groups in (
            [np.arange(n), np.arange(3), np.arange(3, n)],
            [np.arange(n), np.arange(0, n, 2), np.arange(1, n, 2), np.empty(0, dtype=int)],
            [np.arange(n), np.arange(n), np.empty(0, dtype=int)],
        ):
            plan = SegmentedReduction(n, groups, 3, dtype=dtype, device=device)
            wp.launch(kernels.measure_rows, n, [arrays[0], arrays[1], ds, terms, rhs], device=device)
            plan.reduce(terms)
            expected = [[np.sum((scale[p] * v[p]) ** 2) for v in (r, reference)] + [np.sum(r[p] ** 2)] for p in groups]
            np.testing.assert_allclose(plan.result.numpy(), expected, rtol=tolerance, atol=0)
            np.testing.assert_array_equal(rhs.numpy(), -scale * r)
            wp.launch(kernels.assessment_rows, n, [arrays[0], arrays[1], dtau, ds, terms], device=device)
            plan.reduce(terms, minimum=2)
            expected = [
                [
                    np.sum((r[p] / tau[p]) ** 2),
                    np.sum((reference[p] / tau[p]) ** 2),
                    np.min(scale[p] * tau[p], initial=np.inf),
                ]
                for p in groups
            ]
            np.testing.assert_allclose(plan.result.numpy(), expected, rtol=tolerance, atol=0)
            wp.launch(kernels.trial_rows, n, [arrays[0], ds, dd, arrays[3], mask, terms], device=device)
            plan.reduce(terms)
            moved = np.where(unscaled, effective, effective / scale)
            expected = [[r[p] @ effective[p], np.sum(abs(r[p] * direction[p])), np.sum(moved[p] ** 2)] for p in groups]
            actual = plan.result.numpy()
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            plan.reduce(terms)
            np.testing.assert_array_equal(plan.result.numpy(), actual)
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    plan.reduce(terms)
                terms.fill_(native(2))
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(plan.result.numpy(), [[2 * len(p)] * 3 for p in groups])
        arrays[0].fill_(native(float("nan")))
        wp.launch(kernels.measure_rows, n, [arrays[0], arrays[1], ds, terms, rhs], device=device)
        plan.reduce(terms)
        test.assertTrue(np.isnan(plan.result.numpy()[0, 0]))
        dtau.fill_(native(float("nan")))
        wp.launch(kernels.assessment_rows, n, [arrays[0], arrays[1], dtau, ds, terms], device=device)
        plan.reduce(terms, minimum=2)
        test.assertTrue(np.isnan(plan.result.numpy()[0, 2]))
        test.assertEqual(plan.result.numpy()[-1, 2], np.inf)


def test_segment_storage(test, device):
    """Keep row and partial storage linear while allowing many tiny or empty groups."""
    n = 1025
    groups = [np.arange(n), *(np.array([i]) for i in range(n)), np.empty(0, dtype=int)]
    plan = SegmentedReduction(n, groups, 3, dtype=wp.float64, device=device)
    test.assertEqual(plan.rows.size, 2 * n)
    test.assertLessEqual(plan.partials.size, 3 * (2 * n // plan.CHUNK_SIZE + len(groups)))
    terms = wp.ones((3, n), dtype=wp.float64, device=device)
    result = plan.reduce(terms, minimum=2).numpy()
    np.testing.assert_array_equal(result[0], [n, n, 1])
    np.testing.assert_array_equal(result[1:-1], np.ones((n, 3)))
    np.testing.assert_array_equal(result[-1], [0, 0, np.inf])
    empty = SegmentedReduction(0, [np.empty(0, dtype=int)], 3, dtype=wp.float64, device=device)
    np.testing.assert_array_equal(
        empty.reduce(wp.zeros((3, 0), dtype=wp.float64, device=device), minimum=2).numpy(), [[0, 0, np.inf]]
    )
    for group in ([n], [-1], [0.5], [[0]]):
        with test.assertRaises(ValueError):
            SegmentedReduction(n, [group], 3, dtype=wp.float64, device=device)


class TestReductions(unittest.TestCase):
    pass


def test_layout_group_norms(test, device):
    """Measure every mixed actor group without assuming a q/x row partition."""
    for precision, dtype, native in (("fp32", wp.float32, np.float32), ("fp64", wp.float64, np.float64)):
        _, solver = material_scene(mechanical=True, rigid=True, tet=True, precision=precision, device=device)
        layout = solver._layout
        values = np.arange(1, layout.scalar_dof_count + 1, dtype=native)
        vector = wp.array(values, dtype=dtype, device=device)
        expected = [np.linalg.norm(values[start : start + count]) for _, start, count in layout.groups]
        actual = kernels.norms(solver, vector)
        test.assertEqual(len(actual), len(expected))
        np.testing.assert_allclose(actual, expected, rtol=32 * np.finfo(native).eps)


add_function_test(TestReductions, "test_layout_group_norms", test_layout_group_norms, devices=get_test_devices())


def test_host_energy_packet(test, device):
    """Keep term order, native bits and error metadata through the fused readback."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for lengths in ((0, 0, 0, 0, 0), (1, 2, 3, 4, 2), (3, 0, 5, 3, 6)):
            host = [np.arange(n, dtype=native) - 2 for n in lengths]
            terms = [wp.array(a, dtype=dtype, device=device) for a in host]
            absolute = wp.array([11, 13], dtype=dtype, device=device)
            tet_flags = wp.array([5], dtype=int, device=device)
            contact_flags = wp.array([9, 7], dtype=int, device=device)
            det = wp.array([-0.5], dtype=dtype, device=device)
            packet = wp.empty(5 + sum(lengths), dtype=dtype, device=device)
            for pooled in (False, True):
                for value in (-0.0, np.inf, np.nan):
                    if lengths[-1]:
                        host[-1][0] = value
                        terms[-1].assign(host[-1])
                    packet.fill_(99)
                    wp.launch(
                        kernels.host_evaluation_packet,
                        packet.size,
                        [*terms, absolute, tet_flags, contact_flags, det, pooled, packet],
                        device=device,
                    )
                    result = packet.numpy()
                    np.testing.assert_array_equal(result[:5], [5, 9, -0.5, 11 if pooled else 0, 13 if pooled else 0])
                    test.assertEqual(result[5:].tobytes(), np.concatenate(host).tobytes())


add_function_test(TestReductions, "test_host_energy_packet", test_host_energy_packet, devices=get_test_devices())
add_function_test(TestReductions, "test_reductions", test_reductions, devices=get_test_devices())
add_function_test(TestReductions, "test_segment_storage", test_segment_storage, devices=get_test_devices())


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
