# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify count-bounded BSR construction without changing assembly semantics."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton._src.solvers.monolithic.linear_types import MonolithicLinearGeneration, MonolithicLinearStatus
from newton.tests.test_monolithic_linear_assembly import _contact, _dense_bsr, _fixed_pattern, _owners, _workspace
from newton.tests.test_monolithic_linear_reduction import _reference
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_prefix_views_and_parity(test, device):
    """Build only the active prefix with shared storage and unchanged matrix sums."""
    ws = _workspace(device, global_capacity=4096, internal_capacity=32)
    original = sparse.bsr_set_from_triplets
    for sequence in (1, 2):
        generation = MonolithicLinearGeneration(1, 0, 1, sequence)
        assembly = ws.begin_assembly(generation)
        wp.launch(
            _owners,
            1,
            [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets],
            device=device,
        )
        wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
        expected = [(ws.k_global_scalar_bsr, ws._global, 199), (ws.ax_internal_bsr3, ws._internal, 8)]
        oracles = []
        for matrix, buffer, count in expected:
            # Invalid stale capacity tail must be ignored, while active zero and
            # duplicate entries retain their original reduction semantics.
            buffer.rows[count:].fill_(-123)
            buffer.values[count:].fill_(float("nan") if buffer.values.dtype == wp.float32 else wp.mat33(float("nan")))
            oracle = sparse.bsr_zeros(matrix.nrow, matrix.ncol, matrix.values.dtype, device=device)
            original(oracle, buffer.rows, buffer.columns, buffer.values, count=buffer.count)
            oracles.append(
                _reference(
                    buffer.rows.numpy()[:count],
                    buffer.columns.numpy()[:count],
                    buffer.values.numpy()[:count],
                    matrix.nrow,
                ).astype(np.float64)
                if matrix is ws.k_global_scalar_bsr
                else _dense_bsr(oracle)
            )
        calls = []
        reads = []
        original_numpy = wp.array.numpy

        def read(array, _reads=reads, _numpy=original_numpy):
            _reads.append(array.shape)
            return _numpy(array)

        def build(matrix, rows, columns, values=None, _expected=expected, _calls=calls, **kwargs):
            _, buffer, count = _expected[len(_calls)]
            test.assertEqual(rows.size, count, "Builder still sorts unused capacity")
            if matrix is ws.k_global_scalar_bsr:
                test.assertIsNone(values, "Global builder must only construct topology")
                pairs = ((rows, ws._scalar_reduction.rows), (columns, buffer.columns))
            else:
                pairs = zip((rows, columns, values), (buffer.rows, buffer.columns, buffer.values), strict=True)
            for view, base in pairs:
                test.assertEqual(view.ptr, base.ptr)
                test.assertEqual(view.strides, base.strides)
                test.assertTrue(view.is_contiguous)
                test.assertIs(view._ref, base)
            test.assertIs(kwargs["count"], buffer.count)
            _calls.append(rows.size)
            return original(matrix, rows, columns, values, **kwargs)

        with patch.object(sparse, "bsr_set_from_triplets", new=build), patch.object(wp.array, "numpy", new=read):
            result = ws.finalize_assembly(generation=generation)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        test.assertEqual(calls, [199, 8])
        # Counts/statuses and final status/nnz cross each boundary together.
        test.assertEqual(reads, [(6,), (1,), (3,)])
        test.assertEqual(result.global_nnz, int(ws.k_global_scalar_bsr.offsets.numpy()[-1]))
        test.assertEqual(result.internal_nnz, int(ws.ax_internal_bsr3.offsets.numpy()[-1]))
        for (matrix, _, _), oracle in zip(expected, oracles, strict=True):
            actual = _dense_bsr(matrix)
            np.testing.assert_array_equal(actual, oracle)
            vector = np.linspace(-1, 1, actual.shape[0]).astype(np.float32)
            output = wp.empty_like(wp.array(vector, device=device))
            sparse.bsr_mv(matrix, wp.array(vector, device=device), output)
            np.testing.assert_allclose(output.numpy(), oracle @ vector, atol=2e-6)


def test_empty_and_full_prefix(test, device):
    """Clear previous matrices and handle both zero capacity and full occupancy."""
    for global_capacity, internal_capacity in ((0, 0), (199, 8), (256, 32)):
        ws = _workspace(device, global_capacity=global_capacity, internal_capacity=internal_capacity)
        for sequence, populated in enumerate((bool(global_capacity), False), 1):
            generation = MonolithicLinearGeneration(1, 0, 1, sequence)
            assembly = ws.begin_assembly(generation)
            if populated:
                wp.launch(
                    _owners,
                    1,
                    [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets],
                    device=device,
                )
                wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
            test.assertEqual(ws.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
            if not populated:
                for matrix in (ws.k_global_scalar_bsr, ws.ax_internal_bsr3):
                    np.testing.assert_array_equal(matrix.offsets.numpy(), 0)


def test_count_guards_before_builder(test, device):
    """Reject bad counts and stale generations before slicing or building."""
    cases = [
        ("global", -1, MonolithicLinearStatus.INVALID_ARGUMENT),
        ("internal", -1, MonolithicLinearStatus.INVALID_ARGUMENT),
        ("global", 257, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW),
        ("internal", 9, MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW),
    ]
    for name, count, status in cases:
        ws = _workspace(device)
        generation = MonolithicLinearGeneration(1, 0, 1, 1)
        ws.begin_assembly(generation)
        getattr(ws, "_" + name).count.fill_(count)
        with patch.object(
            sparse, "bsr_set_from_triplets", new=lambda *args, **kwargs: test.fail("Invalid assembly reached builder")
        ):
            test.assertEqual(ws.finalize_assembly(generation=generation).status, status)
            test.assertEqual(
                ws.finalize_assembly(generation=generation).status, MonolithicLinearStatus.STALE_GENERATION
            )


def test_count_bounded_initialization(test, device):
    """Clear reserved slots only and ignore poisoned tails across contact count changes."""
    for fixed in (False, True):
        ws = _workspace(device, global_capacity=256, internal_capacity=8)
        if fixed:
            ws._set_fixed_triplet_pattern(**_fixed_pattern())
        for sequence, contact in enumerate((True, False, True), 1):
            for buffer in (ws._global, ws._internal):
                value = float("nan") if buffer.values.dtype == wp.float32 else wp.mat33(float("nan"))
                buffer.values.fill_(value)
            generation = MonolithicLinearGeneration(1, 0, 1, sequence)
            with patch.object(
                wp.array, "numpy", side_effect=AssertionError("Initialization must not read back counts")
            ):
                assembly = ws.begin_assembly(generation)
            for buffer, count in ((ws._global, 10 if fixed else 0), (ws._internal, 1 if fixed else 0)):
                values = buffer.values.numpy()
                np.testing.assert_array_equal(values[:count], 0)
                test.assertTrue(np.isnan(values[count:]).all(), "Initialization cleared unused capacity")
            if contact:
                wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
            result = ws.finalize_assembly(generation=generation)
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
            test.assertEqual(result.global_triplet_count, (10 if fixed else 0) + (121 if contact else 0))
            g = np.r_[2.0, -1.0, np.arange(1, 10) * 0.1]
            expected = 3 * np.outer(g, g) if contact else np.zeros((11, 11))
            np.testing.assert_allclose(_dense_bsr(ws.k_global_scalar_bsr), expected, atol=8e-7)
            np.testing.assert_array_equal(_dense_bsr(ws.ax_internal_bsr3), 0)


def test_fixed_active_sort_span(test, device):
    """Reuse fixed storage while bounding sort work and ignoring stale capacity tails."""
    from newton._src.solvers.monolithic import bsr_reduction  # noqa: PLC0415

    ws = _workspace(device, global_capacity=4096, internal_capacity=1)
    ws._set_fixed_triplet_pattern(**_fixed_pattern())
    ws._enable_fixed_bsr_storage()
    ws.use_cuda_graph = device.is_cuda
    ws._fixed_bsr[0].graphs.enabled = device.is_cuda
    original = bsr_reduction.radix_sort_pairs
    for sequence, contact in enumerate((True, False, True), 1):
        generation = MonolithicLinearGeneration(1, 0, 1, sequence)
        assembly = ws.begin_assembly(generation)
        if contact:
            wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
        count = 10 + (121 if contact else 0)
        ws._global.rows[count:].fill_(-123)
        ws._global.values[count:].fill_(float("nan"))
        calls = []

        def sort(keys, values, length, calls=calls, count=count, **kwargs):
            calls.append(length)
            test.assertGreaterEqual(length, count)
            test.assertLess(length, 2 * count)
            return original(keys, values, length, **kwargs)

        with patch.object(bsr_reduction, "radix_sort_pairs", sort):
            result = ws.finalize_assembly(generation=generation)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        if sequence < 3 or not device.is_cuda:
            test.assertTrue(calls)
        g = np.r_[2.0, -1.0, np.arange(1, 10) * 0.1]
        expected = 3 * np.outer(g, g) if contact else np.zeros((11, 11))
        np.testing.assert_allclose(_dense_bsr(ws.k_global_scalar_bsr), expected, atol=8e-7)


class TestMonolithicBsrPrefix(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_fixed_active_sort_span,
        test_prefix_views_and_parity,
        test_empty_and_full_prefix,
        test_count_guards_before_builder,
        test_count_bounded_initialization,
    ):
        add_function_test(TestMonolithicBsrPrefix, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
