# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-order scalar assembly independently of producer insertion order."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.bsr_reduction import _position_keys
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
)
from newton._src.solvers.monolithic.segment_graph import _SegmentGraphCache
from newton.tests.test_monolithic_linear_assembly import _dense_bsr, _workspace
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _tree(values):
    """Evaluate the pair tree in the values' precision by recursively splitting at powers of two."""
    if len(values) == 1:
        return values[0]
    split = 1 << ((len(values) - 1).bit_length() - 1)
    return values.dtype.type(_tree(values[:split]) + _tree(values[split:]))


def _reference(rows, columns, values, size):
    """Group on the host and prescribe unsigned value-bit order within each entry."""
    values = np.asarray(values)
    bits = np.uint64 if values.dtype == np.float64 else np.uint32
    result = np.zeros((size, size), dtype=values.dtype)
    for row, column in set(zip(rows, columns, strict=True)):
        terms = values[(rows == row) & (columns == column)]
        terms = terms[np.argsort(terms.view(bits), kind="stable")]
        result[row, column] = _tree(terms)
    return result


def _assemble(workspace, sequence, rows, columns, values):
    """Fill an active prefix while leaving any capacity tail untouched."""
    generation = MonolithicLinearGeneration(1, 0, 1, sequence)
    assembly = workspace.begin_assembly(generation)
    buffer = assembly.global_scalar_triplets
    if len(values):
        buffer.rows[: len(values)].assign(np.asarray(rows, dtype=np.int32))
        buffer.columns[: len(values)].assign(np.asarray(columns, dtype=np.int32))
        buffer.values[: len(values)].assign(np.asarray(values, dtype=np.float32))
    buffer.count.fill_(len(values))
    return generation, workspace.finalize_assembly(generation=generation)


def test_fixed_order_tree(test, device):
    """Preserve small terms and reproduce the pair tree across input permutations."""
    values = np.array([2**24, 1, 1, *([1] * 1024), 2**24, 1e8, -1e8, 1, 3, -2], np.float32)
    rows = np.r_[np.zeros(3, np.int32), np.ones(1025, np.int32), np.full(5, 2, np.int32)]
    columns = rows.copy()
    expected = _reference(rows, columns, values, 11)
    test.assertEqual(expected[0, 0], np.float32(2**24 + 2))
    test.assertEqual(expected[1, 1], np.float32(2**24 + 1024))
    workspace = _workspace(device, global_capacity=len(values) + 8)
    rng = np.random.default_rng(405)
    for sequence in range(1, 9):
        order = np.arange(len(values)) if sequence == 1 else rng.permutation(len(values))
        _, result = _assemble(workspace, sequence, rows[order], columns[order], values[order])
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        np.testing.assert_array_equal(_dense_bsr(workspace.k_global_scalar_bsr), expected)


def test_coordinate_key_width(test, device):
    """Keep distinct coordinates when flattened positions exceed 32 bits."""
    size = 2**31 - 1
    rows = np.array([0, 100000, size - 1], dtype=np.int32)
    columns = np.array([size - 1, 17, size - 1], dtype=np.int32)
    keys = wp.empty(3, dtype=wp.uint64, device=device)
    wp.launch(
        _position_keys,
        3,
        [
            wp.array(rows, device=device),
            wp.array(columns, device=device),
            wp.array([0, 1, 2], dtype=int, device=device),
            size,
            keys,
        ],
        device=device,
    )
    np.testing.assert_array_equal(keys.numpy(), rows.astype(np.uint64) * size + columns.astype(np.uint64))


def test_zero_topology_and_active_prefix(test, device):
    """Prune zero-only entries, retain cancellations and ignore stale unused capacity."""
    workspace = _workspace(device, global_capacity=32)
    rows = np.array([0, 0, 1, 1, 2, 2, 2], np.int32)
    values = np.array([0, -0.0, 1, -1, 2**24, 1, 1], np.float32)
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    buffer = assembly.global_scalar_triplets
    buffer.rows.fill_(-123)
    buffer.columns.fill_(-456)
    buffer.values.fill_(float("nan"))
    buffer.rows[: len(values)].assign(rows)
    buffer.columns[: len(values)].assign(rows)
    buffer.values[: len(values)].assign(values)
    buffer.count.fill_(len(values))
    reads = []
    original = wp.array.numpy

    def numpy(array):
        reads.append(array.shape)
        return original(array)

    with patch.object(wp.array, "numpy", new=numpy):
        result = workspace.finalize_assembly(generation=generation)
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(reads, [(6,), (1,), (3,)])  # Packed counts/status, reduction flag, final status/nnz.
    matrix = workspace.k_global_scalar_bsr
    np.testing.assert_array_equal(matrix.offsets.numpy()[:4], [0, 0, 1, 2])
    np.testing.assert_array_equal(_dense_bsr(matrix), _reference(rows, rows, values, 11))
    np.testing.assert_array_equal(buffer.values.numpy()[: len(values)].view(np.uint32), values.view(np.uint32))
    _, result = _assemble(workspace, 2, [], [], [])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    np.testing.assert_array_equal(matrix.offsets.numpy(), 0)


def test_reduction_failure_and_generation(test, device):
    """Reject finite-input sum overflow and stale generations before exposing an operator."""
    workspace = _workspace(device)
    values = np.full(2, np.finfo(np.float32).max, dtype=np.float32)
    generation, result = _assemble(workspace, 1, [0, 0], [0, 0], values)
    test.assertEqual(result.status, MonolithicLinearStatus.NONFINITE_CONTRIBUTION)
    with test.assertRaises(ValueError):
        workspace.densify_for_test(generation=generation)
    test.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.STALE_GENERATION)
    _, result = _assemble(workspace, 2, [1], [1], [3.0])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(_dense_bsr(workspace.k_global_scalar_bsr)[1, 1], 3.0)


def test_reduction_graph_storage(test, device):
    """Reuse graphs across values and rebuild bindings for count or storage changes."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    workspace = _workspace(device, global_capacity=64)
    cache = _SegmentGraphCache(wp.get_device(device), enabled=True)
    workspace._segment_graphs = cache
    for sequence, values in enumerate(([2**24, 1, 1], [1, 1, 2**24], [2, 3, 4]), 1):
        _, result = _assemble(workspace, sequence, [0] * 3, [0] * 3, values)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        test.assertEqual(_dense_bsr(workspace.k_global_scalar_bsr)[0, 0], 2**24 + 2 if sequence < 3 else 9)
    test.assertEqual(cache.captures["scalar_reduction"], 1)
    for field in ("rows", "columns", "values"):
        setattr(workspace._global, field, wp.clone(getattr(workspace._global, field)))
    _, result = _assemble(workspace, 4, [1] * 3, [1] * 3, [1, 2, 3])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(cache.captures["scalar_reduction"], 2)
    workspace.k_global_scalar_bsr.values = wp.clone(workspace.k_global_scalar_bsr.values)
    _, result = _assemble(workspace, 5, [1] * 3, [1] * 3, [1, 2, 3])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(cache.captures["scalar_reduction"], 3)
    _, result = _assemble(workspace, 6, [1], [1], [7])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(cache.captures["scalar_reduction"], 4)
    test.assertEqual(_dense_bsr(workspace.k_global_scalar_bsr)[1, 1], 7)


class TestMonolithicLinearReduction(unittest.TestCase):
    def test_sort_capacity_before_allocation(self):
        """Reject double-buffer sorting overflow before allocating device storage."""
        capacities = MonolithicLinearCapacities(2**30, 0, 0, 1, 1)
        mapping = wp.array([], dtype=int, device="cpu")
        with patch.object(wp, "zeros", side_effect=AssertionError("Allocated before capacity validation")):
            with self.assertRaisesRegex(ValueError, "sort.*int32"):
                MonolithicLinearWorkspace(MonolithicLinearLayout(0, 0), capacities, mapping, device="cpu")


for device in get_test_devices():
    for function in (
        test_fixed_order_tree,
        test_coordinate_key_width,
        test_zero_topology_and_active_prefix,
        test_reduction_failure_and_generation,
        test_reduction_graph_storage,
    ):
        add_function_test(TestMonolithicLinearReduction, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
