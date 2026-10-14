# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check bounded scalar topology against the existing FP32 assembly contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton._src.solvers.monolithic.bsr_reduction import _FixedScalarBsrReduction
from newton._src.solvers.monolithic.linear_types import MonolithicLinearStatus
from newton.tests.test_monolithic_linear_assembly import _dense_bsr
from newton.tests.test_monolithic_linear_reduction import _reference
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _inputs(capacity, device):
    return SimpleNamespace(
        rows=wp.zeros(capacity, dtype=int, device=device),
        columns=wp.zeros(capacity, dtype=int, device=device),
        values=wp.zeros(capacity, dtype=float, device=device),
        count=wp.zeros(1, dtype=int, device=device),
        status=wp.zeros(1, dtype=int, device=device),
    )


def _fill(inputs, rows, columns, values):
    inputs.status.zero_()
    inputs.count.fill_(len(values))
    inputs.rows.fill_(-123)
    inputs.columns.fill_(-456)
    inputs.values.fill_(float("nan"))
    if len(values):
        inputs.rows[: len(values)].assign(np.asarray(rows, np.int32))
        inputs.columns[: len(values)].assign(np.asarray(columns, np.int32))
        inputs.values[: len(values)].assign(np.asarray(values, np.float32))


def _assert_matrix(test, owner, rows, columns, values):
    size = owner.size
    rows, columns, values = np.asarray(rows, np.int32), np.asarray(columns, np.int32), np.asarray(values, np.float32)
    expected = _reference(rows, columns, values, size) if len(values) else np.zeros((size, size), np.float32)
    positions = sorted({(int(r), int(c)) for r, c, v in zip(rows, columns, values, strict=True) if v != 0})
    for row, column in set(zip(rows, columns, strict=True)) - set(positions):
        expected[row, column] = 0  # All-zero coordinates have no stored sign bit.
    np.testing.assert_array_equal(_dense_bsr(owner.matrix).astype(np.float32).view(np.uint32), expected.view(np.uint32))
    offsets = np.zeros(size + 1, np.int32)
    for row, _ in positions:
        offsets[row + 1] += 1
    np.cumsum(offsets, out=offsets)
    np.testing.assert_array_equal(owner.matrix.offsets.numpy(), offsets)
    np.testing.assert_array_equal(owner.matrix.columns.numpy()[: len(positions)], [c for _, c in positions])
    test.assertEqual(owner.matrix.nnz_sync(), len(positions))
    test.assertEqual(owner.status.numpy()[0], MonolithicLinearStatus.SUCCESS)
    vector = np.arange(1, size + 1, dtype=np.float32)
    x, y, diag = (
        wp.array(vector, device=owner.device),
        wp.zeros(size, device=owner.device),
        wp.zeros(size, device=owner.device),
    )
    sparse.bsr_mv(owner.matrix, x, y, tile_size=-1)
    np.testing.assert_allclose(y.numpy(), expected @ vector, rtol=3e-6, atol=1e-5)
    sparse.bsr_get_diag(owner.matrix, out=diag)
    np.testing.assert_array_equal(diag.numpy(), expected.diagonal())


def test_fixed_topology_reuse(test, device):
    """Keep zeros in the value tree despite stale scratch and changing active topology."""
    capacity = 65
    inputs = _inputs(capacity, device)
    owner = _FixedScalarBsrReduction(5, capacity, device, use_cuda_graph=device.is_cuda)
    rows = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4], np.int32)
    columns = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1], np.int32)
    values = np.array([0, -0.0, 1, -1, 0, 2**24, 1, 1, -0.0, -3, 1e8, -1e8], np.float32)
    for n in (0, 12, 5, 0, 12):
        _fill(inputs, rows[:n], columns[:n], values[:n])
        snapshot = None
        for seed in range(3):
            for array in owner.buffers():
                array.fill_(seed)
            owner.build(inputs, layout_identity="fixed")
            _assert_matrix(test, owner, rows[:n], columns[:n], values[:n])
            for name in ("starts", "work"):
                np.testing.assert_array_equal(
                    getattr(owner, name).numpy()[n:], seed, err_msg="Build touched unused scratch capacity"
                )
            current = tuple(
                array.numpy().tobytes() for array in (owner.matrix.offsets, owner.matrix.columns, owner.matrix.values)
            )
            if snapshot is not None:
                test.assertEqual(current, snapshot)
            snapshot = current
    if device.is_cuda:
        test.assertEqual(owner.graphs.captures["fixed_scalar_bsr"], 1)
    _fill(inputs, rows, columns, values)
    with patch.object(wp, "empty", side_effect=AssertionError("Allocated during build")):
        with patch.object(sparse, "bsr_set_from_triplets", side_effect=AssertionError("Native topology builder")):
            owner.build(inputs, layout_identity="fixed")
    _assert_matrix(test, owner, rows, columns, values)
    actual = sum(a.capacity for a in owner.buffers() if a.device == device)
    test.assertEqual(owner.payload_bytes, actual)


def test_active_count_reuse(test, device):
    """Preserve the value tree across empty, full and changing bounded sort spans."""
    owner = _FixedScalarBsrReduction(3, 65, device, use_cuda_graph=device.is_cuda)
    inputs = _inputs(65, device)
    values = np.resize(np.array([2**24, 1, -(2**24), -0.0, 1, -1], np.float32), 65)
    for count in (65, 0, 1, 33, 7, 0, 65):
        rows = np.zeros(count, np.int32)
        _fill(inputs, rows, rows, values[:count])
        with patch.object(wp.array, "numpy", side_effect=AssertionError("Builder added a host read")):
            owner.build(inputs, active_count=count)
        _assert_matrix(test, owner, rows, rows, values[:count])
    for count in (-1, 66, True, 1.5):
        with test.assertRaises(ValueError):
            owner.build(inputs, active_count=count)


def test_fixed_topology_long_tree(test, device):
    """Preserve value-bit order and odd carries for long duplicate groups and permutations."""
    rng = np.random.default_rng(893)
    for count in (20736, 131072):
        values = np.r_[np.full(count, 1, np.float32), np.float32(2**24), np.float32(-(2**24)), np.float32(-0.0)]
        rows = np.zeros(len(values), np.int32)
        columns = rows.copy()
        owner = _FixedScalarBsrReduction(3, len(values), device, use_cuda_graph=device.is_cuda)
        inputs = _inputs(len(values), device)
        for _ in range(4):
            order = rng.permutation(len(values))
            _fill(inputs, rows[order], columns[order], values[order])
            owner.build(inputs)
            _assert_matrix(test, owner, rows, columns, values)
        _fill(inputs, [], [], [])
        owner.build(inputs)
        _assert_matrix(test, owner, [], [], [])


def test_fixed_topology_errors(test, device):
    """Reject invalid active inputs and duplicate-sum overflow without partial output."""
    owner = _FixedScalarBsrReduction(3, 8, device, use_cuda_graph=device.is_cuda)
    inputs = _inputs(8, device)
    cases = (
        ([3], [0], [0], MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ([0], [-1], [0], MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ([0], [0], [float("nan")], MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
        ([0], [0], [float("inf")], MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
        ([0, 0], [0, 0], [np.finfo(np.float32).max] * 2, MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
    )
    for rows, columns, values, status in cases:
        _fill(inputs, rows, columns, values)
        owner.build(inputs)
        test.assertEqual(inputs.status.numpy()[0], status)
        test.assertEqual(owner.status.numpy()[0], status)
        for array in (owner.matrix.offsets, owner.matrix.columns, owner.matrix.values):
            np.testing.assert_array_equal(array.numpy(), np.zeros(array.shape))
        _fill(inputs, [2], [2], [3])
        owner.build(inputs)
        _assert_matrix(test, owner, [2], [2], [3])
    for count in (-1, 9, 16, 17):
        inputs.status.zero_()
        inputs.count.fill_(count)
        owner.build(inputs)
        test.assertEqual(inputs.status.numpy()[0], MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW)
        test.assertEqual(owner.matrix.nnz_sync(), 0)
    _fill(inputs, [0], [0], [1])
    inputs.status.fill_(int(MonolithicLinearStatus.STALE_GENERATION))
    owner.build(inputs)
    test.assertEqual(owner.matrix.nnz_sync(), 0)
    test.assertEqual(owner.status.numpy()[0], MonolithicLinearStatus.STALE_GENERATION)


def test_fixed_topology_bindings(test, device):
    """Rebind inputs, all scratch/output storage and streams before graph replay."""
    owner = _FixedScalarBsrReduction(3, 8, device, use_cuda_graph=device.is_cuda)
    inputs = _inputs(8, device)
    _fill(inputs, [0, 0, 2], [1, 1, 2], [2, 3, 4])
    owner.build(inputs)
    captures = int(device.is_cuda)
    for name in ("rows", "columns", "values", "count", "status"):
        setattr(inputs, name, wp.clone(getattr(inputs, name)))
        owner.build(inputs)
        _assert_matrix(test, owner, [0, 0, 2], [1, 1, 2], [2, 3, 4])
        captures += int(device.is_cuda)
        test.assertEqual(owner.graphs.captures["fixed_scalar_bsr"], captures)
    for name in owner._spec:
        setattr(owner, name, wp.clone(getattr(owner, name)))
        owner.build(inputs)
        _assert_matrix(test, owner, [0, 0, 2], [1, 1, 2], [2, 3, 4])
        captures += int(device.is_cuda)
        test.assertEqual(owner.graphs.captures["fixed_scalar_bsr"], captures)
    for name in ("offsets", "columns", "values"):
        setattr(owner.matrix, name, wp.clone(getattr(owner.matrix, name)))
        owner.build(inputs)
        _assert_matrix(test, owner, [0, 0, 2], [1, 1, 2], [2, 3, 4])
        captures += int(device.is_cuda)
        test.assertEqual(owner.graphs.captures["fixed_scalar_bsr"], captures)
    if device.is_cuda:
        wp.synchronize_device(device)
        with wp.ScopedStream(wp.Stream(device)):
            owner.build(inputs)
            wp.synchronize_stream()
            _assert_matrix(test, owner, [0, 0, 2], [1, 1, 2], [2, 3, 4])
        test.assertEqual(owner.graphs.captures["fixed_scalar_bsr"], captures + 1)
        test.assertEqual(len(owner.graphs.entries["fixed_scalar_bsr"]), 6)


def test_fixed_topology_empty(test, device):
    """Support a zero model, zero capacity and a dense-bound-smaller-than-triplet output."""
    for size, capacity in ((0, 0), (3, 0), (0, 8), (1, 8)):
        owner = _FixedScalarBsrReduction(size, capacity, device, use_cuda_graph=device.is_cuda)
        inputs = _inputs(capacity, device)
        _fill(inputs, [], [], [])
        owner.build(inputs)
        _assert_matrix(test, owner, [], [], [])
        if size == 1:
            _fill(inputs, [0] * 8, [0] * 8, [1] * 8)
            owner.build(inputs)
            _assert_matrix(test, owner, [0] * 8, [0] * 8, [1] * 8)


def test_fixed_topology_storage_guards(test, device):
    """Reject malformed or overlapping arrays on the host before kernels touch them."""
    owner = _FixedScalarBsrReduction(3, 8, device, use_cuda_graph=device.is_cuda)
    inputs = _inputs(8, device)
    for object_, name, wrong in (
        (inputs, "rows", wp.zeros(7, dtype=int, device=device)),
        (inputs, "values", wp.zeros(8, dtype=wp.float64, device=device)),
        (owner, "prefix", wp.zeros(1, dtype=int, device=device)),
        (owner, "work", wp.zeros(8, dtype=wp.float64, device=device)),
        (owner, "work", owner.starts.view(wp.float32)),
        (owner.matrix, "columns", inputs.rows[: owner.output_capacity]),
    ):
        original = getattr(object_, name)
        setattr(object_, name, wrong)
        with patch.object(wp, "launch", side_effect=AssertionError("Launched before validation")):
            with test.assertRaises(ValueError):
                owner.build(inputs)
        setattr(object_, name, original)


def test_fixed_topology_random_groups(test, device):
    """Preserve cancellation and zero placement in heterogeneous duplicate groups."""
    capacity = 513
    owner = _FixedScalarBsrReduction(11, capacity, device, use_cuda_graph=False)
    inputs = _inputs(capacity, device)
    rng = np.random.default_rng(779)
    # Force worker reuse across both sort passes, the scan and the float tree.
    with patch.object(_FixedScalarBsrReduction, "MAX_WORKERS", 7):
        for count in (3, capacity, 1, 2, 7, 8, 9, 255, 256, 257, 511, 512, 0, capacity):
            rows, columns = rng.integers(0, 11, size=(2, count), dtype=np.int32)
            values = rng.choice(np.array([0, -0.0, 1, -1, 2**24, -(2**24), 1e8, -1e8, 0.25], np.float32), count)
            if count == 3:
                rows[:] = 0
                columns[:] = 0
                values[:] = -0.0
            _fill(inputs, rows, columns, values)
            owner.build(inputs)
            _assert_matrix(test, owner, rows, columns, values)


def test_fixed_topology_key_width(test, device):
    """Keep distinct coordinates above uint32 without allocating a dense oracle."""
    size = 100001
    owner = _FixedScalarBsrReduction(size, 3, device, use_cuda_graph=device.is_cuda)
    inputs = _inputs(3, device)
    _fill(inputs, [99999, 100000, 99999], [99999, 100000, 99999], [1, 3, 2])
    owner.build(inputs)
    test.assertEqual(owner.matrix.nnz_sync(), 2)
    np.testing.assert_array_equal(owner.matrix.offsets.numpy()[-3:], [0, 1, 2])
    np.testing.assert_array_equal(owner.matrix.columns.numpy()[:2], [99999, 100000])
    np.testing.assert_array_equal(owner.matrix.values.numpy()[:2], [3, 3])
    test.assertEqual(owner.status.numpy()[0], 0)


class TestMonolithicLinearFixedStorage(unittest.TestCase):
    def test_capacity_preflight(self):
        """Check all allocation dimensions before allocating GPU or host transfer storage."""
        with patch.object(wp, "empty", side_effect=AssertionError("Allocated before validation")):
            with patch.object(wp, "zeros", side_effect=AssertionError("Allocated before validation")):
                for size, capacity in ((True, 3), (3, True), (-1, 0), (0, -1), (2**31 - 1, 0), (0, 2**30), (1.5, 1)):
                    with self.assertRaises(ValueError):
                        _FixedScalarBsrReduction(size, capacity, "cpu", use_cuda_graph=False)


for device in get_test_devices():
    for function in (
        test_active_count_reuse,
        test_fixed_topology_reuse,
        test_fixed_topology_long_tree,
        test_fixed_topology_errors,
        test_fixed_topology_bindings,
        test_fixed_topology_empty,
        test_fixed_topology_storage_guards,
        test_fixed_topology_random_groups,
        test_fixed_topology_key_width,
    ):
        add_function_test(TestMonolithicLinearFixedStorage, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
