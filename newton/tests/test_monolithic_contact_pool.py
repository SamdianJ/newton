# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic native contact rows with bounded device storage."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact_pool import _ContactRows
from newton.tests.unittest_utils import add_function_test, get_test_devices


def row_inputs(rows, identities, local_ids, values, device, dtype):
    return (
        wp.array(rows, dtype=int, device=device),
        wp.array(identities, dtype=wp.uint64, device=device),
        wp.array(local_ids, dtype=int, device=device),
        wp.array(values, dtype=dtype, device=device),
        wp.array([len(rows)], dtype=int, device=device),
    )


def fixed_tree(values, dtype):
    count = len(values)
    work = np.zeros(1 << max(0, (count - 1).bit_length()), dtype=dtype)
    work[:count] = values
    stride = 1
    while stride < len(work):
        for left in range(0, len(work), 2 * stride):
            if left + stride < len(work):
                work[left] = dtype(work[left] + work[left + stride])
        stride *= 2
    return work[0]


def test_zero_padding(test, device):
    """Pad an odd native row with positive zero while preserving a single negative zero."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for count in (1, 2, 3, 5, 257):
            reducer = _ContactRows(1, count, count, 1, dtype, device)
            values = np.full(count, -0.0, dtype=native)
            reducer.reduce(*row_inputs(np.zeros(count), np.arange(count), np.zeros(count), values, device, dtype))
            expected = np.array([fixed_tree(values, native)], dtype=native)
            test.assertEqual(reducer.result.numpy().tobytes(), expected.tobytes())


def test_sorted_native_rows(test, device):
    """Reduce native rows independently of discovery order, including repeated DoFs."""
    rows = np.array([2, 0, 2, 2, 0, 4, 2], dtype=np.int32)
    identities = np.array([2, 0, 0, 1, 0, 3, 0], dtype=np.uint64)
    local = np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    rng = np.random.default_rng(531)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        big = native(2 ** (24 if dtype == wp.float32 else 53))
        values = np.array([3, 1, 1, -big, big, -7, big], dtype=native)
        reducer = _ContactRows(6, len(rows), 4, 2, dtype, device)
        expected = np.zeros(6, dtype=native)
        for row in range(6):
            indices = sorted(np.flatnonzero(rows == row), key=lambda i: (int(identities[i]), int(local[i])))
            expected[row] = fixed_tree(values[indices], native)
        for _ in range(8):
            order = rng.permutation(len(rows))
            inputs = row_inputs(rows[order], identities[order], local[order], values[order], device, dtype)
            reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            np.testing.assert_array_equal(reducer.offsets.numpy(), [0, 2, 2, 6, 6, 7, 7])
            np.testing.assert_array_equal(reducer.result.numpy(), expected)


def test_long_rows(test, device):
    """Handle odd tails and multiple fixed blocks with an independent error bound."""
    rng = np.random.default_rng(831)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for count in (1, 3, 255, 256, 257, 20736, 20739):
            # Powers of two make exact cancellation and a broad exponent range.
            values = np.ldexp(rng.choice([-1.0, 1.0], count), rng.integers(-20, 20, count)).astype(native)
            order = rng.permutation(count)
            reducer = _ContactRows(3, count, count, 1, dtype, device)
            inputs = row_inputs(
                np.ones(count, np.int32), order, np.zeros(count, np.int32), values[order], device, dtype
            )
            reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            result = reducer.result.numpy()
            np.testing.assert_array_equal(result, [0, fixed_tree(values, native), 0])
            np.testing.assert_array_equal(reducer.offsets.numpy(), [0, 0, count, count])
            levels = math.ceil(math.log2(count))
            u = np.finfo(native).eps / 2
            bound = levels * u / (1 - levels * u) * math.fsum(abs(float(v)) for v in values)
            test.assertLessEqual(abs(float(result[1]) - math.fsum(map(float, values))), bound)
            if device.is_cuda:
                reducer.graphs.enabled = True
                reducer.reduce(*inputs)
                test.assertEqual(int(reducer.flags.numpy()[0]), 0)
                test.assertEqual(reducer.result.numpy().tobytes(), result.tobytes())


def test_counts_and_tails(test, device):
    """Overwrite all derived storage while active counts shrink, empty and grow."""
    reducer = _ContactRows(3, 9, 9, 1, wp.float64, device)
    inputs = row_inputs(np.arange(9) % 3, np.arange(9), np.zeros(9), np.arange(9) + 1.0, device, wp.float64)
    for count in (9, 3, 0, 7, 9):
        values = np.arange(9, dtype=np.float64) + 1
        values[count:] = np.nan
        inputs[3].assign(values)
        inputs[4].assign([count])
        for _ in range(3):
            reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            expected = np.zeros(3)
            np.add.at(expected, np.arange(count) % 3, values[:count])
            np.testing.assert_array_equal(reducer.result.numpy(), expected)
            test.assertEqual(int(reducer.offsets.numpy()[-1]), count)
    for row_count in (0, 3):
        empty = _ContactRows(row_count, 0, 1, 1, wp.float64, device, use_cuda_graph=device.is_cuda)
        empty.reduce(*row_inputs([], [], [], [], device, wp.float64))
        test.assertEqual(int(empty.flags.numpy()[0]), 0)
        np.testing.assert_array_equal(empty.offsets.numpy(), np.zeros(row_count + 1, np.int32))
        np.testing.assert_array_equal(empty.result.numpy(), np.zeros(row_count))


def test_invalid_rows(test, device):
    """Fail illegal keys, duplicates, counts and nonfinite sums without publishing rows."""
    reducer = _ContactRows(2, 3, 4, 2, wp.float32, device)
    cases = (
        ([-1, 0, 1], [0, 1, 2], [0, 0, 0], [1, 2, 3], 3),
        ([2, 0, 1], [0, 1, 2], [0, 0, 0], [1, 2, 3], 3),
        ([0, 0, 1], [4, 1, 2], [0, 0, 0], [1, 2, 3], 3),
        ([0, 0, 1], [0, 1, 2], [-1, 0, 0], [1, 2, 3], 3),
        ([0, 0, 1], [0, 1, 2], [2, 0, 0], [1, 2, 3], 3),
        ([0, 0, 1], [0, 0, 2], [0, 0, 0], [1, 2, 3], 3),
        ([0, 0, 1], [0, 1, 2], [0, 0, 0], [np.nan, 2, 3], 3),
        ([0, 0, 1], [0, 1, 2], [0, 0, 0], [np.inf, 2, 3], 3),
        ([0, 0, 1], [0, 1, 2], [0, 0, 0], [np.finfo(np.float32).max] * 3, 3),
        ([0, 0, 1], [0, 1, 2], [0, 0, 0], [1, 2, 3], -1),
        ([0, 0, 1], [0, 1, 2], [0, 0, 0], [1, 2, 3], 4),
    )
    for rows, identities, local, values, count in cases:
        inputs = row_inputs(rows, identities, local, values, device, wp.float32)
        inputs[4].assign([count])
        reducer.reduce(*inputs)
        test.assertNotEqual(int(reducer.flags.numpy()[0]), 0)
        np.testing.assert_array_equal(reducer.result.numpy(), [0, 0])
        np.testing.assert_array_equal(reducer.offsets.numpy(), [0, 0, 0])
    # Duplicate contact identity with distinct local numbers is legitimate.
    reducer.reduce(*row_inputs([0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 2, 3], device, wp.float32))
    test.assertEqual(int(reducer.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(reducer.result.numpy(), [3, 3])


def test_key_and_storage_contracts(test, device):
    """Check sentinel, uint64 packing and input storage before launching or allocating."""
    with patch.object(wp, "empty", side_effect=AssertionError("allocation before validation")):
        for args in (
            (True, 1, 1, 1),
            (1, 1.0, 1, 1),
            (1, -1, 1, 1),
            (2**31 - 1, 1, 1, 1),
            (1, 2**31, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (2, 1, 2**63, 1),
            (1, 1, 2**64 - 1, 2),
        ):
            with test.assertRaises(ValueError):
                _ContactRows(*args, wp.float64, device)
    reducer = _ContactRows(1, 2, 2**63, 1, wp.float64, device)
    inputs = row_inputs([0, 0], [2**63 - 1, 2**40], [0, 0], [2, 3], device, wp.float64)
    reducer.reduce(*inputs)
    test.assertEqual(int(reducer.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(reducer.result.numpy(), [5])
    sentinel_boundary = _ContactRows(1, 2, 2**64 - 1, 1, wp.float64, device)
    sentinel_boundary.reduce(*row_inputs([0, 0], [2**64 - 2, 2**64 - 3], [0, 0], [1, 2], device, wp.float64))
    test.assertEqual(int(sentinel_boundary.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(sentinel_boundary.result.numpy(), [3])
    for index, replacement in (
        (0, wp.zeros(3, dtype=int, device=device)),
        (1, wp.zeros(2, dtype=int, device=device)),
        (2, wp.zeros((1, 2), dtype=int, device=device)),
        (3, wp.zeros(2, dtype=wp.float32, device=device)),
        (4, wp.zeros(2, dtype=int, device=device)),
        (0, reducer.indices),
        (1, reducer.keys),
        (2, reducer.index_scratch),
        (3, reducer.work),
        (4, reducer.flags),
    ):
        bad = list(inputs)
        bad[index] = replacement
        with test.assertRaises(ValueError):
            reducer.reduce(*bad)
    arrays = reducer.buffers()
    test.assertEqual(len({a.ptr for a in arrays if a.size}), len([a for a in arrays if a.size]))
    test.assertEqual(reducer.storage_bytes, sum(a.capacity for a in arrays))
    test.assertEqual(reducer.storage_bytes, 2 * (24 + 8) + 2 * 4 + 8 + 4)


def test_explicit_scratch(test, device):
    """Use only owned scratch after warmup on both the ordinary and captured paths."""
    reducer = _ContactRows(2, 257, 257, 1, wp.float64, device, use_cuda_graph=device.is_cuda)
    inputs = row_inputs(np.arange(257) % 2, np.arange(257), np.zeros(257), np.ones(257), device, wp.float64)
    reducer.reduce(*inputs)
    with patch.object(wp, "empty", side_effect=AssertionError("runtime allocation")):
        for _ in range(5):
            reducer.reduce(*inputs)
    test.assertEqual(int(reducer.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(reducer.result.numpy(), [129, 128])


def test_graph_rebinding(test, device):
    """Reuse count changes and recapture changed inputs, outputs, scratch and streams."""
    reducer = _ContactRows(3, 9, 9, 1, wp.float64, device, use_cuda_graph=True)
    inputs = list(row_inputs(np.arange(9) % 3, np.arange(9), np.zeros(9), np.arange(9) + 1.0, device, wp.float64))
    inputs[-1].assign([0])
    reducer.reduce(*inputs)
    test.assertEqual(reducer.graphs.captures["contact_rows"], 1)
    np.testing.assert_array_equal(reducer.result.numpy(), [0, 0, 0])
    for count in (3, 0, 7, 9):
        inputs[-1].assign([count])
        for _ in range(5):
            reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            expected = np.zeros(3)
            np.add.at(expected, np.arange(count) % 3, np.arange(count) + 1)
            np.testing.assert_array_equal(reducer.result.numpy(), expected)
    test.assertEqual(reducer.graphs.captures["contact_rows"], 1)
    inputs[3].fill_(np.nan)
    reducer.reduce(*inputs)
    test.assertNotEqual(int(reducer.flags.numpy()[0]), 0)
    np.testing.assert_array_equal(reducer.result.numpy(), [0, 0, 0])
    np.testing.assert_array_equal(reducer.offsets.numpy(), [0, 0, 0, 0])
    inputs[3].assign(np.arange(9) + 1.0)
    reducer.reduce(*inputs)
    test.assertEqual(int(reducer.flags.numpy()[0]), 0)
    test.assertEqual(reducer.graphs.captures["contact_rows"], 1)
    for i in range(5):
        captures = reducer.graphs.captures["contact_rows"]
        inputs[i] = wp.clone(inputs[i])
        reducer.reduce(*inputs)
        test.assertEqual(reducer.graphs.captures["contact_rows"], captures + 1)
    for name in ("keys", "key_scratch", "indices", "index_scratch", "work", "offsets", "result", "flags"):
        captures = reducer.graphs.captures["contact_rows"]
        setattr(reducer, name, wp.clone(getattr(reducer, name)))
        reducer.reduce(*inputs)
        test.assertEqual(reducer.graphs.captures["contact_rows"], captures + 1)
        np.testing.assert_array_equal(reducer.result.numpy(), [12, 15, 18])
    stream = wp.Stream(device)
    # Input writes completed on the original stream before another stream reads.
    wp.synchronize_device(device)
    captures = reducer.graphs.captures["contact_rows"]
    with wp.ScopedStream(stream):
        reducer.reduce(*inputs)
    wp.synchronize_stream(stream)
    test.assertEqual(reducer.graphs.captures["contact_rows"], captures + 1)
    np.testing.assert_array_equal(reducer.result.numpy(), [12, 15, 18])


class TestContactPool(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_zero_padding,
        test_sorted_native_rows,
        test_long_rows,
        test_counts_and_tails,
        test_invalid_rows,
        test_key_and_storage_contracts,
        test_explicit_scratch,
    ):
        add_function_test(TestContactPool, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestContactPool, "test_graph_rebinding", test_graph_rebinding, devices=[_device])


if __name__ == "__main__":
    unittest.main()
