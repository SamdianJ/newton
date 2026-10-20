# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic native contact rows with bounded device storage."""

import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton._src.solvers.monolithic.bsr import (
    ASSEMBLY_STRUCTS,
    _append_monolithic_contact_factor_triplets,
    _create_triplets,
)
from newton._src.solvers.monolithic.bsr_reduction import _FixedScalarBsrReduction, _ScalarBsrReduction
from newton._src.solvers.monolithic.contact_pool import _ContactPool, _ContactRows
from newton.tests.test_monolithic_linear_assembly import _dense_bsr
from newton.tests.test_monolithic_linear_reduction import _reference
from newton.tests.unittest_utils import add_function_test, get_test_devices

_FP32 = ASSEMBLY_STRUCTS[wp.float32]


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
            with patch("newton._src.solvers.monolithic.contact_pool._CONTACT_WORKERS", 3):
                reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            np.testing.assert_array_equal(reducer.offsets.numpy(), [0, 2, 2, 6, 6, 7, 7])
            np.testing.assert_array_equal(reducer.result.numpy(), expected)


def test_long_rows(test, device):
    """Handle odd tails and multiple fixed blocks with an independent error bound."""
    rng = np.random.default_rng(831)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for count in (1, 3, 255, 256, 257, 20736, 20739, 131075):
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
    """Overwrite row outputs despite poisoned scratch as counts shrink, empty and grow."""
    capacity = 17
    reducer = _ContactRows(3, capacity, capacity, 1, wp.float64, device, use_cuda_graph=device.is_cuda)
    inputs = row_inputs(
        np.arange(capacity) % 3, np.arange(capacity), np.zeros(capacity), np.arange(capacity) + 1.0, device, wp.float64
    )
    for count in (capacity, 3, 0, 7, capacity):
        values = np.arange(capacity, dtype=np.float64) + 1
        values[count:] = np.nan
        inputs[3].assign(values)
        inputs[4].assign([count])
        snapshot = None
        for seed in range(3):
            for array in (reducer.keys, reducer.indices):
                array.fill_(seed)
            reducer.reduce(*inputs)
            test.assertEqual(int(reducer.flags.numpy()[0]), 0)
            expected = np.zeros(3)
            np.add.at(expected, np.arange(count) % 3, values[:count])
            np.testing.assert_array_equal(reducer.result.numpy(), expected)
            test.assertEqual(int(reducer.offsets.numpy()[-1]), count)
            current = tuple(array.numpy().tobytes() for array in (reducer.offsets, reducer.result, reducer.flags))
            if snapshot is not None:
                test.assertEqual(current, snapshot)
            snapshot = current
    if device.is_cuda:
        test.assertEqual(reducer.graphs.captures["contact_rows"], 1)
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
            (1, 2**30, 1, 1),
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
        (0, reducer.indices[:2]),
        (0, reducer.indices[2:]),
        (1, reducer.keys[:2]),
        (1, reducer.keys[2:]),
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
    for name in ("keys", "indices", "work", "offsets", "result", "flags"):
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


def contact_payload(dtype):
    data = {
        "count": [5],
        "overflow": [0],
        "identities": [30, 10, 40, 20, 50],
        "support_counts": [3, 0, 2, 1, 2],
        "columns": np.zeros((5, 3), np.int32),
        "residual": np.zeros((5, 3), dtype),
        "weights": np.zeros((5, 3), dtype),
        "factors": np.zeros((5, 3, 3), dtype),
        "energy": np.zeros((5, 2), dtype),
        "force_counts": np.zeros(5, np.int32),
        "force_rows": np.zeros((5, 15), np.int32),
        "forces": np.zeros((5, 15), dtype),
        "pairs": np.zeros((5, 2, 6), dtype),
    }
    data["columns"][0] = [0, 1, 0]
    data["residual"][0] = [2, -3, -1]
    data["weights"][0] = [4, 1, 2]
    data["factors"][0] = [[1, 2, -1], [2, -1, 1], [1, 1, 1]]
    data["energy"][0] = [0.5, 0.125]
    # Zero-DoF contact still carries energy and both world-origin wrenches.
    data["energy"][1] = [3, 2]
    data["pairs"][1, :, 2] = [-2, 2]
    # A tangent-only contribution must survive even with zero E and residual.
    data["columns"][2, :2] = [2, 3]
    data["weights"][2, 1] = 5
    data["factors"][2, 1, :2] = [1, -2]
    # Contact 3 is truly zero, including its projected B and endpoint forces.
    data["weights"][3] = 1
    data["columns"][4, :2] = [1, 3]
    data["residual"][4, :2] = [0.5, -0.75]
    data["weights"][4] = [2, 1, 0]
    data["factors"][4, :, :2] = [[1, 2], [0.5, -1], [0, 0]]
    data["force_counts"][0] = data["force_counts"][4] = 2
    data["force_rows"][[0, 4], :2] = [0, 1]
    data["forces"][0, :2] = [-3, 3]
    data["forces"][4, :2] = [2, -2]
    data["pairs"][0, :, 0] = [-3, 3]
    data["pairs"][4, :, 1] = [-2, 2]
    return data


def assign_contacts(pool, data, order=None):
    for name, values in data.items():
        array = np.asarray(values)
        if order is not None and name not in ("count", "overflow"):
            array = array[order]
        getattr(pool.candidates, name).assign(array)


def check_admitted(test, pool, data, *, stable_order=True):
    record = pool.records
    stored = {
        name: getattr(record, name).numpy()
        for name in (
            "accepted",
            "identities",
            "pairs",
            "factor_offsets",
            "factor_columns",
            "factor_values",
            "factor_weights",
            "triplet_rows",
            "triplet_columns",
            "triplet_values",
        )
    }
    mask = stored["accepted"].astype(bool)
    ids = np.asarray(data["identities"])
    accepted = sorted(np.flatnonzero(mask), key=lambda i: int(ids[i]))
    n = len(accepted)
    test.assertEqual(int(record.count.numpy()[0]), n)
    if not stable_order:
        slots = record.source_slots.numpy()[:n]
        test.assertEqual(set(map(int, slots)), set(accepted))
        accepted = slots

    np.testing.assert_array_equal(stored["identities"][:n], ids[accepted])
    test.assertEqual(int(record.used.numpy()[0]), sum(max(1, int(data["support_counts"][i])) for i in accepted))
    expected_r = np.zeros(pool.row_count)
    expected_f = np.zeros(pool.force_row_count)
    expected_e = np.zeros(4)
    expected_b = np.zeros((pool.row_count, pool.row_count))
    supports, forces, triplets = 0, 0, 0
    for contact, i in enumerate(accepted):
        k = int(data["support_counts"][i])
        columns = data["columns"][i, :k]
        np.add.at(expected_r, columns, data["residual"][i, :k])
        energy = data["energy"][i]
        expected_e[:2] += energy
        expected_e[2:] += np.abs(energy)
        f = int(data["force_counts"][i])
        np.add.at(expected_f, data["force_rows"][i, :f], data["forces"][i, :f])
        np.testing.assert_array_equal(stored["pairs"][contact], data["pairs"][i])
        for component in range(3):
            factor = 3 * contact + component
            offsets = stored["factor_offsets"]
            begin, end = offsets[factor : factor + 2]
            test.assertEqual(end - begin, k)
            np.testing.assert_array_equal(stored["factor_columns"][begin:end], columns)
            g = data["factors"][i, component, :k]
            np.testing.assert_array_equal(stored["factor_values"][begin:end], g)
            test.assertEqual(stored["factor_weights"][factor], data["weights"][i, component])
            # Independent FP64 scatter of the operator's native multiply order.
            native = data["weights"].dtype.type
            values = (native(data["weights"][i, component] * g[:, None]) * g[None, :]).astype(np.float64)
            for a in range(k):
                np.add.at(expected_b[columns[a]], columns, values[a])
        supports += k
        forces += f
        triplets += int(np.count_nonzero(data["weights"][i] > 0)) * k * k
    test.assertEqual(int(record.support_count.numpy()[0]), supports)
    test.assertEqual(int(record.factor_count.numpy()[0]), 3 * n)
    test.assertEqual(int(record.triplet_count.numpy()[0]), triplets)
    test.assertEqual(int(record.force_count.numpy()[0]), forces)
    test.assertEqual(int(record.energy_count.numpy()[0]), 4 * n)
    np.testing.assert_allclose(pool.residual.result.numpy(), expected_r, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.force.result.numpy(), expected_f, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pool.energy.result.numpy(), expected_e, rtol=1e-6, atol=1e-6)
    dense = np.zeros_like(expected_b)
    for row, col, value in zip(
        stored["triplet_rows"][:triplets],
        stored["triplet_columns"][:triplets],
        stored["triplet_values"][:triplets],
        strict=True,
    ):
        dense[row, col] += float(value)
    np.testing.assert_array_equal(dense, expected_b)
    test.assertEqual(int(record.flags.numpy()[0]), 0)
    return mask


def test_whole_contact_admission(test, device):
    """Commit E/r/factors/triplets/forces together at zero, exact and insufficient capacity."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        data = contact_payload(native)
        for capacity in (0, 1, 2, 7, 8, 9):
            pool = _ContactPool(4, 2, 5, capacity, 3, 64, dtype, device)
            assign_contacts(pool, data)
            for _ in range(5):
                pool.evaluate()
                mask = check_admitted(test, pool, data)
                test.assertFalse(mask[3], "All-zero contact consumed budget")
                test.assertEqual(int(pool.records.dropped.numpy()[0]), 4 - int(mask.sum()))
                test.assertLessEqual(int(pool.records.used.numpy()[0]), capacity)
                test.assertEqual(int(pool.records.requested.numpy()[0]), 8)
                if capacity >= 8:
                    np.testing.assert_array_equal(mask, [True, True, True, False, True])
                if capacity == 0:
                    test.assertFalse(mask.any())


def test_admission_order(test, device):
    """Preserve complete record and reduced output bits when producer order changes without overflow."""
    pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device, use_cuda_graph=device.is_cuda)
    data = contact_payload(np.float64)
    baseline = None
    rng = np.random.default_rng(923)
    for _ in range(8):
        order = rng.permutation(5)
        assign_contacts(pool, data, order)
        pool.evaluate()
        outputs = [
            pool.energy.result,
            pool.residual.result,
            pool.force.result,
            pool.records.identities,
            pool.records.pairs,
            pool.records.triplet_rows,
            pool.records.triplet_columns,
            pool.records.triplet_values,
        ]
        bits = tuple(a.numpy().tobytes() for a in outputs)
        if baseline is None:
            baseline = bits
        test.assertEqual(bits, baseline)
        test.assertEqual(int(pool.records.flags.numpy()[0]), 0)
    if device.is_cuda:
        test.assertEqual(pool.graphs.captures["contact_pool"], 1)


def test_invalid_contact_payload(test, device):
    """Reject detected illegal or nonfinite payloads before a full pool can hide them."""
    for capacity in (0, 8):
        pool = _ContactPool(4, 2, 5, capacity, 3, 64, wp.float64, device, use_cuda_graph=device.is_cuda)
        for name, index, value in (
            ("support_counts", (4,), -1),
            ("support_counts", (4,), 4),
            ("columns", (4, 0), -1),
            ("columns", (4, 0), 4),
            ("weights", (4, 0), -1),
            ("weights", (4, 0), np.inf),
            ("factors", (4, 0, 0), np.inf),
            ("factors", (4, 0, 0), np.finfo(np.float64).max),
            ("residual", (4, 0), np.nan),
            ("energy", (4, 0), np.inf),
            ("force_counts", (4,), 16),
            ("force_rows", (4, 0), 2),
            ("forces", (4, 0), np.nan),
            ("pairs", (4, 1, 5), np.inf),
            ("identities", (4,), 30),
            ("identities", (4,), 64),
            ("count", (0,), 6),
            ("count", (0,), -1),
            ("overflow", (0,), -1),
        ):
            data = contact_payload(np.float64)
            data[name] = np.asarray(data[name])
            data[name][index] = value
            assign_contacts(pool, data)
            pool.evaluate()
            test.assertNotEqual(int(pool.records.flags.numpy()[0]), 0, (name, index, value))
            np.testing.assert_array_equal(pool.records.accepted.numpy(), 0)
            test.assertEqual(int(pool.records.used.numpy()[0]), 0)
            test.assertEqual(int(pool.records.count.numpy()[0]), 0)
            np.testing.assert_array_equal(pool.energy.result.numpy(), 0)
            np.testing.assert_array_equal(pool.residual.result.numpy(), 0)
            np.testing.assert_array_equal(pool.force.result.numpy(), 0)
            np.testing.assert_array_equal(pool.records.triplet_values.numpy(), 0)
        assign_contacts(pool, contact_payload(np.float64))
        pool.evaluate()
        check_admitted(test, pool, contact_payload(np.float64))


def test_force_and_dense_operator(test, device):
    """Verify both endpoint forces, COM/world torques, virtual work and the existing FP32 B tree."""
    bary = np.array([2 / 3, 1 / 6, 1 / 6])
    vertices = np.array([[0.5, 0.1, 0.2], [0.3, -0.2, 0.4], [0.7, 0.3, -0.1]])
    point = bary @ vertices
    center = np.array([0.4, 0.0, 0.1])
    local_r = np.array([2.0, -3.0, 4.0])
    j = np.zeros((3, 15))
    for node in range(3):
        j[:, 3 * node : 3 * node + 3] = bary[node] * np.eye(3)
    j[:, 9:12] = -np.eye(3)
    j[:, 12:15] = np.cross(point - center, np.eye(3)).T
    directions = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        pool = _ContactPool(15, 15, 1, 15, 15, 2, dtype, device, use_cuda_graph=device.is_cuda)
        a = pool.candidates
        a.count.assign([1])
        a.identities.assign([1])
        a.support_counts.assign([15])
        a.columns.assign(np.arange(15)[None])
        a.residual.assign((j.T @ local_r)[None].astype(native))
        a.factors.assign((directions @ j)[None].astype(native))
        a.weights.assign([[7, 3, 5]])
        a.energy.assign([[0.5, 0.125]])
        forces = np.r_[(-bary[:, None] * local_r).ravel(), local_r, np.cross(point - center, local_r)]
        a.force_counts.assign([15])
        a.force_rows.assign(np.arange(15)[None])
        a.forces.assign(forces[None].astype(native))
        pair = np.r_[local_r, np.cross(point, local_r)]
        a.pairs.assign(np.stack([-pair, pair])[None].astype(native))
        pool.evaluate()
        test.assertEqual(int(pool.records.flags.numpy()[0]), 0)
        np.testing.assert_array_equal(pool.records.accepted.numpy(), [1])
        tolerance = 2e-6 if native == np.float32 else 2e-14
        published = pool.force.result.numpy().astype(np.float64)
        np.testing.assert_allclose(published, forces, rtol=0, atol=tolerance)
        np.testing.assert_allclose(published, -pool.residual.result.numpy(), rtol=0, atol=tolerance)
        np.testing.assert_allclose(published[:9].reshape(3, 3).sum(axis=0) + published[9:12], 0, atol=tolerance)
        total_torque = np.cross(vertices, published[:9].reshape(3, 3)).sum(axis=0)
        total_torque += published[12:15] + np.cross(center, published[9:12])
        np.testing.assert_allclose(total_torque, 0, atol=tolerance)
        np.testing.assert_array_equal(pool.records.pairs.numpy()[0].sum(axis=0), 0)
        delta = np.linspace(-0.2, 0.3, 15)
        test.assertAlmostEqual(float(delta @ published), -float(local_r @ (j @ delta)), delta=tolerance)
        record = pool.records
        count = int(record.triplet_count.numpy()[0])
        triplets = SimpleNamespace(
            rows=record.triplet_rows,
            columns=record.triplet_columns,
            values=record.triplet_values,
            count=record.triplet_count,
            status=wp.zeros(1, dtype=int, device=device),
        )
        matrix = sparse.bsr_zeros(15, 15, dtype, device=device)
        reduction = _ScalarBsrReduction(pool.triplet_capacity, device, use_cuda_graph=device.is_cuda, dtype=dtype)
        reduction.build(matrix, triplets, count, layout_identity=("contact-pool", 15), graphs=None)
        test.assertEqual(int(triplets.status.numpy()[0]), 0)
        reference = _reference(
            record.triplet_rows.numpy()[:count],
            record.triplet_columns.numpy()[:count],
            record.triplet_values.numpy()[:count],
            15,
        )
        np.testing.assert_array_equal(_dense_bsr(matrix), reference)
        fixed = _FixedScalarBsrReduction(15, pool.triplet_capacity, device, use_cuda_graph=device.is_cuda, dtype=dtype)
        fixed.build(triplets, layout_identity=("contact-pool", 15))
        test.assertEqual(int(triplets.status.numpy()[0]), 0)
        np.testing.assert_array_equal(_dense_bsr(fixed.matrix), reference)
        expected = sum(
            weight * np.outer(j.T @ direction, j.T @ direction)
            for weight, direction in zip([7, 3, 5], directions, strict=True)
        )
        np.testing.assert_allclose(_dense_bsr(matrix), expected, rtol=3e-7, atol=2e-6)


def test_pool_capacity_contracts(test, device):
    """Check every resource before allocation and honor existing factor and triplet limits."""
    for args, options in (
        ((4, 2, 5, True, 3, 64), {}),
        ((4, 2, -1, 8, 3, 64), {}),
        ((4, 2, 2**30, 8, 3, 64), {}),
        ((4, 2, 5, 2**31, 3, 64), {}),
        ((4, 2, 5, 8, 0, 64), {}),
        ((4, 2, 5, 8, 2**31 - 1, 64), {}),
        ((4, 2, 5, 8, 3, 2**63), {}),
        ((4, 2, 2**30 - 1, 0, 2, 64), {}),
        ((4, 2, 5, 8, 3, 64), {"contact_factor_capacity": True}),
        ((4, 2, 5, 8, 3, 64), {"contact_factor_capacity": -1}),
        ((4, 2, 5, 8, 3, 64), {"contact_triplet_capacity": 1.5}),
        ((4, 2, 5, 8, 3, 64), {"contact_triplet_capacity": 2**31}),
    ):
        with (
            patch.object(wp, "zeros", side_effect=AssertionError("allocation before guard")),
            patch.object(wp, "empty", side_effect=AssertionError("allocation before guard")),
        ):
            with test.assertRaises(ValueError):
                _ContactPool(*args, wp.float64, device, **options)
    for options, expected in (
        ({}, 100),
        ({"contact_factor_capacity": 15}, 100),
        ({"contact_factor_capacity": 12}, 4),
        ({"contact_factor_capacity": 0}, 0),
        ({"contact_triplet_capacity": 27}, 3),
        ({"contact_factor_capacity": 6, "contact_triplet_capacity": 27}, 2),
    ):
        pool = _ContactPool(4, 2, 5, 100, 3, 64, wp.float64, device, **options)
        test.assertEqual(pool.capacity, expected)
        test.assertEqual(pool.payload_bytes, sum(a.capacity for a in pool.buffers()))
        assign_contacts(pool, contact_payload(np.float64))
        pool.evaluate()
        check_admitted(test, pool, contact_payload(np.float64))
        if "contact_factor_capacity" in options:
            test.assertLessEqual(int(pool.records.factor_count.numpy()[0]), options["contact_factor_capacity"])
        if "contact_triplet_capacity" in options:
            test.assertLessEqual(int(pool.records.triplet_count.numpy()[0]), options["contact_triplet_capacity"])
    empty = _ContactPool(0, 0, 0, 0, 1, 1, wp.float64, device, use_cuda_graph=device.is_cuda)
    empty.evaluate()
    test.assertEqual(int(empty.records.flags.numpy()[0]), 0)
    test.assertEqual(empty.payload_bytes, sum(a.capacity for a in empty.buffers()))
    pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device)
    assign_contacts(pool, contact_payload(np.float64))
    pool.graphs.enabled = device.is_cuda
    pool.evaluate()
    original = pool.records.identities
    pool.records.identities = pool.candidates.identities
    with patch.object(wp, "launch", side_effect=AssertionError("aliased storage reached a kernel")):
        for _ in range(2):
            with test.assertRaisesRegex(ValueError, "alias"):
                pool.evaluate()
    pool.records.identities = original
    pool.evaluate()
    check_admitted(test, pool, contact_payload(np.float64))


def test_pool_reuse_and_failure(test, device):
    """Replay changed activation/counts, ignore stale staging tails and clear all channels on sum overflow."""
    pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device, use_cuda_graph=device.is_cuda)
    for count in (0, 5, 2, 0, 5):
        data = contact_payload(np.float64)
        data["count"] = [count]
        for name in ("residual", "weights", "factors", "energy", "forces", "pairs"):
            data[name][count:] = np.nan
        assign_contacts(pool, data)
        for _ in range(5):
            pool.evaluate()
            check_admitted(test, pool, data)
    if device.is_cuda:
        test.assertEqual(pool.graphs.captures["contact_pool"], 1)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for channel in ("energy", "residual", "forces"):
            overflow = _ContactPool(4, 2, 5, 8, 3, 64, dtype, device, use_cuda_graph=device.is_cuda)
            data = contact_payload(native)
            if channel == "energy":
                data[channel][[0, 1], 0] = np.finfo(native).max
            elif channel == "residual":
                data[channel][0, [0, 2]] = np.finfo(native).max
            else:
                data[channel][[0, 4], 0] = np.finfo(native).max
            assign_contacts(overflow, data)
            overflow.evaluate()
            test.assertNotEqual(int(overflow.records.flags.numpy()[0]), 0)
            np.testing.assert_array_equal(overflow.records.accepted.numpy(), 0)
            test.assertEqual(int(overflow.records.used.numpy()[0]), 0)
            for owner in (overflow.energy, overflow.residual, overflow.force):
                np.testing.assert_array_equal(owner.result.numpy(), 0)
            np.testing.assert_array_equal(overflow.records.pairs.numpy(), 0)
            np.testing.assert_array_equal(overflow.records.triplet_values.numpy(), 0)
    assign_contacts(pool, contact_payload(np.float64))
    pool.evaluate()
    with patch.object(wp, "empty", side_effect=AssertionError("runtime temporary allocation")):
        for _ in range(5):
            pool.evaluate()
    check_admitted(test, pool, contact_payload(np.float64))


def test_pool_graph_rebinding(test, device):
    """Recapture complete pooled chains when staging, records, scratch or stream storage changes."""
    pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device, use_cuda_graph=True)
    data = contact_payload(np.float64)
    assign_contacts(pool, data)
    pool.evaluate()
    for owner, name in (
        (pool.candidates, "count"),
        (pool.candidates, "residual"),
        (pool.candidates, "pairs"),
        (pool.records, "accepted"),
        (pool.records, "triplet_values"),
        (pool, "_prefix"),
        (pool.residual, "work"),
        (pool.force, "result"),
    ):
        captures = pool.graphs.captures["contact_pool"]
        setattr(owner, name, wp.clone(getattr(owner, name)))
        pool.evaluate()
        check_admitted(test, pool, data)
        test.assertEqual(pool.graphs.captures["contact_pool"], captures + 1)
    stream = wp.Stream(device)
    wp.synchronize_device(device)
    captures = pool.graphs.captures["contact_pool"]
    with wp.ScopedStream(stream):
        pool.evaluate()
        check_admitted(test, pool, data)
    test.assertEqual(pool.graphs.captures["contact_pool"], captures + 1)


def test_many_contact_reservations(test, device):
    """Validate 512 concurrent whole records and an exclusive scan across multiple blocks."""
    m, k = 512, 17
    rng = np.random.default_rng(203)
    counts = rng.choice([0, 1, k], m).astype(np.int32)
    for dtype in (wp.float32, wp.float64):
        full = int(np.maximum(1, counts).sum())
        for capacity in (0, 1, 127, full - 1, full):
            pool = _ContactPool(8, 2, m, capacity, k, m, dtype, device, use_cuda_graph=device.is_cuda)
            data = {name: getattr(pool.candidates, name).numpy() for name in pool._specs[0]}
            data["count"] = [m]
            data["overflow"] = [3]
            data["identities"] = rng.permutation(m).astype(np.uint64)
            data["support_counts"] = counts
            data["columns"][:] = np.arange(k) % 8
            data["residual"][:] = 1
            data["weights"][:, 0] = 1
            data["factors"][:, 0, :] = np.linspace(0.125, 1, k)
            data["energy"][:, 0] = 1
            data["force_counts"][:] = 2
            data["force_rows"][:, :2] = [0, 1]
            data["forces"][:, :2] = [-1, 1]
            data["pairs"][:, :, 2] = [-1, 1]
            assign_contacts(pool, data)
            for _ in range(3):
                pool.evaluate()
                mask = check_admitted(test, pool, data)
                test.assertEqual(int(pool.records.requested.numpy()[0]), full)
                test.assertEqual(int(pool.records.dropped.numpy()[0]), m - int(mask.sum()))
                test.assertEqual(int(pool.candidates.overflow.numpy()[0]), 3)
                if capacity == full:
                    test.assertTrue(mask.all())


def test_pool_native_energy_bits(test, device):
    """Preserve singleton native energy signs while separately recording absolute energy."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        pool = _ContactPool(0, 0, 1, 1, 1, 1, dtype, device)
        pool.candidates.count.assign([1])
        pool.candidates.energy.assign(np.array([[-0.0, -0.0]], dtype=native))
        pair = np.zeros((1, 2, 6), dtype=native)
        pair[0, :, 0] = [1, -1]
        pool.candidates.pairs.assign(pair)
        pool.evaluate()
        test.assertEqual(int(pool.records.flags.numpy()[0]), 0)
        np.testing.assert_array_equal(pool.records.accepted.numpy(), [1])
        test.assertEqual(pool.energy.result.numpy().tobytes(), np.array([-0.0, -0.0, 0.0, 0.0], native).tobytes())


def test_pool_invalid_scratch(test, device):
    """Reject replacement scratch with invalid capacity or dtype before any kernel starts."""
    for target in ("prefix", "row_work"):
        pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device)
        assign_contacts(pool, contact_payload(np.float64))
        pool.graphs.enabled = device.is_cuda
        pool.evaluate()
        if target == "prefix":
            pool._prefix = wp.zeros(1, dtype=wp.vec4i, device=device)
        else:
            pool.residual.work = wp.zeros(8, dtype=wp.float32, device=device)
        with patch.object(wp, "launch", side_effect=AssertionError("invalid storage reached a kernel")):
            with test.assertRaisesRegex(ValueError, "storage"):
                pool.evaluate()


@wp.kernel
def _legacy_pool_triplets(factors: Any, triplets: Any):
    _append_monolithic_contact_factor_triplets(factors, wp.tid(), triplets)


def test_pool_zero_weight_b_policy(test, device):
    """Skip zero-weight factors while preserving zero products in positive factors' B tree."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        _zero_weight_b_policy(test, device, dtype, native)


def _zero_weight_b_policy(test, device, dtype, native):
    pool = _ContactPool(1, 0, 3, 3, 1, 3, dtype, device, use_cuda_graph=device.is_cuda)
    a = pool.candidates
    a.count.assign([3])
    a.identities.assign([0, 1, 2])
    a.support_counts.fill_(1)
    factors = np.zeros((3, 3, 1), native)
    factors[:, 0, 0] = 1
    a.factors.assign(factors)
    a.weights.assign([[2**24, 1, 0], [1, 1, 0], [1, 1, 0]])
    pool.evaluate()
    test.assertEqual(pool.records.flags.numpy()[0], 0)
    np.testing.assert_array_equal(pool.records.accepted.numpy(), [1, 1, 1])
    record = pool.records
    token = wp.ones(1, dtype=int, device=device)
    structs = ASSEMBLY_STRUCTS[dtype]
    legacy = _create_triplets(structs.scalar, 9, 1, dtype, device, token, 1)
    factor = structs.factors()
    for name, values in (
        ("offsets", record.factor_offsets),
        ("columns", record.factor_columns),
        ("values", record.factor_values),
        ("local_indices", record.factor_local),
        ("weights", record.factor_weights),
        ("kind", record.factor_kind),
        ("candidate_tid", record.factor_contact),
        ("count", record.factor_count),
        ("status", legacy.status),
        ("active_token", token),
    ):
        setattr(factor, name, values)
    factor.capacity, factor.scalar_dof_count, factor.token = 9, 1, 1
    wp.launch(_legacy_pool_triplets, 9, [factor, legacy], device=device)
    old_matrix = sparse.bsr_zeros(1, 1, dtype, device=device)
    old = _ScalarBsrReduction(9, device, use_cuda_graph=device.is_cuda, dtype=dtype)
    old.build(old_matrix, legacy, int(legacy.count.numpy()[0]), layout_identity="zero-weight", graphs=None)
    new = _FixedScalarBsrReduction(1, pool.triplet_capacity, device, use_cuda_graph=device.is_cuda, dtype=dtype)
    new.build(
        SimpleNamespace(
            rows=record.triplet_rows,
            columns=record.triplet_columns,
            values=record.triplet_values,
            count=record.triplet_count,
            status=wp.zeros(1, dtype=int, device=device),
        )
    )
    # FP32 absorbs both unit terms into 2**24; FP64 keeps them.
    test.assertEqual(old_matrix.values.numpy()[0], native(2**24) if native == np.float32 else native(2**24 + 2))
    np.testing.assert_array_equal(new.matrix.values.numpy()[:1], old_matrix.values.numpy()[:1])
    test.assertEqual(int(record.triplet_count.numpy()[0]), int(legacy.count.numpy()[0]))
    test.assertEqual(int(record.triplet_count.numpy()[0]), 6)
    test.assertEqual(int(record.factor_count.numpy()[0]), 9)


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
        test_whole_contact_admission,
        test_admission_order,
        test_invalid_contact_payload,
        test_force_and_dense_operator,
        test_pool_capacity_contracts,
        test_pool_reuse_and_failure,
        test_many_contact_reservations,
        test_pool_native_energy_bits,
        test_pool_invalid_scratch,
        test_pool_zero_weight_b_policy,
    ):
        add_function_test(TestContactPool, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestContactPool, "test_graph_rebinding", test_graph_rebinding, devices=[_device])
        add_function_test(TestContactPool, "test_pool_graph_rebinding", test_pool_graph_rebinding, devices=[_device])


if __name__ == "__main__":
    unittest.main()
