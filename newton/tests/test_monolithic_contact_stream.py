# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Whole-contact transactions across bounded producer batches."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.bsr_reduction import _FixedScalarBsrReduction
from newton._src.solvers.monolithic.contact_pool import _ContactPool
from newton.tests.test_monolithic_contact_pool import assign_contacts, check_admitted, contact_payload
from newton.tests.unittest_utils import add_function_test, get_test_devices


def append_payload(pool, data, indices):
    batch = {}
    count = len(indices)
    for name, values in data.items():
        if name in ("count", "overflow"):
            batch[name] = [count if name == "count" else 0]
        else:
            selected = np.asarray(values)[indices]
            # Invalid inactive tails must never be inspected or copied.
            padded = np.zeros((pool.batch_capacity, *selected.shape[1:]), dtype=selected.dtype)
            padded[:count] = selected
            if np.issubdtype(padded.dtype, np.floating):
                padded[count:] = np.nan
            batch[name] = padded
    assign_contacts(pool, batch)
    pool.append()


def test_stream_order(test, device):
    """Match complete records and native output bits across batch size and discovery order."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        data = contact_payload(native)
        full = _ContactPool(4, 2, 5, 8, 3, 64, dtype, device)
        assign_contacts(full, data)
        full.evaluate()
        baseline = tuple(owner.result.numpy().tobytes() for owner in (full.energy, full.residual, full.force))
        matrix = _FixedScalarBsrReduction(4, full.triplet_capacity, device, use_cuda_graph=device.is_cuda)

        def matrix_bits(records, matrix):
            status = wp.zeros(1, dtype=int, device=device)
            matrix.build(
                SimpleNamespace(
                    rows=records.triplet_rows,
                    columns=records.triplet_columns,
                    values=records.triplet_values,
                    count=records.triplet_count,
                    status=status,
                )
            )
            test.assertEqual(int(status.numpy()[0]), 0)
            nnz = matrix.matrix.nnz_sync()
            return tuple(
                a.numpy()[:n].tobytes()
                for a, n in ((matrix.matrix.offsets, 5), (matrix.matrix.columns, nnz), (matrix.matrix.values, nnz))
            )

        baseline_matrix = matrix_bits(full.records, matrix)
        for batch_size in (1, 2, 5):
            pool = _ContactPool(
                4, 2, 5, 8, 3, 64, dtype, device, batch_capacity=batch_size, use_cuda_graph=device.is_cuda
            )
            test.assertEqual(pool.candidates.factors.shape, (batch_size, 3, 3))
            test.assertEqual(pool.payload_bytes, sum(a.capacity for a in pool.buffers()))
            for order in (np.arange(5), np.array([4, 2, 0, 3, 1])):
                pool.begin()
                for start in range(0, 5, batch_size):
                    append_payload(pool, data, order[start : start + batch_size])
                    test.assertFalse(pool.records.accepted.numpy().any(), "Published before all batches finished")
                pool.finish()
                ordered = {
                    name: np.asarray(values)[order] if name not in ("count", "overflow") else values
                    for name, values in data.items()
                }
                check_admitted(test, pool, ordered, stable_order=False)
                test.assertEqual(matrix_bits(pool.records, matrix), baseline_matrix)
                test.assertEqual(
                    tuple(owner.result.numpy().tobytes() for owner in (pool.energy, pool.residual, pool.force)),
                    baseline,
                )
            if device.is_cuda:
                test.assertEqual(pool.graphs.captures["contact_stream_finish"], 1)


def test_stream_overflow(test, device):
    """Share one leak-free budget across batches and publish complete endpoint records."""
    data = contact_payload(np.float64)
    for capacity in (0, 1, 7, 8):
        pool = _ContactPool(
            4, 2, 5, capacity, 3, 64, wp.float64, device, batch_capacity=2, use_cuda_graph=device.is_cuda
        )
        for _ in range(3):
            pool.begin()
            for indices in ([0, 1], [2, 3], [4]):
                append_payload(pool, data, indices)
            pool.finish()
            mask = check_admitted(test, pool, data, stable_order=False)
            test.assertEqual(int(pool.records.requested.numpy()[0]), 8)
            test.assertEqual(int(pool.records.dropped.numpy()[0]), 4 - int(mask.sum()))
            test.assertLessEqual(int(pool.records.used.numpy()[0]), capacity)


def test_stream_invalid_late_batch(test, device):
    """Invalidate the whole evaluation for duplicates or bad payloads after the pool fills."""
    for capacity in (0, 3, 8):
        pool = _ContactPool(
            4, 2, 5, capacity, 3, 64, wp.float64, device, batch_capacity=2, use_cuda_graph=device.is_cuda
        )
        for name, index, value in (
            ("identities", (4,), 30),
            ("identities", (3,), 30),
            ("identities", (4,), 64),
            ("columns", (4, 0), -1),
            ("pairs", (4, 1, 5), np.inf),
            ("factors", (4, 0, 0), np.nan),
        ):
            data = contact_payload(np.float64)
            data[name] = np.asarray(data[name]).copy()
            data[name][index] = value
            pool.begin()
            for indices in ([0, 1], [2, 3], [4]):
                append_payload(pool, data, indices)
            pool.finish()
            test.assertNotEqual(int(pool.records.flags.numpy()[0]), 0, name)
            for field in ("accepted", "count", "used", "support_count", "triplet_count", "pairs"):
                test.assertFalse(getattr(pool.records, field).numpy().any(), field)
            for owner in (pool.energy, pool.residual, pool.force):
                test.assertFalse(owner.result.numpy().any())
                test.assertFalse(owner.offsets.numpy().any())


def test_stream_lifecycle(test, device):
    """Reject malformed batches before unsafe access; reset, rebind and replay without double append."""
    pool = _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device, batch_capacity=2, use_cuda_graph=device.is_cuda)
    with test.assertRaises(ValueError):
        pool.append()
    with test.assertRaises(ValueError):
        pool.finish()
    with test.assertRaises(ValueError):
        pool.evaluate()
    data = contact_payload(np.float64)
    for count in (-1, 3):
        pool.begin()
        append_payload(pool, data, [0, 1])
        pool.candidates.count.assign([count])
        pool.append()
        pool.finish()
        test.assertNotEqual(int(pool.records.flags.numpy()[0]), 0)
        test.assertFalse(pool.records.accepted.numpy().any())
    pool.begin()
    for _ in range(3):
        append_payload(pool, data, [0, 1])
    pool.finish()
    test.assertNotEqual(int(pool.records.flags.numpy()[0]) & 1, 0, "Input ledger exceeded capacity")
    # The graph must follow current bindings, including ledger and output arrays.
    for struct, spec in zip((pool.candidates, pool.records), pool._specs, strict=True):
        for name in spec:
            setattr(struct, name, wp.clone(getattr(struct, name)))
    pool._stream = tuple(wp.clone(a) for a in pool._stream)
    for count in (5, 0, 1, 5):
        pool.begin()
        for start in range(0, count, 2):
            append_payload(pool, data, list(range(start, min(count, start + 2))))
        pool.finish()
        test.assertEqual(int(pool.records.flags.numpy()[0]), 0)
        test.assertEqual(int(pool.records.requested.numpy()[0]), [0, 3, 4, 6, 6, 8][count])
    if device.is_cuda:
        captures = pool.graphs.captures["contact_stream_finish"]
        stream = wp.Stream(device)
        # Order existing default-stream writes before overwriting on this stream.
        wp.synchronize_device(device)
        with wp.ScopedStream(stream):
            pool.begin()
            for indices in ([0, 1], [2, 3], [4]):
                append_payload(pool, data, indices)
            pool.finish()
        wp.synchronize_stream(stream)
        check_admitted(test, pool, data, stable_order=False)
        test.assertEqual(pool.graphs.captures["contact_stream_finish"], captures + 1)
    with test.assertRaises(ValueError):
        pool.finish()
    empty = _ContactPool(0, 0, 0, 0, 1, 1, wp.float64, device, batch_capacity=0, use_cuda_graph=device.is_cuda)
    empty.begin()
    empty.append()
    empty.finish()
    test.assertEqual(int(empty.records.flags.numpy()[0]), 0)
    test.assertEqual(empty.payload_bytes, sum(a.capacity for a in empty.buffers()))
    with patch.object(wp, "zeros", side_effect=AssertionError("Allocated before checking batch bound")):
        for batch_capacity in (-1, 6, True, 1.0):
            with test.assertRaises(ValueError):
                _ContactPool(4, 2, 5, 8, 3, 64, wp.float64, device, batch_capacity=batch_capacity)


def test_stream_many_contacts(test, device):
    """Concatenate scans across blocks with k=0/1/17, full endpoint forces and bounded scratch."""
    m, k, b = 512, 17, 129
    rng = np.random.default_rng(801)
    counts = rng.choice([0, 1, k], m).astype(np.int32)
    full = int(np.maximum(1, counts).sum())
    # Make data independently of the batch arrays: all candidates are represented.
    data = {
        "count": [m],
        "overflow": [0],
        "identities": rng.permutation(m).astype(np.uint64),
        "support_counts": counts,
        "columns": np.tile(np.arange(k) % 8, (m, 1)),
        "residual": np.ones((m, k)),
        "weights": np.tile([1, 0, 0], (m, 1)).astype(np.float32),
        "factors": np.zeros((m, 3, k), np.float32),
        "energy": np.tile([1, 0], (m, 1)),
        "force_counts": np.full(m, 15),
        "force_rows": np.tile(np.arange(15) % 12, (m, 1)),
        "forces": np.tile(np.arange(15) - 7, (m, 1)),
        "pairs": np.zeros((m, 2, 6)),
    }
    data["factors"][:, 0, :] = np.arange(1, k + 1) / 32
    data["pairs"][:, :, 2] = [-1, 1]
    for capacity in (127, full - 1, full):
        pool = _ContactPool(
            8, 12, m, capacity, k, m, wp.float64, device, batch_capacity=b, use_cuda_graph=device.is_cuda
        )
        pool.begin()
        for start in range(0, m, b):
            append_payload(pool, data, np.arange(start, min(start + b, m)))
        pool.finish()
        mask = check_admitted(test, pool, data, stable_order=False)
        test.assertEqual(int(pool.records.requested.numpy()[0]), full)
        test.assertEqual(int(pool.records.dropped.numpy()[0]), m - int(mask.sum()))
        test.assertLessEqual(int(pool.records.used.numpy()[0]), capacity)
        test.assertEqual(pool.candidates.factors.shape[0], b)
        test.assertEqual(pool.payload_bytes, sum(a.capacity for a in pool.buffers()))
        if capacity == full:
            test.assertTrue(mask.all())


class TestContactStream(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_stream_order,
        test_stream_overflow,
        test_stream_invalid_late_batch,
        test_stream_lifecycle,
        test_stream_many_contacts,
    ):
        add_function_test(TestContactStream, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
