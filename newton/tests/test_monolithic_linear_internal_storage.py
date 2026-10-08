# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve native owner-block summation while bounding CUDA topology storage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton._src.solvers.monolithic.bsr_reduction import _FixedInternalBsrReduction
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicPcgWarmStart,
)
from newton.tests.test_monolithic_linear_assembly import _pcg_config
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _triplets(rows, columns, device):
    count = len(rows)
    return SimpleNamespace(
        rows=wp.array(rows, dtype=int, device=device),
        columns=wp.array(columns, dtype=int, device=device),
        values=wp.zeros(count, dtype=wp.mat33, device=device),
        count=wp.full(1, count, dtype=int, device=device),
        status=wp.zeros(1, dtype=int, device=device),
    )


def _assert_native(test, owner, inputs):
    reference = sparse.bsr_zeros(owner.size, owner.size, wp.mat33, device=owner.device)
    sparse.bsr_set_from_triplets(reference, inputs.rows, inputs.columns, inputs.values, count=inputs.count)
    count = reference.nnz_sync()
    test.assertEqual(owner.matrix.nnz_sync(), count)
    np.testing.assert_array_equal(owner.matrix.offsets.numpy(), reference.offsets.numpy())
    np.testing.assert_array_equal(owner.matrix.columns.numpy()[:count], reference.columns.numpy()[:count])
    np.testing.assert_array_equal(
        owner.matrix.values.numpy()[:count].view(np.uint32), reference.values.numpy()[:count].view(np.uint32)
    )
    test.assertEqual(owner.status.numpy()[0], 0)
    test.assertEqual(inputs.status.numpy()[0], 0)


def test_internal_storage_native_order(test, device):
    """Keep whole-block pruning, original FP32 serial order and topology shrink/recovery."""
    rows = np.r_[np.zeros(41, np.int32), [1, 1, 2, 2, 0]]
    columns = np.r_[np.zeros(41, np.int32), [1, 1, 2, 2, 2]].astype(np.int32)
    owner = _FixedInternalBsrReduction(3, rows, columns, device, use_cuda_graph=device.is_cuda)
    inputs = _triplets(rows, columns, device)
    rng = np.random.default_rng(861)
    values = rng.choice(np.array([0, -0.0, 1, -1, 2**24, -(2**24)], np.float32), (len(rows), 3, 3))
    values[41:43] = np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32)
    for active in (0, len(rows), 3, 0, len(rows)):
        current = values.copy()
        current[active:] = -0.0
        inputs.values.assign(current)
        for _ in range(3):
            owner.build(inputs)
            _assert_native(test, owner, inputs)
    if device.is_cuda:
        test.assertEqual(owner.graphs.captures["fixed_internal_bsr"], 1)
        with patch.object(wp, "empty", side_effect=AssertionError("runtime allocation")):
            with patch.object(sparse, "bsr_set_from_triplets", side_effect=AssertionError("native GPU builder")):
                owner.build(inputs)
        _assert_native(test, owner, inputs)
    test.assertEqual(owner.payload_bytes, sum(a.capacity for a in owner.buffers()))
    diagonal = wp.zeros(3, dtype=wp.mat33, device=device)
    sparse.bsr_get_diag(owner.matrix, out=diagonal)
    np.testing.assert_array_equal(diagonal.numpy()[1], np.zeros((3, 3)))


def test_internal_storage_failure(test, device):
    """Treat nonfinite sums, changed owner indices and counts as hard failures."""
    rows, columns = np.array([0, 0, 1], np.int32), np.array([0, 0, 1], np.int32)
    owner = _FixedInternalBsrReduction(3, rows, columns, device, use_cuda_graph=device.is_cuda)
    inputs = _triplets(rows, columns, device)
    for mode in ("nan", "sum", "index", "count", "incoming"):
        values = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        inputs.status.zero_()
        inputs.rows.assign(rows)
        inputs.count.fill_(3)
        expected = MonolithicLinearStatus.NONFINITE_CONTRIBUTION
        if mode == "nan":
            values[0, 1, 2] = np.nan
        elif mode == "sum":
            values[:2, 0, 0] = np.finfo(np.float32).max
        elif mode == "index":
            inputs.rows.assign([2, 0, 1])
            expected = MonolithicLinearStatus.INVALID_ARGUMENT
        elif mode == "count":
            inputs.count.fill_(2)
            expected = MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW
        else:
            inputs.status.fill_(int(MonolithicLinearStatus.STALE_GENERATION))
            expected = MonolithicLinearStatus.STALE_GENERATION
        inputs.values.assign(values)
        owner.build(inputs)
        test.assertEqual(inputs.status.numpy()[0], expected)
        test.assertEqual(owner.status.numpy()[0], expected)
        for array in (owner.matrix.offsets, owner.matrix.columns, owner.matrix.values):
            np.testing.assert_array_equal(array.numpy(), np.zeros(array.numpy().shape))
        inputs.rows.assign(rows)
        inputs.count.fill_(3)
        inputs.status.zero_()
        inputs.values.assign(np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0))
        owner.build(inputs)
        _assert_native(test, owner, inputs)


def test_internal_storage_bindings(test, device):
    """Rebind every fixed input, output and CUDA scratch before graph replay."""
    rows, columns = np.array([0, 0, 1], np.int32), np.array([0, 0, 1], np.int32)
    owner = _FixedInternalBsrReduction(3, rows, columns, device, use_cuda_graph=device.is_cuda)
    inputs = _triplets(rows, columns, device)
    inputs.values.assign(np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0))
    owner.build(inputs)
    if device.is_cuda:
        captures = 1
        for object_, names in (
            (inputs, ("rows", "columns", "values", "count", "status")),
            (owner, tuple(owner._spec)),
            (owner.matrix, ("offsets", "columns", "values")),
        ):
            for name in names:
                setattr(object_, name, wp.clone(getattr(object_, name)))
                owner.build(inputs)
                _assert_native(test, owner, inputs)
                captures += 1
                test.assertEqual(owner.graphs.captures["fixed_internal_bsr"], captures)
        wp.synchronize_device(device)
        with wp.ScopedStream(wp.Stream(device)):
            owner.build(inputs)
            wp.synchronize_stream()
            _assert_native(test, owner, inputs)
        test.assertEqual(owner.graphs.captures["fixed_internal_bsr"], captures + 1)
    saved = owner.prefix
    owner.prefix = wp.zeros(1, dtype=int, device=device)
    with patch.object(wp, "launch", side_effect=AssertionError("launch before guard")):
        with test.assertRaises(ValueError):
            owner.build(inputs)
    owner.prefix = saved
    saved = inputs.values
    inputs.values = owner.matrix.values
    with patch.object(wp, "launch", side_effect=AssertionError("launch before guard")):
        with test.assertRaises(ValueError):
            owner.build(inputs)
    inputs.values = saved


def test_internal_storage_empty(test, device):
    """Construct zero blocks and empty rows on CPU and CUDA graph paths."""
    for size in (0, 3):
        rows = np.empty(0, np.int32)
        owner = _FixedInternalBsrReduction(size, rows, rows, device, use_cuda_graph=device.is_cuda)
        inputs = _triplets(rows, rows, device)
        owner.build(inputs)
        _assert_native(test, owner, inputs)


def test_fixed_workspace_solve(test, device):
    """Assemble dynamic contact coordinates and solve through guarded scaling/PCG generations."""
    for mode in ("diagnostic", "production"):
        workspace = MonolithicLinearWorkspace(
            MonolithicLinearLayout(2, 3),
            MonolithicLinearCapacities(64, 4, 0, 100, 102),
            wp.array([0, 1, 2], dtype=int, device=device),
            device=device,
            pcg_mode=mode,
            use_cuda_graph=device.is_cuda and mode == "production",
        )
        indices = np.arange(11, dtype=np.int32)
        internal_indices = np.array([0, 0, 1, 2], np.int32)
        workspace._set_fixed_triplet_pattern(
            global_rows=indices,
            global_columns=indices,
            internal_rows=internal_indices,
            internal_columns=internal_indices,
        )
        workspace._enable_fixed_bsr_storage()
        with patch.object(wp, "empty", side_effect=AssertionError("reconfigured storage")):
            with test.assertRaises(ValueError):
                workspace._enable_fixed_bsr_storage()
        fixed = workspace._fixed_bsr
        config = _pcg_config(linear_tolerance=2e-6, linear_absolute_tolerance=1e-7, true_residual_interval=2)
        diagonal = np.arange(2, 13, dtype=np.float32)
        for sequence, contact in enumerate((False, True, False, True), 1):
            generation = MonolithicLinearGeneration(1, sequence, 1, sequence)
            assembly = workspace.begin_assembly(generation)
            triplets = assembly.global_scalar_triplets
            triplets.values[:11].assign(diagonal)
            assembly.aq_actor_dense.assign(np.diag(diagonal[:2]))
            blocks = np.array([np.diag(diagonal[2 + 3 * i : 5 + 3 * i]) for i in range(3)])
            assembly.ax_internal_triplets.values.assign(np.concatenate([blocks[:1] / 2, blocks[:1] / 2, blocks[1:]]))
            expected = np.diag(diagonal)
            if contact:
                triplets.rows[11:15].assign([0, 0, 3, 3])
                triplets.columns[11:15].assign([0, 3, 0, 3])
                triplets.values[11:15].assign([1, -1, -1, 1])
                triplets.count.fill_(15)
                v = np.zeros(11, np.float32)
                v[0], v[3] = 1, -1
                expected += np.outer(v, v)
            if device.is_cuda and sequence > 1:
                with patch.object(wp, "empty", side_effect=AssertionError("GPU finalize allocated")):
                    with patch.object(sparse, "bsr_set_from_triplets", side_effect=AssertionError("native GPU build")):
                        result = workspace.finalize_assembly(generation=generation)
            else:
                result = workspace.finalize_assembly(generation=generation)
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
            np.testing.assert_array_equal(workspace.densify_for_test(generation=generation).raw_matrix, expected)
            test.assertEqual(
                workspace.build_scaling(wp.array(diagonal, device=device), epsilon_d=0.01, generation=generation),
                MonolithicLinearStatus.SUCCESS,
            )
            test.assertEqual(workspace.set_regularization(0.03, generation=generation), MonolithicLinearStatus.SUCCESS)
            test.assertEqual(
                workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=1e-10),
                MonolithicLinearStatus.SUCCESS,
            )
            rhs, y = wp.ones(11, dtype=float, device=device), wp.zeros(11, dtype=float, device=device)
            solve = workspace.solve_pcg(
                rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config
            )
            test.assertEqual(solve.status, MonolithicLinearStatus.SUCCESS)
            scaled = workspace.densify_for_test(generation=generation).scaled_matrix
            np.testing.assert_allclose(y.numpy(), np.linalg.solve(scaled, np.ones(11)), rtol=2e-5, atol=1e-6)
            stale = MonolithicLinearGeneration(0, 0, 0, 0)
            test.assertEqual(
                workspace.build_scaling(rhs, epsilon_d=0.01, generation=stale), MonolithicLinearStatus.STALE_GENERATION
            )
        if workspace.use_cuda_graph:
            test.assertEqual(fixed[0].graphs.captures["fixed_scalar_bsr"], 1)
            test.assertEqual(fixed[1].graphs.captures["fixed_internal_bsr"], 1)
        # Finite duplicate blocks can overflow in the internal reduction. Its
        # error must propagate even after that component clears all its output.
        generation = MonolithicLinearGeneration(2, 0, 2, 1)
        assembly = workspace.begin_assembly(generation)
        assembly.global_scalar_triplets.values[:11].assign(diagonal)
        blocks = np.zeros((4, 3, 3), np.float32)
        blocks[:2, 0, 0] = np.finfo(np.float32).max
        assembly.ax_internal_triplets.values.assign(blocks)
        test.assertEqual(
            workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.NONFINITE_CONTRIBUTION
        )
        with test.assertRaises(ValueError):
            workspace._enable_fixed_bsr_storage()


class TestMonolithicLinearInternalStorage(unittest.TestCase):
    def test_pattern_preflight(self):
        """Reject invalid immutable owner patterns before allocating any storage."""
        oversized = np.broadcast_to(np.array([0], np.int32), (2**31 // 9 + 1,))
        with patch.object(wp, "empty", side_effect=AssertionError("allocation before validation")):
            with patch.object(wp, "zeros", side_effect=AssertionError("allocation before validation")):
                for size, rows, columns in (
                    (True, [], []),
                    (-1, [], []),
                    (3, [0.5], [0]),
                    (3, [0], [3]),
                    (3, [0], []),
                    (3, [[0]], [[0]]),
                    (1, oversized, oversized),
                ):
                    with self.assertRaises(ValueError):
                        _FixedInternalBsrReduction(size, rows, columns, "cpu", use_cuda_graph=False)


for device in get_test_devices():
    for function in (
        test_internal_storage_native_order,
        test_internal_storage_failure,
        test_internal_storage_bindings,
        test_internal_storage_empty,
        test_fixed_workspace_solve,
    ):
        add_function_test(TestMonolithicLinearInternalStorage, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
