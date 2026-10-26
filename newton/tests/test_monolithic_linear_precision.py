# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that the linear workspace stores and solves in one selected precision."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from newton.tests.test_monolithic_linear_assembly import _dense_bsr
from newton.tests.test_monolithic_linear_reduction import _reference
from newton.tests.unittest_utils import add_function_test, get_test_devices

_PRECISIONS = ((wp.float32, np.float32, wp.mat33), (wp.float64, np.float64, wp.mat33d))
# Convergence thresholds stay FP32 in either precision.
_THRESHOLDS = {"group_floors", "group_limits"}


def _workspace(device, dtype, *, fixed=False, capacity=256):
    layout = MonolithicLinearLayout(2, 3)
    workspace = MonolithicLinearWorkspace(
        layout,
        MonolithicLinearCapacities(capacity, 3, 0, 200, 202),
        wp.array([0, 1, 2], dtype=int, device=device),
        device=device,
        pcg_mode="production",
        dtype=dtype,
    )
    if fixed:
        dynamic = np.arange(3, dtype=np.int32)
        workspace._set_fixed_triplet_pattern(
            global_rows=np.empty(0, np.int32),
            global_columns=np.empty(0, np.int32),
            internal_rows=dynamic,
            internal_columns=dynamic,
        )
        workspace._enable_fixed_bsr_storage()
    return workspace


def _assemble(workspace, sequence, rows, columns, values, native):
    generation = MonolithicLinearGeneration(1, 0, 1, sequence)
    assembly = workspace.begin_assembly(generation)
    buffer = assembly.global_scalar_triplets
    start = int(buffer.count.numpy()[0])
    end = start + len(values)
    buffer.rows[start:end].assign(np.asarray(rows, dtype=np.int32))
    buffer.columns[start:end].assign(np.asarray(columns, dtype=np.int32))
    buffer.values[start:end].assign(np.asarray(values, dtype=native))
    buffer.count.fill_(end)
    return generation, workspace.finalize_assembly(generation=generation)


def _float_arrays(owner, prefix=""):
    """Yield every floating-point array held by an owner, its structs and its BSR matrices."""
    for name, value in vars(owner).items():
        if hasattr(value, "_cls"):
            values = {field: getattr(value, field) for field in value._cls.vars}
        elif hasattr(value, "block_shape"):
            values = {"values": value.values}
        else:
            values = {"": value}
        for field, array in values.items():
            scalar = getattr(getattr(array, "dtype", None), "_wp_scalar_type_", getattr(array, "dtype", None))
            if isinstance(array, wp.array) and scalar in (wp.float16, wp.float32, wp.float64):
                yield prefix + name + (f".{field}" if field else ""), array


def _spd(rng, size, condition):
    basis, _ = np.linalg.qr(rng.normal(size=(size, size)))
    return (basis * np.geomspace(1.0, condition, size)) @ basis.T


def test_storage_follows_precision(test, device):
    """Allocate every matrix, scaling, factor and PCG value array in the selected precision."""
    for dtype, _, matrix_type in _PRECISIONS:
        for fixed in (False, True):
            with test.subTest(dtype=dtype.__name__, fixed=fixed):
                workspace = _workspace(device, dtype, fixed=fixed)
                test.assertIs(workspace.dtype, dtype)
                arrays = dict(_float_arrays(workspace))
                reductions = workspace._fixed_bsr or (workspace._scalar_reduction,)
                for index, reduction in enumerate(reductions):
                    arrays.update(_float_arrays(reduction, f"reduction{index}."))
                test.assertIn("k_global_scalar_bsr.values", arrays)
                test.assertIn("_global.values", arrays)
                test.assertIn("_factors.weights", arrays)
                for name, array in arrays.items():
                    if name in _THRESHOLDS:
                        test.assertEqual(array.dtype, wp.float32, name)
                    else:
                        test.assertIn(array.dtype, (dtype, matrix_type), name)


def test_fixed_order_tree(test, device):
    """Reduce duplicates in a fixed pair tree over native value bits, for any input order."""
    one = np.float64(1.0)
    tiny = np.float64(2.0**-40)
    for dtype, native, _ in _PRECISIONS:
        values = np.array([one, tiny, tiny, 2.0**24, 1, 1, 1e8, -1e8, 3, -2, *([tiny] * 37)], dtype=native)
        rows = np.r_[np.zeros(3), np.ones(3), np.full(4, 2), np.full(37, 4)].astype(np.int32)
        columns = rows.copy()
        expected = _reference(rows, columns, values, 11)
        if native == np.float64:
            # Terms below FP32 resolution survive only in FP64 storage.
            test.assertEqual(expected[0, 0], one + 2 * tiny)
        for fixed in (False, True):
            with test.subTest(dtype=dtype.__name__, fixed=fixed):
                workspace = _workspace(device, dtype, fixed=fixed)
                rng = np.random.default_rng(405)
                for sequence in range(1, 6):
                    order = np.arange(len(values)) if sequence == 1 else rng.permutation(len(values))
                    if fixed:
                        # The fixed internal owner pattern is part of every assembly.
                        workspace._internal.values.assign(np.zeros((3, 3, 3), dtype=native))
                    _, result = _assemble(workspace, sequence, rows[order], columns[order], values[order], native)
                    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
                    test.assertEqual(workspace.k_global_scalar_bsr.values.dtype, dtype)
                    dense = _dense_bsr(workspace.k_global_scalar_bsr).astype(native)
                    np.testing.assert_array_equal(dense, expected)


def test_dense_solve_matches_oracle(test, device):
    """Solve coupled systems to the accuracy each precision can attain."""
    rows, columns = np.indices((11, 11)).reshape(2, -1).astype(np.int32)
    # FP32 attains about eps32 * condition; FP64 resolves a 1e6 condition number.
    cases = ((wp.float32, np.float32, 1e2, 1e-5, 1e-4), (wp.float64, np.float64, 1e6, 1e-10, 1e-6))
    for dtype, native, condition, tolerance, accuracy in cases:
        with test.subTest(dtype=dtype.__name__):
            rng = np.random.default_rng(7)
            matrix = _spd(rng, 11, condition)
            rhs = rng.normal(size=11)
            exact = np.linalg.solve(matrix, rhs)
            workspace = _workspace(device, dtype)
            generation, result = _assemble(workspace, 1, rows, columns, matrix.reshape(-1).astype(native), native)
            test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
            dynamic = wp.array(np.diag(matrix).astype(native), dtype=dtype, device=device)
            for status in (
                workspace.build_scaling(dynamic, epsilon_d=1e-3, generation=generation),
                workspace.set_regularization(0.0, generation=generation),
                workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
            ):
                test.assertEqual(status, MonolithicLinearStatus.SUCCESS)
            scale = workspace.scale.numpy().astype(np.float64)
            scaled_rhs = wp.array((scale * rhs).astype(native), dtype=dtype, device=device)
            y, delta = wp.zeros(11, dtype=dtype, device=device), wp.zeros(11, dtype=dtype, device=device)
            solved = workspace.solve_pcg(
                scaled_rhs,
                y,
                generation=generation,
                warm_start=MonolithicPcgWarmStart.ZERO,
                config=MonolithicPcgConfig(200, 10, tolerance, 1e-30, 1e-30, 1e-30, linear_absolute_tolerance=0.0),
                delta=delta,
            )
            test.assertEqual(solved.status, MonolithicLinearStatus.SUCCESS, solved)
            error = np.linalg.norm(delta.numpy() - exact) / np.linalg.norm(exact)
            test.assertLess(error, accuracy)


class TestMonolithicLinearPrecision(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (test_storage_follows_precision, test_fixed_order_tree, test_dense_solve_matches_oracle):
        add_function_test(TestMonolithicLinearPrecision, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
