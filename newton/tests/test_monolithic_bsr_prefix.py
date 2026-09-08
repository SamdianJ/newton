# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify count-bounded BSR construction without changing assembly semantics."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp
from warp import sparse

from newton._src.solvers.monolithic.linear import MonolithicLinearGeneration, MonolithicLinearStatus
from newton.examples.softbody.monolithic_contact_friction import FrictionCase
from newton.tests.test_solver_monolithic_linear import _contact, _dense_bsr, _owners, _workspace
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.validate_bsr_prefix import full_capacity_builder


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
            oracles.append(_dense_bsr(oracle))
        calls = []
        reads = []
        original_numpy = wp.array.numpy

        def read(array, _reads=reads, _numpy=original_numpy):
            _reads.append(array.shape)
            return _numpy(array)

        def build(matrix, rows, columns, values, _expected=expected, _calls=calls, **kwargs):
            _, buffer, count = _expected[len(_calls)]
            test.assertEqual(rows.size, count, "Builder still sorts unused capacity")
            for view, base in zip((rows, columns, values), (buffer.rows, buffer.columns, buffer.values), strict=True):
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
        # Existing protocol: 3 counts, 3 statuses, 2 validation statuses, 2 offsets.
        test.assertEqual(reads, [(1,)] * 8 + [(12,), (4,)])
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


def test_contact_trajectory_parity(test, device):
    """Preserve contacting motion and committed friction history against the old builder."""
    trajectories = []
    for capacity in (True, False):
        case = FrictionCase(device)
        context = (
            patch.object(
                sparse,
                "bsr_set_from_triplets",
                new=full_capacity_builder(case.solver._linear, sparse.bsr_set_from_triplets),
            )
            if capacity
            else nullcontext()
        )
        trace = []
        with context:
            for step in range(150):
                case.step()
                test.assertFalse(case.solver.last_stats.rolled_back)
                history = case.solver._contact.committed
                if step % 25 == 24:
                    trace.append(
                        [
                            case.state.joint_q.numpy(),
                            case.state.particle_q.numpy(),
                            history.valid.numpy(),
                            history.xi_local.numpy(),
                            history.normal_local.numpy(),
                        ]
                    )
        test.assertGreater(history.valid.numpy().sum(), 0)
        test.assertGreater(np.linalg.norm(history.xi_local.numpy()), 0)
        test.assertEqual(case.solver._history_epoch, 150)
        trajectories.append(trace)
    for old, new in zip(*trajectories, strict=True):
        # Independent CUDA trajectories and nnz-based matvec selection can
        # change reduction roundoff. Reuse the existing G5 tolerances.
        for before, after in zip(old[:2], new[:2], strict=True):
            error = np.linalg.norm(before.astype(float) - after) / max(np.linalg.norm(before), 1e-12)
            test.assertLessEqual(error, 1e-5 if device.is_cpu else 5e-5)
        np.testing.assert_array_equal(old[2], new[2])
        np.testing.assert_allclose(old[3], new[3], atol=1e-7)
        np.testing.assert_allclose(old[4], new[4], rtol=1e-5 if device.is_cpu else 5e-5, atol=1e-7)


class TestMonolithicBsrPrefix(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_prefix_views_and_parity,
        test_empty_and_full_prefix,
        test_count_guards_before_builder,
        test_contact_trajectory_parity,
    ):
        add_function_test(TestMonolithicBsrPrefix, function.__name__, function, devices=[device])


if __name__ == "__main__":
    unittest.main()
