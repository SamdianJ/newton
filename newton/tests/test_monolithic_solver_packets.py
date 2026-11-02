# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check nonlinear summary boundaries against independent arithmetic."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import iteration_device as kernels
from newton._src.solvers.monolithic.iteration import blocks
from newton._src.solvers.monolithic.reductions import SegmentedReduction
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.tests.test_monolithic_solver_coupled import scene, solver
from newton.tests.test_monolithic_solver_direction import identity_workspace, qualify
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_assessment_readback(test, device):
    """Collect one assessment packet while preserving physical and linear group meanings."""
    for precision in ("fp32", "fp64"):
        run = solver(scene(device), precision=precision, mode="production", graph=device.is_cuda)
        state = run.create_state(dt=0.001)
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        reads = []
        original = wp.array.numpy

        def read(array, *args, reads=reads, original=original, **kwargs):
            reads.append(array)
            return original(array, *args, **kwargs)

        with patch.object(wp.array, "numpy", read):
            assessment = run._problem.assess(run._accepted_r, run._reference_device)
        with test.subTest(precision=precision):
            test.assertEqual(len(reads), 1)
            native = run._numpy_dtype
            r, tau, scale = (v.numpy() for v in (run._accepted_r, run._criterion.tau, run._linear.scale))
            np.testing.assert_array_equal(run._rhs.numpy(), -scale * r)
            test.assertEqual(assessment.converged, bool(np.all(abs(r) <= tau)))
            np.testing.assert_allclose(
                assessment.merit, np.sqrt(np.mean((r / tau) ** 2)), rtol=64 * np.finfo(native).eps
            )
            for index, (name, part) in enumerate(blocks(run)):
                test.assertEqual(assessment.caps[name], native(0.5) * np.min(scale[part] * tau[part], initial=np.inf))
                np.testing.assert_allclose(
                    run._rhs_norms_host[index], np.linalg.norm(scale[part] * r[part]), rtol=64 * np.finfo(native).eps
                )


def test_posterior_readback(test, device):
    """Read all external and internal direction norms at one posterior boundary."""
    run = identity_workspace(device, "production", device.is_cuda)
    residual = np.linspace(-0.5, 0.5, run._delta.size)
    original, posterior = wp.array.numpy, run._physical_linear_error
    captures = None
    for regularization in (0.0, 0.25, 0.0):
        reads, ready = [], [False]

        def compute(*args, ready=ready, **kwargs):
            result = posterior(*args, **kwargs)
            ready[0] = True
            return result

        def read(array, *args, ready=ready, reads=reads, **kwargs):
            if ready[0]:
                reads.append(array)
            return original(array, *args, **kwargs)

        with patch.object(run, "_physical_linear_error", compute), patch.object(wp.array, "numpy", read):
            success = qualify(run, residual, dict.fromkeys(dict(blocks(run)), 1e-10), regularization=regularization)
        with test.subTest(regularization=regularization):
            test.assertTrue(success)
            test.assertEqual(len(reads), 1)
            post = run._variational["linear_postchecks"][-1]["blocks"]
            error, original_error, term = (
                v.numpy() for v in (run._post_error, run._post_original, run._post_regularization)
            )
            internal = run._linear._masked_pcg_plan.norms.numpy()
            for index, (name, part) in enumerate(blocks(run)):
                np.testing.assert_allclose(post[name]["residual"], np.linalg.norm(error[part]), rtol=1e-12, atol=1e-15)
                np.testing.assert_allclose(
                    post[name]["original_residual"], np.linalg.norm(original_error[part]), rtol=1e-12, atol=1e-15
                )
                np.testing.assert_allclose(
                    post[name]["regularization_term"], np.linalg.norm(term[part]), rtol=1e-12, atol=1e-15
                )
                test.assertEqual(post[name]["internal_residual"], internal[index])
            if device.is_cuda:
                current = run._graphs.summary()["segments"]["nonlinear_posterior_norms"]["captures"]
                captures = current if captures is None else captures
                test.assertEqual(current, captures)


def test_nonfinite_assessment(test, device):
    """Reject a nonfinite residual even if the physical maximum reduction drops its NaN."""
    run = solver(scene(device, contact=False), mode="production", graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    run.step(state, state, run.model.control(), dt=0.001)
    for bad in (np.nan, np.inf):
        residual = np.zeros(run._accepted_r.size, dtype=run._numpy_dtype)
        residual[0] = bad
        run._accepted_r.assign(residual)
        with test.subTest(value=bad), test.assertRaises(_StepFailure):
            run._problem.assess(run._accepted_r, run._reference_device)


def test_posterior_schema(test, device):
    """Preserve fixed-order multi-vector norms, empty groups and native lambda-zero semantics."""
    rng = np.random.default_rng(307)
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for n in (0, 1025):
            groups = [np.arange(n), np.arange(min(n, 3)), np.empty(0, dtype=int), np.arange(3, n), np.arange(0, n, 2)]
            reduction = SegmentedReduction(n, groups, 5, dtype=dtype, device=device)
            host = (rng.normal(size=(5, n)) * np.geomspace(1e-12, 1e12, n)).astype(native)
            arrays = [wp.array(v, dtype=dtype, device=device) for v in host]
            coefficient = wp.zeros(1, dtype=dtype, device=device)
            internal_values = np.arange(len(groups), dtype=native) + 0.5
            internal = wp.array(internal_values, dtype=dtype, device=device)
            packet = wp.empty((len(groups), 6), dtype=dtype, device=device)

            def execute(arrays=arrays, coefficient=coefficient, reduction=reduction, internal=internal, packet=packet):
                wp.launch(
                    kernels._posterior_chunks,
                    reduction.starts.size,
                    [*arrays, coefficient, reduction.rows, reduction.starts, reduction.ends, reduction.partials],
                    device=device,
                )
                wp.launch(
                    kernels._posterior_groups,
                    packet.shape,
                    [reduction.partials, reduction.offsets, internal, packet],
                    device=device,
                )

            execute()
            graph = None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    execute()
                graph = capture.graph
            for requested in (0.0, 0.25, 0.0, 1e-50):
                coefficient.fill_(native(requested))
                for index, array in enumerate(arrays):
                    array.assign(host[index] if native(requested) or index < 3 else np.full(n, np.nan, dtype=native))
                if graph is None:
                    execute()
                else:
                    wp.capture_launch(graph)
                actual = packet.numpy().copy()
                expected = np.array([[np.sum(v[p].astype(float) ** 2) for v in host] for p in groups])
                if not native(requested):
                    expected[:, 3], expected[:, 4] = expected[:, 0], 0
                np.testing.assert_allclose(actual[:, :5], expected, rtol=64 * np.finfo(native).eps, atol=0)
                np.testing.assert_array_equal(actual[:, 5], internal_values)
                execute()
                np.testing.assert_array_equal(packet.numpy(), actual)
            if n:
                coefficient.fill_(1)
                for index in range(5):
                    for bad in (np.nan, np.inf):
                        for array, values in zip(arrays, host, strict=True):
                            array.assign(values)
                        values = host[index].copy()
                        values[13] = bad
                        arrays[index].assign(values)
                        execute()
                        test.assertFalse(np.isfinite(packet.numpy()[0, index]))
                        np.testing.assert_array_equal(packet.numpy()[2, :5], 0)
            for array in arrays:
                array.zero_()
            execute()
            np.testing.assert_array_equal(packet.numpy()[:, :5], 0)


def test_assessment_schema(test, device):
    """Keep physical counts exact and overwrite padding when group counts differ."""
    for dtype, native in ((wp.float32, np.float32), (wp.float64, np.float64)):
        for linear_count, physical_count in ((2, 4), (4, 2), (0, 0)):
            host = np.arange(linear_count * 5, dtype=native).reshape(linear_count, 5) / native(3)
            sums = wp.array(host, dtype=dtype, device=device)
            worst = wp.full(physical_count, 0.5, dtype=dtype, device=device)
            counts = np.full(physical_count, 2**24 + 1, dtype=np.int32)
            limited = wp.array(counts, dtype=int, device=device)
            total = wp.array([native(1.0 / 3)], dtype=dtype, device=device)
            packet = wp.empty(max(1, linear_count, physical_count), dtype=kernels._AssessmentPacket, device=device)
            wp.launch(kernels._pack_assessment, packet.size, [sums, worst, limited, total, packet], device=device)
            values = packet.numpy()
            for index, name in enumerate(
                ("rhs_square", "raw_square", "residual_square", "reference_square", "minimum")
            ):
                np.testing.assert_array_equal(values[name][:linear_count], host[:, index])
                np.testing.assert_array_equal(values[name][linear_count:], 0)
            np.testing.assert_array_equal(values["worst"][:physical_count], 0.5)
            np.testing.assert_array_equal(values["limited"][:physical_count], counts)
            np.testing.assert_array_equal(values["worst"][physical_count:], 0)
            np.testing.assert_array_equal(values["limited"][physical_count:], 0)
            test.assertEqual(values["merit_square"][0], native(1.0 / 3))
            np.testing.assert_array_equal(values["merit_square"][1:], 0)


class TestPackets(unittest.TestCase):
    pass


for function in (
    test_assessment_readback,
    test_posterior_readback,
    test_posterior_schema,
    test_assessment_schema,
    test_nonfinite_assessment,
):
    add_function_test(TestPackets, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
