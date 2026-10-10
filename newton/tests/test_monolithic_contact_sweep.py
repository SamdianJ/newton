# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify ancestor motion envelopes without equating sampled coverage with a floating-point proof."""

import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import _TreeKinematics
from newton._src.solvers.monolithic.contact_sweep import _next_upper, _product_upper, _sum_upper
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_oracle import Kinematics


@wp.kernel(enable_backward=False)
def _rounding_values(a: wp.array[wp.float64], b: wp.array[wp.float64], out: wp.array2d[wp.float64]):
    i = wp.tid()
    out[i, 0] = _next_upper(a[i])
    out[i, 1] = _sum_upper(a[i], b[i])
    out[i, 2] = _product_upper(a[i], b[i])


def test_outward_arithmetic(test, device):
    """Check device next-up bits and operation upper bounds against exact rational arithmetic."""
    values = np.array([0.0, -0.0, np.nextafter(0.0, 1.0), 1e-300, 1e-150, 0.5, 1, np.nextafter(1, 2), 1e150])
    a = np.repeat(values, len(values))
    b = np.tile(values, len(values))
    out = wp.zeros((len(a), 3), dtype=wp.float64, device=device)
    wp.launch(
        _rounding_values,
        len(a),
        [wp.array(a, dtype=wp.float64, device=device), wp.array(b, dtype=wp.float64, device=device), out],
        device=device,
    )
    actual = out.numpy()
    np.testing.assert_array_equal(actual[:, 0], np.nextafter(a, np.inf))
    for left, right, row in zip(a, b, actual, strict=True):
        x, y = Fraction(float(left)), Fraction(float(right))
        test.assertGreaterEqual(Fraction(float(row[1])), x + y)
        test.assertGreaterEqual(Fraction(float(row[2])), x * y)


def sweep_tree(device, dtype):
    """Build a branched noncoplanar scalar tree with fixed joints and offset frames."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    bodies = [builder.add_link(mass=1, com=(0.03, -0.07, 0.11), inertia=wp.diag(wp.vec3(0.1))) for _ in range(8)]
    joints = []
    for i, (parent, kind) in enumerate(zip((-1, 0, 1, 2, 3, 4, 5, 1), (3, 1, 1, 0, 3, 1, 0, 1), strict=True)):
        options = {
            "parent_xform": wp.transform((0.13, -0.07, 0.21), wp.quat_rpy(0.2, -0.1, 0.4)),
            "child_xform": wp.transform((-0.04, 0.02, 0.09), wp.quat_rpy(-0.3, 0.2, 0.1)),
        }
        if kind != 3:
            options.update(axis=(1, 2 + i, -1), armature=0.01, limit_ke=0, limit_kd=0)
        factory = {3: builder.add_joint_fixed, 1: builder.add_joint_revolute, 0: builder.add_joint_prismatic}[kind]
        joints.append(factory(-1 if parent < 0 else bodies[parent], bodies[i], **options))
    builder.add_articulation(joints)
    return _TreeKinematics(builder.finalize(device=device), dtype=dtype)


def test_sweep_coverage(test, device):
    """Enclose independently evaluated offset points along small, large and whole-turn joint paths."""
    from newton._src.solvers.monolithic.contact_sweep import _ArticulationSweep  # noqa: PLC0415

    rng = np.random.default_rng(804)
    points = rng.uniform(-0.2, 0.2, (8, 12, 3)) + np.array([0.13, 0.04, -0.09])
    radii = np.linalg.norm(points, axis=2).max(axis=1)
    for dtype in (wp.float64, wp.float32):
        k = sweep_tree(device, dtype)
        oracle = Kinematics(k.model)
        sweep = _ArticulationSweep(k, radii)
        q0_host = np.array([0.2, -0.4, -0.8, 0.3, 0.7, -0.2], dtype=k.numpy_dtype)
        q0, q1 = (wp.array(q0_host, dtype=dtype, device=device) for _ in range(2))
        initial_com, initial_rotation = oracle.evaluate(q0_host)[:2]
        initial_origin = initial_com - np.einsum("bij,bj->bi", initial_rotation, k.model.body_com.numpy())
        initial = initial_origin[:, None] + np.einsum("bij,bpj->bpi", initial_rotation, points)
        starts = []
        for delta in (np.zeros(6), np.full(6, 1e-7), np.array([0.7, -1.2, 1.7, np.pi, -1.6, 2 * np.pi])):
            final = (q0_host + delta).astype(k.numpy_dtype)
            q1.assign(final)
            sweep.update(q0, q1)
            test.assertEqual(int(sweep.status.numpy()[0]), 0)
            bound = sweep.displacement.numpy().copy()
            starts.append(bound)
            for alpha in np.linspace(0, 1, 41):
                coordinates = ((1 - alpha) * q0_host.astype(float) + alpha * final.astype(float)).astype(k.numpy_dtype)
                com, rotation = oracle.evaluate(coordinates)[:2]
                origin = com - np.einsum("bij,bj->bi", rotation, k.model.body_com.numpy())
                independent = origin[:, None] + np.einsum("bij,bpj->bpi", rotation, points)
                # Native FK roundoff is observed, not silently made into law padding.
                tolerance = 2e-12 if dtype == wp.float64 else 2e-6
                test.assertLessEqual(
                    float(np.max(np.linalg.norm(independent - initial, axis=2) - bound[:, None])), tolerance
                )
                q1.assign(coordinates)
                k.evaluate(q1)
                pose = k.poses.numpy().astype(float)
                native = pose[:, None, :3] + np.einsum("bij,bpj->bpi", k.rotations.numpy().astype(float), points)
                np.testing.assert_allclose(native, independent, atol=tolerance, rtol=tolerance)
                test.assertLessEqual(
                    float(np.max(np.linalg.norm(native - initial, axis=2) - bound[:, None])), tolerance
                )
            q1.assign(final)
        test.assertLess(float(starts[0].max()), 1e-280)
        test.assertGreater(float(starts[-1][7]), 0.5)
        # A whole revolution returns near its endpoint but must retain its swept path.
        q1.assign(q0_host + np.array([0, 0, 0, 0, 0, 2 * np.pi]))
        sweep.update(q0, q1)
        test.assertGreater(float(sweep.displacement.numpy()[7]), 0.5)


def test_sweep_storage_and_errors(test, device):
    """Reject invalid geometry and native inputs, keep fixed scratch and clear hard failures."""
    from newton._src.solvers.monolithic.contact_sweep import _ArticulationSweep  # noqa: PLC0415

    k = sweep_tree(device, wp.float64)
    for invalid in (np.ones(7), np.full(8, -1.0), np.full(8, np.nan), np.full(8, np.inf)):
        with test.assertRaises(ValueError):
            _ArticulationSweep(k, invalid)
    sweep = _ArticulationSweep(k, np.full(8, 0.5))
    q0 = wp.zeros(k.count, dtype=k.dtype, device=device)
    q1 = wp.full(k.count, 0.01, dtype=k.dtype, device=device)
    baseline_ptrs = [a.ptr for a in sweep.buffers()]
    test.assertEqual(sweep.payload_bytes, sum(a.capacity for a in sweep.buffers()))
    sweep.update(q0, q1)
    with patch.object(device.get_allocator(), "allocate", side_effect=AssertionError("runtime allocation")):
        for _ in range(3):
            sweep.update(q0, q1)
    test.assertEqual(baseline_ptrs, [a.ptr for a in sweep.buffers()])
    expected = sweep.displacement.numpy().copy()
    q1.fill_(np.nan)
    sweep.update(q0, q1)
    test.assertNotEqual(int(sweep.status.numpy()[0]), 0)
    np.testing.assert_array_equal(sweep.displacement.numpy(), 0)
    q1.fill_(0.01)
    sweep.update(q0, q1)
    np.testing.assert_array_equal(sweep.displacement.numpy(), expected)
    for invalid in (
        wp.zeros(k.count, dtype=wp.float32, device=device),
        wp.zeros(k.count + 1, dtype=k.dtype, device=device),
    ):
        with test.assertRaises(ValueError):
            sweep.update(q0, invalid)
    aliased = wp.array(ptr=sweep.displacement.ptr, shape=k.count, dtype=k.dtype, device=device, copy=False)
    with test.assertRaisesRegex(ValueError, "aliases"):
        sweep.update(q0, aliased)
    shifted = wp.array(ptr=sweep.displacement.ptr + 8, shape=k.count, dtype=k.dtype, device=device, copy=False)
    with test.assertRaisesRegex(ValueError, "aliases"):
        sweep.update(q0, shifted)
    parents = k.parents.numpy()
    k.parents.assign(np.zeros_like(parents))
    sweep.update(q0, q1)
    test.assertNotEqual(int(sweep.status.numpy()[0]), 0)
    np.testing.assert_array_equal(sweep.displacement.numpy(), 0)
    k.parents.assign(parents)
    q1.fill_(np.finfo(np.float64).max)
    sweep.update(q0, q1)
    test.assertNotEqual(int(sweep.status.numpy()[0]), 0)
    np.testing.assert_array_equal(sweep.displacement.numpy(), 0)
    q1.fill_(0.01)
    original = sweep.radii
    sweep.radii = wp.clone(original)
    with test.assertRaises(ValueError):
        sweep.update(q0, q1)
    sweep.radii = original
    sweep.radii.fill_(-1)
    sweep.update(q0, q1)
    test.assertNotEqual(int(sweep.status.numpy()[0]), 0)
    np.testing.assert_array_equal(sweep.displacement.numpy(), 0)


def test_sweep_graph(test, device):
    """Replay changed endpoints in one captured graph without retaining old motion or failure."""
    if not device.is_cuda:
        test.skipTest("CUDA graph requires CUDA")
    from newton._src.solvers.monolithic.contact_sweep import _ArticulationSweep  # noqa: PLC0415

    k = sweep_tree(device, wp.float64)
    sweep = _ArticulationSweep(k, np.full(8, 0.5))
    q0, q1 = (wp.zeros(k.count, dtype=k.dtype, device=device) for _ in range(2))
    sweep.update(q0, q1)
    with wp.ScopedCapture(device=device) as capture:
        sweep.update(q0, q1)
    for value in (0.01, 2 * np.pi, np.nan, 0.01):
        q1.fill_(value)
        wp.capture_launch(capture.graph)
        actual, status = sweep.displacement.numpy(), sweep.status.numpy()
        sweep.update(q0, q1)
        np.testing.assert_array_equal(actual, sweep.displacement.numpy())
        np.testing.assert_array_equal(status, sweep.status.numpy())


class TestContactSweep(unittest.TestCase):
    def test_outward_lengths(self):
        """Check outward construction norms with exact rational arithmetic, including underflow."""
        from newton._src.solvers.monolithic.contact_sweep import _length_upper  # noqa: PLC0415

        vectors = np.array([[0, 0, 0], [1e-300, 0, 0], [1, 2, 3], [1e150, 1e149, -1e150]])
        for vector, bound in zip(vectors, _length_upper(vectors), strict=True):
            exact_square = sum(Fraction(float(value)) ** 2 for value in vector)
            self.assertGreaterEqual(Fraction(float(bound)) ** 2, exact_square)
        with self.assertRaises(ValueError):
            _length_upper([[1e300, 1, 0]])


for _test in (test_outward_arithmetic, test_sweep_coverage, test_sweep_storage_and_errors, test_sweep_graph):
    add_function_test(TestContactSweep, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
