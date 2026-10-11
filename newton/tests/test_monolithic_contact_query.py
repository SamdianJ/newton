# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check BVH candidates against a small independent allowed-pair enumeration."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact_query import _ContactQuery
from newton.tests.unittest_utils import add_function_test, get_test_devices


def fixture(device, capacity=32):
    """Build mixed tet/shape patches, directed permissions and an unbounded plane."""
    faces = np.array([[-1, 0], [-1, 1], [0, 0], [0, 1], [2, 0]])
    targets = [1, 3, 4]
    allowed = [[-1, 1], [0, 1], [2, 3], [-1, 4], [0, 4], [2, 4], [0, 1]]
    query = _ContactQuery(faces, targets, allowed, planes=[4], capacity=capacity, device=device)
    lower = np.array([[x - 0.25, -0.25, -0.25] for x in (0, 5, 0, 10, 0)])
    upper = lower + 0.5
    target_lo = np.array([[0.25, -0.25, -0.25], [-0.25, -0.25, -0.25], [0, 0, 0]])
    target_hi = target_lo.copy()
    bounds = [wp.array(a, dtype=wp.vec3d, device=device) for a in (lower, upper, target_lo, target_hi)]
    return query, bounds


def enumerate_keys(query, bounds):
    """Use original binary64 boxes and host pair membership, without BVH or its conversion."""
    lo, hi, target_lo, target_hi = [a.numpy() for a in bounds]
    allowed = set(map(tuple, query.allowed))
    keys = []
    for face, (source, _) in enumerate(query.faces):
        for target, shape in enumerate(query.targets):
            overlap = shape in query.planes or (
                np.all(lo[face] <= target_hi[target]) and np.all(target_lo[target] <= hi[face])
            )
            if (source, shape) in allowed and overlap:
                keys.extend((face * len(query.targets) + target) * 3 + slot for slot in range(3))
    return np.asarray(keys, dtype=np.uint64)


def selected(query):
    """Copy only published candidate identities, ignoring stale fixed-capacity tails."""
    return query.keys.numpy()[: int(query.count.numpy()[0])].copy()


def test_allowed_candidates(test, device):
    """Retain tangency and all slots, filter directed pairs, and replay changed bounds deterministically."""
    query, bounds = fixture(device)
    for offset in (0, 5, 0):
        targets = bounds[2].numpy().copy()
        targets[0, 0] = 0.25 + offset
        bounds[2].assign(targets)
        bounds[3].assign(targets)
        query.update(*bounds)
        query.check()
        expected = enumerate_keys(query, bounds)
        np.testing.assert_array_equal(selected(query), expected)
        test.assertEqual(int(query.dropped.numpy()[0]), 0)
    # BVH traversal and permission duplicates cannot create duplicate contacts.
    test.assertEqual(len(selected(query)), len(set(selected(query))))
    plane = _ContactQuery([[0, 0]], [1], [[0, 1]], planes=[1], capacity=3, device=device)
    point = wp.zeros(1, dtype=wp.vec3d, device=device)
    plane.update(point, point, point, point)
    plane.check()
    test.assertIsNone(plane.bvh)
    np.testing.assert_array_equal(selected(plane), [0, 1, 2])


def test_query_capacity(test, device):
    """Drop whole candidate records at zero, partial and exact capacity, without requiring overflow determinism."""
    for capacity in (0, 1, 2, 7, 24):
        query, bounds = fixture(device, capacity)
        query.update(*bounds)
        expected = enumerate_keys(query, bounds)
        keys = selected(query)
        test.assertEqual(len(keys), min(capacity, len(expected)))
        test.assertEqual(int(query.dropped.numpy()[0]), max(0, len(expected) - capacity))
        test.assertTrue(set(keys).issubset(expected))
        test.assertEqual(len(set(keys)), len(keys))
        if capacity < len(expected):
            with test.assertWarns(RuntimeWarning):
                query.check()
        else:
            query.check()


def test_query_bounds_and_storage(test, device):
    """Round binary64 boxes outward, keep fixed scratch, and discard invalid generations before traversal."""
    query, bounds = fixture(device)
    value = np.nextafter(float(np.float32(0.25)), np.inf)
    host = bounds[0].numpy().copy()
    host[0, 0] = value
    bounds[0].assign(host)
    bounds[1].assign(host + 0.5)
    query.update(*bounds)
    test.assertTrue(np.all(query.face_lo.numpy().astype(float) <= bounds[0].numpy()))
    test.assertTrue(np.all(query.face_hi.numpy().astype(float) >= bounds[1].numpy()))
    test.assertTrue(set(enumerate_keys(query, bounds)).issubset(selected(query)))
    pointers = [a.ptr for a in query.buffers()]
    with patch.object(device.get_allocator(), "allocate", side_effect=AssertionError("runtime allocation")):
        query.update(*bounds)
    test.assertEqual(pointers, [a.ptr for a in query.buffers()])
    good = bounds[0].numpy().copy()
    for invalid in (np.nan, np.inf, np.finfo(np.float64).max, 99):
        host = good.copy()
        host[0, 0] = invalid
        bounds[0].assign(host)
        query.update(*bounds)
        with test.assertRaises(ValueError):
            query.check()
        test.assertEqual(int(query.count.numpy()[0]), 0)
        test.assertEqual(int(query.dropped.numpy()[0]), 0)
    bounds[0].assign(good)
    query.update(*bounds)
    query.check()
    with test.assertRaises(ValueError):
        query.update(wp.zeros(len(query.faces) + 1, dtype=wp.vec3d, device=device), *bounds[1:])
    with test.assertRaises(ValueError):
        query.update(query.face_lo, *bounds[1:])


def test_query_graph(test, device):
    """Refit and query changing boxes in one graph, including hard failure and recovery."""
    if not device.is_cuda:
        test.skipTest("CUDA graph requires CUDA")
    query, bounds = fixture(device)
    query.update(*bounds)
    with wp.ScopedCapture(device=device) as capture:
        query.update(*bounds)
    original = bounds[2].numpy().copy()
    for offset in (5, np.nan, 0):
        host = original.copy()
        host[0, 0] += offset
        bounds[2].assign(host)
        bounds[3].assign(host)
        wp.capture_launch(capture.graph)
        if np.isnan(offset):
            with test.assertRaises(ValueError):
                query.check()
            test.assertEqual(int(query.count.numpy()[0]), 0)
        else:
            query.check()
            np.testing.assert_array_equal(selected(query), enumerate_keys(query, bounds))


def test_query_rounding(test, device):
    """Enclose original binary64 point boxes at signed, subnormal and large float32 rounding boundaries."""
    query = _ContactQuery([[0, 0]], [1], [[0, 1]], capacity=3, device=device)
    point = wp.zeros(1, dtype=wp.vec3d, device=device)
    values = [0, 1e-300, np.nextafter(0.25, np.inf), np.nextafter(0.25, -np.inf), 1000000001.0]
    for sign in (-1, 1):
        for value in values:
            point.assign([[sign * value] * 3])
            query.update(point, point, point, point)
            query.check()
            test.assertTrue(np.all(query.face_lo.numpy().astype(float) <= point.numpy()))
            test.assertTrue(np.all(query.face_hi.numpy().astype(float) >= point.numpy()))
            np.testing.assert_array_equal(selected(query), [0, 1, 2])
    single = wp.zeros(1, dtype=wp.vec3, device=device)
    query.update(single, single, single, single)
    query.check()
    np.testing.assert_array_equal(selected(query), [0, 1, 2])


class TestContactQuery(unittest.TestCase):
    def test_static_metadata_and_empty(self):
        """Reject ambiguous face identities and malformed permissions, while supporting empty models."""
        for faces, targets, allowed in (
            ([[0, 0], [0, 0]], [1], [[0, 1]]),
            ([[0, -1]], [1], [[0, 1]]),
            ([[0.5, 0]], [1], [[0, 1]]),
            ([[0, 0]], [1], [[2, 1]]),
            ([[0, 0]], [1], [[0, 2]]),
        ):
            with self.assertRaises(ValueError):
                _ContactQuery(faces, targets, allowed, capacity=4, device="cpu")
        query = _ContactQuery([], [], [], capacity=0, device="cpu")
        empty = wp.zeros(0, dtype=wp.vec3d, device="cpu")
        query.update(empty, empty, empty, empty)
        query.check()
        self.assertEqual(int(query.count.numpy()[0]), 0)


for _test in (
    test_allowed_candidates,
    test_query_capacity,
    test_query_bounds_and_storage,
    test_query_graph,
    test_query_rounding,
):
    add_function_test(TestContactQuery, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
