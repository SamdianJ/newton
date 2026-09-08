# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that detailed BSR instrumentation preserves solver behavior."""

import gc
import unittest
import weakref

import numpy as np
import warp as wp
from warp import sparse

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.normal_loading import build_scene, load_fixture
from scripts.monolithic_reference.profile_sharpa_bsr import _AssemblyProfile, _count_nonzero, _matrix_digest


def test_profile_parity(test, device):
    """Preserve returned states and matrices while observing both native builders."""
    results = []
    for instrumented in (False, True):
        _model, state, control, solver = build_scene(load_fixture(), device=device)
        if instrumented:
            with _AssemblyProfile(solver._linear) as timer:
                solver.step(state, state, control, None, 0.001)
            summary = timer.summary()
            for name in ("global_scalar", "internal_block3"):
                test.assertGreater(summary[f"{name}/_bsr_set_from_triplets_native"]["calls"], 0)
            test.assertGreater(summary["host_readback"]["calls"], 0)
        else:
            solver.step(state, state, control, None, 0.001)
        test.assertFalse(solver.last_stats.rolled_back)
        results.append(
            (state.joint_q.numpy(), state.particle_q.numpy(), _matrix_digest(solver._linear.k_global_scalar_bsr))
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    test.assertEqual(results[0][2], results[1][2])


def test_profile_exception_cleanup(test, device):
    """Release temporary launch arguments and restore patches after an exception."""
    _model, _state, _control, solver = build_scene(load_fixture(), device=device)
    originals = (wp.launch, wp.empty, wp.array.numpy, sparse.bsr_set_from_triplets, solver._linear.finalize_assembly)
    with test.assertRaisesRegex(RuntimeError, "injected"):
        with _AssemblyProfile(solver._linear) as timer:

            def fail():
                scratch = wp.ones(32, dtype=float, device=device)
                count = wp.zeros(1, dtype=int, device=device)
                reference = weakref.ref(scratch)
                wp.launch(_count_nonzero, 32, [scratch, count], device=device)
                test.assertEqual(int(count.numpy()[0]), 32)
                del scratch
                gc.collect()
                test.assertIsNone(reference(), "Instrumentation retained a temporary device array")
                raise RuntimeError("injected")

            timer.timed("test_scope", fail)()
    test.assertEqual(
        originals, (wp.launch, wp.empty, wp.array.numpy, sparse.bsr_set_from_triplets, solver._linear.finalize_assembly)
    )
    test.assertEqual(timer.frames, [])


class TestMonolithicBsrProfile(unittest.TestCase):
    pass


for device in get_test_devices():
    add_function_test(TestMonolithicBsrProfile, "test_profile_parity", test_profile_parity, devices=[device])
    add_function_test(
        TestMonolithicBsrProfile, "test_profile_exception_cleanup", test_profile_exception_cleanup, devices=[device]
    )


if __name__ == "__main__":
    unittest.main()
