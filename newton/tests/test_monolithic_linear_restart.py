# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Keep smart restart independent of optional observations."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear_types import MonolithicPcgWarmStart
from newton.tests import test_monolithic_linear_assembly as fixtures
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_observation_does_not_change_solution(test, device):
    """Restart and solve identically whether or not every checkpoint is observed."""
    results = []
    for mode in ("diagnostic", "production"):
        workspace, generation = fixtures._ready_system(device, heterogeneous=True, pcg_mode=mode)
        rhs = wp.array(np.linspace(-1, 2, 11, dtype=np.float32), device=device)
        y = wp.zeros(11, dtype=float, device=device)
        result = workspace.solve_pcg(
            rhs,
            y,
            generation=generation,
            warm_start=MonolithicPcgWarmStart.ZERO,
            config=fixtures._pcg_config(linear_tolerance=1e-5, true_residual_interval=3),
        )
        results.append((result, y.numpy()))
    a, b = [item[0] for item in results]
    test.assertEqual(
        (a.status, a.iterations, a.residual_replacements), (b.status, b.iterations, b.residual_replacements)
    )
    np.testing.assert_array_equal(results[0][1], results[1][1])
    test.assertEqual(len(a.checkpoint_history), a.true_residual_checks)
    test.assertEqual(b.checkpoint_history, ())


class TestMonolithicPcgRestart(unittest.TestCase):
    pass


add_function_test(
    TestMonolithicPcgRestart,
    "test_observation_does_not_change_solution",
    test_observation_does_not_change_solution,
    devices=get_test_devices(),
)
