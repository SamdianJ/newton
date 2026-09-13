# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Keep smart restart independent of optional observations."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear import MonolithicPcgWarmStart
from newton.tests import test_solver_monolithic_linear as fixtures
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_without_diagnostics(test, device):
    """Preserve restart decisions and solutions after removing optional buffers."""
    results = []
    for observe in (True, False):
        workspace, generation = fixtures._ready_system(device, heterogeneous=True)
        if not observe:
            for key in list(vars(workspace)):
                if key.startswith("_diagnostic_"):
                    delattr(workspace, key)
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
    test.assertTrue(np.isnan(b.initial_guess_norm))


class TestMonolithicPcgRestart(unittest.TestCase):
    pass


add_function_test(
    TestMonolithicPcgRestart, "test_without_diagnostics", test_without_diagnostics, devices=get_test_devices()
)
