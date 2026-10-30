# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that nonlinear convergence is judged per row in physical units."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.monolithic import convergence
from newton._src.solvers.monolithic.convergence import ConvergencePolicy, Criterion, Unit
from newton._src.solvers.monolithic.linear_types import MonolithicLinearLayout
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.tet_benchmarks import TetBenchmark


def mixed_layout():
    """Articulation (3 DoF), one rigid, a 2-particle tet and a 2-vertex cable with one free angle."""
    return MonolithicLinearLayout(
        3,
        2,
        0,
        9,
        "mixed",
        (("articulation", 0, 3), ("rigid", 3, 6), ("tet", 9, 6), ("cable", 15, 7)),
        ((4, 3),),
        (("cable", 0, 15, 6, 1),),
    )


def final_residual(solver):
    """Return the published step's physical residual [N, N m] as a NumPy array."""
    residual = solver.last_stats.variational["residual"]
    return residual.numpy() if hasattr(residual, "numpy") else np.asarray(residual, dtype=float)


def test_local_load_is_not_diluted(test, device):
    """Resolve a small tip load on a 2x2-section beam instead of accepting the inertial prediction."""
    case = TetBenchmark(
        "cantilever",
        cells=(20, 2, 2),
        device=device,
        pcg_mode="production",
        use_cuda_graph=device.is_cuda,
    )
    # A dynamic step keeps the operator well conditioned, isolating the
    # convergence criterion from the slender beam's quasi-static PCG cost.
    case.dt = 1e-3
    case.solver.reset_state(case.state, dt=case.dt)
    case.step()
    load = float(np.abs(case.state.particle_f.numpy()).max())
    residual = float(np.abs(final_residual(case.solver)).max())
    # The tip load is a few 1e-5 N per node; an unresolved step leaves a residual of that size.
    test.assertLess(residual, 1e-3 * load)
    test.assertGreater(case.solver.last_stats.nonlinear_iterations, 0)
    test.assertLess(case.history[-1][2], 0.0)


def test_criterion_matches_oracle(test, device):
    """Compute identical row tolerances and group assessments on device and in NumPy."""
    rows = convergence.row_groups(mixed_layout(), [Unit.TORQUE, Unit.FORCE, Unit.TORQUE])
    policy = ConvergencePolicy(
        absolute=(1e-9, 1e-10), relative=(1e-6, 1e-5), noise=(32.0, 16.0), relative_fp32=(1e-4, 1e-3)
    )
    rng = np.random.default_rng(7)
    n = rows.units.size
    for dtype, native in ((wp.float64, np.float64), (wp.float32, np.float32)):
        with test.subTest(dtype=native.__name__):
            reference = (rng.standard_normal(n) * 10.0 ** rng.uniform(-12, 2, n)).astype(native)
            magnitude = (10.0 ** rng.uniform(-3, 7, n)).astype(native)
            residual = (rng.standard_normal(n) * 10.0 ** rng.uniform(-14, -2, n)).astype(native)
            tau, physical = convergence.tolerances(reference, magnitude, rows, policy, native)
            worst, limited = convergence.assess(residual, tau, physical, rows)
            criterion = Criterion(rows, policy, dtype=dtype, device=device)
            criterion.freeze_reference(wp.array(reference, dtype=dtype, device=device))
            criterion.update(wp.array(magnitude, dtype=dtype, device=device))
            criterion.assess(wp.array(residual, dtype=dtype, device=device))
            np.testing.assert_array_equal(criterion.tau.numpy(), tau)
            np.testing.assert_array_equal(criterion.physical.numpy(), physical)
            np.testing.assert_array_equal(criterion.worst.numpy(), worst)
            np.testing.assert_array_equal(criterion.limited.numpy(), limited)
            expected = convergence.merit(residual, tau)
            actual = np.sqrt(criterion.square_sum.numpy()[0] / n)
            np.testing.assert_allclose(actual, expected, rtol=8 * np.finfo(native).eps * n)


def test_nonfinite_tolerance_never_converges(test, device):
    """Refuse convergence on a row whose rounding magnitude is not finite, in both precisions.

    An infinite magnitude would make |r| / tau zero and a NaN one would drop out of
    the maximum, so the row could pass whatever its residual. The oracle and the
    kernel both give such a row an infinite tau and an infinite ratio.
    """
    rows = convergence.RowGroups(("x",), np.zeros(3, np.int32), np.zeros(3, np.int32))
    policy = ConvergencePolicy()
    for dtype, native in ((wp.float64, np.float64), (wp.float32, np.float32)):
        for bad in (np.inf, np.nan):
            with test.subTest(dtype=native.__name__, magnitude=bad):
                reference = np.ones(3, dtype=native)
                magnitude = np.array([0.0, bad, 0.0], dtype=native)
                residual = np.array([0.0, 1.0e3, 0.0], dtype=native)
                tau, physical = convergence.tolerances(reference, magnitude, rows, policy, native)
                worst, _ = convergence.assess(residual, tau, physical, rows)
                test.assertTrue(np.isposinf(tau[1]))
                test.assertTrue(np.isinf(worst[0]))
                criterion = Criterion(rows, policy, dtype=dtype, device=device)
                criterion.freeze_reference(wp.array(reference, dtype=dtype, device=device))
                criterion.update(wp.array(magnitude, dtype=dtype, device=device))
                criterion.assess(wp.array(residual, dtype=dtype, device=device))
                test.assertTrue(np.isposinf(criterion.tau.numpy()[1]))
                test.assertTrue(np.isinf(criterion.worst.numpy()[0]))


def test_nonfinite_magnitude_fails_the_step(test, device):
    """End a step whose rounding magnitude overflows with a NONFINITE status, not an exception."""
    from scripts.monolithic_reference.run_precision_checks import _moving_tet  # noqa: PLC0415

    for precision in ("fp64", "fp32"):
        with test.subTest(precision=precision):
            model, solver = _moving_tet(0.0, 9.81, precision, device)
            criterion = solver._criterion
            original = criterion.round

            def overflowing(*args, criterion=criterion, original=original, **kwargs):
                original(*args, **kwargs)
                magnitude = criterion.magnitude.numpy()
                magnitude[0] = np.inf
                criterion.magnitude.assign(magnitude)
                criterion.update(criterion.magnitude)

            criterion.round = overflowing
            state, control = solver.create_state(dt=1e-3, source=model.state()), model.control()
            solver.step(state, state, control, dt=1e-3)
            test.assertEqual(solver.last_stats.status, solver.Status.NONFINITE)
            test.assertTrue(solver.last_stats.rolled_back)


def test_moving_body_resolves_its_load(test, device):
    """Resolve a small load on a uniformly moving body instead of stopping at the predictor.

    The step-start residual carries the inertia M v / h that the predictor
    cancels; as a reference for the relative tolerance it would loosen the
    whole block with the speed, so a fast body converged without moving.
    """
    from scripts.monolithic_reference.run_precision_checks import _moving_tet  # noqa: PLC0415

    dt = 1e-3
    for precision in ("fp64", "fp32"):
        for g, vx in ((0.05, 1.0), (0.05, 100.0), (9.81, 100.0)):
            with test.subTest(precision=precision, g=g, vx=vx):
                model, solver = _moving_tet(vx, g, precision, device)
                state = solver.create_state(dt=dt)
                solver.step(state, state, model.control(), dt=dt)
                test.assertTrue(solver.last_stats.converged, solver.last_stats.status)
                test.assertGreaterEqual(solver.last_stats.nonlinear_iterations, 1)
                vz = state.particle_qd.numpy()[:, 2].astype(float).mean()
                test.assertLess(abs(vz / (-g * dt) - 1), 1e-2)


class TestMonolithicConvergence(unittest.TestCase):
    def test_default_policy(self):
        """Keep the approved per-unit targets, relative tolerances and rounding multipliers."""
        policy = ConvergencePolicy()
        self.assertEqual(policy.absolute, (1e-11, 1e-13))
        self.assertEqual(policy.relative, (1e-6, 1e-6))
        self.assertEqual(policy.relative_fp32, (1e-4, 1e-4))
        self.assertEqual(policy.noise, (32.0, 32.0))
        self.assertEqual(policy.relative_for(np.float64), policy.relative)
        self.assertEqual(policy.relative_for(np.float32), policy.relative_fp32)

    def test_row_groups_split_units(self):
        """Separate revolute and prismatic rows, rigid translation/rotation and cable position/angle."""
        rows = convergence.row_groups(mixed_layout(), [Unit.TORQUE, Unit.FORCE, Unit.TORQUE])
        self.assertEqual(
            rows.names, ("q_force", "q_torque", "rigid_4_t", "rigid_4_r", "x", "cable_0_x", "cable_0_theta")
        )
        np.testing.assert_array_equal(rows.groups, [1, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6])
        np.testing.assert_array_equal(rows.units, [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "Joint units"):
            convergence.row_groups(mixed_layout(), [Unit.FORCE])

    def test_tolerance_terms(self):
        """Take the largest of the absolute, relative and rounding terms per row."""
        rows = convergence.RowGroups(("a", "b"), np.array([0, 0, 1], np.int32), np.array([0, 0, 1], np.int32))
        policy = ConvergencePolicy(absolute=(1e-9, 1e-10), relative=(1e-6, 1e-6), noise=(1.0, 1.0))
        eps = np.finfo(np.float64).eps
        tau, physical = convergence.tolerances([1e-2, -3e-2, 0.0], [0.0, 1e9 / eps, 0.0], rows, policy, np.float64)
        np.testing.assert_allclose(tau, [3e-8, 1e9, 1e-10], rtol=1e-15)
        np.testing.assert_allclose(physical, [3e-8, 3e-8, 1e-10], rtol=1e-15)
        worst, limited = convergence.assess(np.array([3e-8, 2e-3, 0.0]), tau, physical, rows)
        np.testing.assert_allclose(worst, [1.0, 0.0], atol=1e-15)
        np.testing.assert_array_equal(limited, [1, 0])
        # FP32 takes its own relative tolerance.
        policy = ConvergencePolicy(
            absolute=(1e-9, 1e-10), relative=(1e-6, 1e-6), noise=(1.0, 1.0), relative_fp32=(1e-4, 1e-4)
        )
        _, physical = convergence.tolerances([1e-2, -3e-2, 0.0], [0.0, 0.0, 0.0], rows, policy, np.float32)
        np.testing.assert_allclose(physical, [3e-6, 3e-6, 1e-10], rtol=1e-6)

    def test_policy_validation(self):
        """Reject missing, nonpositive or nonfinite policy values."""
        for values in ((1e-9,), (0.0, 1.0), (1.0, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ConvergencePolicy(absolute=values)


add_function_test(
    TestMonolithicConvergence,
    "test_criterion_matches_oracle",
    test_criterion_matches_oracle,
    devices=get_test_devices(),
)
add_function_test(
    TestMonolithicConvergence,
    "test_nonfinite_tolerance_never_converges",
    test_nonfinite_tolerance_never_converges,
    devices=get_test_devices(),
)
add_function_test(
    TestMonolithicConvergence,
    "test_nonfinite_magnitude_fails_the_step",
    test_nonfinite_magnitude_fails_the_step,
    devices=get_test_devices(),
)
add_function_test(
    TestMonolithicConvergence,
    "test_moving_body_resolves_its_load",
    test_moving_body_resolves_its_load,
    devices=get_test_devices(),
)
add_function_test(
    TestMonolithicConvergence,
    "test_local_load_is_not_diluted",
    test_local_load_is_not_diluted,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
