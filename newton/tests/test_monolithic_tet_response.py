# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify demonstration input, peak measurement and state isolation."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

import newton.viewer
from newton.examples.softbody.example_monolithic_tet_response import Example
from newton.examples.softbody.monolithic_tet_compare import FIXTURE
from newton.examples.softbody.monolithic_tet_response import (
    ResponseCase,
    gravity_acceleration,
    traction,
    vibration_peaks,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.run_gravity_resolution import comparison, performance


def test_response_fixture(test, device):
    """Keep frozen G2H inputs and physical states independent of display scaling."""
    frozen = copy.deepcopy(FIXTURE)
    example = Example(
        newton.viewer.ViewerNull(),
        SimpleNamespace(device=device, experiment="mass", display_scale=1, output=None),
    )
    example.step()
    states = [(c.state.particle_q.numpy().copy(), c.state.particle_qd.numpy().copy()) for c in example.cases]
    for scale in (1, 5):
        example.display_scale = scale
        example.render()
        test.assertFalse(example.cases[0].summary()["demo_verified"])
        np.testing.assert_array_equal(example.cases[0].state.particle_f.numpy(), 0)
        for case, (x, v) in zip(example.cases, states, strict=True):
            np.testing.assert_array_equal(case.state.particle_q.numpy(), x)
            np.testing.assert_array_equal(case.state.particle_qd.numpy(), v)
    test.assertEqual(FIXTURE, frozen)
    with test.assertRaises(ValueError):
        ResponseCase(device, experiment="unknown", variant=0)
    with test.assertRaises(ValueError):
        ResponseCase(device, experiment="mass", variant=2)


class TestTetResponse(unittest.TestCase):
    def test_gravity_schedule(self):
        """Check smooth activation followed by sustained full gravity."""
        for t, expected in ((-1, 0), (0, 0), (1, 4.905), (2, 9.81), (24, 9.81)):
            self.assertAlmostEqual(gravity_acceleration(t), expected)

    def test_traction(self):
        """Check finite support, loading/hold/unloading and smooth transitions."""
        for t, force in ((-1, 0), (0, 0), (2, 150), (4, 300), (8, 300), (10, 150), (12, 0), (14, 0)):
            self.assertEqual(traction(t), force)
        for t in (0, 4, 8, 12):
            self.assertLess(abs(traction(t + 1e-5) - traction(t - 1e-5)), 1e-7)

    def test_period(self):
        """Recover a known damped sinusoid's period and reject zero response peaks."""
        records = [
            {"time": float(t), "tip_m": float(0.04 * np.exp(-0.02 * t) * np.cos(2 * np.pi * t / 1.43))}
            for t in np.arange(0, 12, 0.01)
        ]
        peaks = vibration_peaks(records)
        self.assertGreaterEqual(len(peaks), 7)
        self.assertAlmostEqual(float(np.mean(np.diff([p[0] for p in peaks]))), 1.43, places=4)
        self.assertEqual(vibration_peaks([{"time": i, "tip_m": 0} for i in range(10)]), [])


add_function_test(TestTetResponse, "test_response_fixture", test_response_fixture, devices=get_test_devices())


def test_gravity_resolution(test, device):
    """Keep density, material and body load invariant while refining the mesh."""
    cases = [ResponseCase(device, experiment="gravity", variant=i) for i in range(3)]
    test.assertEqual([c.model.tet_count for c in cases], [15, 120, 405])
    test.assertEqual([c.model.particle_count for c in cases], [16, 63, 160])
    for case in cases:
        test.assertAlmostEqual(case.total_mass, 1.08, places=6)
        np.testing.assert_allclose(case.rest.min(axis=0), [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(case.rest.max(axis=0), [1.2, 0.3, 0.3], atol=1e-7)
        np.testing.assert_array_equal(case.materials[0], cases[0].materials[0])
        test.assertEqual(case.manifest["density_kg_m3"], 10)
        test.assertGreater(case.linear_static_tip, 0)
        test.assertFalse(case.summary()["demo_verified"])
        case.step()
        np.testing.assert_allclose(case.model.gravity.numpy(), [[0, 0, -gravity_acceleration(case.dt)]], rtol=1e-6)
        np.testing.assert_array_equal(case.state.particle_f.numpy(), 0)
        # Merely having a long time axis cannot certify an oscillating tail.
        entry = dict(case.records[-1], time=36, tip_m=0.2, max_speed_m_s=0.2)
        case.records = [case.records[0], entry]
        case.steps = 1800
        test.assertFalse(case.summary()["near_static"])
        test.assertFalse(case.summary()["demo_verified"])
    test.assertEqual(len({c.manifest["topology_sha256"] for c in cases}), 3)


add_function_test(TestTetResponse, "test_gravity_resolution", test_gravity_resolution, devices=get_test_devices())


def test_extended_gravity(test, device):
    """Exercise refined geometry and synchronized timing without changing physical inputs."""
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=4)
    test.assertEqual(case.model.tet_count, 960)
    test.assertEqual(case.model.particle_count, 325)
    test.assertAlmostEqual(case.total_mass, 1.08, places=6)
    test.assertEqual(case.label, "r4")
    timing = case.step(profile=True)
    test.assertGreater(timing["solver_ms"], 0)
    test.assertGreaterEqual(timing["step_ms"], timing["solver_ms"] + timing["measurement_ms"])
    test.assertEqual(timing["linear_iterations"], case.solver.last_stats.linear_iterations)
    test.assertEqual(case.steps, 1)
    test.assertEqual(case.records[-1]["time"], 0.02)
    for refinement in (0, 9, 2.5, True):
        with test.assertRaises(ValueError):
            ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)


add_function_test(TestTetResponse, "test_extended_gravity", test_extended_gravity, devices=get_test_devices())


class TestGravityAnalysis(unittest.TestCase):
    def test_invalid_run_not_used_as_reference(self):
        """Exclude failed fine meshes and avoid comparing across a missing mesh result."""
        rows = [
            {"refinement": i, "verified": valid, "tip_mean_m": tip}
            for i, valid, tip in ((1, True, 0.1), (2, False, None), (3, True, 0.2), (4, False, None))
        ]
        result = comparison(rows)
        self.assertFalse(result["all_verified"])
        self.assertEqual(result["reference_refinement"], 3)
        self.assertEqual(rows[0]["relative_difference_to_finest"], 0.5)
        self.assertIsNone(rows[2]["relative_change_from_previous"])
        self.assertIsNone(rows[3]["relative_difference_to_finest"])

    def test_performance_units(self):
        """Compute throughput from arithmetic wall time rather than median step latency."""
        records = [
            {"solver_ms": t, "step_ms": t + 2, "measurement_ms": 1, "nonlinear_iterations": 1, "linear_iterations": 10}
            for t in (2, 2, 8)
        ]
        stats = performance(records, 0.02)
        self.assertEqual(stats["solver_ms"]["mean"], 4)
        self.assertEqual(stats["solver_real_time_factor"], 5)
        self.assertAlmostEqual(stats["measured_step_real_time_factor"], 20 / 6)


if __name__ == "__main__":
    unittest.main()
