# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify timing attribution and unchanged production state."""

import unittest

import numpy as np
import warp as wp

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.profile_solver import profile
from scripts.monolithic_reference.profiling import SolverProfiler
from scripts.monolithic_reference.workload_scene import build_scene


def test_spatial_solver_profile(test, device):
    """Observe full reference/query/publication without changing the spatial trajectory."""
    from newton.examples.softbody.monolithic_recording import state_arrays  # noqa: PLC0415
    from scripts.monolithic_reference.workload_scene import compare_arrays  # noqa: PLC0415

    routes = [
        build_scene(
            "tiny-contact",
            device=device,
            graph=device.is_cuda,
            contact_dataflow="spatial",
            spatial={"candidate_capacity": 256},
        )
        for _ in range(2)
    ]
    for _, run, state, _ in routes:
        run.step(state, state, run.model.control(), dt=0.001)
    _, run, state, _ = routes[1]
    with SolverProfiler(run) as profiler:
        run.step(state, state, run.model.control(), dt=0.001)
    _, baseline, expected, _ = routes[0]
    baseline.step(expected, expected, baseline.model.control(), dt=0.001)
    compare_arrays(state_arrays(state), state_arrays(expected))
    modules = profiler.report()["modules"]
    for name in ("contact.refresh", "contact.begin_step", "contact.publish"):
        test.assertGreater(modules[name]["calls"], 0)


def test_contact_pair_diagnostics_switch(test, device):
    """Skip pair reports while preserving native states, forces and solver work."""
    from newton.examples.softbody.monolithic_recording import state_arrays  # noqa: PLC0415
    from scripts.monolithic_reference.profiling import contact_pair_diagnostics  # noqa: PLC0415
    from scripts.monolithic_reference.rigid_stack_scene import DT, stack_scene, stack_solver  # noqa: PLC0415

    for graph in (False, True) if device.is_cuda else (False,):
        model = stack_scene(device, columns=((0.0, 0.0),), layers=2)
        outputs, work = [], []
        for enabled in (True, False):
            run = stack_solver(model, spatial=True, graph=graph)
            state, control = run.create_state(dt=DT), model.control()
            original = run._contact.pair_diagnostics
            counts = []
            with contact_pair_diagnostics(run, enabled=enabled):
                for _ in range(20):
                    run.step(state, state, control, dt=DT)
                    stats = run.last_stats
                    test.assertTrue(stats.converged)
                    counts.append((stats.nonlinear_iterations, stats.linear_iterations, stats.line_search_iterations))
            test.assertEqual(run._contact.pair_diagnostics, original)
            test.assertEqual(bool(stats.variational["actor_contact_pairs"]), enabled)
            outputs.append(state_arrays(state))
            work.append(counts)
            test.assertGreater(np.linalg.norm(outputs[-1]["history.contact_body_force"]), 0)
        test.assertEqual(work[0], work[1])
        for name in outputs[0]:
            np.testing.assert_array_equal(outputs[0][name], outputs[1][name], err_msg=name)
        with test.assertRaisesRegex(RuntimeError, "probe failure"):
            with contact_pair_diagnostics(run, enabled=False):
                raise RuntimeError("probe failure")
        test.assertEqual(run._contact.pair_diagnostics, original)


@wp.kernel
def _increment(value: wp.array[int]):
    value[0] += 1


def test_graph_events(test, device):
    """Keep each repeated and conditional graph launch independently measured."""
    _, run, _, _ = build_scene("tiny-contact", device=device)
    value = wp.zeros(1, dtype=int, device=device)
    condition = wp.ones(1, dtype=int, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device) as capture:
        if wp.is_conditional_graph_supported():
            wp.capture_if(condition, lambda: wp.launch(_increment, 1, [value], device=device))
        else:
            wp.launch(_increment, 1, [value], device=device)
    profiler = SolverProfiler(run)
    for _ in range(5):
        with profiler.scope("graph"):
            wp.capture_launch(capture.graph)
    profiler.collect()
    test.assertEqual(value.numpy()[0], 5)
    test.assertEqual(len(profiler.rows), 5)
    test.assertTrue(all(r["gpu_interval_ms"] >= 0 for r in profiler.rows))
    test.assertTrue(all("events" not in r for r in profiler.rows))


def test_solver_profile(test, device):
    """Preserve active-contact state across three production steps with profiling."""
    for graph in (False, True) if device.is_cuda else (False,):
        result = profile("tiny-contact", device=device, graph=graph, steps=3, activities=device.is_cuda)
        test.assertEqual(result["status"], "PASS")
        report = result["profile"]
        test.assertIn("linear.solve_pcg", report["modules"])
        test.assertGreater(result["baseline_seconds"], 0)
        for metric in ("host_ms", "gpu_interval_ms"):
            fractions = [m[metric]["self_fraction"] for m in report["modules"].values()]
            if device.is_cuda or metric == "host_ms":
                test.assertAlmostEqual(sum(fractions), 1.0)
        test.assertFalse("step" in vars(build_scene("tiny-contact", device=device, graph=graph)[1]))


class TestProfiling(unittest.TestCase):
    def test_contact_pair_diagnostics(self):
        """Suppress host pair reports without changing native state or force publication."""
        from newton.examples.softbody.monolithic_recording import state_arrays  # noqa: PLC0415
        from scripts.monolithic_reference.profiling import contact_pair_diagnostics  # noqa: PLC0415
        from scripts.monolithic_reference.run_actor_coupling import build_model, make_solver  # noqa: PLC0415

        model = build_model(device="cpu")
        outputs = []
        for enabled in (True, False):
            run = make_solver(model, graph=False)
            state, control = run.create_state(dt=0.001), model.control()
            original = run._contact.pair_diagnostics
            with contact_pair_diagnostics(run, enabled=enabled):
                for _ in range(3):
                    run.step(state, state, control, dt=0.001)
                    self.assertTrue(run.last_stats.converged)
            self.assertEqual(run._contact.pair_diagnostics, original)
            self.assertEqual(bool(run.last_stats.variational["actor_contact_pairs"]), enabled)
            outputs.append(state_arrays(state))
        for name in outputs[0]:
            np.testing.assert_array_equal(outputs[0][name], outputs[1][name], err_msg=name)

    def test_nested_accounting(self):
        """Subtract only immediate children and exclude capture-contaminated calls."""
        _, run, _, _ = build_scene("tiny-contact", device="cpu", graph=False)
        profiler = SolverProfiler(run)
        profiler.rows = [
            {"id": 0, "parent": None, "name": "step", "host_ms": 10, "capture_affected": False},
            {"id": 1, "parent": 0, "name": "child", "host_ms": 6, "capture_affected": False},
            {"id": 2, "parent": 1, "name": "leaf", "host_ms": 2, "capture_affected": False},
            {"id": 3, "parent": None, "name": "cold", "host_ms": 100, "capture_affected": True},
        ]
        modules = profiler.report()["modules"]
        self.assertEqual(modules["cold"]["steady_calls"], 0)
        self.assertEqual(modules["cold"]["host_ms"]["steady_inclusive"], 0)
        np.testing.assert_equal([modules[n]["host_ms"]["self"] for n in ("step", "child", "leaf")], [4, 4, 2])


for device in get_test_devices():
    add_function_test(TestProfiling, "test_solver_profile", test_solver_profile, devices=[device])
    add_function_test(TestProfiling, "test_spatial_solver_profile", test_spatial_solver_profile, devices=[device])
    add_function_test(
        TestProfiling, "test_contact_pair_diagnostics_switch", test_contact_pair_diagnostics_switch, devices=[device]
    )
    if device.is_cuda:
        add_function_test(TestProfiling, "test_graph_events", test_graph_events, devices=[device])


if __name__ == "__main__":
    unittest.main()
