# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot window timing boundaries and publication qualification."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.monolithic_reference import benchmark_robot_windows as benchmark


class _Window:
    def __init__(self, *, hard=False, stopped=False):
        self.clock, self.starts, self.weight = 0, [], 1.0
        self.hard, self.stopped = hard, stopped
        self.solver = SimpleNamespace(model=SimpleNamespace(device="cpu"), step=self.step)

    def start(self, values, model_values, step):
        self.starts.append(step)
        self.steps = step
        self.clock += 100  # Snapshot restoration is outside the measured step.

    def step(self):
        self.clock += 2
        self.solver.last_stats = SimpleNamespace(
            linear_iterations=3, linear_scheduled_slots=4, line_search_iterations=1, regularization_retries=0
        )

    def advance(self):
        self.clock += 10  # Controller work.
        self.solver.step()
        self.clock += 20  # Offline physical measurements.
        self.steps += 1
        return {
            "step": self.steps,
            "publication": "HARD_FAILURE" if self.hard else "STOPPED_ACCEPTED" if self.stopped else "CONVERGED",
            "newton": 1,
            "min_det_f": 1,
            "penetration": None if self.hard else 0,
            "ground_force_z": None if self.hard else 0,
        }


class TestRobotBenchmark(unittest.TestCase):
    def measure(self, window):
        case = {
            "snapshot": {"step": 100},
            "steps": 3,
            "gates": {"fp32": {"hard_failures_max": 0, "stopped_max": 3, "penetration_max_m": 0.002, "det_f_min": 0.2}},
        }
        with (
            patch.object(benchmark, "check_gpu_idle"),
            patch.object(benchmark, "capture_count", return_value=0),
            patch.object(benchmark.wp, "synchronize_device"),
            patch.object(benchmark.time, "perf_counter", side_effect=lambda: window.clock),
        ):
            return benchmark.measure_window(window, case, {}, {}, "fp32")

    def test_timing_and_restore(self):
        """Exclude setup, controls and recording while restoring the same measured input."""
        window = _Window()
        result = self.measure(window)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(window.starts, [100, 100])
        self.assertEqual([s["phase"] for s in result["samples"]], ["warmup", "measured"])
        for sample in result["samples"]:
            self.assertEqual(sample["total_seconds"], 6)
            self.assertEqual(sample["median_ms"], 2000)
            self.assertEqual([r["step"] for r in sample["rows"]], [101, 102, 103])
            self.assertEqual([r["pcg"] for r in sample["rows"]], [3, 3, 3])

    def test_failure_and_stopped(self):
        """Reject incomplete hard failures and preserve Stopped as a separate count."""
        result = self.measure(_Window(hard=True))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["samples"][1]["counts"], {"HARD_FAILURE": 1})
        result = self.measure(_Window(stopped=True))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["samples"][1]["counts"], {"STOPPED_ACCEPTED": 3})


if __name__ == "__main__":
    unittest.main()
