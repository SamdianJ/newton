# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regress P2 measurement units, wrapper lifetime and loaded replay."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import p2_measurement as measurement
from scripts.monolithic_reference import run_p2_sharpa_study as sharpa
from scripts.monolithic_reference import run_p2_tet_study as tet
from scripts.monolithic_reference.p2_measurement import evidence_index, frame_samples, throughput


class TestP2Measurement(unittest.TestCase):
    def test_tet_phase_boundaries(self):
        for time_s, phase in (
            (0, "loading"),
            (2, "loading"),
            (2.02, "hold"),
            (33.98, "hold"),
            (34, "tail"),
            (36, "tail"),
        ):
            self.assertEqual(tet.phase_of(time_s), phase)

    def test_matrix_capture_uses_actual_call_in_window(self):
        case = SimpleNamespace(
            steps=0,
            dt=0.02,
            manifest={},
            solver=SimpleNamespace(_linear=SimpleNamespace(solve_pcg=lambda: None)),
        )

        def step(case):
            if case.steps + 1 in (50, 154, 1703):
                case.solver._linear.solve_pcg()
            case.steps += 1
            return {"rollback": False, "time": case.steps * case.dt}

        for steps, expected in ((1750, {"loading": 1.0, "hold": 3.08, "tail": 34.06}), (100, {"loading": 1.0})):
            case.steps = 0
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(tet, "ResponseCase", return_value=case),
                patch.object(tet, "timing_step", side_effect=step),
                patch.object(tet, "export_candidate", side_effect=lambda *a, **kw: kw["time_s"]),
            ):
                saved = tet.matrix_run("cpu", 1, Path(directory), steps=steps)
                self.assertEqual(saved, expected)
                sampling = json.loads((Path(directory) / "sampling.json").read_text())
                self.assertEqual(sampling["windows"]["loading"]["status"], "CAPTURED")
                self.assertEqual(
                    sampling["windows"]["tail"]["status"], "CAPTURED" if steps == 1750 else "NOT_REQUESTED"
                )

    def test_common_frames_excludes_rollback(self):
        rows = [
            {
                "time": t,
                "stage": "hold",
                "ball_com": [0, 0, 0],
                "deformation_rms": 0,
                "finger_force_n": 1,
                "penetration": p,
                "min_det_f": 1,
                "rollback": failed,
            }
            for t, p, failed in ((0.01, 0.001, False), (0.02, 0.1, True))
        ]
        result = sharpa.common_frames(rows, 0.01)
        self.assertEqual([row["time"] for row in result["samples"]], [0.01])
        self.assertEqual(result["per_step_peaks"]["penetration"], 0.001)

    def test_zero_accepted_time_has_no_per_simulated_rates(self):
        row = {
            "stage": "hold",
            "time": 0.01,
            "step_seconds": 0.002,
            "nonlinear_iterations": 1,
            "linear_iterations": 2,
            "matrix_assembly_count": 2,
            "pcg_iterations_sum": 2,
            "status": "FAILED",
            "rollback": True,
        }
        result = sharpa.work_stats([row], 0.01)
        self.assertIsNone(result["solver_wall_seconds_per_simulated_second"])
        self.assertTrue(all(value is None for value in result["per_simulated_second"].values()))

    def test_nested_time_accounting(self):
        """Partition enclosing time exactly and retain samples when a child fails."""
        solver = SimpleNamespace(model=SimpleNamespace(device="cpu"), _linear=SimpleNamespace(preconditioner=None))
        profile = measurement.P2StageProfile(solver, "actor_block")

        def fail():
            raise RuntimeError("timed failure")

        child = profile._timed("pcg", fail)
        parent = profile._timed("solver_step", child)
        with (
            patch.object(measurement.wp, "synchronize_device"),
            patch.object(measurement.time, "perf_counter", side_effect=[0, 1, 3, 5]),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed failure"):
                parent()
        report = profile.summary()
        self.assertEqual(report["solver_step"]["inclusive_total_ms"], 5000)
        self.assertEqual(report["solver_step"]["exclusive_total_ms"], 3000)
        self.assertEqual(report["pcg"]["exclusive_total_ms"], 2000)
        self.assertEqual(profile.active, [])

    def test_frame_sum_and_failed_coverage(self):
        """Aggregate real frames and count failed attempts only in elapsed wall time."""
        rows = [{"time": i, "solver_seconds": v} for i, v in enumerate([1, 9, 9, 1, 5])]
        self.assertEqual([r["solver_seconds"] for r in frame_samples(rows, 2, "solver_seconds")], [10, 10])
        result = throughput(25, 4, 0.001, 2)
        self.assertEqual(result["completed_output_frames"], 2)
        self.assertAlmostEqual(result["solver_only_realtime_factor"], 0.004 / 25)
        self.assertIsNone(throughput(1, 0, 0.001, 2)["solver_wall_seconds_per_simulated_second"])

    def test_evidence_index_includes_npz(self):
        """Hash binary artifacts while excluding the index and digest sidecars."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            np.savez(output / "state.npz", x=[1])
            (output / "state.npz.sha256").write_text("old digest")
            first = evidence_index(output)
            self.assertEqual(set(first), {"state.npz"})
            self.assertEqual(evidence_index(output), first)

    def test_material_candidate_does_not_evolve(self):
        """Evaluate one loaded candidate without advancing a material trajectory."""
        case = tet.ResponseCase("cpu", experiment="gravity", variant=0, refinement=1)
        for _ in range(5):
            case.step()
        inputs = case.state.particle_q.numpy().copy()
        captured = {
            "candidate_x": inputs,
            "particle_q_n": case.solver._transaction._original.particle_q.numpy(),
            "particle_qd_n": case.solver._transaction._original.particle_qd.numpy(),
            "frozen_particle_f": case.state.particle_f.numpy(),
            "gravity": case.model.gravity.numpy(),
        }
        observed = {}
        allocations = []
        original = tet.assemble_tet_residual_tangent

        def record(model, **kwargs):
            observed.setdefault(model, []).append(kwargs["candidate_particle_q"].numpy().copy())
            allocator = model.device.get_allocator()
            allocate = allocator.allocate

            def tracked_allocate(size):
                allocations.append(size)
                return allocate(size)

            with patch.object(allocator, "allocate", new=tracked_allocate):
                return original(model, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(tet, "assemble_tet_residual_tangent", new=record):
            tet.material_replay("cpu", 1, captured, Path(directory))
        self.assertEqual(len(observed), 3)
        self.assertEqual(allocations, [])
        for calls in observed.values():
            self.assertGreaterEqual(len(calls), 5)
            for x in calls:
                np.testing.assert_array_equal(x, inputs)

    def test_reject_positions_only_replay(self):
        """Require the complete loaded tet input instead of silently inventing velocities or loads."""
        with self.assertRaisesRegex(ValueError, "loaded inputs"):
            tet.material_replay("cpu", 1, np.zeros((16, 3)), Path("unused"))

    def test_fps_units(self):
        """Distinguish output frames per second from simulated seconds per second."""
        rows = [
            {
                "stage": "hold",
                "time": (i + 1) * 0.001,
                "step_seconds": 0.002,
                "nonlinear_iterations": 1,
                "linear_iterations": 2,
                "matrix_assembly_count": 2,
                "pcg_iterations_sum": 2,
                "status": "SUCCESS",
                "rollback": False,
            }
            for i in range(20)
        ]
        result = sharpa.work_stats(rows, 0.001)
        self.assertAlmostEqual(result["solver_only_fps_at_frame_dt"], 50.0)

    def test_five_independent_sharpa_repeats(self):
        """Schedule at least five fresh baseline trajectories for each mesh."""
        for mesh in ("r2", "r3"):
            names = {j["name"] for j in sharpa.jobs() if j["task"] == "0b-baseline" and j["mesh"] == mesh}
            self.assertGreaterEqual(len(names), 5)

    def test_preserve_actual_pcg_calls(self):
        """Keep retries and zero-call steps distinguishable in the saved trace."""
        stats = SimpleNamespace(
            **dict.fromkeys(
                (
                    "nonlinear_iterations",
                    "linear_iterations",
                    "line_search_iterations",
                    "regularization_retries",
                    "matrix_assembly_count",
                    "matrix_nnz",
                    "triplet_count",
                    "rho",
                    "rho_q",
                    "rho_x",
                ),
                0,
            )
        )
        for calls in (
            [],
            [
                {"iterations": 2, "status": "SUCCESS", "seconds": 0.01},
                {"iterations": 80, "status": "SUCCESS", "seconds": 0.04},
            ],
        ):
            legacy = {"collision_seconds": 1, "pcg_seconds": 2, "other_step_seconds": 3}
            record = {"pcg_calls": calls, "finger_forces": {}, "time": 0.01, "q": [1], "qd": [2], **legacy}
            row = sharpa.slim_record(record, stats, None)
            self.assertEqual(row.get("pcg_calls"), calls)
            self.assertTrue(set(legacy).isdisjoint(row))
            self.assertEqual(row.get("q"), [1])
            self.assertEqual(row.get("qd"), [2])
            between = sharpa.slim_record({**record, "time": 0.011}, stats, None)
            self.assertNotIn("q", between)
            self.assertNotIn("qd", between)

    def _window(self, fail, capture=False):
        device = SimpleNamespace(is_cuda=False)

        def original():
            return SimpleNamespace(merit=1, merit_q=1, merit_x=1, min_det_f=1)

        solver = SimpleNamespace(
            _evaluate_current=original,
            model=SimpleNamespace(device=device),
            _linear=SimpleNamespace(solve_pcg=lambda: None),
        )
        array = SimpleNamespace(numpy=lambda: np.zeros((1, 3)))
        case = SimpleNamespace(
            solver=solver,
            model=SimpleNamespace(device=device),
            step_count=0,
            failure=None,
            records=[],
            manifest={},
            state=SimpleNamespace(joint_q=array, joint_qd=array, particle_q=array, particle_qd=array),
        )
        counts = (1, 3, 2, 1)

        def solver_step():
            if fail and case.step_count == 2:
                raise RuntimeError("injected step failure")
            count = counts[case.step_count]
            for _ in range(count):
                solver._evaluate_current()
            if capture and case.step_count == 2:
                solver._linear.solve_pcg()
            solver.last_stats = SimpleNamespace(matrix_assembly_count=count)

        solver.step = solver_step

        def step():
            solver.step()
            count = counts[case.step_count]
            case.step_count += 1
            case.records.append(
                {
                    "time": case.step_count * 0.001,
                    "stage": "hold",
                    "converged": True,
                    "status": "SUCCESS",
                    "rollback": False,
                    "finger_force_n": 1,
                    "penetration": 0,
                    "min_det_f": 1,
                    "anchor_unchanged": True,
                    "residual_ratio": 0,
                    "q_residual_ratio": 0,
                    "x_residual_ratio": 0,
                    "pcg_iterations_max": 0,
                    "pcg_calls": [],
                    "nonlinear_iterations": 0,
                    "matrix_assembly_count": count,
                }
            )

        case.step = step
        profiles = []

        class Profile:
            def __init__(self, solver, variant):
                self.solver, self.calls, self.closed = solver, 0, False
                profiles.append(self)

            def __enter__(self):
                self.original = self.solver._evaluate_current

                def wrapped(*args, **kwargs):
                    self.calls += 1
                    return self.original(*args, **kwargs)

                self.solver._evaluate_current = wrapped
                return self

            def __exit__(self, *args):
                self.solver._evaluate_current = self.original
                self.closed = True

            def summary(self):
                return {"current_evaluation_overlapping_total": {"calls": self.calls}}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sharpa, "GraspCase", return_value=case),
            patch.object(sharpa, "_StageProfile", Profile),
            patch.object(sharpa, "DURATION", 0.004),
            patch.object(sharpa, "STAGE_WINDOWS", {"hold": (0, 0.004)}),
            patch.object(sharpa, "CURVE_TIMES", (0.002,)),
            patch.object(sharpa, "MATRIX_WINDOWS", ((0.001, 0.004),), create=True),
            patch.object(sharpa, "export_candidate", return_value={}) as exported,
            patch.object(
                solver._linear,
                "solve_pcg",
                return_value=SimpleNamespace(
                    iterations=1, status=SimpleNamespace(name="SUCCESS"), rho=0, rho_q=0, rho_x=0
                ),
            ),
            patch.object(sharpa, "slim_record", side_effect=lambda row, *_: row),
            patch.object(sharpa, "work_stats", return_value={}),
            patch.object(sharpa, "common_frames", return_value={}),
            patch.object(sharpa, "environment", return_value={}),
            patch.object(sharpa.wp, "synchronize_device"),
        ):
            job = {"name": "test", "task": "test", "mesh": "r3", "physical_dt_s": 0.001, "newton_max_iterations": 10}
            job["capture_matrices"] = capture
            if fail:
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    sharpa.run_case(job, Path(directory), stage_profile=True)
            else:
                sharpa.run_case(job, Path(directory), stage_profile=True)
        self.assertTrue(all(p.closed for p in profiles))
        self.assertIs(solver._evaluate_current, original)
        self.assertEqual(sum(p.calls for p in profiles), sum(counts[:2] if fail else counts))
        if capture:
            self.assertEqual(exported.call_count, 1)
            self.assertEqual(exported.call_args.kwargs["time_s"], 0.003)

    def test_sharpa_matrix_waits_for_actual_pcg(self):
        self._window(False, capture=True)

    def test_current_window_coverage(self):
        """Count all current calls across varying Newton work and a curve step."""
        self._window(False)

    def test_current_window_exception_cleanup(self):
        """Restore every installed profile wrapper when a step raises."""
        self._window(True)


def test_loaded_timing_parity(test, device):
    """Match ResponseCase state and actual gravity through the ramp and into hold."""
    reference = tet.ResponseCase(device, experiment="gravity", variant=0, refinement=1)
    measured = tet.ResponseCase(device, experiment="gravity", variant=0, refinement=1)
    for step in range(105):
        reference.step()
        row = tet.timing_step(measured)
        test.assertFalse(row["rollback"])
        test.assertEqual(reference.steps, measured.steps)
        test.assertEqual(reference.solver.last_stats.status, measured.solver.last_stats.status)
        np.testing.assert_array_equal(reference.model.gravity.numpy(), measured.model.gravity.numpy())
        if step in (0, 49, 99, 104):
            for name in ("particle_q", "particle_qd", "particle_f", "joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(reference.state, name).numpy(),
                    getattr(measured.state, name).numpy(),
                    atol=5e-5 if wp.get_device(device).is_cuda else 1e-5,
                    rtol=1e-5,
                )
    test.assertGreater(np.linalg.norm(measured.state.particle_q.numpy() - measured.rest), 0.01)
    test.assertGreater(sum(r["gravity_m_s2"] for r in reference.records), 0)


def test_loaded_matrix_export(test, device):
    """Export a real scaled solve with finite SPD algebra and a bound loaded candidate."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        saved = tet.matrix_run(device, 1, output, steps=51)
        data = saved["loading"]
        metadata = json.loads((output / "loading.json").read_text())
        test.assertEqual(measurement.array_fingerprint(data), metadata["input_sha256"])
        np.testing.assert_allclose(data["rhs_hat"], -data["S"] * data["R"], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(data["S"], data["D"] ** -0.5, rtol=1e-6)
        test.assertAlmostEqual(float(data["gravity"][0, 2]), -4.905, places=5)
        test.assertEqual(metadata["time_s"], 1.0)
        n = len(data["D"])
        dense = np.zeros((n, n))
        for row in range(n):
            start, end = data["K_offsets"][row : row + 2]
            dense[row, data["K_columns"][start:end]] = data["K_values"][start:end].reshape(-1)
        scaled = data["S"][:, None] * dense * data["S"][None, :] + metadata["lambda"] * np.eye(n)
        np.testing.assert_allclose(scaled, scaled.T, rtol=5e-5, atol=5e-5)
        test.assertGreater(np.linalg.eigvalsh(scaled).min(), 0)


add_function_test(TestP2Measurement, "test_loaded_timing_parity", test_loaded_timing_parity, devices=get_test_devices())
add_function_test(TestP2Measurement, "test_loaded_matrix_export", test_loaded_matrix_export, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
