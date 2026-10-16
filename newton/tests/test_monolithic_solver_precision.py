# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native coupled arithmetic, immutable precision and checkpoint transactions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.iteration import measure
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, SmoothSDFSet, sample_smooth
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_sdf_fields import grid
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, snapshot, solver
from scripts.monolithic_reference.build_smooth_sdf import build_coefficients
from scripts.monolithic_reference.smooth_sdf_oracle import evaluate


class TestSolverPrecision(unittest.TestCase):
    def test_native_step_and_graph(self):
        for device in ("cpu", "cuda:0"):
            model = scene(device)
            for graph in (False, True) if device != "cpu" else (False,):
                for law in ("C-EE", "C-IE", "C-II"):
                    runs = [
                        solver(
                            model, law, precision=p, graph=graph, mode="production", publish_policy="current_iterate"
                        )
                        for p in ("fp64", "fp32")
                    ]
                    states = [run.create_state(dt=0.001) for run in runs]
                    for _ in range(2):
                        for run, state in zip(runs, states, strict=True):
                            with self.subTest(device=device, graph=graph, law=law, precision=run.precision):
                                output = run._states.copy(state)
                                run.step(state, output, model.control(), dt=0.001)
                                self.assertFalse(run.last_stats.rolled_back, run.last_stats)
                                state.assign(output)
                                native = np.float64 if run.precision == run.Precision.FP64 else np.float32
                                for field in (
                                    state.joint_q,
                                    state.particle_q,
                                    run._complete_r,
                                    run._objective.residual,
                                    run._tet.residual,
                                    run._contact.residual,
                                    state._monolithic_coupled.link_vsym,
                                ):
                                    self.assertEqual(field.numpy().dtype, native)
                                forces = run.get_contact_forces(state)
                                self.assertEqual(forces.particle_force.dtype, native)
                                with self.assertRaises(ValueError):
                                    forces.particle_force.flat[0] = 0
                                saved = forces.particle_force.copy()
                                run.reset_state(output, dt=0.001)
                                np.testing.assert_array_equal(saved, forces.particle_force)
                                self.assertEqual(run._linear.scale.dtype, wp.float32)
                    if graph:
                        for run in runs:
                            segment = run._graphs.summary()["segments"]["coupled_objective_owners"]
                            self.assertGreater(segment["captures"], 0)
                            self.assertGreater(segment["replays"], segment["captures"])

    def test_fp32_absolute_tolerance(self):
        model = scene("cpu")
        default = solver(model, precision="fp32")
        relaxed = solver(model, precision="fp32", fp32_absolute_tolerance=1e-6)
        for name in ("global", "q", "x"):
            self.assertEqual(getattr(relaxed._config, "merit_absolute_" + name), 1e-6)
            self.assertEqual(
                getattr(relaxed._config, "merit_relative_" + name), getattr(default._config, "merit_relative_" + name)
            )
        self.assertNotEqual(default._kinematics.digest, relaxed._kinematics.digest)
        size = relaxed._linear.layout.scalar_dof_count
        residual = np.zeros(size, dtype=np.float32)
        residual[: model.joint_dof_count] = 2e-6
        relaxed._metrics = {}
        values = measure(relaxed, residual, np.zeros_like(residual), np.ones_like(residual))
        self.assertGreater(values["q"][0], values["q"][2])
        self.assertLessEqual(values["x"][0], values["x"][2])
        target = relaxed.create_state(dt=0.001)
        before = snapshot(target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            default.save_checkpoint(default.create_state(dt=0.001), path)
            with self.assertRaises(ValueError):
                relaxed.load_checkpoint(target, path)
            assert_snapshot(self, target, before)
            relaxed.save_checkpoint(target, path)
            relaxed.load_checkpoint(target, path)
        for value in (True, 0, -1, np.nan, np.inf, 1e100, 1e-100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solver(model, precision="fp32", fp32_absolute_tolerance=value)
        with self.assertRaises(ValueError):
            solver(model, precision="fp64", fp32_absolute_tolerance=1e-6)

    def test_checkpoint_and_cross_precision(self):
        model = scene("cpu")
        run64, run32 = solver(model), solver(model, precision="fp32")
        state64, state32 = run64.create_state(dt=0.001), run32.create_state(dt=0.001)
        self.assertEqual(run64.precision, SolverMonolithic.Precision.FP64)
        self.assertNotIn("quaternion_norm_tolerance", run64._coupled_config)
        np.testing.assert_array_equal(run32._coupled_config["quaternion_norm_tolerance"], [1e-6])
        with self.assertRaises(AttributeError):
            run32.precision = "fp64"
        for value in ("double", None, 32):
            with self.assertRaises(ValueError):
                solver(model, precision=value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            for run, state, other, target in ((run64, state64, run32, state32), (run32, state32, run64, state64)):
                before = snapshot(target)
                run.save_checkpoint(state, path)
                with self.assertRaises(ValueError):
                    other.load_checkpoint(target, path)
                assert_snapshot(self, target, before)
                with self.assertRaises(ValueError):
                    other.step(state, target, model.control(), dt=0.001)
                assert_snapshot(self, target, before)
                run.load_checkpoint(state, path)
            # Simulate the exact old field schema, including its original physical digest.
            legacy = snapshot(state64)
            for name in list(legacy):
                if name in {
                    "history.precision",
                    "history.layout_version",
                    "history.layout_digest",
                    "history.actor_layout",
                } or name.startswith("history.rigid_"):
                    del legacy[name]
            legacy["history.schema_version"][:] = 2
            np.savez(path, **legacy)
            run64.load_checkpoint(state64, path)
            self.assertEqual(state64._monolithic_coupled.precision.numpy()[0], 64)
            before = snapshot(state64)
            legacy["history.config_digest"][0] ^= 1
            np.savez(path, **legacy)
            with self.assertRaises(ValueError):
                run64.load_checkpoint(state64, path)
            assert_snapshot(self, state64, before)
        before = snapshot(state32)
        for dt in (1e-100, 1e100):
            with self.assertRaises(ValueError), np.errstate(over="ignore"):
                run32.step(state32, state32, model.control(), dt=dt)
            assert_snapshot(self, state32, before)

    def test_previous_fp32_guard_checkpoint_rejected(self):
        model = scene("cpu")
        make_kinematics = SolverMonolithic._make_kinematics

        def previous_identity(run, *args, **kwargs):
            # Reproduce the pre-adjustment identity without rewriting a saved digest.
            del run._coupled_config["quaternion_norm_tolerance"]
            return make_kinematics(run, *args, **kwargs)

        with patch.object(SolverMonolithic, "_make_kinematics", previous_identity):
            previous = solver(model, precision="fp32")
        current = solver(model, precision="fp32")
        state = current.create_state(dt=0.001)
        before = snapshot(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous.npz"
            previous.save_checkpoint(previous.create_state(dt=0.001), path)
            with self.assertRaises(ValueError):
                current.load_checkpoint(state, path)
        assert_snapshot(self, state, before)

    def test_sdf_native_reference_and_source_binding(self):
        lo, hi, values = grid()
        for device in ("cpu", "cuda:0"):
            for kind in ("c1-hermite", "c2-bspline"):
                coefficients = build_coefficients(values, (hi - lo) / (np.array(values.shape[::-1]) - 1), kind)
                field = SmoothSDF(coefficients, lo, hi, values.shape[::-1], kind, device=device)
                reference = SmoothSDFSet([field], source_identity={"fixture": "native-test"})
                native = reference.for_precision(wp.float32)
                np.testing.assert_array_equal(
                    native.fields[0].data.coefficients.numpy(), coefficients.ravel().astype(np.float32)
                )
                points = np.random.default_rng(31).uniform(lo * 0.8, hi * 0.8, (17, 3)).astype(np.float32)
                expected = evaluate(values, lo, hi, points.astype(np.float64), kind)
                output = wp.empty((len(points), 13), dtype=wp.float32, device=device)
                wp.launch(
                    sample_smooth,
                    len(points),
                    [native.fields[0].data, wp.array(points, dtype=wp.vec3, device=device), output],
                    device=device,
                )
                actual = output.numpy()
                np.testing.assert_allclose(actual[:, :4], expected[:, :4], atol=5e-6, rtol=5e-5)
                np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], atol=3e-3, rtol=3e-3)
                field.data.coefficients.zero_()
                with self.assertRaisesRegex(ValueError, "coefficients changed"):
                    native.validate_identity()


if __name__ == "__main__":
    unittest.main()
