# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Native coupled arithmetic, immutable precision and checkpoint transactions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, SmoothSDFSet, sample_smooth
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_sdf_fields import grid
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, scene, snapshot, solver
from scripts.monolithic_reference.build_smooth_sdf import build_coefficients
from scripts.monolithic_reference.smooth_sdf_oracle import evaluate


def far_slide_model(offset):
    """Build a tet sliding at 5 mm/s on a ground plane ``offset`` m from the origin."""
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    builder.add_ground_plane()
    # The solver requires an articulation; this one has no shapes.
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01))
    builder.add_articulation([builder.add_joint_revolute(-1, root, axis=(0, 1, 0))])
    builder.add_soft_mesh(
        pos=(offset, 0.0, 0.0008),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.005, 0, 0),
        vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02)],
        indices=[0, 1, 2, 3],
        density=1000,
        k_mu=10000 / 2.6,
        k_lambda=10000 * 0.3 / (1.3 * 0.4),
        k_damp=0,
        particle_radius=0.001,
        tri_ke=0,
        tri_ka=0,
        tri_kd=0,
        tri_drag=0,
        tri_lift=0,
        edge_ke=0,
        edge_kd=0,
    )
    return builder.finalize(device="cpu")


def skip_unavailable(test, device):
    """Skip the current subtest when ``device`` is not present on this machine."""
    if device != "cpu" and not wp.is_cuda_available():
        test.skipTest(f"{device} is not available")


class TestSolverPrecision(unittest.TestCase):
    def test_native_step_and_graph(self):
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                skip_unavailable(self, device)
                model = scene(device)
                for graph in (False, True) if device != "cpu" else (False,):
                    runs = [
                        solver(model, precision=p, graph=graph, mode="production", publish_policy="current_iterate")
                        for p in ("fp64", "fp32")
                    ]
                    states = [run.create_state(dt=0.001) for run in runs]
                    for _ in range(2):
                        for run, state in zip(runs, states, strict=True):
                            with self.subTest(device=device, graph=graph, precision=run.precision):
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
                                self.assertEqual(run._linear.scale.numpy().dtype, native)
                    if graph:
                        for run in runs:
                            segment = run._graphs.summary()["segments"]["coupled_objective_owners"]
                            self.assertGreater(segment["captures"], 0)
                            self.assertGreater(segment["replays"], segment["captures"])

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
            # Older field schemas, even with the current physical digest, are not migrated.
            before = snapshot(state64)
            for version in (2, 3):
                legacy = {name: value.copy() for name, value in before.items()}
                for name in list(legacy):
                    if name in {
                        "history.precision",
                        "history.layout_version",
                        "history.layout_digest",
                        "history.actor_layout",
                    } or name.startswith("history.rigid_"):
                        del legacy[name]
                legacy["history.schema_version"][:] = version
                np.savez(path, **legacy)
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint schema version"):
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
            with self.subTest(device=device):
                skip_unavailable(self, device)
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

    def test_far_steps_in_float32(self):
        """Resolve rigid and joint steps that are small next to their coordinates."""
        case = (
            Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures/rigid_dynamics_short_v1.json"
        )
        spec = {**json.loads(case.read_text())["rigids"][0], "origin": [1000.0, -700.0, 400.0]}
        spec.update(velocity=[0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.05])
        for name, rigid_specs, q in (("rigid", [spec], None), ("joint", [], [0.3, 100.0])):
            model = scene("cpu", contact=False, rigid_specs=rigid_specs)
            velocities, iterations = [], []
            for precision in ("fp64", "fp32"):
                run = solver(model, precision=precision)
                state = run.create_state(dt=0.001)
                if q is not None:
                    state.joint_q.assign(q)
                    run.reset_state(state, dt=0.001)
                counts = []
                for _ in range(5):
                    output = run._states.copy(state)
                    run.step(state, output, model.control(), dt=0.001)
                    self.assertTrue(run.last_stats.converged, run.last_stats)
                    counts.append(run.last_stats.nonlinear_iterations)
                    state.assign(output)
                velocities.append(state.joint_qd.numpy().astype(float))
                iterations.append(counts)
            with self.subTest(name):
                # A 1 mm/s fall at 1 km is below the FP32 position spacing; the
                # step increments still resolve it to rounding.
                reference, single = velocities
                self.assertLess(np.abs(single - reference).max(), 1e-4 * np.abs(reference).max())
                # Every FP32 step takes a Newton update (no acceptance by coordinate
                # rounding); its looser relative tolerance never needs more than FP64.
                self.assertTrue(all(1 <= a <= b for a, b in zip(iterations[1], iterations[0], strict=True)), iterations)

    def test_far_friction_in_float32(self):
        """Resolve frictional slip of a tet sliding on a ground plane 100 m from the origin."""
        model = far_slide_model(100.0)
        velocities = []
        for precision in ("fp64", "fp32"):
            run = solver(model, precision=precision)
            state = run.create_state(dt=0.001)
            for step in range(3):
                output = run._states.copy(state)
                run.step(state, output, model.control(), dt=0.001)
                # Slip within the friction falloff (1e-5 m) is far below the
                # FP32 spacing of a 100 m coordinate; increments resolve it.
                self.assertTrue(run.last_stats.converged, (precision, step, run.last_stats.status))
                state.assign(output)
                if step == 0:
                    velocities.append(state.particle_qd.numpy().astype(float))
        reference, single = velocities
        self.assertLess(np.abs(single - reference).max(), 1e-3 * np.abs(reference).max())

    def test_far_contact_converges_in_float32(self):
        """Converge far contact to FP32 arithmetic, not to the rounding of world positions."""
        model = far_slide_model(100.0)
        runs = [solver(model, precision=p) for p in ("fp64", "fp32")]
        states = [run.create_state(dt=0.001) for run in runs]
        for step in range(5):
            for run, state in zip(runs, states, strict=True):
                output = run._states.copy(state)
                run.step(state, output, model.control(), dt=0.001)
                self.assertTrue(run.last_stats.converged, (run.precision, step, run.last_stats.status))
                state.assign(output)
            # Store the reference positions as FP32 does, so the comparison
            # isolates the solver arithmetic from the storage of positions.
            reference = states[0]
            velocity = reference.particle_qd.numpy()
            reference.particle_q.assign(reference.particle_q.numpy().astype(np.float32).astype(np.float64))
            runs[0].reset_state(reference, dt=0.001)
            reference.particle_qd.assign(velocity)
            runs[0].reset_state(reference, dt=0.001)
            single = states[1].particle_qd.numpy().astype(float)
            # Accepting a step at the rounding of 100 m coordinates leaves
            # velocity errors of several percent.
            self.assertLess(np.abs(single - velocity).max(), 1e-3 * np.abs(velocity).max(), step)


if __name__ == "__main__":
    unittest.main()
