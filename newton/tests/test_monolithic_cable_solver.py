# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise cable physics through the public coupled solver's real step transaction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_material_state import scene, snapshot
from scripts.monolithic_reference.cable_benchmarks import CableBenchmark


class TestMonolithicCableSolver(unittest.TestCase):
    device = "cpu"
    pcg_mode = SolverMonolithic.PCGMode.DIAGNOSTIC
    use_cuda_graph = False

    def scene(self, **options):
        return scene(**options, device=self.device, pcg_mode=self.pcg_mode, use_cuda_graph=self.use_cuda_graph)

    def test_single_edge_and_prescribed_cable_blocks(self):
        """Advance a single edge beside empty or fully prescribed material blocks."""
        for fixed_count in (0, 2, 3):
            with self.subTest(fixed_count=fixed_count):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                for i in range(fixed_count):
                    builder.add_particle((float(i), 0, 0), (0, 0, 0), mass=0)
                builder.add_particle((0, 2, 0), (0, 0, 0), mass=0)
                builder.add_particle((1, 2, 0), (0, 0, 0), mass=0.5)
                model = builder.finalize(device=self.device)
                chains = [tuple(range(fixed_count))] if fixed_count else []
                chains.append((fixed_count, fixed_count + 1))
                cables = [
                    SolverMonolithic.CableInput(
                        particle_ids=ids,
                        EA=10,
                        bending=np.eye(2),
                        GJ=1,
                        radius=0.01,
                        linear_density=1,
                        rest_theta=np.zeros(len(ids) - 1),
                        rest_director=np.tile([0.0, 1.0, 0.0], (len(ids) - 1, 1)),
                        fixed_theta=np.ones(len(ids) - 1, dtype=bool),
                    )
                    for ids in chains
                ]
                solver = SolverMonolithic(
                    model,
                    cable_inputs=cables,
                    friction_velocity_scale=0.1,
                    mass_mode="lumped",
                    pcg_mode=self.pcg_mode,
                    use_cuda_graph=self.use_cuda_graph,
                )
                state, control = solver.create_state(dt=0.1), solver.create_control()
                target = state.particle_q.numpy().copy()
                if fixed_count:
                    target[fixed_count - 1, 1] = 0.02
                    control._monolithic_cable.target_position.assign(target)
                force = np.zeros((fixed_count + 2, 3), dtype=np.float32)
                force[-1, 0] = 0.001
                state.particle_f.assign(force)
                solver.step(state, state, control, dt=0.1)
                self.assertTrue(solver.last_stats.converged, solver.last_stats.status)
                np.testing.assert_array_equal(state.particle_q.numpy()[:-1], target[:-1])
                self.assertAlmostEqual(state.particle_q.numpy()[-1, 0] - 1, float(force[-1, 0]) / 60, delta=1e-10)

    def test_tiny_loads_resolved_per_row(self):
        """Resolve sub-nanonewton forces and picometre torques row by row instead of accepting rest."""
        _, solver = self.scene(cloth=False, cables=1, fixed_theta=(True, False))
        state = solver.create_state(dt=0.1)
        force = state.particle_f.numpy().copy()
        force[-1, 2] = 2e-10
        state.particle_f.assign(force)
        control = solver.create_control()
        control._monolithic_cable.torque.assign([0.0, 2e-12])
        solver.step(state, state, control, dt=0.1)
        self.assertTrue(solver.last_stats.converged, solver.last_stats.status)
        self.assertGreater(state.particle_q.numpy()[-1, 2], 0.0)
        self.assertGreater(state._monolithic_coupled.cable_theta.numpy()[1], 7.2)
        residual = np.asarray(solver.last_stats.variational["residual"])
        rows = solver._criterion.rows
        for unit, load in ((0, 2e-10), (1, 2e-12)):
            # Every row lies within its own tolerance, far below the applied load.
            self.assertLess(np.abs(residual[rows.units == unit]).max(), 0.5 * load)

    def test_transverse_force_and_theta_boundary(self):
        """Advance positions and massless theta together through ordinary step()."""
        _, solver = self.scene(cloth=False, cables=1, fixed_theta=(True, False))
        state = solver.create_state(dt=0.01)
        control = solver.create_control()
        force = state.particle_f.numpy()
        force[-1, 2] = -0.01
        state.particle_f.assign(force)
        target = control._monolithic_cable.target_theta.numpy()
        target[0] += 0.05
        control._monolithic_cable.target_theta.assign(target)
        initial = snapshot(state)
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        self.assertLess(state.particle_q.numpy()[-1, 2], 0)
        self.assertEqual(state._monolithic_coupled.cable_theta.numpy()[0], target[0])
        self.assertNotEqual(state._monolithic_coupled.cable_theta.numpy()[1], initial["history.cable_theta"][1])
        self.assertIn("cable_0_theta", solver.last_stats.variational["material_blocks"])

    def test_free_common_angle_and_incompatible_torque(self):
        """Select a temporary common-angle representative and reject unbalanced torque."""
        _, solver = self.scene(cloth=False, cables=1, fixed_theta=(False, False))
        state = solver.create_state(dt=0.01)
        control = solver.create_control()
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        before = snapshot(state)
        control._monolithic_cable.torque.assign(np.array([0.01, 0.0]))
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.rolled_back)
        self.assertFalse(solver.last_stats.converged)
        for name, values in snapshot(state).items():
            np.testing.assert_array_equal(values, before[name], err_msg=name)

    def test_moving_boundary_checkpoint_and_mixed_actors(self):
        """Continue moving cable clamps through a fresh solver alongside A/R/T actors."""
        _, solver = self.scene(cloth=False, cables=2, mechanical=True, rigid=True, tet=True)
        state = solver.create_state(dt=0.01)
        control = solver.create_control()
        target = control._monolithic_cable.target_position.numpy().copy()
        target[0, 1] += 1e-4
        control._monolithic_cable.target_position.assign(target)
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        np.testing.assert_array_equal(state.particle_q.numpy()[0], target[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cable.npz"
            solver.save_checkpoint(state, path)
            _, restored_solver = self.scene(cloth=False, cables=2, mechanical=True, rigid=True, tet=True)
            restored = restored_solver.create_state(dt=0.01)
            restored_solver.load_checkpoint(restored, path)
        restored_control = restored_solver.create_control()
        restored_control._monolithic_cable.target_position.assign(target)
        solver.step(state, state, control, dt=0.01)
        restored_solver.step(restored, restored, restored_control, dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        self.assertTrue(restored_solver.last_stats.converged, restored_solver.last_stats)
        for name, values in snapshot(state).items():
            np.testing.assert_array_equal(snapshot(restored)[name], values, err_msg=name)

    def test_collapsed_boundary_rolls_back(self):
        """Reject a prescribed collapsed edge and preserve every published array."""
        _, solver = self.scene(cloth=False, cables=1, fixed_particles=(1,))
        state = solver.create_state(dt=0.01)
        control = solver.create_control()
        target = control._monolithic_cable.target_position.numpy().copy()
        target[1] = target[0]
        control._monolithic_cable.target_position.assign(target)
        before = snapshot(state)
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.rolled_back)
        for name, values in snapshot(state).items():
            np.testing.assert_array_equal(values, before[name], err_msg=name)

    def test_trial_path_cannot_cross_edge_collapse(self):
        """Reject collapse between valid endpoints inside the same temporal frame chart."""
        _, solver = self.scene(cloth=False, cables=1)
        state = solver.create_state(dt=0.01)
        solver.step(state, state, solver.create_control(), dt=0.01)
        accepted, trial = solver._accepted, solver._trial
        start = solver._original.particle_q.numpy()

        def place(target, x):
            # The solver advances positions only through their step displacement.
            target.particle_q.assign(x)
            solver._displacement(target).assign(x - start)

        x = state.particle_q.numpy().copy()
        x[1] = x[0] + [0, 1, 0]
        x[2] = x[1] + [1, 0, 0]
        place(accepted, x)
        x[1] = x[0] + [0, -1, 0]
        x[2] = x[1] + [1, 0, 0]
        place(trial, x)
        self.assertFalse(solver._domain(trial))

    def test_permanent_gauge_checks_omitted_torque(self):
        """Reject a small incompatible torque applied only to the omitted gauge row."""
        _, solver = self.scene(cloth=False, cables=1, fixed_theta=(False, False), bending=((1, 0), (0, 1)))
        state = solver.create_state(dt=0.01)
        control = solver.create_control()
        control._monolithic_cable.torque.assign(np.array([5e-10, 0.0]))
        before = snapshot(state)
        solver.step(state, state, control, dt=0.01)
        self.assertTrue(solver.last_stats.rolled_back)
        for name, values in snapshot(state).items():
            np.testing.assert_array_equal(values, before[name], err_msg=name)

    def test_slender_cantilever_direction(self):
        """Qualify slender-beam directions by one true-residual check per solve."""
        case = CableBenchmark(
            "cantilever",
            segments=128,
            device=self.device,
            pcg_mode=self.pcg_mode.value,
            use_cuda_graph=self.use_cuda_graph,
        )
        for _ in range(10):
            case.step()
            checks = case.solver.last_stats.variational["linear_postchecks"]
            qualified = [row for row in checks if all(b["residual"] <= b["limit"] for b in row["blocks"].values())]
            self.assertTrue(qualified)

    def test_anisotropic_bending_releases_common_angle(self):
        """Let a formerly straight anisotropic cable choose its physical bend orientation."""
        _, solver = self.scene(
            cloth=False,
            cables=1,
            fixed_theta=(False, False),
            newton_max_iterations=200,
            linear_tolerance=0.5,
        )
        state = solver.create_state(dt=0.1)
        control = solver.create_control()
        theta = state._monolithic_coupled.cable_theta.numpy().copy()
        f = state.particle_f.numpy().copy()
        f[-1, 1:] = [0.01, 0.02]
        state.particle_f.assign(f)
        solver.step(state, state, control, dt=0.1)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        self.assertFalse(solver._temporary_gauges)
        self.assertGreater(abs(np.mean(state._monolithic_coupled.cable_theta.numpy() - theta)), 1e-3)

    def test_stopped_publication_remains_distinct(self):
        """Report optional stopped cable publication without claiming convergence."""
        _, solver = self.scene(cloth=False, cables=1, newton_max_iterations=1, publish_policy="last_accepted")
        state = solver.create_state(dt=0.1)
        force = state.particle_f.numpy().copy()
        force[-1, 2] = -0.1
        state.particle_f.assign(force)
        solver.step(state, state, solver.create_control(), dt=0.1)
        self.assertFalse(solver.last_stats.converged)
        self.assertFalse(solver.last_stats.rolled_back)
        detail = solver.last_stats.variational
        self.assertEqual(detail["publication_status"], "STOPPED_ACCEPTED")
        self.assertIn("cable_0_theta", detail["stopped_candidate"]["final_ratios"])

    def test_fp32_frames_follow_exact_edges(self):
        """Admit FP32 frames that follow the exact edges of rounded far, short-edged coordinates."""
        from newton._src.solvers.monolithic.material_inputs import _cable_geometry  # noqa: PLC0415

        case = CableBenchmark("cantilever", segments=256, precision="fp32", device=self.device)
        states = case.solver._states
        state = states.copy(case.state)
        history = state._monolithic_coupled
        # A helix 4 m from the origin: rounding the stored coordinates tilts its 6 mm edges
        # by ~1e-5, while the frames follow the exact increment-form edges.
        s = np.linspace(0.0, 1.0, 257)
        x = np.c_[4.0 + s, 0.3 * np.cos(4 * s), 0.3 * np.sin(4 * s)]
        tangent = np.diff(x, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        director = np.cross(tangent, [1.0, 0.0, 0.0])
        director /= np.linalg.norm(director, axis=1)[:, None]
        twist = _cable_geometry(x, director)[3]
        state.particle_q.assign(x.astype(np.float32))
        history.bound_x.assign(x.astype(np.float32))

        def frames(director, twist):
            for prefix in ("cable_", "cable_bound_"):
                getattr(history, prefix + "ref_director").assign(director.astype(np.float32))
                getattr(history, prefix + "ref_twist").assign(twist.astype(np.float32))

        frames(director, twist)
        states.validate(state)
        # A frame tilted beyond the rounding bound is still rejected.
        tilted = director.copy()
        tilted[100] = np.cos(1e-3) * director[100] + np.sin(1e-3) * tangent[100]
        frames(tilted, twist)
        with self.assertRaisesRegex(ValueError, "perpendicular"):
            states.validate(state)
        changed = twist.copy()
        changed[100] += 1e-3
        frames(director, changed)
        with self.assertRaisesRegex(ValueError, "twist"):
            states.validate(state)

    def test_fp32_published_frames_stay_unit(self):
        """Keep FP32 reference directors unit across steps instead of accumulating transport rounding."""
        case = CableBenchmark("cantilever", segments=32, precision="fp32", device=self.device)
        for _ in range(120):
            case.step()
        directors = case.state._monolithic_coupled.cable_ref_director.numpy().astype(float)
        self.assertLess(np.abs(np.linalg.norm(directors, axis=1) - 1).max(), np.finfo(np.float32).eps)

    def test_published_frames_stay_perpendicular(self):
        """Publish each director perpendicular to its step's exact edge, so transport rounding cannot accumulate."""
        from newton._src.solvers.monolithic.cable_system import _publish_frame  # noqa: PLC0415

        rng = np.random.default_rng(4)
        tangent = rng.standard_normal((16, 3))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        director = np.cross(tangent, rng.standard_normal((16, 3)))
        director /= np.linalg.norm(director, axis=1)[:, None]
        # A transported director a little off its edge's normal plane and unit length.
        drifted = (director + 1e-5 * tangent) * (1 + 1e-6)
        published = wp.zeros(16, dtype=wp.vec3, device=self.device)
        twist = wp.zeros(15, dtype=float, device=self.device)
        wp.launch(
            _publish_frame,
            16,
            [
                wp.array(drifted, dtype=wp.vec3, device=self.device),
                wp.array(tangent, dtype=wp.vec3, device=self.device),
                wp.zeros(15, dtype=float, device=self.device),
                0,
                0,
                published,
                twist,
            ],
            device=self.device,
        )
        result = published.numpy().astype(float)
        eps = np.finfo(np.float32).eps
        self.assertLess(np.abs(np.sum(result * tangent, axis=1)).max(), 4 * eps)
        self.assertLess(np.abs(np.linalg.norm(result, axis=1) - 1).max(), 4 * eps)
        np.testing.assert_allclose(result, director, atol=1e-4)

    def test_fp32_straight_rod_responds_to_small_load(self):
        """Resolve a small load on an axis-aligned FP32 rod instead of accepting it under the rounding floor."""
        case = CableBenchmark("cantilever", segments=128, precision="fp32", device=self.device)
        for _ in range(3):
            case.step()
            self.assertGreater(case.solver.last_stats.nonlinear_iterations, 0)
        self.assertGreater(abs(case.state.particle_q.numpy()[-1, case.axis]), 0.0)

    def test_fp32_rotated_rod_solves_unregularized(self):
        """Solve a rotated slender FP32 rod's load step without regularizing its direction.

        Assembled FP32 entries of the stiff stretch blocks round by about eps EA/L,
        far above the soft bending modes: the stored matrix turns indefinite in a
        general orientation, so directions were only accepted after strong
        regularization and the tip responded a fraction of its load.
        """
        from scripts.monolithic_reference.precision_checks import SolverTrace  # noqa: PLC0415

        tips = {}
        for precision in ("fp64", "fp32"):
            case = CableBenchmark("cantilever", segments=64, angle=45.0, precision=precision, device=self.device)
            direction = case.rotation[:, case.axis]
            force = np.zeros((65, 3), dtype=np.float32)
            force[-1] = 3e-6 * direction
            case.state.particle_f.assign(force)
            with SolverTrace(case.solver) as trace:
                step = trace.step(lambda case=case: case.solver.step(case.state, case.state, case.control, dt=case.dt))
            self.assertTrue(case.solver.last_stats.converged, case.solver.last_stats.status)
            self.assertFalse(step.regularized, step.attempts)
            tips[precision] = float((case.state.particle_q.numpy()[-1].astype(float) - case.rest[-1]) @ direction)
        self.assertGreater(tips["fp64"], 0.0)
        self.assertLess(abs(tips["fp32"] / tips["fp64"] - 1), 0.02, tips)

    def test_consistent_tet_mass_excludes_cable_particles(self):
        """Assemble a consistent-mass tet and a line-density cable in one solve."""
        _, solver = self.scene(cloth=False, cables=1, tet=True, mass_mode="consistent")
        state = solver.create_state(dt=0.01)
        force = state.particle_f.numpy().copy()
        force[2, 2] = -0.01
        force[-1, 1] = 0.001
        state.particle_f.assign(force)
        solver.step(state, state, solver.create_control(), dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats.status)
        self.assertLess(state.particle_q.numpy()[2, 2], 0)
        self.assertGreater(state.particle_q.numpy()[-1, 1], 0)


class TestMonolithicCableSolverProduction(TestMonolithicCableSolver):
    pcg_mode = SolverMonolithic.PCGMode.PRODUCTION


@unittest.skipUnless(wp.is_cuda_available(), "CUDA unavailable")
class TestMonolithicCableSolverCUDA(TestMonolithicCableSolverProduction):
    device = "cuda:0"
    use_cuda_graph = True


if __name__ == "__main__":
    unittest.main()
