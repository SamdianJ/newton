# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise shell producers through the public monolithic transaction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_material_state import scene, snapshot


class TestClothSolver(unittest.TestCase):
    def test_independent_and_mixed_materials(self):
        """Step one/many shells and mixed existing actors without a placeholder tet."""
        modes = [
            ("cpu", SolverMonolithic.PCGMode.DIAGNOSTIC, False),
            ("cpu", SolverMonolithic.PCGMode.PRODUCTION, False),
        ]
        if wp.is_cuda_available():
            modes.append((wp.get_cuda_device(), SolverMonolithic.PCGMode.PRODUCTION, True))
        for device, mode, graph in modes:
            for cloth, cable, mixed in ((1, 0, False), (2, 0, False), (1, 1, True)):
                with self.subTest(device=str(device), mode=mode, cloth=cloth, cable=cable):
                    model, solver = scene(
                        cloth=cloth,
                        cables=cable,
                        mechanical=mixed,
                        rigid=mixed,
                        tet=mixed,
                        device=device,
                        pcg_mode=mode,
                        use_cuda_graph=graph,
                        newton_max_iterations=40,
                    )
                    state = solver.create_state(dt=0.01)
                    original = state.particle_q.numpy().copy()
                    force = np.zeros((model.particle_count, 3), dtype=np.float32)
                    force[1] = [0.01, 0.02, 0.03]
                    state.particle_f.assign(force)
                    control = solver.create_control()
                    solver.step(state, state, control, dt=0.01)
                    self.assertTrue(solver.last_stats.converged, solver.last_stats.status)
                    np.testing.assert_array_equal(state.particle_q.numpy()[0], original[0])
                    self.assertGreater(np.linalg.norm(state.particle_q.numpy()[1] - original[1]), 1e-7)
                    self.assertIn("cloth_0_x", solver.last_stats.variational["material_blocks"])

    def test_checkpoint_and_failure_are_atomic(self):
        """Restore into a fresh solver and preserve published arrays on a failed step."""
        for device in ["cpu"] + ([wp.get_cuda_device()] if wp.is_cuda_available() else []):
            with self.subTest(device=str(device)):
                options = {"cloth": 2, "cables": 0, "device": device, "newton_max_iterations": 40}
                model, solver = scene(**options)
                state = solver.create_state(dt=0.01)
                force = np.zeros((model.particle_count, 3), dtype=np.float32)
                force[1, 2] = 0.1
                state.particle_f.assign(force)
                solver.step(state, state, solver.create_control(), dt=0.01)
                self.assertTrue(solver.last_stats.converged)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "cloth.npz"
                    solver.save_checkpoint(state, path)
                    _, other = scene(**options)
                    restored = other.create_state(dt=0.01)
                    other.load_checkpoint(restored, path)
                for owner, current in ((solver, state), (other, restored)):
                    owner.step(current, current, owner.create_control(), dt=0.01)
                    self.assertTrue(owner.last_stats.converged, owner.last_stats)
                for name, values in snapshot(state).items():
                    np.testing.assert_array_equal(snapshot(restored)[name], values, err_msg=name)
                # A nonzero load with no Newton budget must not publish a STOP.
                other.newton_max_iterations = 0
                before = snapshot(restored)
                other.step(restored, restored, other.create_control(), dt=0.01)
                self.assertTrue(other.last_stats.rolled_back)
                self.assertFalse(other.last_stats.converged)
                for name, values in snapshot(restored).items():
                    np.testing.assert_array_equal(values, before[name], err_msg=name)

    def test_trial_path_rejects_interior_collapse(self):
        """Both valid endpoints can hide a collapsed intermediate triangle."""
        _, solver = scene(cloth=True, cables=0)
        state = solver.create_state(dt=0.01)
        solver.step(state, state, solver.create_control(), dt=0.01)
        original = state.particle_q.numpy().copy()
        start = solver._original.particle_q.numpy()

        def place(x):
            # The solver advances positions only through their step displacement.
            solver._trial.particle_q.assign(x)
            solver._displacement(solver._trial).assign(x - start)

        x = original.copy()
        x[1:3] *= -1
        place(x)
        self.assertFalse(solver._domain(solver._trial))
        x = original.copy()
        x[:, 2] += 0.1
        place(x)
        self.assertTrue(solver._domain(solver._trial))
        # Predictor fallback must not read a failed solve's stale direction.
        solver._delta.fill_(float("nan"))
        solver._cloths[0].candidate(
            solver._displacement(solver._original),
            solver._original,
            solver._trial,
            solver._displacement(solver._trial),
            solver._delta,
            0.0,
            0.01,
        )
        np.testing.assert_array_equal(solver._trial.particle_q.numpy(), start)

    def test_prescribed_object_and_gravity(self):
        """A fully prescribed shell has no fake rows while another object accelerates."""
        model, solver = scene(cloth=2, cables=0, fixed_particles=(1, 2))
        model.gravity.assign(np.array([[0, 0, -2]], dtype=np.float32))
        state = solver.create_state(dt=0.01)
        solver.step(state, state, solver.create_control(), dt=0.01)
        self.assertTrue(solver.last_stats.converged, solver.last_stats)
        np.testing.assert_array_equal(state.particle_q.numpy()[:3], model.particle_q.numpy()[:3])
        self.assertLess(state.particle_q.numpy()[4, 2], -1e-5)


if __name__ == "__main__":
    unittest.main()
