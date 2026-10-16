# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Short material contact loading and fresh-solver continuation through the public step API."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_materials import contact_solver
from newton.tests.test_monolithic_material_state import snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def rollout(test, *, cable, device, mode, graph):
    """Run the frozen 20-step ramp, hold and unload with a step-30 checkpoint."""
    options = {
        "cable": cable,
        "body": True,
        "device": device,
        "mu": 0.4,
        "law": "C-IE",
        "pcg_mode": mode,
        "use_cuda_graph": graph,
    }
    solver = contact_solver(**options)
    a, b = solver.create_state(dt=0.001), solver.create_state(dt=0.001)
    control = solver.create_control()
    restored_solver = restored = next_restored = None
    frames, restart_error = [], 0.0
    with tempfile.TemporaryDirectory() as directory:
        for step in range(60):
            scale = min((step + 1) / 20, 1.0, (59 - step) / 19)
            force = np.zeros((solver.model.particle_count, 3), dtype=np.float32)
            if cable:
                control._monolithic_cable.torque.fill_(scale * 1e-7)
            else:
                force[:, 1] = scale * 0.01
            a.particle_f.assign(force)
            solver.step(a, b, control, dt=0.001)
            test.assertTrue(solver._last_stats.converged, (step, solver._last_stats.failure_reason))
            test.assertFalse(solver._last_stats.rolled_back)
            published = solver.get_contact_forces(b)
            resultant = published.particle_force.sum(axis=0) + published.body_wrench[:, :3].sum(axis=0)
            test.assertLess(np.linalg.norm(resultant), 1e-8 + 1e-6 * np.linalg.norm(published.particle_force))
            h = b._monolithic_coupled
            frames.append(
                {
                    "step": step + 1,
                    "x": b.particle_q.numpy().tolist(),
                    "theta": h.cable_theta.numpy().tolist(),
                    "com": h.rigid_com.numpy().tolist(),
                    "force": published.particle_force.tolist(),
                    "torque": published.cable_torque.tolist(),
                }
            )
            if restored is not None:
                restored.particle_f.assign(force)
                restored_solver.step(restored, next_restored, control, dt=0.001)
                test.assertTrue(restored_solver._last_stats.converged, restored_solver._last_stats.failure_reason)
                expected, actual = snapshot(b), snapshot(next_restored)
                for name in expected:
                    np.testing.assert_array_equal(expected[name], actual[name], err_msg=f"step {step}: {name}")
                    if expected[name].size and expected[name].dtype.kind == "f":
                        restart_error = max(restart_error, float(np.max(abs(expected[name] - actual[name]))))
                restored, next_restored = next_restored, restored
            a, b = b, a
            if step == 29:
                path = Path(directory) / "material-contact.npz"
                solver.save_checkpoint(a, path)
                restored_solver = contact_solver(**options)
                restored, next_restored = restored_solver.create_state(dt=0.001), restored_solver.create_state(dt=0.001)
                restored_solver.load_checkpoint(restored, path)
    test.assertGreater(np.linalg.norm(np.asarray(frames[-1]["x"]) - solver.model.particle_q.numpy()), 1e-8)
    test.assertTrue(all(np.linalg.norm(frame["force"]) > 1e-8 for frame in frames))
    if cable:
        test.assertGreater(np.max(np.abs([frame["theta"] for frame in frames])), 1e-6)
        test.assertGreater(np.max(np.abs([frame["torque"] for frame in frames])), 1e-9)
    return {
        "status": "PASS",
        "cable": cable,
        "device": str(device),
        "mode": mode.value,
        "graph": graph,
        "steps": 60,
        "restart_steps": 30,
        "restart_max_difference": restart_error,
        "frames": frames,
    }


def test_material_contact_rollout(test, device):
    """Complete the short loading protocol in supported CPU/CUDA execution modes."""
    modes = ((SolverMonolithic.PCGMode.DIAGNOSTIC, False), (SolverMonolithic.PCGMode.PRODUCTION, False))
    if device.is_cuda:
        modes = ((SolverMonolithic.PCGMode.PRODUCTION, False), (SolverMonolithic.PCGMode.PRODUCTION, True))
    for cable in (False, True):
        for mode, graph in modes:
            with test.subTest(cable=cable, mode=mode.value, graph=graph):
                rollout(test, cable=cable, device=device, mode=mode, graph=graph)


class TestMaterialContactRollout(unittest.TestCase):
    pass


add_function_test(TestMaterialContactRollout, "test_loading", test_material_contact_rollout, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
