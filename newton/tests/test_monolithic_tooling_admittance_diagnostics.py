# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify offline direction diagnostics after a production material step."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_monolithic_contact_materials import contact_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.diagnose_franka_admittance import observe_linear, probe_direction


def test_direction_components(test, device):
    """Recover every residual component after graph-backed production evaluation."""
    run = contact_solver(
        body="articulation",
        device=device,
        precision="fp64",
        pcg_mode="production",
        use_cuda_graph=device.is_cuda,
        newton_max_iterations=1,
        linear_max_iterations=100,
        publish_policy="current_iterate",
    )
    state = run.create_state(dt=0.001)
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        captured = observe_linear(run, directory, direction_iteration=0)
        run.step(state, state, run.create_control(), dt=0.001)
        test.assertFalse(run.last_stats.rolled_back)
        test.assertTrue(captured)
        with np.load(directory / f"linear-{captured['call']:02d}.npz", allow_pickle=False) as recorded:
            for name, value in captured["arrays"].items():
                np.testing.assert_array_equal(value, recorded[name], err_msg=name)
        before = state.particle_q.numpy().copy()
        probe_direction(run, captured, directory, 0.001)
        np.testing.assert_array_equal(state.particle_q.numpy(), before)
        report = json.loads((directory / "direction-probe.json").read_text())
        test.assertEqual(len(report["rows"]), 16)
        test.assertEqual(set(report["derivatives"][0]["components"]), {"articulation", "cloth_0", "contact"})
        with np.load(directory / "direction-probe.npz", allow_pickle=False) as vectors:
            for row in report["rows"]:
                prefix = f"{row['direction']}_{row['alpha']}"
                total = sum(vectors[f"{prefix}_{name}"] for name in ("articulation", "cloth_0", "contact"))
                np.testing.assert_allclose(total, vectors[f"{prefix}_residual"], rtol=1e-12, atol=1e-12)
                test.assertTrue(np.isfinite(row["merit"]))


class TestAdmittanceDiagnostics(unittest.TestCase):
    pass


add_function_test(
    TestAdmittanceDiagnostics, "test_direction_components", test_direction_components, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
