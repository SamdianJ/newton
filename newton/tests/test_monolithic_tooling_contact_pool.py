# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate complete contact-pool snapshots through the production workload tools."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton.examples.softbody.monolithic_recording import state_arrays
from newton.tests.test_monolithic_contact_actor_pairs import physical
from newton.tests.test_monolithic_contact_batch import batch_matrix
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.benchmark_workload import benchmark
from scripts.monolithic_reference.contact_full_oracle import evaluate_full
from scripts.monolithic_reference.profiling import capture_count
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.replay_workload import replay
from scripts.monolithic_reference.workload_io import load_arrays
from scripts.monolithic_reference.workload_objective import replay_objective
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays, set_control


def test_pooled_workload(test, device):
    """Record and recompute real pooled windows, all three hotspots and complete physical authority."""
    for scene in ("tiny-contact", "actor-coupling-permuted"):
        for graph in (False, True) if device.is_cuda else (False,):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bundle"
                manifest = record(
                    root,
                    scene=scene,
                    device=device,
                    graph=graph,
                    contact_dataflow="pooled",
                    steps=2,
                    windows=[(2, 2)],
                    boundaries=("objective", "assembly", "pcg"),
                    call_ordinals={"objective": [0, 1, 2], "assembly": [0], "pcg": [0]},
                )
                test.assertEqual(manifest["contact_dataflow"], "pooled")
                for mode in ("window", "hotspot"):
                    test.assertEqual(replay(root, mode=mode, device=device, repeat=2)["status"], "PASS")
                model, run, state, samples = build_scene(scene, device=device, graph=graph, contact_dataflow="pooled")
                control = model.control()
                for step in (1, 2):
                    set_control(control, samples, step)
                    run.step(state, state, control, dt=0.001)
                    test.assertTrue(run.last_stats.converged, run.last_stats)
                compare_arrays(
                    state_arrays(state), load_arrays(root / "state-2.npz", manifest["array_schemas"]["state-2.npz"])
                )
                c = run._contact
                stiffness, mu, velocity, width, radius = c.parameters
                full = evaluate_full(
                    model,
                    run._actors,
                    physical(state),
                    physical(run._original),
                    c.contact_model,
                    radius=radius,
                    stiffness=stiffness,
                    mu=mu,
                    width=width,
                    falloff=velocity * 0.001,
                )
                test.assertEqual(len(full["records"]), c.count)
                np.testing.assert_allclose(c.energy.numpy(), full["energy"], rtol=1e-8, atol=1e-12)
                np.testing.assert_allclose(c.residual.numpy(), full["residual"], rtol=1e-8, atol=1e-10)
                test.assertLess(
                    np.linalg.norm(batch_matrix(c._batch) - full["tangent"]) / max(1, np.linalg.norm(full["tangent"])),
                    1e-5,
                )
                forces = run.get_contact_forces(state)
                for name in ("particle_force", "joint_force"):
                    np.testing.assert_allclose(getattr(forces, name), full[name], rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(forces.body_wrench, full["body_wrench"], rtol=1e-8, atol=1e-10)
                path = next(root / p for p in manifest["snapshots"] if "objective" in p)
                metadata = json.loads((path / "call.json").read_text())
                arrays = load_arrays(path / "input.npz", metadata["input_schema"])
                expected = load_arrays(path / "output.npz", metadata["output_schema"])
                test.assertEqual(metadata["adapter"], "pool-objective/1")
                test.assertTrue(metadata["pool"]["started"])
                test.assertGreater(metadata["pool"]["candidate_generation"], 0)
                test.assertGreater(metadata["pool"]["evaluation_generation"], 0)
                for key in ("contact.batch.start.y", "contact.pool.records.accepted", "contact.batch.support.columns"):
                    test.assertIn(key, arrays)
                    damaged = {k: v for k, v in arrays.items() if k != key}
                    with test.assertRaises(ValueError):
                        replay_objective(damaged, metadata, expected, scene=scene, device=device, graph=graph, repeat=1)
                damaged = {k: v.copy() for k, v in arrays.items()}
                damaged["contact.batch.support.columns"][0] += 1
                with test.assertRaises(ValueError):
                    replay_objective(damaged, metadata, expected, scene=scene, device=device, graph=graph, repeat=1)
                for key in ("pool", "source_hashes", "contact_dataflow"):
                    damaged = copy.deepcopy(metadata)
                    del damaged[key]
                    with test.assertRaises(ValueError):
                        replay_objective(arrays, damaged, expected, scene=scene, device=device, graph=graph, repeat=1)
                damaged = copy.deepcopy(metadata)
                damaged["pool"]["evaluation_generation"] = -1
                with test.assertRaises(ValueError):
                    replay_objective(arrays, damaged, expected, scene=scene, device=device, graph=graph, repeat=1)
                if scene == "tiny-contact":
                    with patch("scripts.monolithic_reference.benchmark_workload.check_gpu_idle"):
                        measured = benchmark(root, device=device, warmup=1, repeat=1)
                    test.assertEqual(measured["status"], "PASS")
                    test.assertTrue(all(s["seconds"] > 0 for s in measured["results"][0]["samples"]))


def test_pooled_capture_accounting(test, device):
    """Count a real isolated fixed-BSR recapture without solver graph or PCG execution."""
    model, run, initial, _ = build_scene("tiny-contact", device=device, graph=True, contact_dataflow="pooled")
    state = run._states.copy(initial)
    control = model.control()
    run.step(state, state, control, dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    before = capture_count(run)
    solver_captures = sum(run._graphs.captures.values())
    builder = run._linear._fixed_bsr[0]
    builder.graphs.entries.clear()
    builder.build(run._linear._global, layout_identity=run._layout.identity)
    test.assertEqual(sum(run._graphs.captures.values()), solver_captures)
    test.assertEqual(capture_count(run), before + 1)


class TestContactPoolWorkload(unittest.TestCase):
    def test_invalid_dataflow(self):
        """Reject unsupported contact algorithms before constructing a Model."""
        with self.assertRaises(ValueError):
            build_scene("tiny-contact", device="cpu", graph=False, contact_dataflow="unknown")


add_function_test(TestContactPoolWorkload, "test_pooled_workload", test_pooled_workload, devices=get_test_devices())
for _device in get_test_devices():
    if _device.is_cuda:
        add_function_test(
            TestContactPoolWorkload, "test_pooled_capture_accounting", test_pooled_capture_accounting, devices=[_device]
        )


if __name__ == "__main__":
    unittest.main()
