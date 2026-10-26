# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check coupled multi-articulation continuation and all-actor failure transactions."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.tests.test_monolithic_contact_multitree import contact_trees, trees_solver
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_multitree_checkpoint(test, device):
    """Restore all tree histories, require the new schema and reject changed owners atomically."""
    model = contact_trees(device)
    run = trees_solver(model, pcg_mode="production", use_cuda_graph=device.is_cuda)
    state = run.create_state(dt=0.001)
    test.assertEqual(int(state._monolithic_coupled.schema_version.numpy()[0]), 7)
    control = model.control()
    run.step(state, state, control, dt=0.001)
    test.assertTrue(run.last_stats.converged)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "multi.npz"
        run.save_checkpoint(state, path)
        expected_start = snapshot(state)
        run.step(state, state, control, dt=0.001)
        expected_end = snapshot(state)
        for repeat in range(2):
            run.load_checkpoint(state, path)
            assert_snapshot(test, state, expected_start)
            run.step(state, state, control, dt=0.001)
            current = snapshot(state)
            test.assertEqual(int(current["history.force_generation"][0]), repeat + 3)
            for name, value in expected_end.items():
                if name != "history.force_generation":
                    np.testing.assert_array_equal(current[name], value, err_msg=name)
        resumed = trees_solver(model, pcg_mode="production", use_cuda_graph=device.is_cuda)
        restored = resumed.create_state(dt=0.001)
        resumed.load_checkpoint(restored, path)
        resumed.step(restored, restored, control, dt=0.001)
        assert_snapshot(test, restored, expected_end)
        for other in (trees_solver(model, reverse=True), trees_solver(contact_trees(device, filtered=True))):
            target = other.create_state(dt=0.001)
            before = snapshot(target)
            with test.assertRaises(ValueError):
                other.load_checkpoint(target, path)
            assert_snapshot(test, target, before)
        with np.load(path, allow_pickle=False) as stored:
            data = {key: stored[key].copy() for key in stored.files}
        data["history.schema_version"][:] = 6
        np.savez(path, **data)
        before = snapshot(state)
        with test.assertRaisesRegex(ValueError, "schema version"):
            run.load_checkpoint(state, path)
        assert_snapshot(test, state, before)


def test_multitree_transactions(test, device):
    """Roll back both trees, FREE rigid and material on hard failures or partial publication."""
    run = trees_solver(
        contact_trees(device, mixed=True), pcg_mode="production", use_cuda_graph=device.is_cuda, mass_mode="lumped"
    )
    state = run.create_state(dt=0.001)
    control = run.model.control()
    run.step(state, state, control, dt=0.001)
    test.assertTrue(run.last_stats.converged, run.last_stats)
    for alias in (False, True):
        output = state if alias else run.create_state(dt=0.001)
        source, target = snapshot(state), snapshot(output)
        forces = run.get_contact_forces(state).body_wrench.copy()
        for method in ("_assemble", "_finish_candidate"):
            with patch.object(run, method, side_effect=_StepFailure(run.Status.NONFINITE, "test hard failure")):
                run.step(state, output, control, dt=0.001)
            test.assertTrue(run.last_stats.rolled_back)
            assert_snapshot(test, state, source)
            assert_snapshot(test, output, target)
            np.testing.assert_array_equal(run.get_contact_forces(state).body_wrench, forces)
        assign = output.assign

        def partial(candidate, assign=assign):
            assign(candidate)
            if candidate is run._accepted:
                raise RuntimeError("test partial publication")

        with patch.object(output, "assign", partial):
            with test.assertRaisesRegex(RuntimeError, "partial publication"):
                run.step(state, output, control, dt=0.001)
        assert_snapshot(test, state, source)
        assert_snapshot(test, output, target)
    for policy in ("strict", "last_accepted"):
        for capacity in ({"contact_candidate_capacity": 1}, {"contact_factor_capacity": 0}):
            overflow = trees_solver(
                contact_trees(device, mixed=True),
                pcg_mode="production",
                use_cuda_graph=device.is_cuda,
                mass_mode="lumped",
                publish_policy=policy,
                **capacity,
            )
            state = overflow.create_state(dt=0.001)
            before = snapshot(state)
            with test.assertWarnsRegex(RuntimeWarning, "capacity exceeded"):
                overflow.step(state, state, overflow.model.control(), dt=0.001)
            test.assertEqual(overflow.last_stats.status, overflow.Status.CONTACT_OVERFLOW)
            test.assertFalse(overflow.last_stats.converged)
            test.assertTrue(overflow.last_stats.rolled_back)
            assert_snapshot(test, state, before)


def test_multitree_routes(test, device):
    """Match native eager and graph trajectories through active A-A/A-R contact plus a tet."""
    for precision in ("fp64", "fp32"):
        frames = []
        modes = (("diagnostic", False), ("production", device.is_cuda))
        for mode, graph in modes:
            run = trees_solver(
                contact_trees(device, mixed=True),
                precision=precision,
                pcg_mode=mode,
                use_cuda_graph=graph,
                mass_mode="lumped",
            )
            state = run.create_state(dt=0.001)
            control = run.model.control()
            for step in range(3):
                effort = control.joint_f.numpy()
                effort[:3] = [0.02 * (step + 1), -0.04, -0.01 * (step + 1)]
                control.joint_f.assign(effort)
                run.step(state, state, control, dt=0.001)
                test.assertTrue(run.last_stats.converged, run.last_stats)
                test.assertFalse(run.last_stats.rolled_back)
                test.assertEqual(len(run.last_stats.variational["articulation_blocks"]), 2)
                kinds = {p["kind"] for p in run.last_stats.variational["actor_contact_pairs"]}
                test.assertTrue({"A-A", "A-R"}.issubset(kinds))
            frames.append(snapshot(state))
        for name, actual in frames[1].items():
            if name.startswith("history.") and ("digest" in name or "layout" in name):
                continue
            if actual.dtype.kind == "f":
                np.testing.assert_allclose(
                    actual,
                    frames[0][name],
                    rtol=1e-4 if precision == "fp32" else 1e-8,
                    atol=1e-5 if precision == "fp32" else 1e-9,
                    err_msg=name,
                )
            else:
                np.testing.assert_array_equal(actual, frames[0][name], err_msg=name)


class TestMultiTreeState(unittest.TestCase):
    pass


for _test in (test_multitree_checkpoint, test_multitree_transactions, test_multitree_routes):
    add_function_test(TestMultiTreeState, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
