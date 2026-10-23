# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bitwise run-to-run determinism and the floating-point atomic scan of the solve path."""

import ast
import json
import unittest
from pathlib import Path

import numpy as np

from newton.tests.test_monolithic_solver_actor_contacts import CASE, contact_scene, contact_solver, contact_state, frame
from newton.tests.unittest_utils import add_function_test, get_test_devices

PACKAGE = Path(__file__).resolve().parents[1] / "_src/solvers/monolithic"
_INTEGER_ARRAYS = ("wp.array[int]", "wp.array[wp.int32]", "wp.array[wp.int64]", "wp.array[wp.uint64]")
_INTEGER_CASTS = ("int", "wp.int32", "wp.int64", "wp.uint64")
# Reviewed struct fields with a nonliteral integer increment; check the declared dtype before adding one.
_INTEGER_FIELDS = {("contact_pool.py", "_reserve_contacts", "records.requested")}


def _integer_increment(value):
    """Return whether Warp's matching-type rule makes the accumulated array integer."""
    if isinstance(value, ast.Constant):
        return type(value.value) is int
    return isinstance(value, ast.Call) and ast.unparse(value.func) in _INTEGER_CASTS


def accumulating_atomics():
    """Return (module, function, target) for every atomic add/sub not proven to be an integer counter."""
    sites = []
    for path in sorted(PACKAGE.glob("*.py")):
        for function in ast.walk(ast.parse(path.read_text())):
            if not isinstance(function, ast.FunctionDef):
                continue
            annotations = {a.arg: ast.unparse(a.annotation) for a in function.args.args if a.annotation is not None}
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("atomic_add", "atomic_sub")
                ):
                    target = ast.unparse(node.args[0])
                    if not _integer_increment(node.args[-1]) and annotations.get(target) not in _INTEGER_ARRAYS:
                        sites.append((path.name, function.name, target))
    return sites


class TestAtomicScan(unittest.TestCase):
    def test_no_floating_point_accumulation(self):
        """Every atomic add/sub in the solve path accumulates an integer counter."""
        sites = accumulating_atomics()
        self.assertEqual([site for site in sites if site not in _INTEGER_FIELDS], [])
        # The reviewed fields must still exist, so the list cannot silently go stale.
        self.assertLessEqual(_INTEGER_FIELDS, set(sites))


def run_chain(device, steps, *, graph):
    case = json.loads(CASE.read_text())
    model = contact_scene(device, "chain")
    run = contact_solver(model, mode="production", graph=graph)
    state = contact_state(run, "chain")
    control = model.control()
    frames = []
    for step in range(steps):
        targets = control.joint_target_q.numpy()
        targets[list(run._articulation_actor.target_indices)] = (step + 1) * case["target_increment_per_step"]
        control.joint_target_q.assign(targets)
        run.step(state, state, control, dt=case["dt"])
        frames.append({**frame(run, state), "iterations": run.last_stats.nonlinear_iterations})
    return frames


def assert_bitwise(test, left, right):
    for step, (a, b) in enumerate(zip(left, right, strict=True)):
        test.assertEqual(a.keys(), b.keys())
        for name, value in a.items():
            np.testing.assert_array_equal(value, b[name], err_msg=f"step {step}: {name}")


def test_repeat_bitwise(test, device):
    """Two runs of one tet + rigid + articulation contact scene agree bit for bit."""
    graphs = (False, True) if device.is_cuda else (False,)
    for graph in graphs:
        with test.subTest(graph=graph):
            assert_bitwise(test, run_chain(device, 6, graph=graph), run_chain(device, 6, graph=graph))


def test_graph_bitwise(test, device):
    """Graph replay reproduces direct launches bit for bit."""
    assert_bitwise(test, run_chain(device, 6, graph=False), run_chain(device, 6, graph=True))


class TestSolverDeterminism(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestSolverDeterminism, "test_repeat_bitwise", test_repeat_bitwise, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestSolverDeterminism, "test_graph_bitwise", test_graph_bitwise, devices=[_device])

if __name__ == "__main__":
    unittest.main()
