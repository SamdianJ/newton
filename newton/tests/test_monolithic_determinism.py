# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bitwise run-to-run determinism and the floating-point atomic scan of the solve path."""

import ast
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.contact_pipeline import CONTACT_BATCH_CAPACITY
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_solver_actor_contacts import CASE, contact_scene, contact_solver, contact_state, frame
from newton.tests.test_monolithic_solver_coupled import assert_snapshot, snapshot
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


def _warp_function(function):
    """Return whether ``function`` is compiled by Warp (``@wp.kernel`` or ``@wp.func``)."""
    for decorator in function.decorator_list:
        name = ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
        if name in ("wp.kernel", "wp.func"):
            return True
    return False


def accumulating_atomics():
    """Return (module, function, target) for every atomic add/sub not proven to be an integer counter.

    Warp also compiles ``array[i] += value`` and ``array[i] -= value`` into atomic add/sub, so these
    augmented assignments on array arguments of Warp functions count as atomics.
    """
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
                elif (
                    _warp_function(function)
                    and isinstance(node, ast.AugAssign)
                    and isinstance(node.op, (ast.Add, ast.Sub))
                    and isinstance(node.target, ast.Subscript)
                ):
                    target = ast.unparse(node.target.value)
                    annotation = annotations.get(target, "")
                    if annotation.startswith("wp.array") and annotation not in _INTEGER_ARRAYS:
                        sites.append((path.name, function.name, target))
    return sites


class TestAtomicScan(unittest.TestCase):
    def test_no_floating_point_accumulation(self):
        """Every atomic add/sub in the solve path accumulates an integer counter."""
        sites = accumulating_atomics()
        self.assertEqual([site for site in sites if site not in _INTEGER_FIELDS], [])
        # The reviewed fields must still exist, so the list cannot silently go stale.
        self.assertLessEqual(_INTEGER_FIELDS, set(sites))


def run_chain(device, steps, *, graph, before=False):
    case = json.loads(CASE.read_text())
    model = contact_scene(device, "chain", before=before)
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


def ordered_scene(device, order, *, arm_boxes=2):
    """Create an arm, a FREE body and a static box in ``order`` before one tet."""
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    inertia = wp.diag(wp.vec3(0.001))

    def arm():
        link = builder.add_link(mass=1, inertia=inertia, label="arm")
        joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, limit_ke=0, limit_kd=0)
        builder.add_articulation([joint])
        for k in range(arm_boxes):
            builder.add_shape_box(
                link,
                xform=wp.transform((0.2, 0.05 * k, 0), wp.quat_identity()),
                hx=0.02,
                hy=0.02,
                hz=0.02,
                cfg=builder.ShapeConfig(density=0),
                label=f"arm/{k}",
            )

    def rigid():
        body = builder.add_link(
            xform=wp.transform((-0.2, 0, 0), wp.quat_identity()), mass=1, inertia=inertia, label="rigid"
        )
        builder.add_articulation([builder.add_joint_free(body)])
        builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=builder.ShapeConfig(density=0), label="rigid/0")

    def static():
        builder.add_shape_box(
            -1,
            xform=wp.transform((0, 0, -0.1), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.05,
            cfg=builder.ShapeConfig(density=0),
            label="static/0",
        )

    for name in order:
        {"arm": arm, "rigid": rigid, "static": static}[name]()
    builder.add_soft_mesh(
        pos=(0, 0, 0.0008),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
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
    )
    return builder.finalize(device=device)


def sample_table(run):
    """Return every contact sample with shapes named by label instead of Model index."""
    contact, names = run._contact, run.model.shape_label
    samples = contact.actor_samples
    return {
        "tet_shape": [names[s] for s in contact.shapes.numpy()],
        "sources": [names[s] if s >= 0 else int(s) for s in samples.source_shapes],
        "targets": [names[s] for s in samples.targets],
        "faces": samples.faces.tolist(),
        "slots": samples.slots.tolist(),
        "points": samples.points.tolist(),
        "weights": samples.weights.tolist(),
    }


def test_creation_order_samples(test, device):
    """Contact samples enumerate in the semantic actor order, whatever the Model creation order."""
    tables = []
    for order in itertools.permutations(("arm", "rigid", "static")):
        model = ordered_scene(device, order)
        run = SolverMonolithic(
            model,
            friction_velocity_scale=0.01,
            tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
        )
        tables.append((order, sample_table(run)))
    reference = tables[0][1]
    test.assertTrue(reference["tet_shape"] and reference["targets"])
    for order, table in tables[1:]:
        with test.subTest(order=order):
            test.assertEqual(table, reference)


def test_creation_order_bitwise(test, device):
    """Creating the FREE bodies before the arm leaves every step bitwise unchanged."""
    assert_bitwise(test, run_chain(device, 6, graph=False), run_chain(device, 6, graph=False, before=True))


def test_device_resources(test, device):
    """Contact batches, and so the checkpoint resource identity, do not depend on the device."""

    def make(target):
        model = ordered_scene(target, ("arm", "rigid", "static"), arm_boxes=4)
        return SolverMonolithic(
            model,
            friction_velocity_scale=0.01,
            tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
        )

    run = make(device)
    count = run._contact.count
    # More samples than the former CPU batch of 256.
    test.assertGreater(count, 256)
    test.assertEqual(run._contact._batch.batch_capacity, min(count, CONTACT_BATCH_CAPACITY))
    if device.is_cuda:
        host = make("cpu")
        np.testing.assert_array_equal(run._contact.resource_identity, host._contact.resource_identity)
        test.assertEqual(run._kinematics.digest, host._kinematics.digest)
        state = run.create_state(dt=0.001)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.npz"
            run.save_checkpoint(state, path)
            restored = host.create_state(dt=0.001)
            host.load_checkpoint(restored, path)
        assert_snapshot(test, restored, snapshot(state))


class TestSolverDeterminism(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (test_repeat_bitwise, test_creation_order_samples, test_creation_order_bitwise, test_device_resources):
        add_function_test(TestSolverDeterminism, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestSolverDeterminism, "test_graph_bitwise", test_graph_bitwise, devices=[_device])

if __name__ == "__main__":
    unittest.main()
