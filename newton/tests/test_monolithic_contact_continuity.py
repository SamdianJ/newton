# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check frozen-input prototypes without changing production contact policies."""

import functools
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.contact_continuity import transport_history
from newton.tests import test_solver_monolithic_contact as contact_tests
from newton.tests import test_solver_monolithic_friction as friction_tests
from newton.tests.test_solver_monolithic_collision import _scene
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_continuity import patch_oracle, query_patch


def _query(values, points, device, inverse_dx=(1.0, 1.0, 1.0)):
    result = wp.empty(len(points), dtype=wp.vec4, device=device)
    valid = wp.empty(len(points), dtype=int, device=device)
    wp.launch(
        query_patch,
        len(points),
        [
            wp.array(values, dtype=float, device=device),
            wp.array(points, dtype=wp.vec3, device=device),
            wp.vec3(*inverse_dx),
            result,
            valid,
        ],
        device=device,
    )
    return result.numpy(), valid.numpy()


def test_patch_oracle(test, device):
    """Match an independent Hermite oracle and the scalar field derivative."""
    rng = np.random.default_rng(381)
    values = rng.normal(size=(8, 8, 8)).astype(np.float32)
    points = rng.uniform(1.1, 5.9, size=(30, 3)).astype(np.float32)
    inverse_dx = np.array([2.0, 3.0, 4.0])
    result, valid = _query(values, points, device, inverse_dx)
    test.assertTrue(valid.all())
    expected = np.array([patch_oracle(values, p, inverse_dx) for p in points])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=4e-6)
    for p in points[:5]:
        h = 1e-5
        fd = [
            (
                patch_oracle(values, p + h * np.eye(3)[a], inverse_dx)[0]
                - patch_oracle(values, p - h * np.eye(3)[a], inverse_dx)[0]
            )
            / (2 * h)
            for a in range(3)
        ]
        np.testing.assert_allclose(np.array(fd) * inverse_dx, patch_oracle(values, p, inverse_dx)[1:], atol=1e-8)


def test_patch_continuity(test, device):
    """Remove finite gradient jumps across every axis of a shared grid."""
    values = np.random.default_rng(22).normal(size=(8, 8, 8)).astype(np.float32)
    for axis in range(3):
        center = np.array([3.25, 3.4, 3.65])
        center[axis] = 3.0
        errors = []
        for h in (1e-2, 1e-3, 1e-4):
            result, valid = _query(values, [center - h * np.eye(3)[axis], center + h * np.eye(3)[axis]], device)
            test.assertTrue(valid.all())
            errors.append(np.linalg.norm(result[1] - result[0]))
        test.assertLess(errors[-1], 0.02 * errors[0])


def test_patch_invalid(test, device):
    """Reject missing stencils, nonfinite samples and degenerate gradients."""
    values = np.ones((8, 8, 8), dtype=np.float32)
    _, valid = _query(values, [[0, 3, 3], [6, 3, 3], [3, 3, 3]], device)
    np.testing.assert_array_equal(valid, 0)
    _, valid = _query(values, [[np.nan, 3, 3], [np.inf, 3, 3]], device)
    np.testing.assert_array_equal(valid, 0)
    values[3, 3, 3] = np.nan
    _, valid = _query(values, [[3.1, 3.1, 3.1]], device)
    np.testing.assert_array_equal(valid, 0)


@wp.kernel
def _history(
    old: wp.array[wp.vec3],
    old_normal: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    retain_cos: float,
    smooth: bool,
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    reset_cos = wp.cos(wp.pi / 4.0)
    value = wp.vec3()
    if smooth:
        value = transport_history(old[i], old_normal[i], normals[i], reset_cos, retain_cos)
    elif wp.dot(old_normal[i], normals[i]) >= reset_cos:
        value = old[i] - wp.dot(normals[i], old[i]) * normals[i]
    output[i] = value


def _apply_history(old, reference, normals, device, smooth=True):
    result = wp.empty(len(old), dtype=wp.vec3, device=device)
    wp.launch(
        _history,
        len(old),
        [
            *[wp.array(a, dtype=wp.vec3, device=device) for a in (old, reference, normals)],
            float(np.cos(np.deg2rad(40))),
            smooth,
            result,
        ],
        device=device,
    )
    return result.numpy()


def test_history_continuity(test, device):
    """Expose the original hard-reset jump and remove it without forgetting all history."""
    angles = np.deg2rad([0, 39, 40, 44.999, 45.001, 70])
    normals = np.stack([np.sin(angles), np.zeros(len(angles)), np.cos(angles)], axis=1)
    old = np.tile([0, 0.001, 0], (len(angles), 1))
    reference = np.tile([0, 0, 1], (len(angles), 1))
    legacy = _apply_history(old, reference, normals, device, False)
    result = _apply_history(old, reference, normals, device)
    test.assertGreater(np.linalg.norm(legacy[3] - legacy[4]), 0.0009)
    test.assertLess(np.linalg.norm(result[3] - result[4]), 1e-8)
    np.testing.assert_allclose(result[:3], old[:3], atol=1e-9)
    np.testing.assert_array_equal(result[4:], 0)
    np.testing.assert_array_equal(result, _apply_history(old, reference, normals, device))


def test_history_energy_objectivity(test, device):
    """Preserve tangency, contract old elastic energy and commute with rigid rotation."""
    rng = np.random.default_rng(54)
    old = rng.normal(size=(200, 3))
    reference = rng.normal(size=(200, 3))
    reference /= np.linalg.norm(reference, axis=1)[:, None]
    old -= np.sum(old * reference, axis=1)[:, None] * reference
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    result = _apply_history(old, reference, normals, device)
    test.assertTrue(np.all(np.linalg.norm(result, axis=1) <= np.linalg.norm(old, axis=1) + 1e-6))
    np.testing.assert_allclose(np.sum(result * normals, axis=1), 0, atol=1e-6)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = _apply_history(old @ rotation, reference @ rotation, normals @ rotation, device)
    np.testing.assert_allclose(rotated, result @ rotation, atol=2e-6)


def test_frozen_close_inputs(test, device):
    """Reproduce the real Close jumps and verify both local candidate responses."""
    path = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/contact_continuity_v1.npz"
    identity = json.loads(path.with_suffix(".json").read_text())
    test.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identity["sha256"])
    with np.load(path) as saved:
        result, valid = _query(saved["values"], saved["coordinates"], device, saved["inverse_dx"])
        test.assertTrue(valid.all())
        expected = np.array([patch_oracle(saved["values"], p, saved["inverse_dx"]) for p in saved["coordinates"]])
        np.testing.assert_allclose(result, expected, atol=3e-6, rtol=1e-5)
        baseline = saved["baseline"]
        old_normals = baseline[:2, 1:] / np.linalg.norm(baseline[:2, 1:], axis=1)[:, None]
        normals = result[:2, 1:] / np.linalg.norm(result[:2, 1:], axis=1)[:, None]
        test.assertGreater(np.linalg.norm(old_normals[1] - old_normals[0]), 0.4)
        test.assertLess(np.linalg.norm(normals[1] - normals[0]), 1e-4)
        np.testing.assert_allclose(result[:, 0], baseline[:, 0], atol=1e-5, rtol=0)

        def rotate(q, v):
            return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)

        old = np.array([rotate(saved["rotation_" + s], saved["old_local"]) for s in ("base", "trial")])
        reference = np.array([rotate(saved["rotation_" + s], saved["old_normal_local"]) for s in ("base", "trial")])
        normals = np.array([saved["normal_base"], saved["normal_trial"]])
        legacy = _apply_history(old, reference, normals, device, False)
        candidate = _apply_history(old, reference, normals, device)
        test.assertGreater(np.linalg.norm(legacy[1] - legacy[0]), 5e-4)
        test.assertLess(np.linalg.norm(candidate[1] - candidate[0]), 1e-8)


def test_cubic_collision(test, device):
    """Keep cubic contact detection conservative and reject field rebinding."""
    if not device.is_cuda:
        test.skipTest("Texture SDF construction requires CUDA")
    model, state = _scene(device, shape="mesh")
    streams = []
    for aabb in (False, True):
        pipeline = MonolithicCollisionPipeline(model, soft_contact_gap=0.002, _cubic_sdf=True, _enable_aabb=aabb)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        keys = contacts.soft_contact_tids.numpy()
        records = keys[keys >= 0]
        test.assertGreater(len(records), 0)
        streams.append(
            (keys >= 0, contacts.soft_contact_normal.numpy()[records], contacts.soft_contact_body_pos.numpy()[records])
        )
        for item in pipeline._cubic_evidence:
            test.assertGreater(item["boundary_lower_bound_m"], item["shell_m"])
        positions = state.particle_q.numpy()
        state.particle_q.assign(positions + 1.0)
        pipeline.collide(state, contacts)
        test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
        state.particle_q.assign(positions)
        pipeline._cubic_fields = wp.clone(pipeline._cubic_fields)
        with test.assertRaisesRegex(ValueError, "replaced_cubic_field"):
            pipeline.collide(state, contacts)
        pipeline._cubic_fields = pipeline._cubic_binding
        pipeline._cubic_storage = []
        with test.assertRaisesRegex(ValueError, "replaced_cubic_field"):
            pipeline.collide(state, contacts)
    for a, b in zip(*streams, strict=True):
        np.testing.assert_array_equal(a, b)


def test_smooth_history_residual(test, device):
    """Remove the reset jump in assembled residuals and preserve trial transactions."""
    differences = []
    for smooth in (False, True):
        with patch.object(
            contact_tests,
            "MonolithicCollisionPipeline",
            functools.partial(MonolithicCollisionPipeline, _smooth_history=smooth),
        ):
            scene = friction_tests._friction_scene(device)
        workspace = scene[-2]
        workspace.committed.valid.fill_(1)
        workspace.committed.xi_local.fill_(wp.vec3(0, 0.001, 0))
        responses = []
        for sequence, angle in enumerate((44.999, 45.001)):
            a = np.deg2rad(angle)
            workspace.committed.normal_local.fill_(wp.vec3(np.sin(a), 0, np.cos(a)))
            before = workspace.committed.xi_local.numpy()
            _, current, _ = friction_tests._evaluate_scene(scene, sequence=sequence)
            _, trial, _ = friction_tests._evaluate_scene(scene, mode=1, sequence=sequence)
            np.testing.assert_allclose(current, trial, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(workspace.committed.xi_local.numpy(), before)
            responses.append(current)
        differences.append(np.linalg.norm(responses[1] - responses[0]))
    test.assertGreater(differences[0], 1e-5)
    test.assertLess(differences[1], differences[0] * 1e-4)
    with patch.object(
        contact_tests,
        "MonolithicCollisionPipeline",
        functools.partial(MonolithicCollisionPipeline, _smooth_history=True),
    ):
        friction_tests.test_history_trials(test, device)
        friction_tests.test_rotating_history(test, device)
        model, _, pipeline, _, _, workspace, contacts = friction_tests._friction_scene(device)
        workspace.transport_retain_cos = float(np.cos(np.deg2rad(39)))
        with test.assertRaisesRegex(ValueError, "configuration is stale"):
            workspace.validate_contract(model, pipeline, contacts)


def test_smooth_history_rollback(test, device):
    """Preserve active history on final-publication and State-copy failures."""
    initialize = MonolithicCollisionPipeline.__init__

    def smooth_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs, _smooth_history=True)

    with patch.object(MonolithicCollisionPipeline, "__init__", smooth_init):
        friction_tests.test_solver_history_transaction(test, device)
        friction_tests.test_active_history_rollback(test, device)


class TestContactContinuity(unittest.TestCase):
    pass


for _test in (
    test_patch_oracle,
    test_patch_continuity,
    test_patch_invalid,
    test_history_continuity,
    test_history_energy_objectivity,
    test_frozen_close_inputs,
    test_cubic_collision,
    test_smooth_history_residual,
    test_smooth_history_rollback,
):
    add_function_test(TestContactContinuity, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
