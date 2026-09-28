# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check shared Sharpa hand/carriage mounting independently of retired scenes."""

import os
import unittest
from pathlib import Path

import numpy as np

import newton
from newton.examples.softbody.monolithic_sharpa_scene import joint_mapping
from newton.examples.softbody.sharpa_close import build_hand, finalize_hand, load_fixture
from newton.tests.monolithic_test_utils import SHARPA_TEST_MOUNT
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_mount(test, device):
    """Keep hand-relative kinematics intact while only the carriage translates it."""
    asset = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
    if not asset:
        test.skipTest("Set NEWTON_SHARPA_ASSET_DIR for real URDF import")
    parameters = load_fixture(Path(newton.__file__).parent / "examples/softbody/sharpa_g1h.json")
    original, manifest = build_hand(asset, device=device, parameters=parameters)
    baseline, _ = finalize_hand(original, manifest, device=device)
    builder, manifest = build_hand(asset, device=device, parameters=parameters, _mount=SHARPA_TEST_MOUNT)
    model, manifest = finalize_hand(builder, manifest, device=device)
    test.assertEqual(
        (model.body_count, model.joint_dof_count, model.shape_count, model.articulation_count), (34, 23, 55, 1)
    )
    mapping = joint_mapping(model, manifest["joint_names"])
    np.testing.assert_array_equal(model.body_mass.numpy()[1:], baseline.body_mass.numpy())
    np.testing.assert_array_equal(model.body_inertia.numpy()[1:], baseline.body_inertia.numpy())
    state, reference = model.state(), baseline.state()
    q = reference.joint_q.numpy()
    q[:] = np.linspace(-0.1, 0.1, len(q))
    reference.joint_q.assign(q)
    newton.eval_fk(baseline, reference.joint_q, reference.joint_qd, reference)
    targets = state.joint_q.numpy()
    for i, name in enumerate(manifest["joint_names"]):
        targets[mapping[name][0]] = q[i]
    targets[mapping["monolithic_lift"][0]] = 0.05
    state.joint_q.assign(targets)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    actual, expected = state.body_q.numpy(), reference.body_q.numpy()
    np.testing.assert_allclose(actual[1:, :3], expected[:, :3] + [0, 0, 0.05], atol=1e-7)
    np.testing.assert_allclose(actual[1:, 3:], expected[:, 3:], atol=1e-7)
    np.testing.assert_array_equal(actual[0], [0, 0, 0, 0, 0, 0, 1])


class TestGraspMount(unittest.TestCase):
    pass


add_function_test(TestGraspMount, "test_mount", test_mount, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
