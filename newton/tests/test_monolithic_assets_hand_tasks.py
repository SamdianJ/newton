# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check hand task attachment metadata and passive object contracts with local assets."""

import os
import unittest
from pathlib import Path

import numpy as np

import newton
from scripts.monolithic_reference.run_hand_tasks import build_model


class TestHandTaskAssets(unittest.TestCase):
    def assets(self):
        path = os.environ.get("NEWTON_SHARPA_INPUT_DIR")
        if not path:
            self.skipTest("Set NEWTON_SHARPA_INPUT_DIR to the retained inputs/sharpa directory")
        return Path(path)

    def test_world_plane_membership(self):
        model, _, _, _, _ = build_model("pull-slider", self.assets(), device="cpu")
        bodies = model.shape_body.numpy()
        for body, shapes in model.body_shapes.items():
            for shape in shapes:
                self.assertEqual(int(bodies[shape]), body)
        planes = np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))
        self.assertEqual(len(planes), 1)
        self.assertEqual(bodies[planes[0]], -1)

    def test_elevated_slider_remains_passive(self):
        model, _, _, _, identity = build_model("pull-slider", self.assets(), device="cpu", height=0.08)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        self.assertAlmostEqual(float(state.body_q.numpy()[identity["object_body"], 2]), 0.1002, places=6)
        self.assertEqual(model.articulation_count, 2)
        for field in ("joint_target_ke", "joint_target_kd", "joint_damping"):
            self.assertEqual(getattr(model, field).numpy()[identity["object_dof"]], 0)
        self.assertTrue(all(kind != int(newton.JointType.FREE) for kind in model.joint_type.numpy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
