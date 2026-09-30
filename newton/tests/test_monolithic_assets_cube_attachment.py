# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cube entry/asset contracts; these tests cannot certify a grasp fixture."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube
from newton.examples.softbody.monolithic_soft_cube import generate_cube
from newton.examples.softbody.sharpa_close import load_fixture
from newton.tests.monolithic_test_utils import SHARPA_TEST_MOUNT


class TestCubeGrasp(unittest.TestCase):
    def test_real_hand_cube_attachment(self):
        """Check the cube attachment with the unchanged 22-DoF hand and carriage."""
        source = os.environ.get("NEWTON_SHARPA_ASSET_DIR")
        contact = os.environ.get("NEWTON_SHARPA_CONTACT_DIR")
        if not source or not contact:
            self.skipTest("Set real Sharpa assets for attachment checks")
        parameters = load_fixture(Path(__file__).parents[1] / "examples/softbody/sharpa_g1h.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.npz"
            generate_cube(2).save(str(path))
            for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
                model, manifest = load_hand_cube(
                    source,
                    contact,
                    path,
                    device=device,
                    parameters=parameters,
                    position=[0.025, -0.03, 0.19],
                    _mount=SHARPA_TEST_MOUNT,
                )
                self.assertEqual(model.joint_dof_count, 23)
                self.assertEqual(model.particle_count, 343)
                self.assertTrue(np.all(model.particle_inv_mass.numpy() > 0))
                self.assertAlmostEqual(float(model.particle_mass.numpy().sum()), 0.064, places=7)
                self.assertEqual(manifest["object_type"], "cube")
                self.assertNotIn("ball_radius_m", manifest)
                np.testing.assert_allclose(model.particle_q.numpy().mean(axis=0), [0.025, -0.03, 0.19], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
