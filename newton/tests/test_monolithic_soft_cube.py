# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_sharpa_grasp import load_calibration
from newton.examples.softbody.monolithic_soft_cube import (
    generate_cube,
    load_cube,
    validate_cube,
    validate_cube_calibration,
)
from scripts.monolithic_reference.calibrate_cube_contact import build_assets, cube_protocol


class TestSoftCube(unittest.TestCase):
    def test_three_grids_and_roundtrip(self):
        """Verify all three grid counts, physical mass and asset round trips."""
        for refinement, nodes, tets in ((1, 125, 384), (2, 343, 1296), (3, 729, 3072)):
            mesh = generate_cube(refinement)
            report = validate_cube(mesh.vertices, np.asarray(mesh.tet_indices).reshape(-1, 4))
            self.assertEqual(report["node_count"], nodes)
            self.assertEqual(report["tet_count"], tets)
            self.assertAlmostEqual(report["mass_kg"], 0.064, places=7)
            self.assertTrue(report["all_nodes_dynamic"])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cube.npz"
                mesh.save(str(path))
                loaded = load_cube(path)
                np.testing.assert_array_equal(loaded.vertices, mesh.vertices)

    def test_invalid_volume(self):
        """Reject changed dimensions and invalid volume connectivity."""
        mesh = generate_cube(1)
        x = np.asarray(mesh.vertices)
        t = np.asarray(mesh.tet_indices).reshape(-1, 4)
        for bad in (np.vstack((t, t[0])), t[:, ::-1][:, [1, 0, 2, 3]], t[:-1]):
            with self.assertRaises(ValueError):
                validate_cube(x, bad)
        with self.assertRaises(ValueError):
            validate_cube(x * 2, t)
        for bad in (0, 4, True):
            with self.assertRaises(ValueError):
                generate_cube(bad)


class TestCubeCalibration(unittest.TestCase):
    def test_binding_and_legacy_rejection(self):
        """Reject calibration from another object, material, timestep or failed scan."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_assets(root)
            protocol = cube_protocol(root)
            result = {
                "status": "FROZEN",
                "protocol": protocol,
                "protocol_sha256": hashlib.sha256((json.dumps(protocol, indent=2) + "\n").encode()).hexdigest(),
                "pcg_iteration_p95_budget": 64,
                "stiffness_floor_n_m3": 5e5,
                "initial_grasp_stiffness_n_m3": 1e7,
                "results": [
                    {"refinement": r, "stiffness": k, "passed": True, "cube_sha256": protocol["assets"][str(r)]}
                    for k in protocol["stiffness_scan_n_m3"]
                    for r in (1, 2, 3)
                ],
            }
            path = root / "calibration.json"
            path.write_text(json.dumps(result))
            validate_cube_calibration(load_calibration(path), root / "cube_r2.npz", dt=0.001)
            for key, value in [("dt", 0.002), ("object_type", "ball"), ("density", 500), ("edge_m", 0.05)]:
                bad = copy.deepcopy(result)
                bad["protocol"][key] = value
                with self.assertRaises(ValueError):
                    validate_cube_calibration(bad, root / "cube_r2.npz", dt=0.001)
            bad = copy.deepcopy(result)
            bad["protocol"]["assets"]["2"] = "wrong"
            with self.assertRaises(ValueError):
                validate_cube_calibration(bad, root / "cube_r2.npz", dt=0.001)
            result["status"] = "FAILED_NO_PHYSICAL_ENVELOPE"
            with self.assertRaises(ValueError):
                validate_cube_calibration(result, root / "cube_r2.npz", dt=0.001)

    def test_all_dynamic_and_mass(self):
        """Check actual CPU/CUDA builder mass and dynamic nodes."""

        for device in ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]:
            for r in (1, 2, 3):
                b = newton.ModelBuilder()
                b.add_soft_mesh(
                    pos=(0, 0, 0),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=(0, 0, 0),
                    mesh=generate_cube(r),
                    particle_radius=0.0002,
                    add_surface_mesh_edges=False,
                )
                m = b.finalize(device=device)
                self.assertTrue(np.all(m.particle_inv_mass.numpy() > 0))
                self.assertAlmostEqual(float(m.particle_mass.numpy().sum()), 0.064, places=7)


if __name__ == "__main__":
    unittest.main()
