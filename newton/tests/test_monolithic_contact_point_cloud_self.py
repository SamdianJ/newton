# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-only self contact on recorded folds and repeated particle support."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cloth_contact_point_cloud import _PointCloudContact
from newton.tests.test_monolithic_contact_point_cloud import _curvature, _pairs, _penalty


def _local_contact(reference, triangles, *, dtype=wp.float64, mu=0.0):
    """Construct the contact producer from a frozen local material fixture."""
    points = reference[triangles]
    areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
    model = SimpleNamespace(device=wp.get_device("cpu"), particle_count=len(reference))
    solver = SimpleNamespace(
        _contact=SimpleNamespace(model=model),
        _dtype=dtype,
        _layout=SimpleNamespace(scalar_dof_count=3 * len(reference)),
        _actors=SimpleNamespace(maps={"particle_to_global": 3 * np.arange(len(reference))}),
        _materials=SimpleNamespace(
            cloth=[
                {
                    "particle_ids": np.arange(len(reference)),
                    "positions": reference,
                    "triangles": triangles,
                    "areas": areas,
                    "thickness": 0.0005,
                    "world": -1,
                }
            ]
        ),
    )
    return _PointCloudContact(
        solver,
        {"radius": 0.012, "normal_stiffness": 1e5, "candidate_padding": 0.003, "candidate_capacity": 128},
        mu=mu,
        smoothing=0.0001,
        velocity_scale=0.01,
        use_cuda_graph=False,
    )


class TestPointCloudSelfContact(unittest.TestCase):
    def test_recorded_reference_far_local_fold(self):
        """Retain responding reference-far neighbors from the recorded crossing fixture."""
        fixture = json.loads((Path(__file__).parent / "assets/monolithic_point_cloud_local_fold.json").read_text())
        vertex_ids = sorted(map(int, fixture["reference_positions_m"]))
        face_ids = sorted(map(int, fixture["original_triangle_ids"]))
        vertices = {p: i for i, p in enumerate(vertex_ids)}
        faces = {f: i for i, f in enumerate(face_ids)}
        reference = np.array([fixture["reference_positions_m"][str(p)] for p in vertex_ids])
        triangles = np.array([[vertices[p] for p in fixture["original_triangle_ids"][str(f)]] for f in face_ids])
        contact = _local_contact(reference, triangles, mu=0.5)
        # Cutting out the local geometry must not change the full mesh's nodal area weights.
        contact.node_area.assign([fixture["global_lumped_node_areas_m2"][str(p)] for p in vertex_ids])
        bary = np.eye(3) * 0.5 + 1 / 6
        for step in (91, 92, 100):
            with self.subTest(step=step):
                start, current = [
                    np.array([fixture["positions_m_by_step"][str(s)][str(p)] for p in vertex_ids])
                    for s in (step - 1, step)
                ]
                contact.start.assign(start)
                contact.refresh(wp.array(current, dtype=wp.vec3d, device="cpu"))
                expected = set()
                for face, tri in enumerate(triangles):
                    for slot, weights in enumerate(bary):
                        for p in range(len(reference)):
                            if np.sum((weights @ reference[tri] - reference[p]) ** 2) < 0.0186**2:
                                continue
                            if min(np.linalg.norm(weights @ x[tri] - x[p]) for x in (start, current)) <= 0.0156:
                                expected.add((3 * face + slot, p))
                self.assertEqual(_pairs(contact), expected)
                restored = [
                    (3 * faces[f] + slot, vertices[p]) for f, slot, p in fixture["restored_active_by_step"][str(step)]
                ]
                self.assertTrue(set(restored) <= expected)
                wp.copy(contact.pairs, wp.array(restored, dtype=wp.vec2i, device="cpu"), count=len(restored))
                contact.pool.candidates.count.fill_(len(restored))
                contact.evaluate(wp.array(current - start, dtype=wp.vec3d, device="cpu"), 0.002)
                self.assertEqual(int(contact.flags.numpy()[0]), 0)
                self.assertAlmostEqual(
                    float(contact.summary.numpy()[5]), fixture["restored_normal_load_N_by_step"][str(step)], delta=1e-12
                )
                self.assertGreater(np.linalg.norm(contact.pool.force.result.numpy()), 0)

    def test_reference_far_incident_node_candidate(self):
        """Apply the same reference rule when a collider node belongs to the source face."""
        reference = np.array([[0, 0, 0], [0.12, 0, 0], [0, 0.12, 0]], dtype=float)
        contact = _local_contact(reference, np.array([[0, 1, 2]]))
        contact.start.assign(0.2 * reference)
        contact.refresh(contact.start)
        self.assertIn((0, 0), _pairs(contact))

    def test_repeated_support_force_residual_and_psd(self):
        """Assemble repeated particle rows as the merged Jacobian coefficient b_j minus one."""
        reference = np.array([[0, 0, 0], [0.12, 0, 0], [0, 0.12, 0]], dtype=float)
        for dtype, tolerance in ((wp.float64, 2e-12), (wp.float32, 3e-6)):
            with self.subTest(dtype=dtype):
                contact = _local_contact(reference, np.array([[0, 1, 2]]), dtype=dtype)
                current = 0.2 * reference
                contact.start.assign(current)
                wp.copy(contact.pairs, wp.array([[0, 0]], dtype=wp.vec2i, device="cpu"), count=1)
                contact.pool.candidates.count.fill_(1)
                contact.evaluate(wp.zeros_like(contact.start), 0.002)
                self.assertEqual(int(contact.pool.records.flags.numpy()[0]), 0)
                coefficients = np.array([-1 / 3, 1 / 6, 1 / 6])
                y = coefficients @ current
                length = np.linalg.norm(y)
                normal = y / length
                jacobian = np.concatenate([b * np.eye(3) for b in coefficients], axis=1)
                stiffness = 1e5 * (0.12**2 / 6) ** 2 / 0.012**2
                energy, load = _penalty(length - 0.0125, stiffness, 0.0001)
                expected_force = jacobian.T @ (load * normal)
                np.testing.assert_allclose(
                    contact.pool.force.result.numpy(), expected_force, rtol=tolerance, atol=tolerance
                )
                np.testing.assert_allclose(
                    contact.pool.residual.result.numpy(), -expected_force, rtol=tolerance, atol=tolerance
                )
                self.assertAlmostEqual(float(contact.energy.numpy()[0]), energy, delta=tolerance)
                records = contact.pool.records
                count = int(records.triplet_count.numpy()[0])
                matrix = np.zeros((9, 9))
                np.add.at(
                    matrix,
                    (records.triplet_rows.numpy()[:count], records.triplet_columns.numpy()[:count]),
                    records.triplet_values.numpy()[:count],
                )
                factor = jacobian.T @ normal
                expected = _curvature(length - 0.0125, stiffness, 0.0001) * np.outer(factor, factor)
                np.testing.assert_allclose(matrix, expected, rtol=tolerance, atol=tolerance)
                self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -tolerance * np.linalg.norm(matrix))
                np.testing.assert_allclose(
                    contact.pool.force.result.numpy().reshape(3, 3).sum(axis=0), 0, atol=tolerance
                )


if __name__ == "__main__":
    unittest.main()
