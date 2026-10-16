# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host input contracts, independent of device allocation or solver execution."""

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from newton._src.solvers.monolithic.material_inputs import _MaterialInputs
from newton._src.solvers.monolithic.solver_types import CableInput, ClothInput
from scripts.monolithic_reference.cable_oracle import CableReference


class HostArray:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data.copy()


def material_model():
    fields = {
        "shape_flags": np.empty(0, dtype=int),
        "particle_q": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
        "particle_mass": [1 / 6, 1 / 6, 1 / 6, 0.5, 1, 0.5],
        "particle_inv_mass": [0, 6, 6, 0, 1, 2],
        "particle_world": [0] * 6,
        "particle_flags": [1] * 6,
        "tet_indices": np.empty((0, 4), dtype=int),
        "tri_indices": [[0, 1, 2]],
        "tri_materials": [[2, 3, 0, 0, 0]],
        "tri_activations": [0],
        "edge_indices": np.empty((0, 4), dtype=int),
        "edge_bending_properties": np.empty((0, 2)),
        "spring_indices": np.empty((0, 2), dtype=int),
        "spring_stiffness": [],
        "spring_damping": [],
    }
    return SimpleNamespace(particle_count=6, world_count=1, **{k: HostArray(v) for k, v in fields.items()})


def cloth_input(**changes):
    return ClothInput(**{"triangle_ids": (0,), "bending": (1, 2), "thickness": 0.01, "surface_density": 1} | changes)


def cable_input(**changes):
    return CableInput(
        **{
            "particle_ids": (3, 4, 5),
            "EA": 10,
            "bending": ((2, 0.2), (0.2, 1)),
            "GJ": 1,
            "radius": 0.01,
            "linear_density": 1,
            "rest_theta": (7.0, 7.2),
            "rest_director": ((0, 1, 0), (0, 1, 0)),
            "fixed_theta": (False, True),
        }
        | changes
    )


class TestMaterialInputs(unittest.TestCase):
    def test_mass_rest_shape_and_ownership(self):
        """Retain physical mass at fixed vertices and natural material strains."""
        values = _MaterialInputs(material_model(), (cloth_input(),), (cable_input(),))
        np.testing.assert_allclose(values.cloth[0]["mass"], [1 / 6] * 3)
        np.testing.assert_allclose(values.cable[0]["mass"], [0.5, 1, 0.5])
        np.testing.assert_allclose(values.cable[0]["rest_twist"], [0.2])
        np.testing.assert_array_equal(values.cable[0]["free_theta"], [0])
        self.assertEqual(values.theta_count, 2)

    def test_owned_immutable_snapshots(self):
        """Detach reference inputs from callers and bind their values into identity."""
        theta = np.array([7.0, 7.2])
        model = material_model()
        values = _MaterialInputs(model, (cloth_input(),), (cable_input(rest_theta=theta),))
        digest = values.identity
        theta[:] = 0
        model.particle_q.data[:] = 10
        np.testing.assert_array_equal(values.cable[0]["rest_theta"], [7.0, 7.2])
        with self.assertRaises(ValueError):
            values.cable[0]["rest_theta"].setflags(write=True)
        self.assertEqual(digest, values.identity)
        changed = _MaterialInputs(material_model(), (cloth_input(),), (cable_input(EA=11),))
        self.assertNotEqual(values.identity, changed.identity)

    def test_permanent_gauge_only_for_isotropic_naturally_straight(self):
        """Distinguish permanent material symmetry from instantaneous straight-rod degeneracy."""
        for bending, fixed, expected in (
            (((2, 0), (0, 2)), (False, False), [1]),
            (((2, 0), (0, 1)), (False, False), [0, 1]),
            (((2, 0), (0, 2)), (False, True), [0]),
        ):
            item = cable_input(bending=bending, fixed_theta=fixed)
            value = _MaterialInputs(material_model(), (), (item,)).cable[0]
            np.testing.assert_array_equal(value["free_theta"], expected)

    def test_invalid_cable_parameters(self):
        """Reject malformed open-chain topology, frames, boundary masks and stiffness."""
        for change in (
            {"particle_ids": (3, 4, 3)},
            {"particle_ids": (3.0, 4.0, 5.0)},
            {"particle_ids": (3, 4, 6)},
            {"particle_ids": ()},
            {"EA": 0},
            {"EA": True},
            {"GJ": np.inf},
            {"radius": -1},
            {"linear_density": 0},
            {"bending": ((1, 2), (2, 1))},
            {"bending": ((1, 0.2), (0, 1))},
            {"rest_theta": (0,)},
            {"rest_theta": (0, np.nan)},
            {"fixed_theta": (0, 1)},
            {"rest_director": ((1, 0, 0), (0, 1, 0))},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _MaterialInputs(material_model(), (), (cable_input(**change),))

    def test_invalid_cloth_parameters(self):
        """Reject invalid triangle selection, bending and physical density."""
        for change in (
            {"triangle_ids": ()},
            {"triangle_ids": (0, 0)},
            {"triangle_ids": (0.0,)},
            {"triangle_ids": (1,)},
            {"bending": (-1, 2)},
            {"bending": (1, 0)},
            {"thickness": 0},
            {"surface_density": np.nan},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _MaterialInputs(material_model(), (cloth_input(**change),), ())

    def test_overlap_world_mass_and_native_forces_rejected(self):
        """Reject shared owners, invalid worlds, inconsistent mass and native force conflicts."""
        for field, data in (
            ("particle_world", [0, 0, 0, 0, 1, 0]),
            ("particle_mass", [1 / 6] * 3 + [0.5, 2, 0.5]),
            ("tet_indices", [[0, 1, 2, 3]]),
            ("tri_materials", [[2, 3, 0.1, 0, 0]]),
            ("tri_activations", [0.1]),
        ):
            model = material_model()
            setattr(model, field, HostArray(data))
            with self.subTest(field=field), self.assertRaises(ValueError):
                _MaterialInputs(model, (cloth_input(),), (cable_input(),))
        with self.assertRaises(ValueError):
            _MaterialInputs(material_model(), (cloth_input(), cloth_input()), ())
        with self.assertRaises(ValueError):
            _MaterialInputs(material_model(), (), (cable_input(), cable_input()))

    def test_cloth_topology_rejected(self):
        """Reject duplicate, unoriented, nonmanifold and disconnected shell faces."""
        for triangles in (
            [[0, 1, 2], [2, 1, 0]],
            [[0, 1, 2], [0, 1, 3]],
            [[0, 1, 2], [1, 0, 3], [1, 0, 4]],
            [[0, 1, 2], [3, 4, 5]],
        ):
            model = material_model()
            model.tri_indices = HostArray(triangles)
            model.tri_materials = HostArray([[2, 3, 0, 0, 0]] * len(triangles))
            model.tri_activations = HostArray([0] * len(triangles))
            with self.subTest(triangles=triangles), self.assertRaises(ValueError):
                _MaterialInputs(model, (cloth_input(triangle_ids=tuple(range(len(triangles)))),), ())

    def test_reference_geometry_rejected(self):
        """Reject collapsed and antiparallel reference edges."""
        for positions in (
            [[2, 0, 0], [2, 0, 0], [4, 0, 0]],
            [[2, 0, 0], [3, 0, 0], [2, 0, 0]],
        ):
            model = material_model()
            model.particle_q.data[3:] = positions
            with self.assertRaises(ValueError):
                _MaterialInputs(model, (), (cable_input(),))

    def test_descriptor_is_frozen(self):
        """Keep the input descriptor immutable after authoring."""
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cable_input().EA = 20

    def test_natural_curvature_matches_independent_reference(self):
        """Match DER rest strains and retain absolute orientation on a naturally bent rod."""
        model = material_model()
        model.particle_q.data[5] = [3, 1, 0]
        item = cable_input(bending=((2, 0), (0, 2)), rest_director=((0, 0, 1), (0, 0, 1)), fixed_theta=(False, False))
        value = _MaterialInputs(model, (), (item,)).cable[0]
        oracle = CableReference(
            model.particle_q.numpy()[3:], item.rest_theta, item.rest_director, (item.EA, item.bending, item.GJ)
        )
        np.testing.assert_allclose(value["rest_curvature"], oracle.curvature, atol=1e-14)
        np.testing.assert_allclose(value["rest_twist"], oracle.twist, atol=1e-14)
        self.assertFalse(value["gauge"])
        np.testing.assert_array_equal(value["free_theta"], [0, 1])

    def test_nonfinite_fixed_mass_rejected(self):
        """Reject invalid Model mass even on a vertex with zero inverse mass."""
        for mass in (np.nan, -1):
            model = material_model()
            model.particle_mass.data[0] = mass
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                _MaterialInputs(model, (cloth_input(),), ())


if __name__ == "__main__":
    unittest.main()
