# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the producer step contract shared by every monolithic physical term."""

import unittest

import numpy as np

from newton._src.solvers.monolithic.producers import Producer, validate_row_partition
from newton.tests.test_monolithic_material_state import scene as material_scene
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver


class _Rows:
    def __init__(self, name, rows):
        self.name, self.rows = name, np.asarray(rows, dtype=np.int32)


class TestProducerRows(unittest.TestCase):
    def test_rows_partition_the_layout(self):
        validate_row_partition([_Rows("a", [0, 1]), _Rows("b", [2])], 3)

    def test_overlap_and_gap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_row_partition([_Rows("a", [0, 1]), _Rows("b", [1, 2])], 3)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_row_partition([_Rows("a", [0, 0, 1])], 2)
        with self.assertRaisesRegex(ValueError, "uncovered"):
            validate_row_partition([_Rows("a", [0])], 2)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_row_partition([_Rows("a", [0, 2])], 2)

    def test_protocol_lists_the_step_contract(self):
        for name in ("begin_step", "candidate", "evaluate", "assemble", "scale", "displacement", "domain", "publish"):
            self.assertTrue(hasattr(Producer, name), name)


class TestSolverProducers(unittest.TestCase):
    def assert_partition(self, solver, names):
        self.assertEqual([p.name for p in solver._producers], names)
        validate_row_partition([p for p in solver._producers if len(p.rows)], solver._layout.scalar_dof_count)

    def test_robot_scene_producers(self):
        solver = contact_solver(contact_scene("cpu", "chain"))
        self.assert_partition(solver, ["articulation", "rigid", "tet", "contact"])
        self.assertTrue(all(p.capturable for p in solver._producers))

    def test_material_scene_producers(self):
        names = ["articulation", "rigid", "tet", "cloth_0", "cable_0", "cable_1"]
        _, solver = material_scene(mechanical=True, rigid=True, tet=True, cloth=True, cables=2)
        # This scene has no collision shape, so it has no contact producer.
        self.assert_partition(solver, names)
        _, solver = material_scene(mechanical=True, rigid=True, tet=True, cloth=True, cables=2, shape_radius=0.5)
        self.assert_partition(solver, [*names, "contact"])
        policies = {p.name: (p.capturable, p.scale_policy) for p in solver._producers}
        self.assertEqual(policies["cloth_0"], (False, "coordinate"))
        self.assertEqual(policies["cable_1"], (False, "coordinate"))
        self.assertEqual(policies["tet"], (True, "diagonal"))


if __name__ == "__main__":
    unittest.main()
