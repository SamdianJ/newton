# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the producer step contract shared by every monolithic physical term."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.producers import Producer, validate_row_partition
from newton.solvers.experimental.monolithic import SolverMonolithic
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


def free_box_scene(device):
    """One free box resting just above a plane, with no articulation or tet auxiliaries."""
    half, mass, gap = 0.02, 0.01, 0.0005
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    cfg = builder.ShapeConfig(density=0, margin=0, has_particle_collision=False)
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, half + gap), wp.quat_identity()),
        mass=mass,
        inertia=wp.diag(wp.vec3(mass * (2 * half) ** 2 / 6)),
        label="box",
    )
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
    builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg)
    return builder.finalize(device=device)


class TestSolverWithoutTet(unittest.TestCase):
    def test_free_box_without_auxiliaries_steps(self):
        model = free_box_scene("cpu")
        solver = SolverMonolithic(model, friction_velocity_scale=0.01, publish_policy="strict")
        state = solver.create_state(dt=0.001)
        for _ in range(3):
            solver.step(state, state, solver.create_control(), dt=0.001)
            self.assertTrue(solver.last_stats.converged, solver.last_stats)
        self.assertEqual([p.name for p in solver._producers], ["rigid", "contact"])
        # The box settles onto the plane instead of falling through it.
        com = state._monolithic_coupled.rigid_com.numpy()[0]
        self.assertGreater(com[2], 0.02 - 0.002)


if __name__ == "__main__":
    unittest.main()
