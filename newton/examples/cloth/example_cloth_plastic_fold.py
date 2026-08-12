# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Plastic Fold
#
# Demonstrates a minimal permanent-fold workflow for VBD cloth. The left
# boundary of a checkerboard paper mesh is fixed, the right edge is pulled
# over a prescribed vertical crease, and the pulling force is then released.
# Plastic bending is enabled only on the crease edges.
#
# Command: python -m newton.examples cloth_plastic_fold
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

GRID_X = 12
GRID_Y = 8
CELL_SIZE = 0.03
PARTICLE_MASS = 0.002
CREASE_YIELD_ANGLE = 0.12
CREASE_HARDENING = 0.02
CREASE_EDGE_STIFFNESS = 60.0
CREASE_EDGE_DAMPING = 2.0
PAPER_EDGE_STIFFNESS = 300.0
PAPER_EDGE_DAMPING = 4.0
FOLD_ANGLE = 2.35
FOLD_START_TIME = 0.25
FOLD_END_TIME = 1.75
RELEASE_TIME = 2.25


@wp.kernel
def apply_fold_force(
    rest_q: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    active_flag: int,
    crease_x: float,
    fold_angle: float,
    force_strength: float,
    force_damping: float,
    particle_f: wp.array[wp.vec3],
):
    particle_index = wp.tid()
    if not particle_flags[particle_index] & active_flag:
        return

    rest_position = rest_q[particle_index]
    side = rest_position[0] - crease_x
    if side < 0.5 * CELL_SIZE:
        return

    target = wp.vec3(
        crease_x + wp.cos(fold_angle) * side,
        rest_position[1],
        wp.sin(fold_angle) * side,
    )
    particle_f[particle_index] = (
        particle_f[particle_index] + force_strength * (target - q[particle_index]) - force_damping * qd[particle_index]
    )


def _grid_index(x: int, y: int) -> int:
    return y * (GRID_X + 1) + x


def _checkerboard_mesh() -> tuple[list[wp.vec3], list[int]]:
    vertices = [wp.vec3(x * CELL_SIZE, y * CELL_SIZE, 0.0) for y in range(GRID_Y + 1) for x in range(GRID_X + 1)]
    indices = []
    for y in range(GRID_Y):
        for x in range(GRID_X):
            v0 = _grid_index(x, y)
            v1 = _grid_index(x + 1, y)
            v2 = _grid_index(x + 1, y + 1)
            v3 = _grid_index(x, y + 1)
            if (x + y) % 2 == 0:
                indices.extend((v0, v1, v3, v1, v2, v3))
            else:
                indices.extend((v0, v1, v2, v0, v2, v3))
    return vertices, indices


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 12
        self.sim_time = 0.0
        self.force_strength = 180.0
        self.force_damping = 1.2

        vertices, indices = _checkerboard_mesh()
        sheet_width = GRID_X * CELL_SIZE
        sheet_height = GRID_Y * CELL_SIZE
        crease_local_x = 0.5 * sheet_width
        crease_world_x = 0.0
        density = PARTICLE_MASS * len(vertices) / (sheet_width * sheet_height)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
        builder.add_cloth_mesh(
            pos=wp.vec3(-0.5 * sheet_width, -0.5 * sheet_height, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=density,
            tri_ke=2.0e4,
            tri_ka=2.0e4,
            tri_kd=2.0,
            edge_ke=PAPER_EDGE_STIFFNESS,
            edge_kd=PAPER_EDGE_DAMPING,
            bending_plasticity=newton.ClothPlasticity(
                yield_angle=CREASE_YIELD_ANGLE,
                hardening_modulus=CREASE_HARDENING,
            ),
            particle_radius=0.004,
        )

        for x in range(2):
            for y in range(GRID_Y + 1):
                particle_index = _grid_index(x, y)
                builder.particle_flags[particle_index] = int(builder.particle_flags[particle_index]) & ~int(
                    ParticleFlags.ACTIVE
                )

        crease_edges = []
        for edge_index, edge in enumerate(builder.edge_indices):
            vertex0 = int(edge[2])
            vertex1 = int(edge[3])
            position0 = vertices[vertex0]
            position1 = vertices[vertex1]
            if (
                abs(float(position0[0]) - crease_local_x) < 1.0e-6
                and abs(float(position1[0]) - crease_local_x) < 1.0e-6
            ):
                crease_edges.append(edge_index)

        if not crease_edges:
            raise RuntimeError("Could not find the checkerboard crease edges")

        crease_edge_set = set(crease_edges)
        for edge_index in range(len(builder.edge_plastic_mask)):
            builder.edge_plastic_mask[edge_index] = int(edge_index in crease_edge_set)
        for edge_index in crease_edges:
            builder.edge_bending_properties[edge_index] = (CREASE_EDGE_STIFFNESS, CREASE_EDGE_DAMPING)

        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 20.0
        self.model.soft_contact_mu = 0.2
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.006,
            particle_self_contact_margin=0.01,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.initial_rest_angles = self.state_0.edge_rest_angle.numpy().copy()
        self.crease_edges = np.asarray(crease_edges, dtype=np.int32)
        self.crease_x = crease_world_x
        self.active_flag = int(ParticleFlags.ACTIVE)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.45, -0.5, 0.38), -25.0, 125.0)

    def _force_scale(self) -> float:
        if self.sim_time < FOLD_START_TIME:
            return 0.0
        if self.sim_time < FOLD_END_TIME:
            return min((self.sim_time - FOLD_START_TIME) / (FOLD_END_TIME - FOLD_START_TIME), 1.0)
        if self.sim_time < RELEASE_TIME:
            return 1.0
        return 0.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            force_scale = self._force_scale()
            if force_scale > 0.0:
                wp.launch(
                    kernel=apply_fold_force,
                    dim=self.model.particle_count,
                    inputs=[
                        self.model.particle_q,
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.model.particle_flags,
                        self.active_flag,
                        self.crease_x,
                        FOLD_ANGLE,
                        self.force_strength * force_scale,
                        self.force_damping,
                    ],
                    outputs=[self.state_0.particle_f],
                    device=self.model.device,
                )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the crease retains a permanent plastic deformation after release."""
        rest_angles = self.state_0.edge_rest_angle.numpy()
        rest_angle_delta = np.abs(
            (rest_angles[self.crease_edges] - self.initial_rest_angles[self.crease_edges] + np.pi) % (2.0 * np.pi)
            - np.pi
        )
        if not np.any(rest_angle_delta > 0.03):
            raise AssertionError("the crease did not accumulate plastic bending")

        positions = self.state_0.particle_q.numpy()
        right_edge = np.array([_grid_index(GRID_X, y) for y in range(GRID_Y + 1)], dtype=np.int32)
        if abs(float(np.mean(positions[right_edge, 2]))) < 0.01:
            raise AssertionError("the folded edge returned to the flat configuration")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
