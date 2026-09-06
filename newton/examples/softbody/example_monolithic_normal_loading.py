# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compress an anchored tetrahedron with an experimental monolithic solver.

Run ``python -m newton.examples monolithic_normal_loading --viewer null --test``.
The 100 frames each contain ten 1 ms substeps; the final target is held afterward.
"""

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 0.01
        self.sim_dt = 0.001
        self.sim_substeps = 10
        self.substeps = 0
        self._test = args.test

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=0.1, inertia=wp.diag(wp.vec3(0.001)))
        joint = builder.add_joint_prismatic(
            -1,
            body,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform((0.0, 0.0, -0.005), wp.quat_identity()),
            armature=0.0,
            damping=0.0,
            friction=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            target_ke=0.0,
            target_kd=0.0,
            actuator_mode=newton.JointTargetMode.NONE,
        )
        builder.add_articulation([joint])
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=builder.ShapeConfig(density=0.0, margin=0.0))
        builder.add_soft_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=[(-0.02, -0.015, 0.0), (0.02, -0.015, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.03)],
            indices=[0, 1, 2, 3],
            density=1000.0,
            k_mu=3846.15380859375,
            k_lambda=5769.23095703125,
            k_damp=0.0,
            particle_radius=0.001,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
        builder.particle_mass[3] = 0.0
        self.model = builder.finalize()
        self.model.request_contact_attributes("force")
        self.state = self.model.state()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.control = self.model.control()
        pipeline = MonolithicCollisionPipeline(self.model, soft_contact_gap=0.002)
        self.solver = SolverMonolithic(self.model, collision_pipeline=pipeline, contact_stiffness=1.0e7)
        self.rest_positions = self.state.particle_q.numpy().copy()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.09, -0.12, 0.07), pitch=-20.0, yaw=125.0)

    @staticmethod
    def _command(time_s):
        if time_s < 0.2:
            start, duration, q0, q1 = 0.0, 0.2, 0.0, 0.003
        elif time_s < 0.7:
            start, duration, q0, q1 = 0.2, 0.5, 0.003, 0.012
        else:
            return 0.012, 0.0
        t = min(max((time_s - start) / duration, 0.0), 1.0)
        return (
            q0 + (q1 - q0) * t**3 * (10.0 - 15.0 * t + 6.0 * t * t),
            (q1 - q0) * 30.0 * t * t * (1.0 - t) ** 2 / duration,
        )

    def step(self):
        for _ in range(self.sim_substeps):
            target, velocity = self._command((self.substeps + 1) * self.sim_dt)
            q, qd = float(self.state.joint_q.numpy()[0]), float(self.state.joint_qd.numpy()[0])
            # Freeze this external PD force for the complete implicit substep.
            self.control.joint_f.assign(np.asarray([2000.0 * (target - q) + 30.0 * (velocity - qd)], dtype=np.float32))
            self.solver.step(self.state, self.state, self.control, None, self.sim_dt)
            self.substeps += 1
            if self._test:
                self.test_post_step()
        self.sim_time = self.substeps * self.sim_dt

    def test_post_step(self):
        """Check convergence, physical guards and the fixed apex each substep."""
        stats = self.solver.last_stats
        assert stats.converged and not stats.rolled_back, stats
        assert stats.min_det_f >= 0.2 and stats.max_penetration <= 0.001, stats
        ratios = (stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x)
        assert all(np.isfinite(value) and value <= 1.0 for value in ratios), ratios
        positions = self.state.particle_q.numpy()
        assert np.isfinite(positions).all()
        np.testing.assert_array_equal(positions[3], self.rest_positions[3])
        np.testing.assert_array_equal(self.state.particle_qd.numpy()[3], np.zeros(3))

    def test_final(self):
        """Require a complete loading trajectory and measurable soft deformation."""
        assert self.substeps >= 1000, "Run at least 100 frames for the full loading test"
        positions = self.state.particle_q.numpy()
        initial_distance = np.linalg.norm(self.rest_positions[3] - self.rest_positions[:3].mean(axis=0))
        final_distance = np.linalg.norm(positions[3] - positions[:3].mean(axis=0))
        assert initial_distance - final_distance >= 0.005
        assert self.solver.last_stats.active_sample_count > 0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
