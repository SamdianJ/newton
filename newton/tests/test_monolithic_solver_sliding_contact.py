# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Converge a curved rigid fingertip that slides on a soft pad."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.unittest_utils import add_function_test, get_test_devices

DT = 0.001
RADIUS = 0.002  # fingertip radius [m]
CELL = 0.04 / 12  # pad cell edge [m]
CLEARANCE = 5e-4  # initial gap between fingertip and pad [m]
DEPTH = 0.002  # commanded press below first touch [m]
SPEED = 0.1  # commanded sliding speed [m/s]
PRESS_STEPS, STEPS = 10, 30


def sliding_scene(device):
    """Build a sphere on press (z) and slide (x) prismatic drives above a soft pad with a fixed bottom."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    cells = (4, 4, 2)
    center = (cells[0] * CELL / 2 - 0.2 * CELL, cells[1] * CELL / 2 + 0.1 * CELL, cells[2] * CELL + RADIUS + CLEARANCE)
    mass, stiffness = 0.01, 2000.0
    drive = {
        "armature": 0.0,
        "damping": 0.0,
        "friction": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "target_ke": stiffness,
        "target_kd": 2.0 * math.sqrt(stiffness * 2 * mass),
        "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
    }
    press = builder.add_link(mass=mass, inertia=wp.diag(wp.vec3(1e-6)))
    slide = builder.add_link(mass=mass, inertia=wp.diag(wp.vec3(1e-6)))
    builder.add_articulation(
        [
            builder.add_joint_prismatic(
                -1, press, axis=(0, 0, 1), parent_xform=wp.transform(center, wp.quat_identity()), **drive
            ),
            builder.add_joint_prismatic(press, slide, axis=(1, 0, 0), **drive),
        ]
    )
    builder.add_shape_sphere(body=slide, radius=RADIUS, cfg=builder.ShapeConfig(density=0.0, margin=0.0002))
    young, poisson = 1e4, 0.3
    builder.add_soft_grid(
        pos=wp.vec3(0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0),
        dim_x=cells[0],
        dim_y=cells[1],
        dim_z=cells[2],
        cell_x=CELL,
        cell_y=CELL,
        cell_z=CELL,
        density=1000.0,
        k_mu=young / (2 * (1 + poisson)),
        k_lambda=young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
        k_damp=0.0,
        add_surface_mesh_edges=False,
        particle_radius=0.0002,
    )
    # Row sums of the consistent mass; the bottom layer is fixed.
    row_mass = np.zeros(len(builder.particle_q))
    for ids, pose in zip(builder.tet_indices, builder.tet_poses, strict=True):
        volume = 1 / (6 * np.linalg.det(np.asarray(pose, dtype=float).reshape(3, 3)))
        np.add.at(row_mass, list(ids), 1000.0 * volume / 4)
    for i, x in enumerate(builder.particle_q):
        builder.particle_mass[i] = 0.0 if x[2] < 1e-9 else float(row_mass[i])
    return builder.finalize(device=device)


def sliding_solver(model):
    return SolverMonolithic(
        model,
        alignment_policy="mask",
        publish_policy="current_iterate",
        friction_velocity_scale=0.01,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        normal_smoothing=1e-4,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
    )


def test_curved_sliding_converges(test, device):
    """Converge every step of a press-and-slide on a curved target within the default Newton budget.

    While the fingertip slides, the contact normal turns with the contact point.
    A tangent that leaves this out converges only linearly and runs out of iterations.
    """
    model = sliding_scene(device)
    run = sliding_solver(model)
    state, control = run.create_state(dt=DT), model.control()
    unconverged, sliding_force = [], []
    for step in range(1, STEPS + 1):
        ramp = min(1.0, step / PRESS_STEPS)
        moving = max(0, step - PRESS_STEPS)
        q = (-(CLEARANCE + DEPTH) * ramp, SPEED * moving * DT)
        qd = (-(CLEARANCE + DEPTH) / (PRESS_STEPS * DT) if step <= PRESS_STEPS else 0.0, SPEED if moving else 0.0)
        control.joint_target_q.assign(np.asarray(q, dtype=np.float32))
        control.joint_target_qd.assign(np.asarray(qd, dtype=np.float32))
        run.step(state, state, control, dt=DT)
        stats = run.last_stats
        test.assertFalse(stats.rolled_back, stats.failure_reason)
        if not stats.converged:
            unconverged.append((step, stats.status.name, stats.nonlinear_iterations, stats.convergence_ratio))
        if moving:
            sliding_force.append(np.linalg.norm(run.get_contact_forces(state).body_wrench[-1][:3]))
    # The fingertip must actually carry load while it slides.
    test.assertGreater(min(sliding_force), 0.02)
    test.assertEqual(unconverged, [])


class TestSolverSlidingContact(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestSolverSlidingContact,
        test_curved_sliding_converges.__name__,
        test_curved_sliding_converges,
        devices=[_device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
