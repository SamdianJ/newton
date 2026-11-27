# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate the straight-start, three-turn cable hitch on a rigid round mandrel."""

from collections import Counter

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton

from .cable_hitch_criteria import evaluate_hitch
from .cable_hitch_fixture import CORE_RADIUS, DT, END, HALF_LENGTH, RADIUS, LockingHitch, smooth
from .cloth_contact_run import publication


def make_solver(points, device, rope_mu):
    points = points.astype(np.float32).astype(float)
    count = len(points)
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    mesh = newton.Mesh.create_cylinder(CORE_RADIUS, HALF_LENGTH, up_axis=newton.Axis.Z, segments=96)
    mesh.build_sdf(
        device=device,
        max_resolution=96,
        margin=0.001,
        narrow_band_range=(-0.004, 0.004),
        texture_format="float32",
        sign_method="winding",
    )
    builder.add_shape_mesh(-1, mesh=mesh, cfg=builder.ShapeConfig(density=0, margin=0))
    density = 0.015
    mass = np.zeros(count)
    halves = np.linalg.norm(np.diff(points, axis=0), axis=1) * density / 2
    mass[1:] += halves
    mass[:-1] += halves
    for p, m in zip(points, mass, strict=True):
        builder.add_particle(wp.vec3(p), wp.vec3(0), mass=float(m), radius=RADIUS)
    cable = SolverMonolithic.CableInput(
        particle_ids=tuple(range(count)),
        EA=14.0,
        bending=np.eye(2) * 1.5e-7,
        GJ=1.0e-7,
        radius=RADIUS,
        linear_density=density,
        rest_theta=np.zeros(count - 1),
        rest_director=np.tile([0.0, 0.0, 1.0], (count - 1, 1)),
        fixed_theta=np.zeros(count - 1, bool),
    )
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        cable_inputs=[cable],
        cable_contact={
            "self": True,
            "inter": False,
            "normal_stiffness": 3e5,
            "candidate_capacity": 2048,
            "support_capacity": 12 * 2048,
            "friction_coefficient": rope_mu,
            "friction_velocity_scale": 0.0005,
        },
        contact_stiffness=6e8,
        friction_coefficient=0.15,
        friction_velocity_scale=0.01,
        smooth_sdf=build_smooth_sdf(model),
        alignment_policy="frozen",
        box_smoothing=0.00015,
        alignment_smoothing=0.05,
        precision=SolverMonolithic.Precision.FP32,
        mass_mode="lumped",
        pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
        use_cuda_graph=True,
        newton_max_iterations=16,
        publish_policy="last_accepted",
    )
    return solver, solver.create_state(dt=DT), mass


def run_hitch(rope_mu, device="cuda:0"):
    """Feed a straight rod through the three turns, withdraw every guide, then load node zero."""
    fixture = LockingHitch()
    solver, state, mass = make_solver(fixture.initial_points, device, rope_mu)
    control = solver.create_control()
    q = state.particle_q.numpy()
    v = state.particle_qd.numpy()
    times = [0.0]
    positions = [q]
    grips = [np.zeros_like(q)]
    loads = [np.zeros(3)]
    contact_y = [0.0]
    stopped = []
    retries = []
    labels = Counter()
    generation = 0
    h = DT
    damping = 1.5 * np.sqrt(100.0 * mass)[:, None]
    previous_target, _ = fixture.targets(0.0)
    while times[-1] < END - 1e-10:
        h = min(h, END - times[-1])
        t = times[-1] + h
        target, strength = fixture.targets(t)
        gain = 1.0 / (1.0 + (damping * h + 100.0 * h * h) / mass[:, None])
        target_v = (target - previous_target) / h
        grip = strength[:, None] * gain * (100.0 * (target - q) + damping * (target_v - v))
        load = np.array([0.0, 0.012, 0.0]) * smooth((t - 1.92) / 0.04)
        forces = -mass[:, None] * 4.0 * v
        forces += grip
        forces[0] += load
        state.particle_f.assign(forces)
        solver.step(state, state, control, dt=h)
        stats = solver.last_stats
        if stats.rolled_back:
            reason = stats.status.value
            retries.append((times[-1], h, reason))
            if (
                reason not in ("regularization_exhausted", "line_search_exhausted", "nonlinear_max_iterations")
                or h <= DT / 16
            ):
                raise RuntimeError(f"Hitch rolled back at {t:.6f}s: {reason}, dt={h}")
            h *= 0.5
            continue
        contact = solver.get_contact_forces(state)
        label = publication(stats, contact, previous_generation=generation)
        if label == "HARD_FAILURE":
            raise RuntimeError(f"Invalid hitch publication at {t:.6f}s: {stats.status.value}")
        generation = stats.step_generation
        labels[label] += 1
        if label == "STOPPED_ACCEPTED":
            stopped.append((len(times), t, stats.status.value))
        q = state.particle_q.numpy()
        v = state.particle_qd.numpy()
        reaction = contact.particle_force.sum(axis=0, dtype=np.float64)
        if not all(np.isfinite(value).all() for value in (q, v, grip, reaction)):
            raise RuntimeError(f"Nonfinite hitch state or force at {t:.6f}s")
        times.append(t)
        positions.append(q)
        grips.append(grip)
        loads.append(load)
        contact_y.append(reaction[1])
        previous_target = target
        h = min(DT, 1.3 * h)
    report = evaluate_hitch(
        fixture, np.array(times), np.array(positions), np.array(grips), np.array(loads), np.array(contact_y)
    )
    report.update(publications=dict(labels), stopped_steps=stopped, rolled_back_attempts=retries)
    return report
