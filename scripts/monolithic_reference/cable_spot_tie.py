# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the straight-start seven-wire spot tie with removable guides in CUDA FP32."""

from collections import Counter

import numpy as np
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cable_hitch_fixture import smooth
from .cable_spot_tie_criteria import evaluate_spot_tie
from .cable_spot_tie_fixture import (
    CORD_RADIUS,
    END,
    HALF_SPAN,
    SERVICE_START,
    WIRE_NODES,
    WIRE_RADIUS,
    SpotTie,
    grip_tensions,
)
from .cloth_contact_run import publication

DT = 0.003


def add_rod(builder, points, radius, *, support_ends, director=(0.0, 1.0, 0.0)):
    density = 0.015  # Common surrogate material, as in the mandrel comparison.
    edges = np.diff(points, axis=0)
    half_mass = 0.5 * density * np.linalg.norm(edges, axis=1)
    masses = np.r_[half_mass, 0.0] + np.r_[0.0, half_mass]
    if support_ends:
        masses[[0, -1]] = 0
    ids = [
        builder.add_particle(tuple(point), (0, 0, 0), mass=float(mass), radius=radius)
        for point, mass in zip(points, masses, strict=True)
    ]
    rod = SolverMonolithic.CableInput(
        particle_ids=tuple(ids),
        EA=14.0,
        bending=np.eye(2) * 1.5e-7,
        GJ=1e-7,
        radius=radius,
        linear_density=density,
        rest_theta=np.zeros(len(edges)),
        rest_director=np.tile(director, (len(edges), 1)),
        fixed_theta=np.zeros(len(edges), dtype=bool),
    )
    return rod, ids


def add_bundle(builder):
    # Preserve the previous 25 mm envelope so the outer contact radius is comparable.
    angles = np.arange(6) * np.pi / 3 + np.pi / 2
    centers = np.vstack(([0, 0], 2 * WIRE_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))))
    z = np.linspace(-HALF_SPAN, HALF_SPAN, WIRE_NODES)
    wires = []
    for x, y in centers:
        points = np.column_stack((np.full_like(z, x), np.full_like(z, y), z))
        rod, _ = add_rod(builder, points, WIRE_RADIUS, support_ends=True)
        wires.append(rod)
    return wires


def run_spot_tie(*, device="cuda:0"):
    """Return measurements from all accepted steps of the complete physical trajectory."""
    fixture = SpotTie()
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    wires = add_bundle(builder)
    initial = fixture.initial_points.astype(np.float32).astype(float)
    cord, ids = add_rod(builder, initial, CORD_RADIUS, support_ends=False, director=(0, 0, 1))
    model = builder.finalize(device=device)
    mass = model.particle_mass.numpy().astype(float)[:, None]
    offset = ids[0]
    solver = SolverMonolithic(
        model,
        cable_inputs=[*wires, cord],
        cable_contact={
            "self": True,
            "inter": True,
            "normal_stiffness": 3e5,
            "candidate_capacity": 16384,
            "support_capacity": 12 * 16384,
            "friction_coefficient": 0.55,
            "friction_velocity_scale": 0.0005,
        },
        friction_velocity_scale=0.01,
        precision=SolverMonolithic.Precision.FP32,
        mass_mode="lumped",
        pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
        alignment_policy="frozen",
        alignment_smoothing=0.05,
        use_cuda_graph=device.startswith("cuda"),
        newton_max_iterations=16,
        publish_policy="last_accepted",
    )
    state = solver.create_state(dt=DT)
    control = solver.create_control()
    q = state.particle_q.numpy().copy()
    v = state.particle_qd.numpy().copy()
    t, h, generation = 0.0, DT, 0
    previous_target, _ = fixture.targets(t)
    damping = 1.5 * np.sqrt(100.0 * mass[offset:])
    times, positions = [t], [q]
    fixture_forces, end_grips = [0.0], [np.zeros((2, 3))]
    loads, reactions = [0.0], [0.0]
    stopped, retries = [], []
    publications = Counter()
    while t < END - 1e-10:
        if t + h >= 1.58:
            h = min(h, 0.0015)
        h = min(h, END - t)
        stamp = t + h
        target, strength = fixture.targets(stamp)
        gain = 1.0 / (1.0 + (damping * h + 100 * h * h) / mass[offset:])
        grip = (
            strength[:, None]
            * gain
            * (100 * (target - q[offset:]) + damping * ((target - previous_target) / h - v[offset:]))
        )
        grip[[0, -1]] += grip_tensions(stamp)
        load = 0.012 * smooth((stamp - SERVICE_START) / 0.04)
        force = -4.0 * mass * v
        force[offset:] += grip
        force[offset, 1] += load
        state.particle_f.assign(force)
        solver.step(state, state, control, dt=h)
        stats = solver.last_stats
        if stats.rolled_back:
            reason = stats.status.value
            retries.append((t, h, reason))
            if not np.array_equal(state.particle_q.numpy(), q) or not np.array_equal(state.particle_qd.numpy(), v):
                raise RuntimeError(f"Rollback changed the published state at {t:.6f} s")
            if (
                reason
                not in ("regularization_exhausted", "nonlinear_max_iterations", "line_search_exhausted", "nonfinite")
                or h <= DT / 32
            ):
                raise RuntimeError(f"Spot tie rolled back at {t:.6f} s: {reason}, dt={h}")
            h *= 0.5
            continue
        feedback = solver.get_contact_forces(state)
        label = publication(stats, feedback, previous_generation=generation)
        if label == "HARD_FAILURE":
            raise RuntimeError(f"Invalid contact publication at {stamp:.6f} s: {stats.status.value}")
        generation = stats.step_generation
        publications[label] += 1
        if not stats.converged:
            stopped.append((len(times), stamp, stats.status.value))
        q, v = state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()
        reaction = feedback.particle_force[offset:].sum(axis=0, dtype=float)
        if not all(np.isfinite(x).all() for x in (q, v, grip, reaction)):
            raise RuntimeError(f"Nonfinite spot tie state or force at {stamp:.6f} s")
        t = stamp
        previous_target = target
        times.append(t)
        positions.append(q)
        fixture_forces.append(float(np.linalg.norm(grip, axis=1).max()))
        end_grips.append(grip[[0, -1]])
        loads.append(load)
        reactions.append(reaction[1])
        h = min(DT, 1.3 * h)
    report = evaluate_spot_tie(
        np.asarray(times),
        np.asarray(positions),
        np.asarray(fixture_forces),
        np.asarray(end_grips),
        np.asarray(loads),
        np.asarray(reactions),
    )
    report.update(publications=dict(publications), stopped_steps=stopped, rolled_back_attempts=retries)
    return report
