# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared numerical fixture for the experimental monolithic solver tests."""

from dataclasses import dataclass

import warp as wp

import newton
from newton._src.usd.utils import _deformable_lame_parameters


@dataclass(frozen=True, slots=True)
class TinyFixtureSpec:
    """Specify SI inputs and scalar ordering [q_revolute, q_prismatic, x0, ..., x3]."""

    fixture_id: str = "monolithic_tiny_rev_pris_one_tet_v1"
    rest_positions: tuple = ((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0), (0.0, 0.0, 0.02))
    boundary_faces: tuple = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    q0: tuple = (0.2, 0.01)
    qd0: tuple = (0.1, -0.02)
    density: float = 1000.0
    particle_radius: float = 0.001
    scalar_dof_count: int = 14


@dataclass(frozen=True, slots=True)
class TinyFixture:
    """Hold independent initialized states and the fixture's model and control."""

    model: newton.Model
    state: newton.State
    state_next: newton.State
    control: newton.Control
    spec: TinyFixtureSpec


def build_tiny_cpu_fixture(*, device="cpu") -> TinyFixture:
    """Build the same frozen two-joint, one-tet data on CPU or a requested device."""
    spec = TinyFixtureSpec()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    link0 = builder.add_link(mass=1.0, com=(0.05, 0.0, 0.0), inertia=wp.mat33(0.002, 0, 0, 0, 0.003, 0, 0, 0, 0.004))
    link1 = builder.add_link(mass=0.5, com=(0.02, 0.0, 0.0), inertia=wp.mat33(0.001, 0, 0, 0, 0.0015, 0, 0, 0, 0.002))
    passive = {
        "armature": 0.0,
        "damping": 0.0,
        "friction": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "actuator_mode": newton.JointTargetMode.NONE,
    }
    revolute = builder.add_joint_revolute(-1, link0, axis=newton.Axis.Z, **passive)
    prismatic = builder.add_joint_prismatic(
        link0, link1, axis=newton.Axis.X, parent_xform=wp.transform((0.1, 0.0, 0.0), wp.quat_identity()), **passive
    )
    builder.add_articulation([revolute, prismatic])
    builder.joint_q[:] = spec.q0
    builder.joint_qd[:] = spec.qd0
    k_mu, k_lambda = _deformable_lame_parameters(10000.0, 0.3)
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=list(spec.rest_positions),
        indices=[0, 1, 2, 3],
        density=spec.density,
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=0.0,
        particle_radius=spec.particle_radius,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
        add_surface_mesh_edges=True,
    )
    model = builder.finalize(device=device)
    state, state_next = model.state(), model.state()
    for initial in (state, state_next):
        newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    return TinyFixture(model, state, state_next, model.control(), spec)
