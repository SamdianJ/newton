# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the approved PR2 fixtures without changing their physical inputs."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

FIXTURE_PATH = Path(__file__).parent / "fixtures/p3_pr2_v3.json"


def specification():
    return json.loads(FIXTURE_PATH.read_text())


def build_case(case, device):
    builder = newton.ModelBuilder()
    builder.gravity = tuple(case["gravity"])
    joints, parent = [], -1
    for row in case["joints"]:
        axis = np.asarray(row.get("axis", row.get("axis_direction")), dtype=float)
        axis /= np.linalg.norm(axis)
        child = builder.add_link(
            mass=row["mass"], com=tuple(row["com"]), inertia=wp.mat33(*np.diag(row["inertia_diag"]).ravel())
        )
        add_joint = builder.add_joint_revolute if row["type"] == "REVOLUTE" else builder.add_joint_prismatic
        options = {
            name: row.get(name, 0.0)
            for name in ("armature", "damping", "target_ke", "target_kd", "friction", "limit_ke", "limit_kd")
        }
        options.update(
            {
                name: row[name]
                for name in ("effort_limit", "velocity_limit", "limit_lower", "limit_upper")
                if name in row
            }
        )
        if row.get("target_ke", 0.0) or row.get("target_kd", 0.0):
            options["actuator_mode"] = newton.JointTargetMode.POSITION_VELOCITY
        joints.append(
            add_joint(
                parent,
                child,
                axis=tuple(axis),
                parent_xform=wp.transform(tuple(row.get("parent_translation", (0.0, 0.0, 0.0))), wp.quat_identity()),
                **options,
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.joint_q[:] = [row["q0"] for row in case["joints"]]
    builder.joint_qd[:] = [row["qd0"] for row in case["joints"]]
    model = builder.finalize(device=device)
    driven = case["id"] == "driven_terms"
    terms = SolverMonolithic.JointTerms(
        implicit_pd=driven,
        limits=driven,
        friction=driven,
        limit_width=tuple(row["limit_width"] for row in case["joints"]) if driven else None,
        friction_velocity_scale=tuple(row["friction_velocity_scale"] for row in case["joints"]) if driven else None,
    )
    return model, terms


def target(case, time):
    """Sample the explicitly declared piecewise-linear target and outgoing slope."""
    n = len(case["joints"])
    if "target_knots" not in case:
        return np.zeros(n), np.zeros(n)
    knots = np.asarray(case["target_knots"], dtype=float)
    if time <= 0:
        return np.full(n, knots[0, 1]), np.zeros(n)
    if time >= knots[-1, 0]:
        return np.full(n, knots[-1, 1]), np.zeros(n)
    i = np.searchsorted(knots[:, 0], time, side="right") - 1
    slope = (knots[i + 1, 1] - knots[i, 1]) / (knots[i + 1, 0] - knots[i, 0])
    return np.full(n, knots[i, 1] + (time - knots[i, 0]) * slope), np.full(n, slope)


def initialize(solver, case, *, dt):
    state = solver._create_articulation_state(dt=dt)
    state.joint_q.assign([row["q0"] for row in case["joints"]])
    state.joint_qd.assign([row["qd0"] for row in case["joints"]])
    solver._reset_articulation_state(state, dt=dt)
    force = np.zeros((solver.model.body_count, 6), dtype=np.float32)
    force[:, :3] = case.get("world_COM_force", (0.0, 0.0, 0.0))
    state.body_f.assign(force)
    return state
