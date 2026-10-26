# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen Sharpa close-grasp scene and its staged joint targets.

Shared by the smooth-SDF qualification, verification and replay tools.
"""

import json
from pathlib import Path

import numpy as np
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube
from newton.examples.softbody.monolithic_sharpa_scene import joint_mapping
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
ARTIFACTS = WORKSPACE / "agents/integration/artifacts"
DT = 0.001
CLOSE_SCHEDULE = {"duration": 9.0, "lift_height": 0.05}


def stage_target(trajectory, time_s):
    """Sample continuous positions and same-segment velocities at physical step end."""
    if not np.isfinite(time_s) or time_s < 0 or time_s > CLOSE_SCHEDULE["duration"] + 1e-12:
        raise ValueError("Time outside the fixed nine-second schedule")
    if time_s <= 0.5:
        stage, source, rate = "prepare", 0.0, 0.0
    elif time_s <= 2.5:
        stage, source, rate = "close", time_s - 0.5, 1.0
    elif time_s <= 4.5:
        stage, source, rate = "hold", time_s - 0.5, 1.0
    elif time_s <= 5.5:
        stage, source, rate = "lift", 4.0, 0.0
    elif time_s <= 6.5:
        stage, source, rate = "hold_high", 4.0, 0.0
    elif time_s <= 8.5:
        stage, source, rate = "release", 2.0 - (time_s - 6.5), -1.0
    else:
        stage, source, rate = "observe", 0.0, 0.0
    q, v = trajectory.sample(source)
    u = min(max(time_s - 4.5, 0.0), 1.0)
    lift = CLOSE_SCHEDULE["lift_height"] * u**3 * (10 - 15 * u + 6 * u * u)
    lift_v = CLOSE_SCHEDULE["lift_height"] * 30 * u * u * (1 - u) ** 2
    return stage, q, rate * v, lift, lift_v


def targets(curve, mapping, time_s, count):
    """Match the original runner's endpoint sampling and FP32 control buffers."""
    stage, q, v, lift, lift_v = stage_target(curve, time_s)
    position, velocity = np.zeros(count, np.float32), np.zeros(count, np.float32)
    for i, name in enumerate(curve.names):
        _, dof, target = mapping[name]
        position[target], velocity[dof] = q[i], v[i]
    _, dof, target = mapping["monolithic_lift"]
    position[target], velocity[dof] = lift, lift_v
    return stage, position, velocity


def build(position):
    """Reuse frozen assets, mount, imported inertia corrections and all joint terms."""
    old = ARTIFACTS / f"p2/close-p0-capture-20260915-{position}"
    manifest = json.loads((old / "manifest.json").read_text())
    parameters = load_fixture(REPO / "newton/examples/softbody/sharpa_g1h.json")
    model, loaded = load_hand_cube(
        WORKSPACE / "assets/left_sharpa_wave",
        ARTIFACTS / "p1/pr6d-assets/hand",
        REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz",
        device="cuda:0",
        parameters=parameters,
        position=manifest["object_position_m"],
        resolution=128,
        _mount=manifest["mount"],
    )
    if model.joint_dof_count != 23 or loaded["inertia_corrections"] != manifest["inertia_corrections"]:
        raise ValueError("Frozen DoF/inertia identity changed")
    mapping = joint_mapping(model, loaded["joint_names"], carriage=True)
    dofs = [mapping[n][1] for n in loaded["joint_names"]]
    curve = ClosureTrajectory(
        WORKSPACE / "assets/trajectory/Ball_catch - Left_hand_motion_rad.csv",
        loaded["joint_names"],
        model.joint_limit_lower.numpy()[dofs],
        model.joint_limit_upper.numpy()[dofs],
        model.joint_velocity_limit.numpy()[dofs],
    )
    width, velocity = np.empty(23), np.empty(23)
    for name, (_, dof, _) in mapping.items():
        cfg = manifest["fixture"]["carriage"] if name == "monolithic_lift" else parameters["joints"][name]
        width[dof], velocity[dof] = cfg["limit_width"], cfg["friction_velocity_scale"]
    terms = SolverMonolithic.JointTerms(
        implicit_pd=True,
        limits=True,
        friction=True,
        limit_width=tuple(width),
        friction_velocity_scale=tuple(velocity),
    )
    return model, manifest, loaded, mapping, curve, terms, old
