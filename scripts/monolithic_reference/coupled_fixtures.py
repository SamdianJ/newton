# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen PR3 T4 scenes and endpoint-sampled controls."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from newton.examples.softbody.monolithic_soft_ball import load_ball
from newton.examples.softbody.monolithic_soft_cube import load_cube
from scripts.monolithic_reference.normal_loading import command_at

DIRECTORY = Path(__file__).with_name("fixtures")
PROTOCOL = json.loads((DIRECTORY / "p3_coupled_t4_v1.json").read_text())
NORMAL = json.loads((DIRECTORY / PROTOCOL["normal_source"]).read_text())


def knots_at(time, knots):
    """Sample the frozen quintic position and its analytic velocity."""
    for (a, x), (b, y) in pairwise(knots):
        if time <= b:
            u = np.clip((time - a) / (b - a), 0.0, 1.0)
            return x + (y - x) * u**3 * (10 - 15 * u + 6 * u * u), (y - x) / (b - a) * 30 * u * u * (1 - u) ** 2
    return knots[-1][1], 0.0


def commands(case, time):
    """Return endpoint position/velocity targets and frozen-feedback gains."""
    spec = PROTOCOL["tiny_fixtures"][case]
    normal = command_at(time, NORMAL)[:2]
    if case == "curved_release":
        normal = knots_at(time, spec["normal_target_knots_s_m"])
    pairs = [normal]
    gains = [(NORMAL["drive"]["kp_n_m"], NORMAL["drive"]["kd_n_s_m"])]
    if case != "normal_loading":
        key = "target_knots_s_m" if case == "tangent_reversal" else "angle_knots_s_rad"
        pairs.append(knots_at(time, spec[key]))
        gains.append((spec["gain_ke"], spec["gain_kd"]))
    return np.asarray(pairs), np.asarray(gains)


def make_tiny(case, device):
    """Keep the base tet, root and force controller identical to the frozen recipe."""
    spec = PROTOCOL["tiny_fixtures"][case]
    rigid, soft = NORMAL["rigid"], NORMAL["soft"]
    builder = newton.ModelBuilder(gravity=tuple(NORMAL["gravity_m_s2"]))
    root = builder.add_link(mass=rigid["mass_kg"], inertia=wp.diag(wp.vec3(*rigid["inertia_kg_m2"])))
    initial = spec.get("sphere_initial_center", [0.0, 0.0, rigid["initial_plane_z_m"]])
    options = {
        "armature": 0.0,
        "damping": 0.0,
        "friction": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "actuator_mode": newton.JointTargetMode.NONE,
    }
    joints = [
        builder.add_joint_prismatic(
            -1, root, axis=(0, 0, 1), parent_xform=wp.transform(initial, wp.quat_identity()), **options
        )
    ]
    body = root
    if case != "normal_loading":
        inertia = spec.get("extra_inertia_diag", spec.get("inertia_diag"))
        body = builder.add_link(mass=spec["extra_mass"], inertia=wp.diag(wp.vec3(*inertia)))
        if case == "tangent_reversal":
            joints.append(builder.add_joint_prismatic(root, body, axis=tuple(spec["extra_prismatic_axis"]), **options))
        else:
            joints.append(builder.add_joint_revolute(root, body, axis=tuple(spec["extra_revolute_axis"]), **options))
    builder.add_articulation(joints)
    config = builder.ShapeConfig(density=0.0, margin=NORMAL["contact"]["shape_margin_m"])
    if case == "curved_release":
        builder.add_shape_sphere(body=body, radius=spec["radius"], cfg=config)
    else:
        builder.add_shape_plane(body=body, width=0.0, length=0.0, cfg=config)
    builder.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        vertices=soft["rest_positions_m"],
        indices=np.asarray(soft["tet_indices"]).ravel().tolist(),
        density=soft["density_kg_m3"],
        k_mu=soft["mu_pa"],
        k_lambda=soft["lambda_pa"],
        k_damp=0.0,
        particle_radius=soft["particle_radius_m"],
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    for node, fixed in enumerate(soft["fixed_nodes"]):
        if fixed:
            builder.particle_mass[node] = 0.0
    return builder.finalize(device=device)


def make_solver(model, case, *, mode="diagnostic", graph=False):
    """Build the frozen fixture solver without modifying physical or stopping parameters."""
    cube = case.startswith("cube")
    return SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        friction_coefficient=0.0 if case == "normal_loading" else 0.5,
        normal_smoothing=1e-4 if cube else 0.0,
        contact_stiffness=1e7,
        material_model="smith_log_stabilized" if cube else "kim_stable_no_log",
        mass_mode="consistent" if cube else "lumped",
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device) if cube else None,
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=cube),
        pcg_mode=mode,
        use_cuda_graph=graph,
    )


def set_control(control, case, time, q, qd):
    """Freeze an external force computed from beginning-state feedback, without implicit PD."""
    targets, gains = commands(case, time)
    force = (gains[:, 0] * (targets[:, 0] - q) + gains[:, 1] * (targets[:, 1] - qd)).astype(np.float32)
    control.joint_f.assign(force)
    return force.astype(float), targets


CLAMP_CONFIG = {
    "schema": "sharpa-contact-calibration/v1",
    "stiffness_scan_n_m3": [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8],
    "refinements": [1, 2, 3],
    "dt": 0.001,
    "ramp_seconds": 0.5,
    "hold_seconds": 1.0,
    "measurement_seconds": 0.2,
    "gravity": [0, 0, 0],
    "load_definition": "per-jaw load = 2 * ball_mass * 9.81 / (2 * mu); gravity represented by equivalent clamp load",
    "mu": 0.5,
    "tangential_stiffness": 2e6,
    "smoothing_width": 0.0001,
    "gap": 0.002,
    "particle_radius": 0.0002,
    "plate_mass": 0.01,
    "plate_velocity_gain": 10.0,
    "pcg_iteration_p95_budget": 64,
    "gates": {"penetration_m": 0.002, "min_det_f": 0.1, "converged_fraction": 0.99, "force_error": 0.05},
}


def make_clamp_model(ball, device, *, cube=False):
    """Clamp a fully dynamic sphere between fixed and damped force-driven planes."""
    if cube:
        mesh = load_cube(ball)
    else:
        mesh = load_ball(ball)
    b = newton.ModelBuilder(gravity=CLAMP_CONFIG["gravity"])
    base = b.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
    plate = b.add_link(mass=CLAMP_CONFIG["plate_mass"], inertia=wp.diag(wp.vec3(1e-5)))
    fixed = b.add_joint_fixed(-1, base)
    drive = b.add_joint_prismatic(
        base,
        plate,
        axis=newton.Axis.Z,
        target_ke=0,
        target_kd=CLAMP_CONFIG["plate_velocity_gain"],
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        armature=0,
        damping=0,
        friction=0,
        limit_ke=0,
        limit_kd=0,
        effort_limit=5.0,
    )
    b.add_articulation([fixed, drive])
    radius = (float(np.max(np.abs(mesh.vertices[:, 2]))) if cube else 0.02) + CLAMP_CONFIG["particle_radius"]
    cfg = b.ShapeConfig(margin=0.0)
    b.add_shape_plane(
        body=base, width=0.0, length=0.0, xform=wp.transform((0, 0, -radius), wp.quat_identity()), cfg=cfg
    )
    b.add_shape_plane(
        body=plate, width=0.0, length=0.0, xform=wp.transform((0, 0, radius), wp.quat(1.0, 0.0, 0.0, 0.0)), cfg=cfg
    )
    b.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        mesh=mesh,
        particle_radius=CLAMP_CONFIG["particle_radius"],
        add_surface_mesh_edges=False,
    )
    model = b.finalize(device=device)
    load = float(model.particle_mass.numpy().sum()) * 9.81 / CLAMP_CONFIG["mu"]
    return model, load
