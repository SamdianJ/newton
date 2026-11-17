# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P9d PR2 small integration: an undriven floating box chain, measured offline in FP64.

    python -m scripts.monolithic_reference.floating_base_integration --device cuda:0 --precision fp32 \\
        --steps 100 --output result.json

Scenes (contact stiffness 1e9 as in the E2E recipes): the chain tumbling in zero gravity (discrete linear and angular momentum), the same chain in
free flight under gravity (|P(t) - P(0) - M g t|), and the chain dropped on the ground plane (every step
must publish; penetration is the exact convex depth from floating_hand_criteria). Momenta use the
native discrete link velocities of the coupled history, the rates the scheme actually conserves.
Newton and PCG iteration counts are recorded as the observation start for later root drives.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference import floating_hand_criteria as fc

GRAVITY = np.array([0.0, 0.0, -9.81])
HALF = (0.03, 0.02, 0.02)
POSE = wp.transform((0.0, 0.0, 0.07), wp.quat_rpy(0.3, 0.2, 0.1))


def chain(gravity, *, ground):
    """Three boxes on two revolute joints under a FREE root, with initial root twist and joint rates."""
    builder = newton.ModelBuilder(gravity=tuple(gravity))
    cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
    if ground:
        builder.add_shape_plane(body=-1, width=0.0, length=0.0, cfg=cfg)
    links = []
    for i in range(3):
        offset = wp.transform((0.08 * i, 0.0, 0.0), wp.quat_identity())
        links.append(builder.add_link(xform=POSE * offset, mass=0.3, inertia=wp.diag(wp.vec3(4e-5, 1.3e-4, 1.3e-4))))
        builder.add_shape_box(links[-1], hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=cfg)
    joints = [builder.add_joint_free(links[0], parent_xform=POSE)]
    for i in (1, 2):
        joints.append(
            builder.add_joint_revolute(
                links[i - 1],
                links[i],
                parent_xform=wp.transform((0.04, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((-0.04, 0.0, 0.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                limit_ke=0.0,
                limit_kd=0.0,
            )
        )
    builder.add_articulation(joints)
    return builder


def scenes():
    return {
        "tumble-zero-g": (chain((0.0, 0.0, 0.0), ground=False), True, False),
        "free-flight": (chain(GRAVITY, ground=False), True, False),
        "drop-plane": (chain(GRAVITY, ground=True), False, True),
    }


def _momenta(model, state, k):
    history = state._monolithic_coupled
    k.evaluate_state(state)
    mass = model.body_mass.numpy().astype(np.float64)
    com = k.com.numpy().astype(np.float64)
    rotations = k.rotations.numpy().astype(np.float64)
    v = history.link_v_com.numpy().astype(np.float64)
    w = history.link_omega.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64)
    linear = mass @ v
    angular = sum(
        m * np.cross(c, u) + r @ i @ r.T @ o
        for m, c, u, r, i, o in zip(mass, com, v, rotations, inertia, w, strict=True)
    )
    return linear, angular


def run(name, builder, twist, ground, device, precision, steps):
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    model = builder.finalize(device=device)
    if twist:
        qd = model.joint_qd.numpy()
        qd[:6] = (0.2, -0.1, 0.3, 1.5, -2.0, 3.0)
        qd[6:8] = (4.0, -3.0)
        model.joint_qd.assign(qd)
    cuda = wp.get_device(device).is_cuda
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        contact_stiffness=1e9,
        friction_coefficient=0.5,
        publish_policy="current_iterate",
        precision=precision,
        pcg_mode="production",
        use_cuda_graph=cuda,
    )
    h = 0.001
    state, control = solver.create_state(dt=h), solver.create_control()
    k = solver._kinematics
    shapes = [
        {
            "index": s,
            "body": int(model.shape_body.numpy()[s]),
            "type": newton.GeoType(int(model.shape_type.numpy()[s])).name,
            "scale": model.shape_scale.numpy()[s].tolist(),
            "transform": model.shape_transform.numpy()[s].tolist(),
        }
        for s in range(model.shape_count)
    ]
    cores = {s["index"]: fc._core_from_record(s) for s in shapes if s["body"] >= 0}
    total = float(model.body_mass.numpy().sum())
    p0, l0 = _momenta(model, state, k)
    counts = {"CONVERGED": 0, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0}
    newton_iterations = linear_iterations = 0
    worst = {"linear_momentum": 0.0, "angular_momentum": 0.0, "free_flight_impulse": 0.0, "plane": 0.0}
    for step in range(steps):
        solver.step(state, state, control, dt=h)
        stats = solver.last_stats
        label = "HARD_FAILURE" if stats.rolled_back else stats.variational["publication_status"]
        counts[label] += 1
        if stats.rolled_back:
            break
        newton_iterations += stats.nonlinear_iterations
        linear_iterations += stats.linear_iterations
        p, angular = _momenta(model, state, k)
        t = (step + 1) * h
        worst["linear_momentum"] = max(worst["linear_momentum"], float(np.abs(p - p0).max()))
        worst["angular_momentum"] = max(worst["angular_momentum"], float(np.abs(angular - l0).max()))
        worst["free_flight_impulse"] = max(
            worst["free_flight_impulse"], float(np.abs(p - p0 - total * np.asarray(model.gravity.numpy()[0]) * t).max())
        )
        q = state.body_q.numpy().astype(np.float64)
        for index, core in cores.items() if ground else ():
            body = shapes[index]["body"]
            worst["plane"] = max(worst["plane"], 0.0, -fc.plane_signed_distance(fc.world_core(core, q[body])))
    return {
        "counts": counts,
        "newton_iterations": newton_iterations,
        "linear_iterations": linear_iterations,
        "initial_linear_momentum": p0.tolist(),
        "initial_angular_momentum": l0.tolist(),
        "worst": worst,
        "root_com": state._monolithic_coupled.floating_com.numpy()[0].tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import subprocess  # noqa: PLC0415

    source = Path(__file__).resolve().parents[2]
    result = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip(),
        "device": args.device,
        "precision": args.precision,
        "steps": args.steps,
        "scenes": {},
    }
    for name, (builder, twist, ground) in scenes().items():
        result["scenes"][name] = run(name, builder, twist, ground, args.device, args.precision, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: (v["counts"], v["worst"]) for k, v in result["scenes"].items()}))


if __name__ == "__main__":
    main()
