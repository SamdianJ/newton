# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P9d PR1 small integration: capsules and a convex hull resting in contact, judged offline in FP64.

    python -m scripts.monolithic_reference.geometry_contact_integration --device cuda:0 --precision fp32 \\
        --steps 100 --output result.json

Scenes: a FREE capsule on the ground plane, a FREE capsule on a FREE box table, a capsule link (prismatic,
falling) pressing a FREE sphere, and on CUDA a FREE convex hull on the plane (the pinned E1b hull). Every
step must publish; penetration is the exact convex depth from floating_hand_criteria.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference import floating_hand_criteria as fc
from scripts.monolithic_reference.floating_hand_scenes import CONVEX_HULL_SDF, convex_hull_asset

SQRT_HALF = float(np.sqrt(0.5))
LYING = (0.0, SQRT_HALF, 0.0, SQRT_HALF)
RADIUS, HALF = 0.02, 0.04


def _free(builder, pos, quat, add_shape, density):
    pose = wp.transform(pos, wp.quat(*quat))
    body = builder.add_link(xform=pose)
    builder.add_articulation([builder.add_joint_free(body, parent_xform=pose)])
    add_shape(body, builder.ShapeConfig(density=density, margin=0.0, has_particle_collision=False))
    return body


def scenes(device):
    """Name -> builder; every scene has a static ground plane."""

    def base():
        b = newton.ModelBuilder()
        b.add_shape_plane(
            body=-1, width=0.0, length=0.0, cfg=b.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
        )
        return b

    def capsule(b):
        return lambda body, cfg: b.add_shape_capsule(body, radius=RADIUS, half_height=HALF, cfg=cfg)

    out = {}
    b = base()
    _free(b, (0.0, 0.0, RADIUS + 5e-4), LYING, capsule(b), 500.0)
    out["capsule-plane"] = b
    b = base()
    _free(
        b,
        (0.0, 0.0, 0.0205),
        (0, 0, 0, 1),
        lambda body, cfg: b.add_shape_box(body, hx=0.06, hy=0.06, hz=0.02, cfg=cfg),
        500.0,
    )
    _free(b, (0.0, 0.0, 0.04 + RADIUS + 1e-3), LYING, capsule(b), 500.0)
    out["capsule-box"] = b
    b = base()
    link = b.add_link(
        xform=wp.transform((0.0, 0.0, 0.0605), wp.quat_identity()), mass=0.2, inertia=wp.diag(wp.vec3(1e-3))
    )
    joint = b.add_joint_prismatic(
        -1,
        link,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, 0.0605), wp.quat_identity()),
        limit_ke=0.0,
        limit_kd=0.0,
    )
    b.add_articulation([joint])
    b.add_shape_capsule(
        link,
        xform=wp.transform((0, 0, 0), wp.quat(*LYING)),
        radius=0.01,
        half_height=0.03,
        cfg=b.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False),
    )
    _free(b, (0.0, 0.0, 0.0205), (0, 0, 0, 1), lambda body, cfg: b.add_shape_sphere(body, radius=0.02, cfg=cfg), 500.0)
    out["link-capsule-sphere"] = b
    if wp.get_device(device).is_cuda:
        points, indices = convex_hull_asset({"radius": RADIUS, "half_height": HALF})
        mesh = newton.Mesh(points, indices)
        mesh.build_sdf(device=device, **CONVEX_HULL_SDF)
        b = base()
        _free(
            b,
            (0.0, 0.0, RADIUS + 5e-4),
            LYING,
            lambda body, cfg: b.add_shape_convex_hull(body, mesh=mesh, cfg=cfg),
            500.0,
        )
        out["convex-plane"] = (b, points)
    return out


def run(name, builder, device, precision, steps, hull_points=None):
    from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf  # noqa: PLC0415

    model = builder.finalize(device=device)
    cuda = wp.get_device(device).is_cuda
    smooth = build_smooth_sdf(model) if cuda and (model._shape_sdf_index.numpy() >= 0).any() else None
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        contact_stiffness=1e9,
        friction_coefficient=0.5,
        publish_policy="current_iterate",
        smooth_sdf=smooth,
        precision=precision,
        pcg_mode="production",
        use_cuda_graph=cuda,
    )
    state, control = solver.create_state(dt=0.001), model.control()
    shapes = []
    for s in range(model.shape_count):
        geo = newton.GeoType(int(model.shape_type.numpy()[s]))
        record = {
            "index": s,
            "label": model.shape_label[s],
            "body": int(model.shape_body.numpy()[s]),
            "type": geo.name,
            "scale": model.shape_scale.numpy()[s].tolist(),
            "transform": model.shape_transform.numpy()[s].tolist(),
        }
        if geo == newton.GeoType.CONVEX_MESH:
            record["hull_points"] = hull_points.tolist()
        shapes.append(record)
    cores = {s["index"]: fc._core_from_record(s) for s in shapes}
    counts, worst = {"CONVERGED": 0, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0}, {"pair": 0.0, "plane": 0.0}
    for _ in range(steps):
        solver.step(state, state, control, dt=0.001)
        stats = solver.last_stats
        label = "HARD_FAILURE" if stats.rolled_back else stats.variational["publication_status"]
        counts[label] += 1
        if stats.rolled_back:
            break
        q = state.body_q.numpy().astype(np.float64)
        dynamic = [s for s in shapes if s["body"] >= 0]
        for i, a in enumerate(dynamic):
            ca = fc.world_core(cores[a["index"]], q[a["body"]])
            worst["plane"] = max(worst["plane"], 0.0, -fc.plane_signed_distance(ca))
            for b in dynamic[i + 1 :]:
                if a["body"] != b["body"]:
                    worst["pair"] = max(
                        worst["pair"], fc.penetration(ca, fc.world_core(cores[b["index"]], q[b["body"]]))
                    )
    forces = solver.get_contact_forces(state).body_wrench if not solver.last_stats.rolled_back else None
    weights = model.body_mass.numpy() * 9.81
    return {
        "counts": counts,
        "penetration_m": worst,
        "contact_fz_over_weight": None if forces is None else (forces[:, 2] / weights).tolist(),
        "body_z": state.body_q.numpy()[:, 2].tolist(),
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
    for name, entry in scenes(args.device).items():
        builder, points = entry if isinstance(entry, tuple) else (entry, None)
        result["scenes"][name] = run(name, builder, args.device, args.precision, args.steps, points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: (v["counts"], {p: round(d * 1e3, 3) for p, d in v["penetration_m"].items()})
                for k, v in result["scenes"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
