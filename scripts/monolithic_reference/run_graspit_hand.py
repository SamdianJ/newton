# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run fixed-palm motion or a static-sphere contact smoke check, not a grasp test."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton
from scripts.monolithic_reference.graspit_hand import build_hand
from scripts.monolithic_reference.retained_assets import sha

DT, STEPS = 0.005, 600


def transform(points, pose):
    """Transform points with an xyzw quaternion without lowering their precision."""
    q = np.asarray(pose[3:], dtype=float)
    t = 2 * np.cross(q[:3], points)
    return points + q[3] * t + np.cross(q[:3], t) + pose[:3]


def shape_vertices(model, poses, shape):
    """Return a mesh's world vertices, including its fixed shape frame."""
    points = model.shape_source[shape].vertices * model.shape_scale.numpy()[shape]
    points = transform(points, model.shape_transform.numpy()[shape])
    body = int(model.shape_body.numpy()[shape])
    return transform(points, poses[body]) if body >= 0 else points


def closed_pose(model, scenario):
    """Use modest angles inside every source limit; move only index for contact."""
    target = np.zeros(20)
    for j, name in enumerate(model.joint_label):
        if model.joint_type.numpy()[j] != newton.JointType.REVOLUTE:
            continue
        i = int(model.joint_q_start.numpy()[j])
        if scenario == "contact" and "index" not in name:
            continue
        target[i] = 0.06 if "palm_" in name else 0.3
    return target


def run(asset, output, *, device, precision, scenario):
    """Record each publication and apply motion/contact gates to all frames."""
    output.mkdir(parents=True, exist_ok=True)
    contact = wp.get_device(device).is_cuda
    if scenario == "contact" and not contact:
        raise ValueError("Contact smoke check requires CUDA")
    builder, manifest = build_hand(asset, device=device, contact=contact)
    probe = None
    if scenario == "contact":
        reference = builder.finalize(device=device)
        state = reference.state()
        state.joint_q.assign(closed_pose(reference, scenario))
        newton.eval_fk(reference, state.joint_q, state.joint_qd, state)
        body = reference.body_label.index("HumanHand20DOF/index3")
        center = transform(reference.body_com.numpy()[body], state.body_q.numpy()[body])
        probe = {"center_m": center.tolist(), "radius_m": 0.006}
        builder.add_shape_sphere(
            -1,
            radius=probe["radius_m"],
            xform=wp.transform(center, wp.quat_identity()),
            cfg=builder.ShapeConfig(density=0, margin=0, has_particle_collision=False),
        )
    model = builder.finalize(device=device)
    smooth = build_smooth_sdf(model) if contact else None
    solver = SolverMonolithic(
        model,
        precision=precision,
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=True, limits=True, limit_width=np.full(20, 0.01)),
        contact_stiffness=1e7,
        friction_coefficient=0,
        friction_velocity_scale=0.001,
        normal_smoothing=1e-4,
        alignment_policy="frozen",
        alignment_smoothing=0.1,
        publish_policy="current_iterate",
        use_cuda_graph=contact,
        pcg_mode="production",
        newton_max_iterations=40,
        line_search_max_iterations=24,
        smooth_sdf=smooth,
    )
    state, control = solver.create_state(dt=DT), solver.create_control()
    target = closed_pose(model, scenario)
    collision_shapes = [
        i
        for i, f in enumerate(model.shape_flags.numpy())
        if f & int(newton.ShapeFlags.COLLIDE_SHAPES) and model.shape_type.numpy()[i] == newton.GeoType.MESH
    ]
    rows, qs, poses = [], [state.joint_q.numpy().copy()], [state.body_q.numpy().copy()]
    failure = None
    with (output / "steps.jsonl").open("w") as stream:
        for step in range(1, STEPS + 1):
            t = step * DT
            u = np.clip(t if t <= 1.5 else 2.5 - t, 0, 1)
            blend = u * u * (3 - 2 * u)
            rate = 6 * u * (1 - u) * (1 if t <= 1.5 else -1)
            control.joint_target_q.assign(blend * target)
            control.joint_target_qd.assign(rate * target)
            try:
                solver.step(state, state, control, dt=DT)
                stats = solver.last_stats
                publication = stats.variational.get("publication_status")
                if stats.rolled_back or publication not in ("CONVERGED", "STOPPED_ACCEPTED"):
                    raise RuntimeError(f"{publication}: {stats.failure_reason}")
                q, qd, body_q = state.joint_q.numpy(), state.joint_qd.numpy(), state.body_q.numpy()
                forces = solver.get_contact_forces(state)
                if not forces.force_valid or not all(np.isfinite(a).all() for a in (q, qd, body_q, forces.body_wrench)):
                    raise RuntimeError("Invalid state or force publication")
                depth = 0.0
                if probe:
                    import trimesh

                    point = np.asarray(probe["center_m"])[None, :]
                    for s in collision_shapes:
                        mesh = trimesh.Trimesh(
                            shape_vertices(model, body_q, s),
                            model.shape_source[s].indices.reshape(-1, 3),
                            process=False,
                        )
                        distance = float(trimesh.proximity.closest_point_naive(mesh, point)[1][0])
                        inside = np.max(np.einsum("ij,ij->i", point - mesh.triangles_center, mesh.face_normals)) <= 0
                        depth = max(depth, probe["radius_m"] + distance if inside else probe["radius_m"] - distance)
                row = {
                    "step": step,
                    "publication": publication,
                    "stop_reason": stats.failure_reason,
                    "newton_iterations": stats.nonlinear_iterations,
                    "convergence_ratio": stats.convergence_ratio,
                    "tracking_error_rad": float(np.max(np.abs(q - blend * target))),
                    "max_speed_rad_s": float(np.max(np.abs(qd))),
                    "penetration_m": depth,
                    "force_N": float(np.linalg.norm(forces.body_wrench[:, :3], axis=1).max()),
                    "limit_violation_rad": float(
                        max(0, np.max(model.joint_limit_lower.numpy() - q), np.max(q - model.joint_limit_upper.numpy()))
                    ),
                }
                rows.append(row)
                qs.append(q.copy())
                poses.append(body_q.copy())
                stream.write(json.dumps(row) + "\n")
            except Exception as exc:
                failure = {"step": step, "error": f"{type(exc).__name__}: {exc}"}
                break
    counts = {k: sum(r["publication"] == k for r in rows) for k in ("CONVERGED", "STOPPED_ACCEPTED")}
    counts["HARD_FAILURE"] = int(failure is not None)
    gates = {
        "complete": len(rows) == STEPS and failure is None,
        "joint_limits": bool(rows) and max(r["limit_violation_rad"] for r in rows) < 0.01,
        "speed": bool(rows) and max(r["max_speed_rad_s"] for r in rows) < 2.0,
        "release_tracking": len(rows) == STEPS and rows[-1]["tracking_error_rad"] < 0.03,
    }
    if probe:
        gates.update(
            contact_response=bool(rows) and max(r["force_N"] for r in rows) > 0.01,
            penetration=bool(rows) and max(r["penetration_m"] for r in rows) < 0.002,
            released=len(rows) == STEPS and max(r["force_N"] for r in rows[-20:]) < 0.01,
        )
    else:
        hold = rows[249:300]
        gates["closed_tracking"] = len(hold) == 51 and max(r["tracking_error_rad"] for r in hold) < 0.03
    source = Path(__file__).resolve().parents[2]
    summary = {
        "passed": all(gates.values()),
        "scenario": scenario,
        "precision": precision,
        "device": str(device),
        "counts": counts,
        "gates": gates,
        "failure": failure,
        "stopped_steps": [
            {"step": r["step"], "reason": r["stop_reason"]} for r in rows if r["publication"] == "STOPPED_ACCEPTED"
        ],
        "body_count": model.body_count,
        "dof_count": model.joint_dof_count,
        "shape_count": model.shape_count,
        "dt_s": DT,
        "requested_steps": STEPS,
        "probe": probe,
        "gravity_m_s2": [0, 0, 0],
        "max_penetration_m": max((r["penetration_m"] for r in rows), default=None),
        "max_force_N": max((r["force_N"] for r in rows), default=None),
        "max_speed_rad_s": max((r["max_speed_rad_s"] for r in rows), default=None),
        "final_tracking_error_rad": rows[-1]["tracking_error_rad"] if rows else None,
        "asset_manifest_sha256": sha(Path(asset) / "manifest.json"),
        "asset_mass_kg": manifest["total_mass_kg"],
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_files": {
            str(p.relative_to(source)): sha(p) for p in (Path(__file__), Path(__file__).with_name("graspit_hand.py"))
        },
        "newton_import": str(Path(newton.__file__).resolve()),
        "smooth_sdf": smooth.build_metadata if smooth else None,
        "scope": "fixed-palm zero-gravity smoke check; no self-contact, grasp or admittance qualification",
    }
    np.savez_compressed(output / "trajectory.npz", joint_q=np.array(qs), body_q=np.array(poses), target=target)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("smooth_sdf", "stopped_steps")}, indent=2))
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--scenario", choices=("motion", "contact"), default="motion")
    args = parser.parse_args()
    raise SystemExit(
        0
        if run(args.asset_dir, args.output, device=args.device, precision=args.precision, scenario=args.scenario)
        else 1
    )
