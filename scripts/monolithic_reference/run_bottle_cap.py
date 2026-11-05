# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Torque-driven cap unscrewing, tool removal and free-flight check on a fixed bottle."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from scripts.monolithic_reference.bottle_cap import CapThread, cap_geometry, rotation
from scripts.monolithic_reference.plastic_bottle import bottle_mesh
from scripts.monolithic_reference.retained_assets import sha

DT = 0.001


def build_scene(device, precision, *, dt=DT):
    """Create one FREE cap with physical cup inertia; the bottle is a fixed fixture."""
    vertices, faces, mass, inertia, seated_y = cap_geometry()
    builder = newton.ModelBuilder(gravity=(0, -9.81, 0))
    body = builder.add_link(
        xform=wp.transform((0, seated_y, 0), wp.quat_identity()),
        mass=mass,
        inertia=wp.diag(wp.vec3(inertia)),
        label="removable_cap",
    )
    builder.add_articulation([builder.add_joint_free(body)])
    # Thread forces are the only fixture interaction. Geometry is independently
    # checked below; arbitrary cap/bottle impacts are outside this mechanism demo.
    builder.add_shape_mesh(
        body,
        mesh=newton.Mesh(vertices=vertices, indices=faces.ravel(), compute_inertia=False),
        cfg=builder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False),
    )
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        precision=precision,
        publish_policy="current_iterate",
        use_cuda_graph=wp.get_device(device).is_cuda,
        pcg_mode="production",
    )
    return model, solver, solver.create_state(dt=dt), solver.create_control(), CapThread(seated_y)


def tool_wrench(time, release_time, velocity, mass):
    """Apply finite torque, then finite translational forces; never assign cap poses."""
    wrench = np.zeros(6)
    if release_time is None:
        if time > 0.2:
            wrench[4] = np.clip(0.00015 * (4.0 - velocity[4]), -0.0008, 0.0008)
        return "unscrew" if time > 0.2 else "seated", wrench
    elapsed = time - release_time
    if elapsed >= 1.6:
        return "free_flight", wrench
    target = np.array([0, 0.05, 0]) if elapsed < 0.8 else np.array([0.08, 0, 0])
    wrench[:3] = np.clip(0.2 * (target - velocity[:3]), -0.08, 0.08)
    wrench[1] += mass * 9.81
    wrench[3:] = np.clip(-0.0001 * velocity[3:], -0.0008, 0.0008)
    return "lift_cap" if elapsed < 0.8 else "move_aside", wrench


def sample_penetration(pose, bottle_samples, cap_vertices):
    """Check original bottle samples against the hollow cup, and cap against the table."""
    r = rotation(pose[3:])
    local = (bottle_samples - pose[:3]) @ r
    radius = np.linalg.norm(local[:, [0, 2]], axis=1)
    bottom, top = cap_vertices[:, 1].min(), cap_vertices[:, 1].max()
    # Signed interiors of the annular wall and solid top disk, including 0.125 mm shell thickness.
    wall = np.minimum.reduce((radius - 0.012, 0.014 - radius, local[:, 1] - bottom, top - local[:, 1]))
    lid = np.minimum.reduce((0.014 - radius, local[:, 1] - (top - 0.002), top - local[:, 1]))
    table = -0.080125 - (cap_vertices @ r.T + pose[:3])[:, 1].min()
    return max(0.0, float(np.maximum(wall, lid).max() + 0.000125), float(table))


def run(args):
    """Record every published step and keep the solver/controller continuation state."""
    if not np.isfinite(args.dt) or not 0 < args.dt <= DT:
        raise ValueError("Explicit fixture forces require 0 < dt <= 1 ms; smaller steps need separate validation")
    dt = args.dt
    max_steps = args.max_steps if args.max_steps is not None else round(6 / dt)
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "source_files": {
            p.name: sha(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("bottle_cap.py"),
                Path(__file__).with_name("plastic_bottle.py"),
            )
        },
        "precision": args.precision,
        "device": args.device,
        "dt_s": dt,
        "max_steps": max_steps,
        "scope": "Fixed bottle fixture, explicit compliant thread, finite tool wrench, no hand or resolved teeth/impacts",
        "limits": {"thread_error_m": 0.0003, "sample_penetration_m": 0.0002, "free_velocity_error_m_s": 0.0001},
    }
    (args.output / "input.json").write_text(json.dumps(identity, indent=2) + "\n")
    _model, solver, state, control, thread = build_scene(args.device, args.precision, dt=dt)
    cap_vertices, cap_faces, mass, _, _ = cap_geometry()
    bottle_vertices, bottle_faces = bottle_mesh()
    samples = np.concatenate((bottle_vertices, bottle_vertices[bottle_faces].mean(1)))
    poses, rows, release_time, failure = [state.body_q.numpy()[0]], [], None, None
    free_start, free_steps = None, 0
    hard_failures = 0
    with (args.output / "steps.jsonl").open("w") as stream:
        for step in range(1, max_steps + 1):
            pose, velocity = state.body_q.numpy()[0].astype(float), state.body_qd.numpy()[0].astype(float)
            time = (step - 1) * dt
            phase, tool = tool_wrench(time, release_time, velocity, mass)
            reaction = thread.wrench(pose, velocity)
            if phase == "free_flight" and free_start is None:
                free_start = (pose.copy(), velocity.copy())
            state.body_f.assign(np.array([tool + reaction], dtype=np.float32))
            published = False
            try:
                solver.step(state, state, control, dt=dt)
                stats = solver.last_stats
                publication = stats.variational.get("publication_status")
                if stats.rolled_back or publication not in ("CONVERGED", "STOPPED_ACCEPTED"):
                    raise RuntimeError(f"{publication}: {stats.failure_reason}")
                pose, velocity = state.body_q.numpy()[0].astype(float), state.body_qd.numpy()[0].astype(float)
                if not np.isfinite(np.r_[pose, velocity]).all():
                    raise RuntimeError("Nonfinite published state")
                published = True
                thread.observe(pose)
                if thread.released and release_time is None:
                    release_time = step * dt
                free_steps += int(phase == "free_flight")
                free_error = 0.0
                if free_start is not None:
                    expected_v = free_start[1][:3] + np.array([0, -9.81 * free_steps * dt, 0])
                    free_error = float(np.linalg.norm(velocity[:3] - expected_v))
                row = {
                    "step": step,
                    "time_s": step * dt,
                    "phase": phase,
                    "publication": publication,
                    "stop_reason": stats.failure_reason,
                    "theta_rad": thread.theta,
                    "rise_m": float(pose[1] - thread.seated_y),
                    "thread_error_m": float(pose[1] - thread.seated_y - thread.lead * thread.theta / (2 * np.pi)),
                    "released": thread.released,
                    "thread_wrench": reaction.tolist(),
                    "tool_wrench": tool.tolist(),
                    "sample_penetration_m": sample_penetration(pose, samples, cap_vertices),
                    "free_velocity_error_m_s": free_error,
                }
                rows.append(row)
                poses.append(pose)
                stream.write(json.dumps(row) + "\n")
                if step % 500 == 0:
                    stream.flush()
                    print(json.dumps(row), flush=True)
                if row["sample_penetration_m"] > 0.0002 or (
                    not thread.released and abs(row["thread_error_m"]) > 0.0003
                ):
                    raise RuntimeError("Physical mechanism gate violated")
                if free_steps * dt >= 0.08:
                    break
            except Exception as exc:
                hard_failures += int(not published)
                failure = {"step": step, "error": repr(exc)}
                break
    solver.save_checkpoint(state, args.output / "last-state.npz")
    (args.output / "mechanism-state.json").write_text(
        json.dumps({"thread": asdict(thread), "release_time_s": release_time}, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output / "trajectory.npz",
        body_q=np.array(poses),
        cap_vertices=cap_vertices,
        cap_faces=cap_faces,
        bottle_vertices=bottle_vertices,
        bottle_faces=bottle_faces,
    )
    engaged = [r for r in rows if not r["released"]]
    free = [r for r in rows if r["phase"] == "free_flight"]
    detached = [r for r in rows if r["phase"] in ("lift_cap", "move_aside", "free_flight")]
    gates = {
        "complete": free_steps * dt >= 0.08 and failure is None,
        "rotation_causes_rise": bool(engaged) and max(abs(r["thread_error_m"]) for r in engaged) <= 0.0003,
        "disengaged": release_time is not None,
        "no_thread_force_after_release": bool(detached) and all(not any(r["thread_wrench"]) for r in detached),
        "cap_removed": bool(rows) and float(poses[-1][0]) >= 0.05 and max(r["rise_m"] for r in rows) >= 0.03,
        "free_dynamics": len(free) * dt >= 0.08 and max(r["free_velocity_error_m_s"] for r in free) <= 0.0001,
        "sample_geometry": bool(rows) and max(r["sample_penetration_m"] for r in rows) <= 0.0002,
    }
    summary = identity | {
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {name: sum(r["publication"] == name for r in rows) for name in ("CONVERGED", "STOPPED_ACCEPTED")}
        | {"HARD_FAILURE": hard_failures},
        "failure": failure,
        "release_time_s": release_time,
        "steps": len(rows),
        "stopped_steps": [
            {"step": r["step"], "reason": r["stop_reason"]} for r in rows if r["publication"] == "STOPPED_ACCEPTED"
        ],
        "import_path": newton.__file__,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dt", type=float, default=DT)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
