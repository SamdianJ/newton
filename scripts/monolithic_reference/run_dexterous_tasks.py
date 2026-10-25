# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run contact-driven task reconstructions with two dynamic fingertip proxies.

These scripted controls are task-level reconstructions, not video estimates or
full-hand retargeting. Objects receive no pose targets or grasp attachments.
"""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

TASKS = ("pinch-transfer", "turn-knob", "cable-pull")
DT = 0.002
DURATION = 4.0
RADIUS = 0.012
REPO = Path(__file__).resolve().parents[2]


def ramp(t, start, end):
    """Return a quintic position blend and its analytic time derivative."""
    u = np.clip((t - start) / (end - start), 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u**2), 30 * u**2 * (1 - u) ** 2 / (end - start)


def targets(task, t):
    """Return finger center positions [m], velocities [m/s], and phase."""
    close, close_v = ramp(t, 0.2, 0.8)
    release, release_v = ramp(t, 3.1, 3.5)
    grip, grip_v = close * (1 - release), close_v * (1 - release) - close * release_v
    move, move_v = ramp(t, 1.2, 2.3)
    center, velocity = np.zeros(3), np.zeros(3)
    angle = angle_v = 0.0
    if task in ("pinch-transfer", "pinch-place"):
        lift, lift_v = ramp(t, 0.9, 1.2)
        lower, lower_v = ramp(t, 2.5, 3.0)
        center[:] = (0.07 * move, 0, 0.016 + 0.045 * (lift - lower))
        velocity[:] = (0.07 * move_v, 0, 0.045 * (lift_v - lower_v))
        if task == "pinch-transfer":
            angle, angle_v = np.pi / 2 * move, np.pi / 2 * move_v
        open_gap, closed_gap = 0.04, 0.025
    elif task == "turn-knob":
        center[2] = 0.04
        angle, angle_v = np.pi / 2 * move, np.pi / 2 * move_v
        open_gap, closed_gap = 0.04, 0.022
    elif task == "cable-pull":
        center[:] = (0.045 * move, 0, 0.07 + 0.025 * move)
        velocity[:] = (0.045 * move_v, 0, 0.025 * move_v)
        open_gap, closed_gap = 0.027, 0.014
    else:
        raise ValueError(f"Unknown task: {task}")
    gap = open_gap + (closed_gap - open_gap) * grip
    gap_v = (closed_gap - open_gap) * grip_v
    direction = np.array((-np.sin(angle), np.cos(angle), 0))
    direction_v = angle_v * np.array((-np.cos(angle), -np.sin(angle), 0))
    positions = np.stack([center + sign * gap * direction for sign in (-1, 1)])
    velocities = np.stack([velocity + sign * (gap_v * direction + gap * direction_v) for sign in (-1, 1)])
    phase = (
        "settle" if t < 0.2 else "pinch" if t < 0.9 else "move" if t < 2.5 else "hold/lower" if t < 3.1 else "release"
    )
    return positions, velocities, phase


class TaskScene:
    """Build a small coupled scene without modifying the solver or frozen assets."""

    def __init__(self, task, *, device="cuda:0", precision="fp64", friction=0.6, newton_iterations=80):
        self.task = task
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        cfg = builder.ShapeConfig(density=0, margin=0.0)
        self.initial, _, _ = targets(task, 0)
        self.fingers = []
        root = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)), label="fixed_support")
        joints = [builder.add_joint_fixed(-1, root)]
        for finger, origin in enumerate(self.initial):
            parent = root
            for axis in range(3):
                body = builder.add_link(
                    mass=0.015 if axis == 2 else 0.001,
                    inertia=wp.diag(wp.vec3(1e-6)),
                    label=f"finger_{finger}_{axis}",
                )
                joints.append(
                    builder.add_joint_prismatic(
                        parent,
                        body,
                        axis=wp.vec3(np.eye(3)[axis]),
                        parent_xform=wp.transform(origin if axis == 0 else (0, 0, 0), wp.quat_identity()),
                        target_ke=8000.0,
                        target_kd=20.0,
                        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                        limit_ke=0,
                        limit_kd=0,
                        damping=0,
                        friction=0,
                    )
                )
                parent = body
            builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
            self.fingers.append(body)
        builder.add_shape_plane(width=0, length=0, cfg=cfg)
        self.object_body = None
        cables = []
        if task in ("pinch-transfer", "pinch-place"):
            builder.add_articulation(joints)
            self.object_body = builder.add_body(
                xform=wp.transform((0, 0, 0.016), wp.quat_identity()),
                mass=0.012,
                inertia=wp.diag(wp.vec3(0.012 * 0.03**2 / 6)),
                label="free_block",
            )
            builder.add_shape_box(self.object_body, hx=0.015, hy=0.015, hz=0.015, cfg=cfg)
        elif task == "turn-knob":
            pose = wp.transform((0, 0, 0.04), wp.quat_identity())
            self.object_body = builder.add_link(mass=0.03, inertia=wp.diag(wp.vec3(8e-6)), label="knob")
            self.knob_joint = builder.add_joint_revolute(
                root,
                self.object_body,
                parent_xform=pose,
                axis=wp.vec3(0, 0, 1),
                target_ke=0,
                target_kd=0,
                limit_ke=0,
                limit_kd=0,
                friction=0.001,
                damping=0,
            )
            builder.add_articulation([*joints, self.knob_joint])
            builder.add_shape_box(self.object_body, hx=0.025, hy=0.012, hz=0.012, cfg=cfg)
        else:
            builder.add_articulation(joints)
            self.rest = np.column_stack((np.zeros(17), np.zeros(17), np.linspace(0.22, 0.06, 17)))
            masses = np.full(17, 0.02 * 0.01)
            masses[[0, -1]] /= 2
            for i, (point, mass) in enumerate(zip(self.rest, masses, strict=True)):
                builder.add_particle(wp.vec3(point), wp.vec3(0), 0 if i < 2 else mass, radius=0.003)
            cables.append(
                SolverMonolithic.CableInput(
                    particle_ids=tuple(range(17)),
                    EA=50.0,
                    bending=((2e-5, 0), (0, 2e-5)),
                    GJ=1e-5,
                    radius=0.003,
                    linear_density=0.02,
                    rest_theta=np.zeros(16),
                    rest_director=np.tile((1.0, 0.0, 0.0), (16, 1)),
                    fixed_theta=np.arange(16) == 0,
                )
            )
        self.model = builder.finalize(device=device)
        self.solver = SolverMonolithic(
            self.model,
            cable_inputs=cables,
            mass_mode="lumped",
            precision=precision,
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True,
                friction=True,
                friction_velocity_scale=tuple([0.001] * (7 if task == "turn-knob" else 6)),
            ),
            contact_stiffness=1e7,
            friction_coefficient=friction,
            friction_velocity_scale=0.001,
            normal_smoothing=1e-4,
            newton_max_iterations=newton_iterations,
            linear_max_iterations=300,
            line_search_max_iterations=24,
            pcg_mode="production",
            use_cuda_graph=wp.get_device(device).is_cuda,
            publish_policy="strict",
        )
        self.state = self.solver.create_state(dt=DT)
        self.control = self.solver.create_control()

    def step(self, t):
        """Only update fingertip actuator targets; leave the object unactuated."""
        position, velocity, phase = targets(self.task, t)
        q = np.zeros(self.control.joint_target_q.shape[0])
        v = np.zeros(self.control.joint_target_qd.shape[0])
        q[:6], v[:6] = (position - self.initial).ravel(), velocity.ravel()
        self.control.joint_target_q.assign(q)
        self.control.joint_target_qd.assign(v)
        self.solver.step(self.state, self.state, self.control, dt=DT)
        return position, phase


def task_metrics(scene, rows):
    """Judge object behavior independently of the nonlinear stopping criterion."""
    contact = max((r["finger_contact_force_N"] for r in rows), default=0.0)
    metrics = {"peak_finger_contact_force_N": contact}
    gates = {"contact_present": contact > 0.01}
    if scene.task in ("pinch-transfer", "pinch-place"):
        poses = np.array([r["object_pose"] for r in rows])
        quat = poses[-1, 3:]
        yaw = np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
        target_yaw = np.pi / 2 if scene.task == "pinch-transfer" else 0.0
        yaw_error = abs(np.arctan2(np.sin(yaw - target_yaw), np.cos(yaw - target_yaw)))
        metrics.update(
            lift_m=float(poses[:, 2].max() - 0.016),
            placement_error_m=float(np.linalg.norm(poses[-1, :3] - (0.07, 0, 0.015))),
            yaw_error_deg=float(np.degrees(yaw_error)),
            final_drift_m=float(np.linalg.norm(poses[-1, :3] - poses[max(0, len(poses) - 100), :3])),
        )
        gates.update(
            lift=metrics["lift_m"] >= 0.03,
            placement=metrics["placement_error_m"] <= 0.01,
            rotation=metrics["yaw_error_deg"] <= 15,
            release_stable=metrics["final_drift_m"] <= 0.002,
        )
    elif scene.task == "turn-knob":
        metrics["angle_error_deg"] = float(abs(np.degrees(rows[-1]["knob_angle_rad"]) - 90))
        gates["rotation"] = metrics["angle_error_deg"] <= 10
    else:
        metrics.update(
            peak_grab_displacement_m=max(r["grab_displacement_m"] for r in rows),
            peak_edge_stretch=max(r["edge_stretch_max"] for r in rows),
        )
        gates.update(pull=metrics["peak_grab_displacement_m"] >= 0.02, stretch=metrics["peak_edge_stretch"] <= 1.1)
    return metrics, gates


def save_json(path, payload):
    """Write a human-readable run receipt."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run_case(task, args):
    """Preserve complete states, first failure, and independent task verdicts."""
    output = args.output / task
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    scene = TaskScene(
        task,
        device=args.device,
        precision=args.precision,
        friction=args.friction,
        newton_iterations=args.newton_iterations,
    )
    frames = {key: [] for key in ("time", "body_q", "particle_q", "joint_q", "target", "phase")}
    rows, error = [], None

    def record(t, target, phase):
        values = (
            t,
            scene.state.body_q.numpy(),
            scene.state.particle_q.numpy(),
            scene.state.joint_q.numpy(),
            target,
            phase,
        )
        for key, value in zip(frames, values, strict=True):
            frames[key].append(value)

    record(0, scene.initial, "initial")
    limit = min(args.max_steps or round(DURATION / DT), round(DURATION / DT))
    try:
        with (output / "trace.jsonl").open("w") as stream:
            for step in range(1, limit + 1):
                target, phase = scene.step(step * DT)
                stats = scene.solver.last_stats
                row = {
                    "step": step,
                    "time": step * DT,
                    "phase": phase,
                    "status": str(stats.status),
                    "converged": bool(stats.converged),
                    "rolled_back": bool(stats.rolled_back),
                    "newton": stats.nonlinear_iterations,
                    "pcg": stats.linear_iterations,
                    "failure_reason": stats.failure_reason,
                }
                for name in ("convergence_ratio", "accepted_alpha", "raw_residual_q_norm"):
                    value = getattr(stats, name)
                    row[name] = float(value) if np.isfinite(value) else None
                body = scene.state.body_q.numpy()
                row["tracking_error_m"] = float(np.max(np.linalg.norm(body[scene.fingers, :3] - target, axis=1)))
                force = scene.solver.get_contact_forces(scene.state)
                row["finger_contact_force_N"] = float(
                    np.max(np.linalg.norm(force.body_wrench[scene.fingers, :3], axis=1))
                )
                if scene.object_body is not None:
                    row["object_pose"] = body[scene.object_body].tolist()
                if task == "turn-knob":
                    row["knob_angle_rad"] = float(scene.state.joint_q.numpy()[6])
                if task == "cable-pull":
                    x = scene.state.particle_q.numpy()
                    row["grab_displacement_m"] = float(np.mean(x[-3:, 0]))
                    row["edge_stretch_max"] = float(np.max(np.linalg.norm(np.diff(x, axis=0), axis=1) / 0.01))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 16 == 0 or step == limit or not stats.converged:
                    record(step * DT, target, phase)
                if not stats.converged or stats.rolled_back:
                    error = f"Step {step}: {stats.status}: {stats.failure_reason}"
                    scene.solver.save_checkpoint(scene.state, output / "last-published-state.npz")
                    break
                if step % 200 == 0:
                    print(
                        f"{task}: {step}/{limit}, Newton {stats.nonlinear_iterations}, contact {row['finger_contact_force_N']:.4g} N",
                        flush=True,
                    )
    except Exception:
        error = traceback.format_exc()
    np.savez_compressed(output / "trajectory.npz", **{key: np.array(value) for key, value in frames.items()})
    metrics, gates = task_metrics(scene, rows) if rows else ({}, {})
    complete = len(rows) == round(DURATION / DT) and error is None
    numerical_pass = complete and all(r["converged"] and not r["rolled_back"] for r in rows)
    result = {
        "task": task,
        "status": "PASS" if numerical_pass and all(gates.values()) else "FAIL" if error or complete else "PARTIAL",
        "complete": complete,
        "numerical_pass": numerical_pass,
        "task_gates": gates,
        "metrics": metrics,
        "steps": len(rows),
        "expected_steps": round(DURATION / DT),
        "dt": DT,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "max_tracking_error_m": max((r["tracking_error_m"] for r in rows), default=None),
        "max_newton_iterations": max((r["newton"] for r in rows), default=None),
    }
    save_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    """Run every selected task, retaining failures without suppressing later cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "pinch-place", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--friction", type=float, default=0.6)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--newton-iterations", type=int, default=80)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.newton_iterations <= 0:
        parser.error("--newton-iterations must be positive")
    if not np.isfinite(args.friction) or args.friction < 0:
        parser.error("--friction must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((REPO / "newton/_src/solvers/monolithic").glob("*.py"))]
    save_json(
        args.output / "manifest.json",
        {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "newton_import": newton.__file__,
            "device": args.device,
            "precision": args.precision,
            "friction": args.friction,
            "max_steps": args.max_steps,
            "newton_iterations": args.newton_iterations,
            "task": args.task,
            "input_kind": "scripted dynamic fingertip proxies; no video or object pose replay",
            "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        },
    )
    results = []
    for task in TASKS if args.task == "all" else (args.task,):
        try:
            results.append(run_case(task, args))
        except Exception:
            result = {"task": task, "status": "ERROR", "error": traceback.format_exc()}
            results.append(result)
            save_json(args.output / task / "summary.json", result)
            print(result["error"], flush=True)
    save_json(args.output / "summary.json", results)
    raise SystemExit(0 if all(r["status"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
