# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Squeeze an empty open elastic bottle with force-feedback human-hand curls."""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton
from newton.examples.softbody.monolithic_paper_box import PaperBoxMetrics
from scripts.monolithic_reference.graspit_hand import build_hand
from scripts.monolithic_reference.hand_admittance import HandAdmittance
from scripts.monolithic_reference.plastic_bottle import add_bottle
from scripts.monolithic_reference.retained_assets import sha
from scripts.monolithic_reference.run_graspit_hand import transform

DT, STEPS = 0.005, 2000
TABLE_Y = -0.080125
CONTACT_STIFFNESS = 5e7


def phase_at(t):
    """Settle, squeeze, hold, then explicitly open the fingers."""
    return "settle" if t <= 0.25 else "squeeze" if t <= 4.0 else "hold" if t <= 6.0 else "release"


class BottleMetrics:
    """Measure shell shape, sampled convex penetration, and torque projection independently."""

    def __init__(self, model):
        from scipy.spatial import ConvexHull

        self.model = model
        self.shell = PaperBoxMetrics(model)
        self.rest = model.particle_q.numpy().astype(float)
        self.centered = self.rest - self.rest.mean(0)
        self.hulls = []
        for s, flags in enumerate(model.shape_flags.numpy()):
            if flags & int(newton.ShapeFlags.COLLIDE_PARTICLES) and model.shape_type.numpy()[s] == newton.GeoType.MESH:
                points = transform(
                    model.shape_source[s].vertices * model.shape_scale.numpy()[s], model.shape_transform.numpy()[s]
                )
                self.hulls.append((int(model.shape_body.numpy()[s]), ConvexHull(points).equations, points))
        self.parents, self.children = model.joint_parent.numpy(), model.joint_child.numpy()
        self.joint_frames, self.axes = model.joint_X_p.numpy(), model.joint_axis.numpy()
        self.q_start = model.joint_q_start.numpy()
        self.qd_start = model.joint_qd_start.numpy()
        self.types = model.joint_type.numpy()
        self.com = model.body_com.numpy()
        self.descendants = []
        for child in self.children:
            bodies = {int(child)}
            for p, c in zip(self.parents, self.children, strict=True):
                if p in bodies:
                    bodies.add(int(c))
            self.descendants.append(sorted(bodies))

    def measure(self, state, forces):
        """Evaluate every publication, including valid STOPPED frames."""
        x, poses = state.particle_q.numpy().astype(float), state.body_q.numpy()
        result = self.shell.measure(state.particle_q)
        # Remove rigid translation and rotation; deformation cannot be credited to drift.
        centered = x - x.mean(0)
        u, _, vt = np.linalg.svd(self.centered.T @ centered)
        correction = np.diag([1, 1, np.linalg.det(u @ vt)])
        fit = self.centered @ u @ correction @ vt
        displacement = np.linalg.norm(centered - fit, axis=1)
        samples = np.concatenate((x, x[self.shell.ids].mean(axis=1)))
        depth = max(0.0, TABLE_Y + 0.000125 - float(samples[:, 1].min()))
        hand_table_depth = 0.0
        for body, equations, vertices in self.hulls:
            pose = poses[body].astype(float)
            inverse = np.r_[np.zeros(3), -pose[3:6], pose[6]]
            local = transform(samples - pose[:3], inverse)
            gap = np.max(local @ equations[:, :3].T + equations[:, 3], axis=1)
            depth = max(depth, float(0.000125 - gap.min()))
            hand_table_depth = max(hand_table_depth, TABLE_Y - float(transform(vertices, pose)[:, 1].min()))
        projected = np.zeros(20)
        com = np.array([transform(c, p) for c, p in zip(self.com, poses, strict=True)])
        for j, kind in enumerate(self.types):
            if kind != newton.JointType.REVOLUTE:
                continue
            frame = wp.transform_multiply(wp.transform(*poses[self.parents[j]]), wp.transform(*self.joint_frames[j]))
            axis = np.array(wp.transform_vector(frame, wp.vec3(self.axes[self.qd_start[j]])))
            ids = self.descendants[j]
            wrench = forces.body_wrench[ids]
            torque = wrench[:, 3:] + np.cross(com[ids] - np.array(frame)[:3], wrench[:, :3])
            projected[self.q_start[j]] = np.sum(torque @ axis)
        return result | {
            "deformation_rms_m": float(np.sqrt(np.mean(displacement**2))),
            "deformation_max_m": float(displacement.max()),
            "sampled_penetration_m": depth,
            "hand_table_penetration_m": hand_table_depth,
            "projection_error_Nm": float(np.max(np.abs(projected - forces.joint_force))),
            "bottle_center_m": x.mean(0).tolist(),
            "limit_violation_rad": float(
                max(
                    0,
                    np.max(self.model.joint_limit_lower.numpy() - state.joint_q.numpy()),
                    np.max(state.joint_q.numpy() - self.model.joint_limit_upper.numpy()),
                )
            ),
            "max_joint_speed_rad_s": float(np.abs(state.joint_qd.numpy()).max()),
        }


def build_scene(asset, *, device, precision):
    """Build free bottle, static support, and a fixed-palm articulated hand."""
    builder, _ = build_hand(asset, device=device, contact=True)
    builder.gravity = (0, -9.81, 0)
    # The zero-gravity import probe used a very soft servo. Hold the palm's
    # finger frames against gravity while letting the outer admittance yield.
    for i in range(builder.joint_dof_count):
        builder.joint_target_ke[i] = 2.0
        builder.joint_target_kd[i] = 0.05
        builder.joint_effort_limit[i] = 0.2
    for s, flags in enumerate(builder.shape_flags):
        if flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
            builder.shape_flags[s] = flags | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    material = add_bottle(builder)
    builder.add_shape_plane(
        plane=(0.0, 1.0, 0.0, -TABLE_Y),
        width=0.0,
        length=0.0,
        cfg=builder.ShapeConfig(density=0, margin=0, has_particle_collision=True),
    )
    model = builder.finalize(device=device)
    controller = HandAdmittance(model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy())
    smooth = build_smooth_sdf(model)
    solver = SolverMonolithic(
        model,
        precision=precision,
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=True, limits=True, limit_width=np.full(20, 0.01)),
        cloth_inputs=(
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(range(model.tri_count)),
                bending=material["bending_Nm"],
                thickness=material["thickness_m"],
                surface_density=material["surface_density_kg_m2"],
            ),
        ),
        mass_mode="lumped",
        contact_stiffness=CONTACT_STIFFNESS,
        friction_coefficient=0.5,
        friction_velocity_scale=0.01,
        normal_smoothing=1e-4,
        alignment_policy="frozen",
        alignment_smoothing=0.1,
        publish_policy="current_iterate",
        use_cuda_graph=True,
        pcg_mode="production",
        linear_max_iterations=1000,
        newton_max_iterations=20,
        line_search_max_iterations=24,
        smooth_sdf=smooth,
    )
    source = model.state()
    source.joint_q.assign(controller.opening)
    newton.eval_fk(model, source.joint_q, source.joint_qd, source)
    state, control = solver.create_state(dt=DT, source=source), solver.create_control()
    control.joint_target_q.assign(controller.opening)
    return model, solver, state, control, controller, material, smooth


def evaluate(rows, *, target_force):
    """Keep numerical publication, deformation, tracking and release gates separate."""
    steady = [r for r in rows if 5.5 < r["time_s"] <= 6.0]
    release = [r for r in rows if r["time_s"] > 9.75]
    return {
        "complete": len(rows) == STEPS,
        "visible_deformation": bool(steady) and max(r["deformation_max_m"] for r in steady) >= 0.003,
        "force_tracking": len(steady) == 100
        and max(max(abs(f - target_force) for f in r["load_N"]) for r in steady) <= 0.3,
        "penetration": bool(rows)
        and max(max(r["sampled_penetration_m"], r["hand_table_penetration_m"]) for r in rows) <= 0.002,
        "joint_limits": bool(rows) and max(r["limit_violation_rad"] for r in rows) <= 0.01,
        "joint_speed": bool(rows) and max(r["max_joint_speed_rad_s"] for r in rows) <= 2.0,
        "no_self_intersection": bool(rows) and max(r["self_intersections"] for r in rows) == 0,
        "area": bool(rows)
        and min(r["area_ratio_min"] for r in rows) >= 0.2
        and max(r["area_ratio_max"] for r in rows) <= 2,
        "projection": bool(rows) and max(r["projection_error_Nm"] for r in rows) <= 1e-4,
        "release": len(release) == 50
        and max(max(abs(f) for f in r["load_N"]) for r in release) <= 0.1
        and max(max(r["command_travel_m"]) for r in release) <= 1e-6,
    }


def run(args):
    """Retain trajectory and per-frame feedback; roll back command state on hard failure."""
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    identity = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_files": {
            p.name: sha(p)
            for p in [
                Path(__file__),
                *[
                    Path(__file__).with_name(n)
                    for n in (
                        "plastic_bottle.py",
                        "hand_admittance.py",
                        "graspit_hand.py",
                        "admittance_control.py",
                    )
                ],
            ]
        },
    }
    (output / "input.json").write_text(
        json.dumps(
            identity
            | {
                "precision": args.precision,
                "target_force_N": args.force,
                "dt_s": DT,
                "requested_steps": args.max_steps,
                "servo_ke_Nm_rad": 2,
                "servo_kd_Nms_rad": 0.05,
                "servo_effort_cap_Nm": 0.2,
                "contact_stiffness_N_m3": CONTACT_STIFFNESS,
            },
            indent=2,
        )
        + "\n"
    )
    model, solver, state, control, controller, material, smooth = build_scene(
        args.asset_dir, device=args.device, precision=args.precision
    )
    metrics = BottleMetrics(model)
    initial = metrics.measure(
        state, SimpleNamespace(body_wrench=np.zeros((model.body_count, 6)), joint_force=np.zeros(20))
    )
    (output / "initial-geometry.json").write_text(json.dumps(initial, indent=2) + "\n")
    if initial["sampled_penetration_m"] > 1e-6 or initial["hand_table_penetration_m"] > 1e-6:
        raise ValueError("Initial hand/bottle/support geometry overlaps")
    rows, xs, poses, qs = [], [state.particle_q.numpy()], [state.body_q.numpy()], [state.joint_q.numpy()]
    failure = None
    early_stop = None
    with (output / "steps.jsonl").open("w") as stream:
        for step in range(1, min(args.max_steps, STEPS) + 1):
            time = step * DT
            phase = phase_at(time)
            before = controller.snapshot()
            old_q, old_v = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
            try:
                if step > 1:
                    target, velocity = controller.command(
                        phase, solver.get_contact_forces(state), DT, target_force=args.force
                    )
                    control.joint_target_q.assign(target)
                    control.joint_target_qd.assign(velocity)
                solver.step(state, state, control, dt=DT)
                stats = solver.last_stats
                publication = stats.variational.get("publication_status")
                if stats.rolled_back or publication not in ("CONVERGED", "STOPPED_ACCEPTED"):
                    raise RuntimeError(f"{publication}: {stats.failure_reason}")
                forces = solver.get_contact_forces(state)
                if not forces.force_valid or not all(
                    np.isfinite(a).all()
                    for a in (
                        state.particle_q.numpy(),
                        state.joint_q.numpy(),
                        state.particle_qd.numpy(),
                        state.joint_qd.numpy(),
                        forces.particle_force,
                        forces.body_wrench,
                    )
                ):
                    raise RuntimeError("Invalid published state or force")
                row = {
                    "step": step,
                    "time_s": time,
                    "phase": phase,
                    "publication": publication,
                    "stop_reason": stats.failure_reason,
                    "converged": stats.converged,
                    "newton_iterations": stats.nonlinear_iterations,
                    "convergence_ratio": stats.convergence_ratio,
                    "load_N": controller.loads(forces).tolist(),
                    "command_travel_m": [a.position for a in controller.axes],
                    "command_velocity_m_s": [a.velocity for a in controller.axes],
                    "target_force_N": args.force if phase in ("squeeze", "hold") else 0,
                    **metrics.measure(state, forces),
                }
                rows.append(row)
                xs.append(state.particle_q.numpy())
                poses.append(state.body_q.numpy())
                qs.append(state.joint_q.numpy())
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if step % 50 == 0:
                    print(json.dumps(row), flush=True)
                # These are all-frame gates: later recovery cannot erase a violation.
                if (
                    max(row["sampled_penetration_m"], row["hand_table_penetration_m"]) > 0.002
                    or row["self_intersections"] > 0
                    or row["area_ratio_min"] < 0.2
                    or row["area_ratio_max"] > 2
                    or row["limit_violation_rad"] > 0.01
                    or row["max_joint_speed_rad_s"] > 2
                ):
                    early_stop = {"step": step, "reason": "Physical all-frame gate violated; checkpoint retained"}
                    break
            except Exception as exc:
                controller.restore(before)
                control.joint_target_q.assign(old_q)
                control.joint_target_qd.assign(old_v)
                failure = {"step": step, "error": f"{type(exc).__name__}: {exc}"}
                break
    solver.save_checkpoint(state, output / "last-state.npz")
    np.savez_compressed(
        output / "trajectory.npz", particle_q=xs, body_q=poses, joint_q=qs, triangles=model.tri_indices.numpy()
    )
    gates = evaluate(rows, target_force=args.force)
    summary = {
        "passed": all(gates.values()) and failure is None,
        "gates": gates,
        "failure": failure,
        "early_stop": early_stop,
        "precision": args.precision,
        "device": args.device,
        "material": material,
        "counts": {k: sum(r["publication"] == k for r in rows) for k in ("CONVERGED", "STOPPED_ACCEPTED")}
        | {"HARD_FAILURE": int(failure is not None)},
        "stopped_steps": [{"step": r["step"], "reason": r["stop_reason"]} for r in rows if not r["converged"]],
        "requested_steps": args.max_steps,
        "dt_s": DT,
        "target_force_N": args.force,
        "max_deformation_m": max((r["deformation_max_m"] for r in rows), default=None),
        "max_penetration_m": max((r["sampled_penetration_m"] for r in rows), default=None),
        "newton_import": str(Path(newton.__file__).resolve()),
        **identity,
        "asset_manifest_sha256": sha(args.asset_dir / "manifest.json"),
        "smooth_sdf": smooth.build_metadata,
        "scope": "four-finger virtual-tendon admittance; fixed palm, held thumb, free elastic bottle on support; no self-contact response or PET calibration",
        "penetration_scope": "all nodes and triangle centroids against original convex hull planes; conservative outside, exact inside; not continuous collision certification",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("smooth_sdf", "stopped_steps", "source_files")}, indent=2
        )
    )
    return summary["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--force", type=float, default=2.0, help="Force [N] per virtual tendon, not per fingertip")
    parser.add_argument("--max-steps", type=int, default=STEPS, help="Short probes cannot qualify the complete task")
    args = parser.parse_args()
    if not np.isfinite(args.force) or args.force <= 0 or args.max_steps <= 0:
        parser.error("Require positive finite force and positive steps")
    raise SystemExit(0 if run(args) else 1)
