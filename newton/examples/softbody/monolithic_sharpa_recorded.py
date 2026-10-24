# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Try the recorded cube grasp with the approved C-IE production graph policy."""

import argparse
import gzip
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import build_smooth_sdf

import newton
from newton.examples.softbody.monolithic_grasp_trajectory import RecordedGrasp
from newton.examples.softbody.monolithic_recording import (
    GRASP_WINDOW_STEPS,
    grasp_metrics,
    measure,
    object_weight_and_edge,
    save,
    serial,
    state_arrays,
)
from newton.examples.softbody.monolithic_sharpa_scene import ASSETS, REPO, build


def build_recorded(payload):
    """Reuse PR3 physical parameters, then bind the recorded scene before solving."""
    trajectory = RecordedGrasp(payload)
    model, inherited, loaded, mapping, curve, terms, _ = build(18)
    # Match the recorder's FP32 CSV interpolation before filling control buffers.
    curve.q = curve.q.astype(np.float32)
    curve.slopes = (np.diff(curve.q, axis=0) / np.diff(curve.time)[:, None]).astype(np.float32)
    lift_joint = next(i for i, label in enumerate(model.joint_label) if label.rsplit("/", 1)[-1] == "monolithic_lift")
    xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    xp[lift_joint] = [*trajectory.position[0], 0, 0, 0, 1]
    # Keep the prismatic axis in world Z while orienting the palm independently.
    xc[lift_joint] = [0, 0, 0, *(-trajectory.quaternion[:3]), trajectory.quaternion[3]]
    model.joint_X_p.assign(xp)
    model.joint_X_c.assign(xc)
    planes = np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))
    if len(planes) != 1:
        raise ValueError("Require the single existing support plane")
    transforms = model.shape_transform.numpy()
    transforms[planes[0]] = [0, 0, 0, 0, 0, 0, 1]
    model.shape_transform.assign(transforms)
    cube_path = REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz"
    with np.load(cube_path) as data:
        model.particle_q.assign(data["vertices"] + np.array([0, 0, 0.02], dtype=np.float32))
    scene = {
        "object_position_m": [0, 0, 0.02],
        "mount": {**inherited["mount"], "support_height": 0.0},
        "palm_initial_position_m": trajectory.position[0],
        "palm_initial_quaternion_xyzw": trajectory.quaternion,
        "source_physical_fixture": inherited,
        "cube_sha256": hashlib.sha256(cube_path.read_bytes()).hexdigest(),
        "loaded_assets": loaded,
    }
    return model, scene, mapping, curve, terms, trajectory, int(model.joint_child.numpy()[lift_joint])


def targets(trajectory, curve, mapping, time_s):
    """Map recorded gripper phase and vertical displacement at each step endpoint."""
    position, grip, velocity, grip_rate, phase = trajectory.sample(time_s)
    q, qd = curve.sample(3.0 * (1.0 - grip))
    target, rate = np.zeros(23, np.float32), np.zeros(23, np.float32)
    for i, name in enumerate(curve.names):
        _, dof, slot = mapping[name]
        target[slot], rate[dof] = q[i], -3.0 * grip_rate * qd[i]
    _, dof, slot = mapping["monolithic_lift"]
    target[slot], rate[dof] = position[2] - trajectory.position[0, 2], velocity[2]
    return target, rate, position, phase


def run(args):
    """Preserve the first failure and record every published complete state."""
    output = args.output
    payload = json.loads(args.trajectory.read_text())
    model, scene, mapping, curve, terms, trajectory, palm = build_recorded(payload)
    smooth = build_smooth_sdf(model, qualified_directory=ASSETS / "sharpa/geometry-rebuilt")
    solver_factory = getattr(args, "solver_factory", None)
    contact_recorder = getattr(args, "contact_recorder", None)
    if solver_factory is None:
        from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory  # noqa: PLC0415

        solver_factory = FrozenRobotFactory("sharpa-recorded-10mm", dataflow="default")
        contact_recorder = solver_factory.measure
    solver = solver_factory(
        model,
        alignment_policy="mask",
        publish_policy="current_iterate",
        friction_velocity_scale=0.01,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        normal_smoothing=1e-4,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
        joint_terms=terms,
        pcg_mode="production",
        use_cuda_graph=True,
        smooth_sdf=smooth,
    )
    source, control = model.state(), model.control()
    q, v, _, _ = targets(trajectory, curve, mapping, 0)
    initial = source.joint_q.numpy()
    for coord, _, slot in mapping.values():
        initial[coord] = q[slot]
    source.joint_q.assign(initial)
    source.joint_qd.zero_()
    newton.eval_fk(model, source.joint_q, source.joint_qd, source)
    state = solver.create_state(dt=trajectory.dt, source=source)
    expected_digest = getattr(args, "expected_config_digest", None)
    certify = getattr(args, "certify_configuration", None)
    if certify is not None:
        certify(solver)
    if expected_digest is not None and solver._kinematics.digest.hex() != expected_digest:
        raise ValueError("Physical configuration digest changed")
    expected_controls = getattr(args, "expected_controls", None)
    if expected_controls is not None:
        samples = np.asarray(
            [targets(trajectory, curve, mapping, i * trajectory.dt)[:2] for i in range(trajectory.steps + 1)],
            dtype=np.float32,
        )
        if not np.array_equal(samples, np.load(expected_controls, allow_pickle=False)):
            raise ValueError("Constructed controls differ from frozen controls")
    pose = solver._kinematics.poses.numpy()[palm].astype(float)
    # The solver evaluates the initial pose in its own precision.
    position_tolerance, rotation_tolerance = (
        (1e-8, 1e-12) if solver.precision == solver.Precision.FP64 else (2e-6, 2e-6)
    )
    np.testing.assert_allclose(pose[:3], trajectory.position[0], atol=position_tolerance, rtol=0)
    if abs(abs(np.dot(pose[3:], trajectory.quaternion)) - 1) > rotation_tolerance:
        raise ValueError("Initial palm orientation does not match recording")
    config = {
        "trajectory": payload,
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "old_manifest": scene,
        "expected_steps": trajectory.steps,
        "dt": trajectory.dt,
        "recorded_duration": trajectory.duration,
        "simulated_duration": trajectory.steps * trajectory.dt,
        "endpoint_hold_seconds": trajectory.steps * trajectory.dt - trajectory.duration,
        "mode": "production",
        "graph": True,
        "law": "C-IE",
        "globalization": "ArmijoOrResNorm",
        "alignment_policy": "mask",
        "publish_policy": "current_iterate",
        "sdf_representation": "c2-bspline",
        "actual_config": solver._config,
        "config_digest": solver._kinematics.digest.hex(),
        "smooth_sdf_identity": smooth.identity,
        "joint_terms": terms,
        "physical_scope": "P3 r2 soft cube, unchanged physical parameters; recorded targets, not MuJoCo dynamics",
        "missing_phases": (["approach"] if not np.any(np.diff(trajectory.position[:, 2]) < 0) else [])
        + ["high_hold", "release"],
        "target_sampling": "1 ms endpoint interpolation; original MuJoCo uses 60 Hz held targets",
        "initial_palm_pose": pose,
        "palm_body": palm,
        "particle_count": model.particle_count,
        "tet_count": model.tet_count,
    }
    save(output / "configuration.json", config)
    np.savez_compressed(output / "initial.npz", **state_arrays(state))
    np.savez_compressed(
        output / "model.npz",
        **{name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)},
    )
    frames, index = [], {"chunks": [], "total_frames": 0, "includes_initial": True}

    def record(step, label, target, rate):
        frames.append((step, label, state_arrays(state), target.copy(), rate.copy()))

    def flush():
        if not frames:
            return
        path = output / f"trajectory/frames-{frames[0][0]:06d}-{frames[-1][0]:06d}.npz"
        path.parent.mkdir(exist_ok=True)
        arrays = {name: np.stack([frame[2][name] for frame in frames]) for name in frames[0][2]}
        arrays.update(
            frame_steps=np.array([f[0] for f in frames]),
            frame_times=np.array([f[0] * trajectory.dt for f in frames]),
            frame_publication=np.array([f[1] for f in frames], dtype="U16"),
            frame_target_q=np.stack([f[3] for f in frames]),
            frame_target_qd=np.stack([f[4] for f in frames]),
        )
        np.savez_compressed(path, **arrays)
        index["chunks"].append(
            {
                "path": str(path.relative_to(output)),
                "count": len(frames),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        index["total_frames"] += len(frames)
        save(output / "trajectory-index.json", index)
        frames.clear()

    record(0, "INITIAL", q, v)
    counts = dict.fromkeys(("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE"), 0)
    weights = model.particle_mass.numpy().astype(float)
    initial_com = np.average(state.particle_q.numpy(), axis=0, weights=weights)
    object_weight, object_edge = object_weight_and_edge(model, state.particle_q.numpy())
    max_penetration, min_det, max_com_rise, last_ground = 0.0, 1.0, 0.0, None
    # The recording ends at the top of the lift, so its final steps form the grasp window.
    grasp_window = []
    begin = time.perf_counter()
    calls = []
    zero_accepted_steps = []
    solve = solver._linear.solve_pcg

    def measured(*a, **kw):
        result = solve(*a, **kw)
        calls.append({"iterations": result.iterations, "status": result.status})
        return result

    solver._linear.solve_pcg = measured
    with gzip.open(output / "steps.jsonl.gz", "wt") as log:
        for step in range(1, min(getattr(args, "max_steps", None) or trajectory.steps, trajectory.steps) + 1):
            q, v, desired, phase = targets(trajectory, curve, mapping, step * trajectory.dt)
            control.joint_target_q.assign(q)
            control.joint_target_qd.assign(v)
            before = state_arrays(state)
            calls.clear()
            started = time.perf_counter()
            solver.step(state, state, control, dt=trajectory.dt)
            stats = solver.last_stats
            label = stats.variational["publication_status"]
            counts[label] += 1
            if stats.variational["zero_accepted_publication"]:
                zero_accepted_steps.append(step)
            row = {
                "step": step,
                "time": step * trajectory.dt,
                "phase": phase,
                "target_q": q,
                "target_qd": v,
                "palm_target": desired,
                "stats": stats,
                "publication": label,
                "published": not stats.rolled_back,
                "solve_seconds": time.perf_counter() - started,
                "pcg_calls": list(calls),
            }
            if not stats.rolled_back:
                fields = contact_recorder(solver, state) if contact_recorder else measure(solver, state)
                com = np.average(fields["x"], axis=0, weights=weights)
                body_force = state._monolithic_coupled.contact_body_force.numpy()
                palm_pose = solver._kinematics.poses.numpy()[palm]
                row.update(
                    cube_com=com,
                    min_cube_z=float(fields["x"][:, 2].min()),
                    palm_pose=palm_pose,
                    palm_tracking_error=palm_pose[:3] - desired,
                    joint_tracking_error=state.joint_q.numpy() - q,
                    ground_force_on_cube=-body_force[0, :3],
                    hand_force_on_cube=-body_force[1:, :3].sum(axis=0),
                    penetration=fields["penetration"],
                    slip=fields["slip"],
                    friction_work=fields["friction_work"],
                    force_balance=fields["force_balance"],
                    mask=fields["mask"],
                    **(
                        {name: fields[name] for name in ("full_system", "contact_pool", "contact_spatial")}
                        if contact_recorder
                        else {}
                    ),
                    force_generation=int(state._monolithic_coupled.force_generation.numpy()[0]),
                    force_valid=int(state._monolithic_coupled.force_valid.numpy()[0]),
                )
                max_penetration = max(max_penetration, fields["penetration"])
                min_det = min(min_det, stats.min_det_f)
                max_com_rise = max(max_com_rise, com[2] - initial_com[2])
                last_ground = row["ground_force_on_cube"]
                if step > trajectory.steps - GRASP_WINDOW_STEPS:
                    grasp_window.append((step, com[2], last_ground[2]))
                record(step, label, q, v)
                if label == "STOPPED_ACCEPTED" and counts[label] == 1:
                    np.savez_compressed(output / "first-stop-input.npz", **before)
                    np.savez_compressed(output / "first-stop-published.npz", **state_arrays(state))
                if len(frames) >= 100:
                    flush()
            else:
                after = state_arrays(state)
                rollback = all(np.array_equal(value, after[name]) for name, value in before.items())
                np.savez_compressed(output / "failure-input.npz", **before)
                for name in ("_accepted", "_trial"):
                    np.savez_compressed(output / f"{name[1:]}-failure.npz", **state_arrays(getattr(solver, name)))
                save(output / "failure.json", {"row": row, "rollback_preserved": rollback})
            log.write(json.dumps(serial(row), allow_nan=False) + "\n")
            log.flush()
            if stats.rolled_back:
                break
            if step % 100 == 0:
                print(step, phase, counts, flush=True)
    flush()
    np.savez_compressed(output / "final.npz", **state_arrays(state))
    published = counts["CONVERGED"] + counts["STOPPED_ACCEPTED"]
    summary = {
        "complete": published == trajectory.steps,
        "publication_counts": counts,
        "publish_policy": "current_iterate",
        "zero_accepted_steps": zero_accepted_steps,
        "published_steps": published,
        "expected_steps": trajectory.steps,
        "last_published_time": published * trajectory.dt,
        "maximum_com_rise_m": max_com_rise,
        "final_com": np.average(state.particle_q.numpy(), axis=0, weights=weights),
        "maximum_penetration_m": max_penetration,
        "minimum_det_f": min_det,
        "final_ground_force": last_ground,
        "elapsed_seconds": time.perf_counter() - begin,
        "strict_complete": counts["CONVERGED"] == trajectory.steps,
        "grasp": grasp_metrics(
            grasp_window, initial_com_z=float(initial_com[2]), weight=object_weight, edge=object_edge
        ),
        "missing_phases": config["missing_phases"],
    }
    save(output / "summary.json", summary)
    print(json.dumps(serial(summary)), flush=True)
    return summary["complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=ASSETS / "sharpa/recorded-control.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
    save(
        args.output / "source-hashes.json",
        {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("newton", "scripts/monolithic_reference")
            for p in (REPO / folder).rglob("*.py")
        },
    )
    try:
        return 0 if run(args) else 1
    except Exception:
        save(args.output / "error.json", {"complete": False, "exception": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
