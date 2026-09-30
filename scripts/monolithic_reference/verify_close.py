# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen PR3 Close from t=0, or reset a historical input for one diagnostic step."""

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube
from newton.examples.softbody.monolithic_sharpa_scene import joint_mapping
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture
from scripts.monolithic_reference.build_smooth_sdf import load_qualified_fields
from scripts.monolithic_reference.verify_coupled import measure, save, serial, state_arrays

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
ARTIFACTS = WORKSPACE / "agents/integration/artifacts"
DT = 0.001
STEPS = 2500


def complete_trace(rows, expected):
    """Reject truncated, failed or shifted traces even if their final state is valid."""
    return len(rows) == expected and all(
        r["success"] and r["step"] == i + 1 and abs(r["time"] - (i + 1) * DT) <= 1e-12 for i, r in enumerate(rows)
    )


def complete_publication(rows, expected):
    """Count approved Stopped publications separately from strict convergence."""
    return len(rows) == expected and all(
        r.get("published", False)
        and r["publication"] in ("CONVERGED", "STOPPED_ACCEPTED")
        and r["step"] == i + 1
        and abs(r["time"] - (i + 1) * DT) <= 1e-12
        for i, r in enumerate(rows)
    )


def mask_statistics(contact):
    """Report omitted loads and penetration, including contacts excluded by the mask."""
    omitted = contact.omitted.numpy()
    keys = np.flatnonzero(np.any(omitted[:, 2:] > 0, axis=1))
    seen = contact.mask_seen.numpy()
    gap = contact.current.distance.numpy() - contact.parameters[4]
    gap -= contact.margins.numpy()[contact.shapes.numpy()[np.arange(contact.count) % contact.shapes.size]]
    return {
        "masked_active_count": len(keys),
        "masked_active_keys": keys,
        "switched_active_keys": np.flatnonzero(seen == 3),
        "omitted_normal_energy": float(omitted[:, 0].sum()),
        "omitted_friction_energy": float(omitted[:, 1].sum()),
        "omitted_normal_load_sum": float(omitted[:, 2].sum()),
        "omitted_friction_load_sum": float(omitted[:, 3].sum()),
        "max_masked_penetration": float(np.maximum(-gap[keys], 0).max()) if len(keys) else 0.0,
        "max_all_penetration": float(np.maximum(-gap, 0).max()) if len(gap) else 0.0,
    }


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


def run(args):
    """Stop on the first rejected step and retain all published and trial state."""
    output = args.output
    compact = getattr(args, "compact", False)
    candidate = getattr(args, "production_candidate", False)
    started = time.perf_counter()
    model, manifest, loaded, mapping, curve, terms, old = build(args.position)
    smooth_sdf = None
    representation = getattr(args, "sdf_representation", "trilinear")
    if representation != "trilinear":
        smooth_sdf = load_qualified_fields(model, args.sdf_qualification, representation)
    run = SolverMonolithic._create_coupled_variational(
        model,
        contact_model=args.law,
        globalization="ArmijoOrResNorm" if candidate else "ResNorm",
        alignment_policy="mask" if candidate else "guard",
        publish_policy="last_accepted" if candidate else "strict",
        allow_nonconservative_energy=candidate,
        friction_velocity_scale=0.01,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        normal_smoothing=1e-4,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
        joint_terms=terms,
        pcg_mode=args.mode,
        use_cuda_graph=args.graph,
        smooth_sdf=smooth_sdf,
    )
    source = model.state()
    control = model.control()
    if args.historical:
        with np.load(old / "physical-input.npz") as data:
            for name in ("joint_q", "joint_qd", "particle_q", "particle_qd"):
                getattr(source, name).assign(data[name])
            control.joint_target_q.assign(data["target_q"])
            control.joint_target_qd.assign(data["target_qd"])
    else:
        _, q, v = targets(curve, mapping, 0.0, 23)
        initial = source.joint_q.numpy()
        for coord, _, target in mapping.values():
            initial[coord] = q[target]
        source.joint_q.assign(initial)
        source.joint_qd.zero_()
    newton.eval_fk(model, source.joint_q, source.joint_qd, source)
    state = run._create_coupled_state(dt=DT, source=source)
    config = {
        "position": args.position,
        "law": args.law,
        "mode": args.mode,
        "graph": args.graph,
        "historical": args.historical,
        "expected_steps": 1 if args.historical else STEPS,
        "dt": DT,
        "globalization": run._globalization,
        "production_candidate": candidate,
        "alignment_policy": run._contact.alignment_policy,
        "publish_policy": run._coupled_publish_policy,
        "allow_nonconservative_energy": run._allow_nonconservative_energy,
        "config_digest": run._kinematics.digest.hex(),
        "actual_config": run._config,
        "old_manifest": manifest,
        "loaded": loaded,
        "joint_terms": terms,
        "contact_capacity": run._coupled_config["guard_capacity"],
        "reset": "Explicit q/qd/x/v import; reconstruct PR2 joint/link/Vsym history, never old-model continuation",
        "python": platform.python_version(),
        "warp": wp.__version__,
        "command": sys.argv,
        "sdf_representation": representation,
        "smooth_sdf_identity": smooth_sdf.identity if smooth_sdf is not None else None,
        "sdf_qualification": str(getattr(args, "sdf_qualification", None)),
        "compact_evidence": compact,
        "smooth_build_seconds": smooth_sdf.build_seconds if smooth_sdf is not None else 0,
    }
    save(output / "configuration.json", config)
    if not compact:
        np.savez_compressed(output / "initial.npz", **state_arrays(state))
        np.savez_compressed(
            output / "contact-identity.npz", shapes=run._contact.shapes.numpy(), faces=model.tri_indices.numpy()
        )
        np.savez_compressed(
            output / "model.npz",
            **{name: value.numpy() for name, value in vars(model).items() if isinstance(value, wp.array)},
        )
    setup_seconds = time.perf_counter() - started
    calls = []
    solve = run._linear.solve_pcg

    def measured(*a, **kw):
        result = solve(*a, **kw)
        calls.append({"iterations": result.iterations, "slots": result.scheduled_slots, "status": result.status})
        return result

    run._linear.solve_pcg = measured
    rows, frames = [], []
    recorded = []
    trajectory_index = {"includes_initial": True, "chunks": [], "total_frames": 0, "published_frames": 0}

    def flush_recording():
        if not recorded:
            return
        first, last = recorded[0][0], recorded[-1][0]
        relative = f"trajectory/frames-{first:06d}-{last:06d}.npz"
        path = output / relative
        path.parent.mkdir(exist_ok=True)
        arrays = {name: np.stack([frame[2][name] for frame in recorded]) for name in recorded[0][2]}
        arrays.update(
            frame_steps=np.array([frame[0] for frame in recorded], np.int64),
            frame_times=np.array([frame[0] * DT for frame in recorded]),
            frame_publication=np.array([frame[1] for frame in recorded], dtype="U16"),
            frame_target_q=np.stack([frame[3] for frame in recorded]),
            frame_target_qd=np.stack([frame[4] for frame in recorded]),
        )
        np.savez_compressed(path, **arrays)
        trajectory_index["chunks"].append(
            {
                "path": relative,
                "count": len(recorded),
                "first_step": first,
                "last_step": last,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
        trajectory_index["total_frames"] += len(recorded)
        trajectory_index["published_frames"] = last
        trajectory_index["last_frame_step"] = last
        save(output / "trajectory-index.json", trajectory_index)
        recorded.clear()

    if candidate and not compact:
        initial = state_arrays(state)
        trajectory_index["state_schema"] = {
            name: {"dtype": str(value.dtype), "shape": value.shape} for name, value in initial.items()
        }
        _, q, v = targets(curve, mapping, 0, 23)
        recorded.append((0, "INITIAL", initial, q, v))
    successful_steps = 0
    published_steps = 0
    first_stopped = False
    expected = 1 if args.historical else STEPS
    count = min(args.steps, expected) if args.steps is not None else expected
    weights = model.particle_mass.numpy().astype(float)
    with (output / "steps.jsonl").open("w") as log:
        for step in range(count):
            before = state_arrays(state)
            calls.clear()
            stage = "historical_reset"
            if not args.historical:
                stage, q, v = targets(curve, mapping, (step + 1) * DT, 23)
                control.joint_target_q.assign(q)
                control.joint_target_qd.assign(v)
            row = {
                "step": step + 1,
                "time": (step + 1) * DT,
                "stage": stage,
                "success": False,
                "target_q": control.joint_target_q.numpy(),
                "target_qd": control.joint_target_qd.numpy(),
            }
            begin = time.perf_counter()
            try:
                run.step(state, state, control, dt=DT)
                row["solve_seconds"] = time.perf_counter() - begin
                row.update(stats=run.last_stats, pcg_calls=list(calls), success=run.last_stats.converged)
                row.update(
                    published=not run.last_stats.rolled_back,
                    publication=run.last_stats.variational["publication_status"],
                )
                successful_steps += int(row["success"])
                if row["published"]:
                    published_steps += 1
                    if not row["success"] and not first_stopped:
                        np.savez_compressed(output / "first-stop-input.npz", **before)
                        np.savez_compressed(output / "first-stop-published.npz", **state_arrays(state))
                        save(output / "first-stop.json", row)
                        first_stopped = True
                    fields = measure(run, state)
                    history = state._monolithic_coupled
                    pf, bf = history.contact_particle_force.numpy(), history.contact_body_force.numpy()
                    projected = np.einsum("bni,bi->n", run._kinematics.jt.numpy(), bf[:, :3])
                    projected += np.einsum("bni,bi->n", run._kinematics.jw.numpy(), bf[:, 3:])
                    moment = np.cross(fields["x"], pf).sum(axis=0)
                    moment += (np.cross(fields["com"], bf[:, :3]) + bf[:, 3:]).sum(axis=0)
                    row.update({k: v for k, v in fields.items() if k not in ("x", "v", "rotation")})
                    row.update(
                        cube_com=np.average(fields["x"], axis=0, weights=weights),
                        moment_imbalance=moment,
                        q_projection_error=projected - history.contact_joint_force.numpy(),
                        x_projection_error=run._contact.residual.numpy()[23:].reshape(-1, 3) + pf,
                        active_keys=np.flatnonzero(np.any(run._contact.weights.numpy() != 0, axis=1)),
                        force_generation=int(history.force_generation.numpy()[0]),
                        force_valid=int(history.force_valid.numpy()[0]),
                        mask=mask_statistics(run._contact),
                    )
                    slide = run._contact.current.y.numpy() - run._contact.start.y.numpy()
                    normal = run._contact.normal.numpy()
                    tangent = slide - np.einsum("ni,ni->n", slide, normal)[:, None] * normal
                    friction = run._contact.weights.numpy()[:, 1, None] * tangent
                    row["friction_force_sum_norm"] = float(np.linalg.norm(friction, axis=1).sum())
                    if candidate and not compact:
                        recorded.append((step + 1, row["publication"], state_arrays(state), q.copy(), v.copy()))
                        if len(recorded) == 100:
                            flush_recording()
                    elif not compact:
                        frames.append({k: fields[k] for k in ("q", "qd", "x", "v", "com", "rotation")})
            except Exception:
                row["success"] = False
                row["published"] = False
                row["exception"] = traceback.format_exc()
            row["seconds"] = time.perf_counter() - begin
            row["memory_bytes"] = wp.get_mempool_used_mem_current(model.device)
            if compact:
                retained = {
                    "step",
                    "time",
                    "stage",
                    "success",
                    "published",
                    "publication",
                    "mask",
                    "cube_com",
                    "q_projection_error",
                    "x_projection_error",
                    "stats",
                    "pcg_calls",
                    "seconds",
                    "solve_seconds",
                    "memory_bytes",
                    "exception",
                    "target_q",
                    "target_qd",
                    "penetration",
                    "force_balance",
                    "force_valid",
                    "force_generation",
                    "moment_imbalance",
                    "friction_force_sum_norm",
                }
                row = {key: value for key, value in row.items() if key in retained}
            rows.append(serial(row))
            log.write(json.dumps(serial(row), allow_nan=False) + "\n")
            log.flush()
            if not row.get("published", False):
                after = state_arrays(state)
                rollback = set(before) == set(after) and all(np.array_equal(before[k], after[k]) for k in before)
                np.savez_compressed(output / "failure-input.npz", **before)
                for name in ("_original", "_accepted", "_trial"):
                    np.savez_compressed(output / (name[1:] + "-failure.npz"), **state_arrays(getattr(run, name)))
                matrix = run._linear.k_global_scalar_bsr
                buffers = {name: getattr(matrix, name).numpy() for name in ("offsets", "columns", "values")}
                buffers.update(
                    {name: getattr(run, name).numpy() for name in ("_rhs", "_delta", "_post_r", "_post_error")}
                )
                buffers["scale"] = run._linear.scale.numpy()
                buffers.update(
                    {
                        "stage." + name: value.numpy()
                        for name, value in vars(run._objective).items()
                        if isinstance(value, wp.array)
                    }
                )
                np.savez_compressed(output / "failure-buffers.npz", **buffers)
                save(output / "failure.json", {"row": rows[-1], "rollback_preserved": rollback})
                break
            if (step + 1) % 100 == 0:
                print(args.position, args.law, args.mode, args.graph, step + 1, flush=True)
    np.savez_compressed(output / "final.npz", **state_arrays(state))
    flush_recording()
    if frames:
        np.savez_compressed(
            output / "trajectory.npz",
            time=np.arange(1, len(frames) + 1) * DT,
            **{k: np.asarray([f[k] for f in frames]) for k in frames[0]},
        )
    strict_complete = complete_trace(rows, expected)
    complete = complete_publication(rows, expected) if candidate else strict_complete
    if candidate and not compact:
        trajectory_index.update(complete_close=complete, status="COMPLETE" if complete else "STOPPED_ON_FAILURE")
        save(output / "trajectory-index.json", trajectory_index)
    summary = {
        "complete": complete,
        "strict_complete": strict_complete,
        "published_steps": published_steps,
        "publication_counts": {
            label: sum(row.get("publication") == label for row in rows)
            for label in ("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE")
        },
        "physical_accuracy": "NOT_CERTIFIED" if candidate else "SEE_T4_AND_SCOPE",
        "successful_steps": successful_steps,
        "attempted_steps": len(rows),
        "expected_steps": expected,
        "historical": args.historical,
        "position": args.position,
        "law": args.law,
        "mode": args.mode,
        "graph": args.graph,
        "setup_seconds": setup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "discrete_close": "PASS" if strict_complete and not args.historical else "NOT_CERTIFIED",
        "close_publication": "PASS" if complete and not args.historical else "NOT_CERTIFIED",
        "continuous_accuracy": "NOT_CERTIFIED",
        "peak_memory_bytes": wp.get_mempool_used_mem_high(model.device),
        "sdf_representation": representation,
        "solve_seconds": sum(row.get("solve_seconds", 0) for row in rows),
        "first_step_solve_seconds": rows[0].get("solve_seconds") if rows else None,
        "steady_solve_seconds": sum(row.get("solve_seconds", 0) for row in rows[1:]),
        "worst_step_solve_seconds": max((row.get("solve_seconds", 0) for row in rows), default=0),
    }
    save(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", type=int, choices=(18, 24), required=True)
    parser.add_argument("--law", choices=("C-EE", "C-IE", "C-II"), required=True)
    parser.add_argument("--mode", choices=("diagnostic", "production"), default="production")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdf-representation", choices=("trilinear", "c1-hermite", "c2-bspline"), default="trilinear")
    parser.add_argument("--sdf-qualification", type=Path)
    parser.add_argument(
        "--production-candidate", action="store_true", help="Opt in to C2/mask/OR/last-accepted publication"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Save per-step statistics, terminal and failed states without duplicate trajectories",
    )
    args = parser.parse_args()
    if args.production_candidate and (args.sdf_representation != "c2-bspline" or args.historical):
        parser.error("Production candidate requires c2-bspline and a from-zero trajectory")
    if args.sdf_representation != "trilinear" and args.sdf_qualification is None:
        parser.error("A smooth candidate requires --sdf-qualification")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("newton", "scripts/monolithic_reference")
        for p in (REPO / folder).rglob("*.py")
    }
    save(args.output / "source-hashes.json", sources)
    save(args.output / "command.json", {"command": sys.argv, "cwd": str(REPO)})
    try:
        return 0 if run(args) else 1
    except Exception:
        save(
            args.output / "summary.json",
            {"complete": False, "initialization_or_diagnostic_error": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
