# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record collision quality and solver cost for the manual 64-box nightly case."""

import gzip
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import save, serial
from scripts.monolithic_reference.articulation_oracle import transform
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit
from scripts.monolithic_reference.profiling import capture_count
from scripts.monolithic_reference.rigid_stack_scene import DT, GAP, HALF_EXTENT, MASS, WEIGHT, stack_scene, stack_solver
from scripts.monolithic_reference.run_baseline import sha

CASE = Path(__file__).with_name("fixtures") / "rigid_stack_large_v1.json"


def box_penetration(com, rotations):
    """Measure ground and all OBB overlaps independently of spatial query/admission."""
    ground = max(0.0, float(-(com[:, 2] - HALF_EXTENT * abs(rotations[:, 2, :]).sum(axis=1)).min()))
    a, b = np.triu_indices(len(com), 1)
    ra, rb = rotations[a], rotations[b]
    axes = np.concatenate(
        (
            ra.transpose(0, 2, 1),
            rb.transpose(0, 2, 1),
            np.cross(ra.transpose(0, 2, 1)[:, :, None], rb.transpose(0, 2, 1)[:, None]).reshape(-1, 9, 3),
        ),
        axis=1,
    )
    lengths = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(lengths[:, :, None], 1e-15)
    radius = HALF_EXTENT * (
        abs(np.einsum("pai,pij->paj", axes, ra)).sum(axis=2) + abs(np.einsum("pai,pij->paj", axes, rb)).sum(axis=2)
    )
    overlap = radius - abs(np.einsum("pai,pi->pa", axes, com[b] - com[a]))
    overlap[lengths < 1e-12] = np.inf
    depth = overlap.min(axis=1)
    return ground, max(0.0, float(depth.max(initial=0)))


def expected_loads(columns, layers):
    """Compute vertical pair reactions directly from supported box weights."""
    ground = columns * layers
    loads = {}
    for column in range(columns):
        bottom = column * layers
        loads[bottom, ground] = layers * WEIGHT
        for layer in range(layers - 1):
            loads[bottom + layer, bottom + layer + 1] = -(layers - layer - 1) * WEIGHT
    return loads


def measure(run, state, expected, layers):
    """Read native motion/forces and independent geometry after the timed step."""
    h = state._monolithic_coupled
    com, quaternion = h.rigid_com.numpy(), h.rigid_rotation.numpy()
    velocity, omega = h.rigid_v_com.numpy(), h.rigid_omega.numpy()
    forces = run.get_contact_forces(state).body_wrench[run._rigid.body_indices]
    rotations = np.asarray([transform(np.r_[np.zeros(3), q])[:3, :3] for q in quaternion])
    ground, pair = box_penetration(com, rotations)
    load, active = {}, {"R-R": 0, "rigid-static": 0}
    balance = torque_balance = peak_force = 0.0
    for entry in run.last_stats.variational["actor_contact_pairs"]:
        wrench = np.asarray(entry["world_wrench"])
        norm = np.linalg.norm(wrench[:, :3], axis=1)
        balance = max(balance, np.linalg.norm(wrench[:, :3].sum(axis=0)) / max(1, norm.sum()))
        torque_balance = max(
            torque_balance,
            np.linalg.norm(wrench[:, 3:].sum(axis=0)) / max(1e-3, np.linalg.norm(wrench[:, 3:], axis=1).sum()),
        )
        peak_force = max(peak_force, float(norm.max()))
        if norm.max() > 1e-4:
            active[entry["kind"]] += 1
        key = entry["source"], entry["target"]
        if key in expected:
            load[key] = float(wrench[0, 2])
    pool, query = run._contact.evaluation_diagnostics(), run._contact.spatial_diagnostics()
    finite = all(np.isfinite(a).all() for a in (com, quaternion, velocity, omega, forces))
    row = {
        "finite": finite,
        "ground_penetration_m": ground,
        "pair_penetration_m": pair,
        "force_balance_relative": float(balance),
        "torque_balance_relative": float(torque_balance),
        "active_pairs": active,
        "max_pair_force_n": peak_force,
        "surface_speed_m_s": float(
            (np.linalg.norm(velocity, axis=1) + np.sqrt(3) * HALF_EXTENT * np.linalg.norm(omega, axis=1)).max()
        ),
        "kinetic_energy_j": float(
            0.5 * MASS * (velocity**2).sum() + MASS * (2 * HALF_EXTENT) ** 2 / 12 * (omega**2).sum()
        ),
        "body_weight_error_relative": float(abs(forces[:, 2] / WEIGHT - 1).max()),
        "layer_load_error_relative": max(abs(load[key] / value - 1) for key, value in expected.items()),
        "layer_order": bool((np.diff(com[:, 2].reshape(-1, layers), axis=1) > 0).all()),
        "pool": pool,
        "query": query,
        "force_generation_valid": bool(
            h.force_valid.numpy()[0] == 1 and h.force_generation.numpy()[0] == run.last_stats.step_generation
        ),
    }
    pose = {"com": com, "rotation": quaternion, "velocity": velocity, "omega": omega, "contact_wrench": forces}
    return row, pose, load


def step_errors(row, gates):
    """Reject nonfinite geometry, overflow, penetration and inconsistent publication."""
    errors = []
    for flag in ("finite", "force_generation_valid", "layer_order"):
        if not row[flag]:
            errors.append(flag)
    for name in ("ground_penetration_m", "pair_penetration_m"):
        if not np.isfinite(row[name]) or row[name] > gates["penetration_m"]:
            errors.append(name)
    for name in ("force_balance_relative", "torque_balance_relative"):
        if not np.isfinite(row[name]) or row[name] > gates["balance_relative"]:
            errors.append(name)
    if row["pool"]["tainted"] or row["pool"]["dropped"] or row["query"]["tainted"] or row["query"]["capacity_dropped"]:
        errors.append("capacity_overflow")
    return errors


def record_run(output, *, max_steps=None, device="cuda:0", precision="fp64"):
    """Run the frozen recipe; shortened runs remain PARTIAL and failed states remain recorded."""
    case, identity = json.loads(CASE.read_text()), sha(CASE)
    if (case["dt"], case["half_extent_m"], case["mass_kg"], case["gap_m"], case["gravity_m_s2"]) != (
        DT,
        HALF_EXTENT,
        MASS,
        GAP,
        9.81,
    ) or case["factor_capacity"] != 3 * case["support_capacity"]:
        raise ValueError("Rigid fixture differs from supported physical/resource recipe")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("Prefix length must be positive")
    columns = [
        (x * case["pitch_m"], y * case["pitch_m"]) for x, y in itertools.product(*(range(n) for n in case["grid"]))
    ]
    model = stack_scene(device, columns=columns, layers=case["layers"])
    graph = model.device.is_cuda
    run = stack_solver(
        model,
        spatial=True,
        graph=graph,
        query_capacity=case["query_capacity"],
        support_capacity=case["support_capacity"],
        precision=precision,
    )
    # Only full-reference batch scratch is needed by the offline auditor.
    full = stack_solver(model, spatial=False, graph=graph, support_capacity=0, precision=precision)
    audit = ContactSystemAudit(run, full)
    state, control = run.create_state(dt=DT), model.control()
    recording = output / "recording"
    recording.mkdir()
    save(
        recording / "scene.json",
        {
            "case": case,
            "columns": columns,
            "body_labels": model.body_label,
            "config_digest": run._kinematics.digest.hex(),
            "effective_policy": asdict(run._config),
            "recipe_sha256": identity,
        },
    )
    expected = expected_loads(len(columns), case["layers"])
    initial = state._monolithic_coupled.rigid_com.numpy().copy()
    rows, hold, seen, chunks, frames = [], [], set(), [], []
    counts = {"CONVERGED": 0, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0}
    errors, full_checks = [], 0
    limit = min(max_steps, case["steps"]) if max_steps is not None else case["steps"]

    def flush():
        if frames:
            path = recording / f"poses-{frames[0]['step']:05d}.npz"
            np.savez_compressed(path, **{k: np.asarray([f[k] for f in frames]) for k in frames[0]})
            chunks.append({"file": path.name, "frames": len(frames), "sha256": sha(path)})
            frames.clear()

    try:
        with gzip.open(output / "steps.jsonl.gz", "wt") as log:
            for step in range(1, limit + 1):
                if graph:
                    wp.synchronize_device(model.device)
                captures = capture_count(run)
                plan = getattr(run._linear, "_masked_pcg_plan", None)
                begin = time.perf_counter()
                run.step(state, state, control, dt=DT)
                if graph:
                    wp.synchronize_device(model.device)
                wall_ms = (time.perf_counter() - begin) * 1000
                stats = run.last_stats
                publication = (
                    "CONVERGED" if stats.converged else "HARD_FAILURE" if stats.rolled_back else "STOPPED_ACCEPTED"
                )
                counts[publication] += 1
                row = {
                    "step": step,
                    "wall_ms": wall_ms,
                    "publication": publication,
                    "new_captures": capture_count(run) - captures,
                    "pcg_plan_changed": plan is not getattr(run._linear, "_masked_pcg_plan", None),
                    "newton_iterations": stats.nonlinear_iterations,
                    "pcg_iterations": stats.linear_iterations,
                    "trials": len(stats.variational["trials"]),
                    "rejected_trials": sum(not t.get("accepted", False) for t in stats.variational["trials"]),
                }
                if stats.converged:
                    values, pose, loads = measure(run, state, expected, case["layers"])
                    row.update(values)
                    frames.append({"step": step, **pose})
                    seen.update(key for key in expected if abs(loads[key]) > case["gates"]["min_pair_force_n"])
                    errors = step_errors(row, case["gates"])
                    if step > case["steps"] - case["hold_steps"]:
                        hold.append(pose["com"].copy())
                        if (
                            max(row["body_weight_error_relative"], row["layer_load_error_relative"])
                            > case["gates"]["support_relative"]
                        ):
                            errors.append("static_support")
                        if row["surface_speed_m_s"] > case["gates"]["surface_speed_m_s"]:
                            errors.append("settling_speed")
                    if not errors and (
                        step in case["audit_steps"] or step % case["audit_stride"] == 0 or step == limit
                    ):
                        row["full_system"] = audit.evaluate()
                        full_checks += 1
                        if row["full_system"]["status"] != "PASS_FULL_RESIDUAL":
                            errors.append("full_system_residual")
                else:
                    errors = [publication, stats.failure_reason]
                row["errors"] = errors
                rows.append(row)
                log.write(json.dumps(serial(row), allow_nan=False) + "\n")
                if len(frames) == 100:
                    flush()
                if errors:
                    break
    finally:
        flush()
        run.save_checkpoint(state, recording / "final.npz")
    if max_steps is None and not errors:
        final = state._monolithic_coupled.rigid_com.numpy()
        if seen != set(expected) or not ((initial - final)[:, 2] > GAP / 2).all():
            errors.append("missing_collision_or_fall")
        if len(hold) != case["hold_steps"] or np.ptp(np.asarray(hold), axis=0).max() > case["gates"]["hold_drift_m"]:
            errors.append("hold_drift")
        if abs(final[:, :2] - initial[:, :2]).max() > case["gates"]["lateral_drift_m"]:
            errors.append("lateral_drift")
    if sha(CASE) != identity:
        errors.append("recipe_changed_during_run")
    warm = [
        r["wall_ms"]
        for r in rows
        if r["step"] > case["warmup_steps"] and not r["new_captures"] and not r["pcg_plan_changed"]
    ]
    performance = {
        "qualification": "OBSERVED_NO_PERFORMANCE_THRESHOLD",
        "warmup_steps": case["warmup_steps"],
        "scope": "synchronized solver.step wall time; excludes audits, metrics and recording",
        "all_step_mean_ms": float(np.mean([r["wall_ms"] for r in rows])),
        "warm_samples": len(warm),
        "warm_median_ms": float(np.median(warm)) if warm else None,
        "warm_p95_ms": float(np.percentile(warm, 95)) if warm else None,
        "warm_p99_ms": float(np.percentile(warm, 99)) if warm else None,
        "warm_fps": 1000 / float(np.mean(warm)) if warm else None,
        "capture_steps": sum(bool(r["new_captures"]) for r in rows),
        "pcg_plan_change_steps": sum(r["pcg_plan_changed"] for r in rows),
        "collision_window_median_ms": float(
            np.median([r["wall_ms"] for r in rows if r["step"] <= case["warmup_steps"]])
        ),
        "collision_window_includes_captures": True,
    }
    metrics = {
        "bodies": model.body_count,
        "expected_pairs": len(expected),
        "seen_pairs": len(seen),
        "full_residual_checks": full_checks,
        "peak_penetration_m": max(max(r.get("ground_penetration_m", 0), r.get("pair_penetration_m", 0)) for r in rows),
        "peak_active_rr_pairs": max(r.get("active_pairs", {}).get("R-R", 0) for r in rows),
        "peak_active_static_pairs": max(r.get("active_pairs", {}).get("rigid-static", 0) for r in rows),
        "peak_query_candidates": max(r.get("query", {}).get("candidates", 0) for r in rows),
        "peak_contacts": max(r.get("pool", {}).get("admitted", 0) for r in rows),
        "query_dropped": sum(r.get("query", {}).get("capacity_dropped", 0) for r in rows),
        "pool_dropped": sum(r.get("pool", {}).get("dropped", 0) for r in rows),
    }
    summary = {
        "status": "FAIL" if errors else "PARTIAL" if max_steps is not None else "PASS",
        "counts": counts,
        "published_steps": counts["CONVERGED"],
        "planned_steps": case["steps"],
        "errors": errors,
        "performance": performance,
        "metrics": metrics,
        "effective_contact_dataflow": "spatial",
        "precision": run.precision.value,
        "scene_input_sha256": identity,
        "reference": {"policy": "ANALYTIC_STACK_PHYSICS; NO_FROZEN_TRAJECTORY"},
        "recipe": {
            "case": case,
            "config_digest": run._kinematics.digest.hex(),
            "resources": run._contact.resource_identity.tolist(),
            "effective_policy": asdict(run._config),
        },
        "environment": {"warp": wp.__version__, "numpy": np.__version__, "device": model.device.name},
        "recording": {"counts": counts, "chunks": chunks},
        "steps_sha256": sha(output / "steps.jsonl.gz"),
    }
    save(output / "scene-summary.json", summary)
    return serial(summary)
