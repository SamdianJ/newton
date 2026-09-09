# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic P2-0A/P2-0B Sharpa study. Does not modify solver sources."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_sharpa_grasp import ANCHORED_FIXTURE, GraspCase
from scripts.monolithic_reference.profile_release import _StageProfile

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/lightwheel/Desktop/newton/SamiulJ")
DEFAULT_OUTPUT = WORKSPACE / "agents/integration/artifacts/p2/p2-0"
ASSET_DIR = WORKSPACE / "assets/left_sharpa_wave"
CONTACT_DIR = WORKSPACE / "agents/integration/artifacts/p1/pr6d-assets/hand"
TRAJECTORY = WORKSPACE / "assets/trajectory/Ball_catch - Left_hand_motion_rad.csv"
CALIBRATION = ROOT / "scripts/monolithic_reference/fixtures/sharpa_contact_calibration_v1.json"
FRAME_DT = 0.01
DURATION = 4.5
NEWTON_SWEEP = (1, 2, 3, 4, 6, 8, 10, 16, 24)
DT_SWEEP = (
    (1, 0.01),
    (2, 0.005),
    (4, 0.0025),
    (5, 0.002),
    (10, 0.001),
    (20, 0.0005),
    (40, 0.00025),
)
CURVE_TIMES = (0.1, 1.5, 3.5)
SNAPSHOT_TIMES = (0.5, 2.5, 4.5)
STAGE_WINDOWS = {"prepare": (0.05, 0.07), "close": (1.00, 1.02), "hold": (3.50, 3.52)}
HEAVY_KEYS = (
    "q",
    "qd",
    "q_target",
    "qd_target",
    "tracking_error",
    "limit_violation",
    "history",
    "query_counts",
    "palm_force",
    "support_force",
    "pd_force",
    "pd_saturated",
    "limit_force",
    "friction_force",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def mempool(device):
    if not getattr(device, "is_cuda", False) or not getattr(device, "is_mempool_supported", False):
        return None
    return {
        "used_bytes": int(wp.get_mempool_used_mem_current(device)),
        "high_bytes": int(wp.get_mempool_used_mem_high(device)),
    }


def environment():
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as error:
        gpu = f"unavailable: {error}"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "warp": getattr(wp, "__version__", None),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "asset_dir": str(ASSET_DIR),
        "contact_dir": str(CONTACT_DIR),
        "trajectory_sha256": sha256(TRAJECTORY),
        "calibration_sha256": sha256(CALIBRATION),
    }


def args_for(mesh, output):
    return SimpleNamespace(
        device="cuda:0",
        asset_dir=str(ASSET_DIR),
        contact_dir=str(CONTACT_DIR),
        trajectory=str(TRAJECTORY),
        calibration=str(CALIBRATION),
        experiment="anchored-close",
        ball=str(ROOT / "scripts/monolithic_reference/fixtures/soft_ball_25mm" / f"ball_{mesh}.npz"),
        ball_radius=0.025,
        friction_off=False,
        disable_aabb=False,
        output=Path(output),
        viewer="null",
        headless=True,
    )


def slim_record(record, stats, curve):
    slim = {key: value for key, value in record.items() if key not in HEAVY_KEYS}
    slim.update(
        nonlinear_iterations=stats.nonlinear_iterations,
        linear_iterations=stats.linear_iterations,
        line_search_iterations=stats.line_search_iterations,
        regularization_retries=stats.regularization_retries,
        matrix_assembly_count=stats.matrix_assembly_count,
        matrix_nnz=stats.matrix_nnz,
        triplet_count=stats.triplet_count,
        rho=stats.rho,
        rho_q=stats.rho_q,
        rho_x=stats.rho_x,
        raw_residual_q_norm=getattr(stats, "raw_residual_q_norm", None),
        raw_residual_x_norm=getattr(stats, "raw_residual_x_norm", None),
        scale_generation=getattr(stats, "scale_generation", None),
        pcg_call_count=len(record["pcg_calls"]),
        pcg_iterations_sum=sum(call["iterations"] for call in record["pcg_calls"]),
        pcg_iterations_max=max((call["iterations"] for call in record["pcg_calls"]), default=0),
        pcg_statuses=[call["status"] for call in record["pcg_calls"]],
        newton_curve=curve,
        finger_force_components=record["finger_forces"],
    )
    slim.pop("pcg_calls", None)
    slim.pop("finger_forces", None)
    return slim


def quality(records, summary, dt):
    gates = ANCHORED_FIXTURE["gates"]
    accepted = [row for row in records if not row["rollback"]]
    soft = sum(row["status"] == "NONLINEAR_MAX_ITERATIONS" for row in accepted)
    hard = int(bool(summary.get("failure")))
    return {
        "complete_schedule": summary["complete_schedule"],
        "stability_passed": summary.get("stability_passed", False),
        "accepted_steps": summary["accepted_steps"],
        "expected_steps": round(DURATION / dt),
        "physical_coverage_s": summary["accepted_steps"] * dt,
        "failure": summary.get("failure"),
        "converged_fraction": summary["converged_fraction"],
        "normal_convergence_fraction": float(np.mean([row["status"] == "SUCCESS" for row in accepted]))
        if accepted
        else 0.0,
        "soft_stop_count": soft,
        "soft_stop_not_normal": True,
        "hard_failure": hard,
        "hold_contact_fraction": summary.get("hold_contact_fraction"),
        "maximum_penetration_m": summary.get("maximum_penetration_m"),
        "minimum_det_f": summary.get("minimum_det_f"),
        "pcg_iteration_p95": summary.get("pcg_iteration_p95"),
        "patch_schur_trigger": summary.get("patch_schur_trigger"),
        "gates": gates,
    }


def work_stats(records, dt):
    if not records:
        return {}
    by_stage = {}
    for stage in ("prepare", "close", "hold"):
        rows = [row for row in records if row["stage"] == stage]
        if not rows:
            continue
        body = rows[1:] if len(rows) > 1 else rows

        def pct(name, body=body):
            values = [row[name] for row in body]
            return {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }

        by_stage[stage] = {
            "count": len(rows),
            "step_ms": pct("step_seconds")
            if False
            else {
                "p50": 1000 * float(np.percentile([r["step_seconds"] for r in body], 50)),
                "p95": 1000 * float(np.percentile([r["step_seconds"] for r in body], 95)),
                "mean": 1000 * float(np.mean([r["step_seconds"] for r in body])),
            },
            "nonlinear_iterations": pct("nonlinear_iterations"),
            "linear_iterations": pct("linear_iterations"),
            "matrix_assembly_count": pct("matrix_assembly_count"),
            "pcg_iterations_sum": pct("pcg_iterations_sum"),
            "limit_hit_fraction": float(np.mean([r["status"] == "NONLINEAR_MAX_ITERATIONS" for r in rows])),
        }
    duration = records[-1]["time"] if records else 0.0
    duration = max(duration, dt)
    return {
        "stages": by_stage,
        "per_simulated_second": {
            "newton_updates": float(sum(r["nonlinear_iterations"] for r in records) / duration),
            "assemblies": float(sum(r["matrix_assembly_count"] for r in records) / duration),
            "pcg_iterations": float(sum(r["pcg_iterations_sum"] for r in records) / duration),
            "wall_seconds": float(sum(r["step_seconds"] for r in records) / duration),
        },
        "solver_only_fps_at_frame_dt": FRAME_DT
        / (sum(r["step_seconds"] for r in records) / max(len(records), 1) * round(FRAME_DT / dt)),
    }


def common_frames(records, dt):
    wanted = {round(i * FRAME_DT, 10) for i in range(round(DURATION / FRAME_DT) + 1)}
    frames = []
    for row in records:
        t = round(row["time"], 10)
        if t in wanted or abs((row["time"] / FRAME_DT) - round(row["time"] / FRAME_DT)) < 1e-9:
            frames.append(
                {
                    "time": row["time"],
                    "stage": row["stage"],
                    "ball_com": row["ball_com"],
                    "deformation_rms": row["deformation_rms"],
                    "finger_force_n": row["finger_force_n"],
                    "finger_force_components": row.get("finger_force_components"),
                    "penetration": row["penetration"],
                    "min_det_f": row["min_det_f"],
                    "q_norm": None,
                }
            )
    peaks = {
        "penetration": max((r["penetration"] for r in records), default=None),
        "finger_force_n": max((r["finger_force_n"] for r in records), default=None),
        "deformation_rms": max((r["deformation_rms"] for r in records), default=None),
        "min_det_f": min((r["min_det_f"] for r in records), default=None),
    }
    return {"frame_dt_s": FRAME_DT, "samples": frames, "per_step_peaks": peaks}


def run_case(job, output, *, nsys_window=None, stage_profile=False):
    output = Path(output)
    if (output / "compact.json").exists() and not job.get("overwrite"):
        existing = json.loads((output / "compact.json").read_text())
        if existing.get("complete") or existing.get("failure"):
            print(f"skip completed {job['name']}", flush=True)
            return existing
    output.mkdir(parents=True, exist_ok=True)
    h = float(job["physical_dt_s"])
    newton_limit = int(job["newton_max_iterations"])
    setup_start = time.perf_counter()
    case = GraspCase(args_for(job["mesh"], output))
    case.fixture = {**ANCHORED_FIXTURE, "dt": h, "duration": DURATION}
    case.total_steps = round(DURATION / h)
    case.solver.newton_max_iterations = newton_limit
    device = case.model.device
    if device.is_cuda and hasattr(wp, "reset_mempool_used_mem_high"):
        wp.reset_mempool_used_mem_high(device)
    setup_s = time.perf_counter() - setup_start
    original_current = case.solver._evaluate_current
    curve_buffer = []

    def current_with_curve(*args, **kwargs):
        result = original_current(*args, **kwargs)
        curve_buffer.append(
            {
                "merit": float(result.merit),
                "merit_q": float(result.merit_q),
                "merit_x": float(result.merit_x),
                "min_det_f": float(result.min_det_f),
            }
        )
        return result

    traces = []
    snapshots = {}
    window_profiles = {}
    profiler = None
    nsys_active = False
    wall_start = time.perf_counter()
    try:
        while case.step_count < case.total_steps and not case.failure:
            next_t = (case.step_count + 1) * h
            want_curve = any(abs(next_t - t) < 0.5 * h for t in CURVE_TIMES)
            case.solver._evaluate_current = current_with_curve if want_curve else original_current
            curve_buffer.clear()
            in_stage_window = stage_profile and any(lo < next_t <= hi + 1e-12 for lo, hi in STAGE_WINDOWS.values())
            if stage_profile and in_stage_window and profiler is None:
                profiler = _StageProfile(case.solver, "actor_block")
                profiler.__enter__()
            if nsys_window:
                lo, hi = STAGE_WINDOWS[nsys_window]
                if not nsys_active and lo < next_t <= hi + 1e-12:
                    wp.cuda_profiler_start(device)
                    nsys_active = True
                elif nsys_active and next_t > hi + 1e-12:
                    wp.cuda_profiler_stop(device)
                    nsys_active = False
            case.step()
            if not case.records:
                break
            stats = case.solver.last_stats
            slim = slim_record(case.records[-1], stats, list(curve_buffer) if want_curve else None)
            case.records[-1] = slim
            traces.append(slim)
            t = slim["time"]
            if any(abs(t - mark) < 0.5 * h for mark in SNAPSHOT_TIMES):
                snapshots[f"{t:.3f}"] = {
                    "q": case.state.joint_q.numpy().copy(),
                    "qd": case.state.joint_qd.numpy().copy(),
                    "x": case.state.particle_q.numpy().copy(),
                    "v": case.state.particle_qd.numpy().copy(),
                }
            if case.step_count % 250 == 0 or case.failure:
                print(
                    f"{job['name']}: t={t:.3f}s force={slim['finger_force_n']:.4f}N "
                    f"pen={1000 * slim['penetration']:.3f}mm detF={slim['min_det_f']:.3f} "
                    f"nl={slim['nonlinear_iterations']} status={slim['status']}",
                    flush=True,
                )
            if stage_profile and profiler is not None:
                still = any(lo < (case.step_count + 1) * h <= hi + 1e-12 for lo, hi in STAGE_WINDOWS.values())
                if not still:
                    profiler.__exit__(None, None, None)
                    for name, (lo, hi) in STAGE_WINDOWS.items():
                        if lo < t <= hi + 1e-12:
                            window_profiles[name] = profiler.summary()
                    profiler = None
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if nsys_active:
            wp.cuda_profiler_stop(device)
    finally:
        case.solver._evaluate_current = original_current
    wall = time.perf_counter() - wall_start
    wp.synchronize_device(device)
    pool = mempool(device)
    gates = ANCHORED_FIXTURE["gates"]
    hold = [row for row in traces if row["stage"] == "hold"]
    pcg_p95 = {}
    for stage in ("close", "hold"):
        calls = [row["pcg_iterations_max"] for row in traces if row["stage"] == stage]
        pcg_p95[stage] = float(np.percentile(calls, 95)) if calls else None
    summary = {
        "complete_schedule": case.step_count == case.total_steps and case.failure is None,
        "accepted_steps": case.step_count,
        "failure": case.failure,
        "converged_fraction": float(np.mean([row["converged"] for row in traces])) if traces else 0.0,
        "hold_contact_fraction": float(
            np.mean([row["finger_force_n"] >= gates["minimum_finger_force_n"] for row in hold])
        )
        if hold
        else 0.0,
        "maximum_penetration_m": max((row["penetration"] for row in traces), default=None),
        "minimum_det_f": min((row["min_det_f"] for row in traces), default=None),
        "pcg_iteration_p95": pcg_p95,
        "patch_schur_trigger": any(v is not None and v > 64 for v in pcg_p95.values()),
        "stability_passed": False,
    }
    if traces:
        summary["stability_passed"] = bool(
            summary["complete_schedule"]
            and summary["converged_fraction"] >= gates["converged_fraction"]
            and summary["hold_contact_fraction"] >= gates["hold_contact_fraction"]
            and all(
                row["anchor_unchanged"]
                and row["penetration"] <= gates["penetration_m"]
                and row["min_det_f"] >= gates["min_det_f"]
                and (
                    not row["converged"]
                    or max(row["residual_ratio"], row["q_residual_ratio"], row["x_residual_ratio"]) <= 1
                )
                for row in traces
            )
        )
    compact = {
        "name": job["name"],
        "task": job["task"],
        "mesh": job["mesh"],
        "physical_dt_s": h,
        "substeps": int(round(FRAME_DT / h)),
        "frame_dt_s": FRAME_DT,
        "newton_max_iterations": newton_limit,
        "setup_seconds": setup_s,
        "run_wall_seconds": wall,
        "wall_seconds_per_simulated_second": wall / max(case.step_count * h, h),
        "complete": bool(summary["complete_schedule"]),
        "failure": summary.get("failure"),
        "quality": quality(traces, summary, h),
        "work": work_stats(traces, h),
        "common_time": common_frames(traces, h),
        "stage_profile_windows": {
            k: {n: {kk: vv for kk, vv in rec.items() if kk != "samples_ms"} for n, rec in v.items()}
            for k, v in window_profiles.items()
        }
        if window_profiles
        else {},
        "mempool": pool,
        "environment": environment(),
        "ball": args_for(job["mesh"], output).ball,
        "ball_sha256": sha256(args_for(job["mesh"], output).ball),
        "manifest": {k: v for k, v in case.manifest.items() if k != "fixture"},
    }
    payload = "".join(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n" for row in traces)
    (output / "trace.jsonl").write_text(payload)
    (output / "trace.jsonl.sha256").write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n")
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")
    (output / "compact.json").write_text(json.dumps(json_safe(compact), indent=2, allow_nan=False) + "\n")
    if snapshots:
        np.savez(output / "common_snapshots.npz", **{k.replace(".", "_"): v for k, v in snapshots.items()})
    np.savez(
        output / "final_state.npz",
        q=case.state.joint_q.numpy(),
        qd=case.state.joint_qd.numpy(),
        x=case.state.particle_q.numpy(),
        v=case.state.particle_qd.numpy(),
    )
    del case
    gc.collect()
    wp.synchronize_device("cuda:0")
    return compact


def jobs():
    rows = []
    for newton in NEWTON_SWEEP:
        rows.append(
            {
                "name": f"0a-newton-r3-n{newton:02d}",
                "task": "0a-newton",
                "mesh": "r3",
                "newton_max_iterations": newton,
                "physical_dt_s": 0.001,
            }
        )
    rows.append(
        {
            "name": "0a-newton-r2-n10",
            "task": "0a-newton-control",
            "mesh": "r2",
            "newton_max_iterations": 10,
            "physical_dt_s": 0.001,
        }
    )
    for substeps, h in DT_SWEEP:
        if substeps == 10:
            continue
        rows.append(
            {
                "name": f"0a-dt-r3-s{substeps:02d}",
                "task": "0a-dt",
                "mesh": "r3",
                "newton_max_iterations": 10,
                "physical_dt_s": h,
            }
        )
    rows.append(
        {
            "name": "0a-dt-r2-s10",
            "task": "0a-dt-control",
            "mesh": "r2",
            "newton_max_iterations": 10,
            "physical_dt_s": 0.001,
        }
    )
    for i in range(1, 5):
        rows.append(
            {
                "name": f"0b-uninstrumented-r3-rep{i + 1:02d}",
                "task": "0b-baseline",
                "mesh": "r3",
                "newton_max_iterations": 10,
                "physical_dt_s": 0.001,
            }
        )
    for i in range(5):
        rows.append(
            {
                "name": f"0b-uninstrumented-r2-rep{i + 1:02d}",
                "task": "0b-baseline",
                "mesh": "r2",
                "newton_max_iterations": 10,
                "physical_dt_s": 0.001,
            }
        )
    rows.append(
        {
            "name": "0a-newton-r3-n10-repeat",
            "task": "0a-repeat",
            "mesh": "r3",
            "newton_max_iterations": 10,
            "physical_dt_s": 0.001,
        }
    )
    rows.append(
        {
            "name": "0b-staged-r3",
            "task": "0b-staged",
            "mesh": "r3",
            "newton_max_iterations": 10,
            "physical_dt_s": 0.001,
            "stage_profile": True,
        }
    )
    rows.append(
        {
            "name": "0b-nsight-r3-hold",
            "task": "0b-nsight",
            "mesh": "r3",
            "newton_max_iterations": 10,
            "physical_dt_s": 0.001,
            "nsys_window": "hold",
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "sharpa")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-nsight", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = jobs()
    if not args.include_nsight:
        selected = [job for job in selected if job.get("task") != "0b-nsight"]
    if args.only:
        selected = [job for job in selected if job["name"] in args.only]
    index = []
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "scope": "P2-0A/P2-0B diagnostic; production solver unmodified",
                "frame_dt_s": FRAME_DT,
                "duration_s": DURATION,
                "newton_sweep": list(NEWTON_SWEEP),
                "dt_sweep": [{"substeps": s, "physical_dt_s": h} for s, h in DT_SWEEP],
                "environment": environment(),
            },
            indent=2,
        )
        + "\n"
    )
    for spec in selected:
        job = {**spec, "overwrite": args.overwrite}
        out = args.output / job["name"]
        print("START", job["name"], flush=True)
        try:
            compact = run_case(
                job,
                out,
                nsys_window=job.get("nsys_window"),
                stage_profile=bool(job.get("stage_profile")),
            )
            index.append({"name": job["name"], "status": "ok", "failure": compact.get("failure"), "output": str(out)})
        except Exception as error:
            index.append({"name": job["name"], "status": "error", "failure": f"{type(error).__name__}: {error}"})
            (out / "error.txt").write_text(f"{type(error).__name__}: {error}\n")
            print("ERROR", job["name"], error, flush=True)
        (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2), flush=True)


if __name__ == "__main__":
    main()
