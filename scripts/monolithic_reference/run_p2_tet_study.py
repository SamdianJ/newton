# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic P2-0B-T tet/gravity study. Does not modify solver sources."""

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
from unittest.mock import patch

import numpy as np
import warp as wp

import newton._src.solvers.monolithic.solver_monolithic as solver_module
from newton._src.solvers.monolithic.tet import assemble_tet_residual_tangent
from newton.examples.softbody.monolithic_tet_response import ResponseCase, gravity_acceleration
from scripts.monolithic_reference.profile_release import _StageProfile, _percentiles

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/lightwheel/Desktop/newton/SamiulJ")
DEFAULT_OUTPUT = WORKSPACE / "agents/integration/artifacts/p2/p2-0/tet"
REFINEMENTS = (3, 4, 5, 6)
PRIMARY = 5
WARMUP = 100
REPEATS = 5
PHASES = ((0.0, 2.0, "loading"), (2.0, 34.0, "hold"), (34.0, 36.0, "tail"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(device):
    gpu = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as error:
        gpu = f"unavailable: {error}"
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "requested_device": device,
        "actual_device": str(wp.get_device(device)),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def mempool(device):
    device = wp.get_device(device)
    if not device.is_cuda or not getattr(device, "is_mempool_supported", False):
        return None
    return {
        "used_bytes": int(wp.get_mempool_used_mem_current(device)),
        "high_bytes": int(wp.get_mempool_used_mem_high(device)),
    }


def phase_of(time_s):
    for start, end, name in PHASES:
        if start <= time_s <= end + 1e-12:
            return name
    return "other"


def summarize_rows(rows):
    if not rows:
        return None
    out = {}
    for key in ("solver_ms", "measurement_ms", "step_ms", "nonlinear_iterations", "linear_iterations"):
        values = [row[key] for row in rows]
        out[key] = {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return out


def diagnostic_run(device, refinement, output):
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    records = []
    failure = None
    t0 = time.perf_counter()
    try:
        for _ in range(round(case.duration / case.dt)):
            timing = case.step(profile=True)
            stats = case.solver.last_stats
            rec = case.records[-1]
            records.append(
                {
                    "time": rec["time"],
                    "phase": phase_of(rec["time"]),
                    "tip_m": rec["tip_m"],
                    "min_detF": rec["min_detF"],
                    "gravity_m_s2": rec.get("gravity_m_s2"),
                    "max_speed_m_s": rec.get("max_speed_m_s"),
                    "converged": rec["converged"],
                    "status": stats.status.name,
                    "reason": stats.failure_reason,
                    "rollback": stats.rolled_back,
                    "nonlinear_iterations": stats.nonlinear_iterations,
                    "linear_iterations": stats.linear_iterations,
                    "matrix_assembly_count": stats.matrix_assembly_count,
                    "residual_ratios": rec["residual_ratios"],
                    **timing,
                }
            )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    wall = time.perf_counter() - t0
    summary = case.summary()
    compact = {
        "kind": "full_diagnostic",
        "requested_device": device,
        "actual_device": str(case.model.device),
        "refinement": refinement,
        "particles": case.model.particle_count,
        "tets": case.model.tet_count,
        "fixed": int(case.fixed.sum()),
        "dynamic": int((~case.fixed).sum()),
        "dt": case.dt,
        "duration_s": case.duration,
        "failure": failure,
        "summary": summary,
        "manifest": case.manifest,
        "wall_seconds": wall,
        "wall_seconds_per_simulated_second": wall / max(case.steps * case.dt, case.dt),
        "by_phase": {
            name: summarize_rows([row for row in records if row["phase"] == name])
            for _, _, name in PHASES
        },
        "overall": summarize_rows(records),
        "mempool": mempool(case.model.device),
        "environment": environment(device),
    }
    output.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
    (output / "trace.jsonl").write_text(payload)
    (output / "compact.json").write_text(json.dumps(compact, indent=2, default=str) + "\n")
    x = case.state.particle_q.numpy().copy()
    del case
    gc.collect()
    return compact, x


def timing_run(device, refinement, output):
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    samples = []
    failure = None
    t0 = time.perf_counter()
    try:
        for _ in range(round(case.duration / case.dt)):
            time_s = (case.steps + 1) * case.dt
            external = np.zeros_like(case.rest, dtype=np.float32)
            case.state.particle_f.assign(external)
            if case.experiment == "gravity":
                case.model.set_gravity((0, 0, -gravity_acceleration(time_s)))
            wp.synchronize_device(case.model.device)
            start = time.perf_counter()
            case.solver.step(case.state, case.state, case.control, None, case.dt)
            wp.synchronize_device(case.model.device)
            solver_ms = 1000 * (time.perf_counter() - start)
            stats = case.solver.last_stats
            if stats.rolled_back:
                failure = f"{stats.status.name}: {stats.failure_reason}"
                break
            case.steps += 1
            samples.append(
                {
                    "time": time_s,
                    "phase": phase_of(time_s),
                    "solver_ms": solver_ms,
                    "measurement_ms": 0.0,
                    "step_ms": solver_ms,
                    "nonlinear_iterations": stats.nonlinear_iterations,
                    "linear_iterations": stats.linear_iterations,
                    "converged": stats.converged,
                    "status": stats.status.name,
                    "min_det_f": stats.min_det_f,
                    "residual_ratio": stats.convergence_ratio,
                }
            )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    wall = time.perf_counter() - t0
    compact = {
        "kind": "solver_only_timing",
        "requested_device": device,
        "actual_device": str(case.model.device),
        "refinement": refinement,
        "particles": case.model.particle_count,
        "tets": case.model.tet_count,
        "failure": failure,
        "accepted_steps": len(samples),
        "wall_seconds": wall,
        "wall_seconds_per_simulated_second": wall / max(len(samples) * case.dt, case.dt),
        "frame_solver_ms_6substeps": {
            name: None
            if not rows
            else {
                "p50": 6 * float(np.percentile([r["solver_ms"] for r in rows], 50)),
                "p95": 6 * float(np.percentile([r["solver_ms"] for r in rows], 95)),
            }
            for name, rows in (
                (label, [r for r in samples if r["phase"] == label]) for _, _, label in PHASES
            )
        },
        "by_phase": {name: summarize_rows([r for r in samples if r["phase"] == name]) for _, _, name in PHASES},
        "overall": summarize_rows(samples),
        "mempool": mempool(case.model.device),
        "environment": environment(device),
        "note": (
            "solver.step with device sync at boundaries; gravity/load applied as in "
            "ResponseCase.step; measure()/viewer/IO excluded. Artifacts collected "
            "before this gravity update are rest-state residual checks, not loaded dynamics."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "compact.json").write_text(json.dumps(compact, indent=2, default=str) + "\n")
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in samples[::6])
    (output / "every_frame.jsonl").write_text(payload)
    del case
    gc.collect()
    return compact


def staged_window(device, refinement, output):
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    for _ in range(50):
        case.step()
    windows = {}
    for label, count in (("loading", 20),):
        with _StageProfile(case.solver, "actor_block") as timer:
            tet_ms = []
            original = assemble_tet_residual_tangent

            def timed_tet(*args, **kwargs):
                wp.synchronize_device(case.model.device)
                start = time.perf_counter()
                result = original(*args, **kwargs)
                wp.synchronize_device(case.model.device)
                tet_ms.append(1000 * (time.perf_counter() - start))
                return result

            with patch.object(solver_module, "assemble_tet_residual_tangent", side_effect=timed_tet):
                for _ in range(count):
                    case.step()
        summary = timer.summary()
        windows[label] = {
            "stages": {k: {kk: vv for kk, vv in rec.items() if kk != "samples_ms"} for k, rec in summary.items()},
            "tet_assemble_ms": _percentiles(tet_ms),
            "tet_per_tet_us": None
            if not tet_ms
            else {
                "p50": 1000 * float(np.percentile(tet_ms, 50)) / case.model.tet_count,
                "p95": 1000 * float(np.percentile(tet_ms, 95)) / case.model.tet_count,
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "staged.json").write_text(json.dumps(windows, indent=2) + "\n")
    x = case.state.particle_q.numpy().copy()
    del case
    gc.collect()
    return windows, x


def material_replay(device, refinement, candidate_x, output):
    from newton.examples.softbody.monolithic_tet_compare import build_case

    rows = []
    for material, mass in (
        ("smith_log_stabilized", "consistent"),
        ("kim_stable_no_log", "consistent"),
        ("smith_log_stabilized", "lumped"),
    ):
        model, solver, state, control, *_ = build_case(
            device,
            material=material,
            mass=mass,
            refinement=refinement,
            direction="axial",
            dt=0.02,
            density=10.0,
        )
        state.particle_q.assign(candidate_x.astype(np.float32))
        original = assemble_tet_residual_tangent
        samples = []

        def timed(*args, _model=model, _original=original, **kwargs):
            wp.synchronize_device(_model.device)
            start = time.perf_counter()
            result = _original(*args, **kwargs)
            wp.synchronize_device(_model.device)
            samples.append(1000 * (time.perf_counter() - start))
            return result

        step_ms = None
        with patch.object(solver_module, "assemble_tet_residual_tangent", side_effect=timed):
            for _ in range(8):
                wp.synchronize_device(model.device)
                t0 = time.perf_counter()
                solver.step(state, state, control, None, 0.02)
                wp.synchronize_device(model.device)
                step_ms = 1000 * (time.perf_counter() - t0)
        rows.append(
            {
                "material": material,
                "mass": mass,
                "tets": model.tet_count,
                "tet_assemble_ms": _percentiles(samples[2:]),
                "tet_per_tet_us_p50": None
                if len(samples) <= 2
                else 1000 * float(np.percentile(samples[2:], 50)) / model.tet_count,
                "step_ms_last": step_ms,
                "status": solver.last_stats.status.name,
                "note": "same captured candidate positions; trajectories are not required to match",
            }
        )
        del model, solver, state, control
        gc.collect()
    output.mkdir(parents=True, exist_ok=True)
    (output / "material_replay.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def warmup(device, refinement):
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    for _ in range(WARMUP):
        case.step()
    del case
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--refinements", nargs="+", type=int, default=list(REFINEMENTS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "task": "P2-0B-T",
                "primary_refinement": PRIMARY,
                "refinements": args.refinements,
                "devices": args.devices,
                "warmup_steps": WARMUP,
                "repeats": args.repeats,
                "scope": "independent gravity cantilever; Sharpa conclusions not reused",
            },
            indent=2,
        )
        + "\n"
    )
    for device in args.devices:
        for refinement in args.refinements:
            tag = f"{device.replace(':', '')}-r{refinement}"
            print("WARMUP", tag, flush=True)
            warmup(device, refinement)
            diag_dir = args.output / tag / "diagnostic"
            if not (diag_dir / "compact.json").exists():
                print("DIAGNOSTIC", tag, flush=True)
                compact, candidate_x = diagnostic_run(device, refinement, diag_dir)
            else:
                compact = json.loads((diag_dir / "compact.json").read_text())
                candidate_x = None
            index.append({"name": f"{tag}-diagnostic", "failure": compact.get("failure")})
            if not args.skip_timing:
                for repeat in range(args.repeats):
                    out = args.output / tag / f"timing-{repeat+1:02d}"
                    if (out / "compact.json").exists():
                        print("skip", out.name, flush=True)
                        continue
                    print("TIMING", tag, repeat + 1, flush=True)
                    timing_run(device, refinement, out)
            staged_dir = args.output / tag / "staged"
            if not (staged_dir / "staged.json").exists():
                print("STAGED", tag, flush=True)
                windows, x = staged_window(device, refinement, staged_dir)
                if candidate_x is None:
                    candidate_x = x
            if refinement == PRIMARY and candidate_x is not None and not (args.output / tag / "material_replay.json").exists():
                print("MATERIAL", tag, flush=True)
                material_replay(device, refinement, candidate_x, args.output / tag)
            (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2), flush=True)


if __name__ == "__main__":
    main()
