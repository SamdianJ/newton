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

from newton._src.solvers.monolithic.tet import TetScatterBuffers, assemble_tet_residual_tangent
from newton.examples.softbody.monolithic_tet_compare import build_case
from newton.examples.softbody.monolithic_tet_response import ResponseCase, gravity_acceleration
from scripts.monolithic_reference.p2_measurement import (
    SCHEMA,
    array_fingerprint,
    evidence_index,
    export_candidate,
    frame_samples,
    snapshot_sources,
    throughput,
)
from scripts.monolithic_reference.p2_measurement import P2StageProfile as _StageProfile
from scripts.monolithic_reference.profile_release import _finite_json, _percentiles

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/lightwheel/Desktop/newton/SamiulJ")
DEFAULT_OUTPUT = WORKSPACE / "agents/integration/artifacts/p2/pr8a-measurement/tet"
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
        "solver_configuration": {
            "newton_max_iterations": case.solver.newton_max_iterations,
            "linear_max_iterations": case.solver.linear_max_iterations,
            "linear_tolerance": case.solver.linear_tolerance,
            "pcg_execution": "existing_diagnostic",
            "cuda_graph": False,
        },
        "wall_seconds": wall,
        "wall_seconds_per_simulated_second": wall / max(case.steps * case.dt, case.dt),
        "by_phase": {name: summarize_rows([row for row in records if row["phase"] == name]) for _, _, name in PHASES},
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


def timing_step(case):
    """Advance the gravity protocol with control preparation outside solver timing."""
    time_s = (case.steps + 1) * case.dt
    case.state.particle_f.zero_()
    case.model.set_gravity((0, 0, -gravity_acceleration(time_s)))
    wp.synchronize_device(case.model.device)
    start = time.perf_counter()
    case.solver.step(case.state, case.state, case.control, None, case.dt)
    wp.synchronize_device(case.model.device)
    solver_ms = 1000 * (time.perf_counter() - start)
    stats = case.solver.last_stats
    if not stats.rolled_back:
        case.steps += 1
    return {
        "time": time_s,
        "phase": phase_of(time_s),
        "solver_ms": solver_ms,
        "solver_seconds": solver_ms / 1000,
        "measurement_ms": 0.0,
        "step_ms": solver_ms,
        "gravity_m_s2": gravity_acceleration(time_s),
        "nonlinear_iterations": stats.nonlinear_iterations,
        "linear_iterations": stats.linear_iterations,
        "matrix_assembly_count": stats.matrix_assembly_count,
        "converged": stats.converged,
        "status": stats.status.name,
        "rollback": stats.rolled_back,
        "min_det_f": stats.min_det_f,
        "residual_ratio": stats.convergence_ratio,
        "q_residual_ratio": stats.convergence_ratio_q,
        "x_residual_ratio": stats.convergence_ratio_x,
    }


def timing_run(device, refinement, output, *, steps=None):
    setup_start = time.perf_counter()
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    wp.synchronize_device(case.model.device)
    setup_seconds = time.perf_counter() - setup_start
    samples = []
    pcg_calls = []
    original_solve = case.solver._linear.solve_pcg

    def measured_solve(*args, **kwargs):
        wp.synchronize_device(case.model.device)
        start = time.perf_counter()
        result = original_solve(*args, **kwargs)
        wp.synchronize_device(case.model.device)
        pcg_calls.append(
            {
                "iterations": result.iterations,
                "status": result.status.name,
                "seconds": time.perf_counter() - start,
                "rho": result.rho,
                "rho_q": result.rho_q,
                "rho_x": result.rho_x,
            }
        )
        return result

    failure = None
    t0 = time.perf_counter()
    try:
        with patch.object(case.solver._linear, "solve_pcg", new=measured_solve):
            for _ in range(steps if steps is not None else round(case.duration / case.dt)):
                pcg_calls.clear()
                row = timing_step(case)
                row["pcg_calls"] = list(pcg_calls)
                samples.append(row)
                if row["rollback"]:
                    failure = f"{row['status']}: {case.solver.last_stats.failure_reason}"
                    break
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    wall = time.perf_counter() - t0
    frames = frame_samples(samples, 6, "solver_seconds")
    compact = {
        "schema": SCHEMA,
        "kind": "solver_only_timing",
        "requested_device": device,
        "actual_device": str(case.model.device),
        "refinement": refinement,
        "particles": case.model.particle_count,
        "tets": case.model.tet_count,
        "failure": failure,
        "accepted_steps": case.steps,
        "complete": case.steps == round(case.duration / case.dt) and failure is None,
        "dt": case.dt,
        "substeps": 6,
        "frame_dt_s": 6 * case.dt,
        "setup_seconds": setup_seconds,
        "manifest": case.manifest,
        "solver_configuration": {
            "newton_max_iterations": case.solver.newton_max_iterations,
            "linear_max_iterations": case.solver.linear_max_iterations,
            "linear_tolerance": case.solver.linear_tolerance,
            "pcg_execution": "existing_diagnostic",
            "cuda_graph": False,
        },
        "work": throughput(sum(r["solver_seconds"] for r in samples), case.steps, case.dt, 6),
        "wall_seconds": wall,
        "wall_seconds_per_simulated_second": wall / (case.steps * case.dt) if case.steps else None,
        "frame_solver_ms_6substeps": {
            name: None
            if not rows
            else {
                "p50": 1000 * float(np.percentile([r["solver_seconds"] for r in rows], 50)),
                "p95": 1000 * float(np.percentile([r["solver_seconds"] for r in rows], 95)),
            }
            for name, rows in ((label, [r for r in frames if r["phase"] == label]) for _, _, label in PHASES)
        },
        "pcg_by_phase": {
            name: {
                "iterations": _percentiles(
                    [c["iterations"] for r in samples if r["phase"] == name for c in r["pcg_calls"]]
                ),
                "solve_ms": _percentiles(
                    [1000 * c["seconds"] for r in samples if r["phase"] == name for c in r["pcg_calls"]]
                ),
            }
            for _, _, name in PHASES
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
    (output / "compact.json").write_text(json.dumps(_finite_json(compact), indent=2, allow_nan=False) + "\n")
    (output / "trace.jsonl").write_text(
        "".join(json.dumps(_finite_json(row), sort_keys=True, allow_nan=False) + "\n" for row in samples)
    )
    (output / "every_frame.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in frames))
    np.savez_compressed(
        output / "final_state.npz",
        x=case.state.particle_q.numpy(),
        v=case.state.particle_qd.numpy(),
        q=case.state.joint_q.numpy(),
        qd=case.state.joint_qd.numpy(),
        gravity=case.model.gravity.numpy(),
    )
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
            for _ in range(count):
                case.step()
        tet_ms = timer.samples["tet_assembly"]
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


def material_replay(device, refinement, candidate, output):
    """Repeat one exported loaded input in preallocated tet scratch, without solver.step."""
    required = ("candidate_x", "particle_q_n", "particle_qd_n", "frozen_particle_f", "gravity")
    if not isinstance(candidate, dict) or any(name not in candidate for name in required):
        raise ValueError("Material replay requires exported loaded inputs, not positions alone")
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
        model.gravity.assign(candidate["gravity"])
        inputs = {name: wp.array(candidate[name], dtype=wp.vec3, device=device) for name in required[:-1]}
        inputs["candidate_particle_q"] = inputs.pop("candidate_x")
        inputs.update(
            dynamic_particle_ids=solver._layout.dynamic_particle_ids,
            particle_to_dynamic=solver._layout.particle_to_dynamic,
        )
        scatter = TetScatterBuffers(
            wp.zeros_like(solver._linear._internal.values),
            wp.zeros(9 * solver._linear._internal.values.size, device=device),
        )
        residual = wp.zeros_like(solver._particle_residual)
        fingerprint_inputs = {**inputs, "gravity": model.gravity}
        before = array_fingerprint(fingerprint_inputs)
        samples = []
        reference = None
        max_difference = 0.0
        for repeat in range(22):
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            assemble_tet_residual_tangent(
                model,
                **inputs,
                residual_x=residual,
                dt=0.02,
                min_det_f_guard=solver._config.det_f_guard,
                workspace=solver._tet_workspace,
                scatter=scatter,
            )
            wp.synchronize_device(model.device)
            elapsed = 1000 * (time.perf_counter() - start)
            if int(solver._tet_workspace.failure_flags.numpy()[0]):
                raise RuntimeError("Fixed candidate tet evaluation failed")
            current = (residual.numpy(), scatter.global_values.numpy())
            if reference is None:
                reference = current
            for observed, expected in zip(current, reference, strict=True):
                max_difference = max(max_difference, float(np.max(np.abs(observed - expected))))
                np.testing.assert_allclose(observed, expected, rtol=5e-5 if model.device.is_cuda else 1e-5, atol=1e-5)
            if repeat >= 2:
                samples.append(elapsed)
        after = array_fingerprint(fingerprint_inputs)
        if before != after:
            raise RuntimeError("Material replay mutated a frozen input")
        rows.append(
            {
                "material": material,
                "mass": mass,
                "tets": model.tet_count,
                "tet_assemble_ms": _percentiles(samples),
                "samples_ms": samples,
                "tet_per_tet_us_p50": 1000 * float(np.percentile(samples, 50)) / model.tet_count,
                "input_before_sha256": before,
                "input_after_sha256": after,
                "max_repeat_output_difference": max_difference,
                "status": "PASS",
                "note": "Fixed loaded candidate, gravity, previous x/v and force; assembly only. "
                "Kim/Smith compute comparison does not assert equivalent physical response.",
            }
        )
        del model, solver, state, control
        gc.collect()
    output.mkdir(parents=True, exist_ok=True)
    (output / "material_replay.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def matrix_run(device, refinement, output, *, steps=1750):
    """Collect first actual PCG calls at loading/hold/tail times on an independent trajectory."""
    case = ResponseCase(device, experiment="gravity", variant=0, refinement=refinement)
    marks = {50: "loading", 150: "hold", 1750: "tail"}
    saved = {}
    original = case.solver._linear.solve_pcg

    def capture(*args, **kwargs):
        step = case.steps + 1
        if step in marks and marks[step] not in saved:
            label = marks[step]
            saved[label] = export_candidate(
                case.solver, output / f"{label}.npz", time_s=step * case.dt, fixture=case.manifest
            )
        return original(*args, **kwargs)

    with patch.object(case.solver._linear, "solve_pcg", new=capture):
        for _ in range(steps):
            row = timing_step(case)
            if row["rollback"]:
                raise RuntimeError(f"Matrix trajectory rollback at {row['time']}")
    return saved


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
    snapshot_sources(args.output / "provenance")
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
                compact, _x = diagnostic_run(device, refinement, diag_dir)
            else:
                compact = json.loads((diag_dir / "compact.json").read_text())
            index.append({"name": f"{tag}-diagnostic", "failure": compact.get("failure")})
            if not args.skip_timing:
                for repeat in range(args.repeats):
                    out = args.output / tag / f"timing-{repeat + 1:02d}"
                    if (out / "compact.json").exists():
                        print("skip", out.name, flush=True)
                        continue
                    print("TIMING", tag, repeat + 1, flush=True)
                    timing_run(device, refinement, out)
            staged_dir = args.output / tag / "staged"
            if not (staged_dir / "staged.json").exists():
                print("STAGED", tag, flush=True)
                staged_window(device, refinement, staged_dir)
            captured = matrix_run(device, refinement, args.output / tag / "matrices")
            if (
                refinement == PRIMARY
                and "loading" in captured
                and not (args.output / tag / "material_replay.json").exists()
            ):
                print("MATERIAL", tag, flush=True)
                material_replay(device, refinement, captured["loading"], args.output / tag)
            (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2), flush=True)
    evidence_index(args.output)


if __name__ == "__main__":
    main()
