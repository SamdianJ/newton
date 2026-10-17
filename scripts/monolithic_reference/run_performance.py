# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure one frozen Franka OR/C-IE or actor trajectory in an isolated process.

Use a fresh process per repetition. Compilation/cache-load and graph setup are
warmed before restoring the initial checkpoint. No trajectory or oracle runs in
the timed loop; native validation, publication and postchecks remain enabled.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.retained_assets import FROZEN, INPUTS
from scripts.monolithic_reference.run_precision import ROOT, build, cases, sha, validate_inputs

COUNTERS = (
    "nonlinear_iterations",
    "line_search_iterations",
    "linear_iterations",
    "linear_scheduled_slots",
    "regularization_retries",
)


def performance_case(scene, precision):
    """Select an existing frozen fixture without changing its physical policy."""
    if scene == "actor-coupling":
        from scripts.monolithic_reference.run_actor_coupling import load_case  # noqa: PLC0415

        if precision != "fp64":
            raise ValueError("The frozen actor recipe requires FP64")
        case = load_case()
        return {**case, "scene": scene, "precision": precision}
    if scene != "franka-cube":
        raise ValueError("Unsupported performance scene")
    case = next(
        c
        for c in cases()
        if c["scene"] == scene
        and c["globalization"] == "ArmijoOrResNorm"
        and c["law"] == "C-IE"
        and c["precision"] == precision
    )
    return case


def build_performance_scene(case, *, contact_dataflow, input_manifest=FROZEN, device="cuda:0", graph=True):
    """Reuse frozen constructors and controls with one explicit contact route."""
    if contact_dataflow not in ("legacy", "spatial"):
        raise ValueError("Unsupported performance contact route")
    if case["scene"] == "actor-coupling":
        from scripts.monolithic_reference.run_actor_coupling import build_model, make_solver, targets  # noqa: PLC0415

        model = build_model(device=device)
        run = make_solver(
            model,
            graph=graph,
            contact_dataflow=contact_dataflow,
            spatial={"candidate_capacity": 720, "law_margin": 0.0} if contact_dataflow == "spatial" else None,
            contact_factor_capacity=2160 if contact_dataflow == "spatial" else None,
        )
        control = model.control()
        position = np.tile(control.joint_target_q.numpy(), (case["steps"] + 1, 1))
        velocity = np.tile(control.joint_target_qd.numpy(), (case["steps"] + 1, 1))
        # Only the prismatic target is driven, at the original absolute endpoint clock.
        arm = run._articulation_actor
        for step in range(case["steps"] + 1):
            q, qd, _ = targets(case, step)
            position[step, arm.target_indices[0]] = q
            velocity[step, arm.dof_indices[0]] = qd
        return model, run, run.create_state(dt=case["dt"]), (position, velocity)
    if device != "cuda:0" or not graph:
        raise ValueError("The frozen Franka recipe requires CUDA0 graph")
    factory = None
    if contact_dataflow == "spatial":
        from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory  # noqa: PLC0415

        if case["precision"] != "fp64":
            raise ValueError("The spatial Franka experiment requires FP64")
        factory = FrozenRobotFactory("franka-cube", dataflow="spatial")
    return build(case, input_manifest=input_manifest, solver_factory=factory)


def control_at(samples, step):
    """Read equal-width robot or unequal-width actor target streams."""
    return (samples[0][step], samples[1][step]) if isinstance(samples, tuple) else samples[step]


def summarize(seconds, captured, counts, expected):
    """Report complete costs without equating advancement with qualification."""
    values = np.asarray(seconds)
    clean = values[~np.asarray(captured, dtype=bool)]
    status = (
        "FAIL" if counts.get("HARD_FAILURE", 0) else "PARTIAL" if len(values) != expected else "COMPLETE_NOT_AUDITED"
    )
    return {
        "status": status,
        "solve_seconds": float(values.sum()),
        "steps_per_second": len(values) / float(values.sum()) if values.sum() else None,
        "step_p50_seconds": float(np.median(values)) if len(values) else None,
        "step_p95_seconds": float(np.percentile(values, 95)) if len(values) else None,
        "step_max_seconds": float(values.max()) if len(values) else None,
        "capture_free_seconds": float(clean.sum()),
        "capture_free_steps": len(clean),
        "capture_affected_steps": int(np.count_nonzero(captured)),
        "qualified_speedup": None,
    }


def worker(args):
    """Build, warm, restore and measure with synchronized step boundaries."""
    import warp as wp  # noqa: PLC0415

    from newton._src.solvers.monolithic.smooth_sdf import SmoothSDFSet  # noqa: PLC0415
    from newton.examples.softbody import monolithic_franka as franka  # noqa: PLC0415
    from newton.examples.softbody.monolithic_recording import save, state_arrays  # noqa: PLC0415
    from scripts.monolithic_reference.benchmark_workload import check_gpu_idle  # noqa: PLC0415
    from scripts.monolithic_reference.profiling import capture_count, contact_pair_diagnostics  # noqa: PLC0415
    from scripts.monolithic_reference.run_baseline import module_origins  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=False)
    if args.kernel_cache:
        wp.config.kernel_cache_dir = str(args.kernel_cache.resolve())
    wall_start = time.perf_counter()
    case = performance_case(args.scene, args.precision)
    receipt = {
        "status": "FAIL",
        "case": case,
        "qualification": "NOT_AUDITED",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for folder in ("newton", "scripts/monolithic_reference")
            for p in (ROOT / folder).rglob("*.py")
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "kernel_cache": str(args.kernel_cache),
        "contact_dataflow": args.contact_dataflow,
        "contact_pair_diagnostics": not args.no_contact_diagnostics,
        "gc_enabled": gc.isenabled(),
        "device": args.device,
        "graph": not args.no_graph,
        "command": sys.argv,
    }
    seconds, captured, rows = [], [], []
    counts, totals, statuses = Counter(), Counter(), Counter()
    try:
        check_gpu_idle(wp.get_device(args.device))
        samples = None
        if args.scene == "franka-cube":
            receipt["input_manifest"] = str(args.input_manifest.resolve())
            receipt["input_manifest_sha256"] = validate_inputs(args.input_manifest)
            manifest = json.loads((args.controls / "manifest.json").read_text())
            if manifest["input_manifest_sha256"] != receipt["input_manifest_sha256"]:
                raise ValueError("Control input identity differs")
            entry = manifest["streams"]["franka-cube"]
            path = args.controls / entry["file"]
            if sha(path) != entry["sha256"]:
                raise ValueError("Frozen controls changed")
            receipt["controls_sha256"] = sha(path)
            samples = np.load(path, allow_pickle=False)
        else:
            from scripts.monolithic_reference.run_actor_coupling import CASE  # noqa: PLC0415

            receipt["fixture_sha256"] = sha(CASE)
        receipt["nvidia_smi_before"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
        parts = Counter()

        def timed(name, fn):
            def wrapped(*a, **kw):
                start = time.perf_counter()
                result = fn(*a, **kw)
                wp.synchronize()
                parts[name] += time.perf_counter() - start
                return result

            return wrapped

        start = time.perf_counter()
        with (
            patch.object(franka, "smooth_fields", timed("fp64_sdf_build_seconds", franka.smooth_fields)),
            patch.object(
                SmoothSDFSet, "for_precision", timed("sdf_precision_binding_seconds", SmoothSDFSet.for_precision)
            ),
        ):
            model, run, state, actual = build_performance_scene(
                case,
                input_manifest=args.input_manifest,
                contact_dataflow=args.contact_dataflow,
                device=args.device,
                graph=not args.no_graph,
            )
        wp.synchronize()
        receipt["scene_build_seconds"] = time.perf_counter() - start
        receipt["build_subtimings"] = dict(parts)
        if samples is not None and (samples.dtype != np.float32 or not np.array_equal(actual, samples)):
            raise ValueError("Constructed controls differ")
        if samples is None:
            samples = actual
            receipt["controls_sha256"] = hashlib.sha256(b"".join(a.tobytes() for a in samples)).hexdigest()
            receipt["control_shapes"] = [list(a.shape) for a in samples]
        receipt["modules"] = module_origins(ROOT, receipt["source_hashes"])
        receipt["config_digest"] = run._kinematics.digest.hex()
        receipt["sdf_identity"] = run._contact.smooth_sdf.identity if run._contact.smooth_sdf is not None else None
        receipt["numerical_policy"] = dataclasses.asdict(run._config)
        receipt["resources"] = run._contact.resource_identity.tolist() if run._contact_pool else None
        receipt["law_margin"] = run._contact.law_margin if run._contact_spatial else None
        initial = state_arrays(state)
        run.save_checkpoint(state, args.output / "initial.npz")
        control = model.control()
        start = time.perf_counter()
        diagnostics = contact_pair_diagnostics(run, enabled=not args.no_contact_diagnostics)
        with diagnostics:
            for step in range(1, 21):
                q, qd = control_at(samples, step)
                control.joint_target_q.assign(q)
                control.joint_target_qd.assign(qd)
                run.step(state, state, control, dt=case["dt"])
                if run.last_stats.rolled_back:
                    raise ValueError("Warmup hard failure")
        wp.synchronize()
        receipt["warmup_20_steps_including_lazy_compile_capture_seconds"] = time.perf_counter() - start
        receipt["graphs_after_warmup"] = run._graphs.summary()
        receipt["all_captures_after_warmup"] = capture_count(run)
        start = time.perf_counter()
        run.load_checkpoint(state, args.output / "initial.npz")
        wp.synchronize()
        receipt["warm_restore_seconds"] = time.perf_counter() - start
        for name, value in state_arrays(state).items():
            np.testing.assert_array_equal(value, initial[name], err_msg=name)
        receipt["initial_restore"] = "EXACT"
        receipt["measurement_generation_start"] = run._step_generation
        limit = min(args.steps, case["steps"])
        loop_start = time.perf_counter()
        previous_plan = getattr(run._linear, "_masked_pcg_plan", None)
        with contact_pair_diagnostics(run, enabled=not args.no_contact_diagnostics):
            for step in range(1, limit + 1):
                if step == 1 or step % 250 == 0:
                    check_gpu_idle(model.device)
                q, qd = control_at(samples, step)
                control.joint_target_q.assign(q)
                control.joint_target_qd.assign(qd)
                wp.synchronize_device(model.device)
                before = capture_count(run)
                start = time.perf_counter()
                run.step(state, state, control, dt=case["dt"])
                wp.synchronize_device(model.device)
                elapsed = time.perf_counter() - start
                seconds.append(elapsed)
                plan = getattr(run._linear, "_masked_pcg_plan", None)
                captured.append(capture_count(run) != before or plan is not previous_plan)
                previous_plan = plan
                stats = run.last_stats
                label = "HARD_FAILURE" if stats.rolled_back else "CONVERGED" if stats.converged else "STOPPED"
                counts[label] += 1
                statuses[str(stats.status)] += 1
                row = {
                    "step": step,
                    "seconds": elapsed,
                    "publication": label,
                    "status": str(stats.status),
                    "capture_affected": captured[-1],
                    **{k: getattr(stats, k) for k in COUNTERS},
                }
                totals.update({k: row[k] for k in COUNTERS})
                rows.append(row)
                for name in ("contact_pool", "contact_spatial"):
                    report = stats.variational.get(name, {})
                    if report.get("tainted") or report.get("dropped", 0) or report.get("capacity_dropped", 0):
                        raise ValueError(f"Capacity-tainted measurement at step {step}: {name}")
                if stats.rolled_back:
                    receipt["first_failure"] = {"step": step, "reason": stats.failure_reason}
                    for name, value in state_arrays(state).items():
                        np.testing.assert_array_equal(value, state_arrays(run._original)[name], err_msg=name)
                    receipt["rollback_verified"] = True
                    break
                if step % 250 == 0:
                    print(
                        json.dumps({"step": step, "precision": args.precision, "solve_seconds": sum(seconds)}),
                        flush=True,
                    )
        check_gpu_idle(model.device)
        receipt["loop_wall_seconds"] = time.perf_counter() - loop_start
        final = state_arrays(state)
        if not all(np.isfinite(value).all() for value in final.values()):
            raise ValueError("Nonfinite final state")
        published = counts["CONVERGED"] + counts["STOPPED"]
        if (
            int(final["history.force_generation"][0]) != receipt["measurement_generation_start"] + published
            or not final["history.force_valid"][0]
        ):
            raise ValueError("Invalid final force generation")
        for history, field in (
            ("bound_q", "joint_q"),
            ("bound_qd", "joint_qd"),
            ("bound_x", "particle_q"),
            ("bound_v", "particle_qd"),
        ):
            np.testing.assert_array_equal(final["history." + history], final[field])
        receipt["final_state_checks"] = "FINITE_HISTORY_FORCE_GENERATION_PASS"
        receipt["graphs_after_measurement"] = run._graphs.summary()
        receipt["all_captures_after_measurement"] = capture_count(run)
        smooth = run._contact.smooth_sdf
        receipt["memory_bytes"] = {
            "cuda_mempool_used_high": wp.get_mempool_used_mem_high(model.device) if model.device.is_cuda else None,
            "cuda_mempool_used_current": wp.get_mempool_used_mem_current(model.device)
            if model.device.is_cuda
            else None,
            "state_logical": sum(v.nbytes for v in final.values()),
            "sdf_active_coefficients": sum(a.capacity for a in smooth._arrays) if smooth is not None else 0,
            "sdf_active_frozen_copy": sum(a.capacity for a in smooth._frozen) if smooth is not None else 0,
            "sdf_retained_fp64_coefficients_and_copy": 0
            if smooth is None or smooth._source is None
            else sum(a.capacity for a in (*smooth._source._arrays, *smooth._source._frozen)),
        }
        start = time.perf_counter()
        np.savez_compressed(args.output / "final-state.npz", **final)
        run.save_checkpoint(state, args.output / "final.npz")
        receipt["final_recording_seconds"] = time.perf_counter() - start
        receipt.update(summarize(seconds, captured, counts, case["steps"]))
        receipt.update(
            counts=dict(counts), totals=dict(totals), status_histogram=dict(statuses), published_steps=published
        )
        receipt["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    except Exception:
        receipt["status"] = "FAIL"
        receipt["exception"] = traceback.format_exc()
    try:
        changed = any(sha(ROOT / path) != digest for path, digest in receipt["source_hashes"].items())
    except OSError:
        changed = True
    if changed:
        receipt["status"] = "FAIL"
        receipt["source_changed_during_measurement"] = True
    # Preserve completed timing rows even when a later capacity/environment guard fails.
    with (args.output / "steps.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    receipt["measured_steps"] = len(rows)
    receipt["end_to_end_seconds"] = time.perf_counter() - wall_start
    save(args.output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output / "receipt.json")}), flush=True)
    return int(receipt["status"] == "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp64", "fp32"), required=True)
    parser.add_argument("--scene", choices=("franka-cube", "actor-coupling"), default="franka-cube")
    parser.add_argument("--contact-dataflow", choices=("legacy", "spatial"), default="legacy")
    parser.add_argument("--no-contact-diagnostics", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--no-graph", action="store_true", help="CPU actor smoke checks only; not formal GPU timing")
    parser.add_argument("--input-manifest", type=Path, default=FROZEN)
    parser.add_argument("--controls", type=Path, default=INPUTS / "controls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-cache", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    try:
        case = performance_case(args.scene, args.precision)
    except ValueError as error:
        parser.error(str(error))
    if args.steps is None:
        args.steps = case["steps"]
    if args.contact_dataflow == "spatial" and args.precision != "fp64":
        parser.error("Spatial performance recipes require FP64")
    if args.scene == "franka-cube" and (args.device != "cuda:0" or args.no_graph):
        parser.error("The frozen Franka recipe requires CUDA0 graph")
    if not 1 <= args.steps <= case["steps"]:
        parser.error(f"--steps must be in [1, {case['steps']}]")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
