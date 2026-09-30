# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure one frozen Franka OR/C-IE trajectory in an isolated process.

Use a fresh process per repetition. Compilation/cache-load and graph setup are
warmed before restoring the initial checkpoint. No trajectory or oracle runs in
the timed loop; native validation, publication and postchecks remain enabled.
"""

import argparse
import json
import platform
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.run_precision import ROOT, build, cases, sha, validate_inputs

COUNTERS = (
    "nonlinear_iterations",
    "line_search_iterations",
    "linear_iterations",
    "linear_scheduled_slots",
    "regularization_retries",
)


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
    from scripts.monolithic_reference.run_baseline import module_origins  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=False)
    if args.kernel_cache:
        wp.config.kernel_cache_dir = str(args.kernel_cache.resolve())
    wall_start = time.perf_counter()
    case = next(
        c
        for c in cases()
        if c["scene"] == "franka-cube"
        and c["globalization"] == "ArmijoOrResNorm"
        and c["law"] == "C-IE"
        and c["precision"] == args.precision
    )
    if args.precision == "fp32":
        case["fp32_absolute_tolerance"] = 1e-6
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
    }
    seconds, captured, rows = [], [], []
    counts, totals, statuses = Counter(), Counter(), Counter()
    try:
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
            model, run, state, actual = build(case, input_manifest=args.input_manifest)
        wp.synchronize()
        receipt["scene_build_seconds"] = time.perf_counter() - start
        receipt["build_subtimings"] = dict(parts)
        if samples.dtype != np.float32 or not np.array_equal(actual, samples):
            raise ValueError("Constructed controls differ")
        receipt["modules"] = module_origins(ROOT, receipt["source_hashes"])
        receipt["config_digest"] = run._kinematics.digest.hex()
        receipt["sdf_identity"] = run._contact.smooth_sdf.identity
        initial = state_arrays(state)
        run.save_checkpoint(state, args.output / "initial.npz")
        control = model.control()
        start = time.perf_counter()
        for step in range(1, 21):
            control.joint_target_q.assign(samples[step, 0])
            control.joint_target_qd.assign(samples[step, 1])
            run.step(state, state, control, dt=case["dt"])
            if run.last_stats.rolled_back:
                raise ValueError("Warmup hard failure")
        wp.synchronize()
        receipt["warmup_20_steps_including_lazy_compile_capture_seconds"] = time.perf_counter() - start
        receipt["graphs_after_warmup"] = run._graphs.summary()
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
        for step in range(1, limit + 1):
            control.joint_target_q.assign(samples[step, 0])
            control.joint_target_qd.assign(samples[step, 1])
            wp.synchronize_device(model.device)
            before = sum(run._graphs.captures.values())
            start = time.perf_counter()
            run.step(state, state, control, dt=case["dt"])
            wp.synchronize_device(model.device)
            elapsed = time.perf_counter() - start
            seconds.append(elapsed)
            plan = getattr(run._linear, "_masked_pcg_plan", None)
            captured.append(sum(run._graphs.captures.values()) != before or plan is not previous_plan)
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
            if stats.rolled_back:
                receipt["first_failure"] = {"step": step, "reason": stats.failure_reason}
                for name, value in state_arrays(state).items():
                    np.testing.assert_array_equal(value, state_arrays(run._original)[name], err_msg=name)
                receipt["rollback_verified"] = True
                break
            if step % 250 == 0:
                print(
                    json.dumps({"step": step, "precision": args.precision, "solve_seconds": sum(seconds)}), flush=True
                )
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
        smooth = run._contact.smooth_sdf
        receipt["memory_bytes"] = {
            "cuda_mempool_used_high": wp.get_mempool_used_mem_high(model.device),
            "cuda_mempool_used_current": wp.get_mempool_used_mem_current(model.device),
            "state_logical": sum(v.nbytes for v in final.values()),
            "sdf_active_coefficients": sum(a.capacity for a in smooth._arrays),
            "sdf_active_frozen_copy": sum(a.capacity for a in smooth._frozen),
            "sdf_retained_fp64_coefficients_and_copy": 0
            if smooth._source is None
            else sum(a.capacity for a in (*smooth._source._arrays, *smooth._source._frozen)),
        }
        start = time.perf_counter()
        np.savez_compressed(args.output / "final-state.npz", **final)
        run.save_checkpoint(state, args.output / "final.npz")
        with (args.output / "steps.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        receipt["final_recording_seconds"] = time.perf_counter() - start
        receipt.update(summarize(seconds, captured, counts, case["steps"]))
        receipt.update(
            counts=dict(counts), totals=dict(totals), status_histogram=dict(statuses), published_steps=published
        )
        receipt["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi", "-q"], text=True)
    except Exception:
        receipt["exception"] = traceback.format_exc()
    receipt["end_to_end_seconds"] = time.perf_counter() - wall_start
    save(args.output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output / "receipt.json")}), flush=True)
    return int(receipt["status"] == "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp64", "fp32"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-cache", type=Path)
    parser.add_argument("--steps", type=int, default=5250)
    args = parser.parse_args()
    if not 1 <= args.steps <= 5250:
        parser.error("--steps must be in [1, 5250]")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
