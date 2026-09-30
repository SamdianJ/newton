# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the ten P4 scene configurations in native precision; default to FP32."""

import argparse
import gzip
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.retained_assets import FROZEN, INPUTS, resolve_input, sha, validate_inputs
from scripts.monolithic_reference.run_p3_baseline import cases as baseline_cases

ROOT = Path(__file__).resolve().parents[2]


def cases():
    """Match the existing ten FP64 baselines; include globalization in every ID."""
    return [
        {
            "scene": c["id"].split("--")[0],
            "law": c["law"],
            "steps": c["steps"],
            "precision": p,
            "id": f"{c['id']}--{p}",
            "reference_case": c["id"],
            "dt": 0.001,
            "globalization": c["globalization"],
            "publish_policy": "current_iterate",
            "pcg_mode": "production",
            "graph": True,
        }
        for c in baseline_cases()
        for p in ("fp64", "fp32")
    ]


def build(case, *, input_manifest=FROZEN):
    """Reuse frozen scene construction and sample the original FP32 controls."""
    import warp as wp  # noqa: PLC0415
    from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf  # noqa: PLC0415

    import newton  # noqa: PLC0415
    from newton.examples.softbody.monolithic_sharpa_recorded import build_recorded, targets  # noqa: PLC0415

    if case["scene"] == "franka-cube":
        from newton.examples.softbody.monolithic_franka import FrankaGrasp  # noqa: PLC0415
        from newton.examples.softbody.monolithic_franka import targets as franka_targets  # noqa: PLC0415

        scene = FrankaGrasp(
            INPUTS / "franka/sdf-cache",
            law=case["law"],
            globalization=case["globalization"],
            precision=case["precision"],
            fp32_absolute_tolerance=case.get("fp32_absolute_tolerance"),
        )
        samples = [franka_targets(scene.high, scene.low, i * case["dt"])[:2] for i in range(case["steps"] + 1)]
        return scene.model, scene.solver, scene.state, np.asarray(samples, dtype=np.float32)
    if case["scene"] != "sharpa-recorded-10mm":
        raise ValueError("T5-C only supports the ten frozen Franka/Sharpa configurations")
    payload = json.loads((INPUTS / "sharpa/recorded-control.json").read_text())
    model, _, mapping, curve, terms, trajectory, _ = build_recorded(payload)
    if (trajectory.steps, trajectory.dt) != (case["steps"], case["dt"]):
        raise ValueError("Frozen recorded schedule changed")
    samples = [targets(trajectory, curve, mapping, i * case["dt"])[:2] for i in range(case["steps"] + 1)]
    samples = np.asarray(samples, dtype=np.float32)
    directory = json.loads(input_manifest.read_text()).get(
        "smooth_sdf_directory", str(INPUTS / "sharpa/geometry-original")
    )
    smooth = build_smooth_sdf(model, qualified_directory=resolve_input(directory))
    run = SolverMonolithic(
        model,
        contact_model=case["law"],
        globalization=case["globalization"],
        friction_velocity_scale=0.01,
        alignment_policy="mask",
        publish_policy=case["publish_policy"],
        allow_nonconservative_energy=True,
        tet_rest_density=wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
        joint_terms=terms,
        pcg_mode="production",
        use_cuda_graph=True,
        smooth_sdf=smooth,
        precision=case["precision"],
        fp32_absolute_tolerance=case.get("fp32_absolute_tolerance"),
    )
    source = model.state()
    q = source.joint_q.numpy()
    for coord, _, slot in mapping.values():
        q[coord] = samples[0, 0, slot]
    source.joint_q.assign(q)
    source.joint_qd.zero_()
    newton.eval_fk(model, source.joint_q, source.joint_qd, source)
    return model, run, run.create_state(dt=case["dt"], source=source), samples


def freeze_controls(output, *, input_manifest=FROZEN):
    """Write scene control streams shared by every law, acceptance mode and precision."""
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"input_manifest_sha256": validate_inputs(input_manifest), "streams": {}, "source_hashes": {}}
    for case in cases():
        name = case["scene"]
        if name in manifest["streams"]:
            continue
        _, _, _, samples = build(case, input_manifest=input_manifest)
        path = output / f"{name}.npy"
        np.save(path, samples, allow_pickle=False)
        manifest["streams"][name] = {
            "file": path.name,
            "sha256": sha(path),
            "shape": list(samples.shape),
            "dtype": str(samples.dtype),
        }
    manifest["source_hashes"] = {
        str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "newton/examples/softbody").glob("monolithic_*.py")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def worker(case, controls, output, max_steps, *, input_manifest=FROZEN):
    """Preserve actual native states and first-failure context, without oracle feedback."""
    import warp as wp  # noqa: PLC0415

    from newton.examples.softbody.monolithic_recording import save, serial, state_arrays  # noqa: PLC0415
    from scripts.monolithic_reference.run_p3_baseline import module_origins  # noqa: PLC0415

    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "case": case,
        "status": "FAIL",
        "qualification": "NOT_AUDITED",
        "source_root": str(ROOT),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for folder in ("newton", "scripts/monolithic_reference")
            for p in (ROOT / folder).rglob("*.py")
        },
    }
    try:
        receipt["input_manifest"] = str(input_manifest.resolve())
        receipt["input_manifest_sha256"] = validate_inputs(input_manifest)
        manifest = json.loads((controls / "manifest.json").read_text())
        if manifest["input_manifest_sha256"] != receipt["input_manifest_sha256"]:
            raise ValueError("Control input identity differs")
        entry = manifest["streams"][case["scene"]]
        path = controls / entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Frozen controls changed")
        receipt["controls_sha256"] = sha(path)
        samples = np.load(path, allow_pickle=False)
        start = time.perf_counter()
        model, run, state, actual = build(case, input_manifest=input_manifest)
        wp.synchronize()
        receipt["build_seconds"] = time.perf_counter() - start
        if samples.dtype != np.float32 or not np.array_equal(actual, samples):
            raise ValueError("Constructed controls differ from frozen controls")
        receipt["modules"] = module_origins(ROOT, receipt["source_hashes"])
        receipt["config_digest"] = run._kinematics.digest.hex()
        receipt["sdf_identity"] = run._contact.smooth_sdf.identity
        control = model.control()
        np.savez_compressed(output / "initial.npz", **state_arrays(state))
        counts = {"CONVERGED": 0, "STOPPED": 0, "HARD_FAILURE": 0}
        chunks, frames = [], []
        limit = min(max_steps, case["steps"]) if max_steps is not None else case["steps"]

        def flush():
            if frames:
                name = f"states-{len(chunks):04d}.npz"
                np.savez_compressed(output / name, **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
                chunks.append({"file": name, "frames": len(frames), "sha256": sha(output / name)})
                frames.clear()

        solve_seconds = 0.0
        with gzip.open(output / "steps.jsonl.gz", "wt") as log:
            for step in range(1, limit + 1):
                control.joint_target_q.assign(samples[step, 0])
                control.joint_target_qd.assign(samples[step, 1])
                start = time.perf_counter()
                run.step(state, state, control, dt=case["dt"])
                wp.synchronize()
                solve_seconds += time.perf_counter() - start
                stats = run.last_stats
                label = "HARD_FAILURE" if stats.rolled_back else "CONVERGED" if stats.converged else "STOPPED"
                counts[label] += 1
                log.write(json.dumps(serial({"step": step, "publication": label, "stats": stats})) + "\n")
                frames.append(state_arrays(state))
                if len(frames) == 100:
                    flush()
                if stats.rolled_back:
                    for name in ("_original", "_accepted", "_trial"):
                        candidate = getattr(run, name)
                        if candidate is not None:
                            np.savez_compressed(output / f"failure{name}.npz", **state_arrays(candidate))
                    for name, value in state_arrays(run._original).items():
                        np.testing.assert_array_equal(state_arrays(state)[name], value, err_msg=name)
                    receipt["rollback_verified"] = True
                    receipt["contact_flags"] = run._contact.flags.numpy().tolist()
                    receipt["first_failure_step"] = step
                    break
        flush()
        run.save_checkpoint(state, output / "final.npz")
        receipt.update(
            counts=counts,
            chunks=chunks,
            solve_including_first_compile_capture_seconds=solve_seconds,
            published_steps=counts["CONVERGED"] + counts["STOPPED"],
            status="FAIL" if counts["HARD_FAILURE"] else "PARTIAL" if max_steps is not None else "COMPLETE_NOT_AUDITED",
        )
    except Exception:
        receipt["exception"] = traceback.format_exc()
    save(output / "receipt.json", receipt)
    return int(receipt["status"] == "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp64", "both"),
        default="fp32",
        help="Default FP32; existing FP64 recordings are reused by T5-C",
    )
    parser.add_argument(
        "--fp32-absolute-tolerance", type=float, help="Experimental global/q/x absolute convergence goal"
    )
    parser.add_argument("--freeze-controls", action="store_true")
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--input-manifest", type=Path, default=FROZEN, help="Explicit frozen asset generation")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, help="Smoke only; always reports PARTIAL")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = [c for c in cases() if args.precision == "both" or c["precision"] == args.precision]
    if args.fp32_absolute_tolerance is not None:
        with np.errstate(over="ignore", under="ignore"):
            tolerance = np.float32(args.fp32_absolute_tolerance)
        if args.precision == "fp64" or not np.isfinite(tolerance) or tolerance <= 0:
            parser.error("--fp32-absolute-tolerance requires selected FP32 and a positive representable value")
        for case in selected:
            if case["precision"] == "fp32":
                case["fp32_absolute_tolerance"] = args.fp32_absolute_tolerance
    if set(args.case) - {c["id"] for c in selected}:
        parser.error("Unknown case ID")
    if args.case:
        selected = [c for c in selected if c["id"] in args.case]
    if args.list:
        print(json.dumps(selected, indent=2))
        return 0
    if args.output is None or (not args.freeze_controls and args.controls is None):
        parser.error("--output and --controls required (freeze needs only --output)")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.freeze_controls:
        freeze_controls(args.output.resolve(), input_manifest=args.input_manifest.resolve())
        return 0
    if args.worker:
        if len(selected) != 1:
            parser.error("Worker requires one case")
        return worker(
            selected[0],
            args.controls.resolve(),
            args.output.resolve(),
            args.max_steps,
            input_manifest=args.input_manifest.resolve(),
        )
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in selected:
        destination = args.output.resolve() / case["id"]
        command = [
            sys.executable,
            "-B",
            "-m",
            "scripts.monolithic_reference.run_p5_precision",
            "--worker",
            "--precision",
            case["precision"],
            "--case",
            case["id"],
            "--controls",
            str(args.controls.resolve()),
            "--input-manifest",
            str(args.input_manifest.resolve()),
            "--output",
            str(destination),
        ]
        if case.get("fp32_absolute_tolerance") is not None:
            command += ["--fp32-absolute-tolerance", str(case["fp32_absolute_tolerance"])]
        if args.max_steps is not None:
            command += ["--max-steps", str(args.max_steps)]
        with (args.output / f"{case['id']}.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt = destination / "receipt.json"
        result = json.loads(receipt.read_text()) if receipt.exists() else {"status": "FAIL"}
        rows.append(
            {"case": case["id"], "exit_code": process.returncode, "status": result["status"], "receipt": str(receipt)}
        )
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "qualification": "NOT_AUDITED",
                    "filtered": bool(args.case),
                    "complete": len(rows) == len(selected),
                    "cases": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)
    return int(any(r["exit_code"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
