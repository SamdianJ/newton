# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manually run retained FP64 scene regressions, including the ten-case nightly profile."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from scripts.monolithic_reference.retained_assets import (
    ASSET_ROOT,
    FROZEN,
    INPUTS,
    resolve_input,
    retained_hashes,
    verify_retained,
)
from scripts.monolithic_reference.retained_assets import validate_inputs as validate_manifest
from scripts.monolithic_reference.run_p3_baseline import cases, module_origins, sha

ROOT = Path(__file__).resolve().parents[2]


def select_cases(profile, filters=()):
    """Expand the shared T0 case definitions before applying explicit filters."""
    full = cases()
    if profile not in ("dev", "full", "nightly"):
        raise ValueError(f"Unknown scene profile: {profile}")
    selected = full if profile in ("full", "nightly") else [full[0], full[-1]]
    if filters:
        unknown = set(filters) - {c["id"] for c in selected}
        if unknown:
            raise ValueError(f"Cases outside the {profile} profile: {sorted(unknown)}")
        selected = [c for c in selected if c["id"] in filters]
    return selected


def aggregate(results, filtered):
    """Never grant profile qualification to a filtered run."""
    if any(row["status"] != "PASS" for row in results):
        return "FAIL"
    identities = {(row.get("commit"), row.get("source_sha256"), row.get("input_manifest_sha256")) for row in results}
    if len(identities) > 1:
        return "FAIL"
    return "PARTIAL" if filtered else "PASS"


def finalize_case(destination, receipt, exit_code, retention):
    """Retain failure recordings and remove only successful newly generated recordings."""
    status = "PASS" if exit_code == 0 and receipt["status"] == "PASS" else "FAIL"
    recording = destination / "recording"
    receipt["retention"] = retention
    receipt["recording_retained"] = recording.is_dir()
    prune = status == "PASS" and retention == "failures" and recording.is_dir()
    if prune:
        if recording.is_symlink():
            raise ValueError("Refuse to prune a linked recording")
        receipt["recording_retained"] = False
    # Persist the decision and numerical summary before dropping full successful states.
    if destination.is_dir():
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if prune:
        shutil.rmtree(recording)
    return status


def write_report(output, summary):
    """Produce a compact report for review after the manually started batch finishes."""
    lines = [
        "# Monolithic scene regression",
        "",
        f"Profile: {summary['profile']}; status: {summary['status']}; complete: {summary['complete']}.",
        "Component tests: NOT_RUN_BY_SCENE_RUNNER. Stopped is not convergence.",
        "",
        "| Case | Result | Converged | Stopped | Recording retained | Receipt |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["cases"]:
        lines.append(
            f"| {row['case']} | {row['status']} | {row['counts'].get('CONVERGED', '-')} | "
            f"{row['counts'].get('STOPPED_ACCEPTED', '-')} | {row['recording_retained']} | "
            f"[receipt]({row['case']}/receipt.json) |"
        )
    lines += ["", "Receipts include source/input/reference identities, environment, scene metrics and first errors."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def validate_inputs():
    """Bind the current runner to the retained, recertified asset generation."""
    verify_retained(FROZEN, retained_hashes())
    manifest = json.loads(FROZEN.read_text())
    if resolve_input(manifest["smooth_sdf_directory"]) != INPUTS / "sharpa/geometry-rebuilt":
        raise ValueError("Unexpected qualified Sharpa asset generation")
    return validate_manifest(FROZEN)


def reference_binding(case):
    """Verify the selected reference before building a scene or taking any steps."""
    rows = json.loads((ASSET_ROOT / "references.json").read_text())
    expected = {f"{c['id']}--fp64" for c in cases()}
    if len(rows) != len(expected) or {row["case"] for row in rows} != expected:
        raise ValueError("Retained reference index must contain the exact ten FP64 cases")
    entry = next(row for row in rows if row["case"] == f"{case['id']}--fp64")
    directory = ASSET_ROOT / entry["retained_path"]
    if not directory.resolve().is_relative_to((ASSET_ROOT / "data/reference").resolve()):
        raise ValueError("Reference outside retained directory")
    hashes = retained_hashes()
    prefix = entry["retained_path"] + "/"
    for name in hashes:
        if name.startswith(prefix):
            verify_retained(ASSET_ROOT / name, hashes)
    if sha(directory / "receipt.json") != entry["receipt_sha256"]:
        raise ValueError("Reference receipt identity changed")
    scene = case["id"].split("--")[0]
    control_manifest = INPUTS / "controls/manifest.json"
    verify_retained(control_manifest, hashes)
    manifest = json.loads(control_manifest.read_text())
    if manifest["input_manifest_sha256"] != sha(FROZEN):
        raise ValueError("Control input identity differs")
    stream = manifest["streams"][scene]
    controls = INPUTS / "controls" / stream["file"]
    verify_retained(controls, hashes)
    if sha(controls) != stream["sha256"]:
        raise ValueError("Frozen control stream changed")
    if entry["state_format"] == "p3_recording" and scene == "franka-cube":
        reference = directory / "recording"
        digest = json.loads((reference / "configuration.json").read_text())["config_digest"]
        protocol = INPUTS / "comparison-tolerances.json"
        verify_retained(protocol, hashes)
        envelopes = json.loads(protocol.read_text())["cases"][case["id"]]["absolute"]
        native_controls = None
        policy = "FRANKA_FROZEN_T0_ENVELOPE"
        if stream["sha256"] != entry["controls_sha256"]:
            raise ValueError("Franka reference/control identity differs")
    elif entry["state_format"] == "p5_native_state_chunks" and scene == "sharpa-recorded-10mm":
        reference = directory
        receipt = json.loads((directory / "receipt.json").read_text())
        digest = receipt["config_digest"]
        if receipt["case"]["id"] != entry["case"] or receipt["input_manifest_sha256"] != sha(FROZEN):
            raise ValueError("Sharpa reference input generation differs")
        if receipt["controls_sha256"] != stream["sha256"]:
            raise ValueError("Sharpa reference/control identity differs")
        # No repeatability envelope was certified for this rebuilt asset generation.
        envelopes, native_controls, policy = {}, controls, "REBUILT_SHARPA_EXACT"
    else:
        raise ValueError("Unsupported retained scene/reference format")
    if digest != entry["config_digest"]:
        raise ValueError("Reference configuration digest differs from its index")
    return reference, controls, native_controls, envelopes, {**entry, "comparison_policy": policy}


def worker(case, output):
    """Run a complete scene and validate recordings against the frozen envelopes."""
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"case": case, "status": "FAIL", "source_root": str(ROOT)}
    source_hashes = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ("newton", "scripts/monolithic_reference")
        for p in (ROOT / folder).rglob("*.py")
    }
    receipt["source_hashes"] = source_hashes
    receipt["commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    try:
        receipt["input_manifest_sha256"] = validate_inputs()
        receipt["input_manifest"] = str(FROZEN)
        reference, controls, native_controls, envelopes, binding = reference_binding(case)
        receipt["reference"] = binding
        import numpy as np  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        from scripts.monolithic_reference.check_p3_recording import check, compare  # noqa: PLC0415

        if not wp.get_device("cuda:0").is_cuda:
            raise ValueError("P4 profiles require CUDA")
        receipt["environment"] = {
            "python": sys.version,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "device": wp.get_device("cuda:0").name,
        }
        recording = output / "recording"
        if case["id"].startswith("franka"):
            from newton.examples.softbody.example_monolithic_franka_grasp import Example  # noqa: PLC0415
            from newton.examples.softbody.monolithic_franka import targets  # noqa: PLC0415
            from newton.viewer import ViewerNull  # noqa: PLC0415

            args = SimpleNamespace(
                device="cuda:0",
                output=recording,
                asset_cache=INPUTS / "franka/sdf-cache",
                law=case["law"],
                globalization=case["globalization"],
            )
            example = Example(ViewerNull(), args)
            if example.case.solver._kinematics.digest.hex() != binding["config_digest"]:
                raise ValueError("Physical configuration digest changed")
            samples = np.asarray(
                [targets(example.case.high, example.case.low, i * 0.001)[:2] for i in range(case["steps"] + 1)],
                dtype=np.float32,
            )
            if not np.array_equal(samples, np.load(controls, allow_pickle=False)):
                raise ValueError("Constructed controls differ from frozen controls")
            while not example.closed:
                example.step()
            example.test_final()
        else:
            from newton.examples.softbody.monolithic_sharpa_recorded import run  # noqa: PLC0415

            recording.mkdir()
            args = SimpleNamespace(
                output=recording,
                trajectory=INPUTS / "sharpa/recorded-control.json",
                expected_config_digest=binding["config_digest"],
                expected_controls=controls,
            )
            if not run(args):
                raise ValueError("Sharpa did not publish its complete trajectory")
        receipt["recording"] = check(recording, case["steps"])
        receipt["comparison"] = compare(
            reference, recording, envelopes, legacy_fp64=True, native_controls=native_controls
        )
        if receipt["comparison"]["status"] not in ("PASS_EXACT", "PASS_FROZEN_ENVELOPE"):
            raise ValueError("Recording differs outside the frozen envelope")
        actual = json.loads((recording / "configuration.json").read_text())
        if actual["config_digest"] != binding["config_digest"]:
            raise ValueError("Physical configuration digest changed")
        receipt["modules"] = module_origins(ROOT, source_hashes)
        receipt["status"] = "PASS"
    except Exception:
        receipt["exception"] = traceback.format_exc()
    finally:
        scene_summary = output / "recording/summary.json"
        if scene_summary.is_file():
            receipt["scene_summary"] = json.loads(scene_summary.read_text())
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(receipt["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("dev", "full", "nightly"), default="dev")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--retain-recordings",
        choices=("failures", "all"),
        default="failures",
        help="Keep compact success receipts and full failure recordings by default",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        selected = select_cases(args.profile, args.case)
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        print(json.dumps(selected, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.worker:
        if len(selected) != 1:
            parser.error("Worker requires exactly one case")
        return worker(selected[0], args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in selected:
        destination = args.output.resolve() / case["id"]
        command = [
            sys.executable,
            "-B",
            "-m",
            "scripts.monolithic_reference.run_p4",
            "--profile",
            args.profile,
            "--case",
            case["id"],
            "--worker",
            "--output",
            str(destination),
        ]
        with (args.output / f"{case['id']}.log").open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        path = destination / "receipt.json"
        receipt = json.loads(path.read_text()) if path.is_file() else {"status": "FAIL"}
        status = finalize_case(destination, receipt, process.returncode, args.retain_recordings)
        results.append(
            {
                "case": case["id"],
                "status": status,
                "receipt": str(path),
                "exit_code": process.returncode,
                "counts": receipt.get("recording", {}).get("counts", {}),
                "recording_retained": receipt["recording_retained"],
                "commit": receipt.get("commit"),
                "source_sha256": hashlib.sha256(
                    json.dumps(receipt.get("source_hashes", {}), sort_keys=True).encode()
                ).hexdigest(),
                "input_manifest_sha256": receipt.get("input_manifest_sha256"),
            }
        )
        summary = {
            "profile": args.profile,
            "status": aggregate(results, bool(args.case)),
            "complete": len(results) == len(selected),
            "filtered": bool(args.case),
            "cases": results,
        }
        if not summary["complete"] and summary["status"] == "PASS":
            summary["status"] = "RUNNING"
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_report(args.output, summary)
        print(json.dumps(results[-1]), flush=True)
    return int(summary["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
