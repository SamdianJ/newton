# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen P4 dev/full scene matrix serially, in isolated CUDA processes."""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from scripts.monolithic_reference.run_p3_baseline import cases, module_origins, sha

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parents[1] / "agents/integration/artifacts"
T0 = ARTIFACTS / "p4/t0-v1"


def select_cases(profile, filters=()):
    """Expand the shared T0 case definitions before applying explicit filters."""
    full = cases()
    selected = full if profile == "full" else [full[0], full[-1]]
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
    return "PARTIAL" if filtered else "PASS"


def validate_inputs():
    """Bind inputs to the frozen manifest, remapping repository files to this tree."""
    manifest = json.loads((T0 / "input-manifest.json").read_text())
    baseline = Path(manifest["source_root"])
    for original, digest in manifest["input_hashes"].items():
        path = Path(original)
        if path.is_relative_to(baseline):
            path = ROOT / path.relative_to(baseline)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input unavailable: {path}")
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return sha(T0 / "input-manifest.json")


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
        import warp as wp  # noqa: PLC0415

        from scripts.monolithic_reference.check_p3_recording import check, compare  # noqa: PLC0415

        if not wp.get_device("cuda:0").is_cuda:
            raise ValueError("P4 profiles require CUDA")
        recording = output / "recording"
        if case["id"].startswith("franka"):
            from newton.examples.softbody.example_monolithic_franka_grasp import Example  # noqa: PLC0415
            from newton.viewer import ViewerNull  # noqa: PLC0415

            args = SimpleNamespace(
                device="cuda:0",
                output=recording,
                asset_cache=ARTIFACTS / "p3/pr3/franka-grasp-v1/cache",
                law=case["law"],
                globalization=case["globalization"],
            )
            example = Example(ViewerNull(), args)
            while not example.closed:
                example.step()
            example.test_final()
            reference = T0 / "baseline" / case["id"] / "recording"
        else:
            from newton.examples.softbody.monolithic_sharpa_recorded import run  # noqa: PLC0415

            recording.mkdir()
            args = SimpleNamespace(
                output=recording, trajectory=ARTIFACTS / "p3/pr3/current-iterate-production-v1/sharpa-trajectory.json"
            )
            if not run(args):
                raise ValueError("Sharpa did not publish its complete trajectory")
            reference = ARTIFACTS / "p3/pr3/current-iterate-production-v1/sharpa"
        receipt["recording"] = check(recording, case["steps"])
        protocol = json.loads((T0 / "comparison-tolerances.json").read_text())
        receipt["comparison"] = compare(
            reference, recording, protocol["cases"][case["id"]]["absolute"], legacy_fp64=True
        )
        if receipt["comparison"]["status"] not in ("PASS_EXACT", "PASS_FROZEN_ENVELOPE"):
            raise ValueError("Recording differs outside the frozen envelope")
        actual = json.loads((recording / "configuration.json").read_text())
        frozen = json.loads((reference / "configuration.json").read_text())
        if actual["config_digest"] != frozen["config_digest"]:
            raise ValueError("Physical configuration digest changed")
        receipt["modules"] = module_origins(ROOT, source_hashes)
        receipt["status"] = "PASS"
    except Exception:
        receipt["exception"] = traceback.format_exc()
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return int(receipt["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("dev", "full"), default="dev")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path)
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
        results.append(
            {
                "case": case["id"],
                "status": "PASS" if process.returncode == 0 and receipt["status"] == "PASS" else "FAIL",
                "receipt": str(path),
                "exit_code": process.returncode,
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
        print(json.dumps(results[-1]), flush=True)
    return int(summary["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
