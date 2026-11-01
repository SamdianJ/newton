# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen quick regression windows and write per-case receipts and a summary.

Each case imports a snapshot, publishes its window with the recorded controls and
applies frozen gates. A failing or broken case never stops the other cases.
Comparison with earlier runs is diagnostic only and not part of this verdict.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import traceback
from pathlib import Path

from scripts.monolithic_reference.quick_gates import evaluate
from scripts.monolithic_reference.quick_manifest import PRECISIONS, load_manifest
from scripts.monolithic_reference.retained_assets import ASSET_ROOT

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("fixtures") / "quick_cases_v1.json"
SNAPSHOT_ROOT = ASSET_ROOT / "data"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_case(case, precision, *, device, asset_root, windows):
    """Return the receipt of one case at one precision."""
    from scripts.monolithic_reference.state_snapshots import load_snapshot  # noqa: PLC0415

    receipt = {"id": case["id"], "precision": precision, "scene": case["scene"], "snapshot": case["snapshot"]}
    if precision not in case["gates"]:
        return {**receipt, "status": "NOT_APPLICABLE"}
    try:
        values, model_values = load_snapshot(asset_root / case["snapshot"]["path"], case["snapshot"]["sha256"])
        window = windows[case["scene"]](precision, device)
        window.start(values, model_values, case["snapshot"]["step"])
        rows = []
        for _ in range(case["steps"]):
            rows.append(window.advance())
            if rows[-1]["publication"] == "HARD_FAILURE":
                break
        gates = evaluate(rows, case["gates"][precision], weight=window.weight, expected_steps=case["steps"])
        status = "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL"
        return {
            **receipt,
            "status": status,
            "gates": gates,
            "rows": rows,
            "scene_checks": window.scene_checks,
            "import": window.import_report,
        }
    except Exception as error:  # one broken case must not stop the others
        return {
            **receipt,
            "status": "ERROR",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def run(manifest_path, output, *, precisions, device, asset_root=SNAPSHOT_ROOT, windows=None, only=()):
    """Run every selected case and precision; return the summary written to ``output``/summary.json."""
    if windows is None:
        from scripts.monolithic_reference.quick_scenes import WINDOWS as windows  # noqa: PLC0415
    manifest = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in manifest["cases"]:
        if only and case["id"] not in only:
            continue
        for precision in precisions:
            receipt = run_case(case, precision, device=device, asset_root=Path(asset_root), windows=windows)
            path = output / case["id"] / precision / "receipt.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            entry = {name: receipt.get(name) for name in ("id", "precision", "status", "gates", "error")}
            cases.append({**entry, "receipt": str(path)})
    summary = {
        "manifest": manifest["path"],
        "manifest_version": manifest["version"],
        "manifest_sha256": manifest["sha256"],
        "commit": _commit(),
        "python": platform.python_version(),
        "precisions": list(precisions),
        "cases": cases,
        "status": "PASS" if all(case["status"] in ("PASS", "NOT_APPLICABLE") for case in cases) else "FAIL",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--precision", default="fp64,fp32", help="Comma-separated fp64,fp32")
    parser.add_argument("--case", action="append", default=[], help="Run only these case ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precisions = tuple(p for p in args.precision.split(",") if p)
    if not precisions or set(precisions) - set(PRECISIONS):
        parser.error("--precision takes fp64, fp32 or both")
    summary = run(args.manifest, args.output, precisions=precisions, device=args.device, only=set(args.case))
    print(
        json.dumps(
            {"status": summary["status"], "cases": [[c["id"], c["precision"], c["status"]] for c in summary["cases"]]}
        )
    )
    return int(summary["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
