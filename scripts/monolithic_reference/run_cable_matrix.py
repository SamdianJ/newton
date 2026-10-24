# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run mesh and finite-EA gates for the frozen cable analytical experiments."""

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np

from .cable_benchmarks import RECIPES


def main():
    """Collect every case, preserving failures and refusing mixed-source aggregate PASS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("cantilever", "localized_helix"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    r = RECIPES[args.kind]
    cases, failures, comparisons = {}, [], {}
    identity = None
    matrix_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    axes = (1, 2) if args.kind == "cantilever" else (1,)
    experiments = [
        (axis, n, ea)
        for axis in axes
        for n, ea in [(n, 1) for n in r["segments"]]
        + [(r["segments"][-1], ea) for ea in r["ea_multipliers"] if ea != 1]
    ]

    def run_case(experiment):
        axis, n, ea = experiment
        name = f"axis-{axis}-n-{n}-ea-{ea}"
        output = args.output / name
        command = [
            sys.executable,
            "-B",
            "-m",
            "scripts.monolithic_reference.run_cable_benchmark",
            args.kind,
            "--axis",
            str(axis),
            "--segments",
            str(n),
            "--ea-multiplier",
            str(ea),
            "--device",
            args.device,
            "--pcg-mode",
            args.pcg_mode,
            "--precision",
            args.precision,
            "--output",
            str(output),
        ]
        if args.use_cuda_graph:
            command.append("--use-cuda-graph")
        if args.resume and output.exists():
            command.append("--resume")
        with (args.output / (name + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt_path = output / "receipt.json"
        receipt = (
            json.loads(receipt_path.read_text())
            if receipt_path.exists()
            else {
                "status": "FAIL",
                "sha256": None,
            }
        )
        return name, process.returncode, receipt

    # Independent CPU experiments may overlap; GPU experiments remain serial.
    with ThreadPoolExecutor(max_workers=4 if args.device == "cpu" else 1) as executor:
        for name, returncode, receipt in executor.map(run_case, experiments):
            cases[name] = {"status": receipt["status"], "metrics": receipt.get("metrics"), "returncode": returncode}
            if identity is None:
                identity = receipt["sha256"]
            if receipt["sha256"] != identity:
                failures.append(name + ": mixed source identities")
            if receipt.get("precision") != args.precision:
                failures.append(name + ": precision differs from the matrix")
            if returncode or receipt["status"] not in ("PASS", "MESH_SAMPLE", "EA_SAMPLE"):
                failures.append(name + ": failed/incomplete experiment")
            print(name, receipt["status"], flush=True)
    for axis in axes:
        mesh = [cases[f"axis-{axis}-n-{n}-ea-1"]["metrics"] for n in r["segments"]]
        if all(m and m["complete"] for m in mesh):
            errors = [m["relative_centerline_error"] for m in mesh]
            ratios = [b / a for a, b in pairwise(errors)]
            comparisons[f"axis-{axis}-mesh"] = {"errors": errors, "refinement_ratios": ratios}
            if max(ratios) > r["gates"]["refinement_ratio"]:
                failures.append(f"axis-{axis}: mesh refinement")
        n = r["segments"][-1]
        base_path = args.output / f"axis-{axis}-n-{n}-ea-1" / "final.npz"
        if not base_path.exists() or not cases[f"axis-{axis}-n-{n}-ea-1"]["metrics"]["complete"]:
            continue
        base = np.load(base_path)["positions"]
        for ea in r["ea_multipliers"]:
            if ea == 1:
                continue
            other_path = args.output / f"axis-{axis}-n-{n}-ea-{ea}" / "final.npz"
            sample = cases[f"axis-{axis}-n-{n}-ea-{ea}"]["metrics"]
            if not other_path.exists() or not sample or not sample["complete"]:
                continue
            other = np.load(other_path)["positions"]
            normalization = (
                r["force"] * r["length"] ** 3 / (3 * r["bending"][axis - 1])
                if args.kind == "cantilever"
                else r["shortening"] / 2
            )
            if args.kind == "localized_helix":
                # Remove only the arbitrary transverse phase between equilibria.
                b, a = base[:, 1] + 1j * base[:, 2], other[:, 1] + 1j * other[:, 2]
                rotated = a * np.exp(-1j * np.angle(np.vdot(b, a)))
                other[:, 1], other[:, 2] = rotated.real, rotated.imag
            change = float(np.max(np.linalg.norm(base - other, axis=1)) / normalization)
            comparisons[f"axis-{axis}-ea-{ea}"] = change
            if ea > 1 and change > r["gates"]["finite_EA_response_change"]:
                failures.append(f"axis-{axis}: finite EA {ea}")
        if args.kind == "localized_helix":
            low = comparisons.get(f"axis-{axis}-ea-0.1")
            high = comparisons.get(f"axis-{axis}-ea-10")
            if low is not None and high is not None:
                ratio = high / max(low, np.finfo(float).tiny)
                comparisons[f"axis-{axis}-ea-refinement"] = ratio
                if ratio > r["gates"]["finite_EA_refinement_ratio"]:
                    failures.append(f"axis-{axis}: finite EA error did not decay")
    summary = {
        "status": "FAIL" if failures else "PASS",
        "kind": args.kind,
        "precision": args.precision,
        "cases": cases,
        "comparisons": comparisons,
        "failures": failures,
        "sha256": identity,
        "matrix_sha256": matrix_identity,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cases", "sha256")}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
