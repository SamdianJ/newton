# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the G2H four-mode matrix and canonical dt/mesh refinements."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import FIXTURE, MODES, Case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("CPU_DIR", "CUDA_DIR"))
    parser.add_argument(
        "--calibration", action="store_true", help="Record exploratory results without claiming acceptance"
    )
    args = parser.parse_args()
    if args.compare:
        errors = {}
        cpu, cuda = map(Path, args.compare)
        if not all(json.loads((directory / "results.json").read_text())["passed"] for directory in (cpu, cuda)):
            raise SystemExit("Device parity requires two accepted formal runs")
        for p in cpu.glob("*/trace.jsonl"):
            other = cuda / p.parent.name / "trace.jsonl"
            if json.loads((p.parent / "manifest.json").read_text()) != json.loads(
                (other.parent / "manifest.json").read_text()
            ):
                raise SystemExit("G2H device fixture identity mismatch")
            a = [json.loads(line) for line in p.read_text().splitlines()]
            b = [json.loads(line) for line in other.read_text().splitlines()]
            if [v["time"] for v in a] != [v["time"] for v in b]:
                raise SystemExit("G2H device trace timestamp mismatch")
            errors[p.parent.name] = float(
                np.max(np.abs(np.array([v["tip"] for v in a]) - np.array([v["tip"] for v in b])))
            )
        passed = len(errors) == 16 and all(v <= FIXTURE["gates"]["device_tip_error"] for v in errors.values())
        result = {"passed": passed, "tip_error_m": errors, "threshold_m": FIXTURE["gates"]["device_tip_error"]}
        (cuda / "device-parity.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not passed:
            raise SystemExit("G2H device parity failed")
        return
    if not args.output:
        parser.error("--output is required for a simulation run")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for direction in ("axial", "transverse"):
        for material, mass in MODES:
            cases.append(
                (direction + "-" + material + "-" + mass, {"material": material, "mass": mass, "direction": direction})
            )
    for material, mass in MODES:
        cases.append(
            (
                "linear-" + material + "-" + mass,
                {"material": material, "mass": mass, "direction": "transverse", "load_scale": 0.25},
            )
        )
    for label, option in [
        ("dt-half", {"dt": 0.01}),
        ("dt-quarter", {"dt": 0.005}),
        ("mesh-coarse", {"refinement": 1}),
        ("mesh-fine", {"refinement": 3}),
    ]:
        cases.append(
            (label, dict(material="smith_log_stabilized", mass="consistent", direction="transverse", **option))
        )
    traces = {}
    summary = {}
    for label, options in cases:
        case = Case(args.device, **options)
        for _ in range(round(FIXTURE["duration"] / case.dt)):
            case.step()
        traces[label] = np.array([r["tip"] for r in case.records])
        directory = output / label
        directory.mkdir(exist_ok=True)
        for name, data in [("manifest", case.manifest), ("summary", case.summary())]:
            (directory / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
        (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in case.records))
        summary[label] = case.summary()
        print(label, summary[label], flush=True)
    baseline = traces["transverse-smith_log_stabilized-consistent"]
    half, quarter = traces["dt-half"], traces["dt-quarter"]
    dt_coarse = float(np.max(np.abs(baseline - quarter[3::4])))
    dt_fine = float(np.max(np.abs(half - quarter[1::2])))
    mesh_coarse = float(np.max(np.abs(traces["mesh-coarse"] - traces["mesh-fine"])))
    mesh_medium = float(np.max(np.abs(baseline - traces["mesh-fine"])))
    refinement = {
        "dt_coarse_error_m": dt_coarse,
        "dt_half_error_m": dt_fine,
        "mesh_coarse_error_m": mesh_coarse,
        "mesh_medium_error_m": mesh_medium,
        "passed": bool(
            dt_fine <= FIXTURE["gates"]["refinement_ratio"] * dt_coarse
            and mesh_medium <= FIXTURE["gates"]["refinement_ratio"] * mesh_coarse
        ),
    }
    result = {
        "schema": "monolithic-g2h-results/v1",
        "device": args.device,
        "calibration": args.calibration,
        "cases": summary,
        "refinement": refinement,
        "passed": bool(not args.calibration and all(v["passed"] for v in summary.values()) and refinement["passed"]),
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("refinement", refinement, flush=True)
    if not args.calibration and not result["passed"]:
        raise SystemExit("G2H failed")


if __name__ == "__main__":
    main()
