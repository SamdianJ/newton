# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""PR8H three-grid cube envelope; retain every failure and continue the scan."""

import argparse
import copy
import json
from pathlib import Path

from newton.examples.softbody.monolithic_soft_cube import generate_cube, load_cube, validate_cube
from newton.examples.softbody.sharpa_close import sha256
from scripts.monolithic_reference.calibrate_sharpa_contact import CONFIG, run_case
from scripts.monolithic_reference.linear_evidence import evidence_index, snapshot_sources

ASSETS = Path(__file__).with_name("fixtures") / "soft_cube_40mm"


def cube_protocol(assets=ASSETS):
    """Bind the complete physical input; graph policy is recorded separately."""
    protocol = copy.deepcopy(CONFIG)
    protocol.update(
        schema="sharpa-cube-contact-calibration/v1",
        object_type="cube",
        edge_m=0.04,
        cells_per_axis=[4, 6, 8],
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        density=1000.0,
        young_modulus=10000.0,
        poisson_ratio=0.3,
        load_definition="per-jaw load = 2 * cube_mass * 9.81 / (2 * mu)",
        assets={str(r): sha256(assets / f"cube_r{r}.npz") for r in (1, 2, 3)},
    )
    return protocol


def build_assets(output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in (1, 2, 3):
        path = output / f"cube_r{r}.npz"
        if not path.exists():
            generate_cube(r).save(str(path))
        mesh = load_cube(path)
        report = validate_cube(mesh.vertices, mesh.tet_indices.reshape(-1, 4))
        if report["refinement"] != r:
            raise ValueError("Asset refinement mismatch")
        rows.append({**report, "file": path.name, "sha256": sha256(path)})
    (output / "manifest.json").write_text(json.dumps({"schema": "tet-cube/v1", "assets": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--build-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.build_assets:
        build_assets(args.assets)
        return 0
    if args.output is None:
        parser.error("--output is required for calibration")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = cube_protocol(args.assets)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    snapshot_sources(args.output)
    rows = []
    for k in CONFIG["stiffness_scan_n_m3"]:
        for r in (1, 2, 3):
            print(f"START cube r{r} k={k:g}", flush=True)
            destination = args.output / f"k{k:g}-r{r}"
            try:
                row = run_case(args.assets / f"cube_r{r}.npz", k, args.device, destination, cube=True)
            except Exception as error:
                row = {
                    "passed": False,
                    "stiffness": k,
                    "cube_sha256": protocol["assets"][str(r)],
                    "exception": f"{type(error).__name__}: {error}",
                }
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append({"refinement": r, **row})
            (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    passing = [
        k
        for k in CONFIG["stiffness_scan_n_m3"]
        if all(
            any(row["refinement"] == r and row["stiffness"] == k and row["passed"] for row in rows) for r in (1, 2, 3)
        )
    ]
    floor = min(passing) if passing else None
    result = {
        "status": "FROZEN" if floor else "FAILED_NO_PHYSICAL_ENVELOPE",
        "protocol": protocol,
        "protocol_sha256": sha256(args.output / "protocol.json"),
        "pcg_iteration_p95_budget": 64,
        "stiffness_floor_n_m3": floor,
        "initial_grasp_stiffness_n_m3": max(1e7, floor) if floor else None,
        "execution": {
            "pcg_mode": "production",
            "use_cuda_graph": args.device != "cpu",
            "rtol": 1e-4,
            "atol": 1e-6,
            "use_optimized_articulation_mass_matrix": True,
        },
        "results": rows,
    }
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    evidence_index(args.output)
    return 0 if floor else 1


if __name__ == "__main__":
    raise SystemExit(main())
