# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize P2-0 diagnostic artifacts into tables used by the report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WORKSPACE = Path("/home/lightwheel/Desktop/newton/SamiulJ")
DEFAULT_ROOT = WORKSPACE / "agents/integration/artifacts/p2/p2-0"


def load_compacts(root: Path, kind: str):
    rows = []
    base = root / kind
    if not base.exists():
        return rows
    for compact in sorted(base.rglob("compact.json")):
        data = json.loads(compact.read_text())
        data["_path"] = str(compact)
        rows.append(data)
    return rows


def newton_table(rows):
    table = []
    for row in rows:
        if row.get("task") not in {"0a-newton", "0a-newton-control", "0a-repeat"} and not str(
            row.get("name", "")
        ).startswith("0a-newton"):
            continue
        q = row.get("quality", {})
        work = row.get("work", {}).get("per_simulated_second", {})
        table.append(
            {
                "name": row.get("name"),
                "mesh": row.get("mesh"),
                "newton_max_iterations": row.get("newton_max_iterations"),
                "complete": row.get("complete"),
                "stability_passed": q.get("stability_passed"),
                "normal_convergence_fraction": q.get("normal_convergence_fraction"),
                "soft_stop_count": q.get("soft_stop_count"),
                "failure": q.get("failure") or row.get("failure"),
                "max_penetration_mm": None
                if q.get("maximum_penetration_m") is None
                else 1000 * q["maximum_penetration_m"],
                "min_det_f": q.get("minimum_det_f"),
                "wall_s_per_sim_s": row.get("wall_seconds_per_simulated_second"),
                "newton_updates_per_sim_s": work.get("newton_updates"),
                "assemblies_per_sim_s": work.get("assemblies"),
                "pcg_iters_per_sim_s": work.get("pcg_iterations"),
                "hold_nl_p50": row.get("work", {})
                .get("stages", {})
                .get("hold", {})
                .get("nonlinear_iterations", {})
                .get("p50"),
                "hold_step_ms_p50": row.get("work", {}).get("stages", {}).get("hold", {}).get("step_ms", {}).get("p50"),
            }
        )
    table.sort(key=lambda r: (r.get("mesh") or "", r.get("newton_max_iterations") or 0, r.get("name") or ""))
    return table


def dt_table(rows):
    table = []
    for row in rows:
        name = row.get("name") or ""
        if not (name.startswith("0a-dt") or name.startswith("0a-newton-r3-n10")):
            continue
        if name.startswith("0a-newton") and row.get("newton_max_iterations") != 10:
            continue
        q = row.get("quality", {})
        work = row.get("work", {}).get("per_simulated_second", {})
        table.append(
            {
                "name": name,
                "mesh": row.get("mesh"),
                "substeps": row.get("substeps"),
                "physical_dt_ms": None if row.get("physical_dt_s") is None else 1000 * row["physical_dt_s"],
                "complete": row.get("complete"),
                "stability_passed": q.get("stability_passed"),
                "normal_convergence_fraction": q.get("normal_convergence_fraction"),
                "physical_coverage_s": q.get("physical_coverage_s"),
                "failure": q.get("failure") or row.get("failure"),
                "max_penetration_mm": None
                if q.get("maximum_penetration_m") is None
                else 1000 * q["maximum_penetration_m"],
                "min_det_f": q.get("minimum_det_f"),
                "wall_s_per_sim_s": row.get("wall_seconds_per_simulated_second"),
                "hold_step_ms_p50": row.get("work", {}).get("stages", {}).get("hold", {}).get("step_ms", {}).get("p50"),
                "pcg_iters_per_sim_s": work.get("pcg_iterations"),
            }
        )
    table.sort(key=lambda r: (r.get("mesh") or "", r.get("physical_dt_ms") or 0, r.get("name") or ""))
    return table


def baseline_repeats(rows):
    selected = [
        r for r in rows if str(r.get("name", "")).startswith("0b-uninstrumented") or r.get("name") == "0a-newton-r3-n10"
    ]
    by_mesh = {}
    for row in selected:
        mesh = row.get("mesh")
        stages = row.get("work", {}).get("stages", {})
        by_mesh.setdefault(mesh, []).append(
            {
                "name": row.get("name"),
                "hold_p50": stages.get("hold", {}).get("step_ms", {}).get("p50"),
                "hold_p95": stages.get("hold", {}).get("step_ms", {}).get("p95"),
                "close_p50": stages.get("close", {}).get("step_ms", {}).get("p50"),
                "prepare_p50": stages.get("prepare", {}).get("step_ms", {}).get("p50"),
                "wall_s_per_sim_s": row.get("wall_seconds_per_simulated_second"),
                "mempool_high": (row.get("mempool") or {}).get("high_bytes"),
                "stability_passed": row.get("quality", {}).get("stability_passed"),
            }
        )
    summary = {}
    for mesh, items in by_mesh.items():
        hold = [i["hold_p50"] for i in items if i["hold_p50"] is not None]
        summary[mesh] = {
            "repeats": len(items),
            "hold_p50_mean": None if not hold else float(np.mean(hold)),
            "hold_p50_std": None if len(hold) < 2 else float(np.std(hold)),
            "runs": items,
        }
    return summary


def tet_summary(rows):
    out = []
    for row in rows:
        out.append(
            {
                "kind": row.get("kind"),
                "device": row.get("actual_device") or row.get("requested_device"),
                "refinement": row.get("refinement"),
                "tets": row.get("tets"),
                "failure": row.get("failure"),
                "wall_s_per_sim_s": row.get("wall_seconds_per_simulated_second"),
                "overall_solver_p50": (row.get("overall") or {}).get("solver_ms", {}).get("p50")
                if row.get("overall")
                else None,
                "overall_solver_p95": (row.get("overall") or {}).get("solver_ms", {}).get("p95")
                if row.get("overall")
                else None,
                "frame_6sub_p50": ((row.get("frame_solver_ms_6substeps") or {}).get("hold") or {}).get("p50"),
                "nl_p50": (row.get("overall") or {}).get("nonlinear_iterations", {}).get("p50")
                if row.get("overall")
                else None,
                "lin_p50": (row.get("overall") or {}).get("linear_iterations", {}).get("p50")
                if row.get("overall")
                else None,
                "path": row.get("_path"),
            }
        )
    return out


def recommend(newton, dt_rows, baselines, tet):
    passing_n = [
        r
        for r in newton
        if r.get("mesh") == "r3" and r.get("stability_passed") and (r.get("normal_convergence_fraction") or 0) >= 0.99
    ]
    passing_h = [
        r
        for r in dt_rows
        if r.get("mesh") == "r3" and r.get("stability_passed") and (r.get("normal_convergence_fraction") or 0) >= 0.99
    ]
    rec = {
        "status": "PENDING_USER_REVIEW",
        "sharpa_newton_limits_that_met_inherited_gates": sorted(
            {r["newton_max_iterations"] for r in passing_n if r.get("newton_max_iterations") is not None}
        ),
        "sharpa_substeps_that_met_inherited_gates": sorted(
            {r["substeps"] for r in passing_h if r.get("substeps") is not None}
        ),
        "notes": [
            "Production defaults remain N=10 and h=1 ms until the user accepts a temporal bound.",
            "N=1 meets the 99% gate at 1 ms but has 29 soft stops and fails the gate at S=4/5.",
            "N=2 and N=10 passed S=4; the other caps above N=2 were not cross-validated at S=4.",
            "S=1 and S=2 pass stability but PCG p95 exceeds 64, which would trigger patch Schur.",
            "S=4 is a coarse-step candidate within the PCG budget; quiet repeated timing remains incomplete.",
            "Finest measured h=0.25 ms is not a converged continuum reference versus 1 ms.",
            "Single-finger 25 mm anchored-close is not a multi-finger grasp target.",
            "Tet gravity targets are recorded separately from Sharpa; PCG iterations dominate r5.",
            "Current profiling covers only the first window step; material samples use evolving candidates.",
            "Original tet timing repeats omitted gravity; original compact FPS fields are real-time factors.",
        ],
    }
    rec["sharpa_newton_candidate"] = 2
    rec["sharpa_substeps_candidate"] = 4
    rec["production_defaults_unchanged"] = {"newton_max_iterations": 10, "substeps": 10, "physical_dt_s": 0.001}
    rec["baselines"] = baselines
    rec["tet_primary_r5"] = [r for r in tet if r.get("refinement") == 5]
    return rec


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def evidence_index(root: Path):
    """Hash all evidence except the index and its own digest."""
    rows = []
    outputs = {root / "evidence-index.json", root / "evidence-index.sha256"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in outputs:
            continue
        rows.append(hash_file(path))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    sharpa = load_compacts(args.root, "sharpa")
    tet = load_compacts(args.root, "tet")
    report = {
        "sharpa_runs": len(sharpa),
        "tet_runs": len(tet),
        "newton": newton_table(sharpa),
        "timestep": dt_table(sharpa),
        "baseline_repeats": baseline_repeats(sharpa),
        "tet": tet_summary(tet),
    }
    report["recommendation"] = recommend(
        report["newton"], report["timestep"], report["baseline_repeats"], report["tet"]
    )
    out = args.root / "summary.json"
    payload = json.dumps(report, indent=2, default=str) + "\n"
    out.write_text(payload)
    (args.root / "summary.sha256").write_text(hashlib.sha256(payload.encode()).hexdigest() + "\n")
    index = evidence_index(args.root)
    index_path = args.root / "evidence-index.json"
    index_payload = json.dumps({"files": index, "count": len(index)}, indent=2) + "\n"
    index_path.write_text(index_payload)
    (args.root / "evidence-index.sha256").write_text(hashlib.sha256(index_payload.encode()).hexdigest() + "\n")
    print(
        json.dumps(
            {"wrote": str(out), "sharpa_runs": len(sharpa), "tet_runs": len(tet), "evidence_files": len(index)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
