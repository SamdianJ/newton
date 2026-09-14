# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare nightly final states without mistaking repeat variability for an accuracy bound."""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def summarize_accuracy(jobs, completed):
    groups = {}
    for job in jobs:
        if "accuracy" in job:
            groups.setdefault(job["accuracy"]["group"], []).append(job)
    results = []
    for name, group_jobs in groups.items():
        states = {}
        errors = []
        sources = {}
        for job in group_jobs:
            metadata = job["accuracy"]
            result = completed.get(job["name"])
            if result is None or not result["screen"]["screen_passed"]:
                errors.append(f"{job['name']}: missing completed run or numerical screen failed")
                continue
            output = Path(result["output"])
            try:
                index = json.loads((output / "evidence-index.json").read_text())
                hashes = {}
                for file in ("final_state.npz", "execution-options.json"):
                    hashes[file] = hashlib.sha256((output / file).read_bytes()).hexdigest()
                    if index[file] != hashes[file]:
                        raise ValueError(f"Evidence hash mismatch: {file}")
                options = json.loads((output / "execution-options.json").read_text())
                production = metadata["mode"] == "production_graph"
                expected = {
                    "pcg_mode": "production" if production else "diagnostic",
                    "use_cuda_graph": production,
                    "use_optimized_articulation_mass_matrix": metadata["mass"] == "owned",
                    "linear_absolute_tolerance": 1e-6,
                }
                if options["failure"] is not None or options["constructions"] != [expected, expected]:
                    raise ValueError("Unexpected actual solver options")
                with np.load(output / "final_state.npz", allow_pickle=False) as final:
                    arrays = {key: final[key].astype(np.float64) for key in ("q", "qd", "x", "v")}
                if not all(np.isfinite(array).all() for array in arrays.values()):
                    raise ValueError("Nonfinite final state")
                states[job["name"]] = arrays
                sources[job["name"]] = {"output": str(output), "sha256": hashes}
            except (FileNotFoundError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as error:
                errors.append(f"{job['name']}: {type(error).__name__}: {error}")

        comparisons = {key: [] for key in ("cross_mode", "diagnostic_repeat", "production_graph_repeat")}
        for pair in combinations(group_jobs, 2):
            left, right = pair
            a, b = left["accuracy"], right["accuracy"]
            if a["mode"] == b["mode"]:
                category = a["mode"] + "_repeat"
            elif a["repeat"] == b["repeat"]:
                category = "cross_mode"
                if a["mode"] != "diagnostic":
                    left, right = right, left
            else:
                continue
            if left["name"] not in states or right["name"] not in states:
                continue
            lhs, rhs = states[left["name"]], states[right["name"]]
            if any(lhs[key].shape != rhs[key].shape for key in lhs):
                errors.append(f"{left['name']} / {right['name']}: final state shape mismatch")
                continue
            metrics = {}
            for key, array in lhs.items():
                delta = rhs[key] - array
                norm = float(np.linalg.norm(array))
                metrics[key] = {
                    "max_abs": float(np.max(np.abs(delta), initial=0)),
                    "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
                    "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
                }
            comparisons[category].append({"left": left["name"], "right": right["name"], "arrays": metrics})
        results.append(
            {
                "group": name,
                "status": "INCOMPLETE" if errors else "REVIEW_REQUIRED",
                "expected_runs": len(group_jobs),
                "available_runs": len(states),
                "errors": errors,
                "sources": sources,
                "comparisons": comparisons,
                "max_position_difference": {
                    category: max((pair["arrays"]["x"]["max_abs"] for pair in pairs), default=None)
                    for category, pairs in comparisons.items()
                },
            }
        )
    return {
        "scope": "Final states only; matched diagnostic versus production graph, separated by mesh and mass mode.",
        "admission": "REVIEW_REQUIRED: physical error limits are uncalibrated; diagnostic is not ground truth. "
        "Within-mode variation is descriptive, never a pass threshold. This does not isolate stable graphs from PCG policy.",
        "metrics": "Float64 differences; max_abs/rms are per coordinate. x: m; v: m/s; q/qd: joint-dependent units. "
        "relative_l2 uses the left state norm and is null for a zero norm. Empty arrays have zero absolute error. "
        "Missing repeat comparisons are null, not zero variability.",
        "groups": results,
    }
