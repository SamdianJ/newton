# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize every frozen PR1 trajectory without selecting a different mode per scene."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.verify_p3_tet import MASSES, MATERIALS, MODES, save

SCENES = ("stationary", "translate", "fall", "axial", "transverse", "release")


def summarize(root):
    """Require the full Cartesian matrix and report matched CPU/CUDA positions independently."""
    jobs, missing = {}, []
    failures, comparison = [], []
    for device, material, mass, scene, mode in itertools.product(("cpu", "cuda_0"), MATERIALS, MASSES, SCENES, MODES):
        key = (device, material, mass, scene, mode)
        file = root / ("-".join(key) + ".json")
        if not file.is_file():
            missing.append(file.name)
            continue
        data = json.loads(file.read_text())
        jobs[key] = data
        if not data["passed"]:
            failure = data.get("failure") or {}
            failures.append(
                {
                    "job": file.name,
                    "gates": data.get("gates"),
                    "status": failure.get("status"),
                    "step": failure.get("step"),
                    "traceback": failure.get("traceback", data.get("traceback")),
                }
            )
    for material, mass, scene, mode in itertools.product(MATERIALS, MASSES, SCENES, MODES):
        cpu = jobs.get(("cpu", material, mass, scene, mode))
        cuda = jobs.get(("cuda_0", material, mass, scene, mode))
        if cpu and cuda and not cpu.get("failure") and not cuda.get("failure") and "states" in cpu and "states" in cuda:
            x, y = np.asarray(cpu["states"]), np.asarray(cuda["states"])
            if x.shape == y.shape and x.size:
                comparison.append(
                    {"case": [material, mass, scene, mode], "position_difference_max": float(np.max(np.abs(x - y)))}
                )
    return {
        "complete": not missing,
        "job_count": len(jobs),
        "passed_count": sum(d["passed"] for d in jobs.values()),
        "common_passing_modes": [
            mode for mode in MODES if not missing and all(d["passed"] for k, d in jobs.items() if k[-1] == mode)
        ],
        "scenes": {
            scene: dict(Counter("pass" if d["passed"] else "fail" for k, d in jobs.items() if k[3] == scene))
            for scene in SCENES
        },
        "failures": failures,
        "missing": missing,
        "cpu_cuda_position_differences": comparison,
        "time_refinement": "NOT_RUN_PENDING_COMMON_PASSING_MODE",
        "status": "BASELINE_ONLY_NOT_PR1_CERTIFICATION",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, summarize(args.input))
