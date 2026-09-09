# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P2-0A cross-checks after separate Newton and dt sweeps."""

import json
from pathlib import Path

from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources
from scripts.monolithic_reference.run_p2_sharpa_study import run_case, warmup

ROOT = Path("/home/lightwheel/Desktop/newton/SamiulJ/agents/integration/artifacts/p2/pr8a-measurement/sharpa-cross")
JOBS = [
    {
        "name": "0a-cross-r3-n01-s04",
        "task": "0a-cross",
        "mesh": "r3",
        "newton_max_iterations": 1,
        "physical_dt_s": 0.0025,
    },
    {
        "name": "0a-cross-r3-n01-s05",
        "task": "0a-cross",
        "mesh": "r3",
        "newton_max_iterations": 1,
        "physical_dt_s": 0.002,
    },
    {
        "name": "0a-cross-r2-n01-s10",
        "task": "0a-cross",
        "mesh": "r2",
        "newton_max_iterations": 1,
        "physical_dt_s": 0.001,
    },
    {
        "name": "0a-cross-r2-n10-s04",
        "task": "0a-cross",
        "mesh": "r2",
        "newton_max_iterations": 10,
        "physical_dt_s": 0.0025,
    },
    {
        "name": "0a-cross-r3-n02-s04",
        "task": "0a-cross",
        "mesh": "r3",
        "newton_max_iterations": 2,
        "physical_dt_s": 0.0025,
    },
]


def main():
    snapshot_sources(ROOT / "provenance")
    index = []
    for job in JOBS:
        print("START", job["name"], flush=True)
        warmup(job["mesh"], ROOT / job["name"])
        compact = run_case(job, ROOT / job["name"])
        index.append(
            {
                "name": job["name"],
                "failure": compact.get("failure"),
                "stability": compact.get("quality", {}).get("stability_passed"),
            }
        )
        (ROOT / "cross-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2), flush=True)
    evidence_index(ROOT)


if __name__ == "__main__":
    main()
