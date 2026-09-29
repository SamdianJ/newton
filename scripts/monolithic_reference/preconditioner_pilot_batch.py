# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run serial, alternating, independent-process PR8F pilot jobs; retain failures and continue."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.monolithic_reference.preconditioner_study import CANDIDATES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("matrix",), required=True)
    parser.add_argument("--matrix-list", type=Path)
    parser.add_argument("--candidates", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.matrix_list is None:
        parser.error("Positive repeats and a matrix list for matrix jobs are required")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for repeat in range(args.repeats):
        candidates = args.candidates if repeat % 2 == 0 else list(reversed(args.candidates))
        scenes = [("matrix", 0)]
        for scene, refinement in scenes:
            for candidate in candidates:
                name = f"rep{repeat:02d}-{scene}{refinement}-{candidate}"
                module = "preconditioner_pilot"
                command = [
                    sys.executable,
                    "-m",
                    f"scripts.monolithic_reference.{module}",
                    "--candidate",
                    candidate,
                    "--output",
                    str(args.output / name),
                ]
                command += ["--matrix-list", str(args.matrix_list.resolve())]
                jobs.append({"name": name, "command": command, "status": "PENDING", "repeat": repeat})
    manifest = {"phase": "PILOT_BUDGET_NOT_FROZEN", "jobs": jobs, "formal_admission": False}

    def save():
        (args.output / "jobs.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for job in jobs:
        job.update(status="RUNNING", start_unix_s=time.time())
        save()
        print("START", job["name"], flush=True)
        with (args.output / (job["name"] + ".log")).open("w") as log:
            try:
                result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, timeout=2400, check=False)
                job.update(returncode=result.returncode, status="COMPLETED" if result.returncode == 0 else "FAILED")
            except (OSError, subprocess.TimeoutExpired) as error:
                job.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        job["seconds"] = time.time() - job["start_unix_s"]
        save()
        print(job["status"], job["name"], flush=True)
    return int(any(job["status"] != "COMPLETED" for job in jobs))


if __name__ == "__main__":
    raise SystemExit(main())
