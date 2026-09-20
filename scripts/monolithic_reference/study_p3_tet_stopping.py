# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test predeclared private stopping candidates without changing the production defaults."""

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.monolithic_reference import verify_p3_tet as verify


def study(output):
    """Preserve every candidate outcome before selecting a production stopping contract."""
    candidates = {"A": 1e-7, "B": 1e-8}
    cases = (("consistent", "fall"), ("lumped", "transverse"), ("consistent", "release"))
    sources = [Path(__file__), Path(verify.__file__), Path("newton/_src/solvers/monolithic/solver_monolithic.py")]
    output.mkdir(parents=True, exist_ok=True)
    verify.save(
        output / "declaration.json",
        {
            "candidates_absolute_rms": candidates,
            "relative": 1e-6,
            "small_step": 1e-10,
            "cases": cases,
            "mode": "ArmijoOrResNorm",
            "material": "kim_stable_no_log",
            "device": "cpu",
            "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        },
    )
    original = verify.solver_for
    rows = []
    for name, absolute in candidates.items():

        def configured(*args, absolute=absolute, **kwargs):
            solver = original(*args, **kwargs)
            solver._config = replace(
                solver._config,
                merit_absolute_global=absolute,
                merit_absolute_x=absolute,
                merit_relative_global=1e-6,
                merit_relative_x=1e-6,
                step_tolerance_global=1e-10,
                step_tolerance_x=1e-10,
            )
            return solver

        for mass, scene in cases:
            with patch.object(verify, "solver_for", side_effect=configured):
                result = verify.trajectory(
                    "cpu", "kim_stable_no_log", mass, "ArmijoOrResNorm", scene, 0.001, "diagnostic", False
                )
            key = f"{name}-{mass}-{scene}"
            verify.save(output / f"{key}.json", result)
            rows.append({"job": key, "passed": result["passed"], "gates": result["gates"]})
            verify.save(output / "summary.json", rows)
            print(key, result["passed"], result["gates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study(args.output)
