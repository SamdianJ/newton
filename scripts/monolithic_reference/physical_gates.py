# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical-plausibility gates for complete robot scene summaries."""

import json
from pathlib import Path

GATES_PATH = Path(__file__).with_name("fixtures") / "physical_gates_v1.json"
_COUNT_KEYS = ("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE")


def load_gates(path=GATES_PATH, *, allow_proposed=False):
    """Load a version-1 gate recipe; proposed recipes are accepted only for diagnostics."""
    gates = json.loads(Path(path).read_text())
    if gates.get("version") != 1:
        raise ValueError("Unsupported physical gate version")
    if gates.get("status") != "FROZEN" and not (allow_proposed and gates.get("status") == "PROPOSED"):
        raise ValueError("Physical gates must be a frozen recipe")
    return gates


def _counts(summary):
    counts = summary.get("counts", summary.get("publication_counts"))
    if counts is None or set(counts) != set(_COUNT_KEYS):
        raise ValueError("Summary lacks complete publication counts")
    return counts


def evaluate_robot(summary, gates):
    """Return the status and every violated gate name for one robot summary."""
    counts = _counts(summary)
    steps = sum(counts.values())
    checks = {
        "complete": bool(summary["complete"]) and steps > 0,
        "hard_failures": counts["HARD_FAILURE"] <= gates["hard_failures_max"],
        "stopped_fraction": steps > 0 and counts["STOPPED_ACCEPTED"] / steps <= gates["stopped_fraction_max"],
        "penetration": summary["maximum_penetration_m"] <= gates["penetration_max_m"],
        "det_f": summary["minimum_det_f"] > gates["det_f_min"],
    }
    if gates["require_held_cube"]:
        checks["held_cube"] = bool(summary.get("held_cube"))
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {"status": "FAIL" if failures else "PASS", "failures": failures, "steps": steps}
