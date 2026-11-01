# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the frozen quick regression case manifest."""

import hashlib
import json
import re
from pathlib import Path

SCENES = ("sharpa", "franka")
PRECISIONS = ("fp64", "fp32")
REQUIRED_GATES = ("hard_failures_max", "stopped_max", "penetration_max_m", "det_f_min")
GRASP_KEYS = ("last_steps", "ground_weight_fraction_max")


def _check_case(case):
    if not isinstance(case.get("id"), str) or not re.fullmatch(r"[a-z0-9-]+", case["id"]):
        raise ValueError(f"Invalid case id: {case.get('id')!r}")
    if case.get("scene") not in SCENES:
        raise ValueError(f"{case['id']}: unknown scene {case.get('scene')!r}")
    snapshot = case.get("snapshot", {})
    if (
        not isinstance(snapshot.get("path"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("sha256", "")))
        or type(snapshot.get("step")) is not int
        or snapshot["step"] < 0
    ):
        raise ValueError(f"{case['id']}: snapshot needs a path, a sha256 and a nonnegative step")
    if type(case.get("steps")) is not int or case["steps"] <= 0:
        raise ValueError(f"{case['id']}: steps must be a positive integer")
    gates = case.get("gates", {})
    if not gates or set(gates) - set(PRECISIONS):
        raise ValueError(f"{case['id']}: gates must be keyed by fp64 and/or fp32")
    for precision, limits in gates.items():
        if set(limits) - {"grasp"} != set(REQUIRED_GATES):
            raise ValueError(f"{case['id']} {precision}: gates must be exactly {REQUIRED_GATES} plus optional grasp")
        grasp = limits.get("grasp")
        if grasp is not None and set(grasp) != set(GRASP_KEYS):
            raise ValueError(f"{case['id']} {precision}: grasp needs {GRASP_KEYS}")
        if grasp is not None and not 0 < grasp["last_steps"] <= case["steps"]:
            raise ValueError(f"{case['id']} {precision}: grasp window must lie inside the case window")


def load_manifest(path):
    """Return the validated manifest with its path and file sha256."""
    path = Path(path)
    payload = json.loads(path.read_text())
    if type(payload.get("version")) is not int or not isinstance(payload.get("cases"), list):
        raise ValueError("Manifest needs an integer version and a case list")
    seen = set()
    for case in payload["cases"]:
        _check_case(case)
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']}")
        seen.add(case["id"])
    return {**payload, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
