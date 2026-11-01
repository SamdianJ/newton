# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate frozen quick regression gates on the per-step rows of one window."""


def _gate(value, limit, passed):
    return {"value": value, "limit": limit, "passed": bool(passed)}


def evaluate(rows, gates, *, weight, expected_steps):
    """Return ``{gate: {value, limit, passed}}`` for one case window.

    A window that ends early fails ``complete``; Stopped steps are counted, never treated as converged.
    """
    published = [row for row in rows if row["publication"] != "HARD_FAILURE"]
    stopped = sum(row["publication"] == "STOPPED_ACCEPTED" for row in rows)
    hard = len(rows) - len(published)
    penetration = max((row["penetration"] for row in published), default=0.0)
    det_f = min((row["min_det_f"] for row in published), default=1.0)
    result = {
        "complete": _gate(len(rows), expected_steps, len(rows) == expected_steps),
        "hard_failures": _gate(hard, gates["hard_failures_max"], hard <= gates["hard_failures_max"]),
        "stopped_steps": _gate(stopped, gates["stopped_max"], stopped <= gates["stopped_max"]),
        "penetration": _gate(penetration, gates["penetration_max_m"], penetration <= gates["penetration_max_m"]),
        "det_f": _gate(det_f, gates["det_f_min"], det_f > gates["det_f_min"]),
    }
    grasp = gates.get("grasp")
    if grasp is not None:
        tail = published[-grasp["last_steps"] :]
        ground = max((abs(row["ground_force_z"]) for row in tail), default=float("inf"))
        limit = grasp["ground_weight_fraction_max"] * weight
        result["grasp"] = _gate(ground, limit, len(tail) == grasp["last_steps"] and ground <= limit)
    return result
