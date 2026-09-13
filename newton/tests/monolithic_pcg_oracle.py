# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent dense host oracle for the new fixed-checkpoint smart policy only."""

import numpy as np


def solve_fixed(matrix, rhs, inverse, nq, config, initial=None):
    """Solve a tiny system with the frozen checkpoints and breakdown recovery rules."""
    a = np.asarray(matrix, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    y = np.zeros_like(b) if initial is None else np.asarray(initial, dtype=np.float32).copy()
    r = np.zeros_like(b)
    p = np.zeros_like(b)
    gamma = np.float32(0)
    iterations = checks = replacements = 0
    waiting = False
    last = -1
    ratios = (np.nan,) * 3
    floors = np.asarray([config.residual_floor_global, config.residual_floor_q, config.residual_floor_x])

    def norms(v):
        return np.array([np.linalg.norm(w.astype(np.float64)) for w in (v, v[:nq], v[nq:])])

    denominator = np.maximum(norms(b), floors)

    def result(status, slot=-1):
        return {
            "status": status,
            "y": y,
            "iterations": iterations,
            "checks": checks,
            "replacements": replacements,
            "last": last,
            "convergence_slot": slot,
            "ratios": ratios,
        }

    def direction(restart, recovering):
        nonlocal gamma, p, waiting
        z = np.asarray(inverse @ r, dtype=np.float32)
        value = np.dot(r, z)
        if not np.isfinite(value):
            return "NONFINITE_ITERATION"
        if value < 0 or (value == 0 and recovering):
            return "NON_POSITIVE_PRECONDITIONED_RESIDUAL"
        if value == 0:
            waiting = True
            return None
        beta = np.float32(0) if restart else np.float32(value / gamma)
        if not np.isfinite(beta):
            return "NONFINITE_ITERATION"
        p = z.copy() if restart else z + beta * p
        gamma = value
        waiting = False
        return None if np.isfinite(p).all() else "NONFINITE_ITERATION"

    for slot in range(config.maximum_iterations + 1):
        if slot and not waiting:
            ap = a @ p
            curvature = np.dot(p, ap)
            if not np.isfinite(curvature):
                return result("NONFINITE_ITERATION")
            if curvature <= 0:
                return result("NON_POSITIVE_CURVATURE")
            alpha = np.float32(gamma / curvature)
            if not np.isfinite(alpha):
                return result("NONFINITE_ITERATION")
            y = y + alpha * p
            r = r - alpha * ap
            if not np.isfinite(y).all() or not np.isfinite(r).all():
                return result("NONFINITE_ITERATION")
            iterations += 1
        restart = recovering = False
        if slot == 0 or slot % config.true_residual_interval == 0 or slot == config.maximum_iterations:
            true = b - a @ y
            checks += 1
            if not np.isfinite(true).all():
                return result("NONFINITE_ITERATION")
            true_norm = norms(true)
            ratios = tuple(true_norm / denominator)
            last = slot
            if max(ratios) <= config.linear_tolerance:
                return result("SUCCESS", slot)
            if slot == config.maximum_iterations:
                return result("MAX_ITERATIONS")
            recovering = waiting
            restart = (
                slot == 0
                or waiting
                or max(norms(r - true) / np.maximum(true_norm, floors)) >= config.residual_replacement_threshold
            )
            if restart:
                r = true.copy()
                replacements += int(slot != 0)
        if not waiting or restart:
            error = direction(restart, recovering)
            if error:
                return result(error)
    raise AssertionError("Final checkpoint must terminate")
