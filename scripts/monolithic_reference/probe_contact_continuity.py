# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect a terminal failed candidate without accepting or advancing it."""

import dataclasses
import json

import numpy as np
import warp as wp


def probe_failure(case, output):
    solver = case.solver
    accepted, trial = solver._transaction.accepted, solver._transaction.trial
    dt = case.fixture["dt"]
    scale = solver._linear.scale.numpy().astype(float)
    delta = solver._delta.numpy().astype(float)
    residual = accepted.residual.numpy().astype(float)
    contact = solver._contact._residual[0].numpy().astype(float)
    dense = solver._linear.densify_for_test(generation=solver._generation)
    predicted = (dense.raw_matrix @ delta) * scale
    z = accepted.z.numpy().astype(float)
    history = {name: getattr(solver._contact.committed, name).numpy() for name in ("valid", "xi_local", "normal_local")}
    zero = wp.zeros_like(solver._delta)
    opposite = wp.array(-delta, dtype=float, device=case.model.device)
    trial.form_trial(accepted, zero, 1.0, dt)
    zero_result = solver._evaluate_trial(trial, dt)
    zero_residual = trial.residual.numpy().astype(float)
    same = solver._evaluate_trial(accepted, dt)
    same_residual = accepted.residual.numpy().astype(float)
    # Use this consistent evaluation as the FD center, reporting any mode offset separately.
    rhat = scale * same_residual
    rows, derivatives = [], []
    arrays = {
        "z": z,
        "delta": delta,
        "scale": scale,
        "current_residual": residual,
        "zero_residual": zero_residual,
        "same_residual": same_residual,
        "raw_matrix": dense.raw_matrix,
        **{"history_" + name: value for name, value in history.items()},
    }
    for alpha in (
        0.25,
        -0.25,
        0.125,
        -0.125,
        0.0625,
        -0.0625,
        0.03125,
        -0.03125,
        0.015625,
        -0.015625,
        0.0078125,
        -0.0078125,
        0.00390625,
        -0.00390625,
    ):
        trial.form_trial(accepted, solver._delta if alpha > 0 else opposite, abs(alpha), dt)
        value = solver._evaluate_trial(trial, dt)
        r = trial.residual.numpy().astype(float)
        c = solver._contact._residual[1].numpy().astype(float)
        actual = trial.z.numpy().astype(float) - z
        suffix = str(alpha)
        arrays["r_" + suffix], arrays["c_" + suffix], arrays["dz_" + suffix] = r, c, actual
        rows.append(
            {
                "alpha": alpha,
                **dataclasses.asdict(value),
                "actual_coordinate_change_max_m_or_rad": float(np.max(np.abs(actual))),
                "residual_change_scaled_rms": float(np.linalg.norm((r - same_residual) * scale) / np.sqrt(len(r))),
                "contact_change_scaled_rms": float(np.linalg.norm((c - contact) * scale) / np.sqrt(len(r))),
            }
        )
    for h in (0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625):
        derivative = scale * (arrays["r_" + str(h)] - arrays["r_" + str(-h)]) / (2 * h)
        direction = (arrays["dz_" + str(h)] - arrays["dz_" + str(-h)]) / (2 * h)
        prediction = scale * (dense.raw_matrix @ direction)
        contact_derivative = scale * (arrays["c_" + str(h)] - arrays["c_" + str(-h)]) / (2 * h)
        derivatives.append(
            {
                "h": h,
                "actual_half_squared_merit_derivative": float(rhat @ derivative / len(rhat)),
                "assembled_half_squared_merit_derivative": float(rhat @ prediction / len(rhat)),
                "contact_half_squared_merit_derivative": float(rhat @ contact_derivative / len(rhat)),
                "jacobian_direction_relative_difference": float(
                    np.linalg.norm(derivative - prediction) / max(np.linalg.norm(derivative), 1e-30)
                ),
            }
        )
    for name, before in history.items():
        np.testing.assert_array_equal(getattr(solver._contact.committed, name).numpy(), before)
    np.savez_compressed(output / "terminal-input.npz", **arrays)
    result = {
        "step": case.step_count + 1,
        "scope": "terminal counterfactual; no accepted step",
        "lambda": solver._linear._lambda,
        "zero": dataclasses.asdict(zero_result),
        "same": dataclasses.asdict(same),
        "zero_current_residual_difference_scaled_rms": float(
            np.linalg.norm(scale * (zero_residual - residual)) / np.sqrt(len(residual))
        ),
        "same_current_residual_difference_scaled_rms": float(
            np.linalg.norm(scale * (same_residual - residual)) / np.sqrt(len(residual))
        ),
        "predicted_half_squared_merit_derivative": float(rhat @ predicted / len(rhat)),
        "sweep": rows,
        "derivatives": derivatives,
        "committed_history_unchanged": True,
    }
    (output / "terminal-probe.json").write_text(json.dumps(result, indent=2) + "\n")
