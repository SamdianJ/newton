# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure repeated residual noise at a real converged released-beam state."""

import argparse
import itertools
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle
from scripts.monolithic_reference.verify_p3_tet import MASSES, MATERIALS, save, solver_for


def measure(device, material, mass):
    """Observe the accepted state before publication without weakening any stopping test."""
    model = build_model(device, cantilever=True)
    solver = solver_for(model, material, mass, "ArmijoOrResNorm")
    state = solver._create_tet_state()
    x = state.particle_q.numpy().copy()
    x[:, 0] += 0.04 * np.sin(np.pi * x[:, 0] / 2.4)
    state.particle_q.assign(x)
    original_finish = solver._finish_step
    samples = []

    def observe(*args):
        for _ in range(32):
            samples.append(solver._evaluate_trial(solver._transaction.accepted, 0.001).merit)
        return original_finish(*args)

    with patch.object(solver, "_finish_step", side_effect=observe):
        solver.step(state, state, None, None, 0.001)
    noise = float(2 * np.ptp(samples))
    goal = (
        solver._config.merit_absolute_global + solver._config.merit_relative_global * solver.last_stats.merit_reference
    )
    original = solver._transaction._original
    oracle = TetOracle(model, solver._tet_physics)
    _energy, reference_residual, _hessian = oracle.evaluate(
        state.particle_q.numpy(),
        original.particle_q.numpy(),
        original.particle_qd.numpy(),
        [0, 0, 0],
        solver._transaction.inputs.particle_f.numpy(),
        0.001,
    )
    scale = solver._linear.scale.numpy().astype(float)
    production_residual = solver._transaction.accepted.residual.numpy().astype(float)
    oracle_rms = float(np.linalg.norm(scale * reference_residual) / np.sqrt(len(scale)))
    independent_error = float(np.linalg.norm(scale * (production_residual - reference_residual)) / np.sqrt(len(scale)))
    return {
        "device": device,
        "material": material,
        "mass": mass,
        "samples": samples,
        "observed_noise": noise,
        "goal": goal,
        "noise_goal_ratio": noise / goal,
        "oracle_rms": oracle_rms,
        "independent_error_rms": independent_error,
        "independent_error_gate": 0.1 * goal,
        "passed": solver.last_stats.converged
        and noise < goal
        and max(samples) <= goal
        and oracle_rms <= goal
        and independent_error <= 0.1 * goal,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [measure(*case) for case in itertools.product(("cpu", "cuda:0"), MATERIALS, MASSES)]
    passed = all(r["passed"] for r in records)
    save(args.output, {"passed": passed, "records": records})
    if not passed:
        raise SystemExit(1)
