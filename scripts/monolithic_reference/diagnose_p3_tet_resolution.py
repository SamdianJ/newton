# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the FP32 coordinate-resolution diagnostic from preserved PR1 failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle
from scripts.monolithic_reference.verify_p3_tet import save, solver_for


def diagnose(root):
    """Compare continuous FP64 corrections with actual rounded production coordinates."""
    results = []
    for scene, mass in (("release", "consistent"), ("axial", "lumped")):
        source = root / "trajectories-v1" / f"cpu-smith_log_stabilized-{mass}-{scene}-ArmijoOrResNorm.json"
        failure = json.loads(source.read_text())["failure"]
        model = build_model("cpu", cantilever=True)
        solver = solver_for(model, "smith_log_stabilized", mass, "ArmijoOrResNorm")
        state = solver._create_tet_state()
        for name, key in (("particle_q", "input_x"), ("particle_qd", "input_v"), ("particle_f", "forces")):
            getattr(state, name).assign(np.asarray(failure[key], dtype=np.float32))
        solver._transaction.begin(state, solver._create_tet_state(), None, 0.001)
        solver._metrics = {"nonlinear_iterations": 0, "matrix_assembly_count": 0}
        candidate = solver._transaction.accepted
        ids = solver._layout.dynamic_particle_ids.numpy()
        accepted = np.asarray(failure["accepted_x"])
        x = accepted.copy()
        candidate.z.assign(x[ids].astype(np.float32).ravel())
        candidate._recover(0.001)
        current = solver._evaluate_current(candidate, 0.001)
        scale = solver._linear.scale.numpy().astype(float)
        oracle = TetOracle(model, solver._tet_physics)

        def evaluate(value, oracle=oracle, failure=failure):
            return oracle.evaluate(value, failure["input_x"], failure["input_v"], [0, 0, 0], failure["forces"], 0.001)

        def rms(residual, scale=scale):
            return float(np.linalg.norm(scale * residual) / np.sqrt(len(scale)))

        initial_residual = evaluate(x)[1]
        for _ in range(4):
            _parts, residual, hessian = evaluate(x)
            x[ids] += np.linalg.solve(hessian, -residual).reshape(-1, 3)
        continuous = evaluate(x)[1]
        rounded = x.astype(np.float32)
        candidate.z.assign(rounded[ids].ravel())
        candidate._recover(0.001)
        actual = solver._evaluate_current(candidate, 0.001)
        half_ulp = 0.5 * np.abs(np.spacing(rounded[ids])).astype(float)
        correction = x[ids] - accepted[ids]
        results.append(
            {
                "scene": scene,
                "mass": mass,
                "source": str(source),
                "current_merit": current.merit,
                "oracle_current_rms": rms(initial_residual),
                "oracle_continuous_rms": rms(continuous),
                "rounded_production_rms": actual.merit,
                "rounded_oracle_rms": rms(evaluate(rounded)[1]),
                "max_required_correction": float(abs(correction).max()),
                "max_half_ulp": float(half_ulp.max()),
                "coordinate_fraction_below_half_ulp": float(np.mean(abs(correction) <= half_ulp)),
                "same_rounded_coordinates": int(np.count_nonzero(rounded[ids] == accepted[ids])),
                "coordinate_count": int(rounded[ids].size),
                "status": "DIAGNOSTIC_ONLY_NOT_ORACLE_CERTIFICATION",
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, diagnose(args.evidence))
