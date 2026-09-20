# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose physical errors using an independent dense FP64 trajectory, never as a PR1 pass."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import load_fraction
from scripts.monolithic_reference.p3_tet_fixtures import build_model
from scripts.monolithic_reference.p3_tet_oracle import TetOracle
from scripts.monolithic_reference.verify_p3_tet import save, solver_for


def diagnose(source):
    """Keep physical inputs fixed and replace only the diagnostic nonlinear solution accuracy."""
    data = json.loads(source.read_text())
    dt = data["dt"]
    times = np.asarray([r["time"] for r in data["records"]], dtype=float)
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Expected a finite positive physical dt")
    if (
        times.size == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(np.r_[0.0, times]) <= 0)
        or not np.allclose(times / dt, np.rint(times / dt), rtol=0, atol=1e-8)
    ):
        raise ValueError("Expected increasing sample times on the physical dt grid")
    sample_steps = np.rint(times / dt).astype(int)
    if np.any(np.diff(np.r_[0, sample_steps]) <= 0):
        raise ValueError("Expected increasing sample times on the physical dt grid")
    if len(data["states"]) != len(times):
        raise ValueError("Expected one production state per sample time")
    dense = bool(np.all(np.diff(np.r_[0, sample_steps]) == 1))
    beam = data["scene"] == "transverse"
    gravity = [0, 0, 0] if beam else [0, 0, -9.81]
    model = build_model("cpu", cantilever=beam, gravity=gravity)
    solver = solver_for(model, data["material"], data["mass"], data["mode"])
    oracle = TetOracle(model, solver._tet_physics)
    x = model.particle_q.numpy().astype(float)
    rest, v = x.copy(), np.zeros_like(x)
    weights = np.zeros(len(x))
    if beam:
        for face in model.tri_indices.numpy():
            vertices = rest[face]
            if np.all(np.abs(vertices[:, 0] - 1.2) < 1e-7):
                area = np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) / 2
                np.add.at(weights, face, area / 3)
        weights /= weights.sum()
    tips, residuals = [], []
    production = np.asarray(data["states"])
    production_residual_sum, defect_displacement, defect_velocity = [], np.zeros(3), np.zeros(3)
    mass_sum = oracle.mass[::3, ::3].sum()
    step = 0
    for i, target_step in enumerate(sample_steps):
        while step < target_step:
            step += 1
            forces = np.zeros_like(x)
            if beam:
                # Match the actual frozen FP32 force at each physical step, including unsaved steps.
                forces[:, 2] = weights * 0.5 * load_fraction(step * dt)
            forces = forces.astype(np.float32).astype(float)
            xn, vn = x.copy(), v.copy()
            x[oracle.free] += dt * v[oracle.free]
            for _ in range(6):
                _energy, residual, hessian = oracle.evaluate(x, xn, vn, gravity, forces, dt)
                if np.linalg.norm(residual, ord=np.inf) < 1e-9:
                    break
                x[oracle.free] += np.linalg.solve(hessian, -residual).reshape(-1, 3)
            residuals.append(float(np.linalg.norm(oracle.evaluate(x, xn, vn, gravity, forces, dt)[1], ord=np.inf)))
            v = (x - xn) / dt
        tips.append(weights @ (x - rest))
        if not beam and dense:
            previous = rest if i == 0 else production[i - 1]
            previous_velocity = (
                np.zeros_like(x) if i == 0 else (previous - (rest if i == 1 else production[i - 2])) / dt
            )
            r = oracle.evaluate(production[i], previous, previous_velocity, gravity, forces, dt)[1]
            summed = r.reshape(-1, 3).sum(axis=0)
            production_residual_sum.append(summed)
            defect_velocity += dt * summed / mass_sum
            defect_displacement += dt * defect_velocity
    result = {
        "status": "DIAGNOSTIC_ONLY_NOT_PR1_CERTIFICATION",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "independent_residual_inf_max": max(residuals),
        "production_zero_iteration_steps": sum(r["newton"] == 0 for r in data["records"]) if dense else None,
        "steps": step,
        "samples": len(tips),
    }
    if beam:
        linear = np.asarray([r["linear_tip"] for r in data["records"]])
        result["independent_linear_relative_error"] = float(np.max(np.abs(tips - linear)) / np.max(np.abs(linear)))
        result["production_linear_relative_error"] = data["gates"]["linear_error"]
    else:
        n = step
        exact = rest + np.asarray(gravity) * dt**2 * n * (n + 1) / 2
        row_mass = oracle.mass[::3, ::3].sum(axis=1)
        result.update(
            independent_analytic_error=float(np.max(np.abs(x - exact))),
            production_analytic_error=data["gates"]["analytic_error"],
            production_center_of_mass_error=row_mass @ (production[-1] - exact) / mass_sum,
            residual_integrated_center_of_mass_error=defect_displacement if dense else None,
            production_residual_sum=production_residual_sum if dense else None,
        )
        if not dense:
            result["production_residual_unavailable_reason"] = (
                "Intermediate physical-step states are missing; sampled positions cannot reconstruct per-step residuals."
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, diagnose(args.source))
