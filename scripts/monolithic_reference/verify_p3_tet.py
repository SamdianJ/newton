# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run frozen PR1 calibration and independent trajectories, preserving failed jobs."""

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton.examples.softbody.monolithic_tet_compare import load_fraction, reference_matrices
from scripts.monolithic_reference.p3_tet_fixtures import build_model

MATERIALS = ("kim_stable_no_log", "smith_log_stabilized")
MASSES = ("lumped", "consistent")
MODES = ("Armijo", "ResNorm", "ArmijoOrResNorm")


def clean(value):
    """Serialize unavailable numeric fields as JSON null rather than nonstandard NaN."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def save(path, payload):
    """Write one complete result without discarding earlier failed jobs."""
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def solver_for(model, material, mass, mode, *, execution="diagnostic", graph=False):
    """Use the strict private solver with fixed physical and numerical parameters."""
    return SolverMonolithic._create_tet_only(
        model,
        globalization=mode,
        material_model=material,
        mass_mode=mass,
        tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device)
        if mass == "consistent"
        else None,
        pcg_mode=execution,
        use_cuda_graph=graph,
    )


def calibrate(device, material, mass):
    """Measure energy, residual and slope independently on 32 identical evaluations."""
    records = []
    for cantilever, fixed, dt in itertools.product((False, True), (False, True), (0.001, 0.0005, 0.00025)):
        if cantilever and fixed:
            continue
        model = build_model(device, cantilever=cantilever, fixed=fixed, gravity=(1, -2, -9.81))
        solver = solver_for(model, material, mass, "ArmijoOrResNorm")
        state = solver._create_tet_state()
        solver._transaction.begin(state, solver._create_tet_state(), None, dt)
        solver._metrics = {"nonlinear_iterations": 0, "matrix_assembly_count": 0}
        current = solver._transaction.accepted
        rest = state.particle_q.numpy()
        ids = solver._layout.dynamic_particle_ids.numpy()
        direction = np.linspace(-0.2, 0.3, current.z.size).astype(np.float32)
        delta = wp.array(direction, dtype=float, device=device)
        for shape in ("rest", "stretch", "shear"):
            f = np.eye(3)
            if shape == "stretch":
                f = np.diag([1.12, 0.91, 1.06])
            if shape == "shear":
                f[0, 1] = 0.2
            current.z.assign((rest @ f.T)[ids].astype(np.float32).ravel())
            current._recover(dt)
            solver._evaluate_current(current, dt)
            samples = []
            bounds = []
            for _ in range(32):
                result = solver._evaluate_trial(current, dt)
                slope, absolute_slope = solver._tet_objective.slope(current.residual, delta)
                samples.append([sum(result.objective), result.merit, slope])
                bounds.append(
                    [
                        64 * np.finfo(float).eps * result.objective_abs_sum,
                        max(solver._config.merit_noise, 64 * np.finfo(np.float32).eps * result.merit),
                        32 * np.finfo(np.float32).eps * absolute_slope,
                    ]
                )
            samples, bounds = np.asarray(samples), np.asarray(bounds)
            noise = 2 * np.ptp(samples, axis=0)
            passed = bool(np.all(noise <= np.min(bounds, axis=0)))
            records.append(
                {
                    "cantilever": cantilever,
                    "fixed": fixed,
                    "dt": dt,
                    "shape": shape,
                    "samples": samples,
                    "observed_noise": noise,
                    "runtime_bound": np.min(bounds, axis=0),
                    "passed": passed,
                }
            )
    return {"passed": all(r["passed"] for r in records), "records": records}


def trajectory(device, material, mass, mode, scene, dt, execution, graph):
    """Stop a physical trajectory at its first failed step and retain the full diagnostic."""
    beam = scene in ("axial", "transverse", "release")
    gravity = (0, 0, -9.81) if scene == "fall" else (0, 0, 0)
    model = build_model(device, cantilever=beam, gravity=gravity)
    solver = solver_for(model, material, mass, mode, execution=execution, graph=graph)
    state = solver._create_tet_state()
    rest = state.particle_q.numpy().copy()
    fixed = model.particle_inv_mass.numpy() == 0
    duration = 4.0 if scene in ("axial", "transverse") else 1.0 if scene == "release" else 0.1
    weights = np.zeros(model.particle_count)
    if beam:
        for face in model.tri_indices.numpy():
            x = rest[face]
            if np.all(np.abs(x[:, 0] - 1.2) < 1e-7):
                area = np.linalg.norm(np.cross(x[1] - x[0], x[2] - x[0])) / 2
                np.add.at(weights, face, area / 3)
        weights /= weights.sum()
    if scene == "release":
        initial = rest.copy()
        initial[:, 0] += 0.04 * np.sin(np.pi * rest[:, 0] / 2.4)
        state.particle_q.assign(initial)
    if scene == "translate":
        state.particle_qd.fill_(wp.vec3d(0.1, 0, 0))
    free = np.flatnonzero(np.repeat(~fixed, 3))
    stiffness, matrix = reference_matrices(model, mass, density=1000)
    m, k = matrix[np.ix_(free, free)], stiffness[np.ix_(free, free)]
    inverse = np.linalg.inv(m + dt**2 * k)
    linear_x = (state.particle_q.numpy() - rest).ravel()[free].astype(float)
    linear_v = state.particle_qd.numpy().ravel()[free].astype(float)
    records, failure, states = [], None, []
    for i in range(round(duration / dt)):
        time_s = (i + 1) * dt
        force = np.zeros_like(rest)
        if scene in ("axial", "transverse"):
            force[:, 0 if scene == "axial" else 2] = weights * (10 if scene == "axial" else 0.5) * load_fraction(time_s)
        state.particle_f.assign(force)
        before_x, before_v = state.particle_q.numpy().copy(), state.particle_qd.numpy().copy()
        try:
            solver.step(state, state, None, None, dt)
            stats = solver.last_stats
            if not stats.converged:
                failure = {
                    "step": i,
                    "time": time_s,
                    "status": stats.status.value,
                    "reason": stats.failure_reason,
                    "ratio": stats.convergence_ratio,
                    "ratio_x": stats.convergence_ratio_x,
                    "diagnostics": dict(stats.variational),
                    "input_x": before_x,
                    "input_v": before_v,
                    "accepted_x": solver._transaction.accepted.state.particle_q.numpy(),
                    "forces": force,
                }
                break
            x, v = state.particle_q.numpy(), state.particle_qd.numpy()
            if not np.array_equal(x[fixed], rest[fixed]) or np.any(v[fixed] != 0):
                raise AssertionError("Fixed node moved")
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                raise AssertionError("Nonfinite returned state")
            if max(stats.convergence_ratio, stats.convergence_ratio_x) > 1 or stats.min_det_f < 0.2:
                raise AssertionError("Physical step bypassed its numerical gate")
            load = (force.astype(float) + (matrix @ np.tile(gravity, model.particle_count)).reshape(-1, 3)).ravel()[
                free
            ]
            next_linear = inverse @ (m @ (linear_x + dt * linear_v) + dt**2 * load)
            linear_v = (next_linear - linear_x) / dt
            linear_x = next_linear
            reference = np.zeros(rest.size)
            reference[free] = linear_x
            if (i + 1) % max(1, round(0.001 / dt)) == 0:
                states.append(x.tolist())
                records.append(
                    {
                        "time": time_s,
                        "tip": weights @ (x - rest),
                        "linear_tip": weights @ reference.reshape(-1, 3),
                        "ratio": stats.convergence_ratio,
                        "ratio_x": stats.convergence_ratio_x,
                        "det_f": stats.min_det_f,
                        "newton": stats.nonlinear_iterations,
                        "pcg": stats.linear_iterations,
                    }
                )
        except Exception:
            failure = {
                "step": i,
                "traceback": traceback.format_exc(),
                "input_x": before_x,
                "input_v": before_v,
                "forces": force,
            }
            break
    gates = {}
    if failure is None:
        tips = np.asarray([r["tip"] for r in records])
        refs = np.asarray([r["linear_tip"] for r in records])
        if scene in ("axial", "transverse"):
            gates["response"] = bool(np.max(np.abs(tips)) >= 0.002)
            gates["linear_error"] = float(np.max(np.abs(tips - refs)) / max(np.max(np.abs(refs)), 1e-30))
            gates["linear_match"] = gates["linear_error"] <= 0.03
        if scene == "release":
            gates["response"] = bool(np.max(np.abs(state.particle_q.numpy() - initial)) > 0.001)
        if scene in ("fall", "translate", "stationary"):
            steps = round(duration / dt)
            exact = rest + (np.array([0.1, 0, 0]) * duration if scene == "translate" else 0)
            if scene == "fall":
                exact = exact + np.array(gravity) * dt**2 * steps * (steps + 1) / 2
            gates["analytic_error"] = float(np.max(np.abs(state.particle_q.numpy() - exact)))
            gates["analytic_match"] = gates["analytic_error"] <= 5e-5
    passed = failure is None and all(v for v in gates.values() if isinstance(v, bool))
    return {
        "passed": passed,
        "scene": scene,
        "dt": dt,
        "duration": duration,
        "material": material,
        "mass": mass,
        "mode": mode,
        "device": device,
        "execution": execution,
        "graph": graph,
        "topology_sha256": hashlib.sha256(model.tet_indices.numpy().tobytes() + rest.tobytes()).hexdigest(),
        "tet_config_identity": solver.tet_config_identity,
        "records": records,
        "states": states,
        "gates": gates,
        "failure": failure,
    }


def main():
    """Run calibration or a fixed Cartesian trajectory batch without skipping failed inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "trajectories"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("stationary", "translate", "fall", "axial", "transverse", "release"),
        default=["stationary", "translate", "fall", "axial", "transverse", "release"],
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--dt", type=float, default=0.001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": platform.python_version(),
        "warp": wp.__version__,
        "arguments": vars(args) | {"output": str(args.output)},
    }
    sources = [
        Path("newton/_src/solvers/monolithic") / name
        for name in ("solver_monolithic.py", "tet.py", "tet_variational.py")
    ]
    sources += [
        Path(__file__),
        Path(__file__).with_name("p3_tet_fixtures.py"),
        Path(__file__).with_name("p3_tet_oracle.py"),
    ]
    provenance["sources"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshot = args.output / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    save(args.output / "provenance.json", provenance)
    summaries = []
    for device, material, mass in itertools.product(args.devices, MATERIALS, MASSES):
        jobs = [("calibration", None)] if args.phase == "calibration" else itertools.product(args.scenes, args.modes)
        for scene, mode in jobs:
            key = "-".join(filter(None, (device.replace(":", "_"), material, mass, scene, mode)))
            try:
                result = (
                    calibrate(device, material, mass)
                    if args.phase == "calibration"
                    else trajectory(
                        device,
                        material,
                        mass,
                        mode,
                        scene,
                        args.dt,
                        "diagnostic" if device == "cpu" else "production",
                        device != "cpu",
                    )
                )
            except Exception:
                result = {"passed": False, "traceback": traceback.format_exc()}
            save(args.output / f"{key}.json", result)
            row = {"job": key, "passed": result["passed"], "file": f"{key}.json"}
            summaries.append(row)
            save(args.output / "summary.json", {"passed": all(r["passed"] for r in summaries), "jobs": summaries})
            print(json.dumps(row), flush=True)
    if not all(row["passed"] for row in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
