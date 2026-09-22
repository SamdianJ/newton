# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose the frozen free-rotation fixture against its analytic discrete root.

This is a stopping-gate diagnostic, not formal calibrated PR2 acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton


def run(device, mode):
    builder = newton.ModelBuilder()
    builder.gravity = (0.0, 0.0, 0.0)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01))
    joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, limit_ke=0.0, limit_kd=0.0)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverMonolithic._create_articulation_only(
        model,
        globalization=mode,
        pcg_mode="production" if device != "cpu" else "diagnostic",
        use_cuda_graph=device != "cpu",
    )
    state = solver._create_articulation_state(dt=0.001)
    state.joint_q.assign([0.5])
    state.joint_qd.assign([1.0])
    solver._reset_articulation_state(state, dt=0.001)
    out = solver._create_articulation_state(dt=0.001)
    control = model.control()
    h = 0.001
    exact_q, delta = 0.5, np.arcsin(h)
    records = []
    max_q = max_v = 0.0
    for step in range(1000):
        exact_q += delta
        exact_v = delta / h
        solver.step(state, out, control, dt=h)
        stats = solver.last_stats
        q, v = float(out.joint_q.numpy()[0]), float(out.joint_qd.numpy()[0])
        max_q, max_v = max(max_q, abs(q - exact_q)), max(max_v, abs(v - exact_v))
        records.append(
            {
                "step": step + 1,
                "status": stats.status.value,
                "q": q,
                "qd": v,
                "oracle_q": exact_q,
                "oracle_qd": exact_v,
                "residual_rms": stats.merit_q_final,
                "convergence_ratio_q": stats.convergence_ratio_q,
                "iterations": stats.nonlinear_iterations,
            }
        )
        if not stats.converged:
            break
        state, out = out, state
        delta = np.arctan2(np.sin(delta), 2 - np.cos(delta))
    gates = json.loads((Path(__file__).parent / "fixtures/p3_pr2_v3.json").read_text())[
        "same_discrete_oracle_max_errors"
    ]
    return {
        "device": device,
        "mode": mode,
        "dt": h,
        "steps": len(records),
        "max_q_error": max_q,
        "q_gate": gates["q_rad"],
        "max_qd_error": max_v,
        "qd_gate": gates["qd_rad_s"],
        "all_success": all(row["status"] == "success" for row in records),
        "passed_discrete_reference": bool(max_q <= gates["q_rad"] and max_v <= gates["qd_rad_s"]),
        "trace": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    args = parser.parse_args()
    records = []
    for device in args.devices:
        for mode in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            result = run(device, mode)
            print({key: value for key, value in result.items() if key != "trace"}, flush=True)
            records.append(result)
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__),
        *sorted((root / "newton/_src/solvers/monolithic").glob("articulation*.py")),
        root / "newton/_src/solvers/monolithic/solver_monolithic.py",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "p3-pr2-stopping-diagnostic/v1",
                "formal_acceptance": False,
                "source_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
                },
                "results": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
