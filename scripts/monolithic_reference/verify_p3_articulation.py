# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen PR2 small-system same-discrete reference matrix."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from newton.solvers.experimental.monolithic import SolverMonolithic

from .p3_articulation_fixtures import FIXTURE_PATH, build_case, initialize, specification, target
from .p3_articulation_oracle import DiscreteOracle


def run_case(case, device, mode, *, h, pcg_mode, graph, control_period=None):
    substeps = 1 if control_period is None else round(control_period / h)
    if control_period is not None and (
        substeps < 1
        or not np.isclose(substeps * h, control_period, rtol=0, atol=1e-15)
        or round(case["duration"] / h) % substeps
    ):
        raise ValueError("Control period must align with the step grid and duration")
    model, terms = build_case(case, device)
    solver = SolverMonolithic._create_articulation_only(
        model, globalization=mode, joint_terms=terms, pcg_mode=pcg_mode, use_cuda_graph=graph
    )
    state = initialize(solver, case, dt=h)
    out = solver._create_articulation_state(dt=h)
    control = model.control()
    oracle = DiscreteOracle(model, solver._joint_terms.params.numpy(), state.joint_q.numpy(), state.joint_qd.numpy(), h)
    force = (
        state.body_f.numpy()[:, :3].astype(float)
        + model.body_mass.numpy().astype(float)[:, None] * model.gravity.numpy().astype(float)[model.body_world.numpy()]
    )
    trace = []
    maxima = {"q": 0.0, "qd": 0.0, "com": 0.0, "rotation": 0.0}
    failure = None
    failure_state = None
    outer_input = None
    for step in range(round(case["duration"] / h)):
        if control_period is None or step % substeps == 0:
            sample_time = (step + 1) * h if control_period is None else (step // substeps) * control_period
            qt, vt = target(case, sample_time)
            control.joint_target_q.assign(qt.astype(np.float32))
            control.joint_target_qd.assign(vt.astype(np.float32))
            if control_period is not None:
                outer_input = solver._states.copy(state)
        try:
            expected = oracle.step(
                h,
                control.joint_target_q.numpy().astype(float),
                control.joint_target_qd.numpy().astype(float),
                force,
                control.joint_f.numpy().astype(float),
            )
        except RuntimeError as error:
            failure = str(error)
            break
        start = time.perf_counter()
        solver.step(state, out, control, dt=h)
        step_ms = 1000 * (time.perf_counter() - start)
        stats = solver.last_stats
        q, v = out.joint_q.numpy().copy(), out.joint_qd.numpy().copy()
        solver._kinematics.evaluate(out.joint_q)
        t = solver._kinematics.com.numpy().copy()
        r = solver._kinematics.rotations.numpy().copy()
        rel = r @ expected["rotation"].transpose(0, 2, 1)
        # atan2 remains accurate near zero; acos(trace) loses small-angle information.
        sine = (
            np.linalg.norm(
                np.stack(
                    (rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]), axis=1
                ),
                axis=1,
            )
            / 2
        )
        cosine = (np.trace(rel, axis1=1, axis2=2) - 1) / 2
        error = {
            "q": float(abs(q - expected["q"]).max()),
            "qd": float(abs(v - expected["qd"]).max()),
            "com": float(np.linalg.norm(t - expected["com"], axis=1).max()),
            "rotation": float(np.arctan2(sine, cosine).max()),
        }
        for key, value in error.items():
            maxima[key] = max(maxima[key], value)
        trace.append(
            {
                "step": step + 1,
                "time": (step + 1) * h,
                "control_time": sample_time,
                "q_target": control.joint_target_q.numpy().tolist(),
                "qd_target": control.joint_target_qd.numpy().tolist(),
                "outer_committed": control_period is None
                and stats.converged
                and stats.status == solver.Status.SUCCESS
                and not stats.rolled_back,
                "status": stats.status.value,
                "q": q.tolist(),
                "qd": v.tolist(),
                "com": t.tolist(),
                "rotation": r.tolist(),
                "oracle_q": expected["q"].tolist(),
                "oracle_qd": expected["qd"].tolist(),
                "error": error,
                "residual_rms": stats.merit_q_final,
                "goal_global": solver._config.merit_absolute_global
                + solver._config.merit_relative_global * stats.merit_reference,
                "goal_q": solver._config.merit_absolute_q + solver._config.merit_relative_q * stats.merit_reference,
                "convergence_ratio_q": stats.convergence_ratio_q,
                "iterations": stats.nonlinear_iterations,
                "linear_iterations": stats.linear_iterations,
                "linear_scheduled_slots": stats.linear_scheduled_slots,
                "step_ms": step_ms,
                "converged": stats.converged,
                "rolled_back": stats.rolled_back,
                "trials": stats.variational["trials"],
                "linear_postchecks": stats.variational["linear_postchecks"],
            }
        )
        if not stats.converged or stats.status != solver.Status.SUCCESS or stats.rolled_back:
            failure = stats.failure_reason or stats.status.value
            break
        state, out = out, state
        if control_period is not None and (step + 1) % substeps == 0:
            for row in trace[-substeps:]:
                row["outer_committed"] = True
    rotational = all(row["type"] == "REVOLUTE" for row in case["joints"])
    gates = specification()["same_discrete_oracle_max_errors"]
    limits = {
        "q": gates["q_rad" if rotational else "q_m"],
        "qd": gates["qd_rad_s" if rotational else "qd_m_s"],
        "com": gates["link_COM_m"],
        "rotation": gates["link_rotation_rad"],
    }
    if failure is not None:
        failure_state = {name: value.numpy().tolist() for name, value in vars(state).items() if hasattr(value, "numpy")}
        failure_state["history"] = {
            name: getattr(state._monolithic_articulation, name).numpy().tolist() for name in solver._states.schema
        }
        failure_state["control"] = {
            name: getattr(control, name).numpy().tolist() for name in ("joint_target_q", "joint_target_qd", "joint_f")
        }
        failure_state["oracle"] = {
            name: getattr(oracle, name).tolist()
            for name in ("q", "v", "t", "r", "linear_velocity", "rotation_velocity")
        }
        failure_state["failed_step"] = step + 1
        if outer_input is not None:
            state.assign(outer_input)
            failure_state["outer_rollback"] = {
                "q": state.joint_q.numpy().tolist(),
                "qd": state.joint_qd.numpy().tolist(),
                "history": {
                    name: getattr(state._monolithic_articulation, name).numpy().tolist()
                    for name in solver._states.schema
                },
            }
    passed = (
        failure is None
        and len(trace) == round(case["duration"] / h)
        and all(maxima[name] <= value for name, value in limits.items())
    )
    return {
        "case": case["id"],
        "device": device,
        "globalization": mode,
        "dt": h,
        "control_period": control_period,
        "substeps_per_control": substeps,
        "control_sampling": "step_end" if control_period is None else "outer_start_float32_zero_order_hold",
        "committed_outer_steps": sum(row["outer_committed"] for row in trace) // substeps,
        "pcg_mode": pcg_mode,
        "stopping_config": asdict(solver._config),
        "graph": graph,
        "passed_same_discrete": passed,
        "failure": failure,
        "failure_state": failure_state,
        "max_errors": maxima,
        "gates": limits,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--modes", nargs="+", default=["Armijo", "ResNorm", "ArmijoOrResNorm"])
    parser.add_argument("--dt", type=float, choices=(0.001, 0.0005, 0.00025, 0.000125), default=0.001)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--control-period", type=float, choices=(0.001,))
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"))
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for device in args.devices:
        pcg_mode = args.pcg_mode or ("diagnostic" if device == "cpu" else "production")
        graph = args.cuda_graph if args.cuda_graph is not None else device != "cpu" and pcg_mode == "production"
        for mode in args.modes:
            for case in specification()["cases"]:
                if args.cases and case["id"] not in args.cases:
                    continue
                result = run_case(
                    case,
                    device,
                    mode,
                    h=args.dt,
                    pcg_mode=pcg_mode,
                    graph=graph,
                    control_period=args.control_period,
                )
                name = f"{case['id']}-{device.replace(':', '_')}-{mode}.json"
                path = args.output_dir / name
                path.write_text(json.dumps(result, indent=2) + "\n")
                summary = {key: value for key, value in result.items() if key != "trace"}
                summary.update(
                    trace_file=name,
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    steps=len(result["trace"]),
                )
                results.append(summary)
                print(summary, flush=True)
                (args.output_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "schema": "p3-pr2-small-discrete/v1",
                            "fixture_sha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
                            "results": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
