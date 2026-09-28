# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record bounded T4 trajectories; failed steps never contribute to passing statistics."""

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

from scripts.monolithic_reference.p3_coupled_fixtures import (
    DIRECTORY,
    NORMAL,
    PROTOCOL,
    make_solver,
    make_tiny,
    set_control,
)
from scripts.monolithic_reference.p3_coupled_oracle import CoupledOracle


def serial(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "items"):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def state_arrays(state):
    arrays = {name: value.numpy().copy() for name, value in vars(state).items() if isinstance(value, wp.array)}
    arrays.update(
        {
            "history." + name: value.numpy().copy()
            for name, value in vars(state._monolithic_coupled).items()
            if isinstance(value, wp.array)
        }
    )
    return arrays


def measure(run, state):
    contact = run._contact
    k = run._kinematics
    gaps = contact.current.distance.numpy().copy()
    shape_ids = contact.shapes.numpy()
    if len(gaps):
        gaps -= contact.margins.numpy()[shape_ids[np.arange(len(gaps)) % len(shape_ids)]] + contact.parameters[4]
    local = contact.local_residual.numpy()
    normal = contact.normal.numpy().copy()
    slide = contact.current.y.numpy() - contact.start.y.numpy()
    slip = slide - np.einsum("ni,ni->n", normal, slide)[:, None] * normal
    friction = contact.weights.numpy()[:, 1, None] * slip
    gradient = contact.current.gradient.numpy()
    normal_load = -np.einsum("ni,ni->n", local - friction, gradient) / np.maximum(np.sum(gradient**2, axis=1), 1e-300)
    physical = state._monolithic_coupled
    pforce = physical.contact_particle_force.numpy()
    bforce = physical.contact_body_force.numpy()
    return {
        "penetration": max(0.0, float(-gaps.min())) if len(gaps) else 0.0,
        "contact_load": float(normal_load.sum()),
        "friction_work": float(np.sum(friction * slide)),
        "slip": float(np.linalg.norm(slip, axis=1).max()) if len(slip) else 0.0,
        "force_balance": float(
            np.linalg.norm(pforce.sum(axis=0) + bforce[:, :3].sum(axis=0))
            / max(
                np.linalg.norm(pforce, axis=1).sum() + np.linalg.norm(bforce[:, :3], axis=1).sum(),
                NORMAL["acceptance"]["force_floor_n"],
            )
        ),
        "body_contact_force": bforce,
        "q": state.joint_q.numpy().copy(),
        "qd": state.joint_qd.numpy().copy(),
        "x": state.particle_q.numpy().copy(),
        "v": state.particle_qd.numpy().copy(),
        "com": k.com.numpy().copy(),
        "rotation": k.rotations.numpy().copy(),
    }


def assess(rows, case, expected, load):
    complete = len(rows) == expected and all(row["success"] for row in rows)
    successful = [r for r in rows if r["success"]]
    result = {
        "complete": complete,
        "successful_steps": len(successful),
        "attempted_steps": len(rows),
        "expected_steps": expected,
        "physical": "NOT_CERTIFIED_INCOMPLETE",
    }
    if not complete:
        return result
    maximum = max(r["penetration"] for r in rows)
    result.update(maximum_penetration=maximum, minimum_det_f=min(r["min_det_f"] for r in rows))
    result["friction_work"] = sum(r["friction_work"] for r in rows)
    if case.startswith("cube"):
        tail = rows[-round(0.2 / rows[0]["time"]) :]
        force = np.mean([r["body_contact_force"][:, :3][:, 2] for r in tail], axis=0)
        error = float(np.max(abs(abs(force) - load)) / load)
        result.update(
            force_error=error,
            physical="PASS" if maximum <= 0.002 and error <= 0.05 and result["minimum_det_f"] >= 0.2 else "FAIL",
        )
    else:
        active = np.array([r["contact_load"] > NORMAL["acceptance"]["force_floor_n"] for r in rows])
        result["signals"] = {
            "loaded": bool(active.any()),
            "sliding": any(r["slip"] > 1e-10 and a for r, a in zip(rows, active, strict=True)),
            "released": bool(active.any() and not active[-1]),
        }
        if case == "normal_loading":
            tail = rows[-round(0.1 / rows[0]["time"]) :]
            rest = np.asarray(NORMAL["soft"]["rest_positions_m"])
            thickness = np.linalg.norm(rest[3] - rest[:3].mean(axis=0))
            response = min(thickness - np.linalg.norm(r["x"][3] - r["x"][:3].mean(axis=0)) for r in tail)
            free = [r for r in rows if r["time"] <= 0.2 + 1e-12]
            completion = float(free[-1]["q"][0] / 0.003)
            gates = {
                "penetration": max(r["penetration"] for r in tail) <= 0.001,
                "soft_response": response >= 0.005,
                "free_space_completion": completion >= 0.95,
                "free_space_no_force": all(not active[i] for i, r in enumerate(rows) if r["time"] <= 0.2),
                "contact": bool(active.any()),
                "force_balance": max(r["force_balance"] for r in rows) <= 1e-4,
            }
            result.update(
                physical="PASS" if all(gates.values()) else "FAIL",
                gates=gates,
                soft_response=response,
                free_space_completion=completion,
            )
        else:
            if case == "tangent_reversal":
                velocities = [r["qd"][1] for r in rows if r["time"] >= 0.4]
                result["signals"]["reversed"] = min(velocities) < -1e-6 and max(velocities) > 1e-6
            needed = (
                ("loaded", "sliding", "released") if case == "curved_release" else ("loaded", "sliding", "reversed")
            )
            result["signal_qualified"] = all(result["signals"][key] for key in needed)
            result["physical"] = "SIGNALS_ONLY"
    return result


def run_case(
    case,
    law,
    device,
    output,
    *,
    dt=0.001,
    mode="diagnostic",
    graph=False,
    reference=False,
    goal_scale=0.1,
    globalization="ResNorm",
    steps=None,
):
    output.mkdir(parents=True, exist_ok=False)
    load = None
    if case.startswith("cube"):
        from scripts.monolithic_reference.calibrate_sharpa_contact import CONFIG, make_case  # noqa: PLC0415

        refinement = int(case[-1])
        asset = DIRECTORY / "soft_cube_40mm" / f"cube_r{refinement}.npz"
        model, old_solver, _, _, load = make_case(asset, 1e7, device, cube=True)
        del old_solver
        duration = 1.5
    else:
        model = make_tiny(case, device)
        duration = PROTOCOL["tiny_fixtures"][case]["duration"]
    solver = make_solver(model, case, law, mode=mode, graph=graph, globalization=globalization)
    state = solver.create_state(dt=dt)
    control = model.control()
    expected = round(duration / dt)
    count = expected if steps is None else min(steps, expected)
    configuration = {
        "case": case,
        "law": law,
        "device": device,
        "dt": dt,
        "mode": mode,
        "graph": graph,
        "reference": reference,
        "goal_scale": goal_scale if reference else None,
        "globalization": globalization,
        "expected_steps": expected,
        "requested_steps": count,
        "actual_config": solver._config,
        "configuration_digest": solver._kinematics.digest.hex(),
        "physical_protocol": PROTOCOL,
        "python": platform.python_version(),
        "warp": wp.__version__,
        "command": __import__("sys").argv,
    }
    root = Path(__file__).resolve().parents[2]
    sources = sorted(
        set(root.glob("newton/**/*.py"))
        | set(root.glob("scripts/monolithic_reference/**/*.py"))
        | set(DIRECTORY.glob("*.json"))
    )
    if case.startswith("cube"):
        sources.append(asset)
    save(
        output / "source-hashes.json",
        {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    save(output / "configuration.json", configuration)
    np.savez_compressed(output / "initial.npz", **state_arrays(state))
    np.savez_compressed(
        output / "model.npz", **{name: v.numpy() for name, v in vars(model).items() if isinstance(v, wp.array)}
    )
    oracle = (
        CoupledOracle(
            model,
            solver._tet.physics,
            solver._objective.params.numpy(),
            law=law,
            mu=0.0 if case == "normal_loading" else 0.5,
            width=0.0,
            goal_scale=goal_scale,
        )
        if reference
        else None
    )
    rows = []
    failure = None
    calls = []
    solve = solver._linear.solve_pcg

    def measured(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append(
            {"iterations": result.iterations, "status": result.status, "scheduled_slots": result.scheduled_slots}
        )
        return result

    solver._linear.solve_pcg = measured
    started = time.perf_counter()
    with (output / "steps.jsonl").open("w") as trace:
        for step in range(count):
            t = (step + 1) * dt
            calls.clear()
            row = {"step": step + 1, "time": t, "success": False}
            begin = time.perf_counter()
            try:
                if case.startswith("cube"):
                    u = min(t / 0.5, 1.0)
                    target = load * u * u * (3 - 2 * u)
                    control.joint_target_qd.assign([-target / CONFIG["plate_velocity_gain"]])
                    row["target_load"] = target
                else:
                    q, v = (oracle.q, oracle.v) if reference else (state.joint_q.numpy(), state.joint_qd.numpy())
                    force, targets = set_control(control, case, t, q, v)
                    row.update(force=force, targets=targets)
                if reference:
                    info = oracle.step(dt, force)
                    row.update(
                        info,
                        q=oracle.q.copy(),
                        qd=oracle.v.copy(),
                        x=oracle.x.copy(),
                        v=oracle.xv.copy(),
                        com=oracle.t.copy(),
                        rotation=oracle.r.copy(),
                        contact_load=info["normal_load"],
                        success=True,
                    )
                else:
                    solver.step(state, state, control, dt=dt)
                    stats = solver.last_stats
                    row.update(
                        stats=serial(stats), pcg_calls=list(calls), min_det_f=stats.min_det_f, success=stats.converged
                    )
                    if stats.converged:
                        row.update(measure(solver, state))
                    else:
                        failure = stats.failure_reason or stats.status.value
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                row["exception"] = failure
            row["seconds"] = time.perf_counter() - begin
            rows.append(row)
            trace.write(json.dumps(serial(row), allow_nan=False) + "\n")
            trace.flush()
            if not row["success"]:
                if not reference:
                    for label in ("_original", "_accepted", "_trial"):
                        np.savez_compressed(
                            output / (label[1:] + "-failure.npz"), **state_arrays(getattr(solver, label))
                        )
                    buffers = {}
                    for label, owner in (("stage", solver._objective), ("linear", solver._linear), ("solver", solver)):
                        buffers.update(
                            {
                                label + "." + key: value.numpy().copy()
                                for key, value in vars(owner).items()
                                if isinstance(value, wp.array)
                            }
                        )
                    np.savez_compressed(output / "failure-buffers.npz", **buffers)
                break
            if (step + 1) % 100 == 0:
                print(case, law, device, "oracle" if reference else mode, dt, step + 1, flush=True)
    summary = assess(rows, case, expected, load)
    costs = [c["iterations"] for row in rows for c in row.get("pcg_calls", [])]
    summary["pcg_cost"] = {
        "calls": len(costs),
        "p95": float(np.percentile(costs, 95)) if costs else None,
        "maximum": max(costs, default=None),
    }
    summary.update(
        case=case,
        law=law,
        reference=reference,
        goal_scale=goal_scale if reference else None,
        dt=dt,
        device=device,
        mode=mode,
        graph=graph,
        failure=failure,
        elapsed_seconds=time.perf_counter() - started,
    )
    if reference:
        np.savez_compressed(
            output / "final.npz",
            q=oracle.q,
            qd=oracle.v,
            x=oracle.x,
            v=oracle.xv,
            com=oracle.t,
            rotation=oracle.r,
            linear_velocity=oracle.tv,
            rotation_velocity=oracle.rv,
        )
    else:
        np.savez_compressed(output / "final.npz", **state_arrays(state))
    if rows:
        successful = [r for r in rows if r["success"]]
        np.savez_compressed(
            output / "trajectory.npz",
            **{k: np.array([r[k] for r in successful]) for k in ("time", "q", "qd", "x", "v", "com", "rotation")},
        )
    summary["configuration_sha256"] = hashlib.sha256((output / "configuration.json").read_bytes()).hexdigest()
    save(output / "summary.json", summary)
    print(json.dumps(serial(summary)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["normal_loading", "tangent_reversal", "curved_release", "cube1", "cube2", "cube3"],
        required=True,
    )
    parser.add_argument("--law", choices=["C-EE", "C-IE", "C-II"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["diagnostic", "production"], default="diagnostic")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--goal-scale", type=float, choices=[0.1, 0.01], default=0.1)
    parser.add_argument("--dt", type=float, choices=[0.001, 0.0005, 0.00025], default=0.001)
    parser.add_argument("--globalization", choices=["ResNorm", "Armijo", "ArmijoOrResNorm"], default="ResNorm")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference and args.case.startswith("cube"):
        parser.error("Dense independent reference is limited to tiny systems")
    result = run_case(
        args.case,
        args.law,
        args.device,
        args.output,
        dt=args.dt,
        mode=args.mode,
        graph=args.graph,
        reference=args.reference,
        goal_scale=args.goal_scale,
        globalization=args.globalization,
        steps=args.steps,
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
