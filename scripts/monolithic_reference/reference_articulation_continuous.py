# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent continuous RK4 references from Cartesian kinetic-energy derivatives."""

import argparse
import json
from pathlib import Path

import numpy as np

from .articulation_fixtures import build_case, specification, target
from .articulation_oracle import Kinematics


def reference(case, subdivisions, *, output_dt=0.001, control_period=None):
    if type(subdivisions) is not int or subdivisions <= 0:
        raise ValueError("Subdivisions must be a positive integer")
    dt = 0.001 / subdivisions
    if not np.isfinite(output_dt) or output_dt <= 0:
        raise ValueError("Output dt must be finite and positive")
    stride = round(output_dt / dt)
    total = round(case["duration"] / dt)
    if stride < 1 or not np.isclose(stride * dt, output_dt, rtol=0, atol=1e-15) or total % stride:
        raise ValueError("Output grid must align with the integration grid and duration")
    hold_stride = None if control_period is None else round(control_period / dt)
    if hold_stride is not None and (
        hold_stride < 1 or not np.isclose(hold_stride * dt, control_period, rtol=0, atol=1e-15) or total % hold_stride
    ):
        raise ValueError("Control period must align with the integration grid and duration")
    model, _ = build_case(case, "cpu")
    kin = Kinematics(model)
    mass = model.body_mass.numpy().astype(float)
    second = kin.second
    armature = model.joint_armature.numpy().astype(float)
    damping = model.joint_damping.numpy().astype(float)
    kp, kd, effort = (
        getattr(model, name).numpy().astype(float)
        for name in ("joint_target_ke", "joint_target_kd", "joint_effort_limit")
    )
    lower, upper, ke, kl, friction = (
        getattr(model, name).numpy().astype(float)
        for name in ("joint_limit_lower", "joint_limit_upper", "joint_limit_ke", "joint_limit_kd", "joint_friction")
    )
    width = np.array([row.get("limit_width", 1.0) for row in case["joints"]], dtype=np.float32).astype(float)
    eps = np.array([row.get("friction_velocity_scale", 1.0) for row in case["joints"]], dtype=np.float32).astype(float)
    force = (
        np.asarray(case.get("world_COM_force", (0.0, 0.0, 0.0)), dtype=np.float32).astype(float)
        + mass[:, None] * model.gravity.numpy().astype(float)[model.body_world.numpy()]
    )
    n = kin.n
    held_target = None

    def rhs(time, state):
        q, v = state[:n], state[n:]
        _, _, jt, jr, ht, hr = kin.evaluate(q)
        matrix = (
            np.einsum("bni,bmi,b->nm", jt, jt, mass)
            + np.einsum("bnij,bmij->nm", jr @ second[:, None], jr)
            + np.diag(armature)
        )
        convective_t = np.einsum("bnmi,n,m->bi", ht, v, v)
        convective_r = np.einsum("bnmij,n,m->bij", hr, v, v)
        bias = np.einsum("bni,bi,b->n", jt, convective_t, mass) + np.einsum(
            "bnij,bij->n", jr @ second[:, None], convective_r
        )
        qt, vt = target(case, time) if held_target is None else held_target
        joint = damping * v + np.clip(kp * (q - qt) + kd * (v - vt), -effort, effort) + friction * v / np.hypot(v, eps)
        for sign, bound in ((1.0, upper), (-1.0, lower)):
            d = sign * (q - bound)
            z = np.clip(d / width, 0.0, 1.0)
            activation = z * z * (3.0 - 2.0 * z)
            joint += sign * (ke * np.maximum(d, 0.0) + kl * activation * np.maximum(sign * v, 0.0))
        acceleration = np.linalg.solve(matrix, np.einsum("bni,bi->n", jt, force) - bias - joint)
        return np.r_[v, acceleration]

    state = np.array([row["q0"] for row in case["joints"]] + [row["qd0"] for row in case["joints"]], dtype=float)
    states = [state.copy()]
    for step in range(total):
        time = step * dt
        if hold_stride is not None and step % hold_stride == 0:
            held_target = tuple(
                value.astype(np.float32).astype(float) for value in target(case, (step // hold_stride) * control_period)
            )
        a = rhs(time, state)
        b = rhs(time + dt / 2, state + dt * a / 2)
        c = rhs(time + dt / 2, state + dt * b / 2)
        d = rhs(time + dt, state + dt * c)
        state += dt * (a + 2 * b + 2 * c + d) / 6
        if (step + 1) % stride == 0:
            states.append(state.copy())
    return {
        "case": case["id"],
        "dt": dt,
        "output_dt": output_dt,
        "control_period": control_period,
        "control_sampling": "continuous" if control_period is None else "outer_start_float32_zero_order_hold",
        "times": (np.arange(len(states)) * output_dt).tolist(),
        "subdivisions": subdivisions,
        "states": np.asarray(states).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--cases", nargs="+", default=["pendulum", "noncollinear_chain", "driven_terms"])
    parser.add_argument("--output-dt", type=float, default=0.001)
    parser.add_argument("--control-period", type=float, choices=(0.001,))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in specification()["cases"]:
        if case["id"] not in args.cases:
            continue
        for subdivisions in args.subdivisions:
            result = reference(case, subdivisions, output_dt=args.output_dt, control_period=args.control_period)
            (args.output_dir / f"{case['id']}-h{subdivisions}.json").write_text(json.dumps(result) + "\n")
            print(case["id"], subdivisions, result["states"][-1], flush=True)


if __name__ == "__main__":
    main()
