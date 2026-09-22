# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure PR2 FP64 E/r/slope error against independent matrix derivatives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton._src.solvers.monolithic.articulation import MonolithicJointTermsWorkspace
from newton._src.solvers.monolithic.articulation_state import ArticulationState, _predict
from newton._src.solvers.monolithic.articulation_variational import ArticulationKinematics, ArticulationObjective
from newton._src.solvers.monolithic.segment_graph import _SegmentGraphCache
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture, load_hand

from .p3_articulation_fixtures import FIXTURE_PATH, build_case, specification, target
from .p3_articulation_oracle import Kinematics, joint_terms, skew


def calibrate(model, terms, qn, vn, samples, h, *, graph, qt, vt, body_force):
    old_terms = MonolithicJointTermsWorkspace(
        model, **{name: getattr(terms, name) for name in terms.__dataclass_fields__}
    )
    k = ArticulationKinematics(model)
    states = ArticulationState(k)
    state = states.create(dt=h)
    state.joint_q.assign(qn)
    state.joint_qd.assign(vn)
    states.reset(state, dt=h)
    history = state._monolithic_articulation
    obj = ArticulationObjective(k, old_terms.params.numpy())
    obj.qn.assign(qn)
    obj.vn.assign(vn)
    obj.qt.assign(np.asarray(qt, dtype=np.float32).astype(float))
    obj.vt.assign(np.asarray(vt, dtype=np.float32).astype(float))
    k.evaluate(state.joint_q)
    wp.copy(obj.tn, k.com)
    wp.copy(obj.rn, k.rotations)
    wp.launch(
        _predict,
        k.bodies,
        [
            obj.tn,
            obj.rn,
            history.link_v_com,
            history.link_omega,
            history.link_vsym,
            wp.float64(h),
            obj.predicted_t,
            obj.predicted_r,
        ],
        device=model.device,
    )
    force = (
        np.asarray(body_force, dtype=np.float32).astype(float)
        + model.body_mass.numpy().astype(float)[:, None] * model.gravity.numpy().astype(float)[model.body_world.numpy()]
    )
    obj.force.assign(force)
    oracle = Kinematics(model)
    tn, rn, jt, jr, _, _ = oracle.evaluate(qn)
    velocity = np.einsum("bni,n->bi", jt, vn)
    rd = np.einsum("bnij,n->bij", jr, vn)
    omega_matrix = rd @ rn.transpose(0, 2, 1)
    omega = np.stack((omega_matrix[:, 2, 1], omega_matrix[:, 0, 2], omega_matrix[:, 1, 0]), axis=1)
    sym = np.array([h * skew(w) @ skew(w) / (1 + np.sqrt(1 - h * h * np.dot(w, w))) for w in omega])
    tp = tn + h * velocity
    rp = rn + h * (omega_matrix + sym) @ rn
    q = wp.empty_like(state.joint_q)
    cache = _SegmentGraphCache(model.device, enabled=graph)
    buffers = [q] + [value for owner in (k, obj) for value in vars(owner).values() if isinstance(value, wp.array)]
    maxima = dict.fromkeys(
        ("energy", "residual", "slope", "scaled_residual", "repeat_energy", "repeat_residual", "fk", "jacobian"), 0.0
    )
    records = []
    direction = np.ones(k.count) / np.sqrt(k.count)
    for sample in samples:
        q.assign(sample)
        eo, ro, _ = oracle.inertia(sample, tp, rp, h, force)
        ej, rj, _ = joint_terms(
            sample,
            qn,
            vn,
            h,
            obj.params.numpy(),
            obj.armature.numpy(),
            obj.damping.numpy(),
            obj.qt.numpy(),
            obj.vt.numpy(),
            obj.joint_force.numpy(),
        )
        expected_e = eo + np.sum(force * tn) + ej.sum()
        expected_r = ro + rj
        values = []
        for _ in range(32):
            cache.run("calibration", (h,), buffers, lambda: obj.evaluate(q, h))
            values.append(np.r_[obj.link_energy.numpy().sum() + obj.joint_energy.numpy().sum(), obj.residual.numpy()])
        values = np.asarray(values)
        residual_error = np.max(abs(values[:, 1:] - expected_r), axis=0)
        b = obj.tangent32.numpy().astype(float)
        scale = 1 / np.sqrt(np.maximum(np.diag(b), 1e-6))
        scaled_error = float(np.linalg.norm(scale * residual_error) / np.sqrt(k.count))
        sample_metrics = {
            "energy": float(np.max(abs(values[:, 0] - expected_e))),
            "residual": float(residual_error.max()),
            "slope": float(np.max(abs((values[:, 1:] - expected_r) @ direction))),
            "scaled_residual": scaled_error,
            "repeat_energy": float(np.ptp(values[:, 0])),
            "repeat_residual": float(np.ptp(values[:, 1:], axis=0).max()),
        }
        t, r, jt, _, _, _ = oracle.evaluate(sample)
        sample_metrics["fk"] = float(max(abs(k.com.numpy() - t).max(), abs(k.rotations.numpy() - r).max()))
        sample_metrics["jacobian"] = float(abs(k.jt.numpy() - jt).max())
        for name, value in sample_metrics.items():
            maxima[name] = max(maxima[name], value)
        records.append({"q": np.asarray(sample).tolist(), **sample_metrics})
    return {
        "dt": h,
        "device": str(model.device),
        "graph": graph,
        "maxima": maxima,
        "samples": records,
        "scaled_error_gate": 2.5e-9,
        "passed": maxima["scaled_residual"] * 8 <= 2.5e-9,
        "graphs": cache.summary(),
    }


def small_samples(case, qn, vn, h):
    eye = np.eye(len(qn))
    directions = [*eye, np.ones(len(qn)) / np.sqrt(len(qn))]
    samples = [qn.copy(), qn + h * vn]
    samples += [qn + sign * 1e-3 * d for d in directions for sign in (-1, 1)]
    if case["id"] == "driven_terms":
        row = case["joints"][0]
        a = row["target_ke"] + row["target_kd"] / h
        points = [
            0.0,
            row["limit_lower"],
            row["limit_upper"],
            row["limit_lower"] - row["limit_width"],
            row["limit_upper"] + row["limit_width"],
            row["effort_limit"] / a,
            -row["effort_limit"] / a,
        ]
        samples += [np.array([point + sign * 1e-6]) for point in points for sign in (-1, 1)]
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--dt", type=float, nargs="+", choices=(0.001, 0.0005, 0.00025, 0.000125), default=[0.001, 0.0005, 0.00025]
    )
    parser.add_argument("--stopping-scale", type=float, choices=(1.0, 0.1, 0.01), default=1.0)
    args = parser.parse_args()
    result = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        for case in specification()["cases"]:
            if args.cases and case["id"] not in args.cases:
                continue
            model, terms = build_case(case, device)
            qn = np.array([r["q0"] for r in case["joints"]], dtype=float)
            vn = np.array([r["qd0"] for r in case["joints"]], dtype=float)
            for h in args.dt:
                for graph in (False, True) if device != "cpu" else (False,):
                    qt, vt = target(case, h)
                    record = calibrate(
                        model,
                        terms,
                        qn,
                        vn,
                        small_samples(case, qn, vn, h),
                        h,
                        graph=graph,
                        qt=qt,
                        vt=vt,
                        body_force=case.get("world_COM_force", (0.0, 0.0, 0.0)),
                    )
                    record["case"] = case["id"]
                    result.append(record)
            print(device, case["id"], result[-1]["maxima"], flush=True)
        if args.cases and "sharpa_fk47" not in args.cases:
            continue
        parameters = load_fixture(Path("newton/examples/softbody/sharpa_g1h.json"))
        model, manifest = load_hand(args.asset_root / "left_sharpa_wave", device=device, parameters=parameters)
        names = manifest["joint_names"]
        trajectory = ClosureTrajectory(
            args.asset_root / "trajectory/Ball_catch - Left_hand_motion_rad.csv",
            names,
            model.joint_limit_lower.numpy(),
            model.joint_limit_upper.numpy(),
            model.joint_velocity_limit.numpy(),
        )
        terms = SolverMonolithic.JointTerms(
            implicit_pd=True,
            limits=True,
            friction=True,
            limit_width=tuple(parameters["joints"][name]["limit_width"] for name in names),
            friction_velocity_scale=tuple(parameters["joints"][name]["friction_velocity_scale"] for name in names),
        )
        qn, vn = trajectory.sample(0.0)
        samples = [qn, trajectory.sample(2.0)[0], trajectory.sample(4.0)[0]] + [
            qn + sign * 1e-3 * d for d in np.eye(22) for sign in (-1, 1)
        ]
        for h in args.dt:
            for graph in (False, True) if device != "cpu" else (False,):
                qt, vt = trajectory.sample(h)
                record = calibrate(
                    model, terms, qn, vn, samples, h, graph=graph, qt=qt, vt=vt, body_force=(0.0, 0.0, 0.0)
                )
                record["case"] = "sharpa_fk47"
                result.append(record)
                print(device, "sharpa_fk47", h, graph, record["maxima"], flush=True)
    frozen = json.loads((Path(__file__).parent / "fixtures/p3_pr2_noise_v1.json").read_text())["envelope"]
    for row in result:
        row["scaled_error_gate"] *= args.stopping_scale
        row["frozen_envelope_pass"] = all(8 * row["maxima"][key] <= bound for key, bound in frozen.items())
        row["passed"] = row["maxima"]["scaled_residual"] * 8 <= row["scaled_error_gate"] and row["frozen_envelope_pass"]
    maxima = {name: max(row["maxima"][name] for row in result) for name in result[0]["maxima"]}
    report = {
        "schema": "p3-pr2-calibration/v1",
        "fixture_sha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        "repeats": 32,
        "stopping_scale": args.stopping_scale,
        "frozen_envelope": frozen,
        "maxima": maxima,
        "envelope": {name: 8 * maxima[name] for name in ("energy", "residual", "slope")},
        "passed": all(row["passed"] for row in result),
        "records": result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("CALIBRATION", report["passed"], maxima, flush=True)


if __name__ == "__main__":
    main()
