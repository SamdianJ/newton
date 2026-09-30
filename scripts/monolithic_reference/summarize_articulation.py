# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Separate PR2 same-discrete agreement, continuous accuracy and time refinement."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .articulation_fixtures import FIXTURE_PATH, build_case, specification
from .articulation_oracle import Kinematics


def angles(a, b):
    rel = a @ np.swapaxes(b, -1, -2)
    sine = (
        np.linalg.norm(
            np.stack(
                (rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0], rel[..., 1, 0] - rel[..., 0, 1]),
                axis=-1,
            ),
            axis=-1,
        )
        / 2
    )
    return np.arctan2(sine, (np.trace(rel, axis1=-2, axis2=-1) - 1) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_dir
    spec = specification()
    cases = {case["id"]: case for case in spec["cases"]}
    references = {}
    kinematics = {}
    convergence = {}
    input_hashes = {}

    def read(path):
        input_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for name, case in cases.items():
        model, _ = build_case(case, "cpu")
        kinematics[name] = Kinematics(model)
        n = len(case["joints"])
        q0 = np.array([v["q0"] for v in case["joints"]], dtype=float)
        v0 = np.array([v["qd0"] for v in case["joints"]], dtype=float)
        times = np.arange(1, round(case["duration"] / 0.001) + 1) * 0.001
        if name in ("pendulum", "noncollinear_chain", "driven_terms"):
            state = np.asarray(read(root / "continuous-reference-v1" / f"{name}-h128.json")["states"])[1:]
            previous = np.asarray(read(root / "continuous-reference-v1" / f"{name}-h64.json")["states"])[1:]
            convergence[name] = {
                "q": float(abs(state[:, :n] - previous[:, :n]).max()),
                "qd": float(abs(state[:, n:] - previous[:, n:]).max()),
            }
            q, v = state[:, :n], state[:, n:]
        else:
            q = q0 + times[:, None] * v0
            v = np.broadcast_to(v0, q.shape).copy()
            if name == "constant_force":
                acceleration = np.asarray(case["world_COM_force"])[0] / float(model.body_mass.numpy()[0])
                q += 0.5 * times[:, None] ** 2 * acceleration
                v += times[:, None] * acceleration
        poses = [kinematics[name].evaluate(value)[:2] for value in q]
        references[name] = {
            "q": q,
            "qd": v,
            "com": np.asarray([x[0] for x in poses]),
            "rotation": np.asarray([x[1] for x in poses]),
        }
    matrix = []
    base = {}
    base_noise = {}
    for directory in ("small-discrete-v1", "small-discrete-cpu-ar-v1", "small-discrete-cuda-v1"):
        for item in read(root / directory / "summary.json")["results"]:
            data = read(root / directory / item["trace_file"])
            name = data["case"]
            trace = data["trace"]
            ref = references[name]
            values = {k: np.asarray([r[k] for r in trace]) for k in ("q", "qd", "com", "rotation")}
            errors = {k: float(abs(values[k] - ref[k][: len(trace)]).max()) for k in ("q", "qd")}
            errors["link_COM_m"] = float(np.linalg.norm(values["com"] - ref["com"][: len(trace)], axis=-1).max())
            errors["link_rotation_rad"] = float(angles(values["rotation"], ref["rotation"][: len(trace)]).max())
            gates = spec["continuous_reference_max_errors"][name]
            mapped = {("q" if k.startswith("q_") else "qd" if k.startswith("qd_") else k): v for k, v in gates.items()}
            flags = {k: errors[k] <= v for k, v in mapped.items()}
            matrix.append(
                {
                    "case": name,
                    "device": data["device"],
                    "mode": data["globalization"],
                    "same_discrete_pass": data["passed_same_discrete"],
                    "continuous_errors": errors,
                    "continuous_gates": mapped,
                    "continuous_flags": flags,
                    "continuous_pass": all(flags.values()),
                }
            )
            if data["globalization"] == "ArmijoOrResNorm":
                base[name, data["device"]] = values
                base_noise[name, data["device"]] = {
                    k: float(np.sqrt(np.mean(np.array([row["error"][k] for row in trace]) ** 2))) for k in values
                }
    refinements = []
    for name in ("noncollinear_chain", "driven_terms"):
        for device in ("cpu", "cuda:0"):
            paths = [
                root / "small-refinement-v1" / f"{name}-{device.replace(':', '_')}-{h}.json" for h in (0.0005, 0.00025)
            ]
            if not all(path.exists() for path in paths):
                continue
            half, quarter = [read(path) for path in paths]
            values = []
            for trace, stride in ((half["trace"], 2), (quarter["trace"], 4)):
                values.append(
                    {
                        k: np.asarray([row[k] for row in trace])[stride - 1 :: stride]
                        for k in ("q", "qd", "com", "rotation")
                    }
                )
            coarse = base[name, device]
            fine, finest = values
            metrics = {}
            for key in coarse:
                a = angles(coarse[key], finest[key]) if key == "rotation" else coarse[key] - finest[key]
                b = angles(fine[key], finest[key]) if key == "rotation" else fine[key] - finest[key]
                ea, eb = float(np.sqrt(np.mean(a * a))), float(np.sqrt(np.mean(b * b)))
                noise = max(
                    base_noise[name, device][key],
                    *[
                        float(np.sqrt(np.mean(np.array([row["error"][key] for row in result["trace"]]) ** 2)))
                        for result in (half, quarter)
                    ],
                    128 * np.finfo(float).eps,
                )
                zero = bool(ea <= 20 * noise)
                metrics[key] = {
                    "coarse_rms": ea,
                    "half_rms": eb,
                    "ratio": eb / ea if ea else None,
                    "noise_bound": noise,
                    "zero_signal": zero,
                    "passed": None if zero else eb <= 0.8 * ea,
                }
            ref = references[name]
            errors = {k: float(abs(finest[k] - ref[k]).max()) for k in ("q", "qd")}
            certified = all(convergence[name][key] <= 0.1 * errors[key] for key in ("q", "qd"))
            refinements.append(
                {
                    "case": name,
                    "device": device,
                    "groups": metrics,
                    "finest_continuous_errors": errors,
                    "reference_deltas": convergence[name],
                    "reference_certified": certified,
                }
            )
    report = {
        "schema": "p3-pr2-physical-analysis/v1",
        "fixture_sha256": hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
        "input_sha256": input_hashes,
        "matrix": matrix,
        "reference_convergence": convergence,
        "refinement": refinements,
        "passed": all(r["same_discrete_pass"] and r["continuous_pass"] for r in matrix),
    }
    (root / "physical-analysis-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "matrix",
        len(matrix),
        "same discrete",
        sum(r["same_discrete_pass"] for r in matrix),
        "continuous",
        sum(r["continuous_pass"] for r in matrix),
    )
    for row in matrix:
        if row["device"] == "cpu" and row["mode"] == "ArmijoOrResNorm":
            print(row, flush=True)
    for row in refinements:
        print(row, flush=True)


if __name__ == "__main__":
    main()
