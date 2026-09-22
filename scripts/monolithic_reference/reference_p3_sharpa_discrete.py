# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Certify independent exact-H Sharpa roots and production refinement noise."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnose_p3_sharpa import SharpaDiagnostic
from .p3_articulation_oracle import DiscreteOracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("newton/examples/softbody/sharpa_g1h.json"))
    parser.add_argument("--refinement", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--goal-scale", type=float, choices=(1.0, 0.1), default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = SharpaDiagnostic(args, "cpu", "ArmijoOrResNorm")
    model, h = case.model, case.sim_dt
    oracle = DiscreteOracle(
        model,
        case.solver._joint_terms.params.numpy(),
        case.state.joint_q.numpy(),
        case.state.joint_qd.numpy(),
        h,
        goal_scale=args.goal_scale,
    )
    force = (
        model.body_mass.numpy().astype(float)[:, None] * model.gravity.numpy().astype(float)[model.body_world.numpy()]
    )
    sources = [Path(__file__), Path(__file__).with_name("p3_articulation_oracle.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    result = {
        "manifest": case.manifest,
        "source_sha256": hashes,
        "goal_scale": args.goal_scale,
        "newton_budget": 100,
        "backtrack_budget": 32,
        "certified": False,
        "steps": 0,
        "failure": None,
    }
    with (args.output_dir / "trace.jsonl").open("w") as stream:
        try:
            for step in range(round(4.0 / h)):
                qt, vt = case.trajectory.sample((step + 1) * h)
                previous_q, previous_r = oracle.q.copy(), oracle.r.copy()
                row = oracle.step(
                    h, qt.astype(np.float32).astype(float), vt.astype(np.float32).astype(float), force, np.zeros(22)
                )
                if np.max(abs(row["q"] - previous_q)) >= np.pi / 2 or np.any(
                    np.einsum("bij,bij->b", row["rotation"], previous_r) <= 1.0
                ):
                    raise RuntimeError("ORACLE_UNCERTIFIED: root outside the local rotation domain")
                row = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in row.items()}
                row.update(step=step + 1, time=(step + 1) * h)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                result["steps"] = step + 1
                if (step + 1) % 250 == 0:
                    stream.flush()
                    print(args.refinement, step + 1, row["residual_rms"], row["goal"], flush=True)
            result["certified"] = True
        except (RuntimeError, np.linalg.LinAlgError) as error:
            result["failure"] = str(error)
            np.savez(
                args.output_dir / "failed-reference-input.npz",
                q=oracle.q,
                qd=oracle.v,
                com=oracle.t,
                rotation=oracle.r,
                linear_velocity=oracle.linear_velocity,
                rotation_velocity=oracle.rotation_velocity,
                parameters=oracle.parameters,
                armature=oracle.armature,
                damping=oracle.damping,
                target_q=qt.astype(np.float32),
                target_qd=vt.astype(np.float32),
                force=force,
                dt=h,
            )
            result["failed_step"] = step + 1
    if hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}:
        result.update(certified=False, failure="Oracle sources changed during execution")
    result["trace_sha256"] = hashlib.sha256((args.output_dir / "trace.jsonl").read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["certified"], result["steps"], result["failure"], flush=True)


if __name__ == "__main__":
    main()
