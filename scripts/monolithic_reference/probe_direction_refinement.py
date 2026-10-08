# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare continuation and native-residual correction on an offline PCG input."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.pcg_production import MaskedPcgConfig, MaskedPcgPlan
from scripts.monolithic_reference.check_recording import sha
from scripts.monolithic_reference.workload_io import load_arrays, write_json


def posterior(arrays, delta, regularization):
    """Multiply actual sparse FP32 B and delivered delta independently in FP64."""
    scale = arrays["scale"].astype(float)
    error = arrays["native_residual"].astype(float).copy()
    for row, (start, end) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
        error[row] += arrays["values"][start:end].astype(float) @ delta[arrays["columns"][start:end]].astype(float)
    return scale * error + float(np.float32(regularization)) * (delta.astype(float) / scale)


def probe(arrays, metadata, limits, strategy, device):
    """Keep the original qualification limits and one total iteration budget."""
    nq = arrays["q_cholesky"].shape[0]

    def norms(vector):
        return np.array([np.linalg.norm(vector), np.linalg.norm(vector[:nq]), np.linalg.norm(vector[nq:])])

    types = {"offsets": int, "columns": int, "x_cholesky": wp.mat33}
    buffers = {
        name: wp.array(arrays[name], dtype=types.get(name, float), device=device)
        for name in ("offsets", "columns", "values", "scale", "q_cholesky", "x_cholesky", "rhs_hat", "y", "delta")
    }
    config = MaskedPcgConfig(**metadata["config"])
    cap = arrays["limits"].copy()
    if strategy == "margin_control":
        cap *= 0.5
        config = replace(
            config,
            linear_absolute_tolerance=config.linear_absolute_tolerance * 0.5,
            linear_tolerance=config.linear_tolerance * 0.5,
        )
    work = 0
    delta = np.zeros_like(arrays["delta"])
    steps = []
    start = time.perf_counter()
    while work < metadata["config"]["maximum_iterations"]:
        plan = MaskedPcgPlan(**buffers, config=config)
        plan.residual_norm_limits.assign(cap)
        plan.set_solve_parameters(
            regularization=metadata["regularization"], zero_initial_guess=not steps or strategy == "correction"
        )
        plan.enqueue_solve()
        result = plan.collect_result()
        work += result.iterations
        if result.output_valid:
            candidate = plan.delta.numpy()
            delta = (delta + candidate).astype(np.float32) if steps and strategy == "correction" else candidate
        error = posterior(arrays, delta, metadata["regularization"])
        error_norms = norms(error)
        gap_norms = norms(error + plan.true_r.numpy())
        steps.append(
            {
                "iterations": result.iterations,
                "status": result.status.name,
                "posterior": error_norms,
                "limits": limits,
                "internal": plan.norms.numpy()[:3],
                "gap": gap_norms,
                "total_iterations": work,
            }
        )
        if not result.output_valid or np.all(error_norms <= limits) or strategy in ("baseline", "margin_control"):
            break
        remaining = metadata["config"]["maximum_iterations"] - work
        if remaining <= 0 or result.iterations == 0:
            break
        config = replace(config, maximum_iterations=remaining)
        if strategy == "correction":
            buffers["rhs_hat"].assign((-error).astype(np.float32))
            config = replace(config, linear_absolute_tolerance=0.0)
            cap = np.maximum(
                np.finfo(np.float32).smallest_subnormal, np.minimum(limits, config.linear_tolerance * error_norms)
            )
        else:
            if np.any(limits - gap_norms <= 0):
                break
            cap = np.minimum(cap, limits - gap_norms)
    return {
        "steps": steps,
        "qualified": bool(np.all(error_norms <= limits)),
        "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    """Compare all strategies using the same authenticated captured input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    baseline = json.loads((args.baseline / "result.json").read_text())
    call = baseline["pcg"][0]
    if sha(args.baseline / "linear-input-00.npz") != call["input_sha256"]:
        raise ValueError("Captured PCG input hash mismatch")
    arrays = load_arrays(args.baseline / "linear-input-00.npz", call["schema"])
    first = baseline["stats"]["variational"]["linear_postchecks"][0]["blocks"]
    limits = np.array([first[name]["limit"] for name in ("global", "q", "x")])
    results = {
        strategy: probe(arrays, call["input"], limits, strategy, args.device)
        for strategy in ("baseline", "continuation", "correction", "margin_control")
    }
    write_json(args.output, results)
    for name, result in results.items():
        print(name, result["qualified"], [step["iterations"] for step in result["steps"]], flush=True)


if __name__ == "__main__":
    main()
