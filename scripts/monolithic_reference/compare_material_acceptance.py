# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare every frame of the complete CPU/CUDA material acceptance matrix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(paths, recipe):
    """Reject partial/mixed receipts and apply the frozen position/theta limits."""
    contract = json.loads(recipe.read_text())
    digest = hashlib.sha256(recipe.read_bytes()).hexdigest()
    expected_modes = {("cpu", "diagnostic", 0), ("cuda:0", "production", 0), ("cuda:0", "production", 1)}
    modes, sources, errors = {}, [], {}
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        assert summary["status"] == "PASS", f"Incomplete input: {path}"
        assert set(summary["cases"]) == set(contract["cases"])
        receipts = {name: json.loads((path / name / "receipt.json").read_text()) for name in contract["cases"]}
        mode = None
        for name, receipt in receipts.items():
            args = receipt["arguments"]
            key = (args["device"], args["pcg_mode"], args["use_cuda_graph"])
            assert mode is None or mode == key, "Mixed mode within a matrix"
            mode = key
            assert receipt["status"] == "PASS" and receipt["steps"] == contract["common"]["steps"]
            assert receipt["recipe_sha256"] == digest and receipt["source_unchanged"]
            sources.append(receipt["source"]["source_hashes"])
            checkpoint = path / name / "checkpoint.npz"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt["checkpoint_sha256"]
        assert mode not in modes, "Duplicate mode"
        modes[mode] = path
    assert set(modes) == expected_modes, "Missing required mode"
    assert all(source == sources[0] for source in sources), "Mixed source identities"
    gates = contract["common"]["gates"]
    for name in contract["cases"]:
        with np.load(modes[("cpu", "diagnostic", 0)] / name / "frames.npz") as baseline:
            for mode, path in modes.items():
                with np.load(path / name / "frames.npz") as actual:
                    assert actual["position"].shape[0] == contract["common"]["steps"]
                    differences = {}
                    for field, limit in (
                        ("position", gates["cpu_cuda_position_error_m"]),
                        ("q", gates["cpu_cuda_position_error_m"]),
                        ("theta", gates["cpu_cuda_theta_error_rad"]),
                    ):
                        assert actual[field].shape == baseline[field].shape
                        assert np.isfinite(actual[field]).all() and np.isfinite(baseline[field]).all()
                        differences[field] = float(np.max(abs(actual[field] - baseline[field]), initial=0))
                        assert differences[field] <= limit, (name, mode, field, differences[field], limit)
                    errors[f"{name}:{mode}"] = differences
    return {"status": "PASS", "errors": errors, "recipe_sha256": digest, "source_hashes": sources[0]}


def main():
    """Write an explicit failure receipt when a complete comparison cannot pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=3)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.runs, args.recipe)
    except (AssertionError, ValueError, KeyError, OSError) as error:
        result = {"status": "FAIL", "error": str(error)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
