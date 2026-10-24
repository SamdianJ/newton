# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one closed-form tet experiment, or summarize a complete recipe matrix.

``case`` executes real solver steps and writes an auditable receipt; truncated
runs are PARTIAL. ``summarize`` applies the mesh-convergence and mass-bias gates
to one same-source receipt per expected case. A PROPOSED recipe never yields PASS.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .tet_benchmarks import KINDS, RECIPES, TetBenchmark, qualify_matrix

SOURCE = Path(__file__).resolve().parents[2]


def source_hashes():
    files = [
        *sorted((SOURCE / "newton/_src/solvers/monolithic").glob("*.py")),
        *(Path(__file__).with_name(name) for name in ("run_tet_benchmark.py", "tet_benchmarks.py", "tet_analytic.py")),
        *(Path(__file__).with_name(name) for name in ("tet_oracle.py", "tet_recipes.json")),
    ]
    return {str(f.relative_to(SOURCE)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def run_case(args):
    """Advance one experiment through the public solver and record the single-case gates."""
    args.output.mkdir(parents=True, exist_ok=False)
    if Path(newton.__file__).resolve() != SOURCE / "newton/__init__.py":
        raise RuntimeError("Benchmark imported Newton outside the tested checkout")
    receipt = {
        "kind": args.kind,
        "case": args.case,
        "cells": args.cells,
        "material_model": args.material_model,
        "mass_mode": args.mass_mode,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "recipe_status": RECIPES["status"],
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "numpy": np.__version__,
            "newton": newton.__file__,
        },
        "sha256": source_hashes(),
        "status": "INCOMPLETE",
    }
    start, case = time.perf_counter(), None
    try:
        case = TetBenchmark(
            args.kind,
            case=args.case,
            cells=args.cells,
            material_model=args.material_model,
            mass_mode=args.mass_mode,
            device=args.device,
            pcg_mode=args.pcg_mode,
            use_cuda_graph=args.use_cuda_graph,
            precision=args.precision,
        )
        receipt["cells"] = list(case.cells)
        receipt["precision"] = case.solver.precision.value
        for _ in range(min(args.max_steps or case.total_steps, case.total_steps)):
            case.step()
        result = case.check_final()
        receipt["metrics"] = result["metrics"]
        receipt["status"] = result["status"] if case.steps == case.total_steps else "PARTIAL"
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - start
        if case is not None:
            np.savez_compressed(
                args.output / "final.npz",
                positions=case.state.particle_q.numpy(),
                history=np.asarray(case.history),
            )
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k not in ("sha256", "error")}), flush=True)


def summarize(args):
    """Collect every receipt under the input directory and apply the matrix gates."""
    receipts = [json.loads(path.read_text()) for path in sorted(args.input.rglob("receipt.json"))]
    receipts = [r for r in receipts if r["kind"] == args.kind]
    result = qualify_matrix(args.kind, receipts, recipe_status=RECIPES["status"])
    result.update(kind=args.kind, recipe_status=RECIPES["status"], receipts=len(receipts))
    (args.input / f"{args.kind}-matrix.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return int(result["status"] not in ("PASS", "PARTIAL"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    case = commands.add_parser("case", help="Run one experiment")
    case.add_argument("kind", choices=KINDS)
    case.add_argument("--case", choices=sorted(RECIPES["patch"]["cases"]))
    case.add_argument("--cells", type=int, nargs=3)
    case.add_argument("--material-model", default="smith_log_stabilized")
    case.add_argument("--mass-mode", default="consistent")
    case.add_argument("--device", default="cpu")
    case.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    case.add_argument("--use-cuda-graph", action="store_true")
    case.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    case.add_argument("--max-steps", type=int)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(func=run_case)
    matrix = commands.add_parser("summarize", help="Apply the matrix gates to collected receipts")
    matrix.add_argument("kind", choices=KINDS)
    matrix.add_argument("--input", type=Path, required=True)
    matrix.set_defaults(func=summarize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
