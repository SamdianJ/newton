# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen no-contact shell experiment and retain its source-bound receipt."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .cloth_benchmarks import ClothBenchmark


def main():
    """Record full or partial execution, including failed trajectories without relabeling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("patch", "prefold"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    if Path(newton.__file__).resolve() != source / "newton/__init__.py":
        raise RuntimeError("Benchmark imported Newton outside tested checkout")
    files = [
        *sorted((source / "newton/_src/solvers/monolithic").glob("*.py")),
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in ("cloth_benchmarks.py", "cloth_recipes.json", "cloth_oracle.py", "material_recipes.py")
        ],
    ]
    receipt = {
        "status": "INCOMPLETE",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source": str(source),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "numpy": np.__version__,
            "newton": newton.__file__,
        },
        "sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    path = args.output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    start, case = time.perf_counter(), None
    try:
        case = ClothBenchmark(args.kind, device=args.device, pcg_mode=args.pcg_mode, use_cuda_graph=args.use_cuda_graph)
        stop = min(args.max_steps or case.total_steps, case.total_steps)
        while case.steps < stop:
            if case.steps == case.recipe["checkpoint_step"] and case.steps + 1 <= stop:
                case.restore_checkpoint(args.output / "checkpoint.npz")
            else:
                case.step()
            if case.steps % 40 == 0:
                print(json.dumps(case.metrics()), flush=True)
        if case.steps == case.total_steps:
            case.check_final()
            assert case.checkpoint_error is not None, "Missing checkpoint continuation"
            receipt["status"] = "PASS"
        else:
            receipt["status"] = "PARTIAL"
    except Exception as error:
        receipt.update(status="FAIL", error=str(error))
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - start
        if case is not None:
            receipt["metrics"] = case.metrics()
            (args.output / "history.json").write_text(json.dumps(case.records, indent=2) + "\n")
            np.savez_compressed(
                args.output / "positions.npz",
                reference=case.rest,
                peak=case.peak_positions,
                final=case.state.particle_q.numpy(),
                triangles=case.triangles,
            )
            if receipt["status"] == "FAIL":
                case.solver.save_checkpoint(case.state, args.output / "failed-state.npz")
                (args.output / "failure-stats.txt").write_text(str(case.solver.last_stats))
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "sha256"}), flush=True)


if __name__ == "__main__":
    main()
