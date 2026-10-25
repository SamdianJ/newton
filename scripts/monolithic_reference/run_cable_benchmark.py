# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a frozen cable experiment and retain an auditable single-case receipt."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .cable_benchmarks import CableBenchmark
from .numerical_gates import loaded_reference_modules


def main():
    """Execute actual solver steps; partial development runs never report PASS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("cantilever", "localized_helix"))
    parser.add_argument("--segments", type=int)
    parser.add_argument("--axis", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ea-multiplier", type=float, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = Path(__file__).resolve().parents[2]
    if Path(newton.__file__).resolve() != source / "newton/__init__.py":
        raise RuntimeError("Benchmark imported Newton outside the tested checkout")
    files = [
        *sorted((source / "newton/_src/solvers/monolithic").glob("*.py")),
        *loaded_reference_modules(),
        Path(__file__).with_name("cable_recipes.json"),
    ]
    receipt = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source": str(source),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "numpy": np.__version__,
            "newton": newton.__file__,
        },
        "sha256": {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "status": "INCOMPLETE",
    }
    progress = None
    if args.resume:
        previous = json.loads((args.output / "receipt.json").read_text())
        if previous["status"] == "FAIL":
            raise ValueError("A failed experiment cannot be resumed into an acceptance PASS")
        ignored = {"output", "resume", "max_steps"}
        if previous["sha256"] != receipt["sha256"] or any(
            previous["arguments"].get(k) != v for k, v in receipt["arguments"].items() if k not in ignored
        ):
            raise ValueError("Resume requires identical source, recipe and execution arguments")
        progress = json.loads((args.output / "progress.json").read_text())
        if hashlib.sha256((args.output / "progress.npz").read_bytes()).hexdigest() != progress["checkpoint_sha256"]:
            raise ValueError("Progress checkpoint and metadata do not match")
        receipt["resumed_from_step"] = progress["step"]
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.perf_counter()
    elapsed = 0.0 if progress is None else progress["seconds"]
    case = None
    try:
        case = CableBenchmark(
            args.kind,
            segments=args.segments,
            axis=args.axis,
            ea_multiplier=args.ea_multiplier,
            device=args.device,
            pcg_mode=args.pcg_mode,
            use_cuda_graph=args.use_cuda_graph,
            precision=args.precision,
        )
        receipt["precision"] = case.solver.precision.value
        if progress is not None:
            case.solver.load_checkpoint(case.state, args.output / "progress.npz")
            case.steps = progress["step"]
            case.maximum_residual = progress["maximum_residual"]
        for _ in range(max(0, min(args.max_steps or case.total_steps, case.total_steps) - case.steps)):
            case.step()
            if case.steps == case.recipe["checkpoint_step"]:
                case.restore_checkpoint(args.output / "checkpoint.npz")
            if case.steps % 100 == 0:
                checkpoint = args.output / "progress.npz"
                case.solver.save_checkpoint(case.state, checkpoint)
                progress = {
                    "step": case.steps,
                    "seconds": elapsed + time.perf_counter() - start,
                    "maximum_residual": case.maximum_residual,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                }
                (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                print(json.dumps({"step": case.steps, "seconds": progress["seconds"]}), flush=True)
        receipt["metrics"] = case.metrics()
        if case.steps == case.total_steps:
            if case.segments == case.recipe["segments"][-1]:
                case.check_final()
                receipt["status"] = "EA_SAMPLE" if args.kind == "localized_helix" and args.ea_multiplier < 1 else "PASS"
            else:
                case.check_final(accuracy=False)
                receipt["status"] = "MESH_SAMPLE"
        else:
            receipt["status"] = "PARTIAL"
    except Exception as error:
        receipt["status"] = "FAIL"
        receipt["error"] = str(error)
        if case is not None:
            receipt["metrics"] = case.metrics()
            try:
                checkpoint = args.output / "failed-state.npz"
                case.solver.save_checkpoint(case.state, checkpoint)
                receipt["failed_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            except Exception as checkpoint_error:
                receipt["checkpoint_error"] = str(checkpoint_error)
        raise
    finally:
        receipt["seconds"] = elapsed + time.perf_counter() - start
        if case is not None:
            np.savez_compressed(
                args.output / "final.npz",
                positions=case.state.particle_q.numpy(),
                theta=case.state._monolithic_coupled.cable_theta.numpy(),
                reference_twist=case.state._monolithic_coupled.cable_ref_twist.numpy(),
            )
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k not in ("sha256", "error")}), flush=True)


if __name__ == "__main__":
    main()
