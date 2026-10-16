# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run source-bound coupled material acceptance and preserve every failed case."""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from .material_acceptance import MaterialAcceptance
from .material_recipes import RECIPE_PATH
from .profiling import provenance


def run_case(name, args, source):
    """Save full histories, checkpoints and a truthful receipt even on failure."""
    output = args.output / name
    output.mkdir(parents=True, exist_ok=False)
    start, case = time.perf_counter(), None
    receipt = {
        "status": "INCOMPLETE",
        "case": name,
        "source": source,
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        case = MaterialAcceptance(
            name,
            recipe=args.recipe,
            device=args.device,
            pcg_mode=args.pcg_mode,
            use_cuda_graph=bool(args.use_cuda_graph),
        )
        stop = min(args.max_steps or case.common["steps"], case.common["steps"])
        while case.steps < stop:
            if case.steps == case.common["checkpoint_step"]:
                case.restore_checkpoint(output / "checkpoint.npz")
                receipt["checkpoint_sha256"] = hashlib.sha256((output / "checkpoint.npz").read_bytes()).hexdigest()
            case.step()
            if case.steps % 100 == 0:
                print(json.dumps(dict(case=name, **case.records[-1])), flush=True)
                (output / "progress.json").write_text(json.dumps(case.records[-1], indent=2) + "\n")
        if case.steps == case.common["steps"]:
            case.check_final()
            receipt["status"] = "PASS"
        else:
            receipt["status"] = "PARTIAL"
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        receipt["seconds"] = time.perf_counter() - start
        receipt["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["cuda_mempool_peak_bytes"] = (
            wp.get_mempool_used_mem_high(args.device) if wp.get_device(args.device).is_cuda else None
        )
        if case is not None:
            receipt.update(
                steps=case.steps,
                resume_position_error=case.resume_position_error,
                resume_theta_error=case.resume_theta_error,
                natural_identity=case.natural_identity,
            )
            (output / "history.json").write_text(json.dumps(case.records, indent=2) + "\n")
            np.savez_compressed(
                output / "frames.npz",
                position=np.asarray(case.positions),
                theta=np.asarray(case.angles),
                q=np.asarray(case.coordinates),
            )
            if case.restart is not None:
                (output / "restart-history.json").write_text(json.dumps(case.restart.records, indent=2) + "\n")
            if receipt["status"] == "FAIL":
                (output / "failure-stats.txt").write_text(str(case.solver.last_stats))
                try:
                    case.solver.save_checkpoint(case.state, output / "failed-state.npz")
                except Exception as error:
                    receipt["failed_state_export_error"] = str(error)
        receipt["source_unchanged"] = provenance()["source_hashes"] == source["source_hashes"]
        if not receipt["source_unchanged"]:
            receipt.update(status="FAIL", source_error="Sources changed during execution")
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "source"}), flush=True)
    return receipt


def main():
    """Run every requested case; a filtered or short matrix remains PARTIAL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    parser.add_argument("--case", choices=("all", "cable-bend-twist", "cloth-press", "precreased-box"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--use-cuda-graph", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source = provenance()
    names = tuple(json.loads(args.recipe.read_text())["cases"]) if args.case == "all" else (args.case,)
    results = [run_case(name, args, source) for name in names]
    status = (
        "FAIL"
        if any(r["status"] == "FAIL" for r in results)
        else "PASS"
        if args.case == "all" and all(r["status"] == "PASS" for r in results)
        else "PARTIAL"
    )
    (args.output / "summary.json").write_text(
        json.dumps({"status": status, "cases": {r["case"]: r["status"] for r in results}}, indent=2) + "\n"
    )
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
