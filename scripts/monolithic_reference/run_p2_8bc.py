# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one CUDA PR-8B/8C comparison using the unchanged PR-8A measurement protocol.

Each invocation is one fresh process, including disposable warmup and the full
trajectory. Schedule repeats serially with separate output directories. Timing
comparisons require matched device state; smoke runs are correctness checks only.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from newton.solvers.experimental.monolithic import SolverMonolithic

from scripts.monolithic_reference import run_p2_8a
from scripts.monolithic_reference.p2_measurement import evidence_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("sharpa", "sharpa-profile", "sharpa-matrix", "tet", "tet-profile", "tet-matrix"),
    )
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--refinement", type=int, default=3)
    parser.add_argument("--mass-matrix", choices=("reference", "owned"), default="reference")
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--smoke", action="store_true")
    launch_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(launch_argv)
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    refinements = (2, 3) if args.kind.startswith("sharpa") else (3, 4, 5, 6)
    if args.refinement not in refinements:
        parser.error(f"{args.kind} requires refinement in {refinements}")

    options = {
        "use_optimized_articulation_mass_matrix": args.mass_matrix == "owned",
        "pcg_mode": args.pcg_mode,
    }
    record = {
        "task": "PR-8B/PR-8C",
        "launch_argv": launch_argv,
        "requested": options,
        "constructions": [],
        "binding": "Constructor-only adapter; production step/solve implementations remain unchanged.",
        "timing_scope": "SMOKE_CORRECTNESS_ONLY" if args.smoke else "REQUIRES_MATCHED_DEVICE_STATE_REVIEW",
        "failure": None,
    }

    def save():
        (args.output / "execution-options.json").write_text(json.dumps(record, indent=2) + "\n")

    original_init = SolverMonolithic.__init__

    def configured_init(self, *positional, **kwargs):
        for name, value in options.items():
            actual = kwargs.get(name, value)
            if getattr(actual, "value", actual) != value:
                raise ValueError(f"Conflicting constructor option: {name}")
            kwargs[name] = value
        original_init(self, *positional, **kwargs)
        actual = {
            "use_optimized_articulation_mass_matrix": self.use_optimized_articulation_mass_matrix,
            "pcg_mode": self.pcg_mode.value,
        }
        if actual != options:
            raise RuntimeError(f"Constructor options were not bound: {actual}")
        record["constructions"].append(actual)
        save()

    delegated_argv = [
        sys.argv[0],
        "--output",
        str(args.output),
        "--kind",
        args.kind,
        "--device",
        args.device,
        "--refinement",
        str(args.refinement),
    ]
    if args.smoke:
        delegated_argv.append("--smoke")
    try:
        with patch.object(SolverMonolithic, "__init__", configured_init), patch.object(sys, "argv", delegated_argv):
            run_p2_8a.main()
    except BaseException as error:
        record["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output.is_dir():
            save()
            evidence_index(args.output)


if __name__ == "__main__":
    main()
