# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Select a private PR8F candidate for an unchanged PR8D production-graph trajectory pilot."""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton._src.solvers.monolithic.soft_preconditioner import _BlockPreconditioner
from scripts.monolithic_reference import preconditioner_study as study
from scripts.monolithic_reference import run_p2_8bc
from scripts.monolithic_reference.p2_measurement import evidence_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=study.CANDIDATES, required=True)
    parser.add_argument("--kind", choices=("tet", "sharpa"), required=True)
    parser.add_argument("--refinement", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    record = {
        "candidate": args.candidate,
        "phase": "PILOT_NOT_FORMAL_ADMISSION",
        "constructions": [],
        "numeric_policy": "Full update for every required generation/scaling/lambda; fixed reference hierarchy",
        "failure": None,
    }
    original_init = SolverMonolithic.__init__

    def configured(self, *positional, **kwargs):
        original_init(self, *positional, **kwargs)
        wp.synchronize_device(self.model.device)
        start = time.perf_counter()
        hierarchy = study.build_hierarchy(
            {
                "rest_x": self.model.particle_q.numpy(),
                "dynamic_particle_ids": self._layout.dynamic_particle_ids.numpy(),
                "tet_indices": self.model.tet_indices.numpy(),
            },
            args.candidate,
        )
        if args.candidate != "actor_block":
            self._linear._study_soft = _BlockPreconditioner(self._linear, hierarchy)
            self._linear._factor_valid = False
        wp.synchronize_device(self.model.device)
        record["constructions"].append(
            {
                "candidate": args.candidate,
                "hierarchy_sha256": hierarchy.fingerprint,
                "level_counts": hierarchy.level_counts,
                "hierarchy_and_allocation_seconds": time.perf_counter() - start,
                "pcg_mode": self.pcg_mode.value,
                "cuda_graph": self.use_cuda_graph,
                "lambda_and_scales": "Updated in-place by the existing linear workspace lifecycle",
            }
        )

    delegated = [
        "--output",
        str(args.output),
        "--kind",
        args.kind,
        "--refinement",
        str(args.refinement),
        "--mass-matrix",
        "owned",
        "--pcg-mode",
        "production",
        "--use-cuda-graph",
    ]
    if args.smoke:
        delegated.append("--smoke")
    try:
        with patch.object(SolverMonolithic, "__init__", configured):
            run_p2_8bc.main(delegated)
    except BaseException as error:
        record["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.output.is_dir():
            (args.output / "preconditioner-options.json").write_text(json.dumps(record, indent=2) + "\n")
            evidence_index(args.output)


if __name__ == "__main__":
    main()
