# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Emit short NVTX-labelled bounded PCG calls for a CUDA API trace audit, not timing."""

import json

import numpy as np
import warp as wp

from newton.tests.test_monolithic_pcg_production import impl, make_plan


def main():
    wp.init()
    device = wp.get_device("cuda:0")
    config = impl.MaskedPcgConfig(maximum_iterations=200, true_residual_interval=10)
    wp.load_module(module=impl.__name__, device=device)
    records = []
    for graph in (False, True):
        plan = make_plan(device, np.eye(4), [1, 2, 3, 4], config=config)
        plan.set_solve_parameters(regularization=0)
        if graph:
            plan.capture()
            # Instantiate/upload outside the warmed replay audit ranges.
            plan.replay()
            plan.collect_result()
        else:
            # Also warm specialized dot modules before the non-graph audit.
            plan.enqueue_solve()
            plan.collect_result()
        for label, sign, warm in (
            ("success", 1, False),
            ("failure", -1, False),
            ("recovery", 1, False),
            ("warm", 1, True),
        ):
            plan.values.assign((np.eye(4) * sign).astype(np.float32).reshape(-1))
            plan.set_solve_parameters(regularization=0, zero_initial_guess=not warm)
            with wp.ScopedTimer(f"masked_pcg_{graph}_{label}", use_nvtx=True, print=False, synchronize=False):
                if graph:
                    plan.replay()
                else:
                    plan.enqueue_solve()
            result = plan.collect_result()
            records.append(
                {
                    "graph": graph,
                    "case": label,
                    "status": result.status.name,
                    "iterations": result.iterations,
                    "slots": result.iteration_slots,
                }
            )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
