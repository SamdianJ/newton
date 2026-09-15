# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One rejected-step counterfactual; never promote an unresolved branch to physical output."""

import argparse
import functools
import inspect
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.examples.softbody import monolithic_sharpa_grasp as grasp
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources


def research_source(original):
    source = textwrap.dedent(inspect.getsource(original))
    marker = (
        "        if not accepted:\n            raise _StepFailure(self.Status.REGULARIZATION_EXHAUSTED, last_reason)"
    )
    if source.count(marker) != 1 or source.count("    small_step = False\n") != 1:
        raise ValueError("Nonlinear source markers changed")
    source = source.replace("    small_step = False\n", "    small_step = False\n    research_used = False\n", 1)
    insertion = """    if not accepted and not research_used and last_reason == "Actual trial merit failed the decrease condition":
        alpha = self._research_alpha
        self._metrics["line_search_iterations"] += 1  # Reevaluate an already-tested alpha for the counterfactual.
        transaction.trial.form_trial(transaction.accepted, self._delta, alpha, dt)
        trial = self._evaluate_trial(transaction.trial, dt)
        self._research_record(iteration, merit, reference, trial, alpha, lambda_value)
        transaction.accept_trial()
        self._metrics["accepted_alpha"] = alpha
        self._metrics.update(scaled_step=alpha * step_rms[0], scaled_step_q=alpha * step_rms[1], scaled_step_x=alpha * step_rms[2])
        small_step = all(alpha * value <= getattr(config, f"step_tolerance_{block}") for block, value in zip(("global", "q", "x"), step_rms, strict=True))
        accepted = True
        research_used = True
"""
    return source.replace(marker, textwrap.indent(insertion, "    ") + marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "contact-dir", "trajectory", "calibration"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--position", choices=(18, 24), type=int, required=True)
    parser.add_argument("--alpha", choices=(1.0, 0.25), type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cube", default="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz")
    parser.set_defaults(
        device="cuda:0", experiment="grasp", pcg_mode="production", friction_off=False, disable_aabb=False
    )
    args = parser.parse_args()
    args.position_offset_mm = (10, -10, -5) if args.position == 18 else (10, 10, -5)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    with patch.object(
        grasp,
        "MonolithicCollisionPipeline",
        functools.partial(MonolithicCollisionPipeline, _cubic_sdf=True, _smooth_history=True),
    ):
        case = grasp.GraspCase(args)
    solver = case.solver
    original = solver._iterate
    source = research_source(original)
    path = args.output / "iterate_research.py"
    path.write_text(source)
    namespace = original.__func__.__globals__.copy()
    exec(compile(source, str(path), "exec"), namespace)
    modified = namespace["_iterate"]
    solver._research_alpha = args.alpha
    interventions, currents = [], []
    raw_status = None
    current = solver._evaluate_current

    def observe(candidate, dt):
        result = current(candidate, dt)
        currents.append(
            {
                "iteration": solver._metrics["nonlinear_iterations"],
                "merit": result.merit,
                "merit_q": result.merit_q,
                "merit_x": result.merit_x,
            }
        )
        return result

    def record(iteration, merit, reference, trial, alpha, lambda_value):
        interventions.append(
            {
                "step": case.step_count + 1,
                "iteration": iteration,
                "remaining_newton_updates": solver.newton_max_iterations - iteration - 1,
                "alpha": alpha,
                "lambda": lambda_value,
                "before": merit,
                "reference": reference,
                "trial": [trial.merit, trial.merit_q, trial.merit_x],
                "original_threshold": (1 - 1e-4 * alpha) * merit[0] + solver._config.merit_noise,
            }
        )
        accepted, candidate = solver._transaction.accepted, solver._transaction.trial
        np.savez_compressed(
            args.output / "intervention.npz",
            z=accepted.z.numpy(),
            trial_z=candidate.z.numpy(),
            residual=accepted.residual.numpy(),
            trial_residual=candidate.residual.numpy(),
            delta=solver._delta.numpy(),
            scale=solver._linear.scale.numpy(),
        )
        print("INTERVENTION", interventions[-1], flush=True)

    def iterate(dt):
        nonlocal raw_status
        try:
            status = modified(solver, dt)
        except _StepFailure as error:
            raw_status = error.status.name
            raise
        raw_status = status.name
        if interventions and status != solver.Status.SUCCESS:
            # The baseline permits a soft stop. A counterfactual MUST NOT use
            # that rule to publish a state that has not met final acceptance.
            raise _StepFailure(
                solver.Status.REGULARIZATION_EXHAUSTED, "Research branch failed final acceptance: " + status.name
            )
        return status

    solver._evaluate_current = observe
    solver._research_record = record
    solver._iterate = iterate
    try:
        while not case.failure and case.step_count < 2500 and not interventions:
            currents.clear()
            case.step()
            if case.step_count % 200 == 0:
                print("step", case.step_count, flush=True)
    finally:
        case.save(args.output)
        stats = solver.last_stats
        result = {
            "scope": "one bypass after all original retries fail; stop after this physical step",
            "position": args.position,
            "alpha": args.alpha,
            "interventions": interventions,
            "currents": currents,
            "raw_status": raw_status,
            "published_status": stats.status.name,
            "failure": case.failure,
            "rolled_back": stats.rolled_back,
            "final_ratios": [stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x],
            "newton_budget": solver.newton_max_iterations,
            "line_search_reevaluations_added": len(interventions),
            "recovered_physical_step": bool(interventions and stats.converged),
            "full_close_fixed": False,
        }
        (args.output / "merit-counterfactual.json").write_text(json.dumps(result, indent=2) + "\n")
        evidence_index(args.output)
    return int(not result["recovered_physical_step"])


if __name__ == "__main__":
    sys.exit(main())
