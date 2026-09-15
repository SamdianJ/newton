# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Research-only full-residual fallback; stop after the first repaired physical step."""

import argparse
import functools
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from threadpoolctl import threadpool_limits

from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.solver_monolithic import _StepFailure
from newton.examples.softbody import monolithic_sharpa_grasp as grasp
from scripts.monolithic_reference.p2_measurement import evidence_index, snapshot_sources


def least_squares_direction(jacobian, residual, damping=0.0):
    if not np.isfinite(jacobian).all() or not np.isfinite(residual).all():
        raise ValueError("Nonfinite residual Jacobian")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("Invalid least-squares damping")
    with threadpool_limits(limits=1):
        if damping:
            normal = jacobian.T @ jacobian + damping * np.eye(jacobian.shape[1])
            direction = np.linalg.solve(normal, -jacobian.T @ residual)
            rank = len(direction)
        else:
            direction, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)
    if not np.isfinite(direction).all() or residual @ (jacobian @ direction) >= 0:
        raise ValueError("No predicted residual descent")
    return direction, int(rank)


def residual_goal_weights(n, nq, targets):
    """Weight each row by its contribution to the three squared acceptance ratios."""
    targets = np.asarray(targets, dtype=float)
    if n <= 0 or not 0 <= nq <= n or not np.isfinite(targets).all() or np.any(targets <= 0):
        raise ValueError("Invalid residual goals")
    weights = np.full(n, 1 / (n * targets[0] ** 2))
    if nq:
        weights[:nq] += 1 / (nq * targets[1] ** 2)
    if nq < n:
        weights[nq:] += 1 / ((n - nq) * targets[2] ** 2)
    weights = np.sqrt(weights)
    return weights / np.max(weights)


def repair_source(original):
    source = textwrap.dedent(inspect.getsource(original))
    start = "        for retry, lambda_value in enumerate(config.regularization_values):\n"
    end = "        if not accepted:\n            raise _StepFailure(self.Status.REGULARIZATION_EXHAUSTED, last_reason)"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Nonlinear source markers changed")
    begin, finish = source.index(start), source.index(end)
    replacement = "        if not research_active:\n" + textwrap.indent(source[begin:finish], "    ")
    replacement += """        if not accepted and (research_active or last_reason == "Actual trial merit failed the decrease condition"):
            research_active = True
            accepted, small_step = self._research_fallback(dt, merit)
            last_reason = "Full-residual fallback failed original line search"
"""
    source = source[:begin] + replacement + source[finish:]
    return source.replace(
        "    small_step = False\n", "    small_step = False\n    research_active = self._research_force_start\n", 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-dir", "contact-dir", "trajectory", "calibration"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--position", choices=(18, 24), type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fd-multiplier", choices=(1.0, 0.5), type=float, default=0.5)
    parser.add_argument(
        "--goal-weighted", action="store_true", help="Replay with a merit normalized by final block goals"
    )
    parser.add_argument("--cube", default="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz")
    parser.set_defaults(
        device="cuda:0", experiment="grasp", pcg_mode="production", friction_off=False, disable_aabb=False
    )
    args = parser.parse_args()
    args.position_offset_mm = (10, -10, -5) if args.position == 18 else (10, 10, -5)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(args.output)
    (args.output / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    with patch.object(
        grasp,
        "MonolithicCollisionPipeline",
        functools.partial(MonolithicCollisionPipeline, _cubic_sdf=True, _smooth_history=True),
    ):
        case = grasp.GraspCase(args)
    solver = case.solver
    original = solver._iterate
    source = repair_source(original)
    (args.output / "iterate_research.py").write_text(source)
    namespace = original.__func__.__globals__.copy()
    exec(compile(source, str(args.output / "iterate_research.py"), "exec"), namespace)
    modified = namespace["_iterate"]
    events, currents = [], []
    initial_state, initial_history = {}, {}
    direction = wp.zeros_like(solver._delta)
    raw_status = None
    output = args.output
    solver._research_force_start = False
    solver._research_damped = False
    solver._research_weighted = False

    def fallback(dt, merit):
        accepted, trial = solver._transaction.accepted, solver._transaction.trial
        if not events:
            for name in ("particle_q", "particle_qd", "joint_q", "joint_qd", "body_q", "body_qd"):
                initial_state[name] = getattr(case.state, name).numpy().copy()
            for name in ("valid", "xi_local", "normal_local"):
                initial_history[name] = getattr(solver._contact.committed, name).numpy().copy()
            np.savez_compressed(
                output / "physical-input.npz",
                **initial_state,
                **{"history_" + k: v for k, v in initial_history.items()},
                target_q=case.control.joint_target_q.numpy(),
                target_qd=case.control.joint_target_qd.numpy(),
            )
        z = accepted.z.numpy().astype(float)
        scale = solver._linear.scale.numpy().astype(float)
        nq, n = solver._layout.q_dof_count, len(z)
        row = {
            "step": case.step_count + 1,
            "iteration": solver._metrics["nonlinear_iterations"],
            "before": merit,
            "trials": [],
        }
        events.append(row)
        start = time.perf_counter()

        def evaluate(offset):
            direction.assign(np.asarray(offset, dtype=np.float32))
            trial.form_trial(accepted, direction, 1.0, dt)
            value = solver._evaluate_trial(trial, dt)
            return trial.residual.numpy().astype(float), trial.z.numpy().astype(float), value

        center, _, base = evaluate(np.zeros(n))
        jacobian = np.empty((n, n))
        for column in range(n):
            offset = np.zeros(n)
            offset[column] = args.fd_multiplier * (2e-4 if column < nq else 2e-6)
            plus, zp, _ = evaluate(offset)
            minus, zm, _ = evaluate(-offset)
            width = zp[column] - zm[column]
            if width <= 0:
                raise _StepFailure(solver.Status.NONFINITE, "Unrepresentable FD increment")
            jacobian[:, column] = scale * (plus - minus) / width * scale[column]
            if column % 250 == 0:
                print("FD", row["step"], row["iteration"], column, n, flush=True)
        rhat = scale * center
        weights = np.ones(n)
        search_merit, noise = merit[0], solver._config.merit_noise
        if solver._research_weighted:
            reference = solver._rms(solver._residual_ref)
            targets = [
                getattr(solver._config, f"merit_absolute_{block}")
                + getattr(solver._config, f"merit_relative_{block}") * ref
                for block, ref in zip(("global", "q", "x"), reference, strict=True)
            ]
            weights = residual_goal_weights(n, nq, targets)
            search_merit = float(np.linalg.norm(weights * scale * accepted.residual.numpy()))
            noise *= float(np.linalg.norm(weights))
            row.update(targets=targets, search_merit=search_merit, search_noise=noise)
        model_jacobian, model_residual = weights[:, None] * jacobian, weights * rhat
        row.update(fd_seconds=time.perf_counter() - start, center_merit=base.merit)
        np.savez_compressed(
            output / f"direction-{len(events):02}.npz",
            z=z,
            scale=scale,
            residual=center,
            jacobian=jacobian,
            goal_weights=weights,
        )
        best = None
        dampings = solver._config.regularization_values if solver._research_damped else (0.0,)
        for damping in dampings:
            try:
                y, rank = least_squares_direction(model_jacobian, model_residual, damping)
            except (ValueError, np.linalg.LinAlgError) as error:
                row["trials"].append({"damping": damping, "failure": str(error)})
                continue
            solver._delta.assign((scale * y).astype(np.float32))
            for search in range(solver.line_search_max_iterations):
                alpha = 0.5**search
                solver._metrics["line_search_iterations"] += 1
                trial.form_trial(accepted, solver._delta, alpha, dt)
                try:
                    result = solver._evaluate_trial(trial, dt)
                except _StepFailure as error:
                    row["trials"].append({"damping": damping, "alpha": alpha, "failure": str(error)})
                    continue
                score = (
                    float(np.linalg.norm(weights * scale * trial.residual.numpy()))
                    if solver._research_weighted
                    else result.merit
                )
                threshold = (1 - 1e-4 * alpha) * search_merit + noise
                passes = score <= threshold
                row["trials"].append(
                    {
                        "damping": damping,
                        "alpha": alpha,
                        "merit": result.merit,
                        "search_merit": score,
                        "q": result.merit_q,
                        "x": result.merit_x,
                        "threshold": threshold,
                        "passes": passes,
                    }
                )
                if passes:
                    if best is None or score < best[0]:
                        best = (score, damping, alpha, y, rank)
                    break
        found, small = False, False
        if best is not None:
            _, damping, alpha, y, rank = best
            solver._delta.assign((scale * y).astype(np.float32))
            solver._y.assign(y.astype(np.float32))
            step_rms = solver._rms(solver._y, scaled=False)
            solver._metrics["line_search_iterations"] += 1
            trial.form_trial(accepted, solver._delta, alpha, dt)
            result = solver._evaluate_trial(trial, dt)
            score = (
                float(np.linalg.norm(weights * scale * trial.residual.numpy()))
                if solver._research_weighted
                else result.merit
            )
            found = score <= (1 - 1e-4 * alpha) * search_merit + noise
            row.update(
                chosen_damping=damping,
                chosen_alpha=alpha,
                chosen_merit=result.merit,
                chosen_search_merit=score,
                predicted_derivative=float(model_residual @ (model_jacobian @ y) / n),
                linear_system_rank=rank,
            )
            np.savez_compressed(output / f"correction-{len(events):02}.npz", correction=scale * y)
            if found:
                np.testing.assert_array_equal(accepted.z.numpy(), z.astype(np.float32))
                for name, before in initial_history.items():
                    np.testing.assert_array_equal(getattr(solver._contact.committed, name).numpy(), before)
                solver._transaction.accept_trial()
                solver._metrics.update(
                    accepted_alpha=alpha,
                    scaled_step=alpha * step_rms[0],
                    scaled_step_q=alpha * step_rms[1],
                    scaled_step_x=alpha * step_rms[2],
                )
                small = all(
                    alpha * value <= getattr(solver._config, f"step_tolerance_{block}")
                    for block, value in zip(("global", "q", "x"), step_rms, strict=True)
                )
        row["accepted"] = found
        (output / "fallbacks.json").write_text(json.dumps(events, indent=2) + "\n")
        print("FALLBACK", {k: v for k, v in row.items() if k != "trials"}, flush=True)
        return found, small

    current = solver._evaluate_current

    def observe(candidate, dt):
        value = current(candidate, dt)
        currents.append(
            {"iteration": solver._metrics["nonlinear_iterations"], "merit": [value.merit, value.merit_q, value.merit_x]}
        )
        if events:
            print("CURRENT", currents[-1], flush=True)
        return value

    def iterate(dt):
        nonlocal raw_status
        try:
            status = modified(solver, dt)
        except _StepFailure as error:
            raw_status = error.status.name
            raise
        raw_status = status.name
        if events and status != solver.Status.SUCCESS:
            raise _StepFailure(
                solver.Status.REGULARIZATION_EXHAUSTED, "Research repair did not converge: " + status.name
            )
        return status

    solver._research_fallback = fallback
    solver._evaluate_current = observe
    solver._iterate = iterate

    def save_result():
        case.save(output)
        stats = solver.last_stats
        unchanged = None
        if events and stats.rolled_back:
            unchanged = all(np.array_equal(getattr(case.state, k).numpy(), v) for k, v in initial_state.items())
            unchanged &= all(
                np.array_equal(getattr(solver._contact.committed, k).numpy(), v) for k, v in initial_history.items()
            )
        result = {
            "scope": "first failure physical step only, full FD research, no production backend change",
            "position": args.position,
            "from_step_start": solver._research_force_start,
            "damped_least_squares": solver._research_damped,
            "goal_weighted_merit": solver._research_weighted,
            "raw_status": raw_status,
            "published_status": stats.status.name,
            "failure": case.failure,
            "rolled_back": stats.rolled_back,
            "rollback_arrays_unchanged": unchanged,
            "currents": currents,
            "fallbacks": events,
            "newton_budget": solver.newton_max_iterations,
            "final_ratios": [stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x],
            "recovered_physical_step": bool(
                events and not case.failure and raw_status == "SUCCESS" and stats.converged and not stats.rolled_back
            ),
            "full_close_fixed": False,
        }
        (output / "repair.json").write_text(json.dumps(result, indent=2) + "\n")
        evidence_index(output)
        return result

    try:
        while not case.failure and case.step_count < 2500 and not events:
            currents.clear()
            case.step()
            if case.step_count % 200 == 0:
                print("step", case.step_count, flush=True)
        result = save_result()
        branches = (
            (("goal-weighted-step-start", True),)
            if args.goal_weighted
            else (("from-step-start", False), ("damped-step-start", True))
        )
        for name, damped in branches:
            if not events or not result["rollback_arrays_unchanged"]:
                break
            # Replay the exact rolled-back physical input with the original ten-update budget.
            output = args.output / name
            output.mkdir()
            solver._research_force_start = True
            solver._research_damped = damped
            solver._research_weighted = args.goal_weighted
            case.failure = None
            events.clear()
            currents.clear()
            initial_state.clear()
            initial_history.clear()
            case.step()
            result = save_result()
    finally:
        result = save_result()
        evidence_index(args.output)
    return int(not result["recovered_physical_step"])


if __name__ == "__main__":
    sys.exit(main())
