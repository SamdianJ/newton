# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton, regularization and line-search policy for one bound nonlinear problem."""

import math

import numpy as np

from .linear_types import MonolithicPcgWarmStart
from .pcg_common import LINEAR_ROUNDING_CAP, linear_rounding
from .solver_types import _StepFailure


def acceptance(
    current,
    trial,
    slope,
    alpha,
    merit,
    trial_merit,
    *,
    energy_noise,
    residual_noise,
    slope_noise,
    best_energy=None,
    best_merit=None,
    dtype=np.float64,
):
    """Accept a trial when either the Armijo or the residual-norm predicate holds.

    Both predicates are returned so callers can report which one accepted.
    """
    if not all(math.isfinite(v) for v in (current, trial, slope, merit, trial_merit)):
        raise ValueError("Nonfinite variational trial")
    if not 0 < alpha <= 1 or not all(math.isfinite(v) and v >= 0 for v in (energy_noise, residual_noise, slope_noise)):
        raise ValueError("Invalid variational acceptance parameters")
    a = slope < -slope_noise and trial <= current + dtype(1.0e-4) * alpha * slope + energy_noise
    r = trial_merit <= merit + residual_noise
    # Either metric may rise, but alternating rises must not undo both records.
    a = a and trial <= (current if best_energy is None else best_energy) + energy_noise
    r = r and trial_merit <= (merit if best_merit is None else best_merit) + residual_noise
    accepted = a or r
    improved = accepted and ((a and trial < current - energy_noise) or (r and trial_merit < merit - residual_noise))
    return bool(accepted), bool(a), bool(r), bool(improved)


def blocks(solver):
    return solver._problem.blocks


def measure(solver, residual, reference, scale):
    return solver._problem.assess(residual, reference)


def solve_direction(solver, residual, scale, limits, pcg, *, iteration, retry, regularization):
    """Solve once, then qualify the direction by its true residual in the solver precision."""
    self = solver
    native = self._numpy_dtype
    attempt = {
        "iteration": iteration,
        "retry": retry,
        "regularization": regularization,
        "iterations": 0,
        "status": None,
        "failure_reason": None,
    }
    self._variational.setdefault("linear_direction_attempts", []).append(attempt)
    try:
        result = self._linear.solve_pcg(
            self._rhs,
            self._y,
            generation=self._generation,
            warm_start=MonolithicPcgWarmStart.ZERO,
            config=pcg,
            delta=self._delta,
        )
        self._metrics["linear_iterations"] += result.iterations
        self._metrics["linear_scheduled_slots"] += result.scheduled_slots
        attempt.update(
            iterations=result.iterations,
            status=result.status.name,
            graph_rebuilt=result.graph_rebuilt,
            capture_seconds=result.capture_seconds,
        )
        self._require_linear(result.status)
        if not result.output_valid:
            raise _StepFailure(self.Status.NONFINITE, "Invalid PCG output")
        self._project_gauges()
    except _StepFailure as error:
        attempt["failure_reason"] = error.status.value
        return False
    direction = self._delta
    error = self._physical_linear_error(None, direction, residual, scale, device_result=True)
    norms = self._problem.posterior_norms(error)
    # The stored direction resolves its residual only to the rounding of
    # |A| |y|; in FP32 an ill-conditioned limit can lie below it.
    rounding = linear_rounding(native)
    postchecks = {}
    for index, (name, _) in enumerate(blocks(self)):
        error_norm, rhs_norm, magnitude, original, term, internal = norms[index]
        floor = native(min(rounding * magnitude, LINEAR_ROUNDING_CAP * rhs_norm))
        postchecks[name] = {
            "residual": error_norm,
            "limit": max(limits[name], floor),
            "rounding_floor": floor,
            "original_residual": original,
            "regularization_term": term,
            "internal_residual": internal,
            "external_minus_internal_norm": error_norm - internal,
        }
    self._variational["linear_postchecks"].append(
        {
            "residual": postchecks[self._problem.aggregate_name]["residual"],
            "limit": min(row["limit"] for row in postchecks.values()),
            "regularization": regularization,
            "actual_regularization": float(native(regularization)),
            "equation": "regularized" if native(regularization) else "original",
            "iteration": iteration,
            "retry": retry,
            "iterations": result.iterations,
            "blocks": postchecks,
        }
    )
    if any(not np.isfinite(row["residual"]) for row in postchecks.values()):
        attempt["failure_reason"] = "nonfinite_posterior"
        return False
    if all(row["residual"] <= row["limit"] for row in postchecks.values()):
        return True
    # A direction whose true residual misses a limit is a linear failure; the
    # caller retries with the next regularization rather than correcting it.
    attempt["failure_reason"] = "direction_posterior"
    return False


def iterate(solver, dt):
    """Keep one direction, one line search, and every row's physical tolerance per iterate."""
    self = solver
    native = getattr(self, "_numpy_dtype", np.float64)
    eps = native(np.finfo(native).eps)
    tiny = native(1e-30)
    config = self._config
    problem = self._problem
    reference = None
    small = False
    best_energy = best_merit = math.inf
    for iteration in range(self.newton_max_iterations + 1):
        problem.refresh()
        e, residual, absolute = problem.evaluate_accepted(dt)
        if iteration == 0:
            # Relative tolerances follow the predictor's residual. The step-start
            # residual r(x_n) carries the inertia M v / h that the predictor
            # cancels, so a fast body would loosen its whole block's tolerance
            # and stop at the predictor with its load unresolved.
            reference = problem.freeze_reference(residual)
        scale = problem.linearize(iteration)
        self._reference_residual = reference
        assessment = measure(self, residual, reference, scale)
        merit = assessment.merit
        best_energy, best_merit = min(best_energy, e), min(best_merit, merit)
        self._metrics["nonlinear_iterations"] = iteration
        if iteration == 0:
            problem.record_initial_merit()
        problem.record_residual(e, residual)
        if assessment.converged:
            return self.Status.SUCCESS
        if small:
            raise _StepFailure(
                self.Status.NONLINEAR_STAGNATION, "Unrepresentable step with unresolved physical residual"
            )
        if iteration == self.newton_max_iterations:
            return self.Status.NONLINEAR_MAX_ITERATIONS
        caps = assessment.caps
        # Inexact Newton with one global forcing term: each direction reduces the
        # whole scaled residual by the linear tolerance, or already lies within
        # half of every row tolerance. A per-block forcing term would ask FP32 PCG
        # for absolute accuracy that coupling to larger blocks cannot provide.
        rhs_norm = self._rhs_norms_host[0]
        forcing = native(self.linear_tolerance) * rhs_norm
        limits = {name: max(caps[name], forcing) for name, _ in blocks(self)}
        pcg = problem.pcg_config(limits)
        accepted = False
        for retry, regularization in enumerate(config.regularization_values):
            self._metrics["regularization_retries"] += int(retry > 0)
            self._metrics["lambda_value"] = regularization
            try:
                self._require_linear(self._linear.set_regularization(regularization, generation=self._generation))
                self._require_linear(
                    self._linear.factor_actor_preconditioner(generation=self._generation, pivot_tolerance=0.0)
                )
            except _StepFailure:
                continue
            if not solve_direction(
                self, residual, scale, limits, pcg, iteration=iteration, retry=retry, regularization=regularization
            ):
                continue
            for search in range(self.line_search_max_iterations):
                alpha = native(0.5**search)
                self._metrics["line_search_iterations"] += 1
                problem.candidate(alpha, dt)
                trial = problem.evaluate_trial(dt)
                if trial is None or not problem.domain(self._accepted, self._trial):
                    continue
                te, _, ta = trial
                trial_value = problem.merit()
                effective = problem.difference(self._accepted, self._trial)
                slope_dot, weighted, moved = problem.trial_summary(residual, effective)
                slope = slope_dot / alpha
                # Rounding allowances in the selected precision for energy, merit and slope.
                allowance = native(128) * eps
                en = allowance * max(tiny, absolute, ta)
                rn = allowance * max(tiny, merit, trial_value)
                sn = allowance * max(tiny, weighted)
                passes, a, r, improved = acceptance(
                    e,
                    te,
                    slope,
                    alpha,
                    merit,
                    trial_value,
                    energy_noise=en,
                    residual_noise=rn,
                    slope_noise=sn,
                    best_energy=best_energy,
                    best_merit=best_merit,
                    dtype=native,
                )
                conservative = problem.conservative_energy()
                self._variational["trials"].append(
                    {
                        "iteration": iteration,
                        "retry": retry,
                        "alpha": alpha,
                        "energy": te,
                        "merit": trial_value,
                        "slope": slope if conservative else None,
                        "Armijo": a if conservative else None,
                        "acceptance_slope": slope,
                        "energy_predicate": a,
                        "energy_change": te - e,
                        "energy_noise": en,
                        "residual_noise": rn,
                        "slope_noise": sn,
                        "ResNorm": r,
                        "accepted": passes,
                        "improved": improved,
                        "best_energy": best_energy,
                        "best_merit": best_merit,
                    }
                )
                if passes:
                    problem.record_step(moved)
                    # No coordinate changed by more than its rounding: further iterations cannot move.
                    small = problem.unrepresentable(effective, eps)
                    problem.accept()
                    self._metrics["accepted_alpha"] = alpha
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            raise _StepFailure(
                self.Status.REGULARIZATION_EXHAUSTED, "No direction/trial passed unchanged physical gates"
            )
    raise AssertionError("Unreachable")
