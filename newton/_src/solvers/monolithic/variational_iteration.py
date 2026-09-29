# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared PR2/PR3 bounded iteration; component/state policy stays with each entry."""

import math

import numpy as np

from .linear import MonolithicPcgConfig, MonolithicPcgWarmStart
from .solver_types import _StepFailure


def acceptance(
    mode,
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
    dtype=np.float64,
):
    """Apply independent A/R predicates after the caller's geometry/context checks."""
    if mode not in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
        raise ValueError("Unknown variational acceptance mode")
    if not all(math.isfinite(v) for v in (current, trial, slope, merit, trial_merit)):
        raise ValueError("Nonfinite variational trial")
    if not 0 < alpha <= 1 or not all(math.isfinite(v) and v >= 0 for v in (energy_noise, residual_noise, slope_noise)):
        raise ValueError("Invalid variational acceptance parameters")
    a = slope < -slope_noise and trial <= current + dtype(1.0e-4) * alpha * slope + energy_noise
    r = trial_merit <= merit + residual_noise
    accepted = a if mode == "Armijo" else r if mode == "ResNorm" else a or r
    improved = accepted and (
        (mode != "ResNorm" and a and trial < current - energy_noise)
        or (mode != "Armijo" and r and trial_merit < merit - residual_noise)
    )
    return bool(accepted), bool(a), bool(r), bool(improved)


def blocks(solver):
    n, q = solver._linear.layout.scalar_dof_count, solver._linear.layout.q_dof_count
    result = [("global", slice(0, n))]
    if q:
        result.append(("q", slice(0, q)))
    if q < n:
        result.append(("x", slice(q, n)))
    return result


def measure(solver, residual, reference, scale):
    """Use the same current scale for the frozen physical reference and candidate."""
    native = getattr(solver, "_numpy_dtype", np.float64)
    values = {}
    for name, part in blocks(solver):
        size = len(residual[part])
        merit = np.linalg.norm(scale[part] * residual[part]) / np.sqrt(native(size))
        ref = np.linalg.norm(scale[part] * reference[part]) / np.sqrt(native(size))
        goal = (
            native(getattr(solver._config, "merit_absolute_" + name))
            + native(getattr(solver._config, "merit_relative_" + name)) * ref
        )
        values[name] = (merit, ref, goal, size)
        suffix = "" if name == "global" else "_" + name
        solver._metrics["merit" + suffix + "_final"] = merit
        solver._metrics["merit" + suffix + "_reference"] = ref
        solver._metrics["convergence_ratio" + suffix] = merit / goal
        if name != "global":
            solver._metrics["raw_residual_" + name + "_norm"] = float(np.linalg.norm(residual[part]))
    return values


def iterate(solver, dt):
    """Keep one direction, one line search, and all active block gates per iterate."""
    self = solver
    native = getattr(self, "_numpy_dtype", np.float64)
    config = self._config
    reference = self._initial_reference(dt)
    small = False
    for iteration in range(self.newton_max_iterations + 1):
        e, residual, absolute = self._evaluate(self._accepted, dt)
        b, scale = self._assemble(iteration)
        if reference is None:
            reference = residual.copy()
        self._reference_residual = reference
        values = measure(self, residual, reference, scale)
        merit = values["global"][0]
        self._metrics["nonlinear_iterations"] = iteration
        if iteration == 0:
            for name, (m, _, _, _) in values.items():
                self._metrics["merit" + ("" if name == "global" else "_" + name) + "_initial"] = m
        self._variational.update(objective_total=e, residual=residual.tolist())
        if all(m <= goal for m, _, goal, _ in values.values()):
            return self.Status.SUCCESS
        if small:
            raise _StepFailure(
                self.Status.NONLINEAR_STAGNATION, "Small effective step with unresolved physical residual"
            )
        if iteration == self.newton_max_iterations:
            return self.Status.NONLINEAR_MAX_ITERATIONS
        rhs = -scale * residual
        self._rhs.assign(rhs.astype(np.float32))
        caps = {name: native(0.5) * np.sqrt(native(size)) * goal for name, (_, _, goal, size) in values.items()}
        pcg = MonolithicPcgConfig(
            maximum_iterations=self.linear_max_iterations,
            true_residual_interval=10,
            linear_tolerance=self.linear_tolerance,
            linear_absolute_tolerance=self.linear_absolute_tolerance,
            residual_norm_limits=tuple(caps.get(name, 1.0) for name in ("global", "q", "x")),
            residual_floor_global=config.residual_floor_global,
            residual_floor_q=config.residual_floor_q,
            residual_floor_x=config.residual_floor_x,
            curvature_absolute_tolerance=0.0,
            curvature_relative_tolerance=1e-12,
            preconditioner_positive_tolerance=0.0,
            stagnation_window=10,
            stagnation_minimum_reduction=1e-3,
        )
        accepted = False
        for retry, regularization in enumerate(config.regularization_values):
            self._metrics["regularization_retries"] += int(retry > 0)
            self._metrics["lambda_value"] = regularization
            try:
                self._require_linear(self._linear.set_regularization(regularization, generation=self._generation))
                self._require_linear(
                    self._linear.factor_actor_preconditioner(generation=self._generation, pivot_tolerance=0.0)
                )
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
                self._require_linear(result.status)
                if self.pcg_mode == self.PCGMode.DIAGNOSTIC:
                    self._require_linear(self._linear.recover_delta(self._y, self._delta, generation=self._generation))
                elif not result.output_valid:
                    raise _StepFailure(self.Status.NONFINITE, "Invalid PCG output")
            except _StepFailure:
                continue
            direction = self._delta.numpy().astype(native)
            error = self._physical_linear_error(b, direction, residual, scale)
            postchecks = {}
            for name, part in blocks(self):
                post = np.linalg.norm(error[part])
                limit = min(
                    caps[name],
                    max(
                        native(self.linear_absolute_tolerance),
                        native(self.linear_tolerance)
                        * max(np.linalg.norm(rhs[part]), native(getattr(config, "residual_floor_" + name))),
                    ),
                )
                postchecks[name] = {"residual": post, "limit": limit}
            self._variational["linear_postchecks"].append(
                {
                    "residual": postchecks["global"]["residual"],
                    "limit": min(p["limit"] for p in postchecks.values()),
                    "regularization": regularization,
                    "blocks": postchecks,
                }
            )
            if any(not np.isfinite(p["residual"]) or p["residual"] > p["limit"] for p in postchecks.values()):
                continue
            for search in range(self.line_search_max_iterations):
                alpha = native(0.5**search)
                self._metrics["line_search_iterations"] += 1
                self._take_trial(alpha, dt)
                trial = self._trial_evaluation(dt)
                if trial is None or not self._domain(self._trial):
                    continue
                te, tr, ta = trial
                effective = self._effective_displacement()
                slope = (residual @ effective) / alpha
                trial_merit = np.linalg.norm(scale * tr) / np.sqrt(native(len(tr)))
                # Keep the PR2 common envelope; the coupled entry qualifies the
                # complete E/r against it before any physical trajectory use.
                eps = native(128 * np.finfo(float).eps)
                en = max(native(5.820766091346741e-11), eps * max(native(1.0), absolute, ta))
                rn = max(
                    native(2.0935893019213836e-9) * np.linalg.norm(scale) / np.sqrt(native(len(scale))),
                    eps * max(native(1.0), merit, trial_merit),
                )
                sn = max(
                    native(1.5498947892952669e-9) * np.linalg.norm(direction),
                    native(2.0935893019213836e-9) * abs(direction).sum(),
                    eps * max(native(1.0), abs(residual * direction).sum()),
                )
                passes, a, r, improved = acceptance(
                    self._globalization,
                    e,
                    te,
                    slope,
                    alpha,
                    merit,
                    trial_merit,
                    energy_noise=en,
                    residual_noise=rn,
                    slope_noise=sn,
                    dtype=native,
                )
                conservative = self._conservative_energy()
                self._variational["trials"].append(
                    {
                        "iteration": iteration,
                        "retry": retry,
                        "alpha": alpha,
                        "energy": te,
                        "merit": trial_merit,
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
                    }
                )
                if passes:
                    steps = {}
                    for name, part in blocks(self):
                        steps[name] = native(
                            np.linalg.norm(effective[part] / scale[part]) / np.sqrt(native(len(effective[part])))
                        )
                        self._metrics["scaled_step" + ("" if name == "global" else "_" + name)] = steps[name]
                    small = all(
                        value <= native(getattr(config, "step_tolerance_" + name)) for name, value in steps.items()
                    )
                    self._accepted, self._trial = self._trial, self._accepted
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
