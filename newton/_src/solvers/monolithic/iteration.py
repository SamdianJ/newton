# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared PR2/PR3 bounded iteration; component/state policy stays with each entry."""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import iteration_device
from .linear_types import MonolithicPcgConfig, MonolithicPcgWarmStart
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
    names = {"articulation": "q", "tet": "x", "global": "global"}
    return [(names.get(kind, kind), slice(start, start + count)) for kind, start, count in solver._linear.layout.groups]


def _residual_floor(config, name):
    """Return a block's linear residual denominator floor in its scaled units."""
    if name.startswith("cable_"):
        return 1e-12
    if name.startswith(("cloth_", "rigid_")) and not name.endswith("_r"):
        return config.residual_floor_x
    if name.startswith("rigid_"):
        return config.residual_floor_q
    return getattr(config, "residual_floor_" + name)


def _record_block(solver, name, key, value):
    if name.startswith(("rigid_", "cable_", "cloth_")):
        category = "rigid_blocks" if name.startswith("rigid_") else "material_blocks"
        solver._variational.setdefault(category, {}).setdefault(name, {})[key] = value
    else:
        solver._metrics[key] = value


@dataclass(frozen=True)
class Assessment:
    """Row-tolerance assessment of the accepted iterate."""

    converged: bool
    merit: float
    """Tolerance-normalized RMS residual."""
    caps: dict
    """Half of each block's smallest scaled row tolerance, the linear solve's sufficient accuracy."""


def _block_of(group):
    return "q" if group.startswith("q_") else group


def measure(solver, residual, reference, scale):
    """Recompute row tolerances at the accepted iterate and assess its residual.

    ``residual`` and ``reference`` are the accepted and step-start residuals;
    the device copy of the accepted residual is read from the solver. The device
    iteration reads back only per-group and per-block results.
    """
    self = solver
    device = isinstance(residual, wp.array)
    norms = None
    if device:
        if not isinstance(reference, wp.array):
            reference = wp.array(reference, dtype=self._dtype, device=self.model.device)
        # The device iteration builds its scaled RHS and block norms here.
        norms = iteration_device.measure(self, residual, reference)
    criterion = self._criterion
    for producer in self._producers:
        producer.coordinates(self, self._accepted, self._coordinates)
    criterion.round(self._linear.k_global_scalar_bsr, self._coordinates, self._complete_a)
    criterion.assess(self._complete_r)
    rows = criterion.rows
    worst, limited = criterion.worst.numpy(), criterion.limited.numpy()
    native = self._numpy_dtype
    ratios = {"global": native(worst.max(initial=0))}
    for group, value in zip(rows.names, worst, strict=True):
        block = _block_of(group)
        ratios[block] = max(ratios.get(block, native(0)), native(value))
    if device:
        sums, minimum = iteration_device.assessment(self, residual, reference)
        summary = {
            name: (sums[i, 0], sums[i, 1], minimum[i], norms[i, 2]) for i, name in enumerate(("global", "q", "x"))
        }
    else:
        tau = criterion.tau.numpy()
        r, r0 = np.asarray(residual), np.asarray(reference)
        s = np.asarray(scale).astype(native)
        summary = {
            name: (
                np.dot(r[part] / tau[part], r[part] / tau[part]),
                np.dot(r0[part] / tau[part], r0[part] / tau[part]),
                np.min(s[part] * tau[part]),
                np.linalg.norm(r[part]),
            )
            for name, part in blocks(self)
        }
    caps = {}
    for name, part in blocks(self):
        squares, reference_squares, smallest, raw = summary[name]
        size = native(part.stop - part.start)
        # A scaled L2 error below half the smallest scaled row tolerance bounds every row.
        caps[name] = native(0.5) * native(smallest)
        suffix = "" if name == "global" else "_" + name
        _record_block(self, name, "convergence_ratio" + suffix, float(ratios.get(name, 0.0)))
        _record_block(self, name, "merit" + suffix + "_final", float(np.sqrt(squares / size)))
        _record_block(self, name, "merit" + suffix + "_reference", float(np.sqrt(reference_squares / size)))
        if name != "global":
            _record_block(self, name, "raw_residual_" + name + "_norm", float(raw))
    self._variational["convergence_groups"] = {
        group: {"ratio": float(value), "rounding_limited": int(count)}
        for group, value, count in zip(rows.names, worst, limited, strict=True)
    }
    merit = native(np.sqrt(criterion.square_sum.numpy()[0] / native(max(1, rows.units.size))))
    return Assessment(bool(ratios["global"] <= 1), merit, caps)


def _rms(values):
    return np.sqrt(np.dot(values, values) / values.size) if values.size else 0.0


def trial_merit(solver):
    """Assess the last evaluated trial residual against the accepted iterate's frozen tolerances."""
    criterion = solver._criterion
    criterion.assess(solver._complete_r)
    native = solver._numpy_dtype
    return native(np.sqrt(criterion.square_sum.numpy()[0] / native(max(1, criterion.tau.size))))


def direction_norms(solver, vector):
    """Measure every active layout group in the nonlinear native precision."""
    if isinstance(vector, wp.array):
        values = iteration_device.norms(solver, vector)
        return dict(zip(("global", "q", "x"), values, strict=True))
    return {name: np.linalg.norm(vector[part]) for name, part in blocks(solver)}


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
    direction = self._delta if self._device_iteration else self._delta.numpy().astype(native)
    error = self._physical_linear_error(None, direction, residual, scale, device_result=self._device_iteration)
    errors = direction_norms(self, error)
    # The stored direction resolves its residual only to the rounding of
    # |A| |y|; in FP32 an ill-conditioned limit can lie below it.
    magnitude = self._post_magnitude if self._device_iteration else self._post_magnitude.numpy()
    rhs = direction_norms(self, self._rhs if self._device_iteration else self._rhs.numpy())
    rounding = linear_rounding(native)
    floors = {
        name: native(min(rounding * value, LINEAR_ROUNDING_CAP * rhs[name]))
        for name, value in direction_norms(self, magnitude).items()
    }
    limits = {name: max(limits[name], floors[name]) for name in limits}
    if native(regularization) == 0:
        originals, terms = errors, dict.fromkeys(errors, 0.0)
    else:
        originals = direction_norms(
            self, self._post_original if self._device_iteration else self._post_original.numpy()
        )
        terms = direction_norms(
            self, self._post_regularization if self._device_iteration else self._post_regularization.numpy()
        )
    internal = self._linear._masked_pcg_plan.norms.numpy()
    postchecks = {
        name: {
            "residual": errors[name],
            "limit": limits[name],
            "rounding_floor": floors[name],
            "original_residual": originals[name],
            "regularization_term": terms[name],
            "internal_residual": internal[index],
            "external_minus_internal_norm": errors[name] - internal[index],
        }
        for index, (name, _) in enumerate(blocks(self))
    }
    self._variational["linear_postchecks"].append(
        {
            "residual": errors["global"],
            "limit": min(limits.values()),
            "regularization": regularization,
            "actual_regularization": float(native(regularization)),
            "equation": "regularized" if native(regularization) else "original",
            "iteration": iteration,
            "retry": retry,
            "iterations": result.iterations,
            "blocks": postchecks,
        }
    )
    if any(not np.isfinite(value) for value in errors.values()):
        attempt["failure_reason"] = "nonfinite_posterior"
        return False
    if all(errors[name] <= limits[name] for name in limits):
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
    device = self._device_iteration
    reference = self._initial_reference(dt, device_result=True) if device else self._initial_reference(dt)
    small = False
    best_energy = best_merit = math.inf
    for iteration in range(self.newton_max_iterations + 1):
        if getattr(self, "_contact_spatial", False):
            self._refresh_candidates()
        e, evaluated, absolute = (
            self._evaluate(self._accepted, dt, device_result=True) if device else self._evaluate(self._accepted, dt)
        )
        if device:
            wp.copy(self._accepted_r, evaluated)
        residual = self._accepted_r if device else evaluated
        _, scale = self._assemble(iteration, device_result=True) if device else self._assemble(iteration)
        self._reference_residual = reference
        assessment = measure(self, residual, reference, scale)
        merit = assessment.merit
        best_energy, best_merit = min(best_energy, e), min(best_merit, merit)
        self._metrics["nonlinear_iterations"] = iteration
        if iteration == 0:
            for name, _ in blocks(self):
                key = "merit" + ("" if name == "global" else "_" + name)
                _record_block(self, name, key + "_initial", _read_block(self, name, key + "_final"))
        if device:
            wp.copy(self._stats_residual, residual)
        self._variational.update(objective_total=e, residual=self._stats_residual if device else residual.tolist())
        if assessment.converged:
            return self.Status.SUCCESS
        if small:
            raise _StepFailure(
                self.Status.NONLINEAR_STAGNATION, "Unrepresentable step with unresolved physical residual"
            )
        if iteration == self.newton_max_iterations:
            return self.Status.NONLINEAR_MAX_ITERATIONS
        rhs = None
        if not device:
            rhs = -scale * residual
            self._rhs.assign(rhs.astype(native))
        caps = assessment.caps
        # Inexact Newton with one global forcing term: each direction reduces the
        # whole scaled residual by the linear tolerance, or already lies within
        # half of every row tolerance. A per-block forcing term would ask FP32 PCG
        # for absolute accuracy that coupling to larger blocks cannot provide.
        rhs_norm = self._rhs_norms_host[0] if device else np.linalg.norm(rhs)
        forcing = native(self.linear_tolerance) * rhs_norm
        limits = {name: max(caps[name], forcing) for name, _ in blocks(self)}
        pcg = MonolithicPcgConfig(
            maximum_iterations=self.linear_max_iterations,
            # Mixed actors use the same actual-operator checkpoint cadence in
            # both backends; otherwise bounded batches can oversolve A/T and
            # give different accepted trajectories at unchanged outer gates.
            true_residual_interval=1 if self._linear.layout.rigid_ranges else 10,
            linear_tolerance=self.linear_tolerance,
            # An absolute term at the largest limit makes every group's target
            # min(max(rtol |b_g|, atol), limit_g) its own limit: PCG stops where
            # the posterior qualifies the direction, without oversolving.
            linear_absolute_tolerance=float(max(limits.values())),
            residual_norm_limits=tuple(limits.get(name, 1.0) for name in ("global", "q", "x")),
            group_norm_limits=tuple(limits[name] for name, _ in blocks(self))
            if self._linear.layout.rigid_ranges or self._linear.layout.material_ranges
            else None,
            group_residual_floors=tuple(_residual_floor(config, name) for name, _ in blocks(self))
            if self._linear.layout.material_ranges
            else None,
            residual_floor_global=config.residual_floor_global,
            residual_floor_q=config.residual_floor_q,
            residual_floor_x=config.residual_floor_x,
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
            except _StepFailure:
                continue
            if not solve_direction(
                self, residual, scale, limits, pcg, iteration=iteration, retry=retry, regularization=regularization
            ):
                continue
            direction = self._delta if device else self._delta.numpy().astype(native)
            coordinates = None if device else self._coordinates.numpy()
            for search in range(self.line_search_max_iterations):
                alpha = native(0.5**search)
                self._metrics["line_search_iterations"] += 1
                self._take_trial(alpha, dt)
                trial = self._trial_evaluation(dt, device_result=True) if device else self._trial_evaluation(dt)
                if trial is None or not self._domain(self._trial):
                    continue
                te, tr, ta = trial
                trial_value = trial_merit(self)
                effective = (
                    self._effective_displacement(device_result=True) if device else self._effective_displacement()
                )
                terms = (
                    iteration_device.trial(self, residual, tr, effective)
                    if device
                    else np.array([residual @ effective, abs(residual * direction).sum()])
                )
                slope = terms[0] / alpha
                weighted = terms[5] if device else terms[1]
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
                conservative = self._conservative_energy()
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
                    for index, (name, part) in enumerate(blocks(self)):
                        if device:
                            moved = np.sqrt(terms[6 + index] / native(part.stop - part.start))
                        else:
                            moved = _rms(
                                effective[part] if name.startswith("rigid_") else effective[part] / scale[part]
                            )
                        _record_block(
                            self, name, "scaled_step" + ("" if name == "global" else "_" + name), float(moved)
                        )
                    # No coordinate changed by more than its rounding: further iterations cannot move.
                    small = (
                        iteration_device.representable(self, effective, eps)
                        if device
                        else bool(np.all(np.abs(effective) <= eps * coordinates))
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


def _read_block(solver, name, key):
    if name.startswith(("rigid_", "cable_", "cloth_")):
        category = "rigid_blocks" if name.startswith("rigid_") else "material_blocks"
        return solver._variational[category][name][key]
    return solver._metrics[key]
