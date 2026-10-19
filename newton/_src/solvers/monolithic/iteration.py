# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared PR2/PR3 bounded iteration; component/state policy stays with each entry."""

import math
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from . import iteration_device
from .linear_types import MonolithicPcgConfig, MonolithicPcgWarmStart
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
    """Qualify the delivered direction, correcting native error within one budget."""
    self = solver
    remaining = pcg.maximum_iterations
    correction = 0
    solve_config = pcg
    material = self._material_blocks
    if material:
        # Each FP32 inner solve need only contract its own residual. Qualification
        # of the native accumulated direction still uses the frozen physical caps.
        inner = direction_norms(self, scale * residual)
        solve_config = replace(
            pcg,
            group_norm_limits=tuple(max(limits[name], pcg.linear_tolerance * inner[name]) for name, _ in blocks(self)),
        )
    rhs, output = self._rhs, self._delta
    attempt = {
        "iteration": iteration,
        "retry": retry,
        "regularization": regularization,
        "passes": [],
        "failure_reason": None,
    }
    self._variational.setdefault("linear_direction_attempts", []).append(attempt)
    while remaining > 0:
        try:
            result = self._linear.solve_pcg(
                rhs,
                self._y,
                generation=self._generation,
                warm_start=MonolithicPcgWarmStart.ZERO,
                config=solve_config,
                delta=output,
            )
            remaining -= result.iterations
            self._metrics["linear_iterations"] += result.iterations
            self._metrics["linear_scheduled_slots"] += result.scheduled_slots
            attempt["passes"].append(
                {
                    "correction": correction,
                    "iterations": result.iterations,
                    "remaining_iterations": remaining,
                    "status": result.status.name,
                    "graph_rebuilt": result.graph_rebuilt,
                    "capture_seconds": result.capture_seconds,
                }
            )
            self._require_linear(result.status)
            if not result.output_valid:
                raise _StepFailure(self.Status.NONFINITE, "Invalid PCG output")
            if self._native_direction:
                changed = self._accumulate_direction(correction)
            else:
                changed = self._combine_direction_correction() if correction else True
        except _StepFailure as error:
            attempt["failure_reason"] = (
                "direction_accuracy_budget_exhausted"
                if correction and remaining == 0 and error.status == self.Status.LINEAR_MAX_ITERATIONS
                else error.status.value
            )
            return False
        delivered = self._native_delta if self._native_direction else self._delta
        direction = delivered if self._device_iteration else delivered.numpy().astype(self._numpy_dtype)
        error = self._physical_linear_error(None, direction, residual, scale, device_result=self._device_iteration)
        errors = direction_norms(self, error)
        if np.float32(regularization) == 0:
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
                "actual_regularization": float(np.float32(regularization)),
                "equation": "regularized" if np.float32(regularization) else "original",
                "iteration": iteration,
                "retry": retry,
                "correction": correction,
                "total_iterations": pcg.maximum_iterations - remaining,
                "blocks": postchecks,
            }
        )
        if any(not np.isfinite(value) for value in errors.values()):
            attempt["failure_reason"] = "nonfinite_posterior"
            return False
        if all(errors[name] <= limits[name] for name in limits):
            return True
        if not changed:
            attempt["failure_reason"] = "direction_precision_unreachable"
            return False
        if remaining == 0:
            break
        # Solve A_lambda z = -e_lambda in FP32, then recheck the recovered,
        # rounded total delta. The original RHS and external gates stay frozen.
        self._prepare_direction_correction()
        rhs, output = self._correction_rhs, self._correction_delta
        # An original RHS floor can otherwise accept the correction's zero
        # guess. Apply the same rtol to its native error norm, without that floor.
        targets = {
            name: max(
                float(np.finfo(np.float32).smallest_subnormal),
                pcg.linear_tolerance * value if material else min(limits[name], pcg.linear_tolerance * value),
            )
            if value
            else limits[name]
            for name, value in errors.items()
        }
        solve_config = replace(
            pcg,
            maximum_iterations=remaining,
            linear_absolute_tolerance=0.0,
            residual_norm_limits=tuple(targets.get(name, 1.0) for name in ("global", "q", "x")),
            group_norm_limits=tuple(targets[name] for name, _ in blocks(self)),
        )
        correction += 1
    attempt["failure_reason"] = "direction_accuracy_budget_exhausted"
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
            self._rhs.assign(rhs.astype(np.float32))
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
            linear_absolute_tolerance=self.linear_absolute_tolerance,
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
            delivered = self._native_delta if self._native_direction else self._delta
            direction = delivered if device else delivered.numpy().astype(native)
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
