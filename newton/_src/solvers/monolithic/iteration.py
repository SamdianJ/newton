# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared PR2/PR3 bounded iteration; component/state policy stays with each entry."""

import math
from dataclasses import replace

import numpy as np
import warp as wp

from . import iteration_device
from .linear_types import MonolithicPcgConfig, MonolithicPcgWarmStart
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
    names = {"articulation": "q", "tet": "x", "global": "global"}
    return [(names.get(kind, kind), slice(start, start + count)) for kind, start, count in solver._linear.layout.groups]


def _block_policy(config, prefix, name):
    if prefix == "step_tolerance" and (
        (name == "q" and config.joint_residual_tolerance is not None)
        or (name.startswith("cloth_") and config.cloth_position_tolerance is not None)
    ):
        # Explicit force goals can require nonzero corrections below the default step tolerance.
        return 0.0
    if name.startswith("cloth_"):
        return getattr(config, prefix + "_x")
    if name.startswith("cable_"):
        if prefix == "merit_absolute":
            return config.merit_absolute_cable_theta if name.endswith("_theta") else config.merit_absolute_cable_x
        if prefix == "merit_relative":
            return 0.0
        if prefix == "step_tolerance":
            # Stiff rods can need representable sub-picometre force corrections.
            return 0.0
        return 1e-12
    if not name.startswith("rigid_"):
        return getattr(config, prefix + "_" + name)
    translation = name.endswith("_t")
    if prefix == "merit_absolute":
        return 1e-7 if translation else 2.5e-8
    if prefix == "merit_relative":
        return 1e-6 if translation else 1e-5
    if prefix == "step_tolerance":
        return 1e-10 if translation else 9.100770221088748e-10
    return getattr(config, prefix + ("_x" if translation else "_q"))


def _record_block(solver, name, key, value):
    if name.startswith(("rigid_", "cable_", "cloth_")):
        category = "rigid_blocks" if name.startswith("rigid_") else "material_blocks"
        solver._variational.setdefault(category, {}).setdefault(name, {})[key] = value
    else:
        solver._metrics[key] = value


def measure(solver, residual, reference, scale):
    """Use the same current scale for the frozen physical reference and candidate."""
    native = getattr(solver, "_numpy_dtype", np.float64)
    values = {}
    norms = iteration_device.measure(solver, residual, reference) if isinstance(residual, wp.array) else None
    for name, part in blocks(solver):
        index = ("global", "q", "x").index(name) if norms is not None else None
        size = part.stop - part.start
        merit = (norms[index, 0] if norms is not None else np.linalg.norm(scale[part] * residual[part])) / np.sqrt(
            native(size)
        )
        ref = (norms[index, 1] if norms is not None else np.linalg.norm(scale[part] * reference[part])) / np.sqrt(
            native(size)
        )
        goal = (
            native(_block_policy(solver._config, "merit_absolute", name))
            + native(_block_policy(solver._config, "merit_relative", name)) * ref
        )
        if name.startswith("cable_"):
            # A fixed per-object unit scale makes this an L2 physical-force cap.
            goal = (
                native(_block_policy(solver._config, "merit_absolute", name))
                * scale[part.start]
                / np.sqrt(native(size))
            )
        cap = (
            solver._config.joint_residual_tolerance
            if name == "q"
            else solver._config.cloth_position_tolerance
            if name.startswith("cloth_")
            else None
        )
        if cap is not None:
            # min(scale) bounds the unscaled L2 norm even for unequal row scales.
            goal = native(min(goal, native(cap) * np.min(scale[part]) / np.sqrt(native(size))))
        raw_norm = float(norms[index, 2] if norms is not None else np.linalg.norm(residual[part]))
        if name in getattr(solver, "_cable_raw_theta_norms", {}):
            raw_norm = solver._cable_raw_theta_norms[name]
            merit = raw_norm * scale[part.start] / np.sqrt(native(size))
        values[name] = (merit, ref, goal, size)
        suffix = "" if name == "global" else "_" + name
        _record_block(solver, name, "merit" + suffix + "_final", merit)
        _record_block(solver, name, "merit" + suffix + "_reference", ref)
        _record_block(solver, name, "convergence_ratio" + suffix, merit / goal)
        if name != "global":
            _record_block(
                solver,
                name,
                "raw_residual_" + name + "_norm",
                raw_norm,
            )
    return values


def direction_norms(solver, vector):
    """Measure every active layout group in the nonlinear native precision."""
    if isinstance(vector, wp.array):
        values = iteration_device.norms(solver, vector)
        return dict(zip(("global", "q", "x"), values, strict=True))
    return {name: np.linalg.norm(vector[part]) for name, part in blocks(solver)}


def material_merit(solver, residual, values, scale):
    """Balance line-search progress against every frozen physical convergence goal."""
    native = solver._numpy_dtype
    ratios = []
    for name, part in blocks(solver):
        norm = np.linalg.norm(scale[part] * residual[part])
        if name in solver._cable_raw_theta_norms:
            norm = solver._cable_raw_theta_norms[name] * scale[part.start]
        ratios.append(norm / np.sqrt(native(part.stop - part.start)) / values[name][2])
    return np.linalg.norm(np.asarray(ratios, dtype=native))


def solve_direction(solver, residual, scale, limits, pcg, *, iteration, retry, regularization):
    """Qualify the delivered direction, correcting native error within one budget."""
    self = solver
    remaining = pcg.maximum_iterations
    correction = 0
    solve_config = pcg
    material = self._materials is not None
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
            if self.pcg_mode == self.PCGMode.DIAGNOSTIC:
                self._require_linear(self._linear.recover_delta(self._y, output, generation=self._generation))
            elif not result.output_valid:
                raise _StepFailure(self.Status.NONFINITE, "Invalid PCG output")
            if getattr(self, "_materials", None) is not None:
                from .material_runtime import accumulate_direction  # noqa: PLC0415

                changed = accumulate_direction(self, correction)
            else:
                changed = self._combine_direction_correction() if correction else True
        except _StepFailure as error:
            attempt["failure_reason"] = (
                "direction_accuracy_budget_exhausted"
                if correction and remaining == 0 and error.status == self.Status.LINEAR_MAX_ITERATIONS
                else error.status.value
            )
            return False
        delivered = self._native_delta if self._materials is not None else self._delta
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
        internal = (
            self._linear._true_residual_norms
            if self.pcg_mode == self.PCGMode.DIAGNOSTIC
            else self._linear._masked_pcg_plan.norms
        ).numpy()
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
    """Keep one direction, one line search, and all active block gates per iterate."""
    self = solver
    native = getattr(self, "_numpy_dtype", np.float64)
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
        values = measure(self, residual, reference, scale)
        merit = values["global"][0]
        if self._materials is not None:
            merit = material_merit(self, residual, values, scale)
            self._variational["residual_merit"] = "block_goal_normalized_l2"
        self._metrics["nonlinear_iterations"] = iteration
        if iteration == 0:
            for name, (m, _, _, _) in values.items():
                _record_block(self, name, "merit" + ("" if name == "global" else "_" + name) + "_initial", m)
        if device:
            wp.copy(self._stats_residual, residual)
        self._variational.update(objective_total=e, residual=self._stats_residual if device else residual.tolist())
        if all(m <= goal for m, _, goal, _ in values.values()):
            return self.Status.SUCCESS
        if small:
            raise _StepFailure(
                self.Status.NONLINEAR_STAGNATION, "Small effective step with unresolved physical residual"
            )
        if iteration == self.newton_max_iterations:
            return self.Status.NONLINEAR_MAX_ITERATIONS
        rhs = None
        if not device:
            rhs = -scale * residual
            self._rhs.assign(rhs.astype(np.float32))
        caps = {name: native(0.5) * np.sqrt(native(size)) * goal for name, (_, _, goal, size) in values.items()}
        pcg = MonolithicPcgConfig(
            maximum_iterations=self.linear_max_iterations,
            # Mixed actors use the same actual-operator checkpoint cadence in
            # both backends; otherwise bounded batches can oversolve A/T and
            # give different accepted trajectories at unchanged outer gates.
            true_residual_interval=1 if self._linear.layout.rigid_ranges else 10,
            linear_tolerance=self.linear_tolerance,
            linear_absolute_tolerance=self.linear_absolute_tolerance,
            residual_norm_limits=tuple(caps.get(name, 1.0) for name in ("global", "q", "x")),
            group_norm_limits=tuple(caps[name] for name, _ in blocks(self))
            if self._linear.layout.rigid_ranges or self._linear.layout.material_ranges
            else None,
            group_residual_floors=tuple(_block_policy(config, "residual_floor", name) for name, _ in blocks(self))
            if self._linear.layout.material_ranges
            else None,
            residual_floor_global=config.residual_floor_global,
            residual_floor_q=config.residual_floor_q,
            residual_floor_x=config.residual_floor_x,
            curvature_absolute_tolerance=0.0,
            curvature_relative_tolerance=1e-12,
            preconditioner_positive_tolerance=0.0,
            stagnation_window=10,
            stagnation_minimum_reduction=1e-3,
        )
        limits = {}
        for name, part in blocks(self):
            rhs_norm = self._rhs_norms_host[("global", "q", "x").index(name)] if device else np.linalg.norm(rhs[part])
            limits[name] = min(
                caps[name],
                max(
                    native(self.linear_absolute_tolerance),
                    native(self.linear_tolerance)
                    * max(rhs_norm, native(_block_policy(config, "residual_floor", name))),
                ),
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
            delivered = self._native_delta if self._materials is not None else self._delta
            direction = delivered if device else delivered.numpy().astype(native)
            for search in range(self.line_search_max_iterations):
                alpha = native(0.5**search)
                self._metrics["line_search_iterations"] += 1
                self._take_trial(alpha, dt)
                trial = self._trial_evaluation(dt, device_result=True) if device else self._trial_evaluation(dt)
                if trial is None or not self._domain(self._trial):
                    continue
                te, tr, ta = trial
                effective = (
                    self._effective_displacement(device_result=True) if device else self._effective_displacement()
                )
                terms = (
                    iteration_device.trial(self, residual, tr, effective)
                    if device
                    else np.array(
                        [
                            residual @ effective,
                            np.linalg.norm(scale * tr) ** 2,
                            np.linalg.norm(scale) ** 2,
                            np.linalg.norm(direction) ** 2,
                            abs(direction).sum(),
                            abs(residual * direction).sum(),
                        ]
                    )
                )
                slope = terms[0] / alpha
                trial_merit = (np.sqrt(terms[1]) if device else np.linalg.norm(scale * tr)) / np.sqrt(native(tr.size))
                if self._materials is not None:
                    trial_merit = material_merit(self, tr, values, scale)
                # Keep the PR2 common envelope; the coupled entry qualifies the
                # complete E/r against it before any physical trajectory use.
                eps = native(128 * np.finfo(float).eps)
                en = max(native(5.820766091346741e-11), eps * max(native(1.0), absolute, ta))
                rn = max(
                    native(2.0935893019213836e-9)
                    * (np.sqrt(terms[2]) if device else np.linalg.norm(scale))
                    / np.sqrt(native(scale.size)),
                    eps * max(native(1.0), merit, trial_merit),
                )
                sn = max(
                    native(1.5498947892952669e-9) * (np.sqrt(terms[3]) if device else np.linalg.norm(direction)),
                    native(2.0935893019213836e-9) * terms[4],
                    eps * max(native(1.0), terms[5]),
                )
                if self._linear.layout.material_ranges:
                    eps = native(128 * np.finfo(native).eps)
                    en = eps * max(native(1e-30), absolute, ta)
                    rn = eps * max(native(1e-30), merit, trial_merit)
                    sn = eps * max(native(1e-30), terms[5])
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
                            (
                                np.sqrt(terms[6 + ("global", "q", "x").index(name)])
                                if device
                                else np.linalg.norm(
                                    effective[part] if name.startswith("rigid_") else effective[part] / scale[part]
                                )
                            )
                            / np.sqrt(native(part.stop - part.start))
                        )
                        _record_block(self, name, "scaled_step" + ("" if name == "global" else "_" + name), steps[name])
                    small = all(
                        value <= native(_block_policy(config, "step_tolerance", name)) for name, value in steps.items()
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
