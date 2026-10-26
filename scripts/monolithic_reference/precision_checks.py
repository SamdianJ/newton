# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-sided precision measurements: rounding coverage and small-load resolution.

A rounding allowance can fail in two directions. Too small, real computation
noise sits above it and the step stalls; too large, a small physical load sits
below it and the step converges without responding. Each helper here measures
one of the quantities that keep those apart, so a check can assert both sides:

- :func:`split_force_error` separates a residual's step-constant bias, its step
  response error and its iteration noise at a shared representable state;
- :class:`SolverTrace` records every iterate's row tolerances and every
  linear-direction attempt of real solver steps;
- :func:`load_margin` compares a load with the rounding allowance on its rows.

These are diagnostics. They never change solver arithmetic, and
:class:`SolverTrace` restores everything it wraps on exit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from newton._src.solvers.monolithic import iteration

Evaluate = Callable[[np.ndarray], np.ndarray]


def split_force_error(
    evaluate32: Evaluate,
    evaluate64: Evaluate,
    step: np.ndarray,
    *,
    rows: np.ndarray | None = None,
    samples: int = 12,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split an FP32 residual's error against matched FP64 arithmetic.

    Both evaluators receive the same FP32-representable step increment and
    return the residual over the same rows in float64. ``evaluate64`` must use
    the FP32 inputs promoted to FP64, so the difference is arithmetic only.

    Args:
        evaluate32: FP32 residual of a step increment.
        evaluate64: FP64 residual of the same increment.
        step: Step increment; it is rounded to FP32 first.
        rows: Rows to report, by default every row.
        samples: Number of one-ulp increment perturbations for the noise.
        seed: Seed of the perturbation signs.

    Returns:
        Per-row absolute errors: ``bias`` at the zero increment, ``response``
        of the step's change from that bias, ``noise`` as the largest change
        under one-ulp increment perturbations, and ``total`` at the step.
    """
    step = np.asarray(step, dtype=np.float32)
    zero = np.zeros_like(step)
    select = slice(None) if rows is None else np.asarray(rows)
    r32_0, r64_0 = evaluate32(zero), evaluate64(zero.astype(np.float64))
    r32, r64 = evaluate32(step), evaluate64(step.astype(np.float64))
    rng = np.random.default_rng(seed)
    spacing = np.spacing(np.abs(step)).astype(np.float32)
    noise = np.zeros_like(r32)
    for _ in range(samples):
        signs = rng.choice(np.array([-1, 1], dtype=np.float32), size=step.shape)
        trial = (step + signs * spacing).astype(np.float32)
        change32 = evaluate32(trial) - r32
        change64 = evaluate64(trial.astype(np.float64)) - r64
        noise = np.maximum(noise, np.abs(change32 - change64))
    return {
        "bias": np.abs(r32_0 - r64_0)[select],
        "response": np.abs((r32 - r32_0) - (r64 - r64_0))[select],
        "noise": noise[select],
        "total": np.abs(r32 - r64)[select],
    }


def load_margin(load: np.ndarray, allowance: np.ndarray, rows: np.ndarray) -> float:
    """Return the largest rounding allowance on ``rows`` divided by the load there.

    A value below one means the load can be resolved: it exceeds what the
    criterion would accept as rounding. Rows without load are ignored.
    """
    rows = np.asarray(rows)
    magnitude = np.abs(np.asarray(load, dtype=float)[rows])
    loaded = magnitude > 0
    if not loaded.any():
        raise ValueError("No loaded row to resolve")
    return float(np.max(np.asarray(allowance, dtype=float)[rows][loaded] / magnitude[loaded]))


def relative_error(value: float, reference: float) -> float:
    """Relative error of one scalar response; a zero reference requires a zero value."""
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return abs(value - reference) / abs(reference)


@dataclasses.dataclass
class StepTrace:
    """Iterates and linear-direction attempts of one solver step."""

    iterates: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    attempts: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def updates(self) -> int:
        """Accepted Newton updates: assessed iterates after the first."""
        return max(0, len(self.iterates) - 1)

    @property
    def regularized(self) -> bool:
        """Whether any accepted direction of the step needed a regularization."""
        return any(a["accepted"] and a["regularization"] > 0 for a in self.attempts)


class SolverTrace:
    """Record each iterate's row tolerances and each direction attempt of one solver.

    Use as a context manager around :meth:`step` calls. ``rows`` selects the
    complete-residual rows whose residual, tolerance, physical limit and
    rounding magnitude are kept per iterate (for example the loaded rows).
    """

    def __init__(self, solver, rows: np.ndarray | None = None):
        self.solver = solver
        self.rows = None if rows is None else np.asarray(rows)
        self.steps: list[StepTrace] = []
        self._saved = None

    def __enter__(self):
        if self._saved is not None:
            raise RuntimeError("SolverTrace is already active")
        self._saved = (iteration.measure, iteration.solve_direction)
        measure, solve = self._saved
        trace = self

        def traced_measure(s, residual, reference, scale):
            out = measure(s, residual, reference, scale)
            if s is trace.solver and trace.steps:
                trace._record_iterate(out)
            return out

        def traced_solve(s, *args, **kwargs):
            accepted = solve(s, *args, **kwargs)
            if s is trace.solver and trace.steps:
                attempt = s._variational["linear_direction_attempts"][-1]
                trace.steps[-1].attempts.append(
                    {
                        "iteration": kwargs["iteration"],
                        "retry": kwargs["retry"],
                        "regularization": float(kwargs["regularization"]),
                        "pcg_iterations": int(attempt["iterations"]),
                        "status": attempt["status"],
                        "failure_reason": attempt["failure_reason"],
                        "accepted": bool(accepted),
                    }
                )
            return accepted

        iteration.measure, iteration.solve_direction = traced_measure, traced_solve
        return self

    def __exit__(self, *exc):
        iteration.measure, iteration.solve_direction = self._saved
        self._saved = None
        return False

    def step(self, advance: Callable[[], Any]):
        """Run one solver step through ``advance`` and return its trace."""
        if self._saved is None:
            raise RuntimeError("SolverTrace.step requires an active context")
        self.steps.append(StepTrace())
        advance()
        return self.steps[-1]

    def _record_iterate(self, assessment):
        criterion = self.solver._criterion
        residual = self.solver._complete_r.numpy().astype(float)
        tau = criterion.tau.numpy().astype(float)
        record = {
            "converged": bool(assessment.converged),
            "worst_ratio": float(np.max(np.abs(residual) / tau, initial=0.0)),
            "residual_inf": float(np.max(np.abs(residual), initial=0.0)),
        }
        if self.rows is not None:
            physical = criterion.physical.numpy().astype(float)
            allowance = (criterion.noise.numpy() * criterion.magnitude.numpy()).astype(float)
            record.update(
                residual=residual[self.rows],
                tau=tau[self.rows],
                physical=physical[self.rows],
                allowance=allowance[self.rows],
            )
        self.steps[-1].iterates.append(record)
