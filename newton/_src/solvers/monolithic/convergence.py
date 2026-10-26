# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-row physical convergence tolerances shared by every producer.

Each residual row ``i`` is converged when ``|r_i| <= tau_i`` with

``tau_i = max(g_u, rtol_u * max_{j in (b, u)} |r0_j|, c_u * eps_T * a_i)``,

where ``u`` is the row's physical unit, ``b`` its convergence block, ``r0`` the
residual at the step's first iterate (the inertial prediction) and ``a_i`` the
row's rounding magnitude (the absolute contributions that were summed into
``r_i``, the tangent-weighted coordinate magnitudes and the operand sizes of
terms that cancel). Rows are never pooled into a norm, so a small local
load cannot be diluted by the size of the model.
"""

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

wp.set_module_options({"fuse_fp": False})


class Unit(enum.IntEnum):
    """Physical unit of one residual row."""

    FORCE = 0
    """Force [N]."""
    TORQUE = 1
    """Torque [N m]."""


@dataclass(frozen=True)
class ConvergencePolicy:
    """Per-unit absolute targets, relative tolerances and rounding multipliers, indexed by :class:`Unit`."""

    # The absolute targets keep the frozen cable L2 residual gates satisfiable row by
    # row (2163 position and 720 angle rows for the finest helix).
    # Rounding multiplier: step-start residuals of component fixtures carry at most
    # 0.83 eps a of FP32 arithmetic error, but replayed stalled FP32 Sharpa steps
    # plateau at 4-8 eps a while closing and 21-31 eps a at the end, on tet rows
    # summing 10-30 terms ((n - 1) eps a bounds such a sum).
    # Relative tolerances: at 1e-4 a step whose load changes accepts 1e-4 of its
    # initial imbalance unresolved, so the stiff precreased box and the prefolded
    # cloth exceed their 1e-6 N force gates; FP64 therefore uses 1e-6. FP32 keeps
    # 1e-4, the setting the rounding multiplier was calibrated with. The tet
    # rounding magnitude now bounds the cancellation in F - I; whether FP32 can
    # return to 1e-6 has not been evaluated.
    absolute: tuple[float, float] = (1e-11, 1e-13)
    """Absolute targets ``g_u`` [N, N m]."""
    relative: tuple[float, float] = (1e-6, 1e-6)
    """Tolerances ``rtol_u`` relative to the block's residual at the step's first iterate."""
    noise: tuple[float, float] = (32.0, 32.0)
    """Multipliers ``c_u`` of ``eps_T * a_i``."""
    relative_fp32: tuple[float, float] = (1e-4, 1e-4)
    """FP32 tolerances ``rtol_u``; ``relative`` applies to FP64."""

    def __post_init__(self):
        for name in ("absolute", "relative", "noise", "relative_fp32"):
            values = getattr(self, name)
            if len(values) != len(Unit) or not all(np.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"Convergence {name} values must be positive and finite per unit")

    def relative_for(self, dtype):
        """Return the relative tolerances of native precision ``dtype``."""
        return self.relative_fp32 if np.dtype(dtype) == np.float32 else self.relative


@dataclass(frozen=True)
class RowGroups:
    """Unit and convergence group of every scalar row."""

    names: tuple[str, ...]
    """Group names, such as ``q_torque``, ``x`` or ``rigid_0_r``."""
    units: np.ndarray
    """Unit of each row, int32."""
    groups: np.ndarray
    """Group index of each row, int32."""


def row_groups(layout, joint_units):
    """Split each layout group by physical unit.

    Args:
        layout: Linear layout providing the actor convergence groups.
        joint_units: :class:`Unit` of each articulation DoF, in local order.
    """
    n = layout.scalar_dof_count
    units = np.full(n, -1, dtype=np.int32)
    groups = np.full(n, -1, dtype=np.int32)
    names = []
    joint_units = np.asarray(joint_units, dtype=np.int32)
    if joint_units.shape != (layout.q_dof_count,):
        raise ValueError("Joint units must match the articulation rows")
    q_units = np.full(n, -1, dtype=np.int32)
    q_units[layout.articulation_indices()] = joint_units
    for kind, start, count in layout.groups:
        if kind == "global" or not count:
            continue
        rows = np.arange(start, start + count)
        if kind == "articulation" or kind.startswith("articulation_"):
            local = q_units[rows]
            prefix = "q" if kind == "articulation" else kind.replace("articulation_", "q_")
            parts = [(f"{prefix}_{unit.name.lower()}", rows[local == unit], unit) for unit in Unit]
        else:
            torque = kind.endswith(("_r", "_theta"))
            parts = [("x" if kind == "tet" else kind, rows, Unit.TORQUE if torque else Unit.FORCE)]
        for name, part, unit in parts:
            if part.size:
                units[part] = unit
                groups[part] = len(names)
                names.append(name)
    if np.any(groups < 0):
        raise ValueError("Every scalar row requires a convergence group")
    return RowGroups(tuple(names), units, groups)


def tolerances(reference, magnitude, rows, policy, dtype):
    """Return per-row tolerances and whether the rounding term decides each one.

    Args:
        reference: Residual ``r0`` at the step's first iterate, in physical units.
        magnitude: Rounding magnitudes ``a``.
        rows: Row units and groups.
        policy: Convergence policy.
        dtype: Native NumPy dtype ``T``.
    """
    reference, magnitude = np.asarray(reference, dtype=dtype), np.asarray(magnitude, dtype=dtype)
    block = np.zeros(len(rows.names), dtype=dtype)
    np.maximum.at(block, rows.groups, np.abs(reference))
    absolute = np.asarray(policy.absolute, dtype=dtype)[rows.units]
    relative = np.asarray(policy.relative_for(dtype), dtype=dtype)[rows.units] * block[rows.groups]
    noise = np.asarray(policy.noise, dtype=dtype)[rows.units] * np.finfo(dtype).eps * magnitude
    physical = np.maximum(absolute, relative)
    # A row whose rounding magnitude is not finite (NaN included) can never pass.
    with np.errstate(invalid="ignore"):
        tau = np.where(np.isfinite(noise), np.maximum(physical, noise), dtype(np.inf))
    return tau, physical


def assess(residual, tau, physical, rows):
    """Return each group's largest ``|r|/tau`` and its count of rounding-limited rows.

    A row is rounding limited when it passes only because of the rounding term. A
    row whose tolerance is not finite (an overflowed rounding magnitude) cannot be
    judged and never passes.
    """
    tau = np.asarray(tau)
    with np.errstate(invalid="ignore"):
        ratio = np.where(np.isfinite(tau), np.abs(residual) / tau, np.inf).astype(tau.dtype)
    worst = np.zeros(len(rows.names), dtype=ratio.dtype)
    np.maximum.at(worst, rows.groups, ratio)
    limited = np.bincount(
        rows.groups,
        weights=((ratio <= 1) & (np.abs(residual) > physical)).astype(np.int64),
        minlength=len(rows.names),
    ).astype(np.int64)
    return worst, limited


def merit(residual, tau):
    """Return the tolerance-normalized RMS residual used by the line search."""
    ratio = np.asarray(residual) / tau
    return np.sqrt(np.dot(ratio, ratio) / ratio.size) if ratio.size else ratio.dtype.type(0)


@wp.func
def abs3(v: Any):
    """Return the componentwise absolute value of a 3-vector."""
    return type(v)(wp.abs(v[0]), wp.abs(v[1]), wp.abs(v[2]))


@wp.func
def axial_absolute(a: Any):
    """Return the summed magnitudes behind each component of ``axial(a - a^T)``."""
    return type(a[0])(
        wp.abs(a[2, 1]) + wp.abs(a[1, 2]), wp.abs(a[0, 2]) + wp.abs(a[2, 0]), wp.abs(a[1, 0]) + wp.abs(a[0, 1])
    )


@wp.kernel
def _group_reference(reference: wp.array[Any], groups: wp.array[int], block: wp.array[Any]):
    i = wp.tid()
    wp.atomic_max(block, groups[i], wp.abs(reference[i]))


@wp.kernel
def _row_tolerance(
    magnitude: wp.array[Any],
    groups: wp.array[int],
    block: wp.array[Any],
    absolute: wp.array[Any],
    relative: wp.array[Any],
    noise: wp.array[Any],
    tau: wp.array[Any],
    physical: wp.array[Any],
):
    i = wp.tid()
    limit = wp.max(absolute[i], relative[i] * block[groups[i]])
    physical[i] = limit
    allowance = noise[i] * magnitude[i]
    # A magnitude that is not finite (NaN included, which wp.max would drop) can never pass.
    tau[i] = type(limit)(wp.inf)
    if wp.isfinite(allowance):
        tau[i] = wp.max(limit, allowance)


@wp.kernel
def _rounding(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    coordinates: wp.array[Any],
    contributions: wp.array[Any],
    magnitude: wp.array[Any],
):
    row = wp.tid()
    total = contributions[row]
    for j in range(offsets[row], offsets[row + 1]):
        total += type(total)(wp.abs(values[j])) * coordinates[columns[j]]
    magnitude[row] = total


@wp.kernel
def _assess_rows(
    residual: wp.array[Any],
    tau: wp.array[Any],
    physical: wp.array[Any],
    groups: wp.array[int],
    worst: wp.array[Any],
    limited: wp.array[int],
    squares: wp.array[Any],
):
    i = wp.tid()
    value = wp.abs(residual[i])
    ratio = value / tau[i]
    # An overflowed rounding magnitude makes tau infinite; such a row cannot pass.
    if not wp.isfinite(tau[i]):
        ratio = type(ratio)(wp.inf)
    wp.atomic_max(worst, groups[i], ratio)
    if ratio <= type(ratio)(1.0) and value > physical[i]:
        wp.atomic_add(limited, groups[i], 1)
    squares[i] = ratio * ratio


class Criterion:
    """Device storage for one solver's row tolerances and assessments."""

    def __init__(self, rows, policy, *, dtype, device):
        self.rows, self.policy = rows, policy
        self.dtype = dtype
        native = np.float64 if dtype == wp.float64 else np.float32
        self.native = native
        n, count = len(rows.units), len(rows.names)
        self.groups = wp.array(rows.groups, dtype=int, device=device)
        self.absolute = wp.array(np.asarray(policy.absolute, dtype=native)[rows.units], dtype=dtype, device=device)
        self.relative = wp.array(
            np.asarray(policy.relative_for(native), dtype=native)[rows.units], dtype=dtype, device=device
        )
        self.noise = wp.array(
            np.asarray(policy.noise, dtype=native)[rows.units] * np.finfo(native).eps, dtype=dtype, device=device
        )
        self.block = wp.zeros(count, dtype=dtype, device=device)
        self.magnitude = wp.zeros(n, dtype=dtype, device=device)
        self.tau = wp.zeros(n, dtype=dtype, device=device)
        self.physical = wp.zeros(n, dtype=dtype, device=device)
        self.worst = wp.zeros(count, dtype=dtype, device=device)
        self.limited = wp.zeros(count, dtype=int, device=device)
        self.squares = wp.zeros(n, dtype=dtype, device=device)
        self.square_sum = wp.zeros(1, dtype=dtype, device=device)

    def buffers(self):
        return (
            self.groups,
            self.absolute,
            self.relative,
            self.noise,
            self.block,
            self.magnitude,
            self.tau,
            self.physical,
            self.worst,
            self.limited,
            self.squares,
            self.square_sum,
        )

    def freeze_reference(self, reference):
        """Record each group's largest residual at the step's first iterate."""
        self.block.zero_()
        wp.launch(_group_reference, reference.size, [reference, self.groups, self.block], device=self.groups.device)

    def round(self, matrix, coordinates, contributions, operator=None):
        """Set rounding magnitudes to the summed absolute contributions plus ``|B| |u|``.

        Args:
            matrix: Assembled physical tangent ``B`` (scalar BSR, unscaled).
            coordinates: Magnitude ``|u|`` of each row's coordinate as represented.
            contributions: Absolute contributions summed into each residual row.
            operator: Element tangents applied outside ``matrix``, whose ``|B| |u|`` is added.
        """
        wp.launch(
            _rounding,
            self.magnitude.size,
            [matrix.offsets, matrix.columns, matrix.values, coordinates, contributions, self.magnitude],
            device=self.groups.device,
        )
        if operator is not None:
            operator.add_abs_coordinates(coordinates, self.magnitude)
        self.update(self.magnitude)

    def update(self, magnitude):
        """Recompute ``tau`` from the current rounding magnitudes."""
        wp.launch(
            _row_tolerance,
            magnitude.size,
            [
                magnitude,
                self.groups,
                self.block,
                self.absolute,
                self.relative,
                self.noise,
                self.tau,
                self.physical,
            ],
            device=self.groups.device,
        )

    def assess(self, residual):
        """Write each group's worst ratio, rounding-limited count and the merit square sum."""
        self.worst.zero_()
        self.limited.zero_()
        wp.launch(
            _assess_rows,
            residual.size,
            [residual, self.tau, self.physical, self.groups, self.worst, self.limited, self.squares],
            device=self.groups.device,
        )
        wp.utils.array_sum(self.squares, out=self.square_sum)
