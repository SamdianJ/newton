# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed-topology, device-masked PCG for the monolithic linear workspace.

The synchronous adapter at the end of this file implements the production
dispatch from ``solve_pcg``; diagnostic retains its separate host policy.
``MaskedPcgPlan.enqueue_solve()`` is the capture-safe part: no host readback, allocation,
data-dependent Python branch, dynamic graph node, or Python early exit.

One enable mask covers the *entire coupled system*, not its q/x blocks.
All expensive kernels, including the CSR/BSR(1x1) matvec and block solves, honor
that mask. Scheduled true-residual kernels use RUNNING instead: a zero recursive
residual may suspend updates while awaiting the next static checkpoint.

Binding/storage and factor validity must be checked outside capture. The matrix
is the workspace's assembled scalar BSR, and the factors are its existing dense
q and per-particle 3x3 Cholesky blocks. This module does not assemble or factor.
"""

import enum
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from .linear import MonolithicLinearSolveResult, MonolithicLinearStatus, MonolithicPcgConfig, MonolithicPcgWarmStart
from .pcg_common import block_drift, max_block_drift

PCG_POLICY = "masked_smart_restart_v1"

wp.set_module_options({"enable_backward": False})

# Terminal codes intentionally match MonolithicLinearStatus in linear.py.
_RUNNING = wp.constant(-1)
_SUCCESS = wp.constant(0)
_NON_POSITIVE_CURVATURE = wp.constant(13)
_BAD_PRECONDITIONED_RESIDUAL = wp.constant(15)
_NONFINITE = wp.constant(16)
_MAX_ITERATIONS = wp.constant(17)


class ProductionStatus(enum.IntEnum):
    RUNNING = -1
    SUCCESS = 0
    NON_POSITIVE_CURVATURE = 13
    NON_POSITIVE_PRECONDITIONED_RESIDUAL = 15
    NONFINITE_ITERATION = 16
    MAX_ITERATIONS = 17


@dataclass(frozen=True)
class MaskedPcgConfig:
    maximum_iterations: int = 200
    true_residual_interval: int = 10
    linear_tolerance: float = 1.0e-4
    residual_floor_global: float = 1.0e-12
    residual_floor_q: float = 1.0e-12
    residual_floor_x: float = 1.0e-12
    residual_replacement_threshold: float = 0.1

    def __post_init__(self):
        for name in ("maximum_iterations", "true_residual_interval"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value >= 2**31:
                raise ValueError(f"{name} must fit int32")
        positive = {"linear_tolerance", "residual_floor_global", "residual_floor_q", "residual_floor_x"}
        for name in positive | {
            "residual_replacement_threshold",
        }:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} must be a real scalar")
            if not math.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
                raise ValueError(f"{name} must be nonnegative and finite in float32")
            if name in positive and value < np.finfo(np.float32).smallest_subnormal:
                raise ValueError(f"{name} must be positive and representable in float32")

    @classmethod
    def from_monolithic(cls, config):
        return cls(**{name: getattr(config, name) for name in cls.__dataclass_fields__})


@wp.struct
class _Parameters:
    tolerance: float
    floors: wp.vec3
    replacement_threshold: wp.float64


@wp.struct
class _DeviceResult:
    status: int
    enabled: int
    pending_check: int  # Zero, or the deferred breakdown reason.
    restart: int
    recovering: int
    slots: int
    effective_iterations: int
    true_residual_checks: int
    residual_replacements: int
    last_true_slot: int
    convergence_slot: int
    failure_slot: int
    residual_current: int
    output_valid: int
    ratios: wp.vec3
    drifts: wp.vec3


@dataclass(frozen=True)
class MaskedPcgResult:
    status: ProductionStatus
    enabled: bool
    pending_check: int
    iteration_slots: int
    iterations: int
    true_residual_checks: int
    residual_replacements: int
    last_true_slot: int
    convergence_slot: int
    failure_slot: int
    residual_matches_y: bool
    output_valid: bool
    ratios: tuple[float, float, float]
    current_drifts: tuple[float, float, float]


@wp.func
def _failed(s: _DeviceResult, code: int, slot: int):
    # Only controller kernels write state. Parallel kernels write error[0].
    s.status = code
    s.enabled = 0
    s.pending_check = 0
    s.restart = 0
    s.failure_slot = slot
    s.output_valid = 0
    return s


@wp.func
def _phase_enabled(s: _DeviceResult, checkpoint: bool):
    if checkpoint:
        return s.status == _RUNNING
    return s.status == _RUNNING and s.enabled != 0


@wp.kernel
def _initialize_state(state: wp.array[_DeviceResult], error: wp.array[int], scalars: wp.array[float], slots: int):
    s = _DeviceResult()
    s.status = _RUNNING
    s.enabled = 1
    s.slots = slots
    s.last_true_slot = -1
    s.convergence_slot = -1
    s.failure_slot = -1
    s.ratios = wp.vec3(wp.nan)
    s.drifts = wp.vec3(wp.nan)
    state[0] = s
    error[0] = 0
    scalars[0] = 0.0  # gamma = r.T z
    scalars[1] = 0.0  # alpha
    scalars[2] = 0.0  # beta


@wp.kernel
def _initialize_vectors(y: wp.array[float], r: wp.array[float], zero_guess: wp.array[int]):
    i = wp.tid()
    if zero_guess[0] != 0:
        y[i] = 0.0
    r[i] = 0.0


@wp.kernel
def _begin_slot(state: wp.array[_DeviceResult], error: wp.array[int], slot: int):
    s = state[0]
    if s.status != _RUNNING:
        return
    s.restart = 0
    s.recovering = 0
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    state[0] = s


@wp.kernel
def _masked_scaled_csr_mv(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    scale: wp.array[float],
    regularization: wp.array[float],
    x: wp.array[float],
    out: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    checkpoint: bool,
):
    # Scalar BSR(1x1) has the same row storage as CSR. Do not read matrix
    # entries at all when disabled; merely masking y updates would waste SpMV.
    if not _phase_enabled(state[0], checkpoint):
        return
    i = wp.tid()
    total = float(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        total += values[k] * (scale[j] * x[j])
    value = scale[i] * total + regularization[0] * x[i]
    out[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _masked_q_solve(
    lower: wp.array2d[float],
    r: wp.array[float],
    temporary: wp.array[float],
    z: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    n = lower.shape[0]
    for i in range(n):
        value = r[i]
        for j in range(i):
            value -= lower[i, j] * temporary[j]
        temporary[i] = value / lower[i, i]
    for k in range(n):
        i = n - 1 - k
        value = temporary[i]
        for j in range(i + 1, n):
            value -= lower[j, i] * z[j]
        value = value / lower[i, i]
        z[i] = value
        if not wp.isfinite(value):
            wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _masked_x_solve(
    lower_blocks: wp.array[wp.mat33],
    q_count: int,
    r: wp.array[float],
    z: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    particle = wp.tid()
    lower = lower_blocks[particle]
    base = q_count + 3 * particle
    v = wp.vec3(0.0)
    for i in range(3):
        value = r[base + i]
        for j in range(i):
            value -= lower[i, j] * v[j]
        v[i] = value / lower[i, i]
    for k in range(3):
        i = 2 - k
        value = v[i]
        for j in range(i + 1, 3):
            value -= lower[j, i] * v[j]
        v[i] = value / lower[i, i]
    for i in range(3):
        z[base + i] = v[i]
        if not wp.isfinite(v[i]):
            wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _clear_products(products: wp.array[float], state: wp.array[_DeviceResult]):
    if state[0].enabled == 0:
        return
    products[0] = 0.0


@wp.kernel
def _masked_products(
    a: wp.array[float],
    b: wp.array[float],
    products: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    i = wp.tid()
    av, bv = a[i], b[i]
    if not wp.isfinite(av) or not wp.isfinite(bv):
        wp.atomic_max(error, 0, _NONFINITE)
        return
    # Retain the existing float32 inner products; changing reduction precision
    # is a separate numerical/performance change, not a graph requirement.
    wp.atomic_add(products, 0, av * bv)


@wp.kernel
def _alpha_guard(
    products: wp.array[float],
    scalars: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
):
    s = state[0]
    if s.enabled == 0:
        return
    curvature = products[0]
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    elif not wp.isfinite(curvature):
        s = _failed(s, _NONFINITE, slot)
    elif curvature < 0.0:
        s = _failed(s, _NON_POSITIVE_CURVATURE, slot)
    elif curvature == 0.0:
        s = _failed(s, _NON_POSITIVE_CURVATURE, slot)
    else:
        alpha = scalars[0] / curvature  # gamma has already been validated.
        if not wp.isfinite(alpha):
            s = _failed(s, _NONFINITE, slot)
        else:
            scalars[1] = alpha
    state[0] = s


@wp.kernel
def _masked_update(
    y: wp.array[float],
    r: wp.array[float],
    p: wp.array[float],
    ap: wp.array[float],
    scalars: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return  # Do not use alpha=0 masking: 0*NaN is still NaN.
    i = wp.tid()
    yi = y[i] + scalars[1] * p[i]
    ri = r[i] - scalars[1] * ap[i]
    if not wp.isfinite(yi) or not wp.isfinite(ri):
        wp.atomic_max(error, 0, _NONFINITE)
        return
    y[i] = yi
    r[i] = ri


@wp.kernel
def _finish_update(state: wp.array[_DeviceResult], error: wp.array[int], slot: int):
    s = state[0]
    if s.enabled == 0:
        return
    # A parallel update may be partially written on error; this y is invalid
    # and will never be published to delta. No transactional promise for y.
    s.residual_current = 0
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    else:
        s.effective_iterations += 1
    state[0] = s


@wp.kernel
def _gamma_and_beta_guard(
    products: wp.array[float],
    scalars: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
):
    s = state[0]
    if s.enabled == 0:
        return
    value = products[0]
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    elif not wp.isfinite(value):
        s = _failed(s, _NONFINITE, slot)
    elif value < 0.0:
        s = _failed(s, _BAD_PRECONDITIONED_RESIDUAL, slot)
    elif value == 0.0:
        if s.recovering != 0:
            # Fresh nonconverged true residual, freshly preconditioned. A zero
            # dot here is not evidence of convergence (e.g. FP32 underflow).
            s = _failed(s, _BAD_PRECONDITIONED_RESIDUAL, slot)
        else:
            s.enabled = 0
            s.pending_check = _BAD_PRECONDITIONED_RESIDUAL
    else:
        beta = float(0.0)
        if s.restart == 0:
            beta = value / scalars[0]  # Old gamma is positive and finite by invariant.
        if not wp.isfinite(beta):
            s = _failed(s, _NONFINITE, slot)
        else:
            scalars[0] = value
            scalars[2] = beta
    state[0] = s


@wp.kernel
def _masked_direction(
    z: wp.array[float],
    p: wp.array[float],
    scalars: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    s = state[0]
    if s.enabled == 0:
        return
    i = wp.tid()
    value = z[i]
    if s.restart == 0:
        value += scalars[2] * p[i]
    p[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _finish_direction(state: wp.array[_DeviceResult], error: wp.array[int], slot: int):
    s = state[0]
    if s.enabled != 0 and error[0] != 0:
        state[0] = _failed(s, error[0], slot)


@wp.kernel
def _clear_norm_work(
    maxima: wp.array[float],
    sums: wp.array[float],
    state: wp.array[_DeviceResult],
    initial: bool,
):
    if state[0].status != _RUNNING:
        return
    i = wp.tid()
    # Layout: true(g,q,x), gap(g,q,x), rhs(g,q,x). RHS norms are cached.
    if i < 6 or initial:
        maxima[i] = 0.0
        sums[i] = 0.0


@wp.kernel
def _residual_and_maxima(
    rhs: wp.array[float],
    ay: wp.array[float],
    recursive: wp.array[float],
    true_r: wp.array[float],
    gap: wp.array[float],
    maxima: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    q_count: int,
    initial: bool,
):
    if state[0].status != _RUNNING:
        return  # A pending checkpoint must still run even though enabled=0.
    i = wp.tid()
    block = int(2)
    if i < q_count:
        block = 1
    value = rhs[i] - ay[i]
    difference = float(0.0)
    if not initial:
        difference = recursive[i] - value
    if not wp.isfinite(value) or not wp.isfinite(difference) or not wp.isfinite(rhs[i]):
        wp.atomic_max(error, 0, _NONFINITE)
        return
    true_r[i] = value
    gap[i] = difference
    wp.atomic_max(maxima, 0, wp.abs(value))
    wp.atomic_max(maxima, block, wp.abs(value))
    wp.atomic_max(maxima, 3, wp.abs(difference))
    wp.atomic_max(maxima, 3 + block, wp.abs(difference))
    if initial:
        wp.atomic_max(maxima, 6, wp.abs(rhs[i]))
        wp.atomic_max(maxima, 6 + block, wp.abs(rhs[i]))


@wp.kernel
def _residual_norm_sums(
    rhs: wp.array[float],
    true_r: wp.array[float],
    gap: wp.array[float],
    maxima: wp.array[float],
    sums: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    q_count: int,
    initial: bool,
):
    if state[0].status != _RUNNING or error[0] != 0:
        return
    i = wp.tid()
    block = int(2)
    if i < q_count:
        block = 1
    for k in range(2):
        part = int(0)
        if k == 1:
            part = block
        if maxima[part] > 0.0:
            v = true_r[i] / maxima[part]
            wp.atomic_add(sums, part, v * v)
        if maxima[3 + part] > 0.0:
            v = gap[i] / maxima[3 + part]
            wp.atomic_add(sums, 3 + part, v * v)
        if initial and maxima[6 + part] > 0.0:
            v = rhs[i] / maxima[6 + part]
            wp.atomic_add(sums, 6 + part, v * v)


@wp.kernel
def _finish_norms(
    maxima: wp.array[float],
    sums: wp.array[float],
    norms: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    initial: bool,
):
    if state[0].status != _RUNNING or error[0] != 0:
        return
    i = wp.tid()
    if i < 6 or initial:
        value = maxima[i] * wp.sqrt(sums[i])
        norms[i] = value
        if not wp.isfinite(value):
            wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _checkpoint_decision(
    norms: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
    initial: bool,
    final: bool,
):
    s = state[0]
    if s.status != _RUNNING:
        return
    s.true_residual_checks += 1
    if error[0] != 0:
        state[0] = _failed(s, error[0], slot)
        return
    ratios = wp.vec3(0.0)
    drifts = wp.vec3(0.0)
    valid = bool(True)
    max_drift64 = max_block_drift(
        wp.vec3(norms[3], norms[4], norms[5]), wp.vec3(norms[0], norms[1], norms[2]), params.floors
    )
    for k in range(3):
        ratios[k] = norms[k] / wp.max(norms[6 + k], params.floors[k])
        # The host diagnostic path uses double arithmetic for this decision.
        # Clamp only the report field, not the comparison; a huge finite gap
        # ratio requests replacement instead of overflowing its float32 field.
        drift64 = block_drift(norms[3 + k], norms[k], params.floors[k])
        drifts[k] = float(wp.min(drift64, wp.float64(3.4028234663852886e38)))
        if not wp.isfinite(ratios[k]):
            valid = False
    if not valid or not wp.isfinite(max_drift64):
        state[0] = _failed(s, _NONFINITE, slot)
        return
    s.ratios = ratios
    s.drifts = drifts
    s.last_true_slot = slot
    s.residual_current = 1
    worst = wp.max(ratios[0], wp.max(ratios[1], ratios[2]))
    if worst <= params.tolerance:
        s.status = _SUCCESS
        s.enabled = 0
        s.pending_check = 0
        s.restart = 0
        s.convergence_slot = slot
    elif final:
        s.status = _MAX_ITERATIONS
        s.enabled = 0
        s.pending_check = 0
        s.restart = 0
    elif initial:
        s.enabled = 1
        s.restart = 1  # Initialization is not counted as residual replacement.
    else:
        recovering = s.pending_check != 0
        if recovering or max_drift64 >= params.replacement_threshold:
            s.enabled = 1
            s.pending_check = 0
            s.restart = 1
            s.recovering = int(recovering)
            s.residual_replacements += 1
    state[0] = s


@wp.kernel
def _replace_recursive(
    true_r: wp.array[float],
    r: wp.array[float],
    state: wp.array[_DeviceResult],
):
    s = state[0]
    if s.enabled == 0 or s.restart == 0:
        return
    i = wp.tid()
    r[i] = true_r[i]


@wp.kernel
def _recover_candidate(
    scale: wp.array[float],
    y: wp.array[float],
    candidate: wp.array[float],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].status != _SUCCESS:
        return
    i = wp.tid()
    value = scale[i] * y[i]
    candidate[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _finalize_result(state: wp.array[_DeviceResult], error: wp.array[int]):
    s = state[0]
    if s.status == _SUCCESS:
        if error[0] != 0:
            s = _failed(s, error[0], s.slots)
        else:
            s.output_valid = 1
    elif s.status == _RUNNING:
        # Defensive fallback. The scheduled final checkpoint should decide it.
        s.status = _MAX_ITERATIONS
        s.enabled = 0
    state[0] = s


@wp.kernel
def _publish_delta(candidate: wp.array[float], delta: wp.array[float], state: wp.array[_DeviceResult]):
    if state[0].output_valid == 0:
        return
    i = wp.tid()
    delta[i] = candidate[i]


def _signature(arr):
    return (arr.ptr, arr.shape, arr.strides, str(arr.dtype), str(arr.device))


def _span(arr):
    start = arr.ptr or 0
    return start, start + arr.size * wp.types.type_size_in_bytes(arr.dtype)


def _overlap(a, b):
    if not a.size or not b.size:
        return False
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    return a0 < b1 and b0 < a1


class MaskedPcgPlan:
    """Preallocated plan bound to one scalar-BSR matrix and actor factors.

    Array addresses, layout and config are immutable for this plan/graph.
    Contents may be updated in place between solves. Rebind (construct another
    plan) when a BSR rebuild reallocates storage. Calls on a plan must be
    serialized on one stream; it is not reentrant.
    """

    def __init__(
        self,
        *,
        offsets,
        columns,
        values,
        scale,
        q_cholesky,
        x_cholesky,
        rhs_hat,
        y,
        delta=None,
        config: MaskedPcgConfig | None = None,
    ):
        self.config = MaskedPcgConfig() if config is None else config
        if not isinstance(self.config, MaskedPcgConfig):
            raise TypeError("config must be MaskedPcgConfig")
        self.device = wp.get_device(y.device)
        if self.device.is_capturing:
            raise RuntimeError("Construct/bind the plan outside graph capture")
        self.n = y.size
        self.nq = q_cholesky.shape[0]
        self.nx = x_cholesky.size
        if self.n <= 0 or self.n != self.nq + 3 * self.nx:
            raise ValueError("Expected n = q_dof_count + 3 * dynamic_particle_count > 0")
        arrays = (offsets, columns, values, scale, q_cholesky, x_cholesky, rhs_hat, y)
        for a in arrays:
            if not isinstance(a, wp.array) or a.device != self.device or not a.is_contiguous:
                raise ValueError("All bound arrays must be contiguous on the same device")
        expected = (
            (offsets, wp.int32, (self.n + 1,)),
            (columns, wp.int32, (columns.size,)),
            (values, wp.float32, (values.size,)),
            (scale, wp.float32, (self.n,)),
            (q_cholesky, wp.float32, (self.nq, self.nq)),
            (x_cholesky, wp.mat33, (self.nx,)),
            (rhs_hat, wp.float32, (self.n,)),
            (y, wp.float32, (self.n,)),
        )
        for a, dtype, shape in expected:
            if a.dtype != dtype or a.shape != shape:
                raise ValueError(f"Unexpected dtype/shape: expected {dtype}, {shape}")
        if columns.size != values.size:
            raise ValueError("columns and values must have equal storage capacities")
        self.offsets, self.columns, self.values = offsets, columns, values
        self.scale, self.q_cholesky, self.x_cholesky = scale, q_cholesky, x_cholesky
        self.rhs, self.y = rhs_hat, y
        self.delta = wp.zeros(self.n, dtype=float, device=self.device) if delta is None else delta
        if (
            not isinstance(self.delta, wp.array)
            or self.delta.device != self.device
            or self.delta.dtype != wp.float32
            or self.delta.shape != (self.n,)
            or not self.delta.is_contiguous
        ):
            raise ValueError("delta must be a contiguous same-device float32 vector")
        for out in (self.y, self.delta):
            for other in (*arrays[:-1], self.delta if out is self.y else self.y):
                if _overlap(out, other):
                    raise ValueError("Outputs must not overlap each other or bound input storage")
        self._binding_signature = tuple(map(_signature, (*arrays, self.delta)))
        for name in ("r", "z", "p", "ap", "true_r", "gap", "temporary", "delta_candidate"):
            setattr(self, name, wp.zeros(self.n, dtype=float, device=self.device))
        self.products = wp.zeros(1, dtype=float, device=self.device)
        self.scalars = wp.zeros(3, dtype=float, device=self.device)
        self.maxima = wp.zeros(9, dtype=float, device=self.device)
        self.sums = wp.zeros(9, dtype=float, device=self.device)
        self.norms = wp.zeros(9, dtype=float, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)
        self.state = wp.zeros(1, dtype=_DeviceResult, device=self.device)
        self.regularization = wp.zeros(1, dtype=float, device=self.device)
        self.zero_guess = wp.ones(1, dtype=int, device=self.device)
        self.graph = None
        self._workspace = None
        self._generation = None
        self._stream = self._stream_id()
        self._in_flight = False
        self._prepared_regularization = None
        self._storage = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        self._storage_signature = tuple(map(_signature, self._storage))
        self._schedule_signature = (self.config, self.n, self.nq, self.nx)
        self._layout_digest = self._layout_signature()
        c = self.config
        self.params = _Parameters()
        self.params.tolerance = float(c.linear_tolerance)
        self.params.floors = wp.vec3(
            float(c.residual_floor_global), float(c.residual_floor_q), float(c.residual_floor_x)
        )
        self.params.replacement_threshold = float(c.residual_replacement_threshold)

    @classmethod
    def from_workspace(cls, workspace, rhs_hat, y, *, config, delta=None):
        matrix = workspace.k_global_scalar_bsr
        if not workspace._scaling_valid or not workspace._factor_valid:
            raise ValueError("Binding requires valid scaling and actor factors")
        plan = cls(
            offsets=matrix.offsets,
            columns=matrix.columns,
            values=matrix.values,
            scale=workspace.scale,
            q_cholesky=workspace._q_cholesky,
            x_cholesky=workspace._x_cholesky,
            rhs_hat=rhs_hat,
            y=y,
            delta=delta,
            config=config,
        )
        plan._workspace = workspace
        plan._generation = workspace._generation
        return plan

    def _stream_id(self):
        return wp.get_stream(self.device).cuda_stream if self.device.is_cuda else None

    def _layout_signature(self):
        offsets = self.offsets.numpy()
        end = int(offsets[-1])
        if offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0) or not 0 <= end <= self.values.size:
            raise ValueError("Invalid BSR offsets or capacity")
        columns = self.columns.numpy()[:end]
        if np.any(columns < 0) or np.any(columns >= self.n):
            raise ValueError("Invalid BSR column index")
        return hashlib.sha256(offsets.tobytes() + columns.tobytes()).digest()

    def _validate_storage(self):
        arrays = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        if (
            self._stream_id() != self._stream
            or tuple(map(_signature, arrays)) != self._storage_signature
            or (self.config, self.n, self.nq, self.nx) != self._schedule_signature
        ):
            raise ValueError("Stream, schedule or bound storage changed; construct a new PCG plan")

    def _validate_workspace(self, regularization):
        if self._workspace is not None:
            workspace = self._workspace
            if (
                not workspace._current(self._generation)
                or not workspace._scaling_valid
                or not workspace._factor_valid
                or regularization != workspace._lambda
            ):
                raise ValueError("Stale generation, scaling or regularization factors")

    def matches_workspace(self, workspace, rhs_hat, y, config, delta=None):
        """Host metadata only. Call before replay, not from an emitted graph."""
        matrix = workspace.k_global_scalar_bsr
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            workspace.scale,
            workspace._q_cholesky,
            workspace._x_cholesky,
            rhs_hat,
            y,
            self.delta if delta is None else delta,
        )
        return (
            self._generation == workspace._generation
            and self._stream == self._stream_id()
            and config == self.config
            and tuple(map(_signature, arrays)) == self._binding_signature
        )

    def set_solve_parameters(self, *, regularization: float, zero_initial_guess: bool = True):
        """Update device parameters before enqueue/replay; matrix/factors must match."""
        if self.device.is_capturing:
            raise RuntimeError("Set host-provided solve parameters outside capture")
        if (
            isinstance(regularization, (bool, np.bool_))
            or not math.isfinite(regularization)
            or regularization < 0
            or regularization > np.finfo(np.float32).max
        ):
            raise ValueError("regularization must be finite, nonnegative float32")
        if type(zero_initial_guess) is not bool:
            raise TypeError("zero_initial_guess must be bool")
        self._validate_storage()
        if self._in_flight:
            raise RuntimeError("Collect the previous result before reusing this plan")
        self._validate_workspace(regularization)
        if self._workspace is None and self._layout_signature() != self._layout_digest:
            raise ValueError("BSR layout changed; construct a new PCG plan")
        self._prepared_regularization = regularization
        self.regularization.fill_(float(regularization))
        self.zero_guess.fill_(int(zero_initial_guess))

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def _emit_matvec(self, x, *, checkpoint):
        self._launch(
            _masked_scaled_csr_mv,
            self.n,
            [
                self.offsets,
                self.columns,
                self.values,
                self.scale,
                self.regularization,
                x,
                self.ap,
                self.state,
                self.error,
                checkpoint,
            ],
        )

    def _emit_products(self, a, b):
        self._launch(_clear_products, 1, [self.products, self.state])
        self._launch(_masked_products, self.n, [a, b, self.products, self.state, self.error])

    def _emit_direction(self, slot):
        if self.nq:  # Static layout test, not a device-dependent branch.
            self._launch(_masked_q_solve, 1, [self.q_cholesky, self.r, self.temporary, self.z, self.state, self.error])
        if self.nx:
            self._launch(_masked_x_solve, self.nx, [self.x_cholesky, self.nq, self.r, self.z, self.state, self.error])
        self._emit_products(self.r, self.z)
        self._launch(_gamma_and_beta_guard, 1, [self.products, self.scalars, self.state, self.error, self.params, slot])
        self._launch(_masked_direction, self.n, [self.z, self.p, self.scalars, self.state, self.error])
        self._launch(_finish_direction, 1, [self.state, self.error, slot])

    def _emit_checkpoint(self, slot, *, initial=False, final=False):
        self._emit_matvec(self.y, checkpoint=True)
        self._launch(_clear_norm_work, 9, [self.maxima, self.sums, self.state, initial])
        self._launch(
            _residual_and_maxima,
            self.n,
            [self.rhs, self.ap, self.r, self.true_r, self.gap, self.maxima, self.state, self.error, self.nq, initial],
        )
        self._launch(
            _residual_norm_sums,
            self.n,
            [self.rhs, self.true_r, self.gap, self.maxima, self.sums, self.state, self.error, self.nq, initial],
        )
        self._launch(_finish_norms, 9, [self.maxima, self.sums, self.norms, self.state, self.error, initial])
        self._launch(_checkpoint_decision, 1, [self.norms, self.state, self.error, self.params, slot, initial, final])
        self._launch(_replace_recursive, self.n, [self.true_r, self.r, self.state])

    def enqueue_solve(self):
        """Emit a fixed N-slot schedule; safe inside an external CUDA capture.

        Does not capture, launch a graph, read device data or return a host
        convergence result. The caller must prepare, warm kernels and serialize
        use of the bound buffers before entering this low-level boundary.
        Every replay resets all lifecycle state. Neither
        convergence nor error changes the host node sequence.
        """
        nslots = self.config.maximum_iterations
        interval = self.config.true_residual_interval
        self._launch(_initialize_state, 1, [self.state, self.error, self.scalars, nslots])
        self._launch(_initialize_vectors, self.n, [self.y, self.r, self.zero_guess])
        self._emit_checkpoint(0, initial=True)
        self._emit_direction(0)
        for slot in range(1, nslots + 1):
            self._launch(_begin_slot, 1, [self.state, self.error, slot])
            self._emit_matvec(self.p, checkpoint=False)
            self._emit_products(self.p, self.ap)
            self._launch(_alpha_guard, 1, [self.products, self.scalars, self.state, self.error, self.params, slot])
            self._launch(
                _masked_update, self.n, [self.y, self.r, self.p, self.ap, self.scalars, self.state, self.error]
            )
            self._launch(_finish_update, 1, [self.state, self.error, slot])
            if slot % interval == 0 or slot == nslots:
                self._emit_checkpoint(slot, final=slot == nslots)
            if slot != nslots:
                self._emit_direction(slot)
        self._launch(_recover_candidate, self.n, [self.scale, self.y, self.delta_candidate, self.state, self.error])
        self._launch(_finalize_result, 1, [self.state, self.error])
        self._launch(_publish_delta, self.n, [self.delta_candidate, self.delta, self.state])
        return self.state

    def capture(self):
        """Capture once after binding. Recreate the plan after storage reallocation."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA capture requires a CUDA device; enqueue_solve() also works on CPU")
        if self.device.is_capturing:
            raise RuntimeError("Call enqueue_solve(), not capture(), inside an external capture")
        self._validate_storage()
        self._validate_workspace(self._prepared_regularization)
        if self._in_flight or self._prepared_regularization is None:
            raise RuntimeError("Prepare an idle plan before capture")
        wp.load_module(module=__name__, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.enqueue_solve()
        self.graph = capture.graph
        return self.graph

    def replay(self):
        if self.graph is None:
            raise RuntimeError("capture() must be called before replay()")
        self._validate_storage()
        if self._in_flight:
            raise RuntimeError("Collect the previous result before replaying")
        self._validate_workspace(self._prepared_regularization)
        self._in_flight = True
        wp.capture_launch(self.graph)

    def collect_result(self):
        """One packet readback at the PCG boundary. Never call during capture."""
        if self.device.is_capturing:
            raise RuntimeError("collect_result() is a host boundary, not capture-safe")
        item = self.state.numpy()[0]
        self._in_flight = False
        status = ProductionStatus(int(item["status"]))
        if status == ProductionStatus.RUNNING or (
            status == ProductionStatus.SUCCESS and not bool(item["output_valid"])
        ):
            raise RuntimeError("PCG has no terminal result")
        return MaskedPcgResult(
            status=status,
            enabled=bool(item["enabled"]),
            pending_check=int(item["pending_check"]),
            iteration_slots=int(item["slots"]),
            iterations=int(item["effective_iterations"]),
            true_residual_checks=int(item["true_residual_checks"]),
            residual_replacements=int(item["residual_replacements"]),
            last_true_slot=int(item["last_true_slot"]),
            convergence_slot=int(item["convergence_slot"]),
            failure_slot=int(item["failure_slot"]),
            residual_matches_y=bool(item["residual_current"]),
            output_valid=bool(item["output_valid"]),
            ratios=tuple(map(float, item["ratios"])),
            current_drifts=tuple(map(float, item["drifts"])),
        )


def solve_pcg_production(workspace, rhs_hat, y, *, generation, warm_start, config, delta=None, use_graph=False):
    """Synchronous compatibility adapter for existing solve_pcg callers.

    Host validation and the final packet read are deliberately outside the
    captured core. The outer Newton loop remains synchronous. To embed PCG in
    a larger graph, bind a MaskedPcgPlan outside it and call plan.enqueue_solve() instead.
    """

    def invalid(code):
        return MonolithicLinearSolveResult(
            status=code,
            generation=generation,
            iterations=0,
            true_residual_checks=0,
            residual_replacements=0,
            rho=math.nan,
            rho_q=math.nan,
            rho_x=math.nan,
            min_p_ap=math.nan,
            min_r_z=math.nan,
            pcg_policy=PCG_POLICY,
        )

    workspace._last_masked_pcg_result = None
    if not isinstance(config, MonolithicPcgConfig):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if workspace.device.is_capturing:
        raise RuntimeError("Use MaskedPcgPlan.enqueue_solve() inside capture, not the synchronous solve adapter")
    if not workspace._current(generation) or not workspace._scaling_valid:
        return invalid(MonolithicLinearStatus.STALE_GENERATION)
    if not workspace._factor_valid:
        return invalid(MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
    if warm_start not in (MonolithicPcgWarmStart.ZERO, MonolithicPcgWarmStart.SAME_GENERATION):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if not workspace._vector_valid(rhs_hat) or not workspace._vector_valid(y):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if warm_start == MonolithicPcgWarmStart.SAME_GENERATION and workspace._warm_start_key != (generation, y.ptr):
        return invalid(MonolithicLinearStatus.STALE_GENERATION)
    prepare_start = time.perf_counter()
    try:
        prod_config = MaskedPcgConfig.from_monolithic(config)
        if prod_config.maximum_iterations > workspace.capacities.pcg_max_iterations:
            return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
        plan = getattr(workspace, "_masked_pcg_plan", None)
        if plan is None or not plan.matches_workspace(workspace, rhs_hat, y, prod_config, delta):
            plan = MaskedPcgPlan.from_workspace(workspace, rhs_hat, y, config=prod_config, delta=delta)
            workspace._masked_pcg_plan = plan
        plan.set_solve_parameters(
            regularization=workspace._lambda, zero_initial_guess=warm_start == MonolithicPcgWarmStart.ZERO
        )
    except (TypeError, ValueError, AttributeError):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    prepare_seconds = time.perf_counter() - prepare_start
    capture_seconds = 0.0
    graph_rebuilt = False
    workspace._warm_start_key = None
    if use_graph and workspace.device.is_cuda:
        if plan.graph is None:
            capture_start = time.perf_counter()
            plan.capture()
            # Warp lazily instantiates/uploads on first launch. Charge that
            # host cost to graph setup, including the first launch submission.
            plan.replay()
            capture_seconds = time.perf_counter() - capture_start
            graph_rebuilt = True
        else:
            plan.replay()
    else:
        plan._in_flight = True
        plan.enqueue_solve()
    result = plan.collect_result()
    workspace._last_masked_pcg_result = result
    if result.status == ProductionStatus.SUCCESS:
        workspace._warm_start_key = (generation, y.ptr)
    # Keep existing rho fields fresh; checkpoint_ratios retain the last valid check.
    ratios = result.ratios if result.residual_matches_y else (math.nan,) * 3
    return MonolithicLinearSolveResult(
        status=MonolithicLinearStatus(int(result.status)),
        generation=generation,
        iterations=result.iterations,
        true_residual_checks=result.true_residual_checks,
        residual_replacements=result.residual_replacements,
        rho=ratios[0],
        rho_q=ratios[1],
        rho_x=ratios[2],
        min_p_ap=math.nan,
        min_r_z=math.nan,
        warm_start=warm_start,
        initial_guess_norm=math.nan,
        recursive_true_residual_gap=(math.nan,) * 3,
        stagnation_window=None,
        pcg_policy=PCG_POLICY,
        scheduled_slots=result.iteration_slots,
        converged_at_slot=result.convergence_slot,
        last_checked_slot=result.last_true_slot,
        failure_slot=result.failure_slot,
        residual_current=result.residual_matches_y,
        output_valid=result.output_valid,
        checkpoint_ratios=result.ratios,
        prepare_seconds=prepare_seconds,
        capture_seconds=capture_seconds,
        graph_rebuilt=graph_rebuilt,
    )
