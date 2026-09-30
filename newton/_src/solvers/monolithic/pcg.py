# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic PCG kernels, true residual checks and host iteration policy."""

import warp as wp

from .linear_types import (
    MonolithicLinearSolveResult,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .pcg_common import decide_restart, residual_converged
from .preconditioner import _combine_vectors


@wp.kernel
def _pcg_products(a: wp.array[float], b: wp.array[float], products: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    av, bv = a[i], b[i]
    if not wp.isfinite(av) or not wp.isfinite(bv):
        wp.atomic_max(status, 0, 16)
    wp.atomic_add(products, 0, av * bv)
    wp.atomic_add(products, 1, av * av)
    wp.atomic_add(products, 2, bv * bv)


@wp.kernel
def _validate_monolithic_preconditioned_residual(
    products: wp.array[float],
    tolerance: float,
    r_z: wp.array[float],
    status: wp.array[int],
):
    value = products[0]
    if not wp.isfinite(value) or not wp.isfinite(products[1]) or not wp.isfinite(products[2]):
        status[0] = 16
    elif value <= tolerance:
        status[0] = 15
    else:
        r_z[0] = value


@wp.kernel
def _compute_monolithic_pcg_alpha(
    products: wp.array[float],
    r_z: wp.array[float],
    absolute_tolerance: float,
    relative_tolerance: float,
    alpha: wp.array[float],
    status: wp.array[int],
):
    curvature = products[0]
    if not wp.isfinite(curvature) or not wp.isfinite(products[1]) or not wp.isfinite(products[2]):
        status[0] = 16
    elif curvature <= 0.0:
        status[0] = 13
    elif curvature <= absolute_tolerance + relative_tolerance * wp.sqrt(products[1]) * wp.sqrt(products[2]):
        status[0] = 14
    else:
        value = r_z[0] / curvature
        if not wp.isfinite(value):
            status[0] = 16
        else:
            alpha[0] = value


@wp.kernel
def _update_monolithic_pcg_solution_residual(
    y: wp.array[float],
    r: wp.array[float],
    p: wp.array[float],
    ap: wp.array[float],
    alpha: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    yi, ri = y[i] + alpha[0] * p[i], r[i] - alpha[0] * ap[i]
    if not wp.isfinite(yi) or not wp.isfinite(ri):
        wp.atomic_max(status, 0, 16)
        return
    y[i] = yi
    r[i] = ri


@wp.kernel
def _compute_monolithic_pcg_beta(
    r_z: wp.array[float],
    old_r_z: wp.array[float],
    beta: wp.array[float],
    status: wp.array[int],
):
    if status[0] != 0:
        return
    if not wp.isfinite(old_r_z[0]) or old_r_z[0] <= 0.0 or not wp.isfinite(r_z[0]) or r_z[0] <= 0.0:
        status[0] = 15
        return
    value = r_z[0] / old_r_z[0]
    if not wp.isfinite(value):
        status[0] = 16
    else:
        beta[0] = value


@wp.kernel
def _pcg_update_direction(z: wp.array[float], p: wp.array[float], beta: wp.array[float], status: wp.array[int]):
    i = wp.tid()
    if status[0] != 0:
        return
    value = z[i] + beta[0] * p[i]
    if not wp.isfinite(value):
        wp.atomic_max(status, 0, 16)
    p[i] = value


@wp.struct
class _MonolithicPcgPacket:
    status: int
    check: int
    ratios: wp.vec3


@wp.kernel
def _pack_monolithic_pcg_residual(
    status: wp.array[int],
    ratios: wp.array[float],
    norms: wp.array[float],
    floors: wp.vec3,
    rtol: float,
    atol: float,
    limits: wp.vec3,
    packet: wp.array[_MonolithicPcgPacket],
):
    value = _MonolithicPcgPacket()
    value.status = status[0]
    value.ratios = wp.vec3(ratios[0], ratios[1], ratios[2])
    value.check = int(
        residual_converged(
            wp.vec3(norms[0], norms[1], norms[2]), wp.vec3(norms[3], norms[4], norms[5]), floors, rtol, atol, limits
        )
    )
    packet[0] = value


@wp.kernel
def _cache_monolithic_pcg_denominator(norms: wp.array[float], floor: wp.float64, denominator: wp.array[wp.float64]):
    denominator[0] = wp.max(wp.float64(norms[3]), floor)


@wp.kernel
def _pack_monolithic_pcg_check(
    products: wp.array[float],
    status: wp.array[int],
    denominator: wp.array[wp.float64],
    tolerance: wp.float64,
    absolute_tolerance: wp.float64,
    scheduled: bool,
    packet: wp.array[_MonolithicPcgPacket],
):
    value = _MonolithicPcgPacket()
    value.status = status[0]
    # The old host decision used Python doubles, including the configured floor.
    threshold = wp.max(tolerance * denominator[0], absolute_tolerance)
    value.check = int(scheduled or wp.float64(products[0]) <= threshold * threshold)
    packet[0] = value


@wp.kernel
def _record_monolithic_pcg_product(products: wp.array[float], summary: wp.array[float], index: int, first: bool):
    value = products[0]
    old = summary[index]
    # Preserve Python min's NaN ordering and the old curvature initialization.
    if first or (index == 0 and wp.isnan(old)) or value < old:
        summary[index] = value


@wp.kernel
def _record_monolithic_pcg_norms(norms: wp.array[float], summary: wp.array[float], initial: bool, first: bool):
    if initial:
        summary[2] = norms[0]
    else:
        for i in range(3):
            old = float(summary[3 + i])
            value = float(norms[i])
            if first:
                summary[3 + i] = value
            elif not wp.isfinite(old) or not wp.isfinite(value):
                summary[3 + i] = wp.nan
            elif value > old:
                summary[3 + i] = value


@wp.kernel
def _true_residual_maxima(
    ay: wp.array[float],
    rhs: wp.array[float],
    q_count: int,
    residual: wp.array[float],
    maxima: wp.array[float],
    status: wp.array[int],
    compute_rhs: bool,
):
    i = wp.tid()
    value = rhs[i] - ay[i]
    residual[i] = value
    if not wp.isfinite(value) or not wp.isfinite(rhs[i]):
        wp.atomic_max(status, 0, 16)
        return
    block = int(1)
    if i >= q_count:
        block = 2
    wp.atomic_max(maxima, 0, wp.abs(value))
    wp.atomic_max(maxima, block, wp.abs(value))
    if compute_rhs:
        wp.atomic_max(maxima, 3, wp.abs(rhs[i]))
        wp.atomic_max(maxima, block + 3, wp.abs(rhs[i]))


@wp.kernel
def _true_residual_sums(
    residual: wp.array[float],
    rhs: wp.array[float],
    q_count: int,
    maxima: wp.array[float],
    sums: wp.array[float],
    status: wp.array[int],
    compute_rhs: bool,
):
    i = wp.tid()
    if status[0] != 0:
        return
    block = int(1)
    if i >= q_count:
        block = 2
    # Each slice has its own scale so a large x block cannot erase a tiny q block.
    for local in range(2):
        index = int(0)
        if local == 1:
            index = block
        if maxima[index] > 0.0:
            value = residual[i] / maxima[index]
            wp.atomic_add(sums, index, value * value)
        if compute_rhs and maxima[index + 3] > 0.0:
            value = rhs[i] / maxima[index + 3]
            wp.atomic_add(sums, index + 3, value * value)


@wp.kernel
def _compute_true_residual_norms(
    maxima: wp.array[float],
    sums: wp.array[float],
    norms: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    value = maxima[i] * wp.sqrt(sums[i])
    if not wp.isfinite(value) or not wp.isfinite(sums[i]):
        wp.atomic_max(status, 0, 16)
        return
    norms[i] = value


@wp.kernel
def _compute_monolithic_true_residual_ratios(
    norms: wp.array[float],
    floor_global: float,
    floor_q: float,
    floor_x: float,
    ratios: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    if status[0] != 0:
        return
    floor = floor_global
    if i == 1:
        floor = floor_q
    elif i == 2:
        floor = floor_x
    value = norms[i] / wp.max(norms[i + 3], floor)
    ratios[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(status, 0, 16)


def _diagnostic_block_norms(self, vector) -> None:
    """Measure scaled vector norms without altering iterative solver status or scratch."""
    if vector is not self._diagnostic_vector:
        wp.copy(self._diagnostic_vector, vector)
    self._diagnostic_status.zero_()
    self._diagnostic_maxima.zero_()
    self._diagnostic_sums.zero_()
    self._diagnostic_norms.fill_(float("nan"))
    wp.launch(
        _true_residual_maxima,
        vector.size,
        [
            self._diagnostic_zero,
            self._diagnostic_vector,
            self.layout.q_dof_count,
            self._diagnostic_vector,
            self._diagnostic_maxima,
            self._diagnostic_status,
            True,
        ],
        device=self.device,
    )
    wp.launch(
        _true_residual_sums,
        vector.size,
        [
            self._diagnostic_vector,
            self._diagnostic_vector,
            self.layout.q_dof_count,
            self._diagnostic_maxima,
            self._diagnostic_sums,
            self._diagnostic_status,
            True,
        ],
        device=self.device,
    )
    wp.launch(
        _compute_true_residual_norms,
        6,
        [self._diagnostic_maxima, self._diagnostic_sums, self._diagnostic_norms, self._diagnostic_status],
        device=self.device,
    )


def _compute_gap_norms_for_restart(self, gap_vector) -> None:
    """Measure algorithm gap once for restart and optional observations."""
    if gap_vector is not self._gap_vector:
        wp.copy(self._gap_vector, gap_vector)
    self._gap_status.zero_()
    self._gap_maxima.zero_()
    self._gap_sums.zero_()
    self._gap_norms.fill_(float("nan"))
    wp.launch(
        _true_residual_maxima,
        gap_vector.size,
        [
            self._gap_zero,
            self._gap_vector,
            self.layout.q_dof_count,
            self._gap_vector,
            self._gap_maxima,
            self._gap_status,
            True,
        ],
        device=self.device,
    )
    wp.launch(
        _true_residual_sums,
        gap_vector.size,
        [
            self._gap_vector,
            self._gap_vector,
            self.layout.q_dof_count,
            self._gap_maxima,
            self._gap_sums,
            self._gap_status,
            True,
        ],
        device=self.device,
    )
    wp.launch(
        _compute_true_residual_norms,
        6,
        [self._gap_maxima, self._gap_sums, self._gap_norms, self._gap_status],
        device=self.device,
    )


def _should_restart(self, config):
    wp.launch(
        decide_restart,
        1,
        [
            self._gap_norms,
            self._true_norms,
            wp.vec3(config.residual_floor_global, config.residual_floor_q, config.residual_floor_x),
            wp.float64(config.residual_replacement_threshold),
            self._gap_status,
            self._restart_decision,
        ],
        device=self.device,
    )
    decision = int(self._restart_decision.numpy()[0])
    return decision


def _products_of(self, a, b):
    self._products.zero_()
    wp.launch(_pcg_products, a.size, [a, b, self._products, self._status], device=self.device)


def _true_residual(self, rhs, y, config):
    self._status.zero_()
    self.operator.matvec(y, self._ap, self._ap, 1.0, 0.0)
    self._true_sums.zero_()
    self._true_maxima.zero_()
    compute_rhs = not (self._pcg_active and self._rhs_norms_cached)
    if compute_rhs:
        self._true_norms.fill_(float("nan"))
    else:
        self._true_residual_norms.fill_(float("nan"))
    self._ratios.fill_(float("nan"))
    wp.launch(
        _true_residual_maxima,
        y.size,
        [self._ap, rhs, self.layout.q_dof_count, self._true_r, self._true_maxima, self._status, compute_rhs],
        device=self.device,
    )
    wp.launch(
        _true_residual_sums,
        y.size,
        [self._true_r, rhs, self.layout.q_dof_count, self._true_maxima, self._true_sums, self._status, compute_rhs],
        device=self.device,
    )
    wp.launch(
        _compute_true_residual_norms,
        6 if compute_rhs else 3,
        [self._true_maxima, self._true_sums, self._true_norms, self._status],
        device=self.device,
    )
    wp.launch(
        _compute_monolithic_true_residual_ratios,
        3,
        [
            self._true_norms,
            config.residual_floor_global,
            config.residual_floor_q,
            config.residual_floor_x,
            self._ratios,
            self._status,
        ],
        device=self.device,
    )
    if compute_rhs:
        wp.launch(
            _cache_monolithic_pcg_denominator,
            1,
            [self._true_norms, config.residual_floor_global, self._rhs_denominator],
            device=self.device,
        )
        self._rhs_norms_cached = True
    wp.launch(
        _pack_monolithic_pcg_residual,
        1,
        [
            self._status,
            self._ratios,
            self._true_norms,
            wp.vec3(config.residual_floor_global, config.residual_floor_q, config.residual_floor_x),
            config.linear_tolerance,
            config.linear_absolute_tolerance,
            wp.vec3(*(config.residual_norm_limits or (float("inf"),) * 3)),
            self._pcg_packet,
        ],
        device=self.device,
    )
    packet = self._pcg_packet.numpy()[0]
    self._pcg_boundary_status = MonolithicLinearStatus(int(packet["status"]))
    self._pcg_converged = bool(packet["check"])
    return tuple(float(value) for value in packet["ratios"])


def _solve_pcg_diagnostic(self, rhs_hat, y, *, generation, warm_start, config):
    """Run dynamic diagnostic scheduling with independent smart restart scratch."""
    iterations, checks, replacements = 0, 0, 0
    ratios = (float("nan"),) * 3
    min_p_ap, min_r_z = float("nan"), float("nan")
    applied_warm_start, initial_guess_norm, stagnation_window = None, float("nan"), None
    residual_gap, gap_samples = (float("nan"),) * 3, 0
    diagnostic = getattr(self, "_diagnostic_summary", None) is not None
    diagnostic_started = False

    def result(status):
        nonlocal min_p_ap, min_r_z, initial_guess_norm, residual_gap
        self._pcg_active = False
        if diagnostic_started:
            summary = self._diagnostic_summary.numpy()
            min_p_ap, min_r_z, initial_guess_norm = map(float, summary[:3])
            residual_gap = tuple(map(float, summary[3:]))
        return MonolithicLinearSolveResult(
            status,
            generation,
            iterations,
            checks,
            replacements,
            *ratios,
            min_p_ap,
            min_r_z,
            applied_warm_start,
            initial_guess_norm,
            residual_gap,
            stagnation_window,
        )

    def record_product(index, first=False):
        if diagnostic:
            wp.launch(
                _record_monolithic_pcg_product,
                1,
                [self._products, self._diagnostic_summary, index, first],
                device=self.device,
            )

    def record_gap():
        nonlocal gap_samples
        wp.launch(
            _combine_vectors, self._r.size, [self._r, self._true_r, self._gap_vector, 1.0, -1.0], device=self.device
        )
        self._compute_gap_norms_for_restart(self._gap_vector)
        if diagnostic:
            wp.launch(
                _record_monolithic_pcg_norms,
                1,
                [self._gap_norms, self._diagnostic_summary, False, gap_samples == 0],
                device=self.device,
            )
        gap_samples += 1

    if not self._current(generation) or not self._scaling_valid:
        return result(MonolithicLinearStatus.STALE_GENERATION)
    if not self._factor_valid:
        return result(MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
    if (
        not isinstance(config, MonolithicPcgConfig)
        or not self._vector_valid(rhs_hat)
        or not self._vector_valid(y)
        or (rhs_hat.size > 0 and rhs_hat.ptr < y.ptr + y.size * 4 and y.ptr < rhs_hat.ptr + rhs_hat.size * 4)
        or warm_start not in (MonolithicPcgWarmStart.ZERO, MonolithicPcgWarmStart.SAME_GENERATION)
        or config.maximum_iterations > self.capacities.pcg_max_iterations
        or config.maximum_iterations + 2 > self.capacities.pcg_true_residual_check_count
    ):
        return result(MonolithicLinearStatus.INVALID_ARGUMENT)
    if warm_start == MonolithicPcgWarmStart.SAME_GENERATION and self._warm_start_key != (generation, y.ptr):
        return result(MonolithicLinearStatus.STALE_GENERATION)
    self._warm_start_key = None
    self._status.zero_()
    if warm_start == MonolithicPcgWarmStart.ZERO:
        y.zero_()
    applied_warm_start = MonolithicPcgWarmStart(warm_start)
    self._pcg_active = True
    self._rhs_norms_cached = False
    if diagnostic:
        self._diagnostic_summary.fill_(float("nan"))
        diagnostic_started = True
        self._diagnostic_block_norms(y)
        wp.launch(
            _record_monolithic_pcg_norms,
            1,
            [self._diagnostic_norms, self._diagnostic_summary, True, True],
            device=self.device,
        )
    stagnation_window = config.stagnation_window
    ratios = self._true_residual(rhs_hat, y, config)
    checks += 1
    status = self._pcg_boundary_status
    if status != MonolithicLinearStatus.SUCCESS:
        return result(status)
    if self._pcg_converged:
        self._warm_start_key = (generation, y.ptr)
        return result(MonolithicLinearStatus.SUCCESS)
    wp.copy(self._r, self._true_r)
    self.preconditioner.matvec(self._r, self._z, self._z, 1.0, 0.0)
    self._products_of(self._r, self._z)
    wp.launch(
        _validate_monolithic_preconditioned_residual,
        1,
        [self._products, config.preconditioner_positive_tolerance, self._r_z, self._status],
        device=self.device,
    )
    record_product(1, first=True)
    status = self._read_status()
    if status != MonolithicLinearStatus.SUCCESS:
        return result(status)
    wp.copy(self._p, self._z)
    history = [max(ratios)]
    # A check can occur on every iteration when a recursive residual is tiny.
    # The configured check capacity therefore bounds that worst case.
    for iteration in range(1, config.maximum_iterations + 1):
        should_replace = False  # Track whether we replaced residual in this iteration
        self.operator.matvec(self._p, self._ap, self._ap, 1.0, 0.0)
        self._products_of(self._p, self._ap)
        wp.launch(
            _compute_monolithic_pcg_alpha,
            1,
            [
                self._products,
                self._r_z,
                config.curvature_absolute_tolerance,
                config.curvature_relative_tolerance,
                self._alpha,
                self._status,
            ],
            device=self.device,
        )
        record_product(0)
        status = self._read_status()
        if status != MonolithicLinearStatus.SUCCESS:
            break
        wp.launch(
            _update_monolithic_pcg_solution_residual,
            y.size,
            [y, self._r, self._p, self._ap, self._alpha, self._status],
            device=self.device,
        )
        iterations = iteration
        self._products_of(self._r, self._r)
        wp.launch(
            _pack_monolithic_pcg_check,
            1,
            [
                self._products,
                self._status,
                self._rhs_denominator,
                config.linear_tolerance,
                config.linear_absolute_tolerance,
                iteration % config.true_residual_interval == 0 or iteration == config.maximum_iterations,
                self._pcg_packet,
            ],
            device=self.device,
        )
        packet = self._pcg_packet.numpy()[0]
        status = MonolithicLinearStatus(int(packet["status"]))
        if status != MonolithicLinearStatus.SUCCESS:
            break
        check = bool(packet["check"])
        if check:
            ratios = self._true_residual(rhs_hat, y, config)
            record_gap()
            checks += 1
            status = self._pcg_boundary_status
            if status != MonolithicLinearStatus.SUCCESS:
                break
            if self._pcg_converged:
                self._warm_start_key = (generation, y.ptr)
                return result(MonolithicLinearStatus.SUCCESS)
            history.append(max(ratios))
            if (
                len(history) > config.stagnation_window
                and history[-1] >= (1.0 - config.stagnation_minimum_reduction) * history[-1 - config.stagnation_window]
            ):
                status = MonolithicLinearStatus.STAGNATION
                break

            decision = self._should_restart(config)
            if decision < 0:
                status = MonolithicLinearStatus.NONFINITE_ITERATION
                break
            should_replace = bool(decision)

            if should_replace:
                wp.copy(self._r, self._true_r)
                replacements += 1
        if iteration == config.maximum_iterations:
            status = MonolithicLinearStatus.MAX_ITERATIONS
            break
        wp.copy(self._old_r_z, self._r_z)
        self.preconditioner.matvec(self._r, self._z, self._z, 1.0, 0.0)
        self._products_of(self._r, self._z)
        wp.launch(
            _validate_monolithic_preconditioned_residual,
            1,
            [self._products, config.preconditioner_positive_tolerance, self._r_z, self._status],
            device=self.device,
        )
        record_product(1)
        status = self._read_status()
        if status != MonolithicLinearStatus.SUCCESS:
            break
        # Only restart PCG (reset conjugacy) if we actually replaced the residual
        if check and should_replace:
            # Residual replacement restarts conjugacy against the actual operator.
            wp.copy(self._p, self._z)
        else:
            wp.launch(
                _compute_monolithic_pcg_beta,
                1,
                [self._r_z, self._old_r_z, self._beta, self._status],
                device=self.device,
            )
            wp.launch(_pcg_update_direction, y.size, [self._z, self._p, self._beta, self._status], device=self.device)
            status = self._read_status()
            if status != MonolithicLinearStatus.SUCCESS:
                break
    # Failure diagnostics also come from an extra actual-operator matvec.
    ratios = self._true_residual(rhs_hat, y, config)
    if iterations:
        record_gap()
    checks += 1
    if self._pcg_boundary_status == MonolithicLinearStatus.NONFINITE_ITERATION:
        status = MonolithicLinearStatus.NONFINITE_ITERATION
    return result(status)
