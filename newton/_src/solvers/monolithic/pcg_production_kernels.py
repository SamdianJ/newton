# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-masked PCG kernels; storage and host dispatch live in pcg_production."""

import functools
from typing import Any

import warp as wp

from .pcg_common import block_drift, max_block_drift, residual_converged

wp.set_module_options({"enable_backward": False})

_RUNNING = wp.constant(-1)


_SUCCESS = wp.constant(0)


_NON_POSITIVE_CURVATURE = wp.constant(13)


_BAD_PRECONDITIONED_RESIDUAL = wp.constant(15)


_NONFINITE = wp.constant(16)


_MAX_ITERATIONS = wp.constant(17)


@wp.struct
class _Parameters:
    tolerance: float
    absolute_tolerance: float
    floors: wp.array[float]
    aliases: wp.array[int]
    replacement_threshold: wp.float64
    residual_norm_limits: wp.array[float]


@wp.struct
class _DeviceResult:
    status: int
    enabled: int
    pending_check: int  # Zero, or the deferred breakdown reason.
    restart: int
    recovering: int
    slots: int
    cursor: int
    conditional: int
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
def _initialize_state(state: wp.array[_DeviceResult], error: wp.array[int], scalars: wp.array[Any], slots: int):
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
    zero = scalars.dtype(0.0)
    scalars[0] = zero  # gamma = r.T z
    scalars[1] = zero  # alpha
    scalars[2] = zero  # beta


@wp.kernel
def _initialize_vectors(y: wp.array[Any], r: wp.array[Any], zero_guess: wp.array[int]):
    i = wp.tid()
    if zero_guess[0] != 0:
        y[i] = y.dtype(0.0)
    r[i] = r.dtype(0.0)


@wp.kernel
def _begin_slot(state: wp.array[_DeviceResult], error: wp.array[int], slot: int):
    s = state[0]
    if slot < 0:
        s.cursor += 1
        slot = s.cursor
        state[0] = s
    if s.status != _RUNNING:
        return
    s.restart = 0
    s.recovering = 0
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    state[0] = s


@wp.kernel
def _loop_condition(state: wp.array[_DeviceResult], condition: wp.array[int], limit: int):
    s = state[0]
    s.conditional = 1
    condition[0] = int(s.status == _RUNNING and s.cursor < limit)
    state[0] = s


@wp.kernel
def _masked_scaled_csr_mv(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    scale: wp.array[Any],
    regularization: wp.array[Any],
    x: wp.array[Any],
    out: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    checkpoint: bool,
):
    # Scalar BSR(1x1) has the same row storage as CSR. Do not read matrix
    # entries at all when disabled; merely masking y updates would waste SpMV.
    if not _phase_enabled(state[0], checkpoint):
        return
    i = wp.tid()
    total = out.dtype(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        j = columns[k]
        total += values[k] * (scale[j] * x[j])
    value = scale[i] * total + regularization[0] * x[i]
    out[i] = value
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _masked_q_solve(
    lower: wp.array2d[Any],
    q_start: int,
    r: wp.array[Any],
    temporary: wp.array[Any],
    z: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    n = lower.shape[0]
    for i in range(n):
        value = r[q_start + i]
        for j in range(i):
            value -= lower[i, j] * temporary[q_start + j]
        temporary[q_start + i] = value / lower[i, i]
    for k in range(n):
        i = n - 1 - k
        value = temporary[q_start + i]
        for j in range(i + 1, n):
            value -= lower[j, i] * z[q_start + j]
        value = value / lower[i, i]
        z[q_start + i] = value
        if not wp.isfinite(value):
            wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _masked_rigid_solve(
    factor: wp.array3d[Any],
    rows: wp.array2d[int],
    r: wp.array[Any],
    z: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    body = wp.tid()
    for i in range(6):
        value = r[rows[body, i]]
        for j in range(i):
            value -= factor[body, i, j] * z[rows[body, j]]
        z[rows[body, i]] = value / factor[body, i, i]
    for k in range(6):
        i = 5 - k
        value = z[rows[body, i]]
        for j in range(i + 1, 6):
            value -= factor[body, j, i] * z[rows[body, j]]
        z[rows[body, i]] = value / factor[body, i, i]
        if not wp.isfinite(z[rows[body, i]]):
            wp.atomic_max(error, 0, 16)


@wp.kernel
def _masked_x_solve(
    lower_blocks: wp.array[Any],
    x_start: int,
    r: wp.array[Any],
    z: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    particle = wp.tid()
    lower = lower_blocks[particle]
    base = x_start + 3 * particle
    v = wp.vector(r[base], r[base + 1], r[base + 2])
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


# Fixed-order reductions: each chunk is summed sequentially by one thread, then
# chunks are combined in index order, so results do not depend on scheduling.
REDUCTION_CHUNK = 256


@wp.kernel
def _chunk_products(
    a: wp.array[Any],
    b: wp.array[Any],
    chunk_start: wp.array[int],
    chunk_end: wp.array[int],
    partials: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
):
    if state[0].enabled == 0:
        return
    c = wp.tid()
    total = partials.dtype(0.0)
    for i in range(chunk_start[c], chunk_end[c]):
        av, bv = a[i], b[i]
        if not wp.isfinite(av) or not wp.isfinite(bv):
            wp.atomic_max(error, 0, _NONFINITE)
            return
        total += av * bv
    partials[c] = total


@wp.kernel
def _sum_products(
    partials: wp.array[Any],
    segment_chunks: wp.array[int],
    segment_products: wp.array[Any],
    products: wp.array[Any],
    state: wp.array[_DeviceResult],
):
    """Sum each world's chunks in index order, then the worlds in world order."""
    if state[0].enabled == 0:
        return
    for w in range(segment_products.shape[0]):
        segment = products.dtype(0.0)
        for c in range(segment_chunks[w], segment_chunks[w + 1]):
            segment += partials[c]
        segment_products[w] = segment
    total = products.dtype(0.0)
    for w in range(segment_products.shape[0]):
        total += segment_products[w]
    products[0] = total


@functools.cache
def _make_products_kernel(tile_size, dtype):
    @wp.kernel(module="unique", enable_backward=False)
    def products_kernel(
        a: wp.array[dtype],
        b: wp.array[dtype],
        products: wp.array[dtype],
        state: wp.array[_DeviceResult],
    ):
        if state[0].enabled == 0:
            return
        _, lane = wp.tid()
        total = dtype(0.0)
        for offset in range(0, a.shape[0], tile_size):
            av = wp.tile_load(a, shape=tile_size, offset=offset)
            bv = wp.tile_load(b, shape=tile_size, offset=offset)
            total += wp.tile_sum(wp.tile_map(wp.mul, av, bv))[0]
        # A single CTA overwrites the result: no clear launch or atomic add.
        # Nonfinite inputs propagate to the existing alpha/gamma guards.
        if lane == 0:
            products[0] = total

    return products_kernel


@wp.kernel
def _alpha_guard(
    products: wp.array[Any],
    scalars: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
):
    s = state[0]
    if slot < 0:
        slot = s.cursor
    if s.enabled == 0:
        return
    curvature = products[0]
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    elif not wp.isfinite(curvature):
        s = _failed(s, _NONFINITE, slot)
    elif curvature < products.dtype(0.0):
        s = _failed(s, _NON_POSITIVE_CURVATURE, slot)
    elif curvature == products.dtype(0.0):
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
    y: wp.array[Any],
    r: wp.array[Any],
    p: wp.array[Any],
    ap: wp.array[Any],
    scalars: wp.array[Any],
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
    if slot < 0:
        slot = s.cursor
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
    products: wp.array[Any],
    scalars: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
):
    s = state[0]
    if slot < 0:
        slot = s.cursor
    if s.enabled == 0:
        return
    value = products[0]
    if error[0] != 0:
        s = _failed(s, error[0], slot)
    elif not wp.isfinite(value):
        s = _failed(s, _NONFINITE, slot)
    elif value < products.dtype(0.0):
        s = _failed(s, _BAD_PRECONDITIONED_RESIDUAL, slot)
    elif value == products.dtype(0.0):
        if s.recovering != 0:
            # Fresh nonconverged true residual, freshly preconditioned. A zero
            # dot here is not evidence of convergence (e.g. FP32 underflow).
            s = _failed(s, _BAD_PRECONDITIONED_RESIDUAL, slot)
        else:
            s.enabled = 0
            s.pending_check = _BAD_PRECONDITIONED_RESIDUAL
    else:
        beta = scalars.dtype(0.0)
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
    z: wp.array[Any],
    p: wp.array[Any],
    scalars: wp.array[Any],
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
    if slot < 0:
        slot = s.cursor
    if s.enabled != 0 and error[0] != 0:
        state[0] = _failed(s, error[0], slot)


@wp.kernel
def _clear_norm_work(
    maxima: wp.array[Any],
    sums: wp.array[Any],
    state: wp.array[_DeviceResult],
    initial: bool,
):
    if state[0].status != _RUNNING:
        return
    i = wp.tid()
    # Layout: true(g,q,x), gap(g,q,x), rhs(g,q,x). RHS norms are cached.
    if i < 2 * maxima.shape[0] / 3 or initial:
        maxima[i] = maxima.dtype(0.0)
        sums[i] = sums.dtype(0.0)


@wp.kernel
def _residual_and_maxima(
    rhs: wp.array[Any],
    ay: wp.array[Any],
    recursive: wp.array[Any],
    true_r: wp.array[Any],
    gap: wp.array[Any],
    maxima: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    groups: wp.array[int],
    initial: bool,
):
    if state[0].status != _RUNNING:
        return  # A pending checkpoint must still run even though enabled=0.
    i = wp.tid()
    block = groups[i]
    group_count = maxima.shape[0] / 3
    value = rhs[i] - ay[i]
    difference = rhs.dtype(0.0)
    if not initial:
        difference = recursive[i] - value
    if not wp.isfinite(value) or not wp.isfinite(difference) or not wp.isfinite(rhs[i]):
        wp.atomic_max(error, 0, _NONFINITE)
        return
    true_r[i] = value
    gap[i] = difference
    wp.atomic_max(maxima, 0, wp.abs(value))
    wp.atomic_max(maxima, block, wp.abs(value))
    wp.atomic_max(maxima, group_count, wp.abs(difference))
    wp.atomic_max(maxima, group_count + block, wp.abs(difference))
    if initial:
        wp.atomic_max(maxima, 2 * group_count, wp.abs(rhs[i]))
        wp.atomic_max(maxima, 2 * group_count + block, wp.abs(rhs[i]))


@wp.kernel
def _chunk_norm_sums(
    rhs: wp.array[Any],
    true_r: wp.array[Any],
    gap: wp.array[Any],
    maxima: wp.array[Any],
    chunk_group: wp.array[int],
    chunk_start: wp.array[int],
    chunk_end: wp.array[int],
    partials: wp.array2d[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    initial: bool,
):
    """Sum maxima-scaled squares of one group chunk, for the global and its own group."""
    if state[0].status != _RUNNING or error[0] != 0:
        return
    c = wp.tid()
    group_count = maxima.shape[0] / 3
    zero = maxima.dtype(0.0)
    for k in range(2):
        part = int(0)
        if k == 1:
            part = chunk_group[c]
        # Fresh values: Warp loops would mutate an aliased zero in place.
        t, d, r = maxima.dtype(0.0), maxima.dtype(0.0), maxima.dtype(0.0)
        for i in range(chunk_start[c], chunk_end[c]):
            if maxima[part] > zero:
                v = true_r[i] / maxima[part]
                t += v * v
            if maxima[group_count + part] > zero:
                v = gap[i] / maxima[group_count + part]
                d += v * v
            if initial and maxima[2 * group_count + part] > zero:
                v = rhs[i] / maxima[2 * group_count + part]
                r += v * v
        partials[c, 3 * k] = t
        partials[c, 3 * k + 1] = d
        partials[c, 3 * k + 2] = r


@wp.kernel
def _group_norm_sums(
    partials: wp.array2d[Any],
    group_chunks: wp.array[int],
    sums: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    initial: bool,
):
    """Combine chunk sums in index order; group 0 is global and spans every chunk."""
    if state[0].status != _RUNNING or error[0] != 0:
        return
    g = wp.tid()
    group_count = sums.shape[0] / 3
    # group_chunks[g - 1] .. group_chunks[g] are the chunks of actor group g >= 1.
    column, start, end = int(0), int(0), int(partials.shape[0])
    if g > 0:
        column, start, end = int(3), int(group_chunks[g - 1]), int(group_chunks[g])
    t, d, r = sums.dtype(0.0), sums.dtype(0.0), sums.dtype(0.0)
    for c in range(start, end):
        t += partials[c, column]
        d += partials[c, column + 1]
        r += partials[c, column + 2]
    sums[g] = t
    sums[group_count + g] = d
    if initial:
        sums[2 * group_count + g] = r


@wp.kernel
def _finish_norms(
    maxima: wp.array[Any],
    sums: wp.array[Any],
    norms: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    initial: bool,
):
    if state[0].status != _RUNNING or error[0] != 0:
        return
    i = wp.tid()
    if i < 2 * norms.shape[0] / 3 or initial:
        value = maxima[i] * wp.sqrt(sums[i])
        norms[i] = value
        if not wp.isfinite(value):
            wp.atomic_max(error, 0, _NONFINITE)


@wp.kernel
def _checkpoint_decision(
    norms: wp.array[Any],
    state: wp.array[_DeviceResult],
    error: wp.array[int],
    params: _Parameters,
    slot: int,
    initial: bool,
    final: bool,
):
    s = state[0]
    if slot < 0:
        slot = s.cursor
        final = slot == s.slots
    if s.status != _RUNNING:
        return
    s.true_residual_checks += 1
    if error[0] != 0:
        state[0] = _failed(s, error[0], slot)
        return
    ratios = wp.vec3(0.0)
    drifts = wp.vec3(0.0)
    valid = bool(True)
    group_count = params.floors.shape[0]
    max_drift64 = max_block_drift(norms, group_count, params.floors)
    for k in range(group_count):
        if not wp.isfinite(norms[k] / wp.max(norms[2 * group_count + k], norms.dtype(params.floors[k]))):
            valid = False
    for k in range(3):
        group = params.aliases[k]
        if group >= 0:
            ratios[k] = float(norms[group] / wp.max(norms[2 * group_count + group], norms.dtype(params.floors[group])))
            drift64 = block_drift(norms[group_count + group], norms[group], params.floors[group])
            drifts[k] = float(wp.min(drift64, wp.float64(3.4028234663852886e38)))
    if not valid or not wp.isfinite(max_drift64):
        state[0] = _failed(s, _NONFINITE, slot)
        return
    s.ratios = ratios
    s.drifts = drifts
    s.last_true_slot = slot
    s.residual_current = 1
    if residual_converged(
        norms,
        2 * group_count,
        params.floors,
        params.tolerance,
        params.absolute_tolerance,
        params.residual_norm_limits,
    ):
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
    true_r: wp.array[Any],
    r: wp.array[Any],
    state: wp.array[_DeviceResult],
):
    s = state[0]
    if s.enabled == 0 or s.restart == 0:
        return
    i = wp.tid()
    r[i] = true_r[i]


@wp.kernel
def _recover_candidate(
    scale: wp.array[Any],
    y: wp.array[Any],
    candidate: wp.array[Any],
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
    if s.conditional != 0:
        s.slots = s.cursor
    state[0] = s


@wp.kernel
def _publish_delta(candidate: wp.array[Any], delta: wp.array[Any], state: wp.array[_DeviceResult]):
    if state[0].output_valid == 0:
        return
    i = wp.tid()
    delta[i] = candidate[i]


@wp.kernel
def _study_mask(state: wp.array[_DeviceResult], active: wp.array[int]):
    active[0] = state[0].enabled
