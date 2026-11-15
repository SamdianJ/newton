# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-cable band Cholesky by block cyclic reduction; the global operator stays coupled.

Each cable's preconditioner block is a band of width 10 in vertex/edge order. Split into 10-row
blocks it is block tridiagonal. Level l eliminates every block k = 2^l, 3 * 2^l, ... of every cable
at once: ``L_o = chol(D_o)``, ``W_L = B_o^T L_o^{-T}`` and ``W_R = B_r L_o^{-T}`` for the coupling
``B_r`` of its right neighbour. Each remaining block then subtracts ``W W^T`` of its left eliminated
neighbour, then of its right one, and takes the coupling ``-W_R W_L^T`` to the block 2^(l+1) to its
left. Block 0 of each cable is factored last. This is the Cholesky factor of the matrix with its
blocks reordered by level, so the preconditioner is the same matrix in exact arithmetic as a serial
band Cholesky, with a different rounding order and O(log n) parallel depth per cable.
"""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .cable_operator import _band
from .material_batch import _DeviceRecords

# Rolled block loops and no adjoints: unrolled 10x10x10 block loops made NVRTC take minutes.
wp.set_module_options({"enable_backward": False})

#: Rows per block; the band half-width, so only neighbouring blocks couple.
BLOCK = wp.constant(10)

# Per-cable factor codes (integer ``atomic_max``): a pivot failure is retried with the next shift.
_PIVOT = wp.constant(1)
_NONFINITE = wp.constant(2)
_INVALID = wp.constant(3)


@cache
def _record_type(dtype):
    @wp.struct
    class Band:
        targets: wp.array[int]
        offsets: wp.array[int]
        kinds: wp.array[int]
        elements: wp.array[int]
        first: wp.array[int]
        second: wp.array[int]
        row_a: wp.array[int]
        row_b: wp.array[int]
        stretch: wp.array3d[dtype]
        bend: wp.array3d[dtype]

    return Band


@wp.kernel
def _assemble_rows(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    scale: wp.array[Any],
    rows: wp.array[int],
    inverse: wp.array[int],
    spans: wp.array2d[int],
    row_actor: wp.array[int],
    band: wp.array[Any],
    template_slots: wp.array[int],
    masked: int,
    assembled: wp.array2d[Any],
    invalid: wp.array[int],
    status: wp.array[int],
):
    # One band row per thread: the element part, then the assembled matrix entries of this
    # row in CSR slot order.
    i = wp.tid()
    actor = row_actor[i]
    start = spans[actor, 0]
    row = rows[i]
    for distance in range(11):
        assembled[i, distance] = band[11 * i + distance]
    for slot in range(offsets[row], offsets[row + 1]):
        column = columns[slot]
        j = inverse[column]
        if j >= start and j <= i:
            if masked != 0:
                # Cable contact: keep only template slots, i.e. whole principal blocks of
                # each contact side; far couplings within one cable are left out. The
                # template covers the cable elements (through ``band``), shape contact on
                # an edge ({x_a, x_{a+1}, theta_e}) and each side of a cable contact; an
                # in-band entry of any other producer would be dropped silently.
                if i - j <= 10:
                    if template_slots[11 * i + i - j] != 0:
                        assembled[i, i - j] = assembled[i, i - j] + values[slot] * scale[row] * scale[column]
            elif i - j <= 10:
                assembled[i, i - j] = assembled[i, i - j] + values[slot] * scale[row] * scale[column]
            elif values[slot] != values.dtype(0.0):
                # Every row is checked, so a failed pivot elsewhere cannot hide an invalid entry.
                invalid[actor] = 1
                wp.atomic_max(status, 0, 1)
                return


# ---- block operations (shared by the level kernels and the serial shift ladder) --------------------
# Loop bounds come from the array shape at run time, which keeps the factor-side loops rolled.


@wp.func
def _block_cholesky(m: wp.array3d[Any], g: int, tolerance: Any) -> int:
    n = m.shape[1]
    for i in range(n):
        for j in range(i + 1):
            value = m[g, i, j]
            for k in range(j):
                value -= m[g, i, k] * m[g, j, k]
            if not wp.isfinite(value):
                return _NONFINITE
            if i == j:
                if value <= tolerance:
                    return _PIVOT
                m[g, i, i] = wp.sqrt(value)
            else:
                m[g, i, j] = value / m[g, j, j]
    return 0


@wp.func
def _block_solve_rows(
    lower: wp.array3d[Any], g: int, coupling: wp.array3d[Any], gc: int, transposed: int, w: wp.array3d[Any]
):
    # w[g] = C L_g^{-T}, with C = coupling[gc] or its transpose: row r solves L w_r = c_r.
    n = w.shape[1]
    for r in range(n):
        for c in range(n):
            value = coupling[gc, r, c]
            if transposed != 0:
                value = coupling[gc, c, r]
            for k in range(c):
                value -= lower[g, c, k] * w[g, r, k]
            w[g, r, c] = value / lower[g, c, c]


@wp.func
def _block_subtract_gram(d: wp.array3d[Any], g: int, w: wp.array3d[Any], gw: int):
    # Lower part of d[g] -= w[gw] w[gw]^T.
    n = d.shape[1]
    for p in range(n):
        for q in range(p + 1):
            value = d[g, p, q]
            for k in range(n):
                value -= w[gw, p, k] * w[gw, q, k]
            d[g, p, q] = value


@wp.func
def _block_couple(coupling: wp.array3d[Any], g: int, right: wp.array3d[Any], left: wp.array3d[Any], go: int):
    # coupling[g] = -W_R[go] W_L[go]^T: block g couples to the left neighbour of go.
    n = coupling.shape[1]
    for p in range(n):
        for q in range(n):
            value = coupling.dtype(0.0)
            for k in range(n):
                value += right[go, p, k] * left[go, q, k]
            coupling[g, p, q] = -value


@wp.func
def _load_block(
    g: int,
    k: int,
    start: int,
    end: int,
    gauge_row: int,
    rows: wp.array[int],
    assembled: wp.array2d[Any],
    shift: Any,
    regularization: Any,
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
):
    # Block k of a cable: the diagonal block (shifted, regularized and gauge-lifted as the serial
    # factor did) and the coupling to block k - 1. Rows past the cable end are identity rows.
    n = d.shape[1]
    for p in range(n):
        for q in range(n):
            d[g, p, q] = d.dtype(0.0)
            coupling[g, p, q] = coupling.dtype(0.0)
        i = start + n * k + p
        if i < end:
            for q in range(p):
                d[g, p, q] = assembled[i, p - q]
            value = assembled[i, 0]
            value = value + regularization + shift * wp.abs(value)
            # Lift only a verified instantaneous null direction in the PRECONDITIONER.
            # The physical matrix/residual are unchanged; the delivered theta increment
            # is projected to zero mean by the common-angle constraint.
            if rows[i] == gauge_row:
                value = value + d.dtype(1.0)
            d[g, p, p] = value
            if k > 0:
                for q in range(p, n):
                    coupling[g, p, q] = assembled[i, n + p - q]
        else:
            d[g, p, p] = d.dtype(1.0)


@wp.func
def _eliminate_odd(
    o: int,
    right: int,
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
    tolerance: Any,
) -> int:
    code = _block_cholesky(d, o, tolerance)
    if code == 0:
        _block_solve_rows(d, o, coupling, o, 1, left_w)
        if right >= 0:
            _block_solve_rows(d, o, coupling, right, 0, right_w)
    return code


@wp.func
def _eliminate_even(
    e: int,
    left_odd: int,
    right_odd: int,
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
):
    if left_odd >= 0:
        _block_subtract_gram(d, e, right_w, left_odd)
    if right_odd >= 0:
        _block_subtract_gram(d, e, left_w, right_odd)
    if left_odd >= 0:
        _block_couple(coupling, e, right_w, left_w, left_odd)


# ---- factor kernels -----------------------------------------------------------------------------------


@wp.kernel
def _begin_factor(invalid: wp.array[int], failed: wp.array[int]):
    actor = wp.tid()
    failed[actor] = 0
    if invalid[actor] != 0:
        failed[actor] = _INVALID


@wp.kernel
def _load_blocks(
    blocks: wp.array2d[int],
    spans: wp.array2d[int],
    rows: wp.array[int],
    gauge_rows: wp.array[int],
    assembled: wp.array2d[Any],
    shifts: wp.array[Any],
    regularization: Any,
    failed: wp.array[int],
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
):
    g = wp.tid()
    actor = blocks[g, 0]
    if failed[actor] != 0:
        return
    _load_block(
        g,
        blocks[g, 1],
        spans[actor, 0],
        spans[actor, 1],
        gauge_rows[actor],
        rows,
        assembled,
        shifts[0],
        regularization,
        d,
        coupling,
    )


@wp.kernel
def _factor_odd(
    jobs: wp.array2d[int],
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
    tolerance: Any,
    failed: wp.array[int],
    status: wp.array[int],
):
    t = wp.tid()
    actor = jobs[t, 0]
    if failed[actor] != 0:
        return
    code = _eliminate_odd(jobs[t, 1], jobs[t, 3], d, coupling, left_w, right_w, tolerance)
    if code == _NONFINITE:
        wp.atomic_max(status, 0, 16)
    if code != 0:
        wp.atomic_max(failed, actor, code)


@wp.kernel
def _factor_even(
    jobs: wp.array2d[int],
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
    failed: wp.array[int],
):
    t = wp.tid()
    if failed[jobs[t, 0]] != 0:
        return
    _eliminate_even(jobs[t, 1], jobs[t, 2], jobs[t, 3], d, coupling, left_w, right_w)


@wp.kernel
def _factor_root(
    roots: wp.array2d[int], d: wp.array3d[Any], tolerance: Any, failed: wp.array[int], status: wp.array[int]
):
    t = wp.tid()
    actor = roots[t, 0]
    if failed[actor] != 0:
        return
    code = _block_cholesky(d, roots[t, 1], tolerance)
    if code == _NONFINITE:
        wp.atomic_max(status, 0, 16)
    if code != 0:
        wp.atomic_max(failed, actor, code)


@wp.kernel
def _finish_factor(
    spans: wp.array2d[int],
    block_spans: wp.array2d[int],
    rows: wp.array[int],
    gauge_rows: wp.array[int],
    assembled: wp.array2d[Any],
    shifts: wp.array[Any],
    first: int,
    regularization: Any,
    tolerance: Any,
    d: wp.array3d[Any],
    coupling: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
    failed: wp.array[int],
    used: wp.array[Any],
    status: wp.array[int],
):
    # One thread per cable. With first = 1 it records the level kernels' attempt 0 or, after a
    # pivot failure, redoes the cable for the next shifts, level by level with the same block
    # functions (so a serial attempt equals the parallel one bit for bit). FP32 sums of stiff
    # stretch entries can leave the band indefinite in a general orientation; only the
    # PRECONDITIONER is then shifted by a growing multiple of its own diagonal, and the operator
    # PCG applies keeps the requested regularization.
    actor = wp.tid()
    code = failed[actor]
    if code == 0 and first > 0:
        used[actor] = shifts[0]
        return
    if code != 0 and code != _PIVOT:
        return
    base, count = block_spans[actor, 0], block_spans[actor, 1]
    start, end = spans[actor, 0], spans[actor, 1]
    for attempt in range(first, shifts.shape[0]):
        shift = shifts[attempt]
        for k in range(count):
            _load_block(base + k, k, start, end, gauge_rows[actor], rows, assembled, shift, regularization, d, coupling)
        result = int(0)
        stride = int(1)
        while stride < count and result == 0:
            k = stride + 0
            while k < count and result == 0:
                right = int(-1)
                if k + stride < count:
                    right = base + k + stride
                result = _eliminate_odd(base + k, right, d, coupling, left_w, right_w, tolerance)
                k += 2 * stride
            if result == 0:
                k = int(0)
                while k < count:
                    left_odd = int(-1)
                    right_odd = int(-1)
                    if k > 0:
                        left_odd = base + k - stride
                    if k + stride < count:
                        right_odd = base + k + stride
                    _eliminate_even(base + k, left_odd, right_odd, d, coupling, left_w, right_w)
                    k += 2 * stride
            stride *= 2
        if result == 0 and count > 0:
            result = _block_cholesky(d, base, tolerance)
        if result == _NONFINITE:
            wp.atomic_max(status, 0, 16)
            return
        if result == 0:
            used[actor] = shift
            return
    wp.atomic_max(status, 0, 12)


# ---- apply kernels ------------------------------------------------------------------------------------


@wp.func
def _forward(lower: wp.array3d[Any], g: int, r: wp.array2d[Any]):
    for c in range(BLOCK):
        value = r[g, c]
        for k in range(c):
            value -= lower[g, c, k] * r[g, k]
        r[g, c] = value / lower[g, c, c]


@wp.func
def _backward(lower: wp.array3d[Any], g: int, r: wp.array2d[Any]):
    for reverse in range(BLOCK):
        c = BLOCK - 1 - reverse
        value = r[g, c]
        for k in range(c + 1, BLOCK):
            value -= lower[g, k, c] * r[g, k]
        r[g, c] = value / lower[g, c, c]


@wp.func
def _subtract_product(r: wp.array2d[Any], g: int, w: wp.array3d[Any], go: int):
    # r[g] -= w[go] r[go]
    for p in range(BLOCK):
        value = r[g, p]
        for k in range(BLOCK):
            value -= w[go, p, k] * r[go, k]
        r[g, p] = value


@wp.func
def _subtract_transposed(r: wp.array2d[Any], g: int, w: wp.array3d[Any], gx: int):
    # r[g] -= w[g]^T r[gx]
    for k in range(BLOCK):
        value = r[g, k]
        for p in range(BLOCK):
            value -= w[g, p, k] * r[gx, p]
        r[g, k] = value


@wp.kernel
def _gather(
    slot_rows: wp.array[int], rows: wp.array[int], rhs: wp.array[Any], r: wp.array2d[Any], active: wp.array[int]
):
    s = wp.tid()
    if active[0] == 0:
        return
    i = slot_rows[s]
    value = r.dtype(0.0)
    if i >= 0:
        value = rhs[rows[i]]
    r[s // BLOCK, s % BLOCK] = value


@wp.kernel
def _forward_odd(jobs: wp.array2d[int], lower: wp.array3d[Any], r: wp.array2d[Any], active: wp.array[int]):
    t = wp.tid()
    if active[0] == 0:
        return
    _forward(lower, jobs[t, 1], r)


@wp.kernel
def _forward_even(
    jobs: wp.array2d[int], left_w: wp.array3d[Any], right_w: wp.array3d[Any], r: wp.array2d[Any], active: wp.array[int]
):
    t = wp.tid()
    if active[0] == 0:
        return
    e, left_odd, right_odd = jobs[t, 1], jobs[t, 2], jobs[t, 3]
    if left_odd >= 0:
        _subtract_product(r, e, right_w, left_odd)
    if right_odd >= 0:
        _subtract_product(r, e, left_w, right_odd)


@wp.kernel
def _solve_root(roots: wp.array2d[int], lower: wp.array3d[Any], r: wp.array2d[Any], active: wp.array[int]):
    t = wp.tid()
    if active[0] == 0:
        return
    g = roots[t, 1]
    _forward(lower, g, r)
    _backward(lower, g, r)


@wp.kernel
def _backward_odd(
    jobs: wp.array2d[int],
    lower: wp.array3d[Any],
    left_w: wp.array3d[Any],
    right_w: wp.array3d[Any],
    r: wp.array2d[Any],
    active: wp.array[int],
):
    t = wp.tid()
    if active[0] == 0:
        return
    o, left, right = jobs[t, 1], jobs[t, 2], jobs[t, 3]
    _subtract_transposed(r, o, left_w, left)
    if right >= 0:
        _subtract_transposed(r, o, right_w, right)
    _backward(lower, o, r)


@wp.kernel
def _scatter(
    row_slots: wp.array[int], rows: wp.array[int], r: wp.array2d[Any], result: wp.array[Any], active: wp.array[int]
):
    i = wp.tid()
    if active[0] == 0:
        return
    s = row_slots[i]
    result[rows[i]] = r[s // BLOCK, s % BLOCK]


def _reduction_tables(spans):
    """Blocks, level jobs ``(cable, block, left, right)`` and row/slot maps of the band spans."""
    blocks, block_spans, roots, odd, even = [], [], [], [], []
    base = 0
    for actor, (start, end) in enumerate(spans):
        count = (end - start + BLOCK - 1) // BLOCK
        block_spans.append((base, count))
        blocks += [(actor, k) for k in range(count)]
        if count:
            roots.append((actor, base))
        stride, level = 1, 0
        while stride < count:
            if len(odd) == level:
                odd.append([])
                even.append([])
            for k in range(stride, count, 2 * stride):
                right = base + k + stride if k + stride < count else -1
                odd[level].append((actor, base + k, base + k - stride, right))
            for k in range(0, count, 2 * stride):
                left = base + k - stride if k > 0 else -1
                right = base + k + stride if k + stride < count else -1
                even[level].append((actor, base + k, left, right))
            stride *= 2
            level += 1
        base += count
    slot_rows = np.full(BLOCK * base, -1, dtype=np.int32)
    row_slots = np.zeros(spans[-1][1] if len(spans) else 0, dtype=np.int32)
    for actor, (start, end) in enumerate(spans):
        first = BLOCK * block_spans[actor][0]
        slot_rows[first : first + end - start] = np.arange(start, end)
        row_slots[start:end] = first + np.arange(end - start)
    return {
        "blocks": blocks,
        "block_spans": block_spans,
        "roots": roots,
        "odd": odd,
        "even": even,
        "slot_rows": slot_rows,
        "row_slots": row_slots,
        "count": base,
    }


class CablePreconditioner:
    """Own fixed band storage and an authenticated per-actor scalar permutation."""

    #: Preconditioner-only diagonal shifts, tried in order until the band factors.
    SHIFTS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

    def __init__(self, layout, order, *, device, dtype=wp.float32, operator=None, contact_template=False):
        expected, spans = [], []
        count = 0
        for kind, _, start, xyz, theta in layout.material_ranges:
            if kind != "cable":
                continue
            spans.append((count, count + xyz + theta))
            expected.extend(range(start, start + xyz + theta))
            count += xyz + theta
        order = np.asarray(order)
        if order.shape != (count,) or order.dtype.kind not in "iu":
            raise ValueError("Cable preconditioning requires an integer scalar permutation")
        for start, end in spans:
            if sorted(order[start:end]) != sorted(expected[start:end]):
                raise ValueError("Cable preconditioner permutation must preserve each actor")
        inverse = np.full(layout.scalar_dof_count, -1, dtype=np.int32)
        inverse[order] = np.arange(count)
        self.device = wp.get_device(device)
        self.rows = wp.array(order, dtype=int, device=device)
        self.inverse = wp.array(inverse, dtype=int, device=device)
        self.spans = wp.array(np.asarray(spans, dtype=np.int32).reshape(-1, 2), dtype=int, device=device)
        self.gauge_rows = wp.full(len(spans), -1, dtype=int, device=device)
        self.dtype = dtype
        # Each band row's element part plus its assembled matrix entries (``band`` keeps the element part).
        self.assembled = wp.zeros((count, 11), dtype=dtype, device=device)
        self.row_actor = wp.array(
            np.concatenate([np.full(end - start, actor, dtype=np.int32) for actor, (start, end) in enumerate(spans)])
            if spans
            else np.empty(0, dtype=np.int32),
            dtype=int,
            device=device,
        )
        self.invalid = wp.zeros(max(len(spans), 1), dtype=int, device=device)
        self.active = wp.ones(1, dtype=int, device=device)
        self.shifts = wp.array(self.SHIFTS, dtype=dtype, device=device)
        self.used_shift = wp.zeros(len(spans), dtype=dtype, device=device)
        self.band = wp.zeros(11 * count, dtype=dtype, device=device)
        self._build_reduction(spans, device, dtype)
        self._operator = operator
        self._band_table = None if operator is None else self._build_band_table(operator, inverse, count, device)
        if contact_template and operator is None:
            raise ValueError("A cable contact preconditioner template requires the cable element operator")
        self.masked = int(bool(contact_template))
        self.template = (
            self._build_template(operator, inverse, count, device)
            if contact_template
            else wp.zeros(1, dtype=int, device=device)
        )
        self._records = (
            None
            if operator is None
            else _DeviceRecords(
                _record_type(dtype), self._views(), [table[0].size for table in self._band_table], device=self.device
            )
        )

    def _build_reduction(self, spans, device, dtype):
        """Allocate the block storage and the per-level job tables of the cyclic reduction."""
        tables = _reduction_tables(spans)

        def table(values, width):
            return wp.array(np.asarray(values, dtype=np.int32).reshape(-1, width), dtype=int, device=device)

        blocks = max(tables["count"], 1)
        self.blocks = table(tables["blocks"], 2)
        self.block_spans = table(tables["block_spans"], 2)
        self.roots = table(tables["roots"], 2)
        self.odd_jobs = [table(level, 4) for level in tables["odd"]]
        self.even_jobs = [table(level, 4) for level in tables["even"]]
        self.slot_rows = wp.array(tables["slot_rows"], dtype=int, device=device)
        self.row_slots = wp.array(tables["row_slots"], dtype=int, device=device)
        # Diagonal blocks (factored in place), couplings, the two eliminated-block multipliers and
        # the apply's right-hand side in block order.
        self.block_factor = wp.zeros((blocks, BLOCK, BLOCK), dtype=dtype, device=device)
        self.block_coupling = wp.zeros((blocks, BLOCK, BLOCK), dtype=dtype, device=device)
        self.block_left = wp.zeros((blocks, BLOCK, BLOCK), dtype=dtype, device=device)
        self.block_right = wp.zeros((blocks, BLOCK, BLOCK), dtype=dtype, device=device)
        self.block_rhs = wp.zeros((blocks, BLOCK), dtype=dtype, device=device)
        self.failed = wp.zeros(max(len(spans), 1), dtype=int, device=device)

    def _views(self):
        return [
            (*table, cable.objective.stretch_b, cable.objective.bend_twist_b)
            for cable, table in zip(self._operator.cables, self._band_table, strict=True)
        ]

    @staticmethod
    def _build_band_table(operator, inverse, count, device):
        """Map each cable's lower band slots to its element entries, in element order."""
        tables = []
        for cable in operator.cables:
            per_slot = {}
            for kind, element, a, b, row_a, row_b in cable.entries():
                i, j = int(inverse[row_a]), int(inverse[row_b])
                if i < 0 or j < 0:
                    raise ValueError("Cable element rows lie outside the cable preconditioner")
                if j > i:
                    continue
                if i - j > 10:
                    raise ValueError("Cable element coupling exceeds the preconditioner band")
                per_slot.setdefault(11 * i + i - j, []).append((kind, element, a, b, row_a, row_b))
            slots = sorted(per_slot)
            offsets = np.zeros(len(slots) + 1, dtype=np.int32)
            offsets[1:] = np.cumsum([len(per_slot[slot]) for slot in slots])
            flat = [entry for slot in slots for entry in per_slot[slot]]
            columns = [np.asarray(c, dtype=np.int32) for c in zip(*flat, strict=True)] if flat else [[]] * 6
            arrays = [np.asarray(slots, dtype=np.int32), offsets, *columns]
            tables.append([wp.array(np.asarray(a, dtype=np.int32), dtype=int, device=device) for a in arrays])
        return tables

    @staticmethod
    def _build_template(operator, inverse, count, device):
        """Band slots a contact-enabled factor reads: the diagonal, element slots and each edge's x/theta stencil."""
        template = np.zeros(11 * count, dtype=np.int32)
        template[11 * np.arange(count)] = 1
        for cable in operator.cables:
            for rows in (cable.edge_rows_host, cable.bend_rows_host):
                for element_rows in rows:
                    slots = inverse[element_rows[element_rows >= 0]]
                    for i in slots:
                        for j in slots:
                            if j <= i and i - j <= 10:
                                template[11 * i + i - j] = 1
        return wp.array(template, dtype=int, device=device)

    def buffers(self):
        """Return every owned array for generation and graph binding."""
        arrays = (
            self.rows,
            self.inverse,
            self.spans,
            self.gauge_rows,
            self.active,
            self.shifts,
            self.used_shift,
            self.band,
            self.template,
            self.assembled,
            self.row_actor,
            self.invalid,
            self.blocks,
            self.block_spans,
            self.roots,
            *self.odd_jobs,
            *self.even_jobs,
            self.slot_rows,
            self.row_slots,
            self.block_factor,
            self.block_coupling,
            self.block_left,
            self.block_right,
            self.block_rhs,
            self.failed,
        )
        if self._records is not None:
            self._records.bind(self._views())
            arrays += self._records.buffers()
        return arrays

    def signature(self):
        """Bind cached plans to owned topology and numerical storage."""
        return tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in self.buffers())

    def factor(self, workspace, tolerance):
        """Factor each cable principal block.

        Without cable contact every nonzero entry of the block must lie in the band. With cable contact
        only template slots are read, so each contact keeps the principal blocks of its two sides.
        """
        matrix = workspace.k_global_scalar_bsr
        self.band.zero_()
        if self._records is not None:
            records = self._records.bind(self._views())
            wp.launch(
                _band,
                self._records.jobs.size,
                [records, self._records.jobs, workspace.scale, self.band],
                device=self.device,
            )
        self.invalid.zero_()
        if self.rows.size:
            wp.launch(
                _assemble_rows,
                self.rows.size,
                [
                    matrix.offsets,
                    matrix.columns,
                    matrix.values,
                    workspace.scale,
                    self.rows,
                    self.inverse,
                    self.spans,
                    self.row_actor,
                    self.band,
                    self.template,
                    self.masked,
                    self.assembled,
                    self.invalid,
                    workspace._status,
                ],
                device=self.device,
            )
        self._factor_blocks(self.dtype(workspace._lambda), self.dtype(tolerance), workspace._status)

    def _factor_blocks(self, regularization, tolerance, status, *, serial=False):
        """Shift 0 by levels for every cable, then the serial ladder for cables whose pivot failed.

        ``serial`` runs every attempt in the per-cable ladder instead (tests compare the two paths).
        """
        device, cables = self.device, self.spans.shape[0]
        factor, coupling, left, right = self.block_factor, self.block_coupling, self.block_left, self.block_right
        wp.launch(_begin_factor, cables, [self.invalid, self.failed], device=device)
        if not serial:
            wp.launch(
                _load_blocks,
                self.blocks.shape[0],
                [
                    self.blocks,
                    self.spans,
                    self.rows,
                    self.gauge_rows,
                    self.assembled,
                    self.shifts,
                    regularization,
                    self.failed,
                    factor,
                    coupling,
                ],
                device=device,
            )
            for odd, even in zip(self.odd_jobs, self.even_jobs, strict=True):
                wp.launch(
                    _factor_odd,
                    odd.shape[0],
                    [odd, factor, coupling, left, right, tolerance, self.failed, status],
                    device=device,
                )
                wp.launch(
                    _factor_even, even.shape[0], [even, factor, coupling, left, right, self.failed], device=device
                )
            wp.launch(
                _factor_root, self.roots.shape[0], [self.roots, factor, tolerance, self.failed, status], device=device
            )
        wp.launch(
            _finish_factor,
            cables,
            [
                self.spans,
                self.block_spans,
                self.rows,
                self.gauge_rows,
                self.assembled,
                self.shifts,
                0 if serial else 1,
                regularization,
                tolerance,
                factor,
                coupling,
                left,
                right,
                self.failed,
                self.used_shift,
                status,
            ],
            device=device,
        )

    def apply(self, rhs, result, *, active=None):
        """Apply the cable factors to disjoint rows of an existing preconditioned vector."""
        device = self.device
        active = self.active if active is None else active
        r = self.block_rhs
        wp.launch(_gather, self.slot_rows.shape[0], [self.slot_rows, self.rows, rhs, r, active], device=device)
        for odd, even in zip(self.odd_jobs, self.even_jobs, strict=True):
            wp.launch(_forward_odd, odd.shape[0], [odd, self.block_factor, r, active], device=device)
            wp.launch(_forward_even, even.shape[0], [even, self.block_left, self.block_right, r, active], device=device)
        wp.launch(_solve_root, self.roots.shape[0], [self.roots, self.block_factor, r, active], device=device)
        for odd in reversed(self.odd_jobs):
            wp.launch(
                _backward_odd,
                odd.shape[0],
                [odd, self.block_factor, self.block_left, self.block_right, r, active],
                device=device,
            )
        wp.launch(_scatter, self.row_slots.shape[0], [self.row_slots, self.rows, r, result, active], device=device)

    def dense_factors(self):
        """Per cable, a dense ``L`` in band order with ``L Lᵀ`` equal to the factored matrix (host; tests).

        ``L`` holds the factored diagonal blocks and the multipliers of each eliminated block; it is
        lower triangular only in the level order, not in band order.
        """
        lower = self.block_factor.numpy()
        left, right = self.block_left.numpy(), self.block_right.numpy()
        jobs = [job.numpy() for job in self.odd_jobs]
        factors = []
        for (start, end), (base, count) in zip(self.spans.numpy(), self.block_spans.numpy(), strict=True):
            size = BLOCK * count
            dense = np.zeros((size, size))
            for k in range(count):
                dense[BLOCK * k : BLOCK * k + BLOCK, BLOCK * k : BLOCK * k + BLOCK] = np.tril(lower[base + k])
            for level in jobs:
                for _, o, eliminated_left, eliminated_right in level[
                    (level[:, 1] >= base) & (level[:, 1] < base + count)
                ]:
                    columns = slice(BLOCK * (o - base), BLOCK * (o - base + 1))
                    a = eliminated_left - base
                    dense[BLOCK * a : BLOCK * a + BLOCK, columns] = left[o]
                    if eliminated_right >= 0:
                        b = eliminated_right - base
                        dense[BLOCK * b : BLOCK * b + BLOCK, columns] = right[o]
            factors.append(dense[: end - start])
        return factors
