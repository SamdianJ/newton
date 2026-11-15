# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable band preconditioner: per-row assembly, factor contracts and status codes on each device."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_preconditioner import _assemble_rows
from newton._src.solvers.monolithic.cable_preconditioner import _factor as _new_factor
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _serial_factor_2826c5a0(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    scale: wp.array[Any],
    rows: wp.array[int],
    inverse: wp.array[int],
    spans: wp.array2d[int],
    gauge_rows: wp.array[int],
    band: wp.array[Any],
    shifts: wp.array[Any],
    template_slots: wp.array[int],
    masked: int,
    regularization: Any,
    tolerance: Any,
    factor: wp.array2d[Any],
    used: wp.array[Any],
    status: wp.array[int],
):
    # The serial factor of 2826c5a0, kept verbatim as the oracle of the per-row assembly split.
    actor = wp.tid()
    start, end = spans[actor, 0], spans[actor, 1]
    for attempt in range(shifts.shape[0]):
        shift = shifts[attempt]
        failed = int(0)
        for i in range(start, end):
            if failed == 0:
                row = rows[i]
                for distance in range(11):
                    factor[i, distance] = band[11 * i + distance]
                for slot in range(offsets[row], offsets[row + 1]):
                    column = columns[slot]
                    j = inverse[column]
                    if j >= start and j <= i:
                        if masked != 0:
                            if i - j <= 10:
                                if template_slots[11 * i + i - j] != 0:
                                    factor[i, i - j] = factor[i, i - j] + values[slot] * scale[row] * scale[column]
                        elif i - j <= 10:
                            factor[i, i - j] = factor[i, i - j] + values[slot] * scale[row] * scale[column]
                        elif values[slot] != values.dtype(0.0):
                            wp.atomic_max(status, 0, 1)
                            return
                factor[i, 0] = factor[i, 0] + regularization + shift * wp.abs(factor[i, 0])
                if row == gauge_rows[actor]:
                    factor[i, 0] = factor[i, 0] + factor.dtype(1.0)
                for j in range(wp.max(start, i - 10), i + 1):
                    if failed == 0:
                        value = factor[i, i - j]
                        for k in range(wp.max(start, i - 10), j):
                            value -= factor[i, i - k] * factor[j, j - k]
                        if not wp.isfinite(value):
                            wp.atomic_max(status, 0, 16)
                            return
                        if i == j:
                            if value <= tolerance:
                                failed = 1
                            else:
                                factor[i, 0] = wp.sqrt(value)
                        else:
                            factor[i, i - j] = value / factor[j, 0]
        if failed == 0:
            used[actor] = shift
            return
    wp.atomic_max(status, 0, 12)


def _synthetic_cables(rng, lengths, *, far, shifted_cable=None):
    """Random SPD cable bands in a permuted global order, split between ``band`` and CSR entries.

    Cables listed in ``far`` also get couplings 12 band rows apart: skipped by a masked factor,
    an invalid argument for an unmasked one.
    """
    count = sum(lengths)
    size = count + 7  # a few rows that belong to no cable
    scale = rng.uniform(0.5, 1.5, size=size)
    permutation = rng.permutation(size)
    rows, outside = permutation[:count].astype(np.int32), int(permutation[count])
    inverse = np.full(size, -1, dtype=np.int32)
    inverse[rows] = np.arange(count)
    spans = np.cumsum([0, *lengths])
    spans = np.column_stack([spans[:-1], spans[1:]]).astype(np.int32)
    band = np.zeros(11 * count)
    triplets = {}
    template = (rng.random(11 * count) < 0.7).astype(np.int32)
    template[11 * np.arange(count)] = 1
    for actor, (start, end) in enumerate(spans):
        n = end - start
        lower = np.tril(rng.normal(size=(n, n)))
        # A lower factor of bandwidth 5 gives an SPD matrix of bandwidth 10.
        lower = np.where(np.subtract.outer(np.arange(n), np.arange(n)) <= 5, lower, 0.0)
        lower[np.diag_indices(n)] = np.abs(lower[np.diag_indices(n)]) + 1.0
        matrix = lower @ lower.T
        if actor == shifted_cable:
            # Push the last pivot to -5e-4 of its remaining diagonal: attempt 0 fails, a 1e-3 shift recovers
            # it (an earlier near-zero pivot would make every later Schur complement indefinite).
            k = n - 1
            pivot = np.linalg.cholesky(matrix[: k + 1, : k + 1])[k, k] ** 2
            matrix[k, k] -= pivot + 5e-4 * (matrix[k, k] - pivot)
        for i in range(n):
            for d in range(min(11, i + 1)):
                j = i - d
                element = 0.5 * matrix[i, j] if rng.random() < 0.5 else 0.0
                band[11 * (start + i) + d] = element
                # The rest as unscaled CSR entries: two parts plus the symmetric (skipped) copy.
                rest = (matrix[i, j] - element) / (scale[rows[start + i]] * scale[rows[start + j]])
                for part in (0.25 * rest, 0.75 * rest):
                    triplets.setdefault((rows[start + i], rows[start + j]), []).append(part)
                if d:
                    triplets.setdefault((rows[start + j], rows[start + i]), []).append(rest)
            if actor in far and i >= 12:
                triplets.setdefault((rows[start + i], rows[start + i - 12]), []).append(0.3)
        # Couplings to rows outside every cable are always skipped.
        if n:
            triplets.setdefault((rows[start], outside), []).append(0.7)
    offsets = np.zeros(size + 1, dtype=np.int32)
    columns, values = [], []
    for row in range(size):
        entries = sorted((c, v) for (r, c), parts in triplets.items() if r == row for v in parts)
        columns += [c for c, _ in entries]
        values += [v for _, v in entries]
        offsets[row + 1] = len(columns)
    gauge = np.full(len(lengths), -1, dtype=np.int32)
    gauge[0] = rows[spans[0, 0] + 1]
    return {
        "offsets": offsets,
        "columns": np.asarray(columns, dtype=np.int32),
        "values": np.asarray(values),
        "scale": scale,
        "rows": rows,
        "inverse": inverse,
        "spans": spans,
        "gauge_rows": gauge,
        "band": band,
        "template": template,
        "row_actor": np.repeat(np.arange(len(lengths)), lengths).astype(np.int32),
    }


def _run_factors(device, dtype, data, masked):
    def array(values, kind):
        return wp.array(np.asarray(values), dtype=kind, device=device)

    count, cables = len(data["rows"]), len(data["spans"])
    shifts = array((0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2), dtype)
    common = {
        "offsets": array(data["offsets"], int),
        "columns": array(data["columns"], int),
        "values": array(data["values"], dtype),
        "scale": array(data["scale"], dtype),
        "rows": array(data["rows"], int),
        "inverse": array(data["inverse"], int),
        "spans": array(data["spans"], int),
        "gauge": array(data["gauge_rows"], int),
        "band": array(data["band"], dtype),
        "template": array(data["template"], int),
    }
    results = []
    for split in (False, True):
        factor = wp.zeros((count, 11), dtype=dtype, device=device)
        used = wp.full(cables, -1.0, dtype=dtype, device=device)
        status = wp.zeros(1, dtype=int, device=device)
        c = common
        if split:
            assembled = wp.zeros((count, 11), dtype=dtype, device=device)
            invalid = wp.zeros(cables, dtype=int, device=device)
            wp.launch(
                _assemble_rows,
                count,
                [
                    c["offsets"],
                    c["columns"],
                    c["values"],
                    c["scale"],
                    c["rows"],
                    c["inverse"],
                    c["spans"],
                    array(data["row_actor"], int),
                    c["band"],
                    c["template"],
                    int(masked),
                    assembled,
                    invalid,
                    status,
                ],
                device=device,
            )
            wp.launch(
                _new_factor,
                cables,
                [
                    c["spans"],
                    c["rows"],
                    c["gauge"],
                    assembled,
                    invalid,
                    shifts,
                    dtype(0.0),  # regularization
                    dtype(0.0),
                    factor,
                    used,
                    status,
                ],
                device=device,
            )
        else:
            wp.launch(
                _serial_factor_2826c5a0,
                cables,
                [
                    c["offsets"],
                    c["columns"],
                    c["values"],
                    c["scale"],
                    c["rows"],
                    c["inverse"],
                    c["spans"],
                    c["gauge"],
                    c["band"],
                    shifts,
                    c["template"],
                    int(masked),
                    dtype(0.0),  # regularization
                    dtype(0.0),
                    factor,
                    used,
                    status,
                ],
                device=device,
            )
        results.append((factor.numpy(), used.numpy(), int(status.numpy()[0])))
    return results


def test_row_assembly_matches_the_serial_factor_bit_for_bit(test, device):
    """Splitting the per-row assembly out of the factor leaves factors, used shifts and status bit-identical."""
    for dtype in (wp.float32, wp.float64):
        for masked in (False, True):
            for seed in (1, 2):
                rng = np.random.default_rng(seed)
                far = range(5) if masked else ()
                data = _synthetic_cables(rng, (37, 1, 24, 0, 51), far=far, shifted_cable=2)
                (old_factor, old_used, old_status), (new_factor, new_used, new_status) = _run_factors(
                    device, dtype, data, masked
                )
                with test.subTest(dtype=dtype.__name__, masked=masked, seed=seed):
                    test.assertEqual(new_status, old_status)
                    np.testing.assert_array_equal(new_used, old_used)
                    np.testing.assert_array_equal(new_factor, old_factor)
                    if not masked:
                        test.assertEqual(new_status, 0)
                    if dtype == wp.float64 and not masked:
                        # The shifted cable needed a shift; the others factored at shift 0. (The masked
                        # template drops entries, so the masked matrix is not the constructed one.)
                        test.assertGreater(float(new_used[2]), 0.0)
                        test.assertEqual(float(new_used[0]), 0.0)


def test_status_codes_of_different_cables_merge_as_before(test, device):
    """Statuses of different cables still merge as 16 > 12 > 1; an out-of-band cable is not factored."""
    cases = (
        # (cables with out-of-band entries, failed pivots, non-finite entries, status before, status now)
        ((0,), (2,), (4,), 16, 16),
        ((0,), (2,), (), 12, 12),
        ((0,), (), (), 1, 1),
        # Both in one cable: the serial factor met the non-finite row (5) before the far one (12);
        # the cable is now rejected by its assembly and never factored.
        ((4,), (), (4,), 16, 1),
    )
    for dtype in (wp.float32, wp.float64):
        for far, negative, nonfinite, before, now in cases:
            data = _synthetic_cables(np.random.default_rng(3), (37, 1, 24, 0, 51), far=far)
            for actor in negative:
                data["band"][11 * data["spans"][actor, 0]] = -1e6
            for actor in nonfinite:
                data["band"][11 * (data["spans"][actor, 0] + 5)] = np.nan
            (_, old_used, old_status), (_, new_used, new_status) = _run_factors(device, dtype, data, masked=False)
            with test.subTest(dtype=dtype.__name__, far=far, negative=negative, nonfinite=nonfinite):
                test.assertEqual(old_status, before)
                test.assertEqual(new_status, now)
                np.testing.assert_array_equal(new_used, old_used)


def _factor_status(device, matrix, order, layout):
    """Assemble ``matrix`` into a bare linear workspace and return the cable factor status."""
    rows, columns = np.nonzero(matrix)
    workspace = MonolithicLinearWorkspace(
        layout,
        MonolithicLinearCapacities(len(rows), 0, 0, 100, 102),
        wp.array([], dtype=int, device=device),
        device=device,
        cable_row_order=order,
    )
    workspace._set_fixed_triplet_pattern(
        global_rows=rows,
        global_columns=columns,
        internal_rows=np.array([], dtype=int),
        internal_columns=np.array([], dtype=int),
    )
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    assembly.global_scalar_triplets.values.assign(matrix[rows, columns].astype(np.float32))
    assert workspace.finalize_assembly(generation=generation).status == MonolithicLinearStatus.SUCCESS
    size = matrix.shape[0]
    scaling = workspace.build_scaling(wp.ones(size, dtype=float, device=device), epsilon_d=1e-12, generation=generation)
    assert scaling == MonolithicLinearStatus.SUCCESS
    return workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)


def test_out_of_band_entry_is_reported_behind_a_failed_pivot(test, device):
    """An out-of-band entry is an invalid argument even when an earlier row fails every shift.

    The serial factor stopped reading a cable at the first row whose pivot failed under every
    shift, so an out-of-band entry further down was reported as a non-positive pivot instead.
    """
    layout = MonolithicLinearLayout(0, 0, actor_ranges=(("cable", 0, 24),), material_ranges=(("cable", 0, 0, 18, 6),))
    order = np.array([[3 * i, 3 * i + 1, 3 * i + 2, 18 + i] for i in range(6)]).ravel()
    matrix = np.eye(24) * 2
    # Scaled, rows 0 and 1 form [[1, 2], [2, 1]]: the second pivot is -3 under every shift.
    matrix[order[1], order[0]] = matrix[order[0], order[1]] = 4.0
    # Permuted positions 0 and 20 of the same cable lie 20 > 10 apart.
    matrix[order[20], order[0]] = matrix[order[0], order[20]] = 0.05
    test.assertEqual(_factor_status(device, matrix, order, layout), MonolithicLinearStatus.INVALID_ARGUMENT)
    # Without the out-of-band entry the same cable is a non-positive pivot.
    matrix[order[20], order[0]] = matrix[order[0], order[20]] = 0.0
    test.assertEqual(_factor_status(device, matrix, order, layout), MonolithicLinearStatus.NON_POSITIVE_PIVOT)


class TestCablePreconditioner(unittest.TestCase):
    pass


devices = get_test_devices()
for _test in (
    test_out_of_band_entry_is_reported_behind_a_failed_pivot,
    test_row_assembly_matches_the_serial_factor_bit_for_bit,
    test_status_codes_of_different_cables_merge_as_before,
):
    add_function_test(TestCablePreconditioner, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
