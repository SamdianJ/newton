# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable band preconditioner: block cyclic reduction against a dense oracle, shift ladder, status codes."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_preconditioner import CablePreconditioner, _assemble_rows
from newton._src.solvers.monolithic.linear import MonolithicLinearWorkspace
from newton._src.solvers.monolithic.linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

LENGTHS = (37, 1, 24, 0, 51, 10, 20)
OUTSIDE = 6  # rows of a cloth actor that belong to no cable


def _layout(lengths):
    actors, materials, cursor = [], [], 0
    for source, n in enumerate(lengths):
        xyz = 3 * (n // 3)
        actors.append(("cable", cursor, n))
        materials.append(("cable", source, cursor, xyz, n - xyz))
        cursor += n
    actors.append(("cloth", cursor, OUTSIDE))
    materials.append(("cloth", 0, cursor, OUTSIDE, 0))
    return MonolithicLinearLayout(0, 0, actor_ranges=tuple(actors), material_ranges=tuple(materials))


def _synthetic(rng, lengths, *, far=(), element_only=(), shifted=None):
    """Strictly diagonally dominant cable bands in band order, split between ``band`` and a CSR.

    Each cable's global rows are its actor block; its band order is a random permutation of them.
    Dropping any symmetric pair of off-diagonal entries keeps such a matrix positive definite, so a
    contact template cannot make it indefinite. ``far`` cables get couplings 12 band rows apart,
    ``element_only`` cables keep all of their entries in ``band``, and the ``shifted`` cable's last
    pivot is pushed below zero so that shift 0 fails and a larger shift recovers it.
    """
    count = sum(lengths)
    size = count + OUTSIDE
    scale = rng.uniform(0.5, 1.5, size=size)
    spans = np.cumsum([0, *lengths])
    spans = np.column_stack([spans[:-1], spans[1:]]).astype(np.int32)
    order = np.concatenate([start + rng.permutation(end - start) for start, end in spans] or [[]]).astype(np.int32)
    inverse = np.full(size, -1, dtype=np.int32)
    inverse[order] = np.arange(count)
    matrices, band, triplets = [], np.zeros((count, 11)), {}
    template = (rng.random((count, 11)) < 0.7).astype(np.int32)
    template[:, 0] = 1
    for actor, (start, end) in enumerate(spans):
        n = end - start
        matrix = np.zeros((n, n))
        for d in range(1, 11):
            values = 0.3 * rng.normal(size=max(n - d, 0))
            matrix[np.arange(d, n), np.arange(n - d)] = values
            matrix[np.arange(n - d), np.arange(d, n)] = values
        matrix[np.diag_indices(n)] = np.abs(matrix).sum(axis=1) + rng.uniform(0.5, 1.5, size=n)
        if actor == shifted:
            k = n - 1
            pivot = np.linalg.cholesky(matrix[: k + 1, : k + 1])[k, k] ** 2
            matrix[k, k] -= pivot + 5e-4 * (matrix[k, k] - pivot)
        matrices.append(matrix)
        for i in range(n):
            for d in range(min(11, i + 1)):
                j = i - d
                element = matrix[i, j] if actor in element_only else (0.5 * matrix[i, j] if rng.random() < 0.5 else 0.0)
                band[start + i, d] = element
                rest = (matrix[i, j] - element) / (scale[order[start + i]] * scale[order[start + j]])
                if rest:
                    for part in (0.25 * rest, 0.75 * rest):
                        triplets.setdefault((order[start + i], order[start + j]), []).append(part)
                    if d:
                        triplets.setdefault((order[start + j], order[start + i]), []).append(rest)
            if actor in far and i >= 12:
                triplets.setdefault((order[start + i], order[start + i - 12]), []).append(0.3)
        if n:
            triplets.setdefault((order[start], count), []).append(0.7)  # a row outside every cable
    rows_, columns, values = [], [], []
    for (r, c), parts in sorted(triplets.items()):
        for v in parts:
            rows_.append(r)
            columns.append(c)
            values.append(v)
    offsets = np.zeros(size + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(np.bincount(np.asarray(rows_, dtype=np.int64), minlength=size))
    gauge = np.full(len(lengths), -1, dtype=np.int32)
    gauge[0] = order[spans[0, 0] + 1]
    return {
        "lengths": lengths,
        "order": order,
        "inverse": inverse,
        "spans": spans,
        "scale": scale,
        "band": band,
        "template": template,
        "offsets": offsets,
        "columns": np.asarray(columns, dtype=np.int32),
        "values": np.asarray(values),
        "gauge": gauge,
    }


def _assembled(data, masked):
    """Float64 oracle of each cable's assembled band matrix (lower part as the factor reads it)."""
    result = []
    order, inverse, scale = data["order"], data["inverse"], data["scale"]
    offsets, columns, values = data["offsets"], data["columns"], data["values"]
    for start, end in data["spans"]:
        n = end - start
        lower = np.zeros((n, n))
        for i in range(n):
            for d in range(min(11, i + 1)):
                lower[i, i - d] = data["band"][start + i, d]
            row = order[start + i]
            for slot in range(offsets[row], offsets[row + 1]):
                j = inverse[columns[slot]]
                if start <= j <= start + i and start + i - j <= 10:
                    if not masked or data["template"][start + i, start + i - j]:
                        lower[i, j - start] += values[slot] * scale[row] * scale[columns[slot]]
        result.append(lower + np.tril(lower, -1).T)
    return result


class _Run:
    """A preconditioner fed directly with synthetic assembled rows (no solver, no element operator)."""

    def __init__(self, device, dtype, data):
        self.device, self.dtype, self.data = device, dtype, data
        self.pre = CablePreconditioner(_layout(data["lengths"]), data["order"], device=device, dtype=dtype)
        self.status = wp.zeros(1, dtype=int, device=device)
        self.pre.gauge_rows.assign(data["gauge"])

    def factor(self, *, masked=False, regularization=0.0, serial=False):
        pre, data, device, dtype = self.pre, self.data, self.device, self.dtype
        self.status.zero_()
        pre.band.assign(data["band"].ravel())
        pre.invalid.zero_()
        if pre.rows.size:
            wp.launch(
                _assemble_rows,
                pre.rows.size,
                [
                    wp.array(data["offsets"], dtype=int, device=device),
                    wp.array(data["columns"], dtype=int, device=device),
                    wp.array(data["values"], dtype=dtype, device=device),
                    wp.array(data["scale"], dtype=dtype, device=device),
                    pre.rows,
                    pre.inverse,
                    pre.spans,
                    pre.row_actor,
                    pre.band,
                    wp.array(data["template"].ravel(), dtype=int, device=device),
                    int(masked),
                    pre.assembled,
                    pre.invalid,
                    self.status,
                ],
                device=device,
            )
        pre._factor_blocks(dtype(regularization), dtype(0.0), self.status, serial=serial)
        return int(self.status.numpy()[0])

    def snapshot(self):
        pre = self.pre
        return [
            a.numpy().copy()
            for a in (pre.block_factor, pre.block_coupling, pre.block_left, pre.block_right, pre.used_shift)
        ]


def _gamma(levels, dtype):
    # Componentwise backward-error constant of a Cholesky whose longest accumulated sum has
    # m = 10 (2 levels + 1) + 1 products (Higham, ASNA 2nd ed., Theorem 10.3: |dA| <= gamma |L||L^T|).
    u = np.finfo(np.float32 if dtype == wp.float32 else np.float64).eps / 2
    m = 10 * (2 * levels + 1) + 1
    return m * u / (1 - m * u)


def test_factor_reproduces_the_shifted_band(test, device):
    """L L^T of every cable equals its assembled, regularized, shifted and gauge-lifted band."""
    for dtype in (wp.float32, wp.float64):
        for masked in (False, True):
            for seed in (1, 2):
                data = _synthetic(np.random.default_rng(seed), LENGTHS, element_only=(2,), shifted=2)
                run = _Run(device, dtype, data)
                status = run.factor(masked=masked, regularization=2e-5)
                used = run.pre.used_shift.numpy().astype(np.float64)
                gamma = 2 * _gamma(len(run.pre.odd_jobs), dtype)
                with test.subTest(dtype=dtype.__name__, masked=masked, seed=seed):
                    test.assertEqual(status, 0)
                    test.assertGreater(used[2], 0.0)
                    test.assertEqual(used[0], 0.0)
                    for actor, (matrix, lower) in enumerate(
                        zip(_assembled(data, masked), run.pre.dense_factors(), strict=True)
                    ):
                        expected = matrix.copy()
                        expected[np.diag_indices_from(expected)] += 2e-5 + used[actor] * np.abs(np.diag(matrix))
                        if actor == 0:
                            expected[1, 1] += 1.0  # gauge row: band position 1 of cable 0
                        product = lower @ lower.T
                        bound = gamma * (np.abs(lower) @ np.abs(lower).T) + np.finfo(np.float64).tiny
                        # Every entry, including those outside the band, where the expected value is zero.
                        np.testing.assert_array_less(np.abs(product - expected), bound, err_msg=f"cable {actor}")


def test_apply_solves_with_the_factor(test, device):
    """Apply returns L^-T L^-1 r on every cable row, leaves other rows untouched and honours ``active``."""
    for dtype in (wp.float32, wp.float64):
        data = _synthetic(np.random.default_rng(3), LENGTHS)
        run = _Run(device, dtype, data)
        test.assertEqual(run.factor(regularization=0.1), 0)
        size = len(data["inverse"])
        rhs_host = np.random.default_rng(4).normal(size=size)
        rhs = wp.array(rhs_host, dtype=dtype, device=device)
        result = wp.full(size, 7.0, dtype=dtype, device=device)
        run.pre.apply(rhs, result)
        x = result.numpy().astype(np.float64)
        gamma = 4 * _gamma(len(run.pre.odd_jobs), dtype)
        for (start, end), lower in zip(data["spans"], run.pre.dense_factors(), strict=True):
            rows = data["order"][start:end]
            residual = lower @ (lower.T @ x[rows]) - rhs_host[rows]
            bound = gamma * (np.abs(lower) @ (np.abs(lower).T @ np.abs(x[rows])) + np.abs(rhs_host[rows]))
            np.testing.assert_array_less(np.abs(residual), bound + np.finfo(np.float64).tiny)
        outside = np.setdiff1d(np.arange(size), data["order"])
        np.testing.assert_array_equal(x[outside], 7.0)
        again = wp.full(size, 7.0, dtype=dtype, device=device)
        run.pre.apply(rhs, again)
        np.testing.assert_array_equal(again.numpy(), result.numpy())
        inactive = wp.full(size, 3.0, dtype=dtype, device=device)
        run.pre.apply(rhs, inactive, active=wp.zeros(1, dtype=int, device=device))
        np.testing.assert_array_equal(inactive.numpy(), 3.0)
        if device.is_cuda:
            replay = wp.full(size, 7.0, dtype=dtype, device=device)
            with wp.ScopedCapture(device) as capture:
                run.pre.apply(rhs, replay)
            replay.fill_(7.0)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(replay.numpy(), result.numpy())


def test_serial_ladder_matches_the_levels_bit_for_bit(test, device):
    """The per-cable shift ladder repeats the level kernels' arithmetic exactly, shift 0 included."""
    for dtype in (wp.float32, wp.float64):
        for masked in (False, True):
            data = _synthetic(np.random.default_rng(5), LENGTHS, element_only=(4,), shifted=4)
            run = _Run(device, dtype, data)
            levels_status = run.factor(masked=masked)
            levels = run.snapshot()
            serial_status = run.factor(masked=masked, serial=True)
            serial = run.snapshot()
            with test.subTest(dtype=dtype.__name__, masked=masked):
                test.assertEqual(levels_status, 0)
                test.assertEqual(serial_status, 0)
                test.assertGreater(float(levels[-1][4]), 0.0)
                for a, b in zip(levels, serial, strict=True):
                    np.testing.assert_array_equal(a, b)
            again = _Run(device, dtype, data)
            again.factor(masked=masked)
            for a, b in zip(levels, again.snapshot(), strict=True):
                np.testing.assert_array_equal(a, b)


def test_status_codes_of_different_cables_merge(test, device):
    """Out-of-band 1, failed pivot 12 and non-finite 16 merge as 16 > 12 > 1; an out-of-band cable is not factored."""
    cases = (
        # (cables with out-of-band entries, failed pivots, non-finite entries, status)
        ((0,), (2,), (4,), 16),
        ((0,), (2,), (), 12),
        ((0,), (), (), 1),
        ((), (2,), (), 12),
        ((), (), (4,), 16),
        # Both in one cable: its assembly rejects it before the non-finite entry is factored.
        ((4,), (), (4,), 1),
    )
    for dtype in (wp.float32, wp.float64):
        for far, negative, nonfinite, expected in cases:
            data = _synthetic(np.random.default_rng(6), LENGTHS, far=far)
            for actor in negative:
                data["band"][data["spans"][actor, 0], 0] = -1e6
            for actor in nonfinite:
                data["band"][data["spans"][actor, 0] + 5, 0] = np.nan
            run = _Run(device, dtype, data)
            run.pre.used_shift.fill_(-1.0)
            status = run.factor()
            used = run.pre.used_shift.numpy()
            with test.subTest(dtype=dtype.__name__, far=far, negative=negative, nonfinite=nonfinite):
                test.assertEqual(status, expected)
                failed = set(far) | set(negative) | set(nonfinite)
                for actor in range(len(LENGTHS)):
                    # A cable records a shift only when it factored.
                    test.assertEqual(float(used[actor]) == -1.0, actor in failed, actor)


def test_launches_follow_the_levels(test, device):
    """Apply launches 3L + 3 kernels whose sizes are the per-level block counts, for any number of cables."""
    sequences = {}
    for lengths in ((41,), (41, 41, 7)):
        data = _synthetic(np.random.default_rng(7), lengths)
        pre = CablePreconditioner(_layout(lengths), data["order"], device=device, dtype=wp.float64)
        size = len(data["inverse"])
        rhs, result = (wp.zeros(size, dtype=wp.float64, device=device) for _ in range(2))
        launches = []
        original = wp.launch

        def launch(kernel, dim, *args, launches=launches, original=original, **kwargs):
            launches.append(int(dim))
            return original(kernel, dim, *args, **kwargs)

        with patch.object(wp, "launch", launch):
            pre.apply(rhs, result)  # counts launches only; the factor is never computed here
        sequences[lengths] = launches
    # 41 rows are 5 blocks: level 0 eliminates blocks 1, 3 (keeps 0, 2, 4), level 1 block 2
    # (keeps 0, 4), level 2 block 4; then the root, the backward levels and the scatter.
    test.assertEqual(sequences[(41,)], [50, 2, 3, 1, 2, 1, 1, 1, 1, 1, 2, 41])
    test.assertEqual(sequences[(41, 41, 7)], [110, 4, 6, 2, 4, 2, 2, 3, 2, 2, 4, 89])


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
    test_apply_solves_with_the_factor,
    test_factor_reproduces_the_shifted_band,
    test_launches_follow_the_levels,
    test_out_of_band_entry_is_reported_behind_a_failed_pivot,
    test_serial_ladder_matches_the_levels_bit_for_bit,
    test_status_codes_of_different_cables_merge,
):
    add_function_test(TestCablePreconditioner, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
