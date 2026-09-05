# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the monolithic assembly contract with synthetic contributions."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import warp as wp
import warp.sparse as sparse

from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
    _accumulate_monolithic_actor_q_entry,
    _accumulate_monolithic_internal_x_block,
    _append_monolithic_contact_factor_triplets,
    _append_monolithic_scalar_triplet,
    _MonolithicContactFactors,
    _MonolithicMat33Triplets,
    _MonolithicScalarTriplets,
    _reserve_monolithic_contact_factor,
)
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _owners(aq: wp.array2d[float], ax: _MonolithicMat33Triplets, triplets: _MonolithicScalarTriplets):
    for i in range(2):
        _accumulate_monolithic_actor_q_entry(aq, triplets, i, i, 2.0)
        _accumulate_monolithic_actor_q_entry(aq, triplets, i, i, 3.0)
    _accumulate_monolithic_actor_q_entry(aq, triplets, 0, 1, 0.5)
    _accumulate_monolithic_actor_q_entry(aq, triplets, 1, 0, 0.5)
    for i in range(3):
        _accumulate_monolithic_internal_x_block(ax, triplets, i, i, wp.diag(wp.vec3(1.0)), 2)
        _accumulate_monolithic_internal_x_block(ax, triplets, i, i, wp.diag(wp.vec3(3.0)), 2)
    _accumulate_monolithic_internal_x_block(ax, triplets, 0, 1, wp.diag(wp.vec3(0.2)), 2)
    _accumulate_monolithic_internal_x_block(ax, triplets, 1, 0, wp.diag(wp.vec3(0.2)), 2)


@wp.kernel
def _contact(factors: _MonolithicContactFactors, triplets: _MonolithicScalarTriplets):
    slot = _reserve_monolithic_contact_factor(factors)
    if slot < 0:
        return
    factors.gq[slot, 0] = 2.0
    factors.gq[slot, 1] = -1.0
    factors.weights[slot] = 3.0
    for j in range(9):
        factors.gx_columns[slot, j] = j
        factors.gx_values[slot, j] = float(j + 1) * 0.1
    _append_monolithic_contact_factor_triplets(factors, slot, triplets, 2)


@wp.kernel
def _append(triplets: _MonolithicScalarTriplets, row: int, value: float):
    _append_monolithic_scalar_triplet(triplets, row, 0, value)


def _workspace(device, *, global_capacity=256, internal_capacity=8, contact_capacity=2):
    layout = MonolithicLinearLayout(2, 3)
    capacities = MonolithicLinearCapacities(global_capacity, internal_capacity, contact_capacity, 10, 3)
    mapping = wp.array([-1, 0, 1, 2], dtype=int, device=device)
    return MonolithicLinearWorkspace(layout, capacities, mapping, device=device)


def _dense_bsr(matrix):
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    block_size = matrix.block_shape[0]
    result = np.zeros((matrix.nrow * block_size, matrix.ncol * block_size))
    for row in range(matrix.nrow):
        for index in range(offsets[row], offsets[row + 1]):
            column = columns[index]
            result[row * block_size : (row + 1) * block_size, column * block_size : (column + 1) * block_size] = values[
                index
            ]
    return result


def test_owner_contact_oracle(test, device):
    """Compare owners, contact four-block expansion, BSR and scaled dense algebra."""
    workspace = _workspace(device)
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    wp.launch(
        _owners,
        1,
        [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets],
        device=device,
    )
    wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
    result = workspace.finalize_assembly(generation=generation)
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(result.global_triplet_count, 199)
    test.assertEqual(result.internal_triplet_count, 8)
    raw = workspace.densify_for_test(generation=generation).raw_matrix
    owners = np.zeros((11, 11))
    owners[:2, :2] = assembly.aq_actor_dense.numpy()
    owners[2:, 2:] = _dense_bsr(workspace.ax_internal_bsr3)
    np.testing.assert_allclose(owners[:2, :2], [[5, 0.5], [0.5, 5]])
    np.testing.assert_allclose(owners[2:5, 5:8], np.eye(3) * 0.2)
    g = np.r_[2.0, -1.0, np.arange(1, 10) * 0.1]
    contact = 3.0 * np.outer(g, g)
    for rows, columns in [
        (slice(2), slice(2)),
        (slice(2), slice(2, None)),
        (slice(2, None), slice(2)),
        (slice(2, None), slice(2, None)),
    ]:
        np.testing.assert_allclose((raw - owners)[rows, columns], contact[rows, columns], atol=8e-7)
    np.testing.assert_allclose(raw, raw.T, atol=8e-7)
    vector = np.linspace(-1, 1, 11).astype(np.float32)
    output = wp.empty(11, dtype=float, device=device)
    sparse.bsr_mv(workspace.k_global_scalar_bsr, wp.array(vector, device=device), output)
    np.testing.assert_allclose(output.numpy(), raw @ vector, atol=2e-6)
    np.testing.assert_allclose(sparse.bsr_get_diag(workspace.k_global_scalar_bsr).numpy(), raw.diagonal())
    # Scaling here is a synthetic oracle identity, not the PR-3 runtime policy.
    diagonal = np.linspace(1, 4, 11)
    scale = 1 / np.sqrt(diagonal)
    regularized = raw + 0.03 * np.diag(diagonal)
    scaled = scale[:, None] * regularized * scale[None, :]
    test.assertGreater(np.linalg.eigvalsh(scaled).min(), 0)
    sparse.bsr_mv(workspace.k_global_scalar_bsr, wp.array((scale * vector).astype(np.float32), device=device), output)
    np.testing.assert_allclose(scaled @ vector, scale * output.numpy() + 0.03 * vector, atol=2e-6)
    np.testing.assert_allclose(np.linalg.solve(regularized, vector), scale * np.linalg.solve(scaled, scale * vector))


def test_generation_freeze(test, device):
    """Reject stale writes after finalization and after a new generation starts."""
    workspace = _workspace(device)
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    old = workspace.begin_assembly(generation)
    wp.launch(_owners, 1, [old.aq_actor_dense, old.ax_internal_triplets, old.global_scalar_triplets], device=device)
    workspace.finalize_assembly(generation=generation)
    before = workspace.densify_for_test(generation=generation).raw_matrix.copy()
    owner_before = old.aq_actor_dense.numpy().copy()
    counts_before = [
        buffer.count.numpy().copy()
        for buffer in (old.global_scalar_triplets, old.ax_internal_triplets, old.contact_factors)
    ]
    wp.launch(_owners, 1, [old.aq_actor_dense, old.ax_internal_triplets, old.global_scalar_triplets], device=device)
    wp.launch(_contact, 1, [old.contact_factors, old.global_scalar_triplets], device=device)
    np.testing.assert_array_equal(workspace.densify_for_test(generation=generation).raw_matrix, before)
    np.testing.assert_array_equal(old.aq_actor_dense.numpy(), owner_before)
    for buffer, count in zip(
        (old.global_scalar_triplets, old.ax_internal_triplets, old.contact_factors), counts_before, strict=True
    ):
        np.testing.assert_array_equal(buffer.count.numpy(), count)
    test.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.STALE_GENERATION)
    with test.assertRaises(ValueError):
        workspace.begin_assembly(generation)
    next_generation = MonolithicLinearGeneration(1, 1, 2, 2)
    current = workspace.begin_assembly(next_generation)
    wp.launch(_owners, 1, [old.aq_actor_dense, old.ax_internal_triplets, old.global_scalar_triplets], device=device)
    wp.launch(_contact, 1, [old.contact_factors, old.global_scalar_triplets], device=device)
    test.assertEqual(current.global_scalar_triplets.count.numpy()[0], 0)
    np.testing.assert_array_equal(current.aq_actor_dense.numpy(), 0)
    with test.assertRaises(ValueError):
        workspace.densify_for_test(generation=generation)
    test.assertEqual(workspace.finalize_assembly(generation=next_generation).status, MonolithicLinearStatus.SUCCESS)
    new_step = MonolithicLinearGeneration(2, 0, 1, 0)
    workspace.begin_assembly(new_step)
    test.assertEqual(workspace.finalize_assembly(generation=new_step).status, MonolithicLinearStatus.SUCCESS)
    with test.assertRaises(ValueError):
        workspace.begin_assembly(next_generation)


def test_capacity_and_invalid_contributions(test, device):
    """Invalidate whole assemblies for overflow, bad indices and nonfinite terms."""
    for kwargs, expected in [
        ({"global_capacity": 1}, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW),
        ({"internal_capacity": 1}, MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW),
        ({"contact_capacity": 0}, MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW),
    ]:
        workspace = _workspace(device, **kwargs)
        generation = MonolithicLinearGeneration(1, 0, 1, 1)
        assembly = workspace.begin_assembly(generation)
        kernel = _contact if "contact_capacity" in kwargs else _owners
        args = (
            [assembly.contact_factors, assembly.global_scalar_triplets]
            if kernel == _contact
            else [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets]
        )
        wp.launch(kernel, 1, args, device=device)
        test.assertEqual(workspace.finalize_assembly(generation=generation).status, expected)
        with test.assertRaises(ValueError):
            workspace.densify_for_test(generation=generation)
    for row, value, expected in [
        (11, 1.0, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        (0, float("nan"), MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
    ]:
        workspace = _workspace(device)
        generation = MonolithicLinearGeneration(1, 0, 1, 1)
        assembly = workspace.begin_assembly(generation)
        wp.launch(_append, 1, [assembly.global_scalar_triplets, row, value], device=device)
        test.assertEqual(workspace.finalize_assembly(generation=generation).status, expected)


@wp.kernel
def _fixture_diagonal(
    aq: wp.array2d[float], ax: _MonolithicMat33Triplets, triplets: _MonolithicScalarTriplets, masses: wp.array[float]
):
    for i in range(2):
        _accumulate_monolithic_actor_q_entry(aq, triplets, i, i, float(i + 1))
    for i in range(4):
        _accumulate_monolithic_internal_x_block(ax, triplets, i, i, wp.diag(wp.vec3(masses[i])), 2)


@wp.kernel
def _masked_contact(
    factors: _MonolithicContactFactors, triplets: _MonolithicScalarTriplets, weight: float, column: int
):
    slot = _reserve_monolithic_contact_factor(factors)
    if slot < 0:
        return
    factors.weights[slot] = weight
    factors.gq[slot, 0] = 1.0
    # Repeated columns must sum; unused fixed-node slots retain -1/0.
    factors.gx_columns[slot, 0] = column
    factors.gx_columns[slot, 1] = column
    factors.gx_values[slot, 0] = 2.0
    factors.gx_values[slot, 1] = 3.0
    _append_monolithic_contact_factor_triplets(factors, slot, triplets, 2)


def test_fixed_contact_and_weight(test, device):
    """Expand repeated columns and fixed-node sentinels, and reject invalid factors."""
    for weight, column, status in [
        (2.0, 0, MonolithicLinearStatus.SUCCESS),
        (0.0, 0, MonolithicLinearStatus.INVALID_CONTACT_WEIGHT),
        (-1.0, 0, MonolithicLinearStatus.INVALID_CONTACT_WEIGHT),
        (float("nan"), 0, MonolithicLinearStatus.INVALID_CONTACT_WEIGHT),
        (2.0, 9, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
    ]:
        workspace = _workspace(device)
        generation = MonolithicLinearGeneration(1, 0, 1, 1)
        assembly = workspace.begin_assembly(generation)
        wp.launch(
            _masked_contact,
            1,
            [assembly.contact_factors, assembly.global_scalar_triplets, weight, column],
            device=device,
        )
        test.assertEqual(workspace.finalize_assembly(generation=generation).status, status)
        if status == MonolithicLinearStatus.SUCCESS:
            g = np.zeros(11)
            g[0], g[2] = 1, 5
            np.testing.assert_array_equal(
                workspace.densify_for_test(generation=generation).raw_matrix, 2 * np.outer(g, g)
            )


def test_shared_fixture_layout(test, device):
    """Scatter synthetic diagonal terms into the frozen fixture's q/x ordering."""
    fixture = build_tiny_cpu_fixture(device=device)
    layout = MonolithicLinearLayout(fixture.model.joint_dof_count, fixture.model.particle_count)
    test.assertEqual(layout.scalar_dof_count, fixture.spec.scalar_dof_count)
    workspace = MonolithicLinearWorkspace(
        layout,
        MonolithicLinearCapacities(38, 4, 0, 10, 3),
        wp.array([0, 1, 2, 3], dtype=int, device=device),
        device=device,
    )
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    wp.launch(
        _fixture_diagonal,
        1,
        [
            assembly.aq_actor_dense,
            assembly.ax_internal_triplets,
            assembly.global_scalar_triplets,
            fixture.model.particle_mass,
        ],
        device=device,
    )
    test.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
    np.testing.assert_array_equal(
        workspace.densify_for_test(generation=generation).raw_matrix,
        np.diag(np.r_[1, 2, np.repeat(fixture.model.particle_mass.numpy(), 3)]),
    )


@wp.kernel
def _overflow_reduction(
    aq: wp.array2d[float],
    ax: _MonolithicMat33Triplets,
    triplets: _MonolithicScalarTriplets,
    q_count: int,
    x_count: int,
    owner_scatter: bool,
):
    for _ in range(2):
        if owner_scatter:
            if q_count > 0:
                _accumulate_monolithic_actor_q_entry(aq, triplets, 0, 0, 3.0e38)
            if x_count > 0:
                _accumulate_monolithic_internal_x_block(ax, triplets, 0, 0, wp.diag(wp.vec3(3.0e38)), q_count)
        else:
            _append_monolithic_scalar_triplet(triplets, 0, 0, 3.0e38)


def test_nonfinite_reduction(test, device):
    """Reject overflow from finite contributions in global and owner reductions."""
    for q_count, x_count, owner_scatter in [(1, 0, False), (1, 0, True), (0, 1, True), (1, 1, True)]:
        with test.subTest(q_count=q_count, x_count=x_count, owner_scatter=owner_scatter):
            workspace = MonolithicLinearWorkspace(
                MonolithicLinearLayout(q_count, x_count),
                MonolithicLinearCapacities(20, 2, 0, 1, 1),
                wp.array(np.arange(x_count), dtype=int, device=device),
                device=device,
            )
            generation = MonolithicLinearGeneration(1, 0, 1, 1)
            assembly = workspace.begin_assembly(generation)
            wp.launch(
                _overflow_reduction,
                1,
                [
                    assembly.aq_actor_dense,
                    assembly.ax_internal_triplets,
                    assembly.global_scalar_triplets,
                    q_count,
                    x_count,
                    owner_scatter,
                ],
                device=device,
            )
            result = workspace.finalize_assembly(generation=generation)
            test.assertFalse(np.isfinite(workspace.k_global_scalar_bsr.values.numpy()[0]))
            if owner_scatter and q_count:
                test.assertFalse(np.isfinite(assembly.aq_actor_dense.numpy()[0, 0]))
            if owner_scatter and x_count:
                test.assertFalse(np.isfinite(workspace.ax_internal_bsr3.values.numpy()[0, 0, 0]))
            test.assertEqual(result.status, MonolithicLinearStatus.NONFINITE_CONTRIBUTION)
            with test.assertRaises(ValueError):
                workspace.densify_for_test(generation=generation)
            next_generation = MonolithicLinearGeneration(1, 1, 2, 2)
            fresh = workspace.begin_assembly(next_generation)
            wp.launch(_append, 1, [fresh.global_scalar_triplets, 0, 1.0], device=device)
            test.assertEqual(
                workspace.finalize_assembly(generation=next_generation).status, MonolithicLinearStatus.SUCCESS
            )


def _fixed_pattern():
    return {
        "global_rows": np.r_[0, np.repeat(np.arange(2, 5), 3)],
        "global_columns": np.r_[0, np.tile(np.arange(2, 5), 3)],
        "internal_rows": np.array([0]),
        "internal_columns": np.array([0]),
    }


@wp.kernel
def _fixed_owners(aq: wp.array2d[float], ax_values: wp.array[wp.mat33], global_values: wp.array[float]):
    aq[0, 0] = 2.0
    ax_values[0] = wp.diag(wp.vec3(4.0))
    global_values[0] = 2.0
    for i in range(3):
        global_values[1 + 3 * i + i] = 4.0


def test_fixed_prefix_contact_and_reset(test, device):
    """Preserve fixed owner slots beside appended contacts and reset each generation."""
    workspace = _workspace(device)
    pattern = _fixed_pattern()
    workspace._set_fixed_triplet_pattern(**pattern)
    pattern["global_rows"].fill(10)
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    test.assertEqual(assembly.global_scalar_triplets.count.numpy()[0], 10)
    test.assertEqual(assembly.ax_internal_triplets.count.numpy()[0], 1)
    wp.launch(
        _fixed_owners,
        1,
        [assembly.aq_actor_dense, assembly.ax_internal_triplets.values, assembly.global_scalar_triplets.values],
        device=device,
    )
    wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
    result = workspace.finalize_assembly(generation=generation)
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(result.global_triplet_count, 131)
    owners = np.zeros((11, 11))
    owners[:2, :2] = assembly.aq_actor_dense.numpy()
    owners[2:, 2:] = _dense_bsr(workspace.ax_internal_bsr3)
    g = np.r_[2.0, -1.0, np.arange(1, 10) * 0.1]
    np.testing.assert_allclose(
        workspace.densify_for_test(generation=generation).raw_matrix, owners + 3 * np.outer(g, g), atol=8e-7
    )
    # A raw producer can alter indices; the next assembly restores the frozen pattern.
    assembly.global_scalar_triplets.rows.fill_(10)
    next_generation = MonolithicLinearGeneration(1, 1, 2, 2)
    fresh = workspace.begin_assembly(next_generation)
    np.testing.assert_array_equal(fresh.global_scalar_triplets.rows.numpy()[:10], _fixed_pattern()["global_rows"])
    test.assertEqual(fresh.global_scalar_triplets.count.numpy()[0], 10)
    test.assertEqual(fresh.ax_internal_triplets.count.numpy()[0], 1)
    test.assertEqual(workspace.finalize_assembly(generation=next_generation).status, MonolithicLinearStatus.SUCCESS)
    np.testing.assert_array_equal(workspace.densify_for_test(generation=next_generation).raw_matrix, 0)
    np.testing.assert_array_equal(_dense_bsr(workspace.ax_internal_bsr3), 0)
    with test.assertRaises(ValueError):
        workspace._set_fixed_triplet_pattern(**_fixed_pattern())


def test_fixed_pattern_validation(test, device):
    """Reject malformed patterns before changing the construction-time reservation."""
    for field, invalid in [
        ("global_rows", np.array([11] * 10)),
        ("global_columns", np.array([-1] * 10)),
        ("global_rows", np.zeros(257, dtype=int)),
        ("global_rows", np.zeros((2, 5), dtype=int)),
        ("global_columns", np.zeros(10, dtype=float)),
        ("internal_rows", np.array([3])),
        ("internal_columns", np.array([-1])),
        ("internal_columns", np.array([2**32])),
        ("internal_rows", np.zeros(9, dtype=int)),
    ]:
        with test.subTest(field=field, invalid=invalid):
            workspace = _workspace(device)
            pattern = _fixed_pattern()
            pattern[field] = invalid
            with test.assertRaises(ValueError):
                workspace._set_fixed_triplet_pattern(**pattern)
            workspace._set_fixed_triplet_pattern(**_fixed_pattern())
            with test.assertRaises(ValueError):
                workspace._set_fixed_triplet_pattern(**_fixed_pattern())
            generation = MonolithicLinearGeneration(1, 0, 1, 1)
            assembly = workspace.begin_assembly(generation)
            test.assertEqual(assembly.global_scalar_triplets.count.numpy()[0], 10)
            test.assertEqual(assembly.ax_internal_triplets.count.numpy()[0], 1)
            test.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)


def test_fixed_raw_validation(test, device):
    """Reject invalid raw producer writes and append overflow after a fixed prefix."""
    for buffer_name, field, value, expected in [
        ("global_scalar_triplets", "rows", 11, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ("global_scalar_triplets", "columns", -1, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ("ax_internal_triplets", "rows", 3, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ("ax_internal_triplets", "columns", -1, MonolithicLinearStatus.TRIPLET_INDEX_OUT_OF_RANGE),
        ("global_scalar_triplets", "values", float("nan"), MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
        ("ax_internal_triplets", "values", wp.mat33(float("inf")), MonolithicLinearStatus.NONFINITE_CONTRIBUTION),
        ("global_scalar_triplets", "count", -1, MonolithicLinearStatus.INVALID_ARGUMENT),
        ("global_scalar_triplets", "count", 9, MonolithicLinearStatus.INVALID_ARGUMENT),
        ("ax_internal_triplets", "count", 0, MonolithicLinearStatus.INVALID_ARGUMENT),
        ("contact_factors", "count", -1, MonolithicLinearStatus.INVALID_ARGUMENT),
    ]:
        with test.subTest(buffer=buffer_name, field=field, value=value):
            workspace = _workspace(device)
            workspace._set_fixed_triplet_pattern(**_fixed_pattern())
            generation = MonolithicLinearGeneration(1, 0, 1, 1)
            assembly = workspace.begin_assembly(generation)
            getattr(getattr(assembly, buffer_name), field).fill_(value)
            test.assertEqual(workspace.finalize_assembly(generation=generation).status, expected)
            with test.assertRaises(ValueError):
                workspace.densify_for_test(generation=generation)
    workspace = _workspace(device, global_capacity=10)
    workspace._set_fixed_triplet_pattern(**_fixed_pattern())
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    wp.launch(_append, 1, [assembly.global_scalar_triplets, 0, 1.0], device=device)
    test.assertEqual(
        workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW
    )


def test_scaling_operator_preconditioner(test, device):
    """Match the production scaled operator and block inverse against dense algebra."""
    workspace = _workspace(device)
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    wp.launch(
        _owners,
        1,
        [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets],
        device=device,
    )
    wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
    test.assertEqual(workspace.finalize_assembly(generation=generation).status, MonolithicLinearStatus.SUCCESS)
    dynamic = wp.array(np.linspace(1, 100, 11).astype(np.float32), device=device)
    test.assertEqual(
        workspace.build_scaling(dynamic, epsilon_d=0.25, generation=generation), MonolithicLinearStatus.SUCCESS
    )
    test.assertEqual(workspace.set_regularization(0.03, generation=generation), MonolithicLinearStatus.SUCCESS)
    test.assertEqual(
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=1e-10),
        MonolithicLinearStatus.SUCCESS,
    )
    oracle = workspace.densify_for_test(generation=generation)
    expected_d = np.maximum(oracle.raw_matrix.diagonal(), 0.25 * dynamic.numpy())
    np.testing.assert_allclose(workspace.diagonal.numpy(), expected_d, rtol=1e-6)
    np.testing.assert_allclose(oracle.scale, 1 / np.sqrt(expected_d), rtol=1e-6)
    vector = np.linspace(-1, 1, 11).astype(np.float32)
    x, out = wp.array(vector, device=device), wp.zeros(11, dtype=float, device=device)
    workspace.operator.matvec(x, out, out, 1.0, 0.0)
    np.testing.assert_allclose(out.numpy(), oracle.scaled_matrix @ vector, atol=5e-7)
    expected_p = np.zeros((11, 11))
    for start, stop in [(0, 2), (2, 5), (5, 8), (8, 11)]:
        expected_p[start:stop, start:stop] = oracle.scaled_matrix[start:stop, start:stop]
    workspace.preconditioner.matvec(x, out, out, 1.0, 0.0)
    np.testing.assert_allclose(out.numpy(), np.linalg.solve(expected_p, vector), rtol=2e-6, atol=5e-7)
    delta = wp.zeros(11, dtype=float, device=device)
    test.assertEqual(workspace.recover_delta(x, delta, generation=generation), MonolithicLinearStatus.SUCCESS)
    np.testing.assert_allclose(delta.numpy(), oracle.scale * vector, atol=1e-7)


def _pcg_config(**kwargs):
    # Explicit test thresholds; these do not claim to be P0 calibration.
    return replace(MonolithicPcgConfig(100, 10, 1e-4, 1e-12, 1e-12, 1e-12, 0.0, 1e-8, 0.0, 20, 0.001), **kwargs)


def _ready_system(device, *, contact=True, heterogeneous=False):
    workspace = MonolithicLinearWorkspace(
        MonolithicLinearLayout(2, 3),
        MonolithicLinearCapacities(256, 8, 2, 100, 102),
        wp.array([0, 1, 2], dtype=int, device=device),
        device=device,
    )
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    assembly = workspace.begin_assembly(generation)
    wp.launch(
        _owners,
        1,
        [assembly.aq_actor_dense, assembly.ax_internal_triplets, assembly.global_scalar_triplets],
        device=device,
    )
    if contact:
        wp.launch(_contact, 1, [assembly.contact_factors, assembly.global_scalar_triplets], device=device)
    coordinate_scale = (
        np.geomspace(1e-3, 1e2, 11).astype(np.float32) if heterogeneous else np.ones(11, dtype=np.float32)
    )
    if heterogeneous:
        wp.launch(
            _rescale_system,
            1,
            [
                assembly.aq_actor_dense,
                assembly.ax_internal_triplets,
                assembly.contact_factors,
                assembly.global_scalar_triplets,
                wp.array(coordinate_scale, device=device),
            ],
            device=device,
        )
    workspace.finalize_assembly(generation=generation)
    dynamic = wp.array(coordinate_scale**2, device=device)
    workspace.build_scaling(dynamic, epsilon_d=0.01, generation=generation)
    workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=1e-10)
    return workspace, generation


@wp.kernel
def _rescale_system(
    aq: wp.array2d[float],
    ax: _MonolithicMat33Triplets,
    factors: _MonolithicContactFactors,
    triplets: _MonolithicScalarTriplets,
    scale: wp.array[float],
):
    for i in range(2):
        for j in range(2):
            aq[i, j] *= scale[i] * scale[j]
    for slot in range(triplets.count[0]):
        triplets.values[slot] = triplets.values[slot] * scale[triplets.rows[slot]] * scale[triplets.columns[slot]]
    for slot in range(ax.count[0]):
        block = ax.values[slot]
        for i in range(3):
            for j in range(3):
                block[i, j] *= scale[2 + 3 * ax.rows[slot] + i] * scale[2 + 3 * ax.columns[slot] + j]
        ax.values[slot] = block
    for c in range(factors.count[0]):
        for i in range(2):
            factors.gq[c, i] = factors.gq[c, i] * scale[i]
        for i in range(9):
            if factors.gx_columns[c, i] >= 0:
                factors.gx_values[c, i] = factors.gx_values[c, i] * scale[2 + factors.gx_columns[c, i]]


def test_pcg_dense_and_allocations(test, device):
    """Solve a heterogeneous coupled system against the same BSR without iteration allocations."""
    workspace, generation = _ready_system(device, heterogeneous=True)
    oracle = workspace.densify_for_test(generation=generation)
    test.assertGreater(np.linalg.eigvalsh(oracle.scaled_matrix).min(), 0)
    rhs_np = np.linspace(-1, 2, 11).astype(np.float32)
    rhs, y = wp.array(rhs_np, device=device), wp.zeros(11, dtype=float, device=device)
    config = _pcg_config(linear_tolerance=1e-5, true_residual_interval=3)

    def solve():
        return workspace.solve_pcg(rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config)

    result = solve()
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    test.assertGreater(result.residual_replacements, 0)
    np.testing.assert_allclose(y.numpy(), np.linalg.solve(oracle.scaled_matrix, rhs_np), rtol=2e-5, atol=2e-5)
    workspace.operator.matvec(y, workspace._ap, workspace._ap, 1.0, 0.0)
    r = workspace._ap.numpy() - rhs_np
    expected = tuple(
        np.linalg.norm(r[part]) / max(np.linalg.norm(rhs_np[part]), 1e-12)
        for part in [slice(None), slice(0, 2), slice(2, None)]
    )
    np.testing.assert_allclose((result.rho, result.rho_q, result.rho_x), expected, atol=1e-10)
    test.assertGreater(result.min_p_ap, 0)
    test.assertGreater(result.min_r_z, 0)
    allocator = wp.get_device(device).get_allocator()
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
        for _ in range(10):
            test.assertEqual(solve().status, MonolithicLinearStatus.SUCCESS)
        test.assertEqual(allocate.call_count, 0)


def test_scaling_guards(test, device):
    """Reject nonpositive or nonfinite raw and dynamic diagonals before applying floors."""
    for source in ["raw", "dynamic"]:
        for value in [0.0, -1.0, float("nan"), float("inf")]:
            workspace, generation = _ready_system(device, contact=False)
            dynamic = np.ones(11, dtype=np.float32)
            if source == "dynamic":
                dynamic[0] = value
            else:
                values = workspace.k_global_scalar_bsr.values.numpy()
                values[0] = value
                workspace.k_global_scalar_bsr.values.assign(values)
            test.assertEqual(
                workspace.build_scaling(wp.array(dynamic, device=device), epsilon_d=10.0, generation=generation),
                MonolithicLinearStatus.INVALID_DIAGONAL,
            )
            with test.assertRaises(ValueError):
                workspace.operator.matvec(workspace.scale, workspace._ap, workspace._ap, 1.0, 0.0)


def test_lambda_and_warm_start(test, device):
    """Invalidate factors after lambda changes and reject stale solution generations."""
    workspace, generation = _ready_system(device)
    rhs, y = wp.ones(11, dtype=float, device=device), wp.zeros(11, dtype=float, device=device)
    config = _pcg_config(linear_tolerance=1e-5)

    def solve(warm=MonolithicPcgWarmStart.ZERO):
        return workspace.solve_pcg(rhs, y, generation=generation, warm_start=warm, config=config)

    test.assertEqual(solve(MonolithicPcgWarmStart.SAME_GENERATION).status, MonolithicLinearStatus.STALE_GENERATION)
    test.assertEqual(solve().status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(solve(MonolithicPcgWarmStart.SAME_GENERATION).iterations, 0)
    test.assertEqual(workspace.set_regularization(0.5, generation=generation), MonolithicLinearStatus.SUCCESS)
    test.assertEqual(solve().status, MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
    with test.assertRaises(ValueError):
        workspace.preconditioner.matvec(rhs, y, y, 1.0, 0.0)
    test.assertEqual(
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=1e-10),
        MonolithicLinearStatus.SUCCESS,
    )
    result = solve(MonolithicPcgWarmStart.SAME_GENERATION)
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    np.testing.assert_allclose(
        y.numpy(),
        np.linalg.solve(workspace.densify_for_test(generation=generation).scaled_matrix, rhs.numpy()),
        atol=1e-4,
    )
    changed = MonolithicLinearGeneration(1, 1, 2, 2)
    workspace.begin_assembly(changed)
    test.assertEqual(solve(MonolithicPcgWarmStart.SAME_GENERATION).status, MonolithicLinearStatus.STALE_GENERATION)
    test.assertEqual(workspace.recover_delta(y, rhs, generation=generation), MonolithicLinearStatus.STALE_GENERATION)


@wp.kernel
def _two_block_matrix(
    aq: wp.array2d[float], ax: _MonolithicMat33Triplets, triplets: _MonolithicScalarTriplets, cross: float
):
    _accumulate_monolithic_actor_q_entry(aq, triplets, 0, 0, 1.0)
    _accumulate_monolithic_internal_x_block(ax, triplets, 0, 0, wp.diag(wp.vec3(1.0)), 1)
    _append_monolithic_scalar_triplet(triplets, 0, 1, cross)
    _append_monolithic_scalar_triplet(triplets, 1, 0, cross)


def _two_block_system(device, cross):
    workspace = MonolithicLinearWorkspace(
        MonolithicLinearLayout(1, 1),
        MonolithicLinearCapacities(12, 1, 0, 100, 102),
        wp.array([0], dtype=int, device=device),
        device=device,
    )
    generation = MonolithicLinearGeneration(1, 0, 1, 1)
    a = workspace.begin_assembly(generation)
    wp.launch(
        _two_block_matrix, 1, [a.aq_actor_dense, a.ax_internal_triplets, a.global_scalar_triplets, cross], device=device
    )
    workspace.finalize_assembly(generation=generation)
    workspace.build_scaling(wp.ones(4, dtype=float, device=device), epsilon_d=0.01, generation=generation)
    workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)
    return workspace, generation


def test_pcg_curvature_guards(test, device):
    """Detect negative and near-zero positive curvature before dividing or updating y."""
    for cross, config, expected in [
        (2.0, _pcg_config(), MonolithicLinearStatus.NON_POSITIVE_CURVATURE),
        (0.5, _pcg_config(curvature_absolute_tolerance=2.0), MonolithicLinearStatus.NEAR_ZERO_CURVATURE),
    ]:
        workspace, generation = _two_block_system(device, cross)
        rhs = wp.array([1.0, -1.0, 0.0, 0.0], dtype=float, device=device)
        y = wp.zeros(4, dtype=float, device=device)
        result = workspace.solve_pcg(
            rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config
        )
        test.assertEqual(result.status, expected, result)
        test.assertEqual(result.iterations, 0)
        np.testing.assert_array_equal(y.numpy(), 0)
        test.assertEqual(result.rho, 1.0)
        if cross == 2.0:
            test.assertLess(result.min_p_ap, 0)


def test_pcg_true_block_gate_and_replacement(test, device):
    """Fail a hidden q block then recover using actual residual replacements."""
    workspace, generation = _two_block_system(device, 1e-6)
    rhs = wp.array([1e-5, 1.0, 0.0, 0.0], dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)
    result = workspace.solve_pcg(
        rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config(maximum_iterations=1)
    )
    test.assertEqual(result.status, MonolithicLinearStatus.MAX_ITERATIONS, result)
    test.assertLess(result.rho, 1e-4)
    test.assertGreater(result.rho_q, 0.01)
    result = workspace.solve_pcg(
        rhs,
        y,
        generation=generation,
        warm_start=MonolithicPcgWarmStart.ZERO,
        config=_pcg_config(true_residual_interval=1),
    )
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    test.assertGreater(result.residual_replacements, 0)
    test.assertLessEqual(result.rho_q, 1e-4)


def test_pcg_stagnation_and_bad_preconditioner(test, device):
    """Report stagnation and invalid preconditioned products without accepting them."""
    workspace, generation = _two_block_system(device, 0.9)
    rhs = wp.array([1.0, 0.0, 0.0, 0.0], dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)
    result = workspace.solve_pcg(
        rhs,
        y,
        generation=generation,
        warm_start=MonolithicPcgWarmStart.ZERO,
        config=_pcg_config(true_residual_interval=1, stagnation_window=1, stagnation_minimum_reduction=0.5),
    )
    test.assertEqual(result.status, MonolithicLinearStatus.STAGNATION, result)
    original = workspace.preconditioner._matvec

    def negative(x, affine, out, alpha, beta):
        original(x, affine, out, -alpha, beta)

    workspace.preconditioner._matvec = negative
    result = workspace.solve_pcg(
        rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
    )
    test.assertEqual(result.status, MonolithicLinearStatus.NON_POSITIVE_PRECONDITIONED_RESIDUAL, result)
    test.assertLess(result.min_r_z, 0)
    np.testing.assert_array_equal(y.numpy(), 0)


def test_pcg_nonfinite_and_zero(test, device):
    """Reject nonfinite operator output and return exact zero without entering PCG."""
    workspace, generation = _ready_system(device)
    rhs, y = wp.zeros(11, dtype=float, device=device), wp.full(11, 7.0, dtype=float, device=device)
    result = workspace.solve_pcg(
        rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
    )
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
    test.assertEqual(result.iterations, 0)
    test.assertEqual(result.true_residual_checks, 1)
    test.assertTrue(np.isnan(result.min_p_ap))
    rhs.fill_(1.0)
    original = workspace.operator._matvec

    def nonfinite(x, affine, out, alpha, beta):
        original(x, affine, out, alpha, beta)
        out.fill_(float("nan"))

    workspace.operator._matvec = nonfinite
    result = workspace.solve_pcg(
        rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
    )
    test.assertEqual(result.status, MonolithicLinearStatus.NONFINITE_ITERATION)
    test.assertEqual(result.iterations, 0)
    test.assertTrue(np.isnan(result.rho))


def test_preconditioner_pivots_and_repeated_factor(test, device):
    """Include repeated contact columns and reject indefinite owner Cholesky pivots."""
    workspace, generation = _ready_system(device, contact=False)
    # Bad owner values cannot be hidden by the independent valid global diagonal.
    workspace.aq_actor_dense.assign(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32))
    test.assertEqual(
        workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0),
        MonolithicLinearStatus.NON_POSITIVE_PIVOT,
    )
    workspace, generation = _ready_system(device, contact=False)
    next_generation = MonolithicLinearGeneration(1, 1, 2, 2)
    a = workspace.begin_assembly(next_generation)
    wp.launch(_owners, 1, [a.aq_actor_dense, a.ax_internal_triplets, a.global_scalar_triplets], device=device)
    wp.launch(_masked_contact, 1, [a.contact_factors, a.global_scalar_triplets, 2.0, 0], device=device)
    workspace.finalize_assembly(generation=next_generation)
    workspace.build_scaling(wp.ones(11, dtype=float, device=device), epsilon_d=0.01, generation=next_generation)
    workspace.factor_actor_preconditioner(generation=next_generation, pivot_tolerance=0.0)
    scaled = workspace.densify_for_test(generation=next_generation).scaled_matrix
    np.testing.assert_allclose(workspace._x_preconditioner.numpy()[0], scaled[2:5, 2:5], atol=1e-7)


def test_pcg_recursive_drift(test, device):
    """Replace a falsely zero recursive residual before allowing linear success."""
    workspace, generation = _ready_system(device)
    rhs, y = wp.ones(11, dtype=float, device=device), wp.zeros(11, dtype=float, device=device)
    products_of = workspace._products_of
    injected = []

    def drift(a, b):
        if a is workspace._r and b is workspace._r and not injected:
            a.zero_()
            injected.append(True)
        products_of(a, b)

    with patch.object(workspace, "_products_of", side_effect=drift):
        result = workspace.solve_pcg(
            rhs,
            y,
            generation=generation,
            warm_start=MonolithicPcgWarmStart.ZERO,
            config=_pcg_config(linear_tolerance=1e-5),
        )
    test.assertEqual(injected, [True])
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    test.assertGreater(result.iterations, 1)
    test.assertGreater(result.residual_replacements, 0)
    test.assertGreaterEqual(result.true_residual_checks, 3)
    np.testing.assert_allclose(
        y.numpy(),
        np.linalg.solve(workspace.densify_for_test(generation=generation).scaled_matrix, rhs.numpy()),
        atol=3e-5,
    )


def test_pcg_input_validation(test, device):
    """Reject invalid scalar configuration and array aliases before modifying the solution."""
    for kwargs in [
        {"maximum_iterations": 0},
        {"true_residual_interval": 0},
        {"stagnation_window": -1},
        {"maximum_iterations": 1.5},
        {"residual_floor_q": 0.0},
        {"residual_floor_x": 1e-300},
        {"residual_floor_global": float("inf")},
        {"linear_tolerance": float("nan")},
        {"curvature_relative_tolerance": -1.0},
        {"stagnation_minimum_reduction": 1.0},
    ]:
        with test.assertRaises(ValueError):
            _pcg_config(**kwargs)
    workspace, generation = _ready_system(device)
    rhs, y = wp.ones(11, dtype=float, device=device), wp.full(11, 7.0, dtype=float, device=device)
    for rhs_arg, y_arg, config in [
        (rhs, rhs, _pcg_config()),
        (rhs, y, _pcg_config(maximum_iterations=101)),
        (wp.ones(10, dtype=float, device=device), y, _pcg_config()),
        (wp.ones(11, dtype=wp.float64, device=device), y, _pcg_config()),
    ]:
        result = workspace.solve_pcg(
            rhs_arg, y_arg, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config
        )
        test.assertEqual(result.status, MonolithicLinearStatus.INVALID_ARGUMENT)
        np.testing.assert_array_equal(y.numpy(), 7.0)
    for value in [-1.0, float("nan"), float("inf")]:
        test.assertEqual(
            workspace.set_regularization(value, generation=generation), MonolithicLinearStatus.INVALID_ARGUMENT
        )
    y.fill_(float("nan"))
    test.assertEqual(workspace.recover_delta(y, rhs, generation=generation), MonolithicLinearStatus.NONFINITE_ITERATION)


def test_pcg_extreme_true_residuals(test, device):
    """Prevent tiny or huge float32 squared residuals from reporting false success."""
    workspace, generation = _two_block_system(device, 0.0)
    rhs = wp.full(4, 1e-25, dtype=float, device=device)
    y = wp.zeros(4, dtype=float, device=device)
    config = _pcg_config(residual_floor_global=1e-30, residual_floor_q=1e-30, residual_floor_x=1e-30)
    result = workspace.solve_pcg(rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config)
    test.assertEqual(result.status, MonolithicLinearStatus.NON_POSITIVE_PRECONDITIONED_RESIDUAL, result)
    np.testing.assert_allclose((result.rho, result.rho_q, result.rho_x), 1.0, atol=1e-7)
    np.testing.assert_array_equal(y.numpy(), 0)
    # Check the actual operator directly: finite vector norms must survive square overflow.
    rhs.fill_(1e20)
    y.fill_(0.999e20)
    workspace._status.zero_()
    ratios = workspace._true_residual(rhs, y, config)
    expected = abs(float(rhs.numpy()[0]) - float(y.numpy()[0])) / float(rhs.numpy()[0])
    np.testing.assert_allclose(ratios, expected, rtol=2e-6)
    test.assertEqual(workspace._read_status(), MonolithicLinearStatus.SUCCESS)
    test.assertGreater(min(ratios), config.linear_tolerance)
    # Each block has its own scale even when a neighboring block is much larger.
    rhs.assign(np.array([1e-25, 1e20, 1e20, 1e20], dtype=np.float32))
    y.assign(np.array([0, 1e20, 1e20, 1e20], dtype=np.float32))
    ratios = workspace._true_residual(rhs, y, config)
    test.assertAlmostEqual(ratios[1], 1.0, places=6)
    test.assertEqual(ratios[2], 0.0)
    rhs.fill_(3e38)
    y.zero_()
    workspace._true_residual(rhs, y, config)
    test.assertEqual(workspace._read_status(), MonolithicLinearStatus.NONFINITE_ITERATION)
    workspace, generation = _two_block_system(device, 1e-15)
    rhs.assign(np.array([1e-25, 1.0, 0.0, 0.0], dtype=np.float32))
    result = workspace.solve_pcg(rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=config)
    test.assertNotEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    test.assertGreater(result.rho_q, config.linear_tolerance)


def test_pcg_partial_overlap(test, device):
    """Reject partially overlapping solution and RHS views without mutating either."""
    workspace, generation = _two_block_system(device, 0.0)
    for reverse in [False, True]:
        original = np.arange(5, dtype=np.float32)
        storage = wp.array(original, device=device)
        rhs, y = (storage[1:], storage[:4]) if reverse else (storage[:4], storage[1:])
        result = workspace.solve_pcg(
            rhs, y, generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
        )
        test.assertEqual(result.status, MonolithicLinearStatus.INVALID_ARGUMENT, result)
        np.testing.assert_array_equal(storage.numpy(), original)
    # Adjacent views share an allocation but have disjoint byte ranges.
    storage = wp.array(np.r_[np.ones(4), np.zeros(4)].astype(np.float32), device=device)
    result = workspace.solve_pcg(
        storage[:4], storage[4:], generation=generation, warm_start=MonolithicPcgWarmStart.ZERO, config=_pcg_config()
    )
    test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS, result)
    np.testing.assert_array_equal(storage.numpy(), 1.0)


class TestMonolithicLinear(unittest.TestCase):
    """Exercise assembly independently on each available device."""


for device in get_test_devices():
    for test_function in [
        test_pcg_extreme_true_residuals,
        test_pcg_partial_overlap,
        test_pcg_recursive_drift,
        test_pcg_input_validation,
        test_pcg_dense_and_allocations,
        test_scaling_guards,
        test_lambda_and_warm_start,
        test_pcg_curvature_guards,
        test_pcg_true_block_gate_and_replacement,
        test_pcg_stagnation_and_bad_preconditioner,
        test_pcg_nonfinite_and_zero,
        test_preconditioner_pivots_and_repeated_factor,
        test_scaling_operator_preconditioner,
        test_owner_contact_oracle,
        test_generation_freeze,
        test_capacity_and_invalid_contributions,
        test_fixed_contact_and_weight,
        test_shared_fixture_layout,
        test_nonfinite_reduction,
        test_fixed_prefix_contact_and_reset,
        test_fixed_pattern_validation,
        test_fixed_raw_validation,
    ]:
        add_function_test(TestMonolithicLinear, test_function.__name__, test_function, devices=[device])


if __name__ == "__main__":
    unittest.main()
