# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the monolithic assembly contract with synthetic contributions."""

import unittest

import numpy as np
import warp as wp
import warp.sparse as sparse

from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
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


class TestMonolithicLinear(unittest.TestCase):
    """Exercise assembly independently on each available device."""


for device in get_test_devices():
    for test_function in [
        test_owner_contact_oracle,
        test_generation_freeze,
        test_capacity_and_invalid_contributions,
        test_fixed_contact_and_weight,
        test_shared_fixture_layout,
    ]:
        add_function_test(TestMonolithicLinear, test_function.__name__, test_function, devices=[device])


if __name__ == "__main__":
    unittest.main()
