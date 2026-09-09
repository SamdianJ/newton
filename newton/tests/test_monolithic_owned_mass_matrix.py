# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the opt-in mass kernel against Newton's articulation dynamics."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic import articulation
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _branched_model(device):
    builder = newton.ModelBuilder(gravity=(1.3, -8.7, -2.0))
    links = [
        builder.add_link(
            mass=0.5 + i * 0.2,
            com=(0.03, -0.02, 0.01),
            inertia=wp.mat33(0.03, 0.002, 0.001, 0.002, 0.02, 0.003, 0.001, 0.003, 0.04),
        )
        for i in range(4)
    ]
    passive = dict.fromkeys(("armature", "damping", "friction", "limit_ke", "limit_kd", "target_ke", "target_kd"), 0.0)
    root = builder.add_joint_revolute(-1, links[2], axis=newton.Axis.Z, **passive)
    fixed = builder.add_joint_fixed(
        links[2], links[0], parent_xform=wp.transform((0.1, 0.02, -0.03), wp.quat_identity())
    )
    slide = builder.add_joint_prismatic(links[2], links[1], axis=newton.Axis.X, **passive)
    tip = builder.add_joint_revolute(
        links[0],
        links[3],
        axis=newton.Axis.Y,
        parent_xform=wp.transform((0.05, -0.08, 0.02), wp.quat_identity()),
        **passive,
    )
    builder.add_articulation([root, fixed, slide, tip])
    return builder.finalize(device=device)


def test_mass_candidate_parity(test, device):
    """Match mass, kinetic energy and inertia force across mixed and branched A-B-A candidates."""
    for model in (build_tiny_cpu_fixture(device=device).model, _branched_model(device)):
        state = model.state()
        reference = articulation.MonolithicArticulationWorkspace(model)
        owned = articulation.MonolithicArticulationWorkspace(model, use_optimized_articulation_mass_matrix=True)
        n = model.joint_dof_count
        a = None
        for i in range(9):
            q = (
                np.linspace(-0.3, 0.2, n, dtype=np.float32)
                if i % 2 == 0
                else np.linspace(0.2, 0.5 + i * 0.1, n, dtype=np.float32)
            )
            state.joint_q.assign(q)
            qd = np.linspace(-0.4, 0.7 if i % 2 == 0 else 1.2, n, dtype=np.float32)
            state.joint_qd.assign(qd)
            articulation.eval_articulation_passive_candidate(model, state, reference)
            articulation.eval_articulation_passive_candidate(model, state, owned)
            for name in ("M", "g", "C"):
                np.testing.assert_allclose(
                    getattr(owned, name).numpy(), getattr(reference, name).numpy(), rtol=1e-5, atol=1e-8
                )
            mass = owned.M.numpy()[0].copy()
            np.testing.assert_allclose(mass, mass.T, rtol=1e-5, atol=1e-8)
            test.assertGreater(np.linalg.eigvalsh(mass.astype(np.float64)).min(), 0.0)
            jacobian = owned.scratch.J.numpy()[0].astype(np.float64)
            # Coriolis reuses body_I_s for composite inertias after mass assembly.
            wp.launch(
                articulation.compute_body_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass, state.body_q, owned.scratch.body_I_s],
                device=device,
            )
            spatial_inertia = owned.scratch.body_I_s.numpy().astype(np.float64)
            body_velocity = (jacobian @ qd).reshape(-1, 6)
            energy = sum(
                0.5 * v @ spatial_inertia[child] @ v
                for v, child in zip(body_velocity, model.joint_child.numpy(), strict=True)
            )
            np.testing.assert_allclose(0.5 * qd @ mass @ qd, energy, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(mass @ qd, reference.M.numpy()[0] @ qd, rtol=1e-5, atol=1e-8)
            if i == 0:
                a = mass
            elif i % 2 == 0:
                np.testing.assert_array_equal(mass, a)


def test_actor_parity(test, device):
    """Match actor residual and tangent with zero and nonzero external generalized forces."""
    model = _branched_model(device)
    state = model.state()
    state.joint_q.assign(np.array([0.4, 0.02, -0.2], dtype=np.float32))
    state.joint_qd.assign(np.array([0.3, -0.2, 0.1], dtype=np.float32))
    n = model.joint_dof_count
    qdd = wp.array([0.7, -0.3, 1.2], dtype=float, device=device)
    force = wp.zeros(n, dtype=float, device=device)
    body_force = wp.zeros_like(force)
    results = []
    for enabled in (False, True):
        workspace = articulation.MonolithicArticulationWorkspace(model, use_optimized_articulation_mass_matrix=enabled)
        articulation.eval_articulation_passive_candidate(model, state, workspace)
        residual = wp.empty_like(force)
        tangent = wp.empty((n, n), dtype=float, device=device)
        rows, cols = (wp.empty(n * n, dtype=int, device=device) for _ in range(2))
        values = wp.empty(n * n, dtype=float, device=device)
        samples = []
        for external in (0.0, 0.4):
            force.fill_(external)
            wp.launch(
                articulation.eval_articulation_actor_residual,
                n,
                inputs=[0, 0, n, workspace.M, qdd, workspace.C, workspace.g, force, body_force, residual],
                device=device,
            )
            wp.launch(
                articulation.scatter_articulation_actor_tangent,
                (n, n),
                inputs=[0, n, 0, 0, 10000.0, workspace.M, tangent, rows, cols, values],
                device=device,
            )
            samples.append((residual.numpy().copy(), tangent.numpy().copy(), values.numpy().copy()))
        results.append(samples)
    for reference, owned in zip(results[0], results[1], strict=True):
        for expected, actual in zip(reference, owned, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_dispatch_and_allocation(test, device):
    """Keep optimized candidate dispatch allocation-free and independent of public mass evaluation."""
    for fallback in (False, True):
        model = _branched_model(device)
        if fallback:
            model._fk_articulation_level_start = None
        stream = wp.Stream(device) if device.is_cuda else None
        with patch.object(articulation, "eval_mass_matrix", side_effect=AssertionError("reference path called")):
            workspace = articulation.MonolithicArticulationWorkspace(
                model, stream=stream, use_optimized_articulation_mass_matrix=True
            )
            states = (model.state(), model.state())
            states[1].joint_q.assign(np.array([-0.4, 0.05, 0.6], dtype=np.float32))
            for state in states:
                articulation.eval_articulation_passive_candidate(model, state, workspace)
            arrays = (workspace.M, workspace.scratch.J, workspace.scratch.body_I_s)
            pointers = [value.ptr for value in arrays]
            allocator = model.device.get_allocator()
            with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
                for i in range(30):
                    articulation.eval_articulation_passive_candidate(model, states[i % 2], workspace)
            test.assertEqual(allocate.call_count, 0)
            test.assertEqual(pointers, [value.ptr for value in arrays])
        with patch.object(articulation, "eval_mass_matrix", wraps=articulation.eval_mass_matrix) as public:
            reference = articulation.MonolithicArticulationWorkspace(model)
            articulation.eval_articulation_passive_candidate(model, states[0], reference)
            test.assertEqual(public.call_count, 2)


def test_padding_overwrite(test, device):
    """Clear invalid padded output entries on every owned mass launch."""
    model = _branched_model(device)
    workspace = articulation.MonolithicArticulationWorkspace(model)
    state = model.state()
    articulation.eval_articulation_passive_candidate(model, state, workspace)
    wp.launch(
        articulation.compute_body_spatial_inertia,
        model.body_count,
        inputs=[model.body_inertia, model.body_mass, state.body_q, workspace.scratch.body_I_s],
        device=device,
    )
    n = model.joint_dof_count
    padded = wp.full((1, n + 2, n + 2), 123.0, dtype=float, device=device)
    wp.launch(
        articulation._eval_owned_mass_matrix,
        (n + 2, n + 2),
        inputs=[
            model.articulation_start,
            model.articulation_end,
            model.joint_child,
            model.joint_qd_start,
            workspace.scratch.body_I_s,
            workspace.scratch.J,
            padded,
        ],
        device=device,
    )
    actual = padded.numpy()[0]
    np.testing.assert_allclose(actual[:n, :n], workspace.M.numpy()[0], rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(actual[n:, :], 0.0)
    np.testing.assert_array_equal(actual[:, n:], 0.0)


def test_option_validation(test, device):
    """Require a construction-time strict bool and reject later option mutation."""
    model = build_tiny_cpu_fixture(device=device).model
    for value in (None, 0, 1, "true", np.bool_(True)):
        with test.assertRaisesRegex(TypeError, "use_optimized_articulation_mass_matrix"):
            articulation.MonolithicArticulationWorkspace(model, use_optimized_articulation_mass_matrix=value)
    workspace = articulation.MonolithicArticulationWorkspace(model)
    test.assertIs(workspace.use_optimized_articulation_mass_matrix, False)
    with test.assertRaises(AttributeError):
        workspace.use_optimized_articulation_mass_matrix = True


class TestMonolithicOwnedMassMatrix(unittest.TestCase):
    """Exercise owned articulation mass assembly on supported devices."""


for _test in (
    test_mass_candidate_parity,
    test_actor_parity,
    test_dispatch_and_allocation,
    test_padding_overwrite,
    test_option_validation,
):
    add_function_test(TestMonolithicOwnedMassMatrix, _test.__name__, _test, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
