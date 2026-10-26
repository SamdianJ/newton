# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit actual native complete-contact assembly without global dense matrices."""

import unittest
import warnings
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp
import warp.sparse as sparse

from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_contact_actor_pairs import physical
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_tooling_contact_stream import grid_scene
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import contact_audit as comparison
from scripts.monolithic_reference.contact_full_oracle import evaluate_full, evaluate_stream
from scripts.monolithic_reference.contact_routes import pooled_audit_route
from scripts.monolithic_reference.workload_scene import build_scene


def reference(run, current, start):
    stiffness, mu, velocity, width, radius = run._contact.parameters
    return evaluate_stream(
        run.model,
        run._actors,
        current,
        start,
        stiffness=stiffness,
        mu=mu,
        width=width,
        radius=radius,
        falloff=velocity * 0.001,
        alignment_policy=run._contact.alignment_policy,
    )


def native_grid(dimension, device, *, graph=False):
    """Use the builder's supported lumped grid mass for a native geometry audit."""
    model, _, frame = grid_scene(dimension, device=device)
    run = SolverMonolithic(
        model,
        mass_mode="lumped",
        friction_velocity_scale=0.01,
        alignment_policy="mask",
        pcg_mode="production",
        use_cuda_graph=graph,
    )
    return run, frame


def test_complete_native_audit(test, device):
    """Compare complete-candidate and spatial raw CSR, E/r and both endpoint forces under real graph modes."""
    for recipe in ("tiny-contact", "actor-coupling-permuted"):
        for algorithm in ("pooled", "spatial"):
            for graph in (False, True) if device.is_cuda else (False,):
                _, run, state, _ = build_scene(recipe, device=device, graph=graph, contact_dataflow=algorithm)
                frame = physical(state)
                with patch.object(run._linear, "densify_for_test", side_effect=AssertionError("global dense B")):
                    result = comparison.audit_contact_state(run, frame, frame, 0.001)
                    again = comparison.audit_contact_state(run, frame, frame, 0.001)
                test.assertEqual(result["status"], "PASS_LOCAL_COMPLETE_CONTACT_STATE")
                test.assertEqual(result["full_count"], run._contact.count)
                # Pool publication retains identical repeated results.
                test.assertEqual(result["errors"], again["errors"])
                test.assertEqual(result["state_sha256"], again["state_sha256"])
                test.assertLess(result["errors"]["tangent_relative"], 1e-5)


def test_legacy_force_publication(test, device):
    """Publish particle-shape audit force operands that match the independent oracle."""
    _, run, state, _ = build_scene("tiny-contact", device=device, graph=device.is_cuda)
    frame = physical(state)
    wanted = reference(run, frame, frame)["full"]
    actual = comparison._contact_at(run, frame, frame, 0.001)
    test.assertGreater(np.linalg.norm(wanted["particle_force"]), 0)
    for name in ("particle_force", "body_wrench", "joint_force"):
        np.testing.assert_allclose(actual[name], wanted[name], rtol=1e-8, atol=1e-10)


def test_tet_static_reactions(test, device):
    """Audit T-static reactions and balance both endpoints about the world origin."""
    for algorithm in ("pooled", "spatial"):
        model, _, start = grid_scene(2, device=device)
        model.shape_body.assign(np.full(model.shape_count, -1))
        context = pooled_audit_route() if algorithm == "pooled" else nullcontext()
        with context:
            run = SolverMonolithic(
                model,
                mass_mode="lumped",
                friction_velocity_scale=0.01,
                alignment_policy="mask",
                pcg_mode="production",
                use_cuda_graph=device.is_cuda,
            )
        frame = {name: value.copy() for name, value in start.items()}
        frame["x"] += [0.0002, 0, -0.0001]
        wanted = reference(run, frame, start)["full"]
        actual = comparison._contact_at(run, frame, start, 0.001, sparse=True)
        test.assertGreater(np.linalg.norm(wanted["static_wrench"]), 0)
        comparison._check_contact_sparse(actual, wanted)
        force = actual["particle_force"].sum(axis=0) + actual["static_wrench"][:, :3].sum(axis=0)
        torque = np.cross(frame["x"], actual["particle_force"]).sum(axis=0) + actual["static_wrench"][:, 3:].sum(axis=0)
        np.testing.assert_allclose(force, 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(torque, 0, rtol=0, atol=1e-10)


def test_sparse_bsr_validation(test, device):
    """Export only used scalar CSR coordinates and reject malformed offsets, indices and values."""
    matrix = sparse.bsr_zeros(3, 3, wp.float32, device=device, nnz_capacity=3)
    matrix.offsets.assign([0, 2, 2, 3])
    matrix.columns.assign([0, 2, 1])
    matrix.values.assign([2, 5, -3])
    got = comparison._sparse_bsr(matrix)
    np.testing.assert_array_equal(got["row"], [0, 0, 2])
    np.testing.assert_array_equal(got["column"], [0, 2, 1])
    np.testing.assert_array_equal(got["value"], [2, 5, -3])
    for name, data in (
        ("offsets", [1, 2, 2, 3]),
        ("offsets", [0, 2, 1, 3]),
        ("offsets", [0, 2, 2, 4]),
        ("columns", [0, 0, 1]),
        ("columns", [0, 3, 1]),
        ("values", [2, np.nan, -3]),
    ):
        previous = getattr(matrix, name).numpy().copy()
        getattr(matrix, name).assign(data)
        with test.assertRaises(ValueError):
            comparison._sparse_bsr(matrix)
        getattr(matrix, name).assign(previous)
    matrix.offsets.zero_()
    matrix.values.fill_(np.nan)
    empty = comparison._sparse_bsr(matrix)
    test.assertEqual(empty["value"].size, 0)


def test_connected_native_sparse(test, device):
    """Audit connected-volume native assembly against all independent sparse coordinates."""
    run, frame = native_grid(8 if device.is_cuda else 6, device=device, graph=device.is_cuda)
    with patch.object(run._linear, "densify_for_test", side_effect=AssertionError("global dense B")):
        result = comparison.audit_contact_state(run, frame, frame, 0.001)
    test.assertGreater(run._layout.scalar_dof_count, 1000)
    test.assertGreater(result["full_count"], 1000)
    test.assertGreater(result["full_B_coordinates"], 0)
    test.assertLess(result["errors"]["tangent_relative"], 1e-5)


def test_independent_alignment_policy(test, device):
    """Retain all keys while selecting whole E/r/B/force terms with the C-IE collider normal."""
    run = contact_solver(contact_scene(device, "R-R"), mode="production", graph=device.is_cuda, alignment_policy="mask")
    start = physical(contact_state(run, "R-R"))
    frame = {name: value.copy() for name, value in start.items()}
    frame["com"][1] -= [0.03, 0.001, 0]
    c = run._contact
    stiffness, mu, velocity, width, radius = c.parameters
    options = {"radius": radius, "stiffness": stiffness, "mu": mu, "width": width, "falloff": velocity * 0.001}
    raw = evaluate_full(run.model, run._actors, frame, start, **options)
    selected = evaluate_full(run.model, run._actors, frame, start, **options, alignment_policy="mask")
    test.assertEqual(raw["records"].keys(), selected["records"].keys())
    masked = {key for key, record in selected["records"].items() if record["alignment_masked"]}
    test.assertTrue(masked)
    test.assertTrue(any(np.any(raw["records"][key]["tangent"]) for key in masked))
    for key in masked:
        for name in ("energy", "residual", "tangent", "world_residual", "body_wrench", "particle_force"):
            np.testing.assert_array_equal(selected["records"][key][name], 0)
    result = comparison.audit_contact_state(run, frame, start, 0.001)
    test.assertEqual(result["full_count"], len(raw["records"]))
    with test.assertRaisesRegex(ValueError, "contact alignment"):
        evaluate_stream(run.model, run._actors, frame, start, **options, alignment_policy="guard")


class TestNativeContactAudit(unittest.TestCase):
    def test_missing_cross_block(self):
        """Reject a missing A-R cross block using all sparse matrix coordinates."""
        run = contact_solver(contact_scene("cpu", "A-R"), mode="production")
        frame = physical(contact_state(run, "A-R"))
        wanted = reference(run, frame, frame)["full"]
        actual = comparison._contact_at(run, frame, frame, 0.001, sparse=True)
        comparison._check_contact_sparse(actual, wanted)
        b = actual["tangent"]
        arm = run._layout.indices("articulation")
        rigid = run._rigid.rows_numpy.ravel()
        mask = np.isin(b["row"], arm) & np.isin(b["column"], rigid)
        self.assertGreater(np.linalg.norm(b["value"][mask]), 0)
        b["value"][mask] = 0
        with self.assertRaises(ValueError):
            comparison._check_contact_sparse(actual, wanted)

    def test_capacity_taint_and_nonfinite(self):
        """Reject truncated or invalid complete-state audits without weakening force or B gates."""
        run = contact_solver(contact_scene("cpu", "A-R"), contact_factor_capacity=0)
        frame = physical(contact_state(run, "A-R"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "Capacity overflow"):
                comparison.audit_contact_state(run, frame, frame, 0.001)
        frame["com"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite native states"):
            comparison.audit_contact_state(run, frame, frame, 0.001)


for function in (
    test_complete_native_audit,
    test_legacy_force_publication,
    test_tet_static_reactions,
    test_sparse_bsr_validation,
    test_connected_native_sparse,
    test_independent_alignment_policy,
):
    add_function_test(TestNativeContactAudit, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
