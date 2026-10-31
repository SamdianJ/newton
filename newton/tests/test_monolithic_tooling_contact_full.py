# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify independent complete enumeration before introducing spatial clipping."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.solvers.experimental.monolithic import build_smooth_sdf
from newton.tests.test_monolithic_contact_actor_pairs import native_batch, perturb, physical, query
from newton.tests.test_monolithic_contact_components import batch_matrix
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_full_oracle import audit_keys, evaluate_full, export_mesh_fields
from scripts.monolithic_reference.coupled_fixtures import make_solver, make_tiny


def native_terms(source):
    """Return E, r and raw B of a native producer's latest evaluation."""
    batch = getattr(source, "_batch", source)
    return batch.pool.energy.result.numpy()[:2].copy(), batch.pool.residual.result.numpy().copy(), batch_matrix(batch)


def compare(test, run, terms, state, start, *, mesh_fields=None):
    c = run._contact
    energy, residual, matrix = terms
    stiffness, mu, velocity, width, radius = c.parameters
    full = evaluate_full(
        run.model,
        run._actors,
        physical(state),
        start,
        radius=radius,
        stiffness=stiffness,
        mu=mu,
        width=width,
        falloff=velocity * 0.001,
        mesh_fields=mesh_fields,
    )
    test.assertEqual(len(full["records"]), c.count)
    np.testing.assert_allclose(energy, full["energy"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(residual, full["residual"], rtol=1e-8, atol=1e-10)
    test.assertLess(np.linalg.norm(matrix - full["tangent"]) / max(1, np.linalg.norm(full["tangent"])), 1e-5)
    keys = list(full["records"])
    test.assertFalse(audit_keys(full, keys)["missing_keys"])
    test.assertTrue(audit_keys(full, [])["missing_contributing_keys"])
    with test.assertRaises(ValueError):
        audit_keys(full, keys + keys[:1])
    # Corrupt the production enumeration without changing Model inputs.
    with patch.object(c, "count", 0):
        again = evaluate_full(
            run.model,
            run._actors,
            physical(state),
            start,
            radius=radius,
            stiffness=stiffness,
            mu=mu,
            width=width,
            falloff=velocity * 0.001,
            mesh_fields=mesh_fields,
        )
    np.testing.assert_array_equal(again["residual"], full["residual"])
    test.assertEqual(again["records"].keys(), full["records"].keys())
    return full


def test_complete_pairs(test, device):
    """Compare all new pairs under two scalar layouts against dense E/r/B."""
    for kind, permuted in itertools.product(("A-R", "R-R", "R-T", "R-static"), (False, True)):
        with test.subTest(kind=kind, permuted=permuted):
            run = contact_solver(contact_scene(device, kind, before=permuted), permuted=permuted)
            batch = native_batch(run)
            state = contact_state(run, kind)
            start = physical(state)
            query(batch, run, state, start=True)
            perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
            query(batch, run, state)
            compare(test, run, native_terms(batch), state, start)


def test_legacy_and_published_force(test, device):
    """Check particle-shape enumeration and native two-endpoint published forces."""
    model = make_tiny("normal_loading", device)
    model.joint_q.fill_(0.005)
    run = make_solver(model, "normal_loading", mode="production", graph=device.is_cuda)
    batch = native_batch(run)
    state = run.create_state(dt=0.001)
    start = physical(state)
    query(batch, run, state, start=True)
    query(batch, run, state)
    compare(test, run, native_terms(batch), state, start)
    for kind in ("chain", "R-static"):
        run = contact_solver(contact_scene(device, kind), mode="production", graph=device.is_cuda)
        state = contact_state(run, kind)
        start = physical(state)
        run.step(state, state, run.model.control(), dt=0.001)
        test.assertTrue(run.last_stats.converged)
        full = compare(test, run, native_terms(run._contact), state, start)
        force = run.get_contact_forces(state)
        np.testing.assert_allclose(force.particle_force, full["particle_force"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(force.body_wrench, full["body_wrench"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(force.joint_force, full["joint_force"], rtol=1e-8, atol=1e-10)


class TestContactFullOracle(unittest.TestCase):
    def test_nonfinite_native_input(self):
        """Fail an invalid native state rather than classify it as a spatial omission."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = physical(contact_state(run, "R-T"))
        start = {name: value.copy() for name, value in state.items()}
        state["x"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite native states"):
            evaluate_full(run.model, run._actors, state, start, radius=0.001)

    def test_invalid_candidate_identity(self):
        """Reject malformed keys before coercion can hide an omitted contact."""
        full = {"records": {(-1, 0, 0, 0): {}}}
        for key in ((-1, 0, 0.5, 0), (-1, 0, False, 0), (-1, 0, 0), (-1, 0, 0, 0, 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_keys(full, [key])

    def test_tangent_only_omission(self):
        """Detect a missing zero-energy, zero-residual sample with a nonzero tangent."""
        key = (-1, 0, 0, 0)
        record = {
            "energy": np.zeros(2),
            "residual": np.zeros(1),
            "tangent": np.ones((1, 1)),
            "world_residual": np.zeros(3),
        }
        full = {"records": {key: record}}
        self.assertEqual(audit_keys(full, [])["missing_contributing_keys"], [key])


def test_mesh_input(test, device):
    """Compare mesh source samples and exported C2 targets without production queries."""
    if not device.is_cuda:
        test.skipTest("Model texture SDF construction requires CUDA")
    model = contact_scene(device, "A-R", mesh=True)
    sdf = build_smooth_sdf(model, representation="c2-bspline")
    run = contact_solver(model, smooth_sdf=sdf)
    batch = native_batch(run)
    state = contact_state(run, "A-R")
    start = physical(state)
    query(batch, run, state, start=True)
    perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
    query(batch, run, state)
    fields = export_mesh_fields(model, sdf)
    compare(test, run, native_terms(batch), state, start, mesh_fields=fields)
    with test.assertRaises(ValueError):
        evaluate_full(model, run._actors, physical(state), start, radius=0.001)


def test_curved_target(test, device):
    """Compare sphere target curvature and local transforms."""
    model = contact_scene(device, "R-T")
    kinds = model.shape_type.numpy()
    scales = model.shape_scale.numpy()
    shape = int(np.flatnonzero(model.shape_body.numpy() == model.body_label.index("rigid_0"))[0])
    kinds[shape] = 3
    scales[shape] = [0.02, 0, 0]
    model.shape_type.assign(kinds)
    model.shape_scale.assign(scales)
    transforms = model.shape_transform.numpy()
    transforms[shape] = np.asarray(wp.transform((0, 0.001, 0.002), wp.quat_identity()))
    model.shape_transform.assign(transforms)
    run = contact_solver(model)
    batch = native_batch(run)
    state = contact_state(run, "R-T")
    start = physical(state)
    query(batch, run, state, start=True)
    perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
    query(batch, run, state)
    compare(test, run, native_terms(batch), state, start)


def test_frozen_complete_pairs(test, device):
    """Match frozen normal selection, in-step work and PSD factors under endpoint motion."""
    for kind, mu in itertools.product(("A-R", "R-R", "R-T", "R-static"), (0.0, 0.5)):
        with test.subTest(kind=kind, mu=mu):
            run = contact_solver(
                contact_scene(device, kind),
                friction_coefficient=mu,
                alignment_policy="frozen",
                box_smoothing=25e-6,
                alignment_smoothing=0.1,
            )
            state, batch = contact_state(run, kind), native_batch(run)
            values = physical(state)
            turn = np.zeros(run._layout.scalar_dof_count)
            turn[run._rigid.actors[0].dof_start + 5] = 1
            perturb(run, state, values, turn, 0.05)
            start = physical(state)
            query(batch, run, state, start=True)
            direction = np.linspace(-0.2, 0.3, run._layout.scalar_dof_count)
            perturb(run, state, start, direction, 1e-7)
            query(batch, run, state)
            test.assertEqual(batch.flags.numpy()[0], 0)
            stiffness, _, velocity, width, radius = run._contact.parameters
            full = evaluate_full(
                run.model,
                run._actors,
                physical(state),
                start,
                radius=radius,
                stiffness=stiffness,
                width=width,
                mu=mu,
                falloff=velocity * 0.001,
                alignment_policy="frozen",
                box_smoothing=25e-6,
                alignment_smoothing=0.1,
            )
            energy, residual, matrix = native_terms(batch)
            np.testing.assert_allclose(energy, full["energy"], rtol=2e-8, atol=1e-11)
            np.testing.assert_allclose(residual, full["residual"], rtol=2e-7, atol=1e-9)
            test.assertLess(np.linalg.norm(matrix - full["tangent"]) / max(1.0, np.linalg.norm(full["tangent"])), 1e-6)
            force = batch.pool.force.result.numpy()
            np.testing.assert_allclose(
                force[: 3 * run.model.particle_count].reshape(-1, 3), full["particle_force"], rtol=2e-7, atol=1e-9
            )
            np.testing.assert_allclose(
                force[3 * run.model.particle_count :].reshape(-1, 6), full["body_wrench"], rtol=2e-7, atol=1e-9
            )


add_function_test(TestContactFullOracle, "test_complete_pairs", test_complete_pairs, devices=get_test_devices())
add_function_test(
    TestContactFullOracle,
    "test_legacy_and_published_force",
    test_legacy_and_published_force,
    devices=get_test_devices(),
)
add_function_test(TestContactFullOracle, "test_mesh_input", test_mesh_input, devices=get_test_devices())
add_function_test(TestContactFullOracle, "test_curved_target", test_curved_target, devices=get_test_devices())
add_function_test(
    TestContactFullOracle, "test_frozen_complete_pairs", test_frozen_complete_pairs, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
