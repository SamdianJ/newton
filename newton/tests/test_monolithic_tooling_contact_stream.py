# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete streamed contact audits without dense global storage."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton.tests.test_monolithic_contact_actor_pairs import perturb, physical
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_contact_pool import pooled_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import contact_full_oracle as oracle
from scripts.monolithic_reference.compare_contact_dataflows import _check_contact, _contact_at


def dense_tangent(result):
    b = result["tangent"]
    matrix = np.zeros(b["shape"])
    matrix[b["row"], b["column"]] = b["value"]
    return matrix


def arguments(run, state, start, law):
    stiffness, mu, velocity, width, radius = run._contact.parameters
    return (run.model, run._actors, physical(state), start, law), {
        "radius": radius,
        "stiffness": stiffness,
        "mu": mu,
        "width": width,
        "falloff": velocity * 0.001,
    }


def static_wrenches(run):
    result = np.zeros((run.model.shape_count, 6))
    bodies = run.model.shape_body.numpy()
    for pair in run._contact.pair_diagnostics():
        for shape, wrench in zip((pair["source"], pair["target"]), pair["world_wrench"], strict=True):
            if shape >= 0 and bodies[shape] < 0:
                result[shape] += wrench
    return result


def grid_scene(dimension, device="cpu"):
    """Construct an active connected-volume audit input, without solver stepping."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    body = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.001)))
    joint = builder.add_joint_prismatic(-1, body, axis=newton.Axis.Z)
    builder.add_articulation([joint])
    builder.add_shape_plane(body=body, cfg=builder.ShapeConfig(density=0, margin=0.0001))
    builder.add_soft_grid(
        pos=wp.vec3(0, 0, 0.0005),
        rot=wp.quat_identity(),
        vel=wp.vec3(0),
        dim_x=dimension,
        dim_y=dimension,
        dim_z=dimension,
        cell_x=0.01,
        cell_y=0.01,
        cell_z=0.01,
        density=1000,
        k_mu=3846.1538,
        k_lambda=5769.231,
        k_damp=0,
        add_surface_mesh_edges=False,
    )
    model = builder.finalize(device=device)
    actors = ActorTable.from_model(model)
    state = {
        "q": model.joint_q.numpy().astype(float),
        "x": model.particle_q.numpy().astype(float),
        "com": np.empty((0, 3)),
        "rotation": np.empty((0, 4)),
    }
    return model, actors, state


def test_streamed_pairs(test, device):
    """Match dense complete E/r/B and endpoint forces for all laws, pairs and layouts."""
    for kind, law, permuted in itertools.product(
        ("A-R", "R-R", "R-T", "R-static"), ("C-EE", "C-IE", "C-II"), (False, True)
    ):
        with test.subTest(kind=kind, law=law, permuted=permuted):
            run = contact_solver(contact_scene(device, kind, before=permuted), law=law, permuted=permuted)
            state = contact_state(run, kind)
            start = physical(state)
            perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
            args, options = arguments(run, state, start, law)
            full = oracle.evaluate_full(*args, **options)
            # The independent iterator does not consume production counts or samples.
            with patch.object(run._contact, "count", 0):
                stream = oracle.evaluate_stream(*args, **options)
            test.assertEqual(stream["full_count"], len(full["records"]))
            test.assertEqual(stream["candidate_count"], stream["full_count"])
            test.assertEqual(stream["admitted_count"], stream["full_count"])
            test.assertEqual(stream["min_gap"], full["min_gap"])
            for group in ("full", "candidate", "admitted"):
                for name in ("energy", "residual", "particle_force", "body_wrench", "static_wrench", "joint_force"):
                    np.testing.assert_array_equal(stream[group][name], full[name])
                np.testing.assert_array_equal(dense_tangent(stream[group]), full["tangent"])
            test.assertEqual(stream["spatial_omission"]["count"], 0)
            test.assertEqual(stream["capacity_omission"]["count"], 0)


def test_projected_sphere_source(test, device):
    """Audit all projected sphere samples against actual native pooled E/r/B and forces."""
    model = contact_scene(device, "R-static")
    shape = int(np.flatnonzero(model.shape_body.numpy() == model.body_label.index("rigid_0"))[0])
    kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
    kinds[shape], scales[shape] = 3, [0.01, 0, 0]
    model.shape_type.assign(kinds)
    model.shape_scale.assign(scales)
    for law in ("C-EE", "C-IE", "C-II"):
        run = pooled_solver(model, law=law, mode="production", graph=device.is_cuda)
        state = contact_state(run, "R-static")
        start = physical(state)
        perturb(run, state, start, np.linspace(-0.2, 0.3, run._layout.scalar_dof_count), 1e-6)
        args, options = arguments(run, state, start, law)
        result = oracle.evaluate_stream(*args, **options)
        test.assertEqual(result["full_count"], 5952)
        test.assertEqual(result["full_count"], run._contact.count)
        expected = {**result["full"], "tangent": dense_tangent(result["full"])}
        actual = _contact_at(run, physical(state), start, 0.001)
        _check_contact(actual, expected)
        np.testing.assert_allclose(static_wrenches(run), expected["static_wrench"], rtol=1e-8, atol=1e-10)
        test.assertGreater(np.linalg.norm(expected["body_wrench"]), 0)
        test.assertFalse(run._contact._tainted)


def test_mesh_stream(test, device):
    """Keep complete C2 mesh source/target auditing independent of production counts."""
    if not device.is_cuda:
        test.skipTest("Model texture SDF construction requires CUDA")
    from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415

    model = contact_scene(device, "A-R", mesh=True)
    sdf = build_smooth_sdf(model, representation="c2-bspline")
    fields = oracle.export_mesh_fields(model, sdf)
    for law in ("C-EE", "C-IE", "C-II"):
        run = pooled_solver(model, law=law, mode="production", graph=True, smooth_sdf=sdf)
        state = contact_state(run, "A-R")
        start = physical(state)
        args, options = arguments(run, state, start, law)
        result = oracle.evaluate_stream(*args, **options, mesh_fields=fields)
        expected = {**result["full"], "tangent": dense_tangent(result["full"])}
        _check_contact(_contact_at(run, physical(state), start, 0.001), expected)
        with test.assertRaises(ValueError):
            oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])


class TestContactStreamOracle(unittest.TestCase):
    def test_static_endpoint_wrench(self):
        """Compare static reactions about the world origin and balance both contact endpoints."""
        run = pooled_solver(contact_scene("cpu", "R-static"), mode="production")
        state = contact_state(run, "R-static")
        start = physical(state)
        args, options = arguments(run, state, start, "C-IE")
        full = oracle.evaluate_stream(*args, **options)["full"]
        expected = full["static_wrench"]
        _contact_at(run, physical(state), start, 0.001)
        np.testing.assert_allclose(static_wrenches(run), expected, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(full["body_wrench"][:, :3].sum(axis=0) + expected[:, :3].sum(axis=0), 0, atol=1e-12)
        centers = np.zeros((run.model.body_count, 3))
        centers[list(run._articulation_actor.body_indices)] = run._kinematics.com.numpy()
        for i, actor in enumerate(run._rigid.actors):
            centers[actor.body_indices[0]] = start["com"][i]
        torque = full["body_wrench"][:, 3:] + np.cross(centers, full["body_wrench"][:, :3])
        np.testing.assert_allclose(torque.sum(axis=0) + expected[:, 3:].sum(axis=0), 0, atol=1e-12)

    def test_spatial_and_capacity_omissions(self):
        """Separate excluded samples from discovered samples dropped by capacity."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        start = physical(state)
        args, options = arguments(run, state, start, "C-IE")
        full = oracle.evaluate_full(*args, **options)
        keys = list(full["records"])
        candidates, admitted = keys[::2], keys[::4]
        result = oracle.evaluate_stream(*args, **options, candidate_keys=candidates, admitted_keys=admitted)
        self.assertEqual(result["spatial_omission"]["count"], len(keys) - len(candidates))
        self.assertEqual(result["capacity_omission"]["count"], len(candidates) - len(admitted))
        for name, selected in (("candidate", candidates), ("admitted", admitted)):
            residual, tangent = np.zeros_like(full["residual"]), np.zeros_like(full["tangent"])
            energy = np.zeros(2)
            for key in selected:
                record = full["records"][key]
                energy += record["energy"]
                residual[record["columns"]] += record["residual"]
                tangent[np.ix_(record["columns"], record["columns"])] += record["tangent"]
            np.testing.assert_array_equal(result[name]["energy"], energy)
            np.testing.assert_array_equal(result[name]["residual"], residual)
            np.testing.assert_array_equal(dense_tangent(result[name]), tangent)
        for name, reference, selected in (
            ("spatial_omission", "full", "candidate"),
            ("capacity_omission", "candidate", "admitted"),
        ):
            expected = np.linalg.norm(dense_tangent(result[reference]) - dense_tangent(result[selected]))
            self.assertAlmostEqual(result[name]["errors"]["tangent"]["l2"], expected)
        expected = np.linalg.norm(full["tangent"] - dense_tangent(result["admitted"]))
        self.assertAlmostEqual(result["full_to_admitted_errors"]["tangent"]["l2"], expected)
        empty = oracle.evaluate_stream(*args, **options, admitted_keys=[])
        self.assertEqual(empty["spatial_omission"]["count"], 0)
        self.assertEqual(empty["capacity_omission"]["count"], len(keys))
        self.assertEqual(empty["admitted"]["tangent"]["value"].size, 0)
        np.testing.assert_array_equal(empty["admitted"]["body_wrench"], 0)

    def test_invalid_keys_and_nonfinite_input(self):
        """Reject invalid identities and native errors even when no contacts are admitted."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        start = physical(state)
        args, options = arguments(run, state, start, "C-IE")
        key = next(iter(oracle.evaluate_full(*args, **options)["records"]))
        for keys in (
            [key, key],
            [(-1, 0, False, 0)],
            [(-1, 0, 0.5, 0)],
            [(-1, 0, 0)],
            [(999, 999, 0, 0)],
            [None],
            [5],
            [np.array(0)],
            [np.zeros((4, 1), dtype=int)],
        ):
            for field in ("candidate_keys", "admitted_keys"):
                with self.subTest(field=field, keys=keys), self.assertRaises(ValueError):
                    oracle.evaluate_stream(*args, **options, **{field: keys})
        with self.assertRaises(ValueError):
            oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[key])
        args[2]["x"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite native states"):
            oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])

    def test_empty_and_repeated_audit(self):
        """Return zero totals for an empty complete set and never accumulate across audits."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        args, options = arguments(run, state, physical(state), "C-IE")
        first = oracle.evaluate_stream(*args, **options)
        repeated = oracle.evaluate_stream(*args, **options)
        for name in ("energy", "residual", "particle_force", "body_wrench", "joint_force"):
            np.testing.assert_array_equal(first["full"][name], repeated["full"][name])
        np.testing.assert_array_equal(first["full"]["tangent"]["value"], repeated["full"]["tangent"]["value"])
        self.assertEqual(first["full_key_sha256"], repeated["full_key_sha256"])
        with patch.object(run.model, "shape_flags", wp.zeros_like(run.model.shape_flags)):
            empty = oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])
        self.assertEqual(empty["full_count"], 0)
        self.assertEqual(empty["min_gap"], 0)
        self.assertEqual(empty["full"]["tangent"]["value"].size, 0)
        np.testing.assert_array_equal(empty["full"]["residual"], 0)

    def test_nonfinite_local_and_aggregate(self):
        """Fail malformed physics even when the spatial or capacity subset is empty."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        args, options = arguments(run, state, physical(state), "C-IE")
        with patch.object(oracle, "local", return_value=(np.full(2, np.nan), np.zeros(3), np.eye(3), np.eye(3))):
            with self.assertRaisesRegex(ValueError, "Nonfinite full-oracle local contribution"):
                oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])
        record = next(oracle.iter_full(*args, **options))
        record["energy"][:] = 1e308
        records = [record, {**record, "key": (-1, record["key"][1], 1, 1)}]
        with (
            patch.object(oracle, "iter_full", return_value=iter(records)),
            np.errstate(over="ignore", invalid="ignore"),
        ):
            with self.assertRaisesRegex(ValueError, "Nonfinite full-oracle aggregate"):
                oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])

    def test_dense_invalid_state_before_allocation(self):
        """Preserve early native validation before the compatible dense wrapper allocates B."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        args, options = arguments(run, state, physical(state), "C-IE")
        args[2]["x"][0, 0] = np.nan
        with patch.object(oracle, "_totals", side_effect=AssertionError("allocation before native validation")):
            with self.assertRaisesRegex(ValueError, "finite native states"):
                oracle.evaluate_full(*args, **options)

    def test_duplicate_model_pair(self):
        """Fail repeated effective Model pairs even when spatial selection is empty."""
        run = contact_solver(contact_scene("cpu", "A-R"))
        state = contact_state(run, "A-R")
        args, options = arguments(run, state, physical(state), "C-IE")
        pairs = run.model.shape_contact_pairs.numpy()
        repeated = wp.array(np.vstack((pairs, pairs)), dtype=wp.vec2i, device="cpu")
        with patch.object(run.model, "shape_contact_pairs", repeated):
            with self.assertRaisesRegex(ValueError, "Duplicate effective Model shape pair"):
                oracle.evaluate_stream(*args, **options, candidate_keys=[], admitted_keys=[])

    def test_tangent_only_omission(self):
        """Count a tangent-only record as contributing in each omission category."""
        run = contact_solver(contact_scene("cpu", "R-T"))
        state = contact_state(run, "R-T")
        start = physical(state)
        args, options = arguments(run, state, start, "C-IE")
        record = next(oracle.iter_full(*args, **options))
        record["energy"][:] = 0
        record["residual"][:] = 0
        record["world_residual"][:] = 0
        record["particle_force"][:] = 0
        record["body_wrench"][:] = 0
        record["tangent"][:] = np.eye(len(record["columns"]))
        for selection, category in (
            ({"candidate_keys": []}, "spatial_omission"),
            ({"admitted_keys": []}, "capacity_omission"),
        ):
            with patch.object(oracle, "iter_full", return_value=iter([record])):
                result = oracle.evaluate_stream(*args, **options, **selection)
            self.assertEqual(result[category]["count"], 1)
            self.assertEqual(result[category]["contributing_count"], 1)
            self.assertGreater(result[category]["errors"]["tangent"]["l2"], 0)

    def test_frozen_load_after_separation(self):
        """Retain separated C-EE start load and C-IE gap minus g dot displacement."""
        run = contact_solver(contact_scene("cpu", "R-static"))
        state = contact_state(run, "R-static")
        start = physical(state)
        com = start["com"].copy()
        com[0] += [0.03, 0.001, 0]
        state._monolithic_coupled.rigid_com.assign(com)
        for law in ("C-EE", "C-IE", "C-II"):
            args, options = arguments(run, state, start, law)
            result = oracle.evaluate_stream(*args, **options, candidate_keys=[])
            self.assertGreater(result["min_gap"], options["width"])
            if law == "C-II":
                np.testing.assert_array_equal(result["full"]["energy"], 0)
                self.assertEqual(result["spatial_omission"]["contributing_count"], 0)
            else:
                self.assertGreater(result["full"]["energy"][1], 0)
                self.assertGreater(result["spatial_omission"]["contributing_count"], 0)

    def test_large_native_model_avoids_dense_storage(self):
        """Audit a real connected tet grid using local support and sparse global B."""
        model, actors, state = grid_scene(12)
        n = actors.scalar_dof_count
        self.assertGreater(n, 6000)
        original = np.zeros

        def bounded_zeros(shape, *args, **kwargs):
            if isinstance(shape, tuple) and shape in ((n, n), (3, n), (n, 3)):
                raise AssertionError("Global dense B/J allocation")
            return original(shape, *args, **kwargs)

        with patch.object(oracle.np, "zeros", bounded_zeros):
            result = oracle.evaluate_stream(model, actors, state, state, "C-IE", radius=0.001)
        self.assertEqual(result["full_count"], 3 * model.tri_count)
        self.assertGreater(result["full"]["energy"][0], 0)
        self.assertGreater(result["full"]["tangent"]["value"].size, 0)
        self.assertLess(result["full"]["tangent"]["value"].size, n * 20)
        self.assertNotIn("records", result)


add_function_test(TestContactStreamOracle, "test_streamed_pairs", test_streamed_pairs, devices=get_test_devices())
add_function_test(
    TestContactStreamOracle, "test_projected_sphere_source", test_projected_sphere_source, devices=get_test_devices()
)
add_function_test(TestContactStreamOracle, "test_mesh_stream", test_mesh_stream, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()
