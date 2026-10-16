# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check strict old/new contact comparisons without relaxing checkpoint identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.contact import ContactObjective
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.record_workload import record
from scripts.monolithic_reference.workload_io import digest, write_json
from scripts.monolithic_reference.workload_scene import build_scene


def test_dataflow_comparison(test, device):
    """Advance independent native trajectories and audit both paths at identical contact states."""
    from scripts.monolithic_reference.compare_contact_dataflows import compare  # noqa: PLC0415

    for scene in ("actor-coupling-before", "actor-coupling-permuted"):
        for graph in (False, True) if device.is_cuda else (False,):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "bundle"
                record(root, scene=scene, device=device, graph=graph, steps=2, windows=[(1, 2)], boundaries=())
                result = compare(root, device=device)
                test.assertEqual(result["status"], "PASS_LOCAL_COMPLETE_CONTACT")
                test.assertEqual(result["steps"], 2)
                test.assertEqual(result["same_state_checks"], 8)
                test.assertEqual(result["algorithm_differences"].keys(), {"contact_dataflow", "contact_resources"})
                test.assertNotEqual(*result["config_digests"])
                test.assertTrue(all(s["converged"] for row in result["work"] for s in row["algorithms"]))
                test.assertTrue(all(s["step_seconds"] > 0 for row in result["work"] for s in row["algorithms"]))
                test.assertTrue(all(v["within_tolerance"] for v in result["trajectory_fields"].values()))
                manifest = json.loads((root / "manifest.json").read_text())
                for key, value in (
                    ("source", {}),
                    ("physical_digest", "00" * 32),
                    ("dt", 0.002),
                    ("comparison", {"rtol": 1, "atol": 1}),
                    ("windows", [[2, 2]]),
                    ("contact_dataflow", "pooled"),
                ):
                    damaged = copy.deepcopy(manifest)
                    damaged[key] = value
                    write_json(root / "manifest.json", damaged)
                    with test.assertRaises(ValueError):
                        compare(root, device=device)
                write_json(root / "manifest.json", manifest)
                # An edited payload with a recomputed file hash still must pass physics/trace checks.
                path = root / "state-1.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = dict(archive)
                arrays["history.config_digest"] = np.zeros(32, dtype=np.uint8)
                np.savez_compressed(path, **arrays)
                manifest["files"][path.name] = digest(path)
                write_json(root / "manifest.json", manifest)
                with test.assertRaises(AssertionError):
                    compare(root, device=device)


def test_contract_rejection(test, device):
    """Reject changed law, Q1, layout, sampling or Model input before permitting algorithm differences."""
    from scripts.monolithic_reference.compare_contact_dataflows import verify_contract  # noqa: PLC0415

    _, left, _, _ = build_scene("actor-coupling", device=device, graph=False)
    _, right, _, _ = build_scene("actor-coupling", device=device, graph=False, contact_dataflow="pooled")
    verify_contract(left, right)
    with patch.object(right, "linear_tolerance", left.linear_tolerance * 2):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    with patch.object(right._contact, "contact_model", "C-EE"):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    for key in ("contact_model", "direction_policy", "iteration_policy", "guard_capacity"):
        original = right._coupled_config[key]
        right._coupled_config[key] = original.copy()
        right._coupled_config[key].flat[0] += 1
        with test.assertRaises(ValueError):
            verify_contract(left, right)
        right._coupled_config[key] = original
    key = "actor_contact.weights"
    original = right._contact.actor_samples.fields[key]
    right._contact.actor_samples.fields[key] = original * 2
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    right._contact.actor_samples.fields[key] = original
    original = right.model.body_mass.numpy().copy()
    right.model.body_mass.assign(original * 2)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    right.model.body_mass.assign(original)
    with patch.object(right._contact, "dataflow_identity", np.frombuffer(b"unknown", dtype=np.uint8)):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    # Original checkpoint guards remain strict across the algorithms.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        left.save_checkpoint(left.create_state(dt=0.001), path)
        with test.assertRaises(ValueError):
            right.load_checkpoint(right.create_state(dt=0.001), path)


def test_hotspot_and_failure(test, device):
    """Audit authentic Newton/trial inputs and reject overflow or a corrupted complete oracle."""
    from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_state  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_contact_pool import pooled_solver  # noqa: PLC0415
    from scripts.monolithic_reference.compare_contact_dataflows import _contact_at, _physical, compare  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "bundle"
        record(
            root,
            scene="actor-coupling",
            device=device,
            graph=device.is_cuda,
            steps=1,
            windows=[(1, 1)],
            boundaries=("objective",),
            call_ordinals={"objective": list(range(10))},
        )
        result = compare(root, device=device)
        test.assertGreater(result["same_state_checks"], 4)
        test.assertTrue(any(row.get("state_owner") == "_trial" for row in result["same_state"]))
        from scripts.monolithic_reference.contact_full_oracle import evaluate_stream  # noqa: PLC0415

        def wrong_oracle(*args, **kwargs):
            full = evaluate_stream(*args, **kwargs)
            full["full"]["energy"] += 1
            return full

        with patch("scripts.monolithic_reference.compare_contact_dataflows.evaluate_stream", wrong_oracle):
            with test.assertRaises(AssertionError):
                compare(root, device=device)
    run = pooled_solver(contact_scene(device, "chain"), contact_factor_capacity=0)
    state = _physical(contact_state(run, "chain"))
    with test.assertWarns(RuntimeWarning), test.assertRaisesRegex(ValueError, "overflow"):
        _contact_at(run, state, state, 0.001)


def test_canonical_contract(test, device):
    """Authenticate legacy tet sampling without permitting metadata or checkpoint drift."""
    from scripts.monolithic_reference.compare_contact_dataflows import verify_contract  # noqa: PLC0415

    _, left, _, _ = build_scene("tiny-contact", device=device, graph=False)
    _, right, _, _ = build_scene("tiny-contact", device=device, graph=False, contact_dataflow="pooled")
    result = verify_contract(left, right)
    test.assertEqual(result["enumeration"], {"legacy": 12, "actor": 0, "complete": 12})
    for name in ("legacy_count", "count"):
        with patch.object(right._contact, name, getattr(right._contact, name) + 1):
            with test.assertRaises(ValueError):
                verify_contract(left, right)
    for name in ("area", "mapping", "shapes", "target_bodies"):
        array = getattr(right._contact, name)
        original = array.numpy().copy()
        damaged = original.copy()
        damaged.flat[0] += 1
        array.assign(damaged)
        with test.assertRaises(ValueError):
            verify_contract(left, right)
        array.assign(original)
    with patch.object(right._contact, "_legacy_ancestors", {}):
        with test.assertRaises(ValueError):
            verify_contract(left, right)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        left.save_checkpoint(left.create_state(dt=0.001), path)
        with test.assertRaises(ValueError):
            right.load_checkpoint(right.create_state(dt=0.001), path)


def test_canonical_comparison(test, device):
    """Compare authentic canonical steps and recorded trial operands under original gates."""
    from scripts.monolithic_reference.compare_contact_dataflows import compare  # noqa: PLC0415

    for graph in (False, True) if device.is_cuda else (False,):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            record(
                root,
                scene="tiny-contact",
                device=device,
                graph=graph,
                steps=2,
                windows=[(1, 2)],
                boundaries=("objective",),
                call_ordinals={"objective": list(range(10))},
            )
            result = compare(root, device=device)
            test.assertEqual(result["status"], "PASS_LOCAL_COMPLETE_CONTACT")
            test.assertEqual(result["enumeration"], {"legacy": 12, "actor": 0, "complete": 12})
            test.assertEqual(result["trajectory_limits"], {"rtol": 2e-5, "atol": 2e-6})
            test.assertGreater(result["same_state_checks"], 8)
            test.assertTrue(any(row.get("state_owner") == "_trial" for row in result["same_state"]))
            test.assertTrue(all(row["full_count"] == 12 for row in result["same_state"]))
            test.assertTrue(all(s["converged"] for row in result["work"] for s in row["algorithms"]))
            manifest = json.loads((root / "manifest.json").read_text())
            for key, value in (("dt", 0.002), ("source", {}), ("comparison", {"rtol": 1, "atol": 1})):
                damaged = copy.deepcopy(manifest)
                damaged[key] = value
                write_json(root / "manifest.json", damaged)
                with test.assertRaises(ValueError):
                    compare(root, device=device)
            write_json(root / "manifest.json", manifest)
            # Rehashed inputs still have to match the allowlisted scene controls.
            path = root / "inputs-1.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = dict(archive)
            arrays["control.joint_f"] += 1
            np.savez_compressed(path, **arrays)
            manifest["files"][path.name] = digest(path)
            write_json(root / "manifest.json", manifest)
            with test.assertRaises(ValueError):
                compare(root, device=device)


def test_owned_geometry_contract(test, device):
    """Compare owned mesh contents while rejecting unknown pointers and changed geometry."""
    from scripts.monolithic_reference.compare_contact_dataflows import verify_contract  # noqa: PLC0415

    finalize = newton.ModelBuilder.finalize

    def with_mesh(builder, *args, **kwargs):
        cfg = builder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.01, 0.02, 0.03), cfg=cfg)
        return finalize(builder, *args, **kwargs)

    with patch.object(newton.ModelBuilder, "finalize", with_mesh):
        _, left, _, _ = build_scene("tiny-contact", device=device, graph=False)
        _, right, _, _ = build_scene("tiny-contact", device=device, graph=False, contact_dataflow="pooled")
    test.assertNotEqual(left.model.shape_source_ptr.numpy()[-1], right.model.shape_source_ptr.numpy()[-1])
    right.model.particle_colors.fill_(987)
    right.model.bvh_shape_enabled.fill_(987)
    result = verify_contract(left, right)
    test.assertEqual(result["model_identity_adapter"], "monolithic-owned-geometry/v1")
    for name in ("particle_colors", "bvh_shape_enabled"):
        array = getattr(right.model, name)
        for size, dtype in ((array.size + 1, array.dtype), (array.size, wp.float32)):
            with patch.object(right.model, name, wp.empty(size, dtype=dtype, device=device)):
                with test.assertRaises(ValueError):
                    verify_contract(left, right)
    mesh = right.model._mesh_keep_alive[-1]
    for array in (mesh.points, mesh.indices):
        original = array.numpy().copy()
        damaged = original.copy()
        damaged.flat[0] += 1
        array.assign(damaged)
        with test.assertRaises(ValueError):
            verify_contract(left, right)
        array.assign(original)
    original = right.model.shape_source_ptr.numpy().copy()
    damaged = original.copy()
    damaged[-1] = 1
    right.model.shape_source_ptr.assign(damaged)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    right.model.shape_source_ptr.assign(original)
    verify_contract(left, right)


def test_owned_sdf_contract(test, device):
    """Reject changed live descriptors, texels and C2 coefficients while allowing owned resource identities."""
    if device.is_cpu:
        test.skipTest("Mesh texture ownership requires CUDA")
    from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_contact_pool import pooled_solver  # noqa: PLC0415
    from scripts.monolithic_reference.compare_contact_dataflows import verify_contract  # noqa: PLC0415

    runs = []
    for factory in (contact_solver, pooled_solver):
        model = contact_scene(device, "A-R", mesh=True)
        sdf = build_smooth_sdf(model, representation="c2-bspline")
        with patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", ContactObjective):
            runs.append(factory(model, mode="production", graph=False, smooth_sdf=sdf))
    left, right = runs
    verify_contract(left, right)
    for name in ("_texture_sdf_coarse_textures", "_texture_sdf_subgrid_textures", "_texture_sdf_subgrid_start_slots"):
        with patch.object(right.model, name, getattr(left.model, name)):
            with test.assertRaises(ValueError):
                verify_contract(left, right)
    table = right.model._texture_sdf_data
    original = table.numpy().copy()
    damaged = original.copy()
    damaged["voxel_radius"][0] += 0.01
    table.assign(damaged)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    table.assign(original)
    texture = right.model._texture_sdf_coarse_textures[0]
    shape = (texture.depth, texture.height, texture.width)
    if texture.num_channels != 1:
        shape += (texture.num_channels,)
    pixels = np.empty(shape, dtype=np.float32)
    texture.copy_to(pixels)
    texture.copy_from(pixels + 0.01)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    texture.copy_from(pixels)
    coefficients = right._contact.smooth_sdf.fields[0].data.coefficients
    original = coefficients.numpy().copy()
    coefficients.assign(original + 0.01)
    with test.assertRaises(ValueError):
        verify_contract(left, right)
    coefficients.assign(original)
    verify_contract(left, right)


def test_prepared_mesh_audit_pair(test, device):
    """Audit both native paths using one independently prepared immutable mesh field."""
    if device.is_cpu:
        test.skipTest("Mesh SDF requires CUDA")
    from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415
    from newton.tests.test_monolithic_solver_actor_contacts import (  # noqa: PLC0415
        contact_scene,
        contact_solver,
        contact_state,
    )
    from newton.tests.test_monolithic_solver_contact_pool import pooled_solver  # noqa: PLC0415
    from scripts.monolithic_reference.compare_contact_dataflows import _audit_pair, _physical  # noqa: PLC0415
    from scripts.monolithic_reference.contact_full_oracle import export_mesh_fields  # noqa: PLC0415

    for law in ("C-IE", "C-EE", "C-II"):
        runs = []
        for factory in (contact_solver, pooled_solver):
            model = contact_scene(device, "A-R", mesh=True)
            sdf = build_smooth_sdf(model, representation="c2-bspline")
            with patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", ContactObjective):
                runs.append(factory(model, law=law, mode="production", graph=device.is_cuda, smooth_sdf=sdf))
        frame = _physical(contact_state(runs[0], "A-R"))
        fields = export_mesh_fields(runs[0].model, runs[0]._contact.smooth_sdf)
        result = _audit_pair(runs, frame, frame, 0.001, {"state_origin": "legacy"}, mesh_fields=fields)
        test.assertEqual([row["algorithm"] for row in result], ["legacy", "pooled"])
        test.assertTrue(all(row["full_count"] == runs[0]._contact.count for row in result))
        for row in result:
            test.assertLess(row["errors"]["tangent_relative"], 1e-5)


class TestContactDataflowComparison(unittest.TestCase):
    def test_absolute_state_envelope(self):
        """Keep frozen absolute, zero and discrete field limits without falling back to a relative tolerance."""
        from scripts.monolithic_reference.compare_contact_dataflows import _compare_absolute  # noqa: PLC0415

        expected = {"bounded": np.array([1.0]), "zero": np.zeros(1), "discrete": np.array([7], dtype=np.int32)}
        limit = 2.0**-30
        limits = {"bounded": limit, "zero": 0.0, "discrete": 1.0}
        actual = {name: value.copy() for name, value in expected.items()}
        actual["bounded"] += limit / 2
        errors = {}
        _compare_absolute(actual, expected, limits, errors)
        self.assertEqual(errors["bounded"], limit / 2)
        for name, delta in (("bounded", 2 * limit), ("zero", 1e-12), ("discrete", 1)):
            damaged = {key: value.copy() for key, value in expected.items()}
            damaged[name] += delta
            with self.assertRaises(AssertionError):
                _compare_absolute(damaged, expected, limits, {})
        with self.assertRaises(AssertionError):
            _compare_absolute(actual, expected, {}, {})
        for value in (np.zeros(2), np.zeros(1, dtype=np.float32), np.array([np.nan])):
            with self.assertRaises(ValueError):
                _compare_absolute({**expected, "zero": value}, expected, limits, {})


for _test in (
    test_dataflow_comparison,
    test_contract_rejection,
    test_hotspot_and_failure,
    test_canonical_contract,
    test_canonical_comparison,
    test_owned_geometry_contract,
    test_owned_sdf_contract,
    test_prepared_mesh_audit_pair,
):
    add_function_test(TestContactDataflowComparison, _test.__name__, _test, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
