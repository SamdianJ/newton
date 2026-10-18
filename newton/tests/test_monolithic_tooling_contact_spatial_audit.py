# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit selected spatial states independently of query and capacity decisions."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.test_monolithic_contact_actor_pairs import physical
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_contact_pool import pooled_solver
from newton.tests.test_monolithic_solver_contact_spatial import spatial_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.contact_audit import audit_spatial_contact_state, verify_contract
from scripts.monolithic_reference.workload_scene import build_scene


def test_spatial_state_audit(test, device):
    """Check selected native E/r/raw B and both forces against independent full enumeration."""
    options = {"mode": "production", "graph": device.is_cuda}
    full = pooled_solver(contact_scene(device, "chain"), **options)
    run = spatial_solver(contact_scene(device, "chain"), **options)
    contract = verify_contract(full, run)
    test.assertEqual(contract["comparison"], "pooled/spatial")
    state = contact_state(run, "chain")
    start = physical(state)
    frame = {k: v.copy() for k, v in start.items()}
    frame["com"][:, 1] += 0.0002
    result = audit_spatial_contact_state(run, frame, start, 0.001, frozen=start)
    test.assertEqual(result["status"], "MEASURED_NOT_QUALIFIED")
    test.assertFalse(result["tainted"])
    test.assertEqual(result["full_count"], run._contact.count)
    test.assertLess(result["native_selected_errors"]["tangent_relative"], 1e-5)
    test.assertEqual(result["capacity_omission"]["contributing_count"], 0)
    run.linear_tolerance *= 2
    with test.assertRaises(ValueError):
        verify_contract(full, run)
    full = pooled_solver(contact_scene(device, "chain"))
    run = spatial_solver(contact_scene(device, "chain"), contact_factor_capacity=3072)
    test.assertEqual(verify_contract(full, run)["comparison"], "pooled/spatial")
    run._coupled_config["guard_capacity"][0] *= 2
    with test.assertRaises(ValueError):
        verify_contract(full, run)


def test_spatial_known_miss_audit(test, device):
    """Detect nonzero contacts outside a frozen query without refreshing at the trial state."""
    run = spatial_solver(contact_scene(device, "A-R"))
    touching = physical(contact_state(run, "A-R"))
    far = {k: v.copy() for k, v in touching.items()}
    far["com"][:] = 10
    result = audit_spatial_contact_state(run, touching, far, 0.001, frozen=far)
    test.assertFalse(result["tainted"])
    test.assertEqual(result["candidate_count"], 0)
    test.assertGreater(result["spatial_omission"]["contributing_count"], 0)
    test.assertGreater(result["spatial_omission"]["errors"]["residual"]["l2"], 0)
    test.assertEqual(run._contact._refreshes, 1)


def test_spatial_capacity_audit(test, device):
    """Keep query overflow, pool overflow and saved zero work distinct from spatial qualification."""
    for options, query_drop in (({"capacity": 0}, True), ({"contact_factor_capacity": 0}, False)):
        run = spatial_solver(contact_scene(device, "chain"), **options)
        start = physical(contact_state(run, "chain"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = audit_spatial_contact_state(run, start, start, 0.001)
        test.assertTrue(result["tainted"])
        test.assertEqual(result["spatial_omission"] is None, query_drop)
        test.assertGreater(result["full_contributing_count"], 0)
        if not query_drop:
            test.assertGreater(result["capacity_omission"]["contributing_count"], 0)
            test.assertEqual(result["pool"]["admitted"], 0)
        with test.assertRaises(ValueError):
            verify_contract(run, run)


def test_owned_geometry_contract(test, device):
    """Compare owned mesh contents while rejecting unknown pointers and changed geometry."""
    finalize = newton.ModelBuilder.finalize

    def with_mesh(builder, *args, **kwargs):
        cfg = builder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.01, 0.02, 0.03), cfg=cfg)
        return finalize(builder, *args, **kwargs)

    with patch.object(newton.ModelBuilder, "finalize", with_mesh):
        _, pooled, _, _ = build_scene("tiny-contact", device=device, graph=False, contact_dataflow="pooled")
        _, spatial, _, _ = build_scene("tiny-contact", device=device, graph=False)
    test.assertNotEqual(pooled.model.shape_source_ptr.numpy()[-1], spatial.model.shape_source_ptr.numpy()[-1])
    spatial.model.particle_colors.fill_(987)
    spatial.model.bvh_shape_enabled.fill_(987)
    result = verify_contract(pooled, spatial)
    test.assertEqual(result["model_identity_adapter"], "monolithic-owned-geometry/v1")
    for name in ("particle_colors", "bvh_shape_enabled"):
        array = getattr(spatial.model, name)
        for size, dtype in ((array.size + 1, array.dtype), (array.size, wp.float32)):
            with patch.object(spatial.model, name, wp.empty(size, dtype=dtype, device=device)):
                with test.assertRaises(ValueError):
                    verify_contract(pooled, spatial)
    mesh = spatial.model._mesh_keep_alive[-1]
    for array in (mesh.points, mesh.indices):
        original = array.numpy().copy()
        damaged = original.copy()
        damaged.flat[0] += 1
        array.assign(damaged)
        with test.assertRaises(ValueError):
            verify_contract(pooled, spatial)
        array.assign(original)
    original = spatial.model.shape_source_ptr.numpy().copy()
    damaged = original.copy()
    damaged[-1] = 1
    spatial.model.shape_source_ptr.assign(damaged)
    with test.assertRaises(ValueError):
        verify_contract(pooled, spatial)
    spatial.model.shape_source_ptr.assign(original)
    verify_contract(pooled, spatial)


def test_owned_sdf_contract(test, device):
    """Reject changed live descriptors, texels and C2 coefficients while allowing owned resource identities."""
    if device.is_cpu:
        test.skipTest("Mesh texture ownership requires CUDA")
    from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415

    runs = []
    for factory in (pooled_solver, contact_solver):
        model = contact_scene(device, "A-R", mesh=True)
        sdf = build_smooth_sdf(model, representation="c2-bspline")
        runs.append(factory(model, mode="production", graph=False, smooth_sdf=sdf))
    pooled, spatial = runs
    verify_contract(pooled, spatial)
    for name in ("_texture_sdf_coarse_textures", "_texture_sdf_subgrid_textures", "_texture_sdf_subgrid_start_slots"):
        with patch.object(spatial.model, name, getattr(pooled.model, name)):
            with test.assertRaises(ValueError):
                verify_contract(pooled, spatial)
    table = spatial.model._texture_sdf_data
    original = table.numpy().copy()
    damaged = original.copy()
    damaged["voxel_radius"][0] += 0.01
    table.assign(damaged)
    with test.assertRaises(ValueError):
        verify_contract(pooled, spatial)
    table.assign(original)
    texture = spatial.model._texture_sdf_coarse_textures[0]
    shape = (texture.depth, texture.height, texture.width)
    if texture.num_channels != 1:
        shape += (texture.num_channels,)
    pixels = np.empty(shape, dtype=np.float32)
    texture.copy_to(pixels)
    texture.copy_from(pixels + 0.01)
    with test.assertRaises(ValueError):
        verify_contract(pooled, spatial)
    texture.copy_from(pixels)
    coefficients = spatial._contact.smooth_sdf.fields[0].data.coefficients
    original = coefficients.numpy().copy()
    coefficients.assign(original + 0.01)
    with test.assertRaises(ValueError):
        verify_contract(pooled, spatial)
    coefficients.assign(original)
    verify_contract(pooled, spatial)


class TestContactSpatialAudit(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _name in (
        "test_spatial_state_audit",
        "test_spatial_known_miss_audit",
        "test_spatial_capacity_audit",
        "test_owned_geometry_contract",
        "test_owned_sdf_contract",
    ):
        add_function_test(TestContactSpatialAudit, _name, globals()[_name], devices=[_device])


if __name__ == "__main__":
    unittest.main()
