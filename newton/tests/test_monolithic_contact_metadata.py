# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact metadata rebuilds deterministically from validated physical inputs."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.actors import ActorTable
from newton._src.solvers.monolithic.contact import ContactObjective
from newton._src.solvers.monolithic.contact_batch import _ContactBatch
from newton.solvers.experimental.monolithic import build_smooth_sdf
from newton.tests.monolithic_test_utils import add_contact_free_tet
from newton.tests.test_monolithic_contact_actor_pairs import native_inputs
from newton.tests.test_monolithic_contact_components import contact, fixture
from newton.tests.test_monolithic_contact_precision import native_inputs as precision_inputs
from newton.tests.test_monolithic_solver_actor_contacts import contact_scene, contact_solver, contact_state
from newton.tests.unittest_utils import add_function_test, get_test_devices


def metadata(owner, **overrides):
    """Rebuild contact metadata from the exact supported input contract of ``owner``."""
    p = owner.parameters
    actors = None if owner.actor_samples is None else owner.actor_samples.actors
    arm = None if actors is None else next(a for a in actors.actors if a.kind == "articulation")
    options = {
        "q_dof_count": owner.nq,
        "normal_stiffness": p[0],
        "friction_coefficient": p[1],
        "friction_velocity_scale": p[2],
        "smoothing_width": p[3],
        "particle_radius": p[4],
        "smooth_sdf": owner.smooth_sdf,
        "alignment_policy": owner.alignment_policy,
        "dtype": owner.dtype,
        "layout": owner.layout,
        "articulation_actor": arm,
        "actors": actors,
    }
    options.update(overrides)
    return ContactObjective(owner.model, owner.mapping.numpy(), **options)


def compact_contract(test, owner):
    """Rebuild without per-key support expansion and verify all source identities and capacity arithmetic."""
    compact = metadata(owner)
    test.assertEqual((compact.count, compact.legacy_count), (owner.count, owner.legacy_count))
    if not hasattr(owner, "pool"):
        # An initialized solver objective replaces this sampling bound with its pool bound.
        test.assertEqual(compact.triplet_capacity, owner.triplet_capacity)
    test.assertEqual(compact.values.keys(), owner.values.keys())
    for name in compact.values:
        np.testing.assert_array_equal(compact.values[name], owner.values[name])
    if owner.actor_samples is not None:
        old, new = owner.actor_samples, compact.actor_samples
        test.assertEqual(
            (new.keys, new.support_size, new.triplet_capacity), (old.keys, old.support_size, old.triplet_capacity)
        )
        test.assertEqual(new.supports, [])
        test.assertEqual(new.selectors, [])
        test.assertEqual(new.fields.keys(), old.fields.keys())
        for name in old.fields:
            np.testing.assert_array_equal(new.fields[name], old.fields[name])
        new.validate_identity()
    return compact


def compare_batches(test, owner, compact, inputs, options, trial_inputs):
    """Rebuilt metadata reproduces generated geometry, references and admitted physics bitwise."""
    b = min(owner.count, 7)
    original = _ContactBatch(
        owner, batch_capacity=b, capacity=owner.count * 18, use_cuda_graph=owner.model.device.is_cuda
    )
    bounded = _ContactBatch(
        compact, batch_capacity=b, capacity=owner.count * 18, use_cuda_graph=owner.model.device.is_cuda
    )
    for batch in (original, bounded):
        batch.begin_step(*inputs, **options)
        batch.evaluate(*trial_inputs, dt=0.001, **options)
        test.assertEqual(int(batch.flags.numpy()[0]), 0)
    for name in ("y", "gradient", "soft_normal", "distance", "uncertified"):
        np.testing.assert_array_equal(getattr(original.start, name).numpy(), getattr(bounded.start, name).numpy())
    for name in ("energy", "residual", "force"):
        np.testing.assert_array_equal(
            getattr(original.pool, name).result.numpy(), getattr(bounded.pool, name).result.numpy()
        )
    for name in ("triplet_rows", "triplet_columns", "triplet_values", "triplet_count", "pairs"):
        np.testing.assert_array_equal(
            getattr(original.pool.records, name).numpy(), getattr(bounded.pool.records, name).numpy()
        )
    return bounded


def test_compact_legacy(test, device):
    """Retain legacy fixed nodes and native FP32/64 law inputs without candidate x DoF tables."""
    for dtype in (wp.float64, wp.float32):
        for kind in ("plane", "sphere", "box"):
            model, xn = fixture(device, kind)
            owner = contact(model, fixed=True, dtype=dtype)
            compact = compact_contract(test, owner)
            trial = xn + np.array([1e-6, 0, 1e-6])
            trial[0] = xn[0]
            compare_batches(
                test,
                owner,
                compact,
                precision_inputs(model, xn, np.zeros(2), dtype),
                {},
                precision_inputs(model, trial, np.array([1e-3, 1e-6]), dtype),
            )


def test_compact_actor(test, device):
    """Preserve Model pair filtering, source sampling, half weights and mixed layout physical identity."""
    for kind in ("A-R", "R-R", "R-T", "R-static", "chain"):
        run = contact_solver(contact_scene(device, kind, before=True), permuted=True)
        s = contact_state(run, kind)
        compact = compact_contract(test, run._contact)
        inputs, options = native_inputs(run, s)
        compare_batches(test, run._contact, compact, inputs, options, inputs)
        if compact.actor_samples.sources:
            original = run.model.shape_contact_pairs
            run.model.shape_contact_pairs = wp.clone(original)
            with test.assertRaisesRegex(ValueError, "Actor contact model changed"):
                compact.actor_samples.validate_identity()
            run.model.shape_contact_pairs = original


def test_compact_guards(test, device):
    """Apply the same physical parameter, map, topology and capacity guards before storage choice."""
    model, _ = fixture(device, "plane")
    owner = contact(model, fixed=True)
    for options, message in (
        ({"normal_stiffness": np.nan}, "Unrepresentable"),
        ({"normal_stiffness": -1}, "Invalid variational"),
        ({"friction_coefficient": 1, "friction_velocity_scale": 0}, "Invalid variational"),
        ({"base_triplet_count": 2**31 - 1}, "capacity exceeds"),
    ):
        with test.assertRaisesRegex(ValueError, message):
            metadata(owner, **options)
    mass = model.particle_inv_mass.numpy().copy()
    invalid = mass.copy()
    invalid[0] = 1
    model.particle_inv_mass.assign(invalid)
    with test.assertRaisesRegex(ValueError, "preserve fixed nodes"):
        metadata(owner)
    model.particle_inv_mass.assign(mass)
    original = model.tri_indices
    indices = original.numpy().copy()
    indices[0, 0] = model.particle_count
    model.tri_indices = wp.array(indices, dtype=int, device=device)
    with test.assertRaisesRegex(ValueError, "Invalid surface topology"):
        metadata(owner)
    model.tri_indices = original


def test_compact_mesh(test, device):
    """Preserve CUDA/C2 target requirements while discarding full actor projection allocations."""
    if device.is_cpu:
        test.skipTest("Actor mesh target requires CUDA")
    model = contact_scene(device, "A-R", mesh=True)
    smooth = build_smooth_sdf(model)
    run = contact_solver(model, smooth_sdf=smooth)
    s = contact_state(run, "A-R")
    compact = compact_contract(test, run._contact)
    inputs, options = native_inputs(run, s)
    compare_batches(test, run._contact, compact, inputs, options, inputs)
    with test.assertRaisesRegex(ValueError, "bound C2 SDF"):
        metadata(run._contact, smooth_sdf=None)
    original = model._shape_sdf_index
    invalid = original.numpy().copy()
    invalid[invalid >= 0] = -1
    model._shape_sdf_index = wp.array(invalid, dtype=int, device=device)
    with test.assertRaisesRegex(ValueError, "CUDA texture SDF"):
        metadata(run._contact)
    model._shape_sdf_index = original


def test_compact_branch_support(test, device):
    """Use actual articulation ancestors even for a standalone legacy metadata owner."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    bodies = [builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01))) for _ in range(3)]
    joints = [
        builder.add_joint_revolute(-1, bodies[0]),
        builder.add_joint_prismatic(bodies[0], bodies[1]),
        builder.add_joint_revolute(bodies[0], bodies[2]),
    ]
    builder.add_articulation(joints)
    builder.add_shape_plane(body=bodies[1], width=0, length=0, cfg=builder.ShapeConfig(density=0))
    add_contact_free_tet(builder)
    builder.particle_mass[0] = 0
    model = builder.finalize(device=device)
    arm = next(a for a in ActorTable.from_model(model).actors if a.kind == "articulation")
    mapping = np.full(model.particle_count, -1, np.int32)
    dynamic = model.particle_inv_mass.numpy() > 0
    mapping[dynamic] = np.arange(np.count_nonzero(dynamic))
    owner = ContactObjective(
        model,
        mapping,
        q_dof_count=arm.dof_count,
        articulation_actor=arm,
        normal_stiffness=1e7,
        friction_coefficient=0.5,
        friction_velocity_scale=0.01,
        smoothing_width=1e-4,
        particle_radius=0.001,
    )
    batch = _ContactBatch(owner, batch_capacity=3, capacity=owner.count * 12)
    offsets, columns = batch.support.offsets.numpy(), batch.support.columns.numpy()
    for body, expected in zip(bodies, ((0,), (0, 1), (0, 2)), strict=True):
        test.assertEqual(tuple(columns[offsets[body] : offsets[body + 1]]), expected)
    test.assertEqual(batch.pool.max_support, 11)


class TestContactMetadata(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_compact_legacy,
        test_compact_actor,
        test_compact_guards,
        test_compact_mesh,
        test_compact_branch_support,
    ):
        add_function_test(TestContactMetadata, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
