# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable self and inter-cable contact through the public coupled solver."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_monolithic_material_state import snapshot
from scripts.monolithic_reference import cable_contact_oracle as oracle

R = 0.002  # cable radius [m]
L = 0.007  # rest edge length [m]
DENSITY = 0.015  # linear density [kg/m]


def cable_input(ids, points, *, radius=R, fixed_theta=None):
    """A soft cable through ``points[ids]`` with unit directors perpendicular to each edge of the FP32 Model."""
    ids = tuple(int(i) for i in ids)
    tangents = np.diff(np.asarray(points, dtype=np.float32).astype(np.float64)[list(ids)], axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    hint = np.where(np.abs(tangents[:, 1:2]) > 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    directors = hint - np.sum(hint * tangents, axis=1, keepdims=True) * tangents
    directors /= np.linalg.norm(directors, axis=1, keepdims=True)
    return SolverMonolithic.CableInput(
        particle_ids=ids,
        EA=1000.0,
        bending=np.eye(2) * 1e-4,
        GJ=1e-4,
        radius=radius,
        linear_density=DENSITY,
        rest_theta=np.zeros(len(ids) - 1),
        rest_director=directors,
        fixed_theta=np.zeros(len(ids) - 1, dtype=bool) if fixed_theta is None else np.asarray(fixed_theta, dtype=bool),
    )


def cable_contact_solver(
    points,
    chains,
    *,
    config=None,
    radii=None,
    fixed=(),
    fixed_theta=None,
    ground=None,
    gravity=(0.0, 0.0, 0.0),
    device="cpu",
    precision=SolverMonolithic.Precision.FP64,
    pcg_mode=SolverMonolithic.PCGMode.DIAGNOSTIC,
    use_cuda_graph=False,
    **options,
):
    """A solver over cables through ``points`` (lumped masses from the linear density) with cable contact.

    ``ground`` adds a fixed plane at that height, so shape contact also acts on the cables.
    """
    points = np.asarray(points, dtype=float)
    mass = np.zeros(len(points))
    for chain in chains:
        ids = list(chain)
        half = np.linalg.norm(np.diff(points[ids], axis=0), axis=1) * DENSITY / 2
        mass[ids[:-1]] += half
        mass[ids[1:]] += half
    mass[list(fixed)] = 0.0
    builder = newton.ModelBuilder(gravity=gravity)
    if ground is not None:
        builder.add_shape_plane(
            xform=wp.transform(wp.vec3(0.0, 0.0, ground), wp.quat_identity()),
            width=0.0,
            length=0.0,
            cfg=builder.ShapeConfig(density=0.0, margin=0.0),
        )
    for point, value in zip(points, mass, strict=True):
        builder.add_particle(tuple(point), (0.0, 0.0, 0.0), mass=float(value))
    model = builder.finalize(device=device)
    radii = [R] * len(chains) if radii is None else radii
    fixed_theta = [None] * len(chains) if fixed_theta is None else fixed_theta
    solver = SolverMonolithic(
        model,
        cable_inputs=[
            cable_input(chain, points, radius=r, fixed_theta=f)
            for chain, r, f in zip(chains, radii, fixed_theta, strict=True)
        ],
        cable_contact=config,
        friction_velocity_scale=0.01,
        mass_mode="lumped",
        precision=precision,
        pcg_mode=pcg_mode,
        use_cuda_graph=use_cuda_graph,
        **{"newton_max_iterations": 20, **options},
    )
    return solver, solver.create_state(dt=0.001)


def hairpin(leg=20, overlap=0.0004):
    """One cable folded into a hairpin whose legs press together; contact pairs are |i - j| >= 4 edges apart."""
    first = np.column_stack([L * np.arange(leg), np.zeros(leg), np.zeros(leg)])
    turn = np.array([[first[-1, 0] + 0.003, 0.0, 0.0018]])
    back = np.column_stack([first[::-1, 0], np.zeros(leg), np.full(leg, 2 * R - overlap)])
    points = np.vstack([first, turn, back])
    return points, [range(len(points))]


def strands(count=12, overlap=0.0004, shift=0.0):
    """Two parallel straight cables, the second laid on the first with the given overlap."""
    line = np.column_stack([L * np.arange(count), np.zeros(count), np.zeros(count)])
    upper = line + np.array([shift, 0.0, 2 * R - overlap])
    return np.vstack([line, upper]), [range(count), range(count, 2 * count)]


def hairpin_and_helix(pitch=0.0045):
    """The hairpin of ``hairpin`` followed by a second cable wound as a helix (default: turns 0.5 mm apart)."""
    points, _ = hairpin()
    u = np.linspace(0, 6 * np.pi, 120)
    helix = np.column_stack([0.02 * np.cos(u), 0.02 * np.sin(u), pitch * u / (2 * np.pi)]) + np.array([0, 0.2, 0])
    return np.vstack([points, helix]), [range(len(points)), range(len(points), len(points) + len(helix))]


def reference_gradient(solver, rest, chains, positions, *, stiffness=1e5, smoothing=1e-4):
    """FP64 reference energy and gradient over every allowed pair at ``positions``.

    The rest geometry is the Model's, whose particle positions are stored in FP32.
    """
    rest = np.asarray(rest, dtype=np.float32).astype(np.float64)
    ranges = [(min(c), max(c) + 1) for c in chains]
    radii = [R] * len(chains)
    contact = solver._cable_contact
    segments = oracle.segment_table(rest, ranges, radii)
    samples = oracle.sample_table(segments, contact.samples_per_edge)
    pairs = oracle.all_pairs(samples, segments, index_span=contact.index_exclusion, arc_factor=contact.arc_factor)
    energy, gradient, _ = oracle.evaluate(positions, segments, samples, pairs, stiffness=stiffness, smoothing=smoothing)
    return energy, gradient, (segments, samples, pairs)


class TestMonolithicCableContact(unittest.TestCase):
    precisions = (SolverMonolithic.Precision.FP64, SolverMonolithic.Precision.FP32)

    def test_published_force_matches_the_reference(self):
        """Pressed strands: the published contact force is minus the reference gradient at the published positions."""
        for precision in self.precisions:
            for shift in (0.0, 0.37 * L):
                points, chains = strands(shift=shift)
                solver, state = cable_contact_solver(points, chains, config={"self": False}, precision=precision)
                solver.step(state, state, solver.create_control(), dt=0.001)
                stats = solver.last_stats
                force = solver.get_contact_forces(state).particle_force.astype(float)
                positions = state.particle_q.numpy().astype(float)
                _, gradient, _ = reference_gradient(solver, points, chains, positions)
                with self.subTest(precision=precision.value, shift=shift):
                    self.assertFalse(stats.rolled_back, stats)
                    scale = np.abs(gradient).max()
                    self.assertGreater(scale, 0.0)
                    tolerance = 1e-11 if precision == SolverMonolithic.Precision.FP64 else 2e-3
                    np.testing.assert_allclose(force, -gradient, rtol=0, atol=tolerance * scale)
                    # Equal and opposite: no net force or torque on the pair of strands.
                    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=tolerance * scale)
                    np.testing.assert_allclose(np.cross(positions, force).sum(axis=0), 0.0, atol=tolerance * scale)

    def test_published_self_force_matches_the_reference(self):
        """Self contact (hairpin; hairpin and a helix with pressed turns): the force is minus the reference gradient.

        The reference counts every allowed self pair of both cables, so a device predicate stricter than the
        self-exclusion rule would show as a missing force here; forces balance with no net torque.
        """
        pressed = hairpin_and_helix(pitch=2 * R - 0.0002)
        for precision in self.precisions:
            for name, (points, chains) in (("hairpin", hairpin()), ("hairpin_and_pressed_helix", pressed)):
                solver, state = cable_contact_solver(points, chains, config={}, precision=precision)
                solver.step(state, state, solver.create_control(), dt=0.001)
                stats = solver.last_stats
                force = solver.get_contact_forces(state).particle_force.astype(float)
                positions = state.particle_q.numpy().astype(float)
                _, gradient, _ = reference_gradient(solver, points, chains, positions)
                with self.subTest(precision=precision.value, scene=name):
                    self.assertFalse(stats.rolled_back, stats)
                    scale = np.abs(gradient).max()
                    self.assertGreater(scale, 0.0)
                    tolerance = 1e-11 if precision == SolverMonolithic.Precision.FP64 else 2e-3
                    np.testing.assert_allclose(force, -gradient, rtol=0, atol=tolerance * scale)
                    np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=tolerance * scale)
                    np.testing.assert_allclose(np.cross(positions, force).sum(axis=0), 0.0, atol=tolerance * scale)
                    for chain in chains:
                        # Every cable carries contact force (the helix through its pressed turns).
                        self.assertGreater(np.abs(force[list(chain)]).max(), 0.1 * scale)

    def test_gauge_decision_is_unchanged_by_contact(self):
        """At the step's first evaluation the twist residual and the gauge decision equal those without contact."""
        points, chains = strands()
        seen = {}
        for config in (None, {}):
            solver, state = cable_contact_solver(points, chains, config=config)
            gauges = solver._gauges
            original = gauges.qualify

            def qualify(owner, gauges=gauges, original=original, key=config is None):
                if key not in seen:
                    raw = [block.raw.numpy()[3 * len(block.ids) :].copy() for block in owner._cables]
                    original(owner)
                    seen[key] = (raw, gauges.blocked.numpy().copy(), gauges.pins.numpy().copy())
                    return None
                return original(owner)

            gauges.qualify = qualify
            solver.step(state, state, solver.create_control(), dt=0.001)
            self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
            self.assertTrue(gauges.eligible.numpy().any())
        (raw_on, blocked_on, pins_on), (raw_off, blocked_off, pins_off) = seen[False], seen[True]
        for on, off in zip(raw_on, raw_off, strict=True):
            np.testing.assert_array_equal(on, off)
        np.testing.assert_array_equal(blocked_on, blocked_off)
        np.testing.assert_array_equal(pins_on, pins_off)

    def test_candidates_obey_both_exclusions(self):
        """Self bounds equal the reference rule; every candidate is allowed; adjacent helix turns are candidates."""
        points, chains = hairpin_and_helix()
        solver, state = cable_contact_solver(points, chains, config={})
        contact = solver._cable_contact
        tables = contact.host_tables
        segments, samples = tables["segments"], tables["samples"]
        count = len(segments["first"])
        sample, target = np.divmod(np.arange(len(samples["edge"]) * count), count)
        allowed = oracle.allowed(samples, segments, sample, target, index_span=3, arc_factor=np.pi / 2)
        allowed = allowed.reshape(len(samples["edge"]), count)
        same = segments["cable"][samples["edge"]][:, None] == segments["cable"][None, :]
        expected = np.where((allowed & same).any(axis=1), np.argmax(allowed & same, axis=1), count)
        np.testing.assert_array_equal(tables["self_bounds"], expected)
        solver.step(state, state, solver.create_control(), dt=0.001)
        self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        pairs = contact.pairs.numpy()[: contact._candidate_count]
        segment_pairs = pairs[pairs[:, 2] == 0]
        self.assertTrue(np.all(allowed[segment_pairs[:, 0], segment_pairs[:, 1]]))
        # Vertex pairs come only with both of their segments, and the flags match those pairs.
        keys = {(int(s), int(j)) for s, j in segment_pairs[:, :2]}
        for s, j, _, _bits in pairs[pairs[:, 2] == 1]:
            self.assertIn((s, j), keys)
            self.assertIn((s, j + 1), keys)
        for s, j, _, bits in segment_pairs:
            self.assertEqual(
                bool(bits & 2), (s, j + 1) in keys and segments["cable"][j] == segments["cable"][min(j + 1, count - 1)]
            )
            self.assertEqual(
                bool(bits & 1), j > 0 and (s, j - 1) in keys and segments["cable"][j] == segments["cable"][j - 1]
            )
        helix = segments["cable"][segment_pairs[:, 1]] == 1
        self.assertGreater(np.count_nonzero(helix & (segments["cable"][samples["edge"][segment_pairs[:, 0]]] == 1)), 50)
        hairpin_legs = (segments["cable"][segment_pairs[:, 1]] == 0) & (
            segment_pairs[:, 1] - samples["edge"][segment_pairs[:, 0]] > 3
        )
        self.assertGreater(np.count_nonzero(hairpin_legs), 50)

    def test_contact_adds_no_twist_reaction(self):
        """Contact acts on centerline positions only: its residual on twist rows and the cable torque stay zero."""
        solver, state = cable_contact_solver(*hairpin(), config={})
        solver.step(state, state, solver.create_control(), dt=0.001)
        self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        contact = solver._cable_contact
        residual = contact.pool.residual.result.numpy()
        theta_rows = np.asarray(solver._actors.maps["theta_to_global"])
        theta_rows = theta_rows[theta_rows >= 0]
        self.assertGreater(len(theta_rows), 0)
        self.assertGreater(np.abs(residual).max(), 0.0)
        np.testing.assert_array_equal(residual[theta_rows], 0.0)
        producer = next(p for p in solver._problem.producers if p.name == "cable_contact")
        self.assertEqual(producer._reactions.jobs.size, 3 * sum(len(block.ids) for block in solver._cables))
        np.testing.assert_array_equal(state._monolithic_coupled.contact_cable_torque.numpy(), 0.0)

    def test_invalid_configuration_names_the_field(self):
        """Every invalid key or value is rejected at construction and names its field."""
        points, chains = strands()
        for config, pattern in (
            ({"mu": 0.1}, "Unknown cable contact fields: \\['mu'\\]"),
            ({"friction": 0.1}, "friction"),
            ({"self": False, "inter": False}, "self or inter"),
            ({"self": 1}, "bool"),
            ({"normal_stiffness": 0.0}, "normal_stiffness"),
            ({"normal_stiffness": float("nan")}, "normal_stiffness"),
            ({"normal_stiffness": True}, "normal_stiffness"),
            ({"normal_stiffness": 1e300}, "normal_stiffness"),
            ({"index_exclusion": 2}, "index_exclusion"),
            ({"index_exclusion": 3.0}, "index_exclusion"),
            ({"arc_exclusion_factor": -1.0}, "arc_exclusion_factor"),
            ({"samples_per_edge": 4}, "samples_per_edge"),
            ({"candidate_capacity": -1}, "candidate_capacity"),
            ({"support_capacity": 1.5}, "support_capacity"),
            ({"candidate_padding": -0.001}, "candidate_padding"),
            ("on", "dict"),
        ):
            precision = SolverMonolithic.Precision.FP32 if config == {"normal_stiffness": 1e300} else None
            options = {} if precision is None else {"precision": precision}
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, pattern):
                cable_contact_solver(points, chains, config=config, **options)
        with self.assertRaisesRegex(ValueError, "longer than 4 x"):
            cable_contact_solver(points, chains, config={}, radii=[R, 0.0015])
        # 7 mm edges at r = 2 mm next to 5 mm edges at r = 1.5 mm: only self contact (own radii) allows it.
        mixed = np.vstack(
            [
                np.column_stack([L * np.arange(6), np.zeros(6), np.zeros(6)]),
                np.column_stack([0.005 * np.arange(6), np.full(6, 0.05), np.zeros(6)]),
            ]
        )
        cable_contact_solver(mixed, [range(6), range(6, 12)], config={"inter": False}, radii=[R, 0.0015])
        with self.assertRaisesRegex(ValueError, "smallest contact radius 0.0015 m"):
            cable_contact_solver(mixed, [range(6), range(6, 12)], config={}, radii=[R, 0.0015])
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for x, y in ((0.0, 0.0), (0.03, 0.0), (0.0, 0.03)):
            builder.add_particle(wp.vec3(x, y, 0.0), wp.vec3(0.0), 0.00015)
        builder.add_triangle(0, 1, 2, tri_ke=2, tri_ka=3, tri_kd=0)
        cloth = SolverMonolithic.ClothInput(triangle_ids=(0,), bending=(0.01, 0.01), thickness=0.01, surface_density=1)
        with self.assertRaisesRegex(ValueError, "requires cable_inputs"):
            SolverMonolithic(builder.finalize(), cloth_inputs=[cloth], cable_contact={}, friction_velocity_scale=0.01)

    def test_configuration_enters_the_checkpoint_identity(self):
        """Enabling cable contact or changing its configuration rejects an older checkpoint unchanged."""
        points, chains = strands()
        solver, state = cable_contact_solver(points, chains, config={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cable.npz"
            solver.save_checkpoint(state, path)
            for config, accepted in (({}, True), ({"normal_stiffness": 2e5}, False), (None, False)):
                other, target = cable_contact_solver(points, chains, config=config)
                before = snapshot(target)
                with self.subTest(config=config):
                    if accepted:
                        other.load_checkpoint(target, path)
                    else:
                        with self.assertRaises(ValueError):
                            other.load_checkpoint(target, path)
                        for name, value in before.items():
                            np.testing.assert_array_equal(snapshot(target)[name], value, err_msg=name)

    def test_disabled_contact_builds_nothing(self):
        """Without cable_contact nothing new is built: no contact, producer, identity field or triplet capacity."""
        points, chains = strands()
        disabled, disabled_state = cable_contact_solver(points, chains, config=None)
        enabled, enabled_state = cable_contact_solver(points, chains, config={})
        # No new state fields: the checkpoint schema version stays that of the trunk material layout.
        versions = [s._monolithic_coupled.schema_version.numpy().tolist() for s in (enabled_state, disabled_state)]
        self.assertEqual(versions, [[6], [6]])
        self.assertIsNone(disabled._cable_contact)
        self.assertNotIn("cable_contact", [p.name for p in disabled._problem.producers])
        self.assertIn("cable_contact", [p.name for p in enabled._problem.producers])
        self.assertNotIn("cable_contact", disabled._coupled_config)
        extra = enabled._cable_contact.pool.triplet_capacity
        self.assertEqual(
            enabled._linear.capacities.global_scalar_triplet_count,
            disabled._linear.capacities.global_scalar_triplet_count + extra,
        )

    def test_cloth_and_cable_contact_together(self):
        """Cloth and cable contact construct and step together; cable contact only pairs cable samples and segments."""
        points, chains = strands()
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        mass = np.zeros(len(points))
        for chain in chains:
            ids = list(chain)
            half = np.linalg.norm(np.diff(points[ids], axis=0), axis=1) * DENSITY / 2
            mass[ids[:-1]] += half
            mass[ids[1:]] += half
        for point, value in zip(points, mass, strict=True):
            builder.add_particle(tuple(point), (0.0, 0.0, 0.0), mass=float(value))
        first, face = builder.particle_count, builder.tri_count
        for x, y in ((0.0, 0.05), (0.03, 0.05), (0.0, 0.08)):
            builder.add_particle(wp.vec3(x, y, 0.0), wp.vec3(0.0), 0.00015)
        builder.add_triangle(first, first + 1, first + 2, tri_ke=2, tri_ka=3, tri_kd=0)
        model = builder.finalize()
        solver = SolverMonolithic(
            model,
            cloth_inputs=[
                SolverMonolithic.ClothInput(
                    triangle_ids=(face,), bending=(0.01, 0.01), thickness=0.01, surface_density=1
                )
            ],
            cable_inputs=[cable_input(chain, points) for chain in chains],
            cloth_contact={"radius": 0.02, "normal_stiffness": 1e3, "candidate_capacity": 64},
            cable_contact={},
            friction_velocity_scale=0.01,
            mass_mode="lumped",
            newton_max_iterations=20,
        )
        state = solver.create_state(dt=0.001)
        solver.step(state, state, solver.create_control(), dt=0.001)
        self.assertFalse(solver.last_stats.rolled_back, solver.last_stats)
        names = [p.name for p in solver._problem.producers]
        self.assertLess(names.index("cloth_contact"), names.index("cable_contact"))
        contact = solver._cable_contact
        pairs = contact.pairs.numpy()[: contact._candidate_count]
        cable_particles = set(contact.particles.tolist())
        segments = contact.host_tables["segments"]
        targets = contact.particles[segments["first"][pairs[:, 1]]]
        self.assertTrue(set(targets.tolist()) <= cable_particles)
        self.assertGreater(len(pairs), 0)


if __name__ == "__main__":
    unittest.main()
