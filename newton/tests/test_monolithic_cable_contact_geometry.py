# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-to-segment cable contact geometry, window and local response against the FP64 reference."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_contact_geometry import (
    GAUSS_RULES,
    build_tables,
    cable_pair_allowed,
    cable_pair_response,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import cable_contact_oracle as oracle
from scripts.monolithic_reference.cable_contact_criteria import contact_pairs
from scripts.monolithic_reference.cable_contact_criteria import segment_table as judge_segments

R = 0.002  # cable radius [m]
K = 1.0e5  # line stiffness [N/m^2]
S = 1.0e-4  # normal smoothing [m]


@wp.kernel
def _pairs(
    xa: wp.array[Any],
    xb: wp.array[Any],
    y0: wp.array[Any],
    y1: wp.array[Any],
    xi: wp.array[Any],
    weight: wp.array[Any],
    radius_sum: wp.array[Any],
    eps_start: wp.array[Any],
    eps_end: wp.array[Any],
    stiffness: Any,
    smoothing: Any,
    status: wp.array[int],
    energy: wp.array[Any],
    ga: wp.array[Any],
    gb: wp.array[Any],
    g0: wp.array[Any],
    g1: wp.array[Any],
    coefficient: wp.array[Any],
    normal: wp.array[Any],
    t: wp.array[Any],
):
    i = wp.tid()
    s, e, a, b, c, d, k, n, u = cable_pair_response(
        xa[i], xb[i], y0[i], y1[i], xi[i], weight[i], radius_sum[i], stiffness, smoothing, eps_start[i], eps_end[i]
    )
    status[i] = s
    energy[i] = e
    ga[i] = a
    gb[i] = b
    g0[i] = c
    g1[i] = d
    coefficient[i] = k
    normal[i] = n
    t[i] = u


@wp.kernel
def _allowed(
    cable_a: wp.array[int],
    cable_b: wp.array[int],
    edge_index: wp.array[int],
    segment_index: wp.array[int],
    sample_arc: wp.array[Any],
    arc0: wp.array[Any],
    arc1: wp.array[Any],
    reach: wp.array[Any],
    index_span: int,
    out: wp.array[int],
):
    i = wp.tid()
    out[i] = wp.where(
        cable_pair_allowed(
            cable_a[i],
            cable_b[i],
            edge_index[i],
            segment_index[i],
            sample_arc[i],
            arc0[i],
            arc1[i],
            reach[i],
            index_span,
        ),
        1,
        0,
    )


def evaluate_pairs(device, dtype, xa, xb, y0, y1, xi, weight, radius_sum, eps_start, eps_end, stiffness=K, smoothing=S):
    """Run the Warp pair response on host arrays; returns NumPy results in FP64."""
    vec = wp.vec3d if dtype == wp.float64 else wp.vec3
    count = len(xi)
    inputs = [wp.array(np.asarray(v), dtype=vec, device=device) for v in (xa, xb, y0, y1)]
    scalars = [
        wp.array(np.broadcast_to(np.asarray(v, dtype=float), (count,)).copy(), dtype=dtype, device=device)
        for v in (xi, weight, radius_sum, eps_start, eps_end)
    ]
    status = wp.zeros(count, dtype=int, device=device)
    energy, coefficient, t = (wp.zeros(count, dtype=dtype, device=device) for _ in range(3))
    grads = [wp.zeros(count, dtype=vec, device=device) for _ in range(4)]
    normal = wp.zeros(count, dtype=vec, device=device)
    wp.launch(
        _pairs,
        count,
        [*inputs, *scalars, dtype(stiffness), dtype(smoothing), status, energy, *grads, coefficient, normal, t],
        device=device,
    )
    gradient = np.stack([g.numpy().astype(float) for g in grads], axis=1)
    return {
        "status": status.numpy(),
        "energy": energy.numpy().astype(float),
        "gradient": gradient,
        "coefficient": coefficient.numpy().astype(float),
        "normal": normal.numpy().astype(float),
        "t": t.numpy().astype(float),
    }


def random_pairs(count, seed, *, offset=0.0):
    """Near-touching sample edges and target segments, including clamped ends and window ramps."""
    rng = np.random.default_rng(seed)
    xa = rng.normal(scale=0.002, size=(count, 3))
    xb = xa + np.array([0.007, 0.0, 0.0]) + rng.normal(scale=0.002, size=(count, 3))
    y0 = rng.normal(scale=0.003, size=(count, 3)) + np.array([0.0, -0.002, 0.0035])
    y1 = y0 + np.array([0.0, 0.007, 0.0]) + rng.normal(scale=0.003, size=(count, 3))
    xi = rng.choice(np.asarray(GAUSS_RULES[3][0]), size=count)
    weight = rng.uniform(0.001, 0.003, size=count)
    eps_start = np.where(rng.uniform(size=count) < 0.7, 0.00175, 0.0)
    eps_end = np.where(rng.uniform(size=count) < 0.7, 0.00175, 0.0)
    shift = np.array([offset, offset, offset])
    return xa + shift, xb + shift, y0 + shift, y1 + shift, xi, weight, np.full(count, 2 * R), eps_start, eps_end


def test_matches_fp64_reference(test, device):
    """Warp FP64 energy, gradient, Gauss-Newton coefficient and normal equal the NumPy reference."""
    args = random_pairs(800, 11)
    warp = evaluate_pairs(device, wp.float64, *args)
    energy, gradient, coefficient, direction = oracle.pair_response(*args[:7], K, S, *args[7:])
    test.assertTrue(np.all(warp["status"] == 0))
    test.assertGreater(np.count_nonzero(energy > 0), 100)
    scale = np.max(np.abs(gradient))
    np.testing.assert_allclose(warp["energy"], energy, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(warp["gradient"], gradient, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(warp["coefficient"], coefficient, rtol=1e-12, atol=1e-12 * np.max(coefficient))
    engaged = coefficient > 0
    c = np.stack([1 - args[4], args[4], -(1 - warp["t"]), -warp["t"]], axis=-1)
    np.testing.assert_allclose(
        (c[..., None] * warp["normal"][:, None, :])[engaged], direction[engaged], rtol=0, atol=1e-12
    )


def test_fp32_with_relative_inputs(test, device):
    """FP32 keeps the reference to its rounding when callers pass anchor-relative positions."""
    args = random_pairs(400, 12)
    warp = evaluate_pairs(device, wp.float32, *args)
    energy, gradient, _, _ = oracle.pair_response(*args[:7], K, S, *args[7:])
    test.assertTrue(np.all(warp["status"] == 0))
    scale = np.max(np.abs(gradient))
    np.testing.assert_allclose(warp["gradient"], gradient, rtol=0, atol=2e-3 * scale)
    np.testing.assert_allclose(warp["energy"], energy, rtol=0, atol=2e-3 * np.max(energy))


class TestCableContactGeometry(unittest.TestCase):
    def test_reference_gradient_matches_finite_differences(self):
        """The window term is part of the exact gradient: central differences agree everywhere."""
        args = random_pairs(150, 13)
        _, gradient, _, _ = oracle.pair_response(*args[:7], K, S, *args[7:])
        x = np.stack(args[:4], axis=1)
        h = 1e-9
        for slot in range(4):
            for axis in range(3):
                plus, minus = x.copy(), x.copy()
                plus[:, slot, axis] += h
                minus[:, slot, axis] -= h
                fd = (
                    oracle.pair_response(*plus.transpose(1, 0, 2), *args[4:7], K, S, *args[7:])[0]
                    - oracle.pair_response(*minus.transpose(1, 0, 2), *args[4:7], K, S, *args[7:])[0]
                ) / (2 * h)
                np.testing.assert_allclose(fd, gradient[:, slot, axis], rtol=0, atol=1e-6 * np.max(np.abs(gradient)))

    def test_energy_is_invariant_under_rigid_motion(self):
        """Action equals reaction and the forces have no net torque, window derivative included."""
        args = random_pairs(300, 14)
        _, gradient, coefficient, _ = oracle.pair_response(*args[:7], K, S, *args[7:])
        x = np.stack(args[:4], axis=1)
        scale = np.max(np.abs(gradient))
        np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-13 * scale)
        np.testing.assert_allclose(np.cross(x, gradient).sum(axis=1), 0.0, atol=1e-13 * scale * 0.01)
        self.assertTrue(np.all(coefficient >= 0))

    def test_clamp_switch_at_a_free_end_is_c1(self):
        """Sliding a sample past the free end of a segment (spherical cap) keeps the gradient continuous."""
        y0, y1 = np.array([0.0, 0.0, 0.0]), np.array([0.007, 0.0, 0.0])
        offsets = 0.007 + np.array([-1e-10, 1e-10])
        grads = []
        for x in offsets:
            xa = np.array([x - 0.0035, -0.001, 0.0035])
            xb = np.array([x + 0.0035, 0.001, 0.0035])
            grads.append(oracle.pair_response(xa, xb, y0, y1, 0.5, 0.002, 2 * R, K, S, 0.0, 0.0)[1])
        np.testing.assert_allclose(grads[0], grads[1], rtol=0, atol=1e-6 * np.max(np.abs(grads[0])))

    def test_near_parallel_sweep_is_smooth(self):
        """Rotating a strand through exactly parallel moves the force continuously (no closest-point jump)."""
        angles = np.linspace(-0.02, 0.02, 401)
        forces = []
        for angle in angles:
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            xa = np.array([0.0035, 0.0, 0.0]) - 0.0035 * d
            xb = np.array([0.0035, 0.0, 0.0]) + 0.0035 * d
            y0, y1 = np.array([0.0, 0.0, 0.0037]), np.array([0.007, 0.0, 0.0037])
            forces.append(oracle.pair_response(xa, xb, y0, y1, 0.5, 0.002, 2 * R, K, S, 0.00175, 0.00175)[1].ravel())
        forces = np.asarray(forces)
        steps = np.abs(np.diff(forces, axis=0)).max()
        self.assertLess(steps, 0.02 * np.abs(forces).max())

    def test_straight_parallel_strands_carry_a_uniform_line_load(self):
        """Pressed parallel strands: interior target vertices carry k p L each, at any axial phase."""
        count, length, overlap = 25, 0.007, 0.0004
        line = np.column_stack([length * np.arange(count), np.zeros(count), np.zeros(count)])
        for phase in (0.0, 0.5 * length, 0.37 * length):
            upper = line + np.array([phase - 3 * length, 0.0, 2 * R - overlap])
            rest = np.vstack([upper, line])
            segments = oracle.segment_table(rest, [(0, count), (count, 2 * count)], [R, R])
            samples = oracle.sample_table(segments, 3)
            sample, target = oracle.all_pairs(samples, segments, index_span=3, arc_factor=np.pi / 2)
            _, gradient, _ = oracle.evaluate(rest, segments, samples, sample, target, stiffness=K, smoothing=S)
            _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
            interior = slice(count + 8, 2 * count - 8)
            # The target strand is pushed down by k P P' per unit length; no axial force on a straight pair.
            np.testing.assert_allclose(-gradient[interior, 2], -load[0] * length, rtol=1e-10)
            np.testing.assert_allclose(gradient[interior, :2], 0.0, atol=1e-12 * load[0] * length)

    def test_culled_contact_energy_equals_the_full_sum(self):
        """The offline energy term culls far pairs without changing the total."""
        rest, ranges = _hairpin_and_helix()
        rng = np.random.default_rng(16)
        positions = rest + rng.normal(scale=0.0005, size=rest.shape)
        kwargs = {"stiffness": K, "smoothing": S, "samples_per_edge": 3, "index_span": 3, "arc_factor": np.pi / 2}
        culled, pairs = oracle.contact_energy(positions, rest, ranges, [R, R], **kwargs)
        segments, samples, sample, target = pairs
        full, _, _ = oracle.evaluate(positions, segments, samples, sample, target, stiffness=K, smoothing=S)
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(culled, full, delta=1e-15 * max(full, 1.0))

    def test_bent_target_window_force_stays_within_its_bound(self):
        """At a bend the window derivative adds a force of at most 0.75 sin(theta) times the normal load.

        Derivation (PR-01.md): on the overlap |sigma - L| <= eps the two segment lines differ in distance by at
        most eps sin(theta), so the penalties differ by at most k P eps sin(theta); the window slope is at most
        1.5 / (2 eps). Their product times w is independent of eps.
        """
        length = 0.007
        for degrees in (5.0, 10.0, 20.0, 45.0):
            bend = np.radians(degrees)
            target = [np.array([-length, 0.0, 0.0]), np.zeros(3), length * np.array([np.cos(bend), 0.0, -np.sin(bend)])]
            for overlap in (0.0002, 0.0004):
                _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
                worst = 0.0
                for s in np.linspace(-0.45, 0.45, 181) * length:
                    centre = np.array([s, 0.0, 2 * R - overlap])
                    xa, xb = centre - np.array([0.0, 0.0035, 0.0]), centre + np.array([0.0, 0.0035, 0.0])
                    window = np.zeros((4, 3))
                    for y0, y1, es, ee in (
                        (target[0], target[1], 0.0, 0.25 * length),
                        (target[1], target[2], 0.25 * length, 0.0),
                    ):
                        window += oracle.pair_response(xa, xb, y0, y1, 0.5, 0.002, 2 * R, K, S, es, ee, split=True)[2]
                    worst = max(worst, np.linalg.norm(window[0] + window[1]) / (load[0] * 0.002))
                with self.subTest(degrees=degrees, overlap=overlap):
                    self.assertLess(worst, 0.75 * np.sin(bend))

    def test_worst_phase_crossing_onset_matches_the_sampling_table(self):
        """A perpendicular crossing at the worst phase first loads at 2r - sqrt(4r^2 - delta^2)."""
        for samples_per_edge in (2, 3):
            for ratio in (2.0, 3.0, 3.5, 4.0):
                length = ratio * R
                nodes = np.asarray(GAUSS_RULES[samples_per_edge][0])
                gaps = np.r_[np.diff(nodes), 1.0 - nodes[-1] + nodes[0]]
                k = int(np.argmax(gaps))
                crossing = (nodes[k] + gaps[k] / 2) * length  # along the sampled edge
                sample_edges = [
                    (np.array([i * length, 0.0, 0.0]), np.array([(i + 1) * length, 0.0, 0.0])) for i in (-1, 0, 1)
                ]

                def loaded(height, crossing=crossing, sample_edges=sample_edges, nodes=nodes):
                    y0, y1 = np.array([crossing, -0.02, height]), np.array([crossing, 0.02, height])
                    total = 0.0
                    for xa, xb in sample_edges:
                        for xi in nodes:
                            total += oracle.pair_response(xa, xb, y0, y1, xi, 1.0, 2 * R, K, 0.0, 0.0, 0.0)[0]
                    return total > 0

                low, high = 0.0, 2 * R
                for _ in range(60):
                    mid = (low + high) / 2
                    low, high = (mid, high) if loaded(mid) else (low, mid)
                measured = (2 * R - high) / R
                expected = oracle.onset_overlap(samples_per_edge, ratio)
                with self.subTest(samples=samples_per_edge, ratio=ratio):
                    self.assertAlmostEqual(measured, expected, delta=1e-9)
        self.assertAlmostEqual(oracle.onset_overlap(3, 3.5), 0.1183, delta=1e-4)
        self.assertAlmostEqual(oracle.onset_overlap(2, 3.5), 0.2740, delta=1e-4)

    def test_build_tables_match_the_reference(self):
        rng = np.random.default_rng(15)
        rest = np.cumsum(rng.uniform(0.006, 0.008, size=(30, 3)) * np.array([1, 0, 0]), axis=0)
        rest = np.vstack([rest, rest + np.array([0.0, 0.05, 0.0])])
        ranges, radii = [(0, 30), (30, 60)], [R, 0.0015]
        segments, samples = build_tables(rest, ranges, radii)
        reference = oracle.segment_table(rest, ranges, radii)
        reference_samples = oracle.sample_table(reference, 3)
        for key in ("first", "cable", "index", "arc0", "arc1", "rest_length", "eps_start", "eps_end", "radius"):
            np.testing.assert_allclose(segments[key], reference[key], rtol=1e-15, err_msg=key)
        for key in ("edge", "xi", "weight", "arc"):
            np.testing.assert_allclose(samples[key], reference_samples[key], rtol=1e-15, err_msg=key)
        self.assertEqual(segments["eps_start"][0], 0.0)
        self.assertEqual(segments["eps_end"][28], 0.0)

    def test_build_tables_reject_unsupported_cables(self):
        rest = np.column_stack([0.007 * np.arange(5), np.zeros(5), np.zeros(5)])
        with self.assertRaisesRegex(ValueError, "radius exceeds its shortest rest edge"):
            build_tables(rest, [(0, 5)], [0.008])
        with self.assertRaisesRegex(ValueError, "samples per edge"):
            build_tables(rest, [(0, 5)], [R], samples_per_edge=4)
        flat = rest.copy()
        flat[2] = flat[1]
        with self.assertRaisesRegex(ValueError, "zero-length rest edge"):
            build_tables(flat, [(0, 5)], [R])

    def test_sample_on_the_target_axis_is_reported(self):
        status = evaluate_pairs(
            "cpu",
            wp.float64,
            [[0.0, 0.0, 0.0]],
            [[0.007, 0.0, 0.0]],
            [[0.0035, -0.0035, 0.0]],
            [[0.0035, 0.0035, 0.0]],
            [0.5],
            [0.002],
            [2 * R],
            [0.0],
            [0.0],
        )["status"]
        self.assertEqual(int(status[0]), 2)


def _hairpin_and_helix():
    """A hairpin (two legs 2.5r apart) followed by a second cable wound as a helix."""
    length = 0.007
    leg = np.column_stack([length * np.arange(20), np.zeros(20), np.zeros(20)])
    turn = np.array([[leg[-1, 0] + 0.003, 0.0025, 0.0]])
    back = np.column_stack([leg[::-1, 0], np.full(20, 0.005), np.zeros(20)])
    hairpin = np.vstack([leg, turn, back])
    u = np.linspace(0, 6 * np.pi, 120)
    helix = np.column_stack([0.02 * np.cos(u), 0.02 * np.sin(u), 0.0045 * u / (2 * np.pi)]) + np.array([0.0, 0.2, 0.0])
    return np.vstack([hairpin, helix]), [(0, 41), (41, 161)]


def test_exclusion_matches_reference(test, device):
    """The Warp predicate equals the reference pair by pair, and every pair the judge keeps is counted."""
    rest, ranges = _hairpin_and_helix()
    radii = [R, R]
    segments, samples = build_tables(rest, ranges, radii)
    count = len(samples["edge"]) * len(segments["first"])
    sample, target = np.divmod(np.arange(count), len(segments["first"]))
    edge = samples["edge"][sample]
    reach = np.pi / 2 * (segments["radius"][edge] + segments["radius"][target])
    arrays = [
        wp.array(segments["cable"][edge], dtype=int, device=device),
        wp.array(segments["cable"][target], dtype=int, device=device),
        wp.array(segments["index"][edge], dtype=int, device=device),
        wp.array(segments["index"][target], dtype=int, device=device),
        *(
            wp.array(v, dtype=wp.float64, device=device)
            for v in (samples["arc"][sample], segments["arc0"][target], segments["arc1"][target], reach)
        ),
    ]
    out = wp.zeros(count, dtype=int, device=device)
    wp.launch(_allowed, count, [*arrays, 3, out], device=device)
    allowed = out.numpy().astype(bool)
    reference = oracle.allowed(
        oracle.sample_table(oracle.segment_table(rest, ranges, radii), 3),
        oracle.segment_table(rest, ranges, radii),
        sample,
        target,
        index_span=3,
        arc_factor=np.pi / 2,
    )
    np.testing.assert_array_equal(allowed, reference)
    same = segments["cable"][edge] == segments["cable"][target]
    near = same & (np.abs(segments["index"][edge] - segments["index"][target]) <= 3)
    test.assertFalse(np.any(allowed & near))
    test.assertFalse(np.any(allowed & same & (segments["index"][edge] >= segments["index"][target])))
    # The hairpin's facing legs and the helix's neighbouring turns stay in contact.
    test.assertTrue(
        np.any(
            allowed[(segments["cable"][edge] == 0) & (segments["index"][edge] == 0) & (segments["index"][target] == 39)]
        )
    )
    turn = 40  # segments of one helix turn
    helix_pair = (
        (segments["cable"][edge] == 1) & (segments["index"][edge] == 5) & (segments["index"][target] == 5 + turn)
    )
    test.assertTrue(np.all(allowed[helix_pair]))
    # The PR0 judge only measures pairs the solver counts: every kept edge pair has all its samples counted.
    judge = judge_segments({"vertex_ranges": np.asarray(ranges), "rest_positions": rest, "radius": np.asarray(radii)})
    i, j, _ = contact_pairs(judge, index_span=3, arc_factor=np.pi / 2)
    edges = len(segments["first"])
    counted = allowed.reshape(len(samples["edge"]), edges)
    per_edge = counted.reshape(edges, 3, edges)
    for a, b in zip(i, j, strict=True):
        lower, upper = (
            (a, b)
            if (segments["cable"][a], segments["index"][a]) < (segments["cable"][b], segments["index"][b])
            else (b, a)
        )
        test.assertTrue(np.all(per_edge[lower, :, upper]), (lower, upper))


devices = get_test_devices()
add_function_test(TestCableContactGeometry, "test_matches_fp64_reference", test_matches_fp64_reference, devices=devices)
add_function_test(
    TestCableContactGeometry, "test_fp32_with_relative_inputs", test_fp32_with_relative_inputs, devices=devices
)
add_function_test(
    TestCableContactGeometry, "test_exclusion_matches_reference", test_exclusion_matches_reference, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
