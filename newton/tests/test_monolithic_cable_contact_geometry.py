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
    cable_vertex_tangent,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import cable_contact_oracle as oracle
from scripts.monolithic_reference.cable_contact_criteria import contact_pairs
from scripts.monolithic_reference.cable_contact_criteria import segment_table as judge_segments

R = 0.002  # cable radius [m]
K = 1.0e5  # line stiffness [N/m^2]
S = 1.0e-4  # normal smoothing [m]
L = 0.007  # edge length [m]
EPS = 0.25 * L  # window half width at an interior vertex [m]


@wp.kernel
def _pairs(
    xa: wp.array[Any],
    xb: wp.array[Any],
    y0: wp.array[Any],
    y1: wp.array[Any],
    tau0: wp.array[Any],
    tau1: wp.array[Any],
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
        xa[i],
        xb[i],
        y0[i],
        y1[i],
        tau0[i],
        tau1[i],
        xi[i],
        weight[i],
        radius_sum[i],
        stiffness,
        smoothing,
        eps_start[i],
        eps_end[i],
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
def _tangents(previous: wp.array[Any], vertex: wp.array[Any], following: wp.array[Any], out: wp.array[Any]):
    i = wp.tid()
    out[i] = cable_vertex_tangent(previous[i], vertex[i], following[i])


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


def evaluate_pairs(device, dtype, pairs, stiffness=K, smoothing=S):
    """Run the Warp pair response on a dict of host arrays; returns NumPy results in FP64."""
    vec = wp.vec3d if dtype == wp.float64 else wp.vec3
    count = len(pairs["xi"])
    vectors = [
        wp.array(np.asarray(pairs[k]), dtype=vec, device=device) for k in ("xa", "xb", "y0", "y1", "tau0", "tau1")
    ]
    scalars = [
        wp.array(np.broadcast_to(np.asarray(pairs[k], dtype=float), (count,)).copy(), dtype=dtype, device=device)
        for k in ("xi", "weight", "radius_sum", "eps_start", "eps_end")
    ]
    status = wp.zeros(count, dtype=int, device=device)
    energy, coefficient, t = (wp.zeros(count, dtype=dtype, device=device) for _ in range(3))
    grads = [wp.zeros(count, dtype=vec, device=device) for _ in range(4)]
    normal = wp.zeros(count, dtype=vec, device=device)
    wp.launch(
        _pairs,
        count,
        [*vectors, *scalars, dtype(stiffness), dtype(smoothing), status, energy, *grads, coefficient, normal, t],
        device=device,
    )
    return {
        "status": status.numpy(),
        "energy": energy.numpy().astype(float),
        "gradient": np.stack([g.numpy().astype(float) for g in grads], axis=1),
        "coefficient": coefficient.numpy().astype(float),
        "normal": normal.numpy().astype(float),
        "t": t.numpy().astype(float),
    }


def reference(pairs, *, split=False):
    return oracle.pair_response(
        pairs["xa"],
        pairs["xb"],
        pairs["y0"],
        pairs["y1"],
        pairs["tau0"],
        pairs["tau1"],
        pairs["xi"],
        pairs["weight"],
        pairs["radius_sum"],
        K,
        S,
        pairs["eps_start"],
        pairs["eps_end"],
        split=split,
    )


def random_pairs(count, seed, *, offset=0.0):
    """Near-touching sample edges and target segments with tilted frozen tangents, ramps and free ends."""
    rng = np.random.default_rng(seed)
    xa = rng.normal(scale=0.002, size=(count, 3))
    xb = xa + np.array([L, 0.0, 0.0]) + rng.normal(scale=0.002, size=(count, 3))
    y0 = rng.normal(scale=0.003, size=(count, 3)) + np.array([0.0, -0.002, 0.0035])
    y1 = y0 + np.array([0.0, L, 0.0]) + rng.normal(scale=0.003, size=(count, 3))
    direction = (y1 - y0) / np.linalg.norm(y1 - y0, axis=1)[:, None]

    def tilted():
        tau = direction + rng.normal(scale=0.2, size=(count, 3))
        return tau / np.linalg.norm(tau, axis=1)[:, None]

    shift = np.full(3, offset)
    return {
        "xa": xa + shift,
        "xb": xb + shift,
        "y0": y0 + shift,
        "y1": y1 + shift,
        "tau0": tilted(),
        "tau1": tilted(),
        "xi": rng.choice(np.asarray(GAUSS_RULES[3][0]), size=count),
        "weight": rng.uniform(0.001, 0.003, size=count),
        "radius_sum": np.full(count, 2 * R),
        "eps_start": np.where(rng.uniform(size=count) < 0.7, EPS, 0.0),
        "eps_end": np.where(rng.uniform(size=count) < 0.7, EPS, 0.0),
    }


def bent_target(degrees):
    """Two edges meeting at the origin, the second turned down by ``degrees`` in the x-z plane."""
    bend = np.radians(degrees)
    second = L * np.array([np.cos(bend), 0.0, -np.sin(bend)])
    tau = np.array([np.cos(bend / 2), 0.0, -np.sin(bend / 2)])
    return [np.array([-L, 0.0, 0.0]), np.zeros(3), second], tau, bend


def bent_load(degrees, offset, overlap, s):
    """Load on a sample at arc s from a bent vertex, ``overlap`` deep on the convex, concave or out-of-plane side.

    Returns the force on the sample divided by k P P' w, and the window part of it on the same scale.
    """
    target, tau, bend = bent_target(degrees)
    if s <= 0:
        base, normal = np.array([s, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    else:
        base, normal = s * np.array([np.cos(bend), 0.0, -np.sin(bend)]), np.array([np.sin(bend), 0.0, np.cos(bend)])
    side = {"convex": normal, "concave": -normal, "out-of-plane": np.array([0.0, 1.0, 0.0])}[offset]
    point = base + side * (2 * R - overlap)
    across = np.cross(side, np.array([1.0, 0.0, 0.0]) if offset == "out-of-plane" else np.array([0.0, 1.0, 0.0]))
    xa, xb = point - 0.5 * L * across, point + 0.5 * L * across
    total, window = np.zeros((4, 3)), np.zeros((4, 3))
    first_edge = (target[0], target[1], np.array([1.0, 0.0, 0.0]), tau, 0.0, EPS)
    second_edge = (target[1], target[2], tau, tau, EPS, 0.0)
    for y0, y1, tau0, tau1, es, ee in (first_edge, second_edge):
        _, normal_part, window_part, _, _ = oracle.pair_response(
            xa, xb, y0, y1, tau0, tau1, 0.5, 0.002, 2 * R, K, S, es, ee, split=True
        )
        total += normal_part + window_part
        window += window_part
    _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
    scale = load[0] * 0.002
    return np.linalg.norm(total[0] + total[1]) / scale, np.linalg.norm(window[0] + window[1]) / scale


def bisect(loaded, low=0.0, high=None):
    """Largest axis distance in [low, high] at which ``loaded`` holds; 60 halvings."""
    high = 2 * R if high is None else high
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if loaded(mid) else (low, mid)
    return high


def hairpin_and_helix():
    """A hairpin (two legs 2.5r apart) followed by a second cable wound as a helix."""
    leg = np.column_stack([L * np.arange(20), np.zeros(20), np.zeros(20)])
    turn = np.array([[leg[-1, 0] + 0.003, 0.0025, 0.0]])
    back = np.column_stack([leg[::-1, 0], np.full(20, 0.005), np.zeros(20)])
    u = np.linspace(0, 6 * np.pi, 120)
    helix = np.column_stack([0.02 * np.cos(u), 0.02 * np.sin(u), 0.0045 * u / (2 * np.pi)]) + np.array([0.0, 0.2, 0.0])
    return np.vstack([leg, turn, back, helix]), [(0, 41), (41, 161)]


def test_matches_fp64_reference(test, device):
    """Warp FP64 energy, gradient, Gauss-Newton coefficient and normal equal the NumPy reference."""
    pairs = random_pairs(800, 11)
    warp = evaluate_pairs(device, wp.float64, pairs)
    energy, gradient, coefficient, direction = reference(pairs)
    test.assertTrue(np.all(warp["status"] == 0))
    test.assertGreater(np.count_nonzero(energy > 0), 100)
    scale = np.max(np.abs(gradient))
    np.testing.assert_allclose(warp["energy"], energy, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(warp["gradient"], gradient, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(warp["coefficient"], coefficient, rtol=1e-12, atol=1e-12 * np.max(coefficient))
    engaged = coefficient > 0
    c = np.stack([1 - pairs["xi"], pairs["xi"], -(1 - warp["t"]), -warp["t"]], axis=-1)
    np.testing.assert_allclose((c[..., None] * warp["normal"][:, None, :])[engaged], direction[engaged], atol=1e-12)


def test_fp32_needs_anchor_relative_inputs(test, device):
    """FP32 meets the reference to 1e-4 with anchor-relative inputs; a metre offset loses most of that."""
    pairs = random_pairs(800, 12)
    energy, gradient, _, _ = reference(pairs)
    near = evaluate_pairs(device, wp.float32, pairs)
    far = evaluate_pairs(device, wp.float32, random_pairs(800, 12, offset=1.0))
    test.assertTrue(np.all(near["status"] == 0))
    scale = np.max(np.abs(gradient))
    np.testing.assert_allclose(near["gradient"], gradient, rtol=0, atol=1e-4 * scale)
    np.testing.assert_allclose(near["energy"], energy, rtol=0, atol=1e-4 * np.max(energy))
    error_far = np.max(np.abs(far["gradient"] - gradient)) / scale
    error_near = np.max(np.abs(near["gradient"] - gradient)) / scale
    test.assertGreater(error_far, 10 * error_near)


def test_vertex_tangent_matches_reference(test, device):
    """The frozen window tangent is the bisector of the adjacent edges, also at a folded vertex."""
    rng = np.random.default_rng(17)
    points = np.cumsum(rng.normal(size=(40, 3)) * 0.006, axis=0)
    points[20] = points[18]  # edges 18-19 and 19-20 exactly antiparallel
    segments = oracle.segment_table(points, [(0, 40)], [R])
    expected = oracle.vertex_tangents(points, segments)[1:]
    arrays = [wp.array(points[k], dtype=wp.vec3d, device=device) for k in (slice(0, 38), slice(1, 39), slice(2, 40))]
    out = wp.zeros(38, dtype=wp.vec3d, device=device)
    wp.launch(_tangents, 38, [*arrays, out], device=device)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-12)


def test_exclusion_matches_reference(test, device):
    """The Warp predicate equals the reference pair by pair, and every pair the judge keeps is counted."""
    rest, ranges = hairpin_and_helix()
    radii = [R, R]
    segments, samples = build_tables(rest, ranges, radii)
    edges = len(segments["first"])
    sample, target = np.divmod(np.arange(len(samples["edge"]) * edges), edges)
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
    out = wp.zeros(len(sample), dtype=int, device=device)
    wp.launch(_allowed, len(sample), [*arrays, 3, out], device=device)
    allowed = out.numpy().astype(bool)
    expected_segments = oracle.segment_table(rest, ranges, radii)
    expected = oracle.allowed(
        oracle.sample_table(expected_segments, 3), expected_segments, sample, target, index_span=3, arc_factor=np.pi / 2
    )
    np.testing.assert_array_equal(allowed, expected)
    same = segments["cable"][edge] == segments["cable"][target]
    test.assertFalse(np.any(allowed & same & (np.abs(segments["index"][edge] - segments["index"][target]) <= 3)))
    test.assertFalse(np.any(allowed & same & (segments["index"][edge] >= segments["index"][target])))
    # The hairpin's facing legs and the helix's neighbouring turns stay in contact.
    first_leg = (segments["cable"][edge] == 0) & (segments["index"][edge] == 0) & (segments["index"][target] == 39)
    test.assertTrue(np.any(allowed[first_leg]))
    helix_pair = (segments["cable"][edge] == 1) & (segments["index"][edge] == 5) & (segments["index"][target] == 45)
    test.assertTrue(np.all(allowed[helix_pair]))
    # The PR0 judge only measures pairs the solver counts: each kept edge pair has all its Gauss samples counted.
    judge = judge_segments({"vertex_ranges": np.asarray(ranges), "rest_positions": rest, "radius": np.asarray(radii)})
    i, j, _ = contact_pairs(judge, index_span=3, arc_factor=np.pi / 2)
    gauss = allowed.reshape(len(samples["edge"]), edges)[~samples["tip"]].reshape(edges, 3, edges)
    for a, b in zip(i, j, strict=True):
        lower, upper = min(a, b), max(a, b)
        test.assertTrue(np.all(gauss[lower, :, upper]), (lower, upper))


class TestCableContactGeometry(unittest.TestCase):
    def test_reference_gradient_matches_finite_differences(self):
        """The window term is part of the exact gradient for fixed tangents: central differences agree."""
        pairs = random_pairs(150, 13)
        _, gradient, _, _ = reference(pairs)
        h = 1e-9
        for slot, key in enumerate(("xa", "xb", "y0", "y1")):
            for axis in range(3):
                plus, minus = dict(pairs), dict(pairs)
                plus[key], minus[key] = pairs[key].copy(), pairs[key].copy()
                plus[key][:, axis] += h
                minus[key][:, axis] -= h
                fd = (reference(plus)[0] - reference(minus)[0]) / (2 * h)
                np.testing.assert_allclose(fd, gradient[:, slot, axis], rtol=0, atol=1e-6 * np.max(np.abs(gradient)))

    def test_forces_balance(self):
        """Action equals reaction; the normal part has no net torque (the frozen tangents give the window part one)."""
        pairs = random_pairs(300, 14)
        _, normal, window, coefficient, _ = reference(pairs, split=True)
        x = np.stack([pairs[k] for k in ("xa", "xb", "y0", "y1")], axis=1)
        scale = np.max(np.abs(normal + window))
        np.testing.assert_allclose((normal + window).sum(axis=1), 0.0, atol=1e-13 * scale)
        np.testing.assert_allclose(np.cross(x, normal).sum(axis=1), 0.0, atol=1e-15 * scale)
        self.assertTrue(np.all(coefficient >= 0))

    def test_clamp_switch_at_a_free_end_is_c1(self):
        """Sliding a sample past the free end of a segment (spherical cap) keeps the gradient continuous."""
        y0, y1, tau = np.zeros(3), np.array([L, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        grads = []
        for x in L + np.array([-1e-10, 1e-10]):
            xa, xb = np.array([x - 0.0035, -0.001, 0.0035]), np.array([x + 0.0035, 0.001, 0.0035])
            grads.append(oracle.pair_response(xa, xb, y0, y1, tau, tau, 0.5, 0.002, 2 * R, K, S, 0.0, 0.0)[1])
        np.testing.assert_allclose(grads[0], grads[1], rtol=0, atol=1e-6 * np.max(np.abs(grads[0])))

    def test_near_parallel_sweep_is_smooth(self):
        """Rotating a strand through exactly parallel moves the force continuously (no closest-point jump)."""
        forces = []
        y0, y1, tau = np.array([0.0, 0.0, 0.0037]), np.array([L, 0.0, 0.0037]), np.array([1.0, 0.0, 0.0])
        for angle in np.linspace(-0.02, 0.02, 401):
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            xa, xb = np.array([0.0035, 0.0, 0.0]) - 0.0035 * d, np.array([0.0035, 0.0, 0.0]) + 0.0035 * d
            forces.append(oracle.pair_response(xa, xb, y0, y1, tau, tau, 0.5, 0.002, 2 * R, K, S, EPS, EPS)[1].ravel())
        forces = np.asarray(forces)
        self.assertLess(np.abs(np.diff(forces, axis=0)).max(), 0.02 * np.abs(forces).max())

    def test_straight_parallel_strands_carry_a_uniform_line_load(self):
        """Pressed parallel strands: interior target vertices carry k P P' L each, at any axial phase."""
        count, overlap = 25, 0.0004
        line = np.column_stack([L * np.arange(count), np.zeros(count), np.zeros(count)])
        for phase in (0.0, 0.5 * L, 0.37 * L):
            rest = np.vstack([line + np.array([phase - 3 * L, 0.0, 2 * R - overlap]), line])
            segments = oracle.segment_table(rest, [(0, count), (count, 2 * count)], [R, R])
            samples = oracle.sample_table(segments, 3)
            sample, target = oracle.all_pairs(samples, segments, index_span=3, arc_factor=np.pi / 2)
            _, gradient, _ = oracle.evaluate(rest, segments, samples, sample, target, stiffness=K, smoothing=S)
            _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
            interior = slice(count + 8, 2 * count - 8)
            np.testing.assert_allclose(-gradient[interior, 2], -load[0] * L, rtol=1e-10)
            np.testing.assert_allclose(gradient[interior, :2], 0.0, atol=1e-12 * load[0] * L)

    def test_windows_partition_unity_off_the_axis(self):
        """With a shared frozen tangent the two windows at a bent vertex sum to one for any sample point."""
        rng = np.random.default_rng(18)
        for degrees in (5.0, 20.0, 45.0, 90.0):
            target, tau, _ = bent_target(degrees)
            points = rng.normal(scale=0.004, size=(200, 3))
            first = oracle.window((points - target[0]) @ np.array([1.0, 0.0, 0.0]), points @ tau, 0.0, EPS)[0]
            second = oracle.window(points @ tau, (points - target[2]) @ tau, EPS, 0.0)[0]
            np.testing.assert_allclose(first + second, 1.0, atol=1e-15)

    def test_bent_target_load_stays_in_its_band(self):
        """Across a bent vertex the load never dips on the convex side and stays near k P P' w out of the plane.

        Convex upper bound (PR-01.md): inside a window the extended line of the next edge is at most
        R_sum (1 - cos theta) closer than the vertex, so the load is at most 1 + R_sum (1 - cos theta) / P.
        """
        arcs = np.linspace(-0.5, 0.5, 101) * L
        for degrees in (5.0, 10.0, 20.0, 45.0):
            bend = np.radians(degrees)
            for overlap in (0.0002, 0.0005):
                convex = [bent_load(degrees, "convex", overlap, s)[0] for s in arcs]
                side = [bent_load(degrees, "out-of-plane", overlap, s)[0] for s in arcs]
                with self.subTest(degrees=degrees, overlap=overlap):
                    self.assertGreater(min(convex), 1.0 - 1e-9)
                    self.assertLess(max(convex), 1.0 + 2 * R * (1 - np.cos(bend)) / overlap)
                    self.assertGreater(min(side), 0.94)
                    self.assertLess(max(side), 1.0 + 1e-9)

    def test_bent_target_window_force_stays_within_its_bound(self):
        """The window force is at most 0.75 (dd / eps)(1 + dd / P) of the normal load, dd = 2 sin(theta/2) |p - v|.

        Derivation (PR-01.md): with a shared coordinate the two window slopes are opposite, so the window force
        is (phi_j - phi_k) psi' tau; the two lines through v differ in distance by at most dd, the penalties by
        at most k (P + dd) dd, and the slope is at most 1.5 / (2 eps).
        """
        arcs = np.linspace(-0.5, 0.5, 101) * L
        for degrees in (5.0, 10.0, 20.0, 45.0):
            bend = np.radians(degrees)
            dd = 2 * np.sin(bend / 2) * np.hypot(0.5 * L, 2 * R)
            for offset in ("convex", "concave", "out-of-plane"):
                for overlap in (0.0002, 0.0005):
                    worst = max(bent_load(degrees, offset, overlap, s)[1] for s in arcs)
                    with self.subTest(degrees=degrees, offset=offset, overlap=overlap):
                        self.assertLess(worst, 0.75 * dd / EPS * (1 + dd / overlap))

    def test_worst_phase_crossing_onset_matches_the_sampling_table(self):
        """A perpendicular crossing at the worst phase first loads at 2r - sqrt(4r^2 - delta^2)."""
        tau = np.array([0.0, 1.0, 0.0])
        for samples_per_edge in (2, 3):
            for ratio in (2.0, 3.0, 3.5, 4.0):
                length = ratio * R
                nodes = np.asarray(GAUSS_RULES[samples_per_edge][0])
                gaps = np.r_[np.diff(nodes), 1.0 - nodes[-1] + nodes[0]]
                k = int(np.argmax(gaps))
                crossing = (nodes[k] + gaps[k] / 2) * length
                edges = [(np.array([i * length, 0.0, 0.0]), np.array([(i + 1) * length, 0.0, 0.0])) for i in (-1, 0, 1)]

                def loaded(height, crossing=crossing, edges=edges, nodes=nodes):
                    y0, y1 = np.array([crossing, -0.02, height]), np.array([crossing, 0.02, height])
                    total = 0.0
                    for xa, xb in edges:
                        for xi in nodes:
                            total += oracle.pair_response(xa, xb, y0, y1, tau, tau, xi, 1.0, 2 * R, K, 0.0, 0.0, 0.0)[0]
                    return total > 0

                measured = (2 * R - bisect(loaded)) / R
                with self.subTest(samples=samples_per_edge, ratio=ratio):
                    self.assertAlmostEqual(measured, oracle.onset_overlap(samples_per_edge, ratio), delta=1e-9)
        self.assertAlmostEqual(oracle.onset_overlap(3, 3.5), 0.1183, delta=1e-4)

    def test_tip_pushed_into_a_side_loads_early_in_either_order(self):
        """A free tip pressed end-on into another strand loads within the K1 onset whichever cable samples."""
        count = 9
        poke = np.column_stack([-L * np.arange(count)[::-1], np.zeros(count), np.zeros(count)])  # tip at x = 0
        for tip_first in (True, False):
            for phase in (0.0, 0.5 * L, 0.21 * L):
                side = np.column_stack([np.zeros(count), L * (np.arange(count) - 4) + phase, np.zeros(count)])

                def loaded(gap, tip_first=tip_first, side=side):
                    other = side + np.array([gap, 0.0, 0.0])
                    rest = np.vstack([poke, other] if tip_first else [other, poke])
                    segments = oracle.segment_table(rest, [(0, count), (count, 2 * count)], [R, R])
                    samples = oracle.sample_table(segments, 3)
                    pairs = oracle.all_pairs(samples, segments, index_span=3, arc_factor=np.pi / 2)
                    energy, _, _ = oracle.evaluate(rest, segments, samples, *pairs, stiffness=K, smoothing=0.0)
                    return energy > 0

                onset = (2 * R - bisect(loaded)) / R
                with self.subTest(tip_first=tip_first, phase=phase):
                    self.assertLess(onset, oracle.onset_overlap(3, 3.5) + 1e-9)
                    if tip_first:
                        self.assertLess(onset, 1e-9)  # the tip sample sits on the tip

    def test_build_tables_match_the_reference(self):
        rng = np.random.default_rng(15)
        rest = np.cumsum(rng.uniform(0.006, 0.008, size=(30, 3)) * np.array([1, 0, 0]), axis=0)
        rest = np.vstack([rest, rest + np.array([0.0, 0.05, 0.0])])
        ranges, radii = [(0, 30), (30, 60)], [R, 0.0021]
        segments, samples = build_tables(rest, ranges, radii)
        expected = oracle.segment_table(rest, ranges, radii)
        expected_samples = oracle.sample_table(expected, 3)
        for key in ("first", "cable", "index", "arc0", "arc1", "rest_length", "eps_start", "eps_end", "radius"):
            np.testing.assert_allclose(segments[key], expected[key], rtol=1e-15, err_msg=key)
        for key in ("edge", "xi", "weight", "arc", "tip"):
            np.testing.assert_allclose(samples[key], expected_samples[key], rtol=1e-15, err_msg=key)
        self.assertEqual(int(samples["tip"].sum()), 4)
        np.testing.assert_allclose(samples["weight"][samples["tip"]], [R, R, 0.0021, 0.0021])

    def test_build_tables_reject_unsupported_cables(self):
        rest = np.column_stack([L * np.arange(5), np.zeros(5), np.zeros(5)])
        for radius, pattern in (
            (0.008, "radius exceeds its shortest rest edge"),
            (float("nan"), "finite and positive"),
            (0.0, "finite and positive"),
            (0.0015, "edges are too long"),
        ):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, pattern):
                build_tables(rest, [(0, 5)], [radius])
        with self.assertRaisesRegex(ValueError, "samples per edge"):
            build_tables(rest, [(0, 5)], [R], samples_per_edge=4)
        flat = rest.copy()
        flat[2] = flat[1]
        with self.assertRaisesRegex(ValueError, "zero-length rest edge"):
            build_tables(flat, [(0, 5)], [R])

    def test_sample_on_the_target_axis_is_reported(self):
        tau = np.array([[0.0, 1.0, 0.0]])
        pairs = {
            "xa": [[0.0, 0.0, 0.0]],
            "xb": [[L, 0.0, 0.0]],
            "y0": [[0.0035, -0.0035, 0.0]],
            "y1": [[0.0035, 0.0035, 0.0]],
            "tau0": tau,
            "tau1": tau,
            "xi": [0.5],
            "weight": [0.002],
            "radius_sum": [2 * R],
            "eps_start": [0.0],
            "eps_end": [0.0],
        }
        self.assertEqual(int(evaluate_pairs("cpu", wp.float64, pairs)["status"][0]), 2)

    def test_culled_contact_energy_equals_the_full_sum(self):
        """The offline energy term culls far pairs without changing the total."""
        rest, ranges = hairpin_and_helix()
        positions = rest + np.random.default_rng(16).normal(scale=0.0005, size=rest.shape)
        kwargs = {"stiffness": K, "smoothing": S, "samples_per_edge": 3, "index_span": 3, "arc_factor": np.pi / 2}
        culled, (segments, samples, sample, target) = oracle.contact_energy(positions, rest, ranges, [R, R], **kwargs)
        full, _, _ = oracle.evaluate(positions, segments, samples, sample, target, stiffness=K, smoothing=S)
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(culled, full, delta=1e-15 * max(full, 1.0))


devices = get_test_devices()
for _test in (
    test_matches_fp64_reference,
    test_fp32_needs_anchor_relative_inputs,
    test_vertex_tangent_matches_reference,
    test_exclusion_matches_reference,
):
    add_function_test(TestCableContactGeometry, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
