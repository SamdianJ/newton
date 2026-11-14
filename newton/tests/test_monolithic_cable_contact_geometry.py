# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sample-to-polyline cable contact geometry and local response against the FP64 reference."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.cable_contact_geometry import (
    GAUSS_RULES,
    build_tables,
    cable_pair_allowed,
    cable_segment_response,
    cable_vertex_response,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import cable_contact_oracle as oracle
from scripts.monolithic_reference.cable_contact_criteria import contact_pairs
from scripts.monolithic_reference.cable_contact_criteria import segment_table as judge_segments

R = 0.002  # cable radius [m]
K = 1.0e5  # line stiffness [N/m^2]
S = 1.0e-4  # normal smoothing [m]
L = 0.007  # edge length [m]


@wp.kernel
def _segments(
    xa: wp.array[Any],
    xb: wp.array[Any],
    previous: wp.array[Any],
    y0: wp.array[Any],
    y1: wp.array[Any],
    xi: wp.array[Any],
    weight: wp.array[Any],
    radius_sum: wp.array[Any],
    stiffness: Any,
    smoothing: Any,
    start: wp.array[int],
    end: wp.array[int],
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
    s, e, a, b, c, d, k, n, u = cable_segment_response(
        xa[i], xb[i], previous[i], y0[i], y1[i], xi[i], weight[i], radius_sum[i], stiffness, smoothing, start[i], end[i]
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
def _vertices(
    xa: wp.array[Any],
    xb: wp.array[Any],
    vertex: wp.array[Any],
    xi: wp.array[Any],
    weight: wp.array[Any],
    radius_sum: wp.array[Any],
    stiffness: Any,
    smoothing: Any,
    status: wp.array[int],
    energy: wp.array[Any],
    ga: wp.array[Any],
    gb: wp.array[Any],
    gv: wp.array[Any],
    normal: wp.array[Any],
):
    i = wp.tid()
    s, e, a, b, c, n = cable_vertex_response(
        xa[i], xb[i], vertex[i], xi[i], weight[i], radius_sum[i], stiffness, smoothing
    )
    status[i] = s
    energy[i] = e
    ga[i] = a
    gb[i] = b
    gv[i] = c
    normal[i] = n


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


def _launch(kernel, device, dtype, vector_keys, scalar_keys, int_keys, pairs, outputs):
    vec = wp.vec3d if dtype == wp.float64 else wp.vec3
    count = len(pairs["xi"])
    vectors = [wp.array(np.asarray(pairs[k]), dtype=vec, device=device) for k in vector_keys]
    scalars = [
        wp.array(np.broadcast_to(np.asarray(pairs[k], dtype=float), (count,)).copy(), dtype=dtype, device=device)
        for k in scalar_keys
    ]
    ints = [wp.array(np.asarray(pairs[k], dtype=np.int32), dtype=int, device=device) for k in int_keys]
    status = wp.zeros(count, dtype=int, device=device)
    arrays = [wp.zeros(count, dtype=vec if kind == "v" else dtype, device=device) for kind in outputs]
    wp.launch(kernel, count, [*vectors, *scalars, dtype(K), dtype(S), *ints, status, *arrays], device=device)
    return status.numpy(), [a.numpy().astype(float) for a in arrays]


def evaluate_segments(device, dtype, pairs):
    status, (energy, ga, gb, g0, g1, coefficient, normal, t) = _launch(
        _segments,
        device,
        dtype,
        ("xa", "xb", "previous", "y0", "y1"),
        ("xi", "weight", "radius_sum"),
        ("start", "end"),
        pairs,
        "svvvvsvs",
    )
    gradient = np.stack([ga, gb, g0, g1], axis=1)
    return {
        "status": status,
        "energy": energy,
        "gradient": gradient,
        "coefficient": coefficient,
        "normal": normal,
        "t": t,
    }


def evaluate_vertices(device, dtype, pairs):
    status, (energy, ga, gb, gv, normal) = _launch(
        _vertices, device, dtype, ("xa", "xb", "vertex"), ("xi", "weight", "radius_sum"), (), pairs, "svvvv"
    )
    gradient = np.stack([ga, gb, gv], axis=1)
    return {"status": status, "energy": energy, "gradient": gradient, "normal": normal}


def random_pairs(count, seed, *, offset=0.0):
    """Near-touching sample edges, a target polyline piece (previous, y0, y1; vertex y0) and random vertex flags."""
    rng = np.random.default_rng(seed)
    xa = rng.normal(scale=0.002, size=(count, 3))
    xb = xa + np.array([L, 0.0, 0.0]) + rng.normal(scale=0.002, size=(count, 3))
    y0 = rng.normal(scale=0.003, size=(count, 3)) + np.array([0.0, -0.002, 0.0035])
    y1 = y0 + np.array([0.0, L, 0.0]) + rng.normal(scale=0.003, size=(count, 3))
    previous = y0 - np.array([0.0, L, 0.0]) + rng.normal(scale=0.003, size=(count, 3))
    shift = np.full(3, offset)
    pairs = {
        "xa": xa,
        "xb": xb,
        "y0": y0,
        "y1": y1,
        "previous": previous,
        "vertex": y0,
        "xi": rng.choice(np.asarray(GAUSS_RULES[3][0]), size=count),
        "weight": rng.uniform(0.001, 0.003, size=count),
        "radius_sum": np.full(count, 2 * R),
        "start": rng.integers(0, 2, size=count),
        "end": rng.integers(0, 2, size=count),
    }
    for key in ("xa", "xb", "y0", "y1", "previous", "vertex"):
        pairs[key] = pairs[key] + shift
    return pairs


def segment_reference(pairs):
    keys = ("xa", "xb", "y0", "y1", "xi", "weight", "radius_sum")
    return oracle.segment_response(
        *(pairs[k] for k in keys),
        K,
        S,
        previous=pairs["previous"],
        start_vertex=pairs["start"],
        end_vertex=pairs["end"],
    )


def vertex_reference(pairs):
    keys = ("xa", "xb", "vertex", "xi", "weight", "radius_sum")
    return oracle.vertex_response(*(pairs[k] for k in keys), K, S)


def polyline_response(point, polyline, weight=0.002):
    """Energy, force on a point sample and assembled 3x3 point tangent against an open polyline."""
    xa, xb = point, point
    energy, force, tangent = 0.0, np.zeros(3), np.zeros((3, 3))
    last = len(polyline) - 2
    for k in range(last + 1):
        e, g, c, d, _ = oracle.segment_response(
            xa,
            xb,
            polyline[k],
            polyline[k + 1],
            0.0,
            weight,
            2 * R,
            K,
            S,
            previous=polyline[max(k - 1, 0)],
            start_vertex=k > 0,
            end_vertex=k < last,
        )
        energy, force, tangent = energy + e, force - g[0], tangent + c * np.outer(d[0], d[0])
    for k in range(1, len(polyline) - 1):
        e, g, _ = oracle.vertex_response(xa, xb, polyline[k], 0.0, weight, 2 * R, K, S)
        energy, force = energy + e, force - g[0]
    return energy, force, tangent


def bent_polyline(degrees):
    bend = np.radians(degrees)
    return [np.array([-L, 0.0, 0.0]), np.zeros(3), L * np.array([np.cos(bend), 0.0, -np.sin(bend)])], bend


def nearest_offset(point, polyline):
    best = None
    for k in range(len(polyline) - 1):
        a, b = polyline[k], polyline[k + 1]
        t = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        offset = point - a - t * (b - a)
        if best is None or np.linalg.norm(offset) < np.linalg.norm(best):
            best = offset
    return best


def true_distance(point, polyline):
    return np.linalg.norm(nearest_offset(point, polyline))


def nearest_normal(point, polyline):
    offset = nearest_offset(point, polyline)
    return offset / np.linalg.norm(offset)


def bent_points(degrees, offset, overlap):
    """Sample points ``overlap`` deep along the local normal across a bent vertex, plus the convex wedge."""
    polyline, bend = bent_polyline(degrees)
    points = []
    for s in np.linspace(-0.5, 0.5, 81) * L:
        if s <= 0:
            base, normal = np.array([s, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        else:
            base, normal = s * np.array([np.cos(bend), 0.0, -np.sin(bend)]), np.array([np.sin(bend), 0.0, np.cos(bend)])
        side = {"convex": normal, "concave": -normal, "out-of-plane": np.array([0.0, 1.0, 0.0])}[offset]
        points.append(base + side * (2 * R - overlap))
    if offset == "convex":
        for angle in np.linspace(0.0, bend, 21):
            points.append(np.array([np.sin(angle), 0.0, np.cos(angle)]) * (2 * R - overlap))
    return polyline, points


def load_ratios(degrees, offset, overlap):
    """|force| / (k P P' w) with P from the true distance to the polyline, over the swept points."""
    polyline, points = bent_points(degrees, offset, overlap)
    ratios = []
    for point in points:
        _, load, _ = oracle.penalty(np.array([true_distance(point, polyline) - 2 * R]), K, S)
        if load[0] > 0:
            ratios.append(np.linalg.norm(polyline_response(point, polyline)[1]) / (load[0] * 0.002))
    return np.asarray(ratios)


def bisect(loaded, low=0.0, high=None):
    """Largest axis distance in [low, high] at which ``loaded`` holds; 60 halvings."""
    high = 2 * R if high is None else high
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if loaded(mid) else (low, mid)
    return high


def strand_energy(rest, ranges, *, smoothing=S):
    segments = oracle.segment_table(rest, ranges, [R] * len(ranges))
    samples = oracle.sample_table(segments, 3)
    pairs = oracle.all_pairs(samples, segments, index_span=3, arc_factor=np.pi / 2)
    return oracle.evaluate(rest, segments, samples, pairs, stiffness=K, smoothing=smoothing)


def hairpin_and_helix():
    """A hairpin (two legs 2.5r apart) followed by a second cable wound as a helix."""
    leg = np.column_stack([L * np.arange(20), np.zeros(20), np.zeros(20)])
    turn = np.array([[leg[-1, 0] + 0.003, 0.0025, 0.0]])
    back = np.column_stack([leg[::-1, 0], np.full(20, 0.005), np.zeros(20)])
    u = np.linspace(0, 6 * np.pi, 120)
    helix = np.column_stack([0.02 * np.cos(u), 0.02 * np.sin(u), 0.0045 * u / (2 * np.pi)]) + np.array([0.0, 0.2, 0.0])
    return np.vstack([leg, turn, back, helix]), [(0, 41), (41, 161)]


def test_matches_fp64_reference(test, device):
    """Warp FP64 segment and vertex responses equal the NumPy reference, including the dropped tangent terms."""
    pairs = random_pairs(800, 11)
    warp = evaluate_segments(device, wp.float64, pairs)
    energy, gradient, coefficient, direction, t = segment_reference(pairs)
    test.assertTrue(np.all(warp["status"] == 0))
    test.assertGreater(np.count_nonzero(energy > 0), 100)
    clamped = (energy > 0) & ((t == 0.0) | (t == 1.0))
    test.assertGreater(np.count_nonzero(clamped & (coefficient == 0)), 10)
    test.assertGreater(np.count_nonzero(clamped & (coefficient > 0)), 10)
    test.assertTrue(np.all(coefficient >= 0))
    np.testing.assert_allclose(warp["energy"], energy, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(warp["gradient"], gradient, rtol=0, atol=1e-11 * np.max(np.abs(gradient)))
    np.testing.assert_array_equal(warp["coefficient"] == 0, coefficient == 0)
    np.testing.assert_allclose(warp["coefficient"], coefficient, rtol=1e-12, atol=1e-12 * np.max(coefficient))
    np.testing.assert_allclose(warp["t"], t, atol=1e-13)
    engaged = energy > 0
    c = np.stack([1 - pairs["xi"], pairs["xi"], -(1 - warp["t"]), -warp["t"]], axis=-1)
    np.testing.assert_allclose((c[..., None] * warp["normal"][:, None, :])[engaged], direction[engaged], atol=1e-12)
    vwarp = evaluate_vertices(device, wp.float64, pairs)
    venergy, vgradient, vdirection = vertex_reference(pairs)
    test.assertTrue(np.all(vwarp["status"] == 0))
    test.assertGreater(np.count_nonzero(venergy < 0), 50)
    np.testing.assert_allclose(vwarp["energy"], venergy, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(vwarp["gradient"], vgradient, rtol=0, atol=1e-11 * np.max(np.abs(vgradient)))
    np.testing.assert_allclose(vwarp["normal"][venergy < 0], -vdirection[venergy < 0, 2], atol=1e-12)


def test_clamp_ties_keep_one_tangent_term(test, device):
    """Around a shared vertex of a straight or convex polyline exactly one segment keeps its term, ties included."""
    overlap = 0.0004  # deeper than the smoothing: every kept coefficient is w k
    height = 2 * R - overlap
    for dtype in (wp.float32, wp.float64):
        for degrees in (0.0, 30.0):
            bend = np.radians(degrees)
            previous, vertex = np.array([-L, 0.0, 0.0]), np.zeros(3)
            following = L * np.array([np.cos(bend), 0.0, -np.sin(bend)])
            # The incoming end plane (x = 0, an exact tie), just either side of it, and the convex wedge.
            points = [np.array([x, 0.0, height]) for x in (-1e-4 * L, 0.0, 1e-4 * L)]
            points += [height * np.array([np.sin(a), 0.0, np.cos(a)]) for a in np.linspace(0.0, bend, 5)[1:]]
            count = len(points)
            rows = {
                "xa": points,
                "xb": points,
                "xi": np.full(count, 0.5),
                "weight": np.full(count, 0.002),
                "radius_sum": np.full(count, 2 * R),
            }
            incoming = {**rows, "previous": [previous] * count, "y0": [previous] * count, "y1": [vertex] * count}
            outgoing = {**rows, "previous": [previous] * count, "y0": [vertex] * count, "y1": [following] * count}
            incoming.update(start=np.zeros(count), end=np.ones(count))
            outgoing.update(start=np.ones(count), end=np.zeros(count))
            first = evaluate_segments(device, dtype, incoming)
            second = evaluate_segments(device, dtype, outgoing)
            corner = evaluate_vertices(device, dtype, {**rows, "vertex": [vertex] * count})
            kept = (first["coefficient"] > 0).astype(int) + (second["coefficient"] > 0).astype(int)
            total = first["coefficient"] + second["coefficient"]
            energy = first["energy"] + second["energy"] + corner["energy"]
            gaps = [true_distance(p, [previous, vertex, following]) - 2 * R for p in points]
            expected, _, _ = oracle.penalty(np.asarray(gaps), K, S)
            tolerance = 1e-5 if dtype == wp.float32 else 1e-12
            with test.subTest(dtype=dtype.__name__, degrees=degrees):
                test.assertTrue(np.all(first["status"] == 0) and np.all(second["status"] == 0))
                np.testing.assert_array_equal(kept, 1)
                np.testing.assert_allclose(total, 0.002 * K, rtol=tolerance)
                np.testing.assert_allclose(energy, 0.002 * expected, rtol=100 * tolerance)
                # At the exact tie the clamped incoming segment and the vertex measure the same distance.
                test.assertEqual(first["t"][1], 1.0)
                test.assertEqual(first["energy"][1] + corner["energy"][1], 0.0)


def configuration_pairs(positions, rest, ranges):
    """Every allowed segment pair of a configuration as response inputs, with its four vertex indices."""
    segments = oracle.segment_table(rest, ranges, [R] * len(ranges))
    samples = oracle.sample_table(segments, 3)
    pairs = oracle.all_pairs(samples, segments, index_span=3, arc_factor=np.pi / 2)
    start, end = oracle.vertex_flags(segments, pairs)
    sample, target = pairs["segment"]
    edge = samples["edge"][sample]
    a, b = segments["first"][edge], segments["first"][target]
    inputs = {
        "xa": positions[a],
        "xb": positions[a + 1],
        "previous": positions[np.where(start, b - 1, b)],
        "y0": positions[b],
        "y1": positions[b + 1],
        "xi": samples["xi"][sample],
        "weight": samples["weight"][sample],
        "radius_sum": segments["radius"][edge] + segments["radius"][target],
        "start": start,
        "end": end,
    }
    return inputs, np.stack([a, a + 1, b, b + 1], axis=1), (segments, samples, pairs)


def assemble(size, indices, coefficient, direction):
    """Dense tangent sum_k coefficient_k J_k^T J_k over ``size`` vertices; J_k places direction rows at indices."""
    active = coefficient != 0
    jacobian = np.zeros((np.count_nonzero(active), 3 * size))
    rows = np.arange(len(jacobian))
    for slot in range(indices.shape[1]):
        for axis in range(3):
            np.add.at(jacobian, (rows, 3 * indices[active, slot] + axis), direction[active, slot, axis])
    return jacobian.T @ (coefficient[active, None] * jacobian)


def pressed_configuration():
    rest, ranges = hairpin_and_helix()
    return rest + np.random.default_rng(16).normal(scale=0.0005, size=rest.shape), rest, ranges


def test_assembled_tangent_matches_reference(test, device):
    """Over a pressed hairpin and helix, the tangent assembled from Warp equals the reference one."""
    positions, rest, ranges = pressed_configuration()
    inputs, indices, tables = configuration_pairs(positions, rest, ranges)
    _, _, results = oracle.evaluate(positions, *tables, stiffness=K, smoothing=S)
    _, _, coefficient, direction, _ = results["segment"]
    expected = assemble(len(positions), indices, coefficient, direction)
    test.assertGreater(np.count_nonzero(coefficient), 50)
    for dtype, tolerance in ((wp.float64, 1e-10), (wp.float32, 1e-3)):
        warp = evaluate_segments(device, dtype, inputs)
        c = np.stack([1 - inputs["xi"], inputs["xi"], -(1 - warp["t"]), -warp["t"]], axis=-1)
        tangent = assemble(len(positions), indices, warp["coefficient"], c[..., None] * warp["normal"][:, None, :])
        with test.subTest(dtype=dtype.__name__):
            test.assertTrue(np.all(warp["status"] == 0))
            np.testing.assert_allclose(tangent, expected, rtol=0, atol=tolerance * np.max(np.abs(expected)))


def test_fp32_needs_anchor_relative_inputs(test, device):
    """FP32 meets the reference to 1e-4 with anchor-relative inputs; a metre offset loses most of that."""
    pairs = random_pairs(800, 12)
    energy, gradient, _, _, _ = segment_reference(pairs)
    near = evaluate_segments(device, wp.float32, pairs)
    far = evaluate_segments(device, wp.float32, random_pairs(800, 12, offset=1.0))
    test.assertTrue(np.all(near["status"] == 0))
    scale = np.max(np.abs(gradient))
    np.testing.assert_allclose(near["gradient"], gradient, rtol=0, atol=1e-4 * scale)
    np.testing.assert_allclose(near["energy"], energy, rtol=0, atol=1e-4 * np.max(energy))
    error_near = np.max(np.abs(near["gradient"] - gradient)) / scale
    error_far = np.max(np.abs(far["gradient"] - gradient)) / scale
    test.assertGreater(error_far, 10 * error_near)
    vgradient = vertex_reference(pairs)[1]
    vnear = evaluate_vertices(device, wp.float32, pairs)
    np.testing.assert_allclose(vnear["gradient"], vgradient, rtol=0, atol=1e-4 * np.max(np.abs(vgradient)))


def test_exclusion_matches_reference(test, device):
    """The Warp predicate equals the reference pair by pair, and every pair the judge keeps is counted."""
    rest, ranges = hairpin_and_helix()
    radii = [R, R]
    segments, vertices, samples = build_tables(rest, ranges, radii)
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
    reference_segments = oracle.segment_table(rest, ranges, radii)
    reference_samples = oracle.sample_table(reference_segments, 3)
    expected = oracle.allowed(reference_samples, reference_segments, sample, target, index_span=3, arc_factor=np.pi / 2)
    np.testing.assert_array_equal(allowed, expected)
    same = segments["cable"][edge] == segments["cable"][target]
    test.assertFalse(np.any(allowed & same & (np.abs(segments["index"][edge] - segments["index"][target]) <= 3)))
    test.assertFalse(np.any(allowed & same & (segments["index"][edge] >= segments["index"][target])))
    first_leg = (segments["cable"][edge] == 0) & (segments["index"][edge] == 0) & (segments["index"][target] == 39)
    test.assertTrue(np.any(allowed[first_leg]))
    helix_pair = (segments["cable"][edge] == 1) & (segments["index"][edge] == 5) & (segments["index"][target] == 45)
    test.assertTrue(np.all(allowed[helix_pair]))
    # A vertex pair exists exactly when both of its segments are counted.
    grid = allowed.reshape(len(samples["edge"]), edges)
    pairs = oracle.all_pairs(reference_samples, reference_segments, index_span=3, arc_factor=np.pi / 2)
    vertex_sample, row = pairs["vertex"]
    np.testing.assert_array_equal(vertices["vertex"], oracle.vertex_table(reference_segments)["vertex"])
    both = grid[:, vertices["incoming"]] & grid[:, vertices["incoming"] + 1]
    np.testing.assert_array_equal(np.argwhere(both), np.column_stack([vertex_sample, row]))
    # The PR0 judge only measures pairs the solver counts: each kept edge pair has all its Gauss samples counted.
    judge = judge_segments({"vertex_ranges": np.asarray(ranges), "rest_positions": rest, "radius": np.asarray(radii)})
    i, j, _ = contact_pairs(judge, index_span=3, arc_factor=np.pi / 2)
    gauss = grid[~samples["tip"]].reshape(edges, 3, edges)
    for a, b in zip(i, j, strict=True):
        test.assertTrue(np.all(gauss[min(a, b), :, max(a, b)]), (a, b))


class TestCableContactGeometry(unittest.TestCase):
    def test_reference_gradients_match_finite_differences(self):
        """Segment and vertex gradients are exact: central differences agree."""
        pairs = random_pairs(150, 13)
        h = 1e-9
        for response, keys in (
            (segment_reference, ("xa", "xb", "y0", "y1")),
            (vertex_reference, ("xa", "xb", "vertex")),
        ):
            gradient = response(pairs)[1]
            for slot, key in enumerate(keys):
                for axis in range(3):
                    plus, minus = dict(pairs), dict(pairs)
                    plus[key], minus[key] = pairs[key].copy(), pairs[key].copy()
                    plus[key][:, axis] += h
                    minus[key][:, axis] -= h
                    fd = (response(plus)[0] - response(minus)[0]) / (2 * h)
                    np.testing.assert_allclose(fd, gradient[:, slot, axis], atol=1e-6 * np.max(np.abs(gradient)))

    def test_energy_is_invariant_under_rigid_motion(self):
        """No frozen data: forces balance and carry no net torque, for segment and vertex pairs."""
        pairs = random_pairs(300, 14)
        for response, keys in (
            (segment_reference, ("xa", "xb", "y0", "y1")),
            (vertex_reference, ("xa", "xb", "vertex")),
        ):
            gradient = response(pairs)[1]
            x = np.stack([pairs[k] for k in keys], axis=1)
            scale = np.max(np.abs(gradient))
            np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-13 * scale)
            np.testing.assert_allclose(np.cross(x, gradient).sum(axis=1), 0.0, atol=1e-15 * scale)

    def test_clamp_switch_at_a_free_end_is_c1(self):
        """Sliding a sample past the free end of a segment (spherical cap) keeps the gradient continuous."""
        y0, y1 = np.zeros(3), np.array([L, 0.0, 0.0])
        grads = []
        for x in L + np.array([-1e-10, 1e-10]):
            xa, xb = np.array([x - 0.0035, -0.001, 0.0035]), np.array([x + 0.0035, 0.001, 0.0035])
            grads.append(oracle.segment_response(xa, xb, y0, y1, 0.5, 0.002, 2 * R, K, S)[1])
        np.testing.assert_allclose(grads[0], grads[1], atol=1e-6 * np.max(np.abs(grads[0])))

    def test_near_parallel_sweep_is_smooth(self):
        """Rotating a strand through exactly parallel moves the force continuously (no closest-point jump)."""
        forces = []
        y0, y1 = np.array([0.0, 0.0, 0.0037]), np.array([L, 0.0, 0.0037])
        for angle in np.linspace(-0.02, 0.02, 401):
            d = np.array([np.cos(angle), np.sin(angle), 0.0])
            xa, xb = np.array([0.0035, 0.0, 0.0]) - 0.0035 * d, np.array([0.0035, 0.0, 0.0]) + 0.0035 * d
            forces.append(oracle.segment_response(xa, xb, y0, y1, 0.5, 0.002, 2 * R, K, S)[1].ravel())
        forces = np.asarray(forces)
        self.assertLess(np.abs(np.diff(forces, axis=0)).max(), 0.02 * np.abs(forces).max())

    def test_straight_parallel_strands_carry_a_uniform_line_load(self):
        """Pressed parallel strands: interior target vertices carry k P P' L each, at any axial phase."""
        count, overlap = 25, 0.0004
        line = np.column_stack([L * np.arange(count), np.zeros(count), np.zeros(count)])
        _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
        for phase in (0.0, 0.5 * L, 0.37 * L):
            rest = np.vstack([line + np.array([phase - 3 * L, 0.0, 2 * R - overlap]), line])
            _, gradient, _ = strand_energy(rest, [(0, count), (count, 2 * count)])
            interior = slice(count + 8, 2 * count - 8)
            np.testing.assert_allclose(-gradient[interior, 2], -load[0] * L, rtol=1e-10)
            np.testing.assert_allclose(gradient[interior, :2], 0.0, atol=1e-12 * load[0] * L)

    def test_free_ends_add_at_most_one_radius_of_side_load(self):
        """Side by side, each free end of the short strand adds load over one radius if it samples, less if it is hit."""
        count, overlap = 25, 0.0004
        line = np.column_stack([L * np.arange(count), np.zeros(count), np.zeros(count)])
        _, load, _ = oracle.penalty(np.array([-overlap]), K, S)
        for phase in (0.0, 0.5 * L, 0.21 * L):
            upper = line[:12] + np.array([5 * L + phase, 0.0, 2 * R - overlap])  # 11 L long, over the lower strand
            for upper_samples in (True, False):
                rest = np.vstack([upper, line] if upper_samples else [line, upper])
                ranges = [(0, 12), (12, 12 + count)] if upper_samples else [(0, count), (count, count + 12)]
                lower = slice(12, 12 + count) if upper_samples else slice(0, count)
                _, gradient, _ = strand_energy(rest, ranges)
                extra = gradient[lower, 2].sum() / load[0] - 11 * L
                with self.subTest(phase=phase, upper_samples=upper_samples):
                    if upper_samples:
                        # Its two tip samples, weighted by the radius each.
                        self.assertAlmostEqual(extra / (2 * R), 1.0, delta=1e-10)
                    else:
                        # The lower strand's Gauss samples meet the upper strand's end caps (PR-01.md: 0.80-1.56 r).
                        self.assertGreater(extra, 0.0)
                        self.assertLess(extra, 2 * R)

    def test_shared_vertices_are_counted_once(self):
        """Straight and convex: exactly the load of the distance to the polyline; concave: about one to two."""
        for degrees in (5.0, 10.0, 20.0, 45.0, 90.0, 120.0):
            for overlap in (0.0002, 0.0005):
                convex = load_ratios(degrees, "convex", overlap)
                side = load_ratios(degrees, "out-of-plane", overlap)
                with self.subTest(degrees=degrees, overlap=overlap):
                    np.testing.assert_allclose(convex, 1.0, atol=1e-6)
                    self.assertGreater(side.min(), 1.0 - 1e-6)
                    self.assertLess(side.max(), 2.0)
                    # The energy is never below that of the nearest segment; the force magnitude dips where the
                    # second segment pushes at an angle, more for sharper bends and deeper overlap (PR-01.md).
                    concave = load_ratios(degrees, "concave", overlap)
                    self.assertGreater(concave.min(), 0.98 if degrees <= 45.0 else 0.9)
                    self.assertLess(concave.max(), 2.0)

    def test_point_tangent_is_exact_where_segments_clamp(self):
        """A straight polyline has the tangent of one long segment; on the convex side, that of the nearest point.

        Up to 120 degrees: past that, convex-side points of one leg come within reach of the other leg's interior.
        """
        straight = [np.array([-L, 0.0, 0.0]), np.zeros(3), np.array([L, 0.0, 0.0])]
        long = [straight[0], straight[2]]
        for s in np.linspace(-0.8, 0.8, 33) * L:
            for height in (2 * R - 0.0002, 2 * R - 0.0006):
                point = np.array([s, 0.001, height])
                np.testing.assert_allclose(
                    polyline_response(point, straight)[2], polyline_response(point, long)[2], atol=1e-9 * K
                )
        for degrees in (20.0, 45.0, 90.0, 120.0):
            for overlap in (0.00005, 0.0005):  # inside the smoothing band and deeper
                polyline, points = bent_points(degrees, "convex", overlap)
                for point in points:
                    distance = true_distance(point, polyline)
                    _, _, stiffness = oracle.penalty(np.array([distance - 2 * R]), K, S)
                    normal = nearest_normal(point, polyline)
                    with self.subTest(degrees=degrees, overlap=overlap):
                        np.testing.assert_allclose(
                            polyline_response(point, polyline)[2],
                            0.002 * stiffness[0] * np.outer(normal, normal),
                            atol=1e-9 * K,
                        )

    def test_full_tangent_is_psd_and_equals_the_vertex_cancellation(self):
        """Over a pressed hairpin and helix all coefficients are non-negative, the full tangent is PSD and equals
        segment terms minus the vertex terms where an adjacent segment clamps."""
        positions, rest, ranges = pressed_configuration()
        inputs, indices, tables = configuration_pairs(positions, rest, ranges)
        segments, samples, pairs = tables
        _, _, results = oracle.evaluate(positions, *tables, stiffness=K, smoothing=S)
        energy, _, coefficient, direction, _ = results["segment"]
        self.assertTrue(np.all(coefficient >= 0))
        self.assertGreater(np.count_nonzero((coefficient == 0) & (energy > 0)), 10)
        tangent = assemble(len(positions), indices, coefficient, direction)
        # Cancellation form: keep every segment term and subtract the vertex term where a neighbour clamps.
        _, _, full, _, _ = oracle.segment_response(
            *(inputs[k] for k in ("xa", "xb", "y0", "y1", "xi", "weight", "radius_sum")), K, S
        )
        cancelled = assemble(len(positions), indices, full, direction)
        vertices = oracle.vertex_table(segments)
        sample, row = pairs["vertex"]
        edge = samples["edge"][sample]
        a, v = segments["first"][edge], vertices["vertex"][row]
        p = (1 - samples["xi"][sample])[:, None] * positions[a] + samples["xi"][sample][:, None] * positions[a + 1]
        incoming, outgoing = positions[v] - positions[v - 1], positions[v + 1] - positions[v]
        clamps = (np.sum((p - positions[v - 1]) * incoming, axis=1) >= np.sum(incoming * incoming, axis=1)) | (
            np.sum((p - positions[v]) * outgoing, axis=1) <= 0
        )
        _, _, ddphi = oracle.penalty(np.linalg.norm(p - positions[v], axis=1) - 2 * R, K, S)
        _, _, vdirection = results["vertex"]
        weight = np.where(clamps, -samples["weight"][sample] * ddphi, 0.0)
        cancelled += assemble(len(positions), np.stack([a, a + 1, v], axis=1), weight, vdirection)
        scale = np.max(np.abs(tangent))
        np.testing.assert_allclose(tangent, cancelled, rtol=0, atol=1e-10 * scale)
        self.assertGreater(np.linalg.eigvalsh(tangent).min(), -1e-10 * scale)

    def test_worst_phase_crossing_onset_matches_the_sampling_table(self):
        """A perpendicular crossing at the worst phase first loads at 2r - sqrt(4r^2 - delta^2)."""
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
                            total += oracle.segment_response(xa, xb, y0, y1, xi, 1.0, 2 * R, K, 0.0)[0]
                    return total > 0

                with self.subTest(samples=samples_per_edge, ratio=ratio):
                    measured = (2 * R - bisect(loaded)) / R
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
                    return strand_energy(rest, [(0, count), (count, 2 * count)], smoothing=0.0)[0] > 0

                onset = (2 * R - bisect(loaded)) / R
                with self.subTest(tip_first=tip_first, phase=phase):
                    self.assertLess(onset, oracle.onset_overlap(3, 3.5) + 1e-9)
                    if tip_first:
                        self.assertLess(onset, 1e-9)  # the tip sample sits on the tip

    def test_own_tips_pushed_into_the_cable_load_early(self):
        """Self-contact: the start tip (a tip sample) loads from zero; the end tip (a target cap) within K1."""
        body = np.column_stack([L * np.arange(11), np.zeros(11), np.zeros(11)])
        hook = [np.array([10 * L, 3 * L, 0.0]), np.array([5.37 * L, 3 * L, 0.0])]

        def path(gap):
            tip = np.array([5.37 * L, gap, 0.0])  # pointing down at the body
            return np.vstack([body, hook, tip])

        for reverse in (False, True):

            def loaded(gap, reverse=reverse):
                rest = path(gap)[::-1] if reverse else path(gap)
                return strand_energy(rest, [(0, len(rest))], smoothing=0.0)[0] > 0

            onset = (2 * R - bisect(loaded)) / R
            with self.subTest(start_tip=reverse):
                self.assertLess(onset, oracle.onset_overlap(3, 3.5) + 1e-9)
                if reverse:
                    self.assertLess(onset, 1e-9)

    def test_build_tables_match_the_reference(self):
        rng = np.random.default_rng(15)
        rest = np.cumsum(rng.uniform(0.006, 0.008, size=(30, 3)) * np.array([1, 0, 0]), axis=0)
        rest = np.vstack([rest, rest + np.array([0.0, 0.05, 0.0])])
        ranges, radii = [(0, 30), (30, 60)], [R, 0.0021]
        segments, vertices, samples = build_tables(rest, ranges, radii)
        expected = oracle.segment_table(rest, ranges, radii)
        expected_samples = oracle.sample_table(expected, 3)
        for key in ("first", "cable", "index", "arc0", "arc1", "rest_length", "radius"):
            np.testing.assert_allclose(segments[key], expected[key], rtol=1e-15, err_msg=key)
        for key in ("edge", "xi", "weight", "arc", "tip"):
            np.testing.assert_allclose(samples[key], expected_samples[key], rtol=1e-15, err_msg=key)
        for key in ("incoming", "vertex"):
            np.testing.assert_array_equal(vertices[key], oracle.vertex_table(expected)[key])
        self.assertEqual(len(vertices["vertex"]), 56)
        np.testing.assert_allclose(samples["weight"][samples["tip"]], [R, R, 0.0021, 0.0021])

    def test_build_tables_reject_unsupported_cables(self):
        rest = np.column_stack([L * np.arange(5), np.zeros(5), np.zeros(5)])
        for radius, pattern in (
            (0.008, "radius exceeds its shortest rest edge"),
            (0.0017, r"Cable 0 rest edge \d is 0.007 m, longer than 4 x its contact radius 0.0017 m"),
            (float("nan"), "finite and positive"),
            (0.0, "finite and positive"),
        ):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, pattern):
                build_tables(rest, [(0, 5)], [radius])
        with self.assertRaisesRegex(ValueError, "samples per edge"):
            build_tables(rest, [(0, 5)], [R], samples_per_edge=4)
        flat = rest.copy()
        flat[2] = flat[1]
        with self.assertRaisesRegex(ValueError, "zero-length rest edge"):
            build_tables(flat, [(0, 5)], [R])

    def test_degenerate_and_nonfinite_inputs_are_reported(self):
        """A sample on the target or a zero-length target is status 2; a NaN or infinite input is status 1."""
        on_target = {
            "xa": [[0.0, 0.0, 0.0]],
            "xb": [[L, 0.0, 0.0]],
            "previous": [[0.0035, -0.0105, 0.0]],
            "y0": [[0.0035, -0.0035, 0.0]],
            "y1": [[0.0035, 0.0035, 0.0]],
            "vertex": [[0.0035, 0.0, 0.0]],
            "xi": [0.5],
            "weight": [0.002],
            "radius_sum": [2 * R],
            "start": [1],
            "end": [1],
        }
        self.assertEqual(int(evaluate_segments("cpu", wp.float64, on_target)["status"][0]), 2)
        self.assertEqual(int(evaluate_vertices("cpu", wp.float64, on_target)["status"][0]), 2)
        flat = {**on_target, "y1": on_target["y0"]}
        self.assertEqual(int(evaluate_segments("cpu", wp.float64, flat)["status"][0]), 2)
        for value in (np.nan, np.inf):
            for key in ("xa", "y0", "y1", "vertex"):
                broken = {**on_target, key: [[value, 0.0, 0.0]]}
                with self.subTest(value=value, key=key):
                    for dtype in (wp.float32, wp.float64):
                        if key != "vertex":
                            self.assertEqual(int(evaluate_segments("cpu", dtype, broken)["status"][0]), 1)
                        if key not in ("y0", "y1"):
                            self.assertEqual(int(evaluate_vertices("cpu", dtype, broken)["status"][0]), 1)

    def test_culled_contact_energy_equals_the_full_sum(self):
        """The offline energy term culls far pairs without changing the total, which stays non-negative."""
        rest, ranges = hairpin_and_helix()
        positions = rest + np.random.default_rng(16).normal(scale=0.0005, size=rest.shape)
        kwargs = {"stiffness": K, "smoothing": S, "samples_per_edge": 3, "index_span": 3, "arc_factor": np.pi / 2}
        culled, (segments, samples, pairs) = oracle.contact_energy(positions, rest, ranges, [R, R], **kwargs)
        full, _, _ = oracle.evaluate(positions, segments, samples, pairs, stiffness=K, smoothing=S)
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(culled, full, delta=1e-15 * max(full, 1.0))


devices = get_test_devices()
for _test in (
    test_matches_fp64_reference,
    test_clamp_ties_keep_one_tangent_term,
    test_assembled_tangent_matches_reference,
    test_fp32_needs_anchor_relative_inputs,
    test_exclusion_matches_reference,
):
    add_function_test(TestCableContactGeometry, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
