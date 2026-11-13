# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Judge a recorded cable-contact run offline in FP64: penetration, crossing, link, energy and rest.

Independent of the solver's contact code: penetration is the exact segment-segment
distance between every non-excluded segment pair of every published frame. Gate
values come from a gates file; a run judged against a draft file is never PASS.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .cable_oracle import CableReference, _reference_twist, _weighted_strains
from .cloth_contact_run import write_json

FIXTURES = Path(__file__).with_name("fixtures")
FROZEN_GATES = FIXTURES / "cable_contact_gates_frozen_v1.json"
GAUSS = ((1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2)
RING = 64


def judge_files():
    """The judge, the modules whose numbers it uses, and the canonical frozen gates."""
    here = Path(__file__).resolve().parent
    return [Path(__file__).resolve(), here / "cable_oracle.py", here / "cloth_contact_run.py", FROZEN_GATES.resolve()]


def evaluator_identity():
    """Commit, tree status and file hashes of the judge and its inputs, so a changed definition shows."""
    files = judge_files()
    root = files[0].parents[2]

    def git(*command):
        try:
            return subprocess.check_output(["git", *command], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "file": str(files[0]),
        "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "inputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files[1:]},
        "source_sha": git("rev-parse", "HEAD"),
        # Uncommitted changes to the judge, its oracle or the canonical gates all count.
        "file_status": git("status", "--porcelain", "--", *(str(path) for path in files)),
    }


# ---- run loading --------------------------------------------------------------------------


def load_run(directory):
    """Return topology, manifest, summary and the frame sequence (initial state first)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    with np.load(directory / "topology.npz") as data:
        topology = {key: data[key] for key in data.files}
    with np.load(directory / "initial.npz") as data:
        initial = {key: data[key] for key in data.files}
    keys = ("x", "v", "theta", "director", "twist")
    frames = {key: [np.asarray(initial[key], dtype=float)[None]] for key in keys}
    frames["step"] = [np.zeros(1, dtype=np.int64)]
    # initial.npz carries the initial targets from c4c1e330 on; for older runs the first step's
    # target stands in, so an angle target that jumps at step 1 would not count as driven there.
    frames["target_theta"] = (
        [np.asarray(initial["target_theta"], dtype=float)[None]] if "target_theta" in initial else []
    )
    for name in summary.get("chunks", []):
        with np.load(directory / name) as data:
            for key in (*keys, "step"):
                frames[key].append(np.asarray(data[key], dtype=float if key != "step" else np.int64))
            frames["target_theta"].append(np.asarray(data["target_theta"], dtype=float))
    series = {key: np.concatenate(value) for key, value in frames.items() if value}
    if "target_theta" not in series:
        raise ValueError(f"Record {directory} has neither initial nor per-step angle targets")
    if len(series["target_theta"]) == len(series["step"]) - 1:
        series["target_theta"] = np.concatenate([series["target_theta"][:1], series["target_theta"]])
    steps = series["step"]
    if np.any(np.diff(steps) != 1) or steps[0] != 0:
        raise ValueError("Recorded frames are not a contiguous step sequence from the initial state")
    return SimpleNamespace(manifest=manifest, summary=summary, topology=topology, frames=series)


# ---- segment pairs and distances ----------------------------------------------------------


def segment_table(topology):
    """Return per-segment (first vertex, cable, local index, rest arc start, rest arc end, radius)."""
    rows = []
    for cable, (a, b) in enumerate(topology["vertex_ranges"]):
        rest = topology["rest_positions"][a:b]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rest, axis=0), axis=1))]
        for k in range(b - a - 1):
            rows.append((a + k, cable, k, arc[k], arc[k + 1], topology["radius"][cable]))
    table = np.asarray(rows, dtype=float)
    return {
        "first": table[:, 0].astype(np.int64),
        "cable": table[:, 1].astype(np.int64),
        "index": table[:, 2].astype(np.int64),
        "arc0": table[:, 3],
        "arc1": table[:, 4],
        "radius": table[:, 5],
    }


def contact_pairs(segments, *, index_span, arc_factor):
    """Every unordered segment pair that the exclusion rule keeps; flags same-cable pairs."""
    n = len(segments["first"])
    i, j = np.triu_indices(n, k=1)
    same = segments["cable"][i] == segments["cable"][j]
    gap = np.maximum.reduce(
        [np.zeros(len(i)), segments["arc0"][i] - segments["arc1"][j], segments["arc0"][j] - segments["arc1"][i]]
    )
    reach = arc_factor * (segments["radius"][i] + segments["radius"][j])
    excluded = same & ((np.abs(segments["index"][i] - segments["index"][j]) <= index_span) | (gap < reach))
    keep = ~excluded
    return i[keep], j[keep], same[keep]


def segment_distance(p1, q1, p2, q2):
    """Exact closest distance and parameters of segment pairs (rows), Ericson's clamped solution."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.sum(d1 * d1, axis=-1), np.sum(d2 * d2, axis=-1)
    b, c, f = np.sum(d1 * d2, axis=-1), np.sum(d1 * r, axis=-1), np.sum(d2 * r, axis=-1)
    if np.any(a <= 0) or np.any(e <= 0):
        raise ValueError("Degenerate cable segment")
    denom = a * e - b * b
    parallel = denom <= 1e-14 * a * e
    s = np.where(parallel, 0.0, np.clip((b * f - c * e) / np.where(parallel, 1.0, denom), 0.0, 1.0))
    t = (b * s + f) / e
    low, high = t < 0, t > 1
    t = np.clip(t, 0.0, 1.0)
    s = np.where(low, np.clip(-c / a, 0.0, 1.0), np.where(high, np.clip((b - c) / a, 0.0, 1.0), s))
    delta = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(delta, axis=-1), s, t


def _near(x, segments, i, j, slack):
    p, q = x[segments["first"][i]], x[segments["first"][i] + 1]
    u, w = x[segments["first"][j]], x[segments["first"][j] + 1]
    reach = (np.linalg.norm(q - p, axis=1) + np.linalg.norm(w - u, axis=1)) / 2
    reach += segments["radius"][i] + segments["radius"][j] + slack
    return np.linalg.norm((p + q) / 2 - (u + w) / 2, axis=1) <= reach


def frame_penetration(x, segments, i, j):
    """Return per-pair penetration [m] (positive = overlap) for the pairs near each other."""
    near = _near(x, segments, i, j, 0.0)
    ii, jj = i[near], j[near]
    first = segments["first"]
    d, _, _ = segment_distance(x[first[ii]], x[first[ii] + 1], x[first[jj]], x[first[jj] + 1])
    return near, segments["radius"][ii] + segments["radius"][jj] - d


def crossing_events(x0, x1, segments, i, j, *, min_sine):
    """Pairs whose line-line signed distance changes sign between two frames with interior closest points."""
    near = _near(x0, segments, i, j, 0.0) | _near(x1, segments, i, j, 0.0)
    ii, jj = i[near], j[near]
    first = segments["first"]
    radius = segments["radius"][ii] + segments["radius"][jj]
    signs, interior, close = [], [], []
    for x in (x0, x1):
        p, q, u, w = x[first[ii]], x[first[ii] + 1], x[first[jj]], x[first[jj] + 1]
        d1, d2 = q - p, w - u
        normal = np.cross(d1, d2)
        size = np.linalg.norm(normal, axis=1)
        sine = size / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1))
        height = np.sum((u - p) * normal, axis=1) / np.where(size > 0, size, 1.0)
        a, e, b = np.sum(d1 * d1, axis=1), np.sum(d2 * d2, axis=1), np.sum(d1 * d2, axis=1)
        r = p - u
        c, f = np.sum(d1 * r, axis=1), np.sum(d2 * r, axis=1)
        denom = np.where(sine > min_sine, a * e - b * b, 1.0)
        s, t = (b * f - c * e) / denom, (a * f - b * c) / denom
        signs.append(np.where(sine > min_sine, np.sign(height), 0.0))
        interior.append((sine > min_sine) & (s > 0) & (s < 1) & (t > 0) & (t < 1))
        distance, _, _ = segment_distance(p, q, u, w)
        close.append(distance < radius)
    flipped = (signs[0] * signs[1] < 0) & interior[0] & interior[1] & (close[0] | close[1])
    return ii[flipped], jj[flipped]


# ---- link: twist and writhe of a clamped cable ------------------------------------------


def writhe(points):
    """Gauss writhe of a closed polygon (Klenin and Langowski 2000), points without repetition."""
    p1 = points
    p2 = np.roll(points, -1, axis=0)
    n = len(points)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    a, b, c, d = p1[i], p2[i], p1[j], p2[j]
    r13, r14, r23, r24 = c - a, d - a, c - b, d - b

    def unit(v):
        size = np.linalg.norm(v, axis=1)
        return v / np.where(size > 0, size, 1.0)[:, None], size > 0

    n1, k1 = unit(np.cross(r13, r14))
    n2, k2 = unit(np.cross(r14, r24))
    n3, k3 = unit(np.cross(r24, r23))
    n4, k4 = unit(np.cross(r23, r13))
    valid = k1 & k2 & k3 & k4

    def angle(u, v):
        return np.arcsin(np.clip(np.sum(u * v, axis=1), -1.0, 1.0))

    omega = angle(n1, n2) + angle(n2, n3) + angle(n3, n4) + angle(n4, n1)
    omega *= np.sign(np.sum(np.cross(d - c, b - a) * r13, axis=1))
    return float(2 * np.sum(np.where(valid, omega, 0.0)) / (4 * np.pi))


def closed_writhe(x, closure):
    """Writhe of a clamped cable closed by a planar rectangle ``closure`` metres beyond its ends."""
    start, end = x[0], x[-1]
    axis = end - start
    axis /= np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0])
    up = up - np.dot(up, axis) * axis
    up /= np.linalg.norm(up)
    loop = [end + closure * axis, end + closure * axis + closure * up, start - closure * axis + closure * up]
    loop.append(start - closure * axis)
    return writhe(np.vstack([x, np.asarray(loop)]))


def total_twist(theta, twist):
    """Total material twist of one cable in turns: sum of (dtheta + reference twist) / 2 pi."""
    return float((np.sum(np.diff(theta)) + np.sum(twist)) / (2 * np.pi))


# ---- energy ---------------------------------------------------------------------------------


def _frame(x, director, twist):
    tangents = np.diff(x, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    director = director - np.sum(director * tangents, axis=1)[:, None] * tangents
    director /= np.linalg.norm(director, axis=1)[:, None]
    return SimpleNamespace(positions=x, directors=director, reference_twist=_reference_twist(tangents, director, twist))


def elastic_energy(rest, x, theta, frame):
    return float(sum(0.5 * np.dot(s, s) for s in _weighted_strains(rest, x, theta, frame)))


def elastic_gradient(rest, x, theta, frame, vertices, edges):
    """dE/dx at chosen local vertices [N] and dE/dtheta at chosen local edges [N m], complex step."""
    gx = np.zeros((len(vertices), 3))
    gt = np.zeros(len(edges))
    h = 1e-30
    for row, vertex in enumerate(vertices):
        for axis in range(3):
            trial = x.astype(np.complex128)
            trial[vertex, axis] += h * 1j
            gx[row, axis] = sum(0.5 * np.dot(s, s) for s in _weighted_strains(rest, trial, theta, frame)).imag / h
    for row, edge in enumerate(edges):
        trial = theta.astype(np.complex128)
        trial[edge] += h * 1j
        gt[row] = sum(0.5 * np.dot(s, s) for s in _weighted_strains(rest, x, trial, frame)).imag / h
    return gx, gt


def penalty(gap, stiffness, smoothing):
    """C-IE normal penalty energy density [J/m^2]: k p^2 / 2 with the C2-smoothed overlap p."""
    u = -np.asarray(gap, dtype=float)
    p = np.where(u >= smoothing, u, 0.0)
    middle = (u > -smoothing) & (u < smoothing)
    t = (u + smoothing) / (2 * smoothing)
    p = np.where(middle, smoothing * t**3 * (2 - t), p)
    return stiffness * p * p / 2


def surface_samples(x, theta, director, radius, rest_lengths):
    """Heights [m] and areas [m^2] of one cable's ring64 surface and end-cap samples (solver quadrature)."""
    edges = np.diff(x, axis=0)
    t = edges / np.linalg.norm(edges, axis=1)[:, None]
    d = director - np.sum(director * t, axis=1)[:, None] * t
    d /= np.linalg.norm(d, axis=1)[:, None]
    perpendicular = np.cross(t, d)
    phi = 2 * np.pi * np.arange(RING) / RING
    heights, areas = [], []
    for a in GAUSS:
        angle = phi[None, :] + theta[:, None]
        centre = (1 - a) * x[:-1] + a * x[1:]
        z = centre[:, None, 2] + radius * (np.cos(angle) * d[:, None, 2] + np.sin(angle) * perpendicular[:, None, 2])
        heights.append(z.ravel())
        areas.append(np.repeat(2 * np.pi * radius * rest_lengths / (2 * RING), RING))
    for end, edge in ((0, 0), (1, len(edges) - 1)):
        for a in GAUSS:
            angle = phi + theta[edge]
            radial = radius * np.sqrt(a)
            z = x[edge + end, 2] + radial * (np.cos(angle) * d[edge, 2] + np.sin(angle) * perpendicular[edge, 2])
            heights.append(z)
            areas.append(np.full(RING, np.pi * radius**2 / (2 * RING)))
    return np.concatenate(heights), np.concatenate(areas)


def ground_contact(x, theta, director, radius, rest_lengths, height, stiffness, smoothing):
    """Penalty energy [J] and deepest overlap [m] of one cable's surface samples against the plane z = height.

    Uses the solver's quadrature points and areas but no orientation weight; for a
    ring resting on a plane only downward samples overlap, so this bounds the solver value.
    """
    z, area = surface_samples(x, theta, director, radius, rest_lengths)
    gap = z - height
    return float(np.sum(area * penalty(gap, stiffness, smoothing))), float(-gap.min())


def driver_work(rest, xs, thetas, directors, twists, vertices, edges):
    """Per-frame work [J] done on one cable by its prescribed vertices and angles (trapezoid rule).

    The prescribed vertices are massless and kinematic, so the force they supply equals
    the elastic gradient at them; a prescribed angle supplies the elastic torque. Entry k is
    the work of the step from frame k - 1 to frame k (entry 0 is zero).
    """
    work = np.zeros(len(xs))
    if not vertices and not edges:
        return work
    previous = None
    for k in range(len(xs)):
        frame = _frame(xs[k], directors[k], twists[k])
        forces = elastic_gradient(rest, xs[k], thetas[k], frame, vertices, edges)
        if previous is not None:
            dx = xs[k][vertices] - xs[k - 1][vertices]
            dtheta = thetas[k][edges] - thetas[k - 1][edges]
            work[k] = 0.5 * np.sum((previous[0] + forces[0]) * dx) + 0.5 * np.sum((previous[1] + forces[1]) * dtheta)
        previous = forces
    return work


def closure_clearance(x, closure):
    """Smallest distance [m] from interior cable vertices to the five writhe-closure segments.

    Approximate: a segment can pass a closure line while both of its vertices stay up to half a
    segment away, so a clearance below half the longest edge flags a possible spurious writhe jump.
    """
    start, end = x[0], x[-1]
    axis = (end - start) / np.linalg.norm(end - start)
    up = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
    up /= np.linalg.norm(up)
    corners = [end, end + closure * axis, end + closure * axis + closure * up, start - closure * axis + closure * up]
    corners += [start - closure * axis, start]
    points = x[2:-2]
    best = np.inf
    for a, b in itertools.pairwise(corners):
        u = b - a
        s = np.clip((points - a) @ u / (u @ u), 0.0, 1.0)
        best = min(best, float(np.min(np.linalg.norm(points - (a + s[:, None] * u), axis=1))))
    return best


# ---- evaluation -----------------------------------------------------------------------------


def max_rise(values):
    """Largest increase max_{t1 <= t2} values[t2] - values[t1] of a time series (0 if never rising)."""
    values = np.asarray(values, dtype=float)
    return float(np.max(values - np.minimum.accumulate(values)))


def evaluate(run, gates):
    """Compute every criterion of ``gates`` on a loaded run; returns (criteria, per-frame series)."""
    topo, frames = run.topology, run.frames
    recipe = run.manifest["scene"]["recipe"]
    scene, kind = recipe["scene"], recipe["kind"]
    if run.manifest["arguments"].get("method", "off") != "off":
        raise NotImplementedError("Cable-cable contact energy needs the PR1 contact oracle; method-off runs only")
    exclusion = gates["exclusion"]
    segments = segment_table(topo)
    i, j, same = contact_pairs(segments, index_span=exclusion["index_span"], arc_factor=exclusion["arc_factor"])
    radius_pair = (segments["radius"][i] + segments["radius"][j]) / 2
    count = len(frames["step"])
    peak = np.zeros(count)
    peak_self, peak_inter = np.zeros(count), np.zeros(count)
    overlapping = np.zeros(count, dtype=np.int64)
    for k in range(count):
        near, depth = frame_penetration(frames["x"][k], segments, i, j)
        if depth.size:
            relative = depth / radius_pair[near]
            peak[k] = max(0.0, relative.max())
            overlapping[k] = int(np.count_nonzero(depth > 0))
            if np.any(same[near]):
                peak_self[k] = max(0.0, relative[same[near]].max())
            if np.any(~same[near]):
                peak_inter[k] = max(0.0, relative[~same[near]].max())
    crossings = []
    for k in range(count - 1):
        a, b = crossing_events(
            frames["x"][k], frames["x"][k + 1], segments, i, j, min_sine=gates["crossing"]["min_sine"]
        )
        crossings.extend(
            {"step": int(frames["step"][k + 1]), "segments": [int(p), int(q)]} for p, q in zip(a, b, strict=True)
        )

    # Energy, driver work and link, per cable in FP64.
    mass, gravity = topo["particle_mass"], topo["gravity"]
    fixed = {int(v) for v in topo["fixed_vertices"]}
    solver = recipe["solver"]
    height = float(topo["ground_height"][0])
    kinetic = 0.5 * np.einsum("i,fij->f", mass, frames["v"] ** 2)
    potential = -np.einsum("i,fi->f", mass, frames["x"] @ gravity)
    elastic, ground = np.zeros(count), np.zeros(count)
    ground_depth = np.full(count, -np.inf)
    work = np.zeros(count)
    link = {"twist_turns": [], "writhe_turns": [], "applied_turns": []}
    closure = []
    for cable, (a, b) in enumerate(topo["vertex_ranges"]):
        e0 = int(topo["theta_offsets"][cable])
        e1 = e0 + (b - a - 1)
        w0 = e0 - cable
        w1 = w0 + (b - a - 2)
        rest = CableReference(
            topo["rest_positions"][a:b],
            np.zeros(b - a - 1),
            topo["rest_directors"][e0:e1],
            (topo["EA"][cable], float(topo["EI"][cable]), topo["GJ"][cable]),
        )
        xs, thetas = frames["x"][:, a:b], frames["theta"][:, e0:e1]
        directors, twists = frames["director"][:, e0:e1], frames["twist"][:, w0:w1]
        local_fixed = [v - a for v in range(a, b) if v in fixed]
        clamped = [k for k in range(b - a - 1) if np.ptp(frames["target_theta"][:, e0 + k]) > 0]
        work += driver_work(rest, xs, thetas, directors, twists, local_fixed, clamped)
        for k in range(count):
            frame = _frame(xs[k], directors[k], twists[k])
            elastic[k] += elastic_energy(rest, xs[k], thetas[k], frame)
            if np.isfinite(height):
                energy, depth = ground_contact(
                    xs[k],
                    thetas[k],
                    directors[k],
                    topo["radius"][cable],
                    rest.lengths,
                    height,
                    solver["contact_stiffness"],
                    solver["normal_smoothing"],
                )
                ground[k] += energy
                ground_depth[k] = max(ground_depth[k], depth)
            if kind == "plectoneme":
                link["twist_turns"].append(total_twist(thetas[k], twists[k]))
                link["writhe_turns"].append(closed_writhe(xs[k], gates["link"]["closure_m"]))
                target = frames["target_theta"][k]
                link["applied_turns"].append((target[e1 - 1] - target[e0]) / (2 * np.pi))
                closure.append(closure_clearance(xs[k], gates["link"]["closure_m"]))
    work = np.cumsum(work)
    total = kinetic + potential + elastic + ground
    adjusted = total - work
    rise = max_rise(adjusted)
    gravity_drop = float(potential[0] - potential[-1])

    dt = float(topo["dt"][0])
    window = frames["step"] * dt >= frames["step"][-1] * dt - gates["rest"]["window_s"]
    free = np.array([v not in fixed for v in range(frames["v"].shape[1])])
    rest_speed = float(np.max(np.linalg.norm(frames["v"][window][:, free], axis=2))) if count > 1 else None
    ground_depth = float(np.max(ground_depth)) if np.isfinite(height) else None
    curvature_radius = []
    for a, b in topo["vertex_ranges"]:
        e = np.diff(frames["x"][:, a:b], axis=1)
        lengths = np.linalg.norm(e, axis=2)
        cosine = np.clip(np.sum(e[:, :-1] * e[:, 1:], axis=2) / (lengths[:, :-1] * lengths[:, 1:]), -1, 1)
        kappa = 2 * np.tan(np.arccos(cosine) / 2) / ((lengths[:, :-1] + lengths[:, 1:]) / 2)
        curvature_radius.append(float(1 / np.max(kappa)) if np.max(kappa) > 0 else float("inf"))

    status = run.summary["status"]
    counts = run.summary["publication_counts"]
    gate_results = {}
    penetration_gate = gates["penetration"]
    if scene in penetration_gate["scenes"]:
        gate_results["penetration_peak_radius_fraction"] = {
            "value": float(peak.max()),
            "limit": penetration_gate["peak_max_radius_fraction"],
            "pass": bool(peak.max() <= penetration_gate["peak_max_radius_fraction"]),
        }
    gate_results["hard_failures"] = {
        "value": counts["HARD_FAILURE"],
        "limit": gates["hard_failure_max"],
        "pass": counts["HARD_FAILURE"] <= gates["hard_failure_max"],
    }
    energy_gate = gates["energy"]
    if scene in energy_gate["scenes"]:
        ratio = rise / gravity_drop if gravity_drop > 0 else float("inf")
        gate_results["energy_rise_fraction_of_gravity_drop"] = {
            "value": ratio,
            "limit": energy_gate["max_rise_fraction_of_gravity_drop"],
            "pass": bool(ratio <= energy_gate["max_rise_fraction_of_gravity_drop"]),
        }
    complete = status == "COMPLETE"
    draft = gates.get("status") != "frozen"
    frozen_recipes = set((gates.get("frozen") or {}).get("recipes_sha256", {}).values())
    recipe_frozen = run.manifest.get("recipe_sha256") in frozen_recipes
    # Missing receipt fields count as unclean: a receipt must show a clean, unchanged tree.
    source_clean = run.manifest.get("git_status") == "" and run.manifest.get("source_files_changed_during_run") == []
    gated = scene in penetration_gate["scenes"]
    verdict = "PASS" if complete and all(g["pass"] for g in gate_results.values()) else "FAIL"
    if not complete:
        verdict = "PARTIAL" if status == "PARTIAL" else "FAIL"
    # A PASS binds a frozen recipe, frozen gates and committed source; anything else is not a PASS.
    if verdict == "PASS":
        if not gated:
            verdict = "NOT_GATED"
        elif draft:
            verdict = "PASS_ON_DRAFT_GATES"
        elif not recipe_frozen:
            verdict = "NOT_FROZEN_RECIPE"
        elif not source_clean:
            verdict = "DIRTY_SOURCE"
    stopped = counts["STOPPED_ACCEPTED"]
    criteria = {
        "scene": scene,
        "run_status": status,
        "source_sha": run.manifest["source_sha"],
        "precision": run.manifest["arguments"]["precision"],
        "device": run.manifest["arguments"]["device"],
        "method": run.manifest["arguments"].get("method", "off"),
        "gates_status": gates.get("status"),
        "evaluator": evaluator_identity(),
        "recipe_sha256": run.manifest.get("recipe_sha256"),
        "recipe_frozen": recipe_frozen,
        "source_clean": source_clean,
        "verdict": verdict,
        "gates": gate_results,
        "diagnostics": {
            "publication_counts": counts,
            "stopped_fraction": stopped / max(1, sum(counts.values())),
            "penetration_peak_radius_fraction": float(peak.max()),
            "penetration_p99_radius_fraction": float(np.percentile(peak, 99)),
            "penetration_peak_self": float(peak_self.max()),
            "penetration_peak_inter": float(peak_inter.max()),
            "penetration_peak_step": int(frames["step"][int(np.argmax(peak))]),
            "overlapping_pairs_max": int(overlapping.max()),
            "crossing_events": len(crossings),
            "crossing_limits": (
                "lower bound: misses passages at less than the min_sine angle (stacked near-parallel turns), "
                "tunnelling with neither frame closer than rA + rB, and a crossing point that moves to the "
                "adjacent segment between frames"
            ),
            "crossing_first": crossings[:20],
            "energy_max_rise_J": rise,
            "energy_scope": (
                "kinetic + elastic + gravity"
                + (" + ground-plane penalty" if np.isfinite(height) else "")
                + " - driver work"
                + "; cable-cable contact energy is zero for method-off runs"
            ),
            "gravity_drop_J": gravity_drop,
            "driver_work_J": float(work[-1]),
            "ground_penetration_max_m": ground_depth,  # deepest surface sample below the plane
            "rest_window_max_speed_m_s": rest_speed,
            "min_curvature_radius_over_radius": [r / float(np.max(topo["radius"])) for r in curvature_radius],
            "contact_pairs_checked": len(i),
            "frames": count,
        },
    }
    if link["twist_turns"]:
        deviation = (
            np.asarray(link["twist_turns"]) + np.asarray(link["writhe_turns"]) - np.asarray(link["applied_turns"])
        )
        criteria["diagnostics"]["link_max_deviation_turns"] = float(np.max(np.abs(deviation)))
        criteria["diagnostics"]["link_within_tolerance"] = bool(
            np.max(np.abs(deviation)) < gates["link"]["tolerance_turns"]
        )
        longest = float(np.max(np.linalg.norm(np.diff(frames["x"], axis=1), axis=2)))
        criteria["diagnostics"]["link_closure_min_clearance_m"] = float(min(closure))
        criteria["diagnostics"]["link_closure_suspect"] = bool(min(closure) < 0.5 * longest)
        criteria["diagnostics"]["link_closure_note"] = (
            "vertex clearance to the zero-thickness closure lines; below half the longest edge "
            f"({0.5 * longest:.4g} m) a segment may cross a closure line and fake a writhe jump"
        )
    series = {
        "step": frames["step"],
        "penetration_peak": peak,
        "penetration_peak_self": peak_self,
        "penetration_peak_inter": peak_inter,
        "kinetic": kinetic,
        "potential": potential,
        "elastic": elastic,
        "ground": ground,
        "driver_work": work,
        **{key: np.asarray(value) for key, value in link.items() if value},
    }
    return criteria, series


def bind_verdict(criteria, gates_path):
    """Record the gates file and demote a PASS judged by other gates or by a modified judge.

    ``evaluate`` sees only the gates content; a PASS also needs the canonical frozen file
    of this commit and an unmodified evaluator, which only the caller can establish.
    """
    gates_path = Path(gates_path)
    criteria["gates_file"] = str(gates_path)
    criteria["gates_sha256"] = hashlib.sha256(gates_path.read_bytes()).hexdigest()
    criteria["gates_canonical"] = gates_path.resolve() == FROZEN_GATES.resolve()
    if criteria["verdict"] == "PASS":
        if not criteria["gates_canonical"]:
            criteria["verdict"] = "NON_CANONICAL_GATES"
        elif criteria["evaluator"].get("file_status") != "":
            criteria["verdict"] = "DIRTY_JUDGE"
    return criteria


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--gates", type=Path, default=FROZEN_GATES)
    parser.add_argument("--output", type=Path, help="criteria.json path; defaults inside the run directory")
    args = parser.parse_args(argv)
    if not args.gates.is_file():
        raise SystemExit(f"Gates file {args.gates} does not exist; pass --gates explicitly")
    gates = json.loads(args.gates.read_text())
    criteria, series = evaluate(load_run(args.run), gates)
    bind_verdict(criteria, args.gates)
    output = args.output or args.run / "criteria.json"
    write_json(output, criteria)
    np.savez_compressed(output.with_name(output.stem + "-series.npz"), **series)
    print(json.dumps({"verdict": criteria["verdict"], "gates": criteria["gates"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
