# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Judge a recorded cable-contact run offline in FP64: penetration, crossing, link, energy and rest.

Independent of the solver's contact code: penetration is the exact segment-segment
distance between every non-excluded segment pair of every published frame. Gate
values come from a gates file; a run judged against a draft file is never PASS.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .cable_oracle import CableReference, _reference_twist, _weighted_strains
from .cloth_contact_run import write_json

FIXTURES = Path(__file__).with_name("fixtures")
FROZEN_GATES = FIXTURES / "cable_contact_gates_frozen_v1.json"
GAUSS = ((1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2)
RING = 64


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
    for key in ("target_theta",):
        frames[key] = []
    for name in summary.get("chunks", []):
        with np.load(directory / name) as data:
            for key in (*keys, "step"):
                frames[key].append(np.asarray(data[key], dtype=float if key != "step" else np.int64))
            frames["target_theta"].append(np.asarray(data["target_theta"], dtype=float))
    series = {key: np.concatenate(value) for key, value in frames.items() if value}
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


def ground_energy(x, theta, director, radius, rest_lengths, height, stiffness, smoothing):
    """Penalty energy of one cable's surface samples against the plane z = height [J].

    Uses the solver's quadrature points and areas but no orientation weight; for a
    ring resting on a plane only downward samples overlap, so this bounds the solver value.
    """
    z, area = surface_samples(x, theta, director, radius, rest_lengths)
    return float(np.sum(area * penalty(z - height, stiffness, smoothing)))


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
    previous_forces = None
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
        local_fixed = [v - a for v in range(a, b) if v in fixed]
        clamped = [k for k in range(b - a - 1) if np.ptp(frames["target_theta"][:, e0 + k]) > 0] if count > 1 else []
        previous_forces = None
        for k in range(count):
            x = frames["x"][k, a:b]
            theta = frames["theta"][k, e0:e1]
            frame = _frame(x, frames["director"][k, e0:e1], frames["twist"][k, w0:w1])
            elastic[k] += elastic_energy(rest, x, theta, frame)
            if np.isfinite(height):
                z, area = surface_samples(x, theta, frames["director"][k, e0:e1], topo["radius"][cable], rest.lengths)
                gap = z - height
                ground[k] += float(np.sum(area * penalty(gap, solver["contact_stiffness"], solver["normal_smoothing"])))
                ground_depth[k] = max(ground_depth[k], float(-gap.min()))
            if local_fixed or clamped:
                forces = elastic_gradient(rest, x, theta, frame, local_fixed, clamped)
                if previous_forces is not None:
                    dx = x[local_fixed] - frames["x"][k - 1, a:b][local_fixed]
                    dtheta = theta[clamped] - frames["theta"][k - 1, e0:e1][clamped]
                    work[k] += 0.5 * np.sum((previous_forces[0] + forces[0]) * dx)
                    work[k] += 0.5 * np.sum((previous_forces[1] + forces[1]) * dtheta)
                previous_forces = forces
            if kind == "plectoneme":
                twist = total_twist(theta, frames["twist"][k, w0:w1])
                link["twist_turns"].append(twist)
                link["writhe_turns"].append(closed_writhe(x, gates["link"]["closure_m"]))
                target = frames["target_theta"][k - 1] if k else np.zeros(e1)
                applied = (target[e1 - 1] - target[e0]) / (2 * np.pi) if k else 0.0
                link["applied_turns"].append(applied)
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
    verdict = "PASS" if complete and all(g["pass"] for g in gate_results.values()) else "FAIL"
    if not complete:
        verdict = "PARTIAL" if status == "PARTIAL" else "FAIL"
    if draft and verdict == "PASS":
        verdict = "PASS_ON_DRAFT_GATES"
    stopped = counts["STOPPED_ACCEPTED"]
    criteria = {
        "scene": scene,
        "run_status": status,
        "source_sha": run.manifest["source_sha"],
        "precision": run.manifest["arguments"]["precision"],
        "device": run.manifest["arguments"]["device"],
        "method": run.manifest["arguments"].get("method", "off"),
        "gates_status": gates.get("status"),
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
            "crossing_first": crossings[:20],
            "energy_max_rise_J": rise,
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
    criteria["gates_file"] = str(args.gates)
    output = args.output or args.run / "criteria.json"
    write_json(output, criteria)
    np.savez_compressed(output.with_name(output.stem + "-series.npz"), **series)
    print(json.dumps({"verdict": criteria["verdict"], "gates": criteria["gates"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
