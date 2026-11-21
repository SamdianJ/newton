# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FP64 offline criteria for floating-hand runs, independent of the solver's contact code.

Every collision shape is a convex core plus a radius: a sphere is a point, a
capsule a segment, a box or convex hull a polytope with radius zero. For convex
sets the translation (penetration) depth is exact:

    depth(A, B) = max(0, r_A + r_B - sd(core_A, core_B)),

where ``sd`` is the core distance (GJK) when the cores are disjoint and minus
their SAT overlap when they intersect. Box pairs reproduce the rigid-stack SAT
depth (``run_rigid_stack.box_penetration``).
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

import newton

_TOL = 1e-13


@dataclass(frozen=True)
class Convex:
    """Shape-local convex core: vertices, face normals, edge directions and an inflation radius."""

    vertices: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    radius: float = 0.0

    def placed(self, rotation, translation):
        """Return the core in world: rotation is 3x3, translation a 3-vector."""
        rotation = np.asarray(rotation, dtype=np.float64)
        return Convex(
            self.vertices @ rotation.T + np.asarray(translation, dtype=np.float64),
            self.normals @ rotation.T,
            self.edges @ rotation.T,
            self.radius,
        )


_EMPTY = np.zeros((0, 3))


def point(radius=0.0):
    return Convex(np.zeros((1, 3)), _EMPTY, _EMPTY, float(radius))


def segment(half_height, radius=0.0):
    """Capsule core along local z, Newton's convention (scale = radius, half_height)."""
    ends = np.array([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]])
    return Convex(ends, _EMPTY, np.array([[0.0, 0.0, 1.0]]), float(radius))


def box(half_extents):
    h = np.asarray(half_extents, dtype=np.float64)
    if h.shape != (3,) or np.any(h <= 0):
        raise ValueError(f"Box half extents must be three positive values, got {half_extents}")
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64) * h
    return Convex(corners, np.eye(3), np.eye(3))


def hull(points):
    """Convex hull of a point cloud: unique face normals and unique edge directions."""
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=np.float64)
    h = ConvexHull(points)
    vertices = points[h.vertices]
    normals = _unique_directions(h.equations[:, :3])
    edges = []
    for simplex in h.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.append(points[simplex[b]] - points[simplex[a]])
    edges = np.asarray(edges)
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    return Convex(vertices, normals, _unique_directions(edges))


def _unique_directions(directions, tol=1e-9):
    """Keep one of each pair of parallel or anti-parallel unit directions."""
    kept = []
    for direction in directions:
        unit = direction / np.linalg.norm(direction)
        if not any(abs(abs(float(unit @ k)) - 1.0) < tol for k in kept):
            kept.append(unit)
    return np.asarray(kept) if kept else _EMPTY


def _closest_on_simplex(simplex):
    """Point of the simplex's convex hull nearest the origin, and the vertices that support it."""
    best, best_index = None, None
    n = len(simplex)
    for size in range(1, n + 1):
        for index in combinations(range(n), size):
            q = simplex[list(index)]
            if size == 1:
                x, weights = q[0], np.ones(1)
            else:
                gram = q @ q.T
                kkt = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
                rhs = np.r_[np.zeros(size), 1.0]
                solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
                weights = solution[:size]
                if np.any(weights < -1e-12):
                    continue
                x = weights @ q
            if best is None or x @ x < best @ best - 1e-30:
                best, best_index = x, index
    return best, best_index


def core_distance(a, b, iterations=200):
    """GJK distance between the convex hulls of two vertex sets (0 when they intersect)."""
    support = lambda d: a.vertices[np.argmax(a.vertices @ d)] - b.vertices[np.argmin(b.vertices @ d)]  # noqa: E731
    simplex = np.array([a.vertices[0] - b.vertices[0]])
    scale = max(1.0, float(np.abs(a.vertices).max(initial=0)), float(np.abs(b.vertices).max(initial=0)))
    previous = np.inf
    for _ in range(iterations):
        x, index = _closest_on_simplex(simplex)
        distance = float(np.linalg.norm(x))
        if distance <= _TOL * scale:
            return 0.0
        s = support(-x)
        kept = simplex[list(index)]
        # Stop when the support adds no progress, repeats a simplex vertex, or the distance stalls.
        if (
            distance * distance - float(x @ s) <= 1e-12 * distance * scale
            or np.any(np.all(np.abs(kept - s) <= 1e-15 * scale, axis=1))
            or distance >= previous * (1 - 1e-14)
        ):
            return distance
        previous = distance
        simplex = np.vstack((kept, s))
    raise RuntimeError(f"GJK did not converge in {iterations} iterations (distance {distance})")


def core_overlap(a, b):
    """SAT overlap of two intersecting polytopes over face normals and edge-edge crosses (0 if none)."""
    axes = [a.normals, b.normals]
    if len(a.edges) and len(b.edges):
        crosses = np.cross(a.edges[:, None, :], b.edges[None, :, :]).reshape(-1, 3)
        lengths = np.linalg.norm(crosses, axis=1)
        axes.append(crosses[lengths > 1e-9] / lengths[lengths > 1e-9, None])
    axes = np.concatenate(axes)
    if not len(axes):
        return 0.0
    pa, pb = a.vertices @ axes.T, b.vertices @ axes.T
    overlap = np.minimum(pa.max(axis=0) - pb.min(axis=0), pb.max(axis=0) - pa.min(axis=0))
    return max(0.0, float(overlap.min()))


def signed_distance(a, b):
    """Signed distance of the inflated shapes: positive gap, negative penetration depth."""
    d = core_distance(a, b)
    core = d if d > 0 else -core_overlap(a, b)
    return core - a.radius - b.radius


def penetration(a, b):
    return max(0.0, -signed_distance(a, b))


def plane_signed_distance(a, normal=(0.0, 0.0, 1.0), offset=0.0):
    """Signed distance of an inflated core to the half-space ``n.x >= offset``."""
    normal = np.asarray(normal, dtype=np.float64)
    return float((a.vertices @ normal).min() - offset - a.radius)


def model_core(model, shape, *, hull_points=None):
    """Body-local convex core of one Model shape; MESH and CONVEX_MESH use the hull of their vertices."""
    geo = int(model.shape_type.numpy()[shape])
    scale = model.shape_scale.numpy()[shape].astype(np.float64)
    if geo == newton.GeoType.SPHERE:
        core = point(scale[0])
    elif geo == newton.GeoType.CAPSULE:
        core = segment(scale[1], scale[0])
    elif geo == newton.GeoType.BOX:
        core = box(scale)
    elif geo in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        source = model.shape_source[shape] if hull_points is None else None
        points = hull_points if hull_points is not None else np.asarray(source.vertices, dtype=np.float64) * scale
        core = hull(points)
    else:
        raise ValueError(f"Shape {shape}: no convex core for {newton.GeoType(geo).name}")
    x = model.shape_transform.numpy()[shape].astype(np.float64)
    return core.placed(quaternion_matrix(x[3:]), x[:3])


def quaternion_matrix(q):
    """Rotation matrix of a unit quaternion stored xyzw (Warp order)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_core(core, body_pose):
    """Place a body-local core with a body pose ``(px, py, pz, qx, qy, qz, qw)``."""
    pose = np.asarray(body_pose, dtype=np.float64)
    return core.placed(quaternion_matrix(pose[3:]), pose[:3])


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------


def _core_from_record(shape):
    geo = shape["type"]
    scale = np.asarray(shape["scale"], dtype=np.float64)
    if geo == "SPHERE":
        core = point(scale[0])
    elif geo == "CAPSULE":
        core = segment(scale[1], scale[0])
    elif geo == "BOX":
        core = box(scale)
    elif geo in ("CONVEX_MESH", "MESH"):
        core = hull(np.asarray(shape["hull_points"], dtype=np.float64))
    elif geo == "PLANE":
        return None
    else:
        raise ValueError(f"Shape {shape['label']}: no convex core for {geo}")
    x = np.asarray(shape["transform"], dtype=np.float64)
    return core.placed(quaternion_matrix(x[3:]), x[:3])


def load_run(run_dir):
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    model = json.loads((run_dir / "model.json").read_text())
    chunks = [np.load(run_dir / name) for name in manifest.get("chunks", [])]
    frames = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0].files} if chunks else {}
    return manifest, model, frames


def _bounding_radius(core):
    return float(np.linalg.norm(core.vertices - core.vertices.mean(axis=0), axis=1).max()) + core.radius


def _rotation_angle(a, b):
    """Geodesic angle between two rotation matrices."""
    c = (np.trace(a @ b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def evaluate(run_dir, recipe, *, settle_s=0.05, grasp_window=500, stride=1):
    """Measure every published frame of a run; returns the criteria dictionary (no PASS/FAIL)."""
    manifest, model, frames = load_run(run_dir)
    dt = manifest["dt_s"]
    out = {
        "source_sha": manifest["source_sha"],
        "status": manifest["status"],
        "counts": manifest["counts"],
        "planned_steps": manifest.get("planned_steps"),
        "published_steps": int(len(frames.get("step", ()))),
    }
    published = out["published_steps"]
    out["stopped_fraction"] = manifest["counts"]["STOPPED_ACCEPTED"] / max(1, published)
    if not published:
        return out
    shapes = {s["index"]: s for s in model["shapes"]}
    cores = {i: _core_from_record(s) for i, s in shapes.items()}
    radii = {i: _bounding_radius(c) for i, c in cores.items() if c is not None}
    body_q = frames["body_q"].astype(np.float64)
    obj = model["object_body"]
    hand = set(model["hand_bodies"])
    plane = [i for i, s in shapes.items() if s["type"] == "PLANE"]
    labels = model["body_labels"]
    exclude = {tuple(sorted(p)) for p in (recipe.get("self_collision") or {}).get("exclude", [])}
    self_pairs = [
        (a, b)
        for a, b in model["same_tree_pairs"]
        if tuple(sorted((labels[shapes[a]["body"]], labels[shapes[b]["body"]]))) not in exclude
    ]

    def placed(i, k):
        return world_core(cores[i], body_q[k, shapes[i]["body"]])

    def depth(a, b, k):
        ca, cb = placed(a, k), placed(b, k)
        center = np.linalg.norm(ca.vertices.mean(axis=0) - cb.vertices.mean(axis=0))
        if center > radii[a] + radii[b]:
            return 0.0
        return penetration(ca, cb)

    hand_shapes = [i for i, s in shapes.items() if s["body"] in hand and cores[i] is not None]
    object_shapes = [i for i, s in shapes.items() if s["body"] == obj and cores[i] is not None]
    dynamic_shapes = [i for i, s in shapes.items() if s["body"] >= 0 and cores[i] is not None]

    def table_depth(group, k):
        return max(max(0.0, -plane_signed_distance(placed(i, k))) for i in group)

    # Each category is measured only when both sides exist; otherwise it is NOT_APPLICABLE, never 0.
    categories = {
        "hand_object": (
            hand_shapes and object_shapes,
            lambda k: max(depth(a, b, k) for a in hand_shapes for b in object_shapes),
        ),
        "object_table": (object_shapes and plane, lambda k: table_depth(object_shapes, k)),
        "hand_table": (hand_shapes and plane, lambda k: table_depth(hand_shapes, k)),
        "dynamic_table": (dynamic_shapes and plane, lambda k: table_depth(dynamic_shapes, k)),
        "self": (self_pairs, lambda k: max(depth(a, b, k) for a, b in self_pairs)),
    }
    frames_checked = list(range(0, published, stride))
    out["penetration_m"] = {}
    for name, (applicable, measure) in categories.items():
        if not applicable:
            out["penetration_m"][name] = "NOT_APPLICABLE"
            continue
        values = [measure(k) for k in frames_checked]
        out["penetration_m"][name] = {"max": float(np.max(values)), "p99": float(np.quantile(values, 0.99))}
    out["frames_checked"] = len(frames_checked)
    out["self_pairs_checked"] = len(self_pairs)
    if model["palm_body"] is not None and np.any(frames["palm_target"][:, 3, 3]):
        first = int(np.ceil(settle_s / dt))
        palm = body_q[:, model["palm_body"]]
        pos_err, ang_err = [], []
        for k in range(first, published):
            target = frames["palm_target"][k]
            pos_err.append(float(np.linalg.norm(palm[k, :3] - target[:3, 3])))
            ang_err.append(_rotation_angle(quaternion_matrix(palm[k, 3:]), target[:3, :3]))
        out["root_tracking"] = {
            "position_max_m": max(pos_err, default=0.0),
            "angle_max_rad": max(ang_err, default=0.0),
            "from_step": first,
            "from_s": first * dt,
        }
    if obj is not None:
        z0 = body_q[0, obj, 2]
        weight = model["body_mass"][obj] * abs(model["gravity"][2])
        o = recipe["object"]
        edge = 2 * min(o["half_extents"]) if o["kind"] == "box" else 2 * o["radius"]
        table = [i for i, idx in enumerate(manifest.get("object_pair_shapes", [])) if idx in plane]
        support = frames["object_pair_force"][:, table, 2].sum(axis=1) if table else np.zeros(published)
        window = slice(max(0, published - grasp_window), published)
        rise = body_q[:, obj, 2] - z0
        out["grasp"] = {
            "window_steps": int(min(grasp_window, published)),
            # Object-table pairs that feed the support force; none would read as zero support.
            "table_pairs": len(table),
            "window_complete": published == manifest.get("planned_steps") and published >= grasp_window,
            "weight_N": weight,
            "max_ground_force_N": float(np.max(np.abs(support[window]))),
            "min_com_rise_m": float(np.min(rise[window])),
            "object_min_edge_m": edge,
            "lift_ratio_min": float(np.min(rise[window]) / recipe["lift_m"]) if "lift_m" in recipe else None,
        }
    if recipe["scene"] == "E2":
        out["chain"] = chain_balance(model, frames, recipe, dt)
    if recipe["scene"] == "E3":
        # Root rows: the first six DoFs of the hand tree (FREE root or the six wrist joints).
        root = slice(model["joint_qd_start"][0], model["joint_qd_start"][0] + 6)
        values = frames["contact_joint_force"][:, root]
        out["root_rows_contact_force"] = {
            "max_abs": float(np.max(np.abs(values))),
            "all_zero": bool(np.all(values == 0)),
        }
    out["solver"] = {
        name: {"mean": float(np.mean(frames[name])), "max": int(np.max(frames[name]))}
        for name in ("newton_iterations", "linear_iterations", "line_search_iterations", "regularization_retries")
    }
    return out


def _penalty_energy(gap, stiffness, smoothing):
    """FP64 copy of the C-IE normal penalty energy (contact.py _penalty64) for gaps in metres."""
    u = -np.asarray(gap, dtype=np.float64)
    p = np.where(u >= smoothing, u, 0.0)
    if smoothing > 0:
        t = (u + smoothing) / (2.0 * smoothing)
        mid = (u > -smoothing) & (u < smoothing)
        p = np.where(mid, smoothing * t**3 * (2.0 - t), p)
    else:
        p = np.maximum(u, 0.0)
    return stiffness * p * p / 2.0


def chain_balance(model, frames, recipe, dt):
    """E2: free-flight linear momentum and the largest energy gain over any window (K14)."""
    gravity = np.asarray(model["gravity"], dtype=np.float64)
    mass = np.asarray(model["body_mass"], dtype=np.float64)
    com = np.asarray(model["body_com"], dtype=np.float64)
    inertia = np.asarray(model["body_inertia"], dtype=np.float64)
    q, qd = frames["body_q"].astype(np.float64), frames["body_qd"].astype(np.float64)
    steps = frames["step"].astype(np.float64)
    n = len(steps)
    rotations = np.array([[quaternion_matrix(q[k, b, 3:]) for b in range(len(mass))] for k in range(n)])
    x_com = q[:, :, :3] + np.einsum("kbij,bj->kbi", rotations, com)
    momentum = np.einsum("b,kbi->ki", mass, qd[:, :, :3])
    flight = round(recipe["phases_s"]["flight"] / dt)
    touching = np.flatnonzero(np.abs(frames["contact_body_wrench"]).sum(axis=(1, 2)) > 0)
    free_end = min(flight, int(touching[0]) if len(touching) else n, n)
    elapsed = (steps[:free_end] - steps[0]) * dt
    impulse = mass.sum() * gravity[None, :] * elapsed[:, None]
    drift = np.linalg.norm(momentum[:free_end] - momentum[0] - impulse, axis=1)
    reference = np.maximum(mass.sum() * np.linalg.norm(gravity) * elapsed, np.finfo(float).tiny)
    omega = qd[:, :, 3:]
    world_inertia = np.einsum("kbij,bjl,kbml->kbim", rotations, inertia, rotations)
    kinetic = 0.5 * np.einsum("b,kbi,kbi->k", mass, qd[:, :, :3], qd[:, :, :3]) + 0.5 * np.einsum(
        "kbi,kbij,kbj->k", omega, world_inertia, omega
    )
    potential = -np.einsum("b,kbi,i->k", mass, x_com, gravity)
    spring_ke = recipe["chain"]["spring_ke"]
    labels, types, starts = model["joint_labels"], model["joint_type"], model["joint_q_start"]
    scalar = [starts[j] for j, t in enumerate(types) if labels[j].startswith("chain_joint")]
    spring = 0.5 * spring_ke * (frames["joint_q"][:, scalar].astype(np.float64) ** 2).sum(axis=1)
    contact, contact_note = np.zeros(n), "C-IE normal penalty on the solver's P1Q3 samples, FP64"
    try:
        from newton._src.solvers.monolithic.contact_sampling import _surface_samples  # noqa: PLC0415

        solver = recipe["solver"]
        for shape in model["shapes"]:
            if shape["body"] < 0:
                continue
            geo = getattr(newton.GeoType, shape["type"])
            points, _, weights = _surface_samples(int(geo), np.asarray(shape["scale"], dtype=np.float64))
            x = np.asarray(shape["transform"], dtype=np.float64)
            local = points @ quaternion_matrix(x[3:]).T + x[:3]
            world = np.einsum("kij,pj->kpi", rotations[:, shape["body"]], local) + q[:, shape["body"], None, :3]
            gap = world[:, :, 2] - recipe["table"]["z"]
            contact += (_penalty_energy(gap, solver["contact_stiffness"], solver["normal_smoothing"]) * weights).sum(
                axis=1
            )
    except ValueError as missing:
        contact, contact_note = None, f"NOT_APPLICABLE: {missing}"
    total = kinetic + potential + spring + (contact if contact is not None else 0.0)
    gain = float(np.max(total - np.minimum.accumulate(total)))
    budget = float(kinetic[0] + max(0.0, potential[0] - potential.min()))
    return {
        "free_flight_steps": int(free_end),
        "momentum_drift_ratio_max": float(np.max(drift / reference)) if free_end > 1 else None,
        "energy_gain_max_J": gain if contact is not None else None,
        "energy_budget_J": budget,
        "energy_gain_ratio": gain / budget if contact is not None and budget > 0 else None,
        "contact_energy": contact_note,
        "energy_initial_J": {
            "kinetic": float(kinetic[0]),
            "potential": float(potential[0]),
            "spring": float(spring[0]),
            "contact": float(contact[0]) if contact is not None else None,
        },
    }


def missing_categories(criteria, gates, scene):
    """Required penetration categories without a measured value (NOT_APPLICABLE or absent): each is a FAIL."""
    measured = criteria.get("penetration_m", {})
    return [name for name in gates["required_categories"].get(scene, []) if not isinstance(measured.get(name), dict)]


# Scene gate keys gate_verdict judges; any other key in the frozen file is an error, not a silent pass.
SCENE_GATE_KEYS = frozenset(
    {
        "self_penetration_max_m",
        "root_tracking",
        "grasp",
        "free_flight_momentum_drift_ratio_max",
        "energy_gain_ratio_max",
        "root_rows_contact_force_all_zero",
        "negative_control",
    }
)


def acceptance_problems(manifest, recipe):
    """Why a run is not an acceptance run (gates: FP32 CUDA production graph, the recipe as frozen)."""
    args = manifest.get("arguments", {})
    problems = []
    if args.get("mount") not in (None, recipe.get("mount")):
        problems.append(f"mount {args.get('mount')!r} is not the recipe mount {recipe.get('mount')!r}")
    if args.get("self_collision") != "recipe":
        problems.append(f"self_collision {args.get('self_collision')!r} is not 'recipe'")
    if args.get("contact_factor_capacity") is not None:
        problems.append(f"contact_factor_capacity overridden to {args.get('contact_factor_capacity')}")
    if args.get("precision") != "fp32":
        problems.append(f"precision {args.get('precision')!r} is not fp32 (FP64 runs are diagnostic)")
    if not str(args.get("device", "")).startswith("cuda"):
        problems.append(f"device {args.get('device')!r} is not CUDA")
    if not args.get("use_cuda_graph"):
        problems.append("the production CUDA graph is off")
    if manifest.get("git_status"):
        problems.append("the worktree had uncommitted changes")
    return problems


def negative_problems(manifest, negative):
    """Why a run is not this run's E3 negative control (same recipe and build, self-collision off)."""
    a, b = manifest.get("arguments", {}), negative.get("arguments", {})
    problems = []
    if b.get("self_collision") != "off":
        problems.append(f"negative run self_collision {b.get('self_collision')!r} is not 'off'")
    if negative.get("status") != "COMPLETE":
        problems.append(f"negative run status {negative.get('status')!r} is not COMPLETE")
    for key in ("recipe_sha256", "source_sha"):
        if negative.get(key) != manifest.get(key):
            problems.append(f"negative run {key} differs")
    for key in ("mount", "precision", "device", "use_cuda_graph", "contact_factor_capacity"):
        if b.get(key) != a.get(key):
            problems.append(f"negative run {key} {b.get(key)!r} differs from {a.get(key)!r}")
    return problems


def gate_verdict(criteria, gates, scene, *, negative=None, negative_issues=(), configuration_issues=()):
    """Judge one run's criteria against the frozen gates: every check PASS, FAIL or NOT_RUN.

    A frozen hard gate whose value or limit is missing or None is a FAIL, never skipped; required
    penetration categories go through ``missing_categories``. ``negative`` is the E3
    self-collision-off run's criteria (the negative control), NOT_RUN when not given and FAIL when
    ``negative_issues`` (from ``negative_problems``) is not empty. The verdict is FAIL if any check
    fails, NOT_RUN if any is not run, DIAGNOSTIC if every gate passes but ``configuration_issues``
    (from ``acceptance_problems``) says the run is not an acceptance run, else PASS. The performance
    budgets of the gates file are judged by the performance protocol, not here.
    """
    if scene not in gates["scenes"]:
        raise ValueError(f"No frozen gates for scene {scene!r}")
    common, own = gates["common"], gates["scenes"][scene]
    unknown = set(own) - SCENE_GATE_KEYS
    if unknown:
        raise ValueError(f"Frozen gates for {scene} have keys gate_verdict does not judge: {sorted(unknown)}")
    if negative is not None and "negative_control" not in own:
        raise ValueError(f"Scene {scene} has no negative control; do not pass one")
    checks = []

    def check(name, value, limit, op):
        # A missing value or limit (e.g. no object weight) cannot pass.
        if value is None or limit is None:
            result = "FAIL"
        elif op == "<=":
            result = "PASS" if value <= limit else "FAIL"
        elif op == ">=":
            result = "PASS" if value >= limit else "FAIL"
        elif op == ">":
            result = "PASS" if value > limit else "FAIL"
        elif op == "==":
            result = "PASS" if value == limit else "FAIL"
        else:
            raise ValueError(f"Unknown gate comparison {op!r}")
        checks.append({"name": name, "value": value, "limit": limit, "op": op, "result": result})

    def get(*path):
        value = criteria
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    published = criteria.get("published_steps")
    check("run_complete", criteria.get("status"), "COMPLETE", "==")
    check("published_steps", published, criteria.get("planned_steps"), "==")
    check("stride", criteria.get("stride"), 1, "==")
    check("frames_checked", criteria.get("frames_checked"), published, "==")
    check("hard_failures", get("counts", "HARD_FAILURE"), common["hard_failures_max"], "<=")
    check("stopped_fraction", criteria.get("stopped_fraction"), common["stopped_fraction_max"], "<=")
    missing = missing_categories(criteria, gates, scene)
    for name in missing:
        limit = own["self_penetration_max_m"] if name == "self" else common["penetration_max_m"].get(name)
        check(f"penetration.{name}", None, limit, "<=")
    for name, limit in common["penetration_max_m"].items():
        measured = get("penetration_m", name)
        if isinstance(measured, dict):
            check(f"penetration.{name}", measured.get("max"), limit, "<=")
    if "self_penetration_max_m" in own and "self" not in missing:
        self_value = get("penetration_m", "self")
        check(
            "penetration.self",
            self_value.get("max") if isinstance(self_value, dict) else None,
            own["self_penetration_max_m"],
            "<=",
        )
    if "root_tracking" in own:
        tracking = own["root_tracking"]
        from_s = get("root_tracking", "from_s")
        check("root_tracking.from_s", None if from_s is None else round(from_s, 9), round(tracking["from_s"], 9), "==")
        check("root_tracking.position_m", get("root_tracking", "position_max_m"), tracking["position_max_m"], "<=")
        check("root_tracking.angle_rad", get("root_tracking", "angle_max_rad"), tracking["angle_max_rad"], "<=")
    if "grasp" in own:
        grasp = own["grasp"]
        check("grasp.window_complete", get("grasp", "window_complete"), True, "==")
        check("grasp.window_steps", get("grasp", "window_steps"), grasp["window_steps"], "==")
        check("grasp.table_pairs", get("grasp", "table_pairs"), 1, ">=")
        weight = get("grasp", "weight_N")
        check(
            "grasp.ground_force_N",
            get("grasp", "max_ground_force_N"),
            None if weight is None else grasp["ground_weight_fraction_max"] * weight,
            "<=",
        )
        edge = get("grasp", "object_min_edge_m")
        check(
            "grasp.min_com_rise_m",
            get("grasp", "min_com_rise_m"),
            None if edge is None else grasp["com_rise_min_edge_fraction"] * edge,
            ">=",
        )
        check("grasp.lift_ratio", get("grasp", "lift_ratio_min"), grasp["lift_ratio_min"], ">=")
    if "free_flight_momentum_drift_ratio_max" in own:
        check(
            "chain.momentum_drift_ratio",
            get("chain", "momentum_drift_ratio_max"),
            own["free_flight_momentum_drift_ratio_max"],
            "<=",
        )
    if "energy_gain_ratio_max" in own:
        check("chain.energy_gain_ratio", get("chain", "energy_gain_ratio"), own["energy_gain_ratio_max"], "<=")
    if own.get("root_rows_contact_force_all_zero"):
        check("root_rows_contact_force_all_zero", get("root_rows_contact_force", "all_zero"), True, "==")
    if "negative_control" in own:
        limit = own["self_penetration_max_m"]
        if negative is None:
            checks.append(
                {"name": "negative_control.self", "value": None, "limit": limit, "op": ">", "result": "NOT_RUN"}
            )
        else:
            value = negative.get("penetration_m", {}).get("self")
            check("negative_control.self", value.get("max") if isinstance(value, dict) else None, limit, ">")
            check("negative_control.provenance", list(negative_issues), [], "==")
    results = {item["result"] for item in checks}
    verdict = "FAIL" if "FAIL" in results else "NOT_RUN" if "NOT_RUN" in results else "PASS"
    if verdict == "PASS" and configuration_issues:
        verdict = "DIAGNOSTIC"
    return {
        "scene": scene,
        "verdict": verdict,
        "checks": checks,
        "configuration_issues": list(configuration_issues),
        "performance": "judged by the performance protocol, not by gate_verdict",
    }


def main(argv=None):
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Evaluate one run_floating_hand output directory in FP64.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stride", type=int, default=1, help="evaluate every n-th frame (diagnostic runs only)")
    parser.add_argument("--gates", type=Path, help="frozen gates JSON: also write verdict.json (PASS/FAIL/NOT_RUN)")
    parser.add_argument("--negative", type=Path, help="E3: the self-collision-off run directory (negative control)")
    args = parser.parse_args(argv)
    import hashlib  # noqa: PLC0415

    from scripts.monolithic_reference.floating_hand_scenes import load_recipe  # noqa: PLC0415

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    recipe_path = Path(manifest["arguments"]["recipe"])
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    if digest != manifest["recipe_sha256"]:
        raise ValueError(f"{recipe_path} changed since the run (sha256 {digest}, run used {manifest['recipe_sha256']})")
    recipe = load_recipe(recipe_path)
    gates, settle = None, {}
    if args.gates is not None:
        gates = json.loads(args.gates.read_text())
        if gates.get("status") != "FROZEN":
            raise ValueError(f"{args.gates}: gates must be FROZEN, got {gates.get('status')!r}")
        tracking = gates["scenes"].get(recipe["scene"], {}).get("root_tracking")
        if tracking is not None:
            settle = {"settle_s": tracking["from_s"]}
    elif args.negative is not None:
        raise ValueError("--negative needs --gates")
    result = evaluate(args.run_dir, recipe, stride=args.stride, **settle)
    result["stride"] = args.stride
    (args.run_dir / "criteria.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if gates is not None:
        negative, issues = None, ()
        if args.negative is not None:
            negative = json.loads((args.negative / "criteria.json").read_text())
            issues = negative_problems(manifest, json.loads((args.negative / "manifest.json").read_text()))
        verdict = gate_verdict(
            result,
            gates,
            recipe["scene"],
            negative=negative,
            negative_issues=issues,
            configuration_issues=acceptance_problems(manifest, recipe),
        )
        run_args = manifest.get("arguments", {})
        verdict["run"] = {
            "source_sha": manifest.get("source_sha"),
            "git_status": manifest.get("git_status"),
            "recipe_sha256": manifest.get("recipe_sha256"),
            **{
                k: run_args.get(k)
                for k in ("mount", "self_collision", "contact_factor_capacity", "precision", "device", "use_cuda_graph")
            },
        }
        verdict["gates_sha256"] = hashlib.sha256(args.gates.read_bytes()).hexdigest()
        (args.run_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps({"verdict": verdict["verdict"]}))
        return 0 if verdict["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
