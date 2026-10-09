# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Enumerate all supported contact samples from Model input, independently.

No production sampling, contact, broad phase, capacity or support map is read.
Native states and actor row descriptions are explicit inputs. Mesh targets
require an independently exported SDF field; CPU texture handles are not used.
"""

import hashlib

import numpy as np

from .actor_contact_oracle import box_face_samples
from .actor_contract_oracle import rotation, sphere_samples
from .articulation_oracle import skew, transform
from .contact_oracle import geometry, local
from .smooth_sdf_oracle import NaturalSpline


def _check_native(state, start):
    if any(not np.isfinite(value).all() for physical in (state, start) for value in physical.values()):
        raise ValueError("Full oracle requires finite native states")


def export_mesh_fields(model, sdf):
    """Export immutable C2 coefficients as independent natural-spline node input."""
    table = model._texture_sdf_data.numpy()
    indices, kinds, scales, flags = (
        getattr(model, name).numpy() for name in ("_shape_sdf_index", "shape_type", "shape_scale", "shape_flags")
    )
    result, prepared = {}, {}
    for shape in np.flatnonzero(kinds == 8):
        index = int(indices[shape])
        if index < -1 or index >= len(sdf.fields) or index >= len(table):
            raise ValueError("Invalid full-oracle mesh SDF index")
        if index == -1:
            if flags[shape] & 4:
                raise ValueError("Full-oracle particle collider requires a mesh SDF")
            continue
        field = sdf.fields[index]
        if field.representation != "c2-bspline":
            raise ValueError("Full mesh oracle requires the supported C2 field")
        dims = np.asarray(field.identity["dims"])
        nodes = field.data.coefficients.numpy().reshape(tuple(dims[::-1] + 2))
        # Tensor-product cubic B-spline values at knots: (c[-1]+4c[0]+c[1])/6.
        for axis in range(3):
            values = np.moveaxis(nodes, axis, 0)
            nodes = np.moveaxis((values[:-2] + 4 * values[1:-1] + values[2:]) / 6, 0, axis)
        unit = 1.0 if table["scale_baked"][index] else float(scales[shape, 0])
        if unit <= 0 or (not table["scale_baked"][index] and np.any(scales[shape] != unit)):
            raise ValueError("Unbaked full-oracle mesh scale must be positive and uniform")
        values = nodes * unit
        lower, upper = np.asarray(field.identity["lower"]) * unit, np.asarray(field.identity["upper"]) * unit
        # Share only identical physical node input, not a production query/cache identity.
        key = (values.shape, hashlib.sha256(values.tobytes()).digest(), tuple(lower), tuple(upper))
        if key not in prepared:
            prepared[key] = NaturalSpline(values, lower, upper)
        natural = prepared[key]
        result[int(shape)] = {
            "smooth_grid": natural.values,
            "lower": natural.lower,
            "upper": natural.upper,
            "representation": "c2-bspline",
            "identity": dict(field.identity),
            "natural_spline": natural,
        }
    return result


def _frames(model, actors, state):
    body_com = model.body_com.numpy().astype(float)
    poses = np.repeat(np.eye(4)[None], model.body_count, axis=0)
    parents, children, kinds, qs, vs = (
        getattr(model, "joint_" + name).numpy() for name in ("parent", "child", "type", "q_start", "qd_start")
    )
    xp, xc, axes = (getattr(model, "joint_" + name).numpy() for name in ("X_p", "X_c", "axis"))
    arm = next(a for a in actors.actors if a.kind == "articulation")
    columns = {d: arm.dof_start + i for i, d in enumerate(arm.dof_indices)}
    screws, ancestors = {}, {}
    for joint in arm.joint_indices:
        parent, child = int(parents[joint]), int(children[joint])
        frame = (np.eye(4) if parent < 0 else poses[parent]) @ transform(xp[joint])
        motion = np.eye(4)
        chain = [] if parent < 0 else list(ancestors[parent])
        if kinds[joint] != 3:
            if kinds[joint] not in (0, 1) or vs[joint + 1] - vs[joint] != 1:
                raise ValueError("Full oracle requires scalar prismatic/revolute or fixed A joints")
            axis = axes[vs[joint]].astype(float)
            axis /= np.linalg.norm(axis)
            if kinds[joint] == 0:
                motion[:3, 3] = axis * state["q"][qs[joint]]
            else:
                motion[:3, :3] = rotation(axis * state["q"][qs[joint]])
            column = columns[int(vs[joint])]
            screws[column] = (int(kinds[joint]), frame[:3, :3] @ axis, frame[:3, 3])
            chain.append(column)
        poses[child] = frame @ motion @ np.linalg.inv(transform(xc[joint]))
        ancestors[child] = chain
    rigid_rows = {}
    for i, actor in enumerate(a for a in actors.actors if a.kind == "rigid"):
        body = actor.body_indices[0]
        r = transform(np.r_[np.zeros(3), state["rotation"][i]])[:3, :3]
        poses[body, :3, :3] = r
        poses[body, :3, 3] = state["com"][i] - r @ body_com[body]
        rigid_rows[body] = actor.dof_start
    centers = np.einsum("bij,bj->bi", poses[:, :3, :3], body_com) + poses[:, :3, 3]

    def jacobian(body, point):
        j = {}
        if body in rigid_rows:
            start = rigid_rows[body]
            values = np.column_stack((np.eye(3), -skew(point - centers[body])))
            j = {start + i: values[:, i] for i in range(6)}
        elif body in ancestors:
            for column in ancestors[body]:
                kind, axis, origin = screws[column]
                j[column] = axis if kind == 0 else np.cross(axis, point - origin)
        return j

    return poses, centers, jacobian, ancestors


def _samples(model, source, kinds, scales):
    if kinds[source] == 7:
        points, weights = box_face_samples(scales[source])
        axis = np.argmax(np.abs(points) / scales[source], axis=1)
        normals = np.zeros_like(points)
        normals[np.arange(len(points)), axis] = np.sign(points[np.arange(len(points)), axis])
        return points, weights, normals
    if kinds[source] == 3:
        points, weights = sphere_samples(float(scales[source, 0]))
        return points, weights, points / np.linalg.norm(points, axis=1)[:, None]
    if kinds[source] != 8:
        raise ValueError("Unsupported full-oracle source shape")
    mesh = model.shape_source[source]
    tri = (np.asarray(mesh.vertices, dtype=float) * scales[source])[np.asarray(mesh.indices).reshape(-1, 3)]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Invalid full-oracle source normals")
    bary = np.eye(3) / 2 + np.ones((3, 3)) / 6
    return (
        np.einsum("qn,fni->fqi", bary, tri).reshape(-1, 3),
        np.repeat(area / 3, 3),
        np.repeat(cross / (2 * area[:, None]), 3, axis=0),
    )


def iter_full(
    model,
    actors,
    state,
    start,
    law,
    *,
    radius,
    stiffness=1e7,
    width=1e-4,
    mu=0.5,
    falloff=1e-5,
    mesh_fields=None,
    alignment_policy=None,
):
    """Yield every supported Model sample, with only its local DoF and endpoint contributions.

    Enumeration ignores spatial candidates and production capacity. Contact keys
    are unique by face/slot/target and directed, deduplicated Model shape pair.
    No full contact-key set or global dense Jacobian/matrix is retained.
    None retains the raw mathematical wrapper; native audits must explicitly
    pass their guard/mask physical policy. Masking retains all enumerated keys.
    """
    _check_native(state, start)
    if alignment_policy not in (None, "guard", "mask"):
        raise ValueError("Unsupported full-oracle alignment policy")
    poses, centers, jacobian, ancestors = _frames(model, actors, state)
    old_poses, _, _, _ = _frames(model, actors, start)
    bodies, kinds, scales, margins, flags = (
        getattr(model, "shape_" + name).numpy() for name in ("body", "type", "scale", "margin", "flags")
    )
    margins, scales = margins.astype(float), scales.astype(float)
    local_frames = [transform(value) for value in model.shape_transform.numpy()]
    current = [
        (np.eye(4) if body < 0 else poses[body]) @ frame for body, frame in zip(bodies, local_frames, strict=True)
    ]
    initial = [
        (np.eye(4) if body < 0 else old_poses[body]) @ frame for body, frame in zip(bodies, local_frames, strict=True)
    ]
    tet = next(a for a in actors.actors if a.kind == "tet")
    particle_rows = {p: tet.dof_start + 3 * i for i, p in enumerate(tet.dynamic_particle_indices)}
    owner = {body: i for i, actor in enumerate(actors.actors) for body in actor.body_indices}
    rigid_bodies = {a.body_indices[0] for a in actors.actors if a.kind == "rigid"}
    dynamic = [body in rigid_bodies or bool(ancestors.get(int(body), ())) for body in bodies]

    def contribution(key, point, old_point, source_j, weight, offset, normal, old_normal, tri=None, bary=None):
        source, target, _, _ = key
        rt = current[target][:3, :3]
        y = rt.T @ (point - current[target][:3, 3])
        yn = initial[target][:3, :3].T @ (old_point - initial[target][:3, 3])
        kind = {1: "plane", 3: "sphere", 7: "box"}.get(int(kinds[target]))
        if kind is None:
            if kinds[target] != 8 or mesh_fields is None or target not in mesh_fields:
                raise ValueError("Full oracle mesh target requires independently exported SDF input")
            kind = mesh_fields[target]
        gap = float(geometry(y, kind, scales[target])[0] - offset)
        e, r, _, b = local(
            y, yn, kind, scales[target], law, offset=offset, stiffness=stiffness * weight, width=width, mu=mu, a=falloff
        )
        if any(not np.isfinite(value).all() for value in (e, r, b)):
            raise ValueError("Nonfinite full-oracle local contribution")
        masked = False
        if alignment_policy is not None:
            old_rt = initial[target][:3, :3]
            old_geometry = geometry(yn, kind, scales[target])
            gradient = geometry(y, kind, scales[target])[1]
            direction = old_geometry[1] if law == "C-EE" else gradient
            masked = np.dot(rt.T @ normal, direction) > 0
            if alignment_policy == "guard":
                if (old_geometry[0] - offset < 0 and np.dot(old_rt.T @ old_normal, old_geometry[1]) > 0) or (
                    gap < 0 and np.dot(rt.T @ normal, gradient) > 0
                ):
                    raise ValueError("Invalid full-oracle contact alignment")
                masked = False
            if masked:
                e, r, b = np.zeros(2), np.zeros(3), np.zeros((3, 3))
        combined = dict(source_j)
        for column, value in jacobian(int(bodies[target]), point).items():
            combined[column] = combined.get(column, np.zeros(3)) - value
        columns = np.asarray(sorted(combined), dtype=np.int64)
        reduced = rt.T @ np.column_stack([combined[c] for c in columns]) if columns.size else np.empty((3, 0))
        active = np.any(reduced != 0, axis=0)
        columns, reduced = columns[active], reduced[:, active]
        rr, bb = reduced.T @ r, reduced.T @ b @ reduced
        world = rt @ r
        particle_ids, particle_force, body_ids, body_wrench, static_ids, static_wrench = [], [], [], [], [], []
        for shape, force in ((source, -world), (target, world)):
            if shape < 0:
                for particle, coefficient in zip(tri, bary, strict=True):
                    particle_ids.append(int(particle))
                    particle_force.append(coefficient * force)
            elif bodies[shape] >= 0:
                body = int(bodies[shape])
                body_ids.append(body)
                body_wrench.append(np.r_[force, np.cross(point - centers[body], force)])
            else:
                static_ids.append(shape)
                static_wrench.append(np.r_[force, np.cross(point, force)])
        record = {
            "key": key,
            "alignment_masked": bool(masked),
            "energy": e,
            "columns": columns,
            "residual": rr,
            "tangent": bb,
            "point": point,
            "world_residual": world,
            "gap": gap,
            "particle_ids": np.asarray(particle_ids, dtype=np.int64),
            "particle_force": np.asarray(particle_force).reshape(-1, 3),
            "body_ids": np.asarray(body_ids, dtype=np.int64),
            "body_wrench": np.asarray(body_wrench).reshape(-1, 6),
            "static_ids": np.asarray(static_ids, dtype=np.int64),
            "static_wrench": np.asarray(static_wrench).reshape(-1, 6),
        }
        if any(
            not np.isfinite(record[name]).all()
            for name in ("residual", "tangent", "particle_force", "body_wrench", "static_wrench", "gap")
        ):
            raise ValueError("Nonfinite full-oracle transformed contribution")
        return record

    rest = model.particle_q.numpy().astype(float)
    for face, tri in enumerate(model.tri_indices.numpy()):
        area = np.linalg.norm(np.cross(rest[tri[1]] - rest[tri[0]], rest[tri[2]] - rest[tri[0]])) / 2
        normals = []
        for frame in (state, start):
            points = frame["x"][tri]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            length = np.linalg.norm(normal)
            if not np.isfinite(length) or length <= 0:
                raise ValueError("Invalid full-oracle source normals")
            normals.append(normal / length)
        for slot in range(3):
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            j = {}
            for particle, coefficient in zip(tri, bary, strict=True):
                if particle in particle_rows:
                    row = particle_rows[particle]
                    for axis in range(3):
                        j[row + axis] = coefficient * np.eye(3)[:, axis]
            for target in np.flatnonzero(flags & 4):
                yield contribution(
                    (-1, int(target), face, slot),
                    bary @ state["x"][tri],
                    bary @ start["x"][tri],
                    j,
                    area / 3,
                    radius + margins[target],
                    *normals,
                    tri,
                    bary,
                )
    seen = set()
    for pair in model.shape_contact_pairs.numpy():
        a, b = map(int, pair)
        if owner.get(int(bodies[a]), -1) == owner.get(int(bodies[b]), -1) or not (
            bodies[a] in rigid_bodies or bodies[b] in rigid_bodies
        ):
            continue
        identity = tuple(sorted((a, b)))
        if identity in seen:
            raise ValueError("Duplicate effective Model shape pair")
        seen.add(identity)
        for source, target in ((a, b), (b, a)):
            if not dynamic[source]:
                continue
            points, weights, normals = _samples(model, source, kinds, scales)
            for sample, (point, weight, normal) in enumerate(zip(points, weights, normals, strict=True)):
                world = current[source][:3, :3] @ point + current[source][:3, 3]
                old = initial[source][:3, :3] @ point + initial[source][:3, 3]
                yield contribution(
                    (source, target, sample // 3, sample % 3),
                    world,
                    old,
                    jacobian(int(bodies[source]), world),
                    weight * (0.5 if dynamic[target] else 1),
                    margins[source] + margins[target],
                    current[source][:3, :3] @ normal,
                    initial[source][:3, :3] @ normal,
                )


def _totals(model, actors, *, dense=False):
    n = actors.scalar_dof_count
    return {
        "energy": np.zeros(2),
        "residual": np.zeros(n),
        "tangent": np.zeros((n, n)) if dense else {},
        "particle_force": np.zeros((model.particle_count, 3)),
        "body_wrench": np.zeros((model.body_count, 6)),
        "static_wrench": np.zeros((model.shape_count, 6)),
        "joint_force": np.zeros(model.joint_dof_count),
    }


def _accumulate(totals, record, dofs):
    columns, bb = record["columns"], record["tangent"]
    totals["energy"] += record["energy"]
    totals["residual"][columns] += record["residual"]
    tangent = totals["tangent"]
    if isinstance(tangent, np.ndarray):
        tangent[np.ix_(columns, columns)] += bb
    else:
        for i, j in zip(*np.nonzero(bb), strict=True):
            key = (int(columns[i]), int(columns[j]))
            tangent[key] = tangent.get(key, 0.0) + bb[i, j]
    for column, value in zip(columns, record["residual"], strict=True):
        if column in dofs:
            totals["joint_force"][dofs[column]] -= value
    np.add.at(totals["particle_force"], record["particle_ids"], record["particle_force"])
    np.add.at(totals["body_wrench"], record["body_ids"], record["body_wrench"])
    np.add.at(totals["static_wrench"], record["static_ids"], record["static_wrench"])


def evaluate_full(
    model,
    actors,
    state,
    start,
    law,
    *,
    radius,
    stiffness=1e7,
    width=1e-4,
    mu=0.5,
    falloff=1e-5,
    mesh_fields=None,
    alignment_policy=None,
):
    """Return the existing dense small-scene E/r/B, forces and complete per-key records."""
    _check_native(state, start)
    totals = _totals(model, actors, dense=True)
    dofs = {a.dof_start + i: d for a in actors.actors if a.kind != "tet" for i, d in enumerate(a.dof_indices)}
    records, min_gap = {}, np.inf
    for record in iter_full(
        model,
        actors,
        state,
        start,
        law,
        radius=radius,
        stiffness=stiffness,
        width=width,
        mu=mu,
        falloff=falloff,
        mesh_fields=mesh_fields,
        alignment_policy=alignment_policy,
    ):
        key = record["key"]
        if key in records:
            raise ValueError("Duplicate independently enumerated contact identity")
        records[key] = record
        min_gap = min(min_gap, record["gap"])
        _accumulate(totals, record, dofs)
    return {**totals, "records": records, "min_gap": min_gap if records else 0.0}


def _key_set(keys):
    if keys is None:
        return None
    result = set()
    for key in keys:
        if (
            not isinstance(key, (tuple, list, np.ndarray))
            or (isinstance(key, np.ndarray) and key.ndim != 1)
            or len(key) != 4
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in key)
        ):
            raise ValueError("Candidate identity requires four integers")
        identity = tuple(map(int, key))
        if identity in result:
            raise ValueError("Duplicate candidate identity")
        result.add(identity)
    return result


def _errors(reference, selected):
    errors = {}
    for name in reference:
        if name == "tangent":
            keys = sorted(reference[name].keys() | selected[name].keys())
            expected = np.asarray([reference[name].get(k, 0.0) for k in keys])
            got = np.asarray([selected[name].get(k, 0.0) for k in keys])
        else:
            expected, got = reference[name], selected[name]
        if not np.isfinite(expected).all() or not np.isfinite(got).all():
            raise ValueError("Nonfinite full-oracle aggregate")
        delta = got - expected
        errors[name] = {
            "max_abs": float(np.max(np.abs(delta), initial=0)),
            "l2": float(np.linalg.norm(delta)),
            "reference_l2": float(np.linalg.norm(expected)),
        }
        if not np.isfinite(list(errors[name].values())).all():
            raise ValueError("Nonfinite full-oracle error metric")
    return errors


def evaluate_stream(model, actors, state, start, law, *, candidate_keys=None, admitted_keys=None, **options):
    """Audit the full system and optional spatial/capacity subsets without dense B.

    Missing selections mean the complete set and the candidate set, respectively.
    Sparse B includes every numerical coordinate; errors use its full Frobenius
    norm, not sampled matvecs. Results measure omissions and grant no acceptance.
    Memory is O(Model inputs + native state + touched B coordinates + supplied
    selection keys + largest source surface/local support). Cancellation zeros
    may retain a touched coordinate; a dense interaction pattern can still take
    O(n_dof squared) memory. No contact records are retained.
    """
    candidates, admitted = _key_set(candidate_keys), _key_set(admitted_keys)
    if candidates is not None and admitted is not None and not admitted <= candidates:
        raise ValueError("Admitted identities must belong to the candidate set")
    _check_native(state, start)
    totals = {name: _totals(model, actors) for name in ("full", "candidate", "admitted")}
    counts = dict.fromkeys(totals, 0)
    active_counts = dict.fromkeys(totals, 0)
    dofs = {a.dof_start + i: d for a in actors.actors if a.kind != "tet" for i, d in enumerate(a.dof_indices)}
    omissions = {name: {"count": 0, "contributing_count": 0} for name in ("spatial_omission", "capacity_omission")}
    hashes = {name: hashlib.sha256() for name in ("full", *omissions)}
    pairs, min_gap = {}, np.inf
    for record in iter_full(model, actors, state, start, law, **options):
        key = record["key"]
        selected = candidates is None or key in candidates
        accepted = selected and (admitted is None or key in admitted)
        contributing = any(np.any(record[name] != 0) for name in ("energy", "residual", "tangent", "world_residual"))
        identity = np.asarray(key, dtype="<i8").tobytes()
        hashes["full"].update(identity)
        min_gap = min(min_gap, record["gap"])
        pair = pairs.setdefault(
            key[:2],
            {
                "full": 0,
                "candidate": 0,
                "admitted": 0,
                "full_contributing": 0,
                "candidate_contributing": 0,
                "admitted_contributing": 0,
                "spatial_contributing": 0,
                "capacity_contributing": 0,
            },
        )
        for name, include in (("full", True), ("candidate", selected), ("admitted", accepted)):
            if include:
                counts[name] += 1
                pair[name] += 1
                active_counts[name] += int(contributing)
                pair[name + "_contributing"] += int(contributing)
                _accumulate(totals[name], record, dofs)
        category = "spatial_omission" if not selected else "capacity_omission" if not accepted else None
        if category:
            omissions[category]["count"] += 1
            omissions[category]["contributing_count"] += int(contributing)
            hashes[category].update(identity)
            pair["spatial_contributing" if not selected else "capacity_contributing"] += int(contributing)
    if (candidates is not None and counts["candidate"] != len(candidates)) or (
        admitted is not None and counts["admitted"] != len(admitted)
    ):
        raise ValueError("Unknown candidate or admitted identity")
    for name, reference, selected in (
        ("spatial_omission", "full", "candidate"),
        ("capacity_omission", "candidate", "admitted"),
    ):
        omissions[name].update(key_sha256=hashes[name].hexdigest(), errors=_errors(totals[reference], totals[selected]))
    full_to_admitted = _errors(totals["full"], totals["admitted"])
    for group in totals.values():
        keys = sorted(group["tangent"])
        group["tangent"] = {
            "shape": (actors.scalar_dof_count, actors.scalar_dof_count),
            "row": np.asarray([k[0] for k in keys], dtype=np.int64),
            "column": np.asarray([k[1] for k in keys], dtype=np.int64),
            "value": np.asarray([group["tangent"][k] for k in keys]),
        }
    return {
        **totals,
        **omissions,
        **{name + "_count": value for name, value in counts.items()},
        **{name + "_contributing_count": value for name, value in active_counts.items()},
        "full_key_sha256": hashes["full"].hexdigest(),
        "law": law,
        "full_to_admitted_errors": full_to_admitted,
        "min_gap": min_gap if counts["full"] else 0.0,
        "pairs": [{"source": key[0], "target": key[1], **pair} for key, pair in sorted(pairs.items())],
    }


def audit_keys(full, candidate_keys):
    """Measure spatial omissions separately from any production capacity decision."""
    keys = []
    for key in candidate_keys:
        if len(key) != 4 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in key
        ):
            raise ValueError("Candidate identity requires four integers")
        keys.append(tuple(map(int, key)))
    if len(keys) != len(set(keys)) or set(keys) - full["records"].keys():
        raise ValueError("Duplicate or unknown candidate identity")
    omitted = full["records"].keys() - set(keys)
    contributing = [
        key
        for key in sorted(omitted)
        if any(np.any(full["records"][key][name] != 0) for name in ("energy", "residual", "tangent", "world_residual"))
    ]
    return {
        "missing_keys": sorted(omitted),
        "missing_contributing_keys": contributing,
        "full_count": len(full["records"]),
        "candidate_count": len(keys),
    }
