# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Enumerate all supported contact samples from Model input, independently.

No production sampling, contact, broad phase, capacity or support map is read.
Native states and actor row descriptions are explicit inputs. Mesh targets
require an independently exported SDF field; CPU texture handles are not used.
"""

import numpy as np

from .actor_contact_oracle import box_face_samples
from .actor_contract_oracle import rotation, sphere_samples
from .articulation_oracle import skew, transform
from .contact_oracle import geometry, local


def export_mesh_fields(model, sdf):
    """Export immutable C2 coefficients as independent natural-spline node input."""
    table = model._texture_sdf_data.numpy()
    indices, kinds, scales = (
        getattr(model, name).numpy() for name in ("_shape_sdf_index", "shape_type", "shape_scale")
    )
    result = {}
    for shape in np.flatnonzero(kinds == 8):
        index = int(indices[shape])
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
        result[int(shape)] = {
            "smooth_grid": nodes * unit,
            "lower": np.asarray(field.identity["lower"]) * unit,
            "upper": np.asarray(field.identity["upper"]) * unit,
            "representation": "c2-bspline",
            "identity": dict(field.identity),
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
        j = np.zeros((3, actors.scalar_dof_count))
        if body in rigid_rows:
            start = rigid_rows[body]
            j[:, start : start + 6] = np.column_stack((np.eye(3), -skew(point - centers[body])))
        elif body in ancestors:
            for column in ancestors[body]:
                kind, axis, origin = screws[column]
                j[:, column] = axis if kind == 0 else np.cross(axis, point - origin)
        return j

    return poses, centers, jacobian, ancestors


def _samples(model, source, kinds, scales):
    if kinds[source] == 7:
        return box_face_samples(scales[source])
    if kinds[source] == 3:
        return sphere_samples(float(scales[source, 0]))
    if kinds[source] != 8:
        raise ValueError("Unsupported full-oracle source shape")
    mesh = model.shape_source[source]
    tri = (np.asarray(mesh.vertices, dtype=float) * scales[source])[np.asarray(mesh.indices).reshape(-1, 3)]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    bary = np.eye(3) / 2 + np.ones((3, 3)) / 6
    return np.einsum("qn,fni->fqi", bary, tri).reshape(-1, 3), np.repeat(area / 3, 3)


def evaluate_full(
    model, actors, state, start, law, *, radius, stiffness=1e7, width=1e-4, mu=0.5, falloff=1e-5, mesh_fields=None
):
    """Return full dense E/r/B, endpoint forces and sparse per-key contribution records."""
    if any(not np.isfinite(value).all() for physical in (state, start) for value in physical.values()):
        raise ValueError("Full oracle requires finite native states")
    n = actors.scalar_dof_count
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
    energy, residual, tangent = np.zeros(2), np.zeros(n), np.zeros((n, n))
    particle_force, body_wrench = np.zeros((model.particle_count, 3)), np.zeros((model.body_count, 6))
    joint_force = np.zeros(model.joint_dof_count)
    dofs = {a.dof_start + i: d for a in actors.actors if a.kind != "tet" for i, d in enumerate(a.dof_indices)}
    records = {}
    min_gap = np.inf

    def append(key, point, old_point, source_j, weight, offset, tri=None, bary=None):
        nonlocal energy, min_gap
        source, target, _, _ = key
        rt = current[target][:3, :3]
        y = rt.T @ (point - current[target][:3, 3])
        yn = initial[target][:3, :3].T @ (old_point - initial[target][:3, 3])
        kind = {1: "plane", 3: "sphere", 7: "box"}.get(int(kinds[target]))
        if kind is None:
            if kinds[target] != 8 or mesh_fields is None or target not in mesh_fields:
                raise ValueError("Full oracle mesh target requires independently exported SDF input")
            kind = mesh_fields[target]
        min_gap = min(min_gap, float(geometry(y, kind, scales[target])[0] - offset))
        e, r, _, b = local(
            y, yn, kind, scales[target], law, offset=offset, stiffness=stiffness * weight, width=width, mu=mu, a=falloff
        )
        if any(not np.isfinite(value).all() for value in (e, r, b)):
            raise ValueError("Nonfinite full-oracle local contribution")
        j = rt.T @ (source_j - jacobian(int(bodies[target]), point))
        columns = np.flatnonzero(np.any(j != 0, axis=0))
        reduced = j[:, columns]
        rr, bb = reduced.T @ r, reduced.T @ b @ reduced
        world = rt @ r
        if key in records:
            raise ValueError("Duplicate independently enumerated contact identity")
        records[key] = {
            "energy": e,
            "columns": columns,
            "residual": rr,
            "tangent": bb,
            "point": point,
            "world_residual": world,
        }
        energy += e
        residual[columns] += rr
        tangent[np.ix_(columns, columns)] += bb
        for column, value in zip(columns, rr, strict=True):
            if column in dofs:
                joint_force[dofs[column]] -= value
        for shape, force in ((source, -world), (target, world)):
            if shape < 0:
                for particle, coefficient in zip(tri, bary, strict=True):
                    particle_force[particle] += coefficient * force
            elif bodies[shape] >= 0:
                body = bodies[shape]
                body_wrench[body] += np.r_[force, np.cross(point - centers[body], force)]

    rest = model.particle_q.numpy().astype(float)
    for face, tri in enumerate(model.tri_indices.numpy()):
        area = np.linalg.norm(np.cross(rest[tri[1]] - rest[tri[0]], rest[tri[2]] - rest[tri[0]])) / 2
        for slot in range(3):
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            j = np.zeros((3, n))
            for particle, coefficient in zip(tri, bary, strict=True):
                if particle in particle_rows:
                    row = particle_rows[particle]
                    j[:, row : row + 3] = coefficient * np.eye(3)
            for target in np.flatnonzero(flags & 4):
                append(
                    (-1, int(target), face, slot),
                    bary @ state["x"][tri],
                    bary @ start["x"][tri],
                    j,
                    area / 3,
                    radius + margins[target],
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
            points, weights = _samples(model, source, kinds, scales)
            for sample, (point, weight) in enumerate(zip(points, weights, strict=True)):
                world = current[source][:3, :3] @ point + current[source][:3, 3]
                old = initial[source][:3, :3] @ point + initial[source][:3, 3]
                append(
                    (source, target, sample // 3, sample % 3),
                    world,
                    old,
                    jacobian(int(bodies[source]), world),
                    weight * (0.5 if dynamic[target] else 1),
                    margins[source] + margins[target],
                )
    return {
        "energy": energy,
        "residual": residual,
        "tangent": tangent,
        "particle_force": particle_force,
        "body_wrench": body_wrench,
        "joint_force": joint_force,
        "records": records,
        "min_gap": min_gap if records else 0.0,
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
