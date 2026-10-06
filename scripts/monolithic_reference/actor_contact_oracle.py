# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent surface quadrature, relative virtual work and short 1D preload reference."""

import json
from pathlib import Path

import numpy as np

from .actor_contract_oracle import rigid_terms, rotation, sphere_samples
from .articulation_oracle import skew, transform
from .contact_oracle import local

CASE = Path(__file__).with_name("fixtures") / "actor_contact_short_v2.json"


def box_samples(half):
    """Integrate twelve independently constructed outward box triangles with P1Q3."""
    triangles = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for sign in (-1, 1):
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[v] = sign * half[axis], a * half[u], b * half[v]
                corners.append(p)
            if sign < 0:
                corners.reverse()
            triangles.extend([np.array(corners)[[0, 1, 2]], np.array(corners)[[0, 2, 3]]])
    triangles = np.asarray(triangles)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    bary = np.eye(3) * 0.5 + np.full((3, 3), 1 / 6)
    return np.einsum("qn,fni->fqi", bary, triangles).reshape(-1, 3), np.repeat(areas / 3, 3)


def endpoint_terms(point, source_com, target_com, target_rotation):
    """Return both six-column point Jacobians without using production endpoint maps."""
    js = np.column_stack((np.eye(3), -skew(point - source_com)))
    jt = -np.column_stack((np.eye(3), -skew(point - target_com)))
    return target_rotation.T @ np.column_stack((js, jt))


def box_face_samples(half):
    """Use the Newton box's documented face and diagonal order, without its sampler."""
    faces = []
    for axis in range(3):
        u, v = ((1, 2), (0, 2), (0, 1))[axis]
        for sign in (-1, 1):
            quad = np.zeros((4, 3))
            quad[:, axis] = sign * half[axis]
            quad[:, u] = np.array([-1, -1, 1, 1]) * half[u]
            quad[:, v] = np.array([-1, 1, 1, -1]) * half[v]
            ids = [(0, 1, 2), (0, 2, 3)]
            if (axis == 1) != (sign == 1):
                ids = [tuple(reversed(i)) for i in ids]
                # Keep the first vertex and swap the two remaining corners.
                ids = [(i[2], i[0], i[1]) for i in ids]
            faces.extend(quad[list(i)] for i in ids)
    tri = np.array(faces)
    bary = np.eye(3) / 2 + np.ones((3, 3)) / 6
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    return np.einsum("qn,fni->fqi", bary, tri).reshape(-1, 3), np.repeat(area / 3, 3)


def fixture_terms(model, actors, state, start, law, *, mu=0.5, width=1e-4):
    """Independent dense contact for the one-prismatic-arm analytic short fixture.

    Only Model physical input and actor row descriptions are read. No production
    contact keys, samples, geometry, Jacobians, scatter or factors are used.
    """
    n = actors.scalar_dof_count
    energy, residual, tangent = np.zeros(2), np.zeros(n), np.zeros((n, n))
    arm = next(a for a in actors.actors if a.kind == "articulation")
    rigids = [a for a in actors.actors if a.kind == "rigid"]
    body_actors = {b: a for a in actors.actors for b in a.body_indices}
    rigid_local = {a.body_indices[0]: i for i, a in enumerate(rigids)}
    bodies, kinds, scales, margins, flags = (
        getattr(model, f"shape_{name}").numpy() for name in ("body", "type", "scale", "margin", "flags")
    )
    margins = margins.astype(float)
    shape_local = [transform(v) for v in model.shape_transform.numpy()]
    body_com = model.body_com.numpy().astype(float)
    parent = transform(model.joint_X_p.numpy()[arm.joint_indices[0]])

    def body_pose(body, value):
        if body < 0:
            return np.eye(4)
        if body in rigid_local:
            i = rigid_local[body]
            r = transform(np.r_[np.zeros(3), value["rotation"][i]])
            r[:3, 3] = value["com"][i] - r[:3, :3] @ body_com[body]
            return r
        r = parent.copy()
        r[:3, 3] += np.array([value["q"][arm.coord_indices[0]], 0, 0])
        return r

    def point_jacobian(body, point, value):
        j = np.zeros((3, n))
        if body < 0:
            return j
        a = body_actors[body]
        if body in rigid_local:
            j[:, a.dof_start : a.dof_start + 6] = np.column_stack(
                (np.eye(3), -skew(point - value["com"][rigid_local[body]]))
            )
        else:
            j[:, arm.dof_start] = (1, 0, 0)
        return j

    def term(source, target, point, old_point, js, weight, offset):
        nonlocal energy, residual, tangent
        current = body_pose(int(bodies[target]), state) @ shape_local[target]
        old = body_pose(int(bodies[target]), start) @ shape_local[target]
        rt = current[:3, :3]
        y, yn = rt.T @ (point - current[:3, 3]), old[:3, :3].T @ (old_point - old[:3, 3])
        kind = {3: "sphere", 7: "box", 1: "plane"}.get(int(kinds[target]))
        # GeoType values are verified by the caller's analytic fixture.
        if kind is None:
            raise ValueError("Oracle requires an analytic sphere/box/plane target")
        e, r, _, b = local(y, yn, kind, scales[target], law, offset=offset, stiffness=1e7 * weight, width=width, mu=mu)
        j = rt.T @ (js - point_jacobian(int(bodies[target]), point, state))
        energy += e
        residual += j.T @ r
        tangent += j.T @ b @ j

    rigid_shapes = {s for a in rigids for s in a.shape_indices}
    triangles, rest = model.tri_indices.numpy(), model.particle_q.numpy().astype(float)
    for tri in triangles:
        area = np.linalg.norm(np.cross(rest[tri[1]] - rest[tri[0]], rest[tri[2]] - rest[tri[0]])) / 2
        for slot in range(3):
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            js = np.zeros((3, n))
            for node, particle in enumerate(tri):
                row = actors.maps["particle_to_global"][particle]
                if row >= 0:
                    js[:, row : row + 3] = bary[node] * np.eye(3)
            for target in sorted(rigid_shapes):
                if flags[target] & 4:
                    term(
                        -1,
                        target,
                        bary @ state["x"][tri],
                        bary @ start["x"][tri],
                        js,
                        area / 3,
                        float(model.particle_radius.numpy()[0]) + margins[target],
                    )
    for pair in model.shape_contact_pairs.numpy():
        a, b = map(int, pair)
        if (
            not (a in rigid_shapes or b in rigid_shapes)
            or actors.maps["shape_owner"][a] == actors.maps["shape_owner"][b]
        ):
            continue
        for source, target in ((a, b), (b, a)):
            if bodies[source] < 0:
                continue
            ps, weights = box_face_samples(scales[source])
            current = body_pose(int(bodies[source]), state) @ shape_local[source]
            old = body_pose(int(bodies[source]), start) @ shape_local[source]
            for p, weight in zip(ps, weights, strict=True):
                point, old_point = current[:3, 3] + current[:3, :3] @ p, old[:3, 3] + old[:3, :3] @ p
                term(
                    source,
                    target,
                    point,
                    old_point,
                    point_jacobian(int(bodies[source]), point, state),
                    weight * (0.5 if bodies[target] >= 0 else 1),
                    margins[source] + margins[target],
                )
    return energy, residual, tangent


def directed_terms(points, weights, source, target, start_source, start_target, *, kind, scale, law, offset, mu=0.5):
    """Assemble dense two-rigid E/r/B by independent point transformations and local law."""
    energy, residual, tangent = np.zeros(2), np.zeros(12), np.zeros((12, 12))
    positions, reactions = [], []
    cs, rs = source
    ct, rt = target
    cn, rn = start_source
    tn, tr = start_target
    for p, area in zip(points, weights, strict=True):
        world = cs + rs @ p
        y, yn = rt.T @ (world - ct), tr.T @ (cn + rn @ p - tn)
        e, r, _, b = local(y, yn, kind, scale, law, offset=offset, stiffness=1e7 * area, mu=mu)
        jac = endpoint_terms(world, cs, ct, rt)
        energy += e
        residual += jac.T @ r
        tangent += jac.T @ b @ jac
        positions.append(world)
        reactions.append(rt @ r)
    return energy, residual, tangent, np.array(positions), np.array(reactions)


def preload_reference(case):
    """Advance an independent small-displacement axial chain to bound the short preload."""
    h = case["dt"]
    initial = np.array(
        [case["centers"][n][0] for n in ("pusher", "rigid_0", "rigid_1")] + [case.get("tet_initial_compression", 0.0)]
    )
    q, v = initial.copy(), np.array([case.get("pusher_initial_velocity", 0), 0, 0, 0])
    masses = np.array([1, 0.01, 0.01, case["tet"]["density"] * case["tet"]["edge"] ** 3 / 2])
    contact = 1e7 * np.prod(case["dimensions"][1:])
    stiffness = case["tet"]["young_modulus"] * case["tet"]["edge"]
    rest = np.array(
        [case["dimensions"][0] + 2 * case["shape_margin"]] * 2
        + [case["dimensions"][0] / 2 + case["tet"]["edge"] / 2 + case["shape_margin"] + case["particle_radius"]]
    )
    jac = np.diff(np.eye(4), axis=0)
    forces = []
    for step in range(case["chain_steps"]):
        predicted = q + h * v
        a = np.diag(masses / h**2) + contact * jac.T @ jac
        rhs = masses / h**2 * predicted + contact * jac.T @ rest
        gain = case["pusher_pd"]
        a[0, 0] += gain["ke"] + gain["kd"] / h
        rhs[0] += gain["ke"] * (initial[0] + (step + 1) * case["target_increment_per_step"]) + gain["kd"] / h * q[0]
        a[3, 3] += stiffness
        updated = np.linalg.solve(a, rhs)
        v, q = (updated - q) / h, updated
        forces.append(contact * (rest - jac @ q))
    return {
        "min_pair_force": float(np.min(forces)),
        "tet_displacement": float(q[3]),
        "all_interfaces_active": bool(np.min(forces) > 0),
    }


def static_reference(case, law):
    """Advance six native rigid coordinates against a plane with independent NumPy formulas."""
    h, mass = case["dt"], case["mass"]["rigid"]
    dimensions = np.asarray(case["dimensions"])
    inertia = mass / 12 * np.diag(np.sum(dimensions**2) - dimensions**2)
    local_com = np.asarray(case["rigid_com"])
    points, weights = box_face_samples(dimensions / 2)
    rt = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    plane = np.array([case["static_plane_x"], 0, 0])
    force = np.asarray(case["static_body_force_world"])
    com = np.asarray(case["centers"]["rigid_0"]) + local_com
    rot, velocity, omega, sym = np.eye(3), np.zeros(3), np.zeros(3), np.zeros((3, 3))
    frames = []

    def contact(c, r, old_points):
        residual, tangent, wrench = np.zeros(6), np.zeros((6, 6)), np.zeros(6)
        current = c + (points - local_com) @ r.T
        for point, old_point, area in zip(current, old_points, weights, strict=True):
            y, yn = rt.T @ (point - plane), rt.T @ (old_point - plane)
            _, local_r, _, b = local(
                y,
                yn,
                "plane",
                1,
                law,
                offset=2 * case["shape_margin"],
                stiffness=case["contact_stiffness"] * area,
                width=case["smoothing_width"],
                mu=case["friction_coefficient"],
                a=case["friction_velocity_scale"] * h,
            )
            j = rt.T @ np.column_stack((np.eye(3), -skew(point - c)))
            residual += j.T @ local_r
            tangent += j.T @ b @ j
            world_force = -rt @ local_r
            wrench += np.r_[world_force, np.cross(point - c, world_force)]
        return residual, tangent, wrench

    for step in range(case["pair_steps"]):
        old_com, old_rot = com.copy(), rot.copy()
        old_points = old_com + (points - local_com) @ old_rot.T
        predicted_com = com + h * velocity
        predicted_rot = rot + h * (skew(omega) + sym) @ rot
        com = predicted_com.copy()
        for _iteration in range(50):
            _, residual, tangent = rigid_terms(
                com, rot, predicted_com, predicted_rot, mass, inertia, h, force, initial_com=old_com
            )
            cr, cb, wrench = contact(com, rot, old_points)
            residual += cr
            if np.linalg.norm(residual) <= 1e-11:
                break
            delta = np.linalg.solve(tangent + cb, -residual)
            com += delta[:3]
            rot = rotation(delta[3:]) @ rot
        else:
            raise ValueError("Independent static reference did not resolve its equations")
        velocity = (com - old_com) / h
        history = (rot - old_rot) @ rot.T / h
        omega = (
            np.array([history[2, 1] - history[1, 2], history[0, 2] - history[2, 0], history[1, 0] - history[0, 1]]) / 2
        )
        sym = (history + history.T) / 2
        angle = np.arctan2(
            np.linalg.norm([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / 2,
            (np.trace(rot) - 1) / 2,
        )
        frames.append(
            {
                "step": step + 1,
                "com": com.tolist(),
                "contact_wrench": wrench.tolist(),
                "normal_force": float(wrench[0]),
                "rotation_rad": float(angle),
                "equation_residual": float(np.linalg.norm(residual)),
                "iterations": _iteration,
            }
        )
    return frames


def calibration():
    """Calibrate area, virtual work, PSD roundoff and preload without production kernels."""
    case = json.loads(CASE.read_text())
    points, weights = box_samples(np.array(case["dimensions"]) / 2)
    expected_area = 2 * sum(np.prod(np.delete(case["dimensions"], i)) for i in range(3))
    src, dst = (np.array([0.01, -0.02, -0.0398]), rotation(np.array([0.03, -0.02, 0.01]))), (np.zeros(3), np.eye(3))
    # The plane cap has smooth distance everywhere, independent of triangle correspondence.
    _e, r, b, _, _ = directed_terms(
        points, weights, src, dst, src, dst, kind="plane", scale=1, law="C-EE", offset=0.0002, mu=0
    )
    errors = []
    for eps in (1e-7, 3e-8):
        numerical = []
        for col in range(12):
            values = []
            for sign in (1, -1):
                s, t = (src[0].copy(), src[1].copy()), (dst[0].copy(), dst[1].copy())
                obj = s if col < 6 else t
                j = col % 6
                if j < 3:
                    obj[0][j] += sign * eps
                else:
                    w = np.eye(3)[j - 3] * sign * eps
                    obj = (obj[0], rotation(w) @ obj[1])
                    if col < 6:
                        s = obj
                    else:
                        t = obj
                values.append(
                    directed_terms(
                        points, weights, s, t, src, dst, kind="plane", scale=1, law="C-EE", offset=0.0002, mu=0
                    )[0].sum()
                )
            numerical.append((values[0] - values[1]) / (2 * eps))
        errors.append(float(np.max(np.abs(np.array(numerical) - r)) / max(1, np.max(np.abs(r)))))
    sphere_p, sphere_w = sphere_samples(0.02)
    preload = preload_reference(case)
    return {
        "box_area_error": float(abs(weights.sum() / expected_area - 1)),
        "sphere_samples": len(sphere_p),
        "sphere_area_error": float(abs(sphere_w.sum() / (4 * np.pi * 0.02**2) - 1)),
        "derivative_errors": errors,
        "fp32_tangent_error": float(np.linalg.norm(b.astype(np.float32).astype(float) - b) / np.linalg.norm(b)),
        "min_eigenvalue": float(np.linalg.eigvalsh(b)[0]),
        "preload_1d": preload,
        "scope": "Independent axial preload and surface/endpoint calibration; not a 3D production trajectory qualification",
    }
