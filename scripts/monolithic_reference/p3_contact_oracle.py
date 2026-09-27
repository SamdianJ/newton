# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy contact oracle in a rotating/translating rigid frame.

No production geometry, law, residual or tangent evaluator is imported.
"""

import itertools

import numpy as np

from scripts.monolithic_reference.p3_smooth_sdf_oracle import evaluate as smooth_reference

GENERATOR = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def geometry(y, kind, scale):
    if isinstance(kind, dict):
        if "smooth_grid" in kind:
            row = smooth_reference(kind["smooth_grid"], kind["lower"], kind["upper"], [y], kind["representation"])[0]
            return row[0], row[1:4], row[4:].reshape(3, 3)
        value = mesh_reference(kind, [y])[0]
        # Independent cell-local derivative of the raw mesh gradient. The
        # geometry tests separately qualify phi -> gradient on actual textures.
        h = 1e-7
        offsets = np.eye(3) * h
        upper = mesh_reference(kind, y + offsets)[:, 1:]
        lower = mesh_reference(kind, y - offsets)[:, 1:]
        return value[0], value[1:], ((upper - lower) / (2 * h)).T
    if kind == "plane":
        return y[2], np.array([0.0, 0.0, 1.0]), np.zeros((3, 3))
    if kind == "sphere":
        length = np.linalg.norm(y)
        g = y / length
        return length - scale[0], g, (np.eye(3) - np.outer(g, g)) / length
    d = np.abs(y) - scale
    v = np.maximum(d, 0)
    length = np.linalg.norm(v)
    if length:
        g = np.sign(y) * v / length
        active = np.diag(d > 0)
        h = (active - np.outer(g, g)) / length
        return length, g, h
    g = np.zeros(3)
    axis = np.argmax(d)
    g[axis] = -1 if y[axis] < 0 else 1
    return d[axis], g, np.zeros((3, 3))


def penalty(gap, stiffness, width):
    u = -gap
    if u <= -width:
        return 0.0, 0.0, 0.0
    if u >= width:
        p, dp, ddp = u, 1.0, 0.0
    else:
        t = (u + width) / (2 * width)
        p, dp, ddp = width * (2 * t**3 - t**4), 3 * t**2 - 2 * t**3, 3 * t * (1 - t) / width
    return stiffness * p**2 / 2, stiffness * p * dp, stiffness * (dp**2 + p * ddp)


def local(y, yn, kind, scale, law, *, offset, stiffness, width=1e-4, mu=0.5, a=1e-5):
    phi, g, h = geometry(y, kind, scale)
    old_phi, old_g, _ = geometry(yn, kind, scale)
    d, dn = phi - offset, old_phi - offset
    en, force, curvature = penalty(d, stiffness, width)
    u = y - yn
    if law == "C-EE":
        n = old_g / np.linalg.norm(old_g)
        load = penalty(dn, stiffness, width)[1]
        dnmat, dload = np.zeros((3, 3)), np.zeros(3)
    else:
        n = g / np.linalg.norm(g)
        dnmat = (np.eye(3) - np.outer(n, n)) @ h / np.linalg.norm(g)
        gap = d - g @ u if law == "C-IE" else d
        _, load, load_curvature = penalty(gap, stiffness, width)
        dload = load_curvature * h @ u if law == "C-IE" else -load_curvature * g
    p = np.eye(3) - np.outer(n, n)
    v = p @ u
    rho = np.linalg.norm(v)
    if rho < a:
        psi, factor, derivative = rho**2 / a - rho**3 / (3 * a**2), 2 / a - rho / a**2, -1 / a**2
    else:
        psi, factor, derivative = rho - a / 3, 1 / rho, -1 / rho**2
    dv = p - (np.outer(n, u) + (n @ u) * np.eye(3)) @ dnmat
    radial = factor * np.eye(3)
    if rho:
        radial += derivative / rho * np.outer(v, v)
    friction = mu * load * factor * v
    dr = curvature * np.outer(g, g) - force * h + mu * (np.outer(factor * v, dload) + load * radial @ dv)
    b = curvature * np.outer(g, g) + mu * load * radial @ p
    return np.array([en, mu * load * psi]), -force * g + friction, dr, b


def evaluate(model, q, x, qn, xn, law, *, kinds, mapping=None, mu=0.5, radius=0.001):
    """Two coordinates: rotation about world Y, translation along world X."""
    if mapping is None:
        mapping = np.arange(len(x))
    n = 2 + 3 * np.count_nonzero(mapping >= 0)
    residual, dr, b = np.zeros(n), np.zeros((n, n)), np.zeros((n, n))
    energies = []
    triangles = model.tri_indices.numpy()
    rest = model.particle_q.numpy().astype(float)
    transforms = model.shape_transform.numpy().astype(float)
    scales, margins = model.shape_scale.numpy().astype(float), model.shape_margin.numpy().astype(float)
    rr, rn = rotation(q[0]), rotation(qn[0])
    for ids in triangles:
        area = np.linalg.norm(np.cross(rest[ids[1]] - rest[ids[0]], rest[ids[2]] - rest[ids[0]])) / 2
        for slot in range(3):
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            for shape, kind in enumerate(kinds):
                y0 = rr.T @ (bary @ x[ids] - [q[1], 0, 0])
                y = y0 - transforms[shape, :3]
                yn = rn.T @ (bary @ xn[ids] - [qn[1], 0, 0]) - transforms[shape, :3]
                j = np.zeros((3, n))
                j[:, 0], j[:, 1] = -GENERATOR @ y0, -rr.T[:, 0]
                for k, particle in enumerate(ids):
                    node = mapping[particle]
                    if node >= 0:
                        j[:, 2 + 3 * node : 5 + 3 * node] = bary[k] * rr.T
                hy = np.zeros((3, n, n))
                hy[:, 0, 0] = GENERATOR @ GENERATOR @ y0
                hy[:, 0, 1:] = -GENERATOR @ j[:, 1:]
                hy[:, 1:, 0] = hy[:, 0, 1:]
                e, r, exact, fitted = local(
                    y, yn, kind, scales[shape], law, offset=radius + margins[shape], stiffness=1e7 * area / 3, mu=mu
                )
                energies.append(e)
                residual += j.T @ r
                dr += j.T @ exact @ j + np.einsum("i,ijk->jk", r, hy)
                b += j.T @ fitted @ j
    return np.asarray(energies), residual, dr, b


def mesh_reference(data, points):
    """Long-double successive interpolation of stored texels and the legacy extension."""
    ld = np.longdouble
    lo = data["min_extents"].astype(np.float32).astype(ld)
    hi = data["max_extents"].astype(np.float32).astype(ld)
    inv = (1 / data["cell_size"]).astype(np.float32).astype(ld)
    sub = int(data["subgrid_size"])
    dims = np.array(data["coarse_sdf"].shape[::-1]) - 1
    rows = []
    for p in np.asarray(points, dtype=ld):
        clamped = np.clip(p, lo, hi)
        raw = (clamped - lo) * inv
        f = np.clip(raw, 0, dims * sub)
        derivative = inv * (raw == f)
        cell = np.clip(np.floor(f).astype(int), 0, dims * sub - 1)
        base = cell // sub
        slot = int(data["subgrid_start_slots"][tuple(base)])
        if slot < 0xFFFFFFFE:
            start = np.array([slot & 1023, (slot >> 10) & 1023, (slot >> 20) & 1023]) * (sub + 1) + cell - base * sub
            t, scale, grid = f - cell, derivative, data["subgrid_data"]
            mul = ld(np.float32(data["subgrids_sdf_value_range"]))
            offset = ld(np.float32(data["subgrids_min_sdf_value"]))
        else:
            start, t, scale, grid, mul, offset = (
                base,
                f / sub - base,
                derivative / sub,
                data["coarse_sdf"],
                ld(1),
                ld(0),
            )
        c = grid[start[2] : start[2] + 2, start[1] : start[1] + 2, start[0] : start[0] + 2].astype(ld)
        alongx = c[:, :, 0] * (1 - t[0]) + c[:, :, 1] * t[0]
        alongy = alongx[:, 0] * (1 - t[1]) + alongx[:, 1] * t[1]
        value = (alongy[0] * (1 - t[2]) + alongy[1] * t[2]) * mul + offset
        gradient = []
        for axis in range(3):
            diff = np.diff(c, axis=2 - axis).squeeze(axis=2 - axis)
            others = [a for a in [2, 1, 0] if a != axis]
            g = ld(0)
            for bits in itertools.product([0, 1], repeat=2):
                weight = ld(1)
                for a, bit in zip(others, bits, strict=True):
                    weight *= t[a] if bit else 1 - t[a]
                g += diff[bits] * weight
            gradient.append(g * scale[axis] * mul)
        gradient = np.asarray(gradient)
        delta = p - clamped
        length = np.linalg.norm(delta)
        if length:
            value += length
            gradient = gradient * ((p >= lo) & (p <= hi)) + delta / length
        rows.append([value, *gradient])
    return np.asarray(rows, dtype=float)
