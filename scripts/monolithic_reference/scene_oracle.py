# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline NumPy FP64 audit of native scene snapshots; never feeds the solver."""

from types import SimpleNamespace

import numpy as np

from scripts.monolithic_reference.articulation_oracle import Kinematics, joint_terms, skew, transform


def unpack_symmetric(values):
    """Decode the six stored independent symmetric entries in FP64."""
    values = np.asarray(values, dtype=np.float64)
    result = np.empty((*values.shape[:-1], 3, 3))
    for i, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        result[..., a, b] = result[..., b, a] = values[..., i]
    return result


def spline(field, index, points):
    """Evaluate captured coefficients using SciPy's independent cardinal basis."""
    from scipy.interpolate import BSpline

    prefix = str(index) + "."
    lower, upper, inv, dims = (np.asarray(field[prefix + k], dtype=float) for k in ("lower", "upper", "inv_dx", "dims"))
    dims = dims.astype(int)
    p = np.clip(points, lower, upper)
    f = (p - lower) * inv
    cell = np.clip(np.floor(f).astype(int), 0, dims - 2)
    t = f - cell
    basis = BSpline.basis_element(np.arange(5), extrapolate=True)
    w = np.stack([basis(t + 3 - i) for i in range(4)], axis=-1)
    d = np.stack([basis.derivative()(t + 3 - i) for i in range(4)], axis=-1) * inv[None, :, None]
    coefficients = field[prefix + "coefficients"].astype(float).reshape(tuple((dims + 2)[::-1]))
    value, gradient = np.zeros(len(points)), np.zeros_like(points, dtype=float)
    for k in range(4):
        for j in range(4):
            for i in range(4):
                c = coefficients[cell[:, 2] + k, cell[:, 1] + j, cell[:, 0] + i]
                value += c * w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                gradient[:, 0] += c * d[:, 0, i] * w[:, 1, j] * w[:, 2, k]
                gradient[:, 1] += c * w[:, 0, i] * d[:, 1, j] * w[:, 2, k]
                gradient[:, 2] += c * w[:, 0, i] * w[:, 1, j] * d[:, 2, k]
    delta = points - p
    length = np.linalg.norm(delta, axis=1)
    exterior = length > 0
    value += length
    gradient[exterior] *= ((points >= lower) & (points <= upper))[exterior]
    gradient[exterior] += delta[exterior] / length[exterior, None]
    return value, gradient, np.any((points <= lower) | (points >= upper), axis=1)


def penalty(gap, stiffness, width):
    """Return normal potential and load with the frozen smooth penalty."""
    u = -gap
    if width == 0:
        p, dp = np.maximum(u, 0), (u > 0).astype(float)
    else:
        t = np.clip((u + width) / (2 * width), 0, 1)
        p = np.where(u >= width, u, width * t**3 * (2 - t))
        dp = np.where(u >= width, 1.0, t**2 * (3 - 2 * t))
    return 0.5 * stiffness * p**2, stiffness * p * dp


class SceneOracle:
    """Recompute FK, prediction, tet and smooth contact from the native history."""

    def __init__(self, model, physics, contact, fields, config):
        self.model, self.physics, self.contact, self.fields, self.config = model, physics, contact, fields, config
        self.k = Kinematics(SimpleNamespace(**model))
        self.ids = np.flatnonzero(model["particle_inv_mass"] > 0)
        self.tri = contact["triangles"]
        self.shapes = contact["shapes"]
        self.transforms = np.asarray([transform(v) for v in contact["transforms"]])
        self.bary = np.full((3, 3), 1 / 6)
        np.fill_diagonal(self.bary, 2 / 3)

    def frame(self, q, x):
        """Compute exact promoted-coordinate geometry and first derivatives."""
        t, r, jt, jr, *_ = self.k.evaluate(q)
        points = np.einsum("sa,fai->fsi", self.bary, x[self.tri]).reshape(-1, 3)
        normal = np.cross(x[self.tri[:, 1]] - x[self.tri[:, 0]], x[self.tri[:, 2]] - x[self.tri[:, 0]])
        lengths = np.linalg.norm(normal, axis=1)
        soft = np.repeat(normal / lengths[:, None], 3, axis=0)
        frames = []
        for shape in self.shapes:
            body = self.contact["bodies"][shape]
            local = self.transforms[shape]
            rot = local[:3, :3] if body < 0 else r[body] @ local[:3, :3]
            shift = local[:3, 3] if body < 0 else t[body] + r[body] @ (local[:3, 3] - self.k.body_com[body])
            y = (points - shift) @ rot
            jacobian = np.zeros((len(y), self.k.n, 3))
            if body >= 0:
                jacobian = np.einsum("ij,nkj,pk->pni", local[:3, :3].T, jr[body], points - t[body])
                jacobian -= (jt[body] @ r[body]) @ local[:3, :3]
            kind = int(self.contact["types"][shape])
            if kind == 1:
                phi, g, outside = y[:, 2], np.tile([0.0, 0.0, 1.0], (len(y), 1)), np.zeros(len(y), bool)
            elif kind == 8:
                index = int(self.contact["sdf_indices"][shape])
                scale = 1.0 if self.config["sdf_scale_baked"][index] else float(self.contact["scales"][shape, 0])
                phi, g, outside = spline(self.fields, index, y / scale)
                phi *= scale
            elif kind == 7:
                d = abs(y) - self.contact["scales"][shape].astype(float)
                positive = np.maximum(d, 0)
                length = np.linalg.norm(positive, axis=1)
                phi = length + np.minimum(d.max(axis=1), 0)
                g = positive / np.where(length > 0, length, 1)[:, None]
                inside = np.flatnonzero(length == 0)
                g[inside, np.argmax(d[inside], axis=1)] = 1
                g *= np.where(y < 0, -1.0, 1.0)
                outside = np.zeros(len(y), bool)
            elif kind == 3:
                length = np.linalg.norm(y, axis=1)
                phi, g, outside = length - self.contact["scales"][shape, 0], y / length[:, None], np.zeros(len(y), bool)
            else:
                raise ValueError(f"Unsupported audit shape {kind}")
            frames.append((y, phi, g, outside, jacobian, rot, soft @ rot, body, points))
        return t, r, lengths, frames

    def evaluate(self, data, prefix="state."):
        """Return independent E/r and geometry for the actual saved candidate."""

        def get(key):
            return np.asarray(data[key], dtype=float)

        q, x = get(prefix + "joint_q"), get(prefix + "particle_q")
        qn, xn, vn = get("original.joint_q"), get("original.particle_q"), get("original.particle_qd")
        dt = float(data["dt"])
        tn, rn, _, old_frames = self.frame(qn, xn)
        t, r, lengths, frames = self.frame(q, x)
        omega = get("original.history.link_omega")
        sym = unpack_symmetric(get("original.history.link_vsym"))
        predicted_t = tn + dt * get("original.history.link_v_com")
        predicted_r = rn + dt * (np.asarray([skew(v) for v in omega]) + sym) @ rn
        gravity = get("native.gravity")
        force = (
            get("original.body_f")[:, :3] + self.k.body_mass[:, None].astype(float) * gravity[self.model["body_world"]]
        )
        energy, qr, _ = self.k.inertia(q, predicted_t, predicted_r, dt, force)
        energy += np.sum(force * tn)
        je, jr, _ = joint_terms(
            q,
            qn,
            get("original.joint_qd"),
            dt,
            get("objective.params"),
            get("objective.armature"),
            get("objective.damping"),
            get("objective.qt"),
            get("objective.vt"),
            get("objective.joint_force"),
        )
        link_delta = t - predicted_t
        rotation_delta = r - predicted_r
        link_energy = self.k.body_mass.astype(float) * np.sum(link_delta**2, axis=1) / (2 * dt**2)
        link_energy += np.sum(rotation_delta * (rotation_delta @ self.k.second), axis=(1, 2)) / (2 * dt**2) - np.sum(
            force * (t - tn), axis=1
        )
        absolute = float(abs(link_energy).sum() + abs(je).sum())
        energy += je.sum()
        qr += jr
        p = self.physics
        delta = x - xn
        displacement = delta - dt * vn
        loads = p["row_mass"][:, None].astype(float) * gravity[p["worlds"]] + get("native.particle_force")
        xr = -loads.copy()
        external = -np.sum(loads * delta)
        inertia = 0.0
        energy += external
        if self.config["mass"] == "consistent":
            for ids, coefficient in zip(p["indices"], p["coefficients"], strict=True):
                d = displacement[ids]
                md = float(coefficient) * (d + d.sum(axis=0))
                np.add.at(xr, ids, md / dt**2)
                inertia += np.sum(d * md) / (2 * dt**2)
        else:
            xr += p["mass"][:, None].astype(float) * displacement / dt**2
            inertia += np.sum(p["mass"][:, None] * displacement**2) / (2 * dt**2)
        energy += inertia
        poses = p["poses"].astype(float)
        ids = p["indices"]
        f = (x[ids[:, 1:]] - x[ids[:, :1]]).transpose(0, 2, 1) @ poses
        det = np.linalg.det(f)
        a, b, c = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        cof = np.stack((np.cross(b, c), np.cross(c, a), np.cross(a, b)), axis=2)
        mu, lam = p["materials"][:, 0].astype(float), p["materials"][:, 1].astype(float)
        ic, dd = np.sum(f * f, axis=(1, 2)), det - 1
        if self.config["material"] == "smith_log_stabilized":
            mh, lh = 4 * mu / 3, lam + 5 * mu / 6
            density = mh / 2 * (ic - 3 - np.log((ic + 1) / 4)) + lh / 2 * dd**2 - mu * dd
            stress = (mh * (1 - 1 / (ic + 1)))[:, None, None] * f + (lh * dd - mu)[:, None, None] * cof
        else:
            density = mu / 2 * (ic - 3) - mu * dd + (lam + mu) / 2 * dd**2
            stress = mu[:, None, None] * f + ((lam + mu) * dd - mu)[:, None, None] * cof
        volume = 1 / (6 * np.linalg.det(poses))
        energy += volume @ density
        absolute += abs(inertia) + abs(external) + abs(volume @ density)
        gradients = np.concatenate((-poses.sum(axis=1)[:, None], poses), axis=1)
        np.add.at(
            xr, ids.ravel(), (volume[:, None, None] * np.einsum("tij,taj->tai", stress, gradients)).reshape(-1, 3)
        )
        particle_force, wrench = np.zeros_like(x), np.zeros((self.k.b, 6))
        contact_q = np.zeros_like(q)
        stiffness, mu, velocity, width, radius = self.config["contact_parameters"]
        native = data["state.joint_q"].dtype
        stiffness, mu, velocity, width, radius = np.asarray(
            [stiffness, mu, velocity, width, radius], dtype=native
        ).astype(float)
        masks, gaps, invalid = [], [], []
        for shape, current, old in zip(self.shapes, frames, old_frames, strict=True):
            y, phi, g, outside, jacobian, rotation, soft, body, points = current
            old_y, old_phi, old_g, old_outside, *_ = old
            area = np.repeat(self.contact["area"].astype(float), 3) / 3
            gap = phi - radius - float(self.contact["margins"][shape])
            u = y - old_y
            en, load_normal = penalty(gap, stiffness * area, width)
            law = self.config["law"]
            direction = old_g if law == "C-EE" else g
            laggap = (
                old_phi - radius - float(self.contact["margins"][shape])
                if law == "C-EE"
                else gap - np.sum(g * u, axis=1)
                if law == "C-IE"
                else gap
            )
            _, load = penalty(laggap, stiffness * area, width)
            if mu == 0:
                load[:] = 0
            length = np.linalg.norm(direction, axis=1)
            n = direction / np.where(length > 0, length, 1)[:, None]
            v = u - np.sum(n * u, axis=1)[:, None] * n
            rho = np.linalg.norm(v, axis=1)
            falloff = float(np.asarray(dt, dtype=native) * np.asarray(velocity, dtype=native))
            psi = np.where(rho < falloff, rho**2 / falloff - rho**3 / (3 * falloff**2), rho - falloff / 3)
            factor = np.where(rho < falloff, 2 / falloff - rho / falloff**2, 1 / np.maximum(rho, 1e-300))
            mask = (
                np.sum(soft * direction, axis=1) > 0
                if self.config.get("alignment", "mask") == "mask"
                else np.zeros(len(y), bool)
            )
            active = (load_normal > 0) | (load > 0)
            invalid.append(
                bool(
                    np.any(active & (outside | ((load > 0) & old_outside)))
                    or np.any(active & (np.linalg.norm(g, axis=1) == 0))
                )
            )
            local = -load_normal[:, None] * g + (mu * load * factor)[:, None] * v
            local[mask] = 0
            energy += np.sum(np.where(mask, 0, en + mu * load * psi))
            absolute += np.sum(np.where(mask, 0, abs(en) + abs(mu * load * psi)))
            contact_q += np.einsum("pni,pi->n", jacobian, local)
            reaction = local @ rotation.T
            face_force = -np.einsum("sa,fsi->fai", self.bary, reaction.reshape(-1, 3, 3))
            np.add.at(particle_force, self.tri.ravel(), face_force.reshape(-1, 3))
            if body >= 0:
                wrench[body, :3] += reaction.sum(axis=0)
                wrench[body, 3:] += np.cross(points - t[body], reaction).sum(axis=0)
            gaps.append(gap)
            masks.append(mask)
        qr += contact_q
        xr -= particle_force
        return {
            "energy": float(energy),
            "absolute": float(absolute),
            "residual": np.r_[qr, xr[self.ids].ravel()],
            "com": t,
            "rotation": r,
            "predicted_t": predicted_t,
            "predicted_r": predicted_r,
            "particle_force": particle_force,
            "body_wrench": wrench,
            "joint_force": -contact_q,
            "penetration": float(max(0, -np.min(gaps))),
            "min_det_f": float(det.min()),
            "mask": np.stack(masks, axis=1),
            "geometry_valid": bool(np.all(lengths > 0) and not any(invalid)),
            "local_domain": bool(
                np.all(np.einsum("bij,bij->b", r, rn) > 1)
                and np.all(abs(q - qn)[self.k.joint_q_start[:-1][self.k.joint_type == 1]] < np.pi / 2)
            ),
        }
