# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent dense FP64 tiny reference, with exact residual Jacobian and history."""

import numpy as np

from scripts.monolithic_reference.articulation_oracle import Kinematics, joint_terms, transform
from scripts.monolithic_reference.contact_oracle import geometry, local, penalty
from scripts.monolithic_reference.tet_oracle import TetOracle


class CoupledOracle:
    """Combine NumPy references; no production residual, tangent, FK or solve calls."""

    def __init__(self, model, physics, parameters, *, mu, width, goal_scale):
        self.k = Kinematics(model)
        self.tet = TetOracle(model, physics)
        self.params = np.asarray(parameters).copy()
        self.armature = model.joint_armature.numpy().astype(float)
        self.damping = model.joint_damping.numpy().astype(float)
        self.q = model.joint_q.numpy().astype(float)
        self.v = model.joint_qd.numpy().astype(float)
        self.x = model.particle_q.numpy().astype(float)
        self.xv = model.particle_qd.numpy().astype(float)
        self.t, self.r, *_ = self.k.evaluate(self.q)
        if np.any(self.v) or np.any(self.xv):
            raise ValueError("These frozen T4 fixtures require zero initial velocity")
        self.tv = np.zeros_like(self.t)
        self.rv = np.zeros_like(self.r)
        self.triangles = model.tri_indices.numpy()
        self.shapes = model.shape_body.numpy()
        self.transforms = [transform(v) for v in model.shape_transform.numpy()]
        # Newton serialized GeoType values; this reference only accepts these two shapes.
        self.kinds = [{1: "plane", 3: "sphere"}[int(kind)] for kind in model.shape_type.numpy()]
        self.scales = model.shape_scale.numpy().astype(float)
        self.offset = model.shape_margin.numpy().astype(float) + float(model.particle_radius.numpy()[0])
        points = self.x[self.triangles]
        self.areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) / 2
        self.mapping = np.full(len(self.x), -1, dtype=int)
        self.mapping[self.tet.free] = np.arange(len(self.tet.free))
        self.mu, self.width, self.goal_scale = mu, width, goal_scale
        self.nq = len(self.q)
        self.n = self.nq + 3 * len(self.tet.free)
        self.gravity = model.gravity.numpy()[0].astype(float)
        self.joint_force = np.zeros(self.nq)
        self.particle_force = np.zeros_like(self.x)

    def frame(self, q, x):
        """Differentiate collider-local quadrature coordinates through exact FK."""
        t, r, jt, jr, ht, hr = self.k.evaluate(q)
        for face, ids in enumerate(self.triangles):
            soft = np.cross(x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]])
            if np.linalg.norm(soft) == 0:
                raise ValueError("Degenerate surface")
            soft /= np.linalg.norm(soft)
            for slot in range(3):
                bary = np.full(3, 1 / 6)
                bary[slot] = 2 / 3
                point = bary @ x[ids]
                for shape, body in enumerate(self.shapes):
                    shift = self.transforms[shape][:3, 3]
                    rotation = self.transforms[shape][:3, :3]
                    j = np.zeros((3, self.n))
                    hy = np.zeros((3, self.n, self.n))
                    if body < 0:
                        world = rotation
                        y = rotation.T @ (point - shift)
                    else:
                        rel = point - t[body]
                        world = r[body] @ rotation
                        y = rotation.T @ (r[body].T @ rel + self.k.body_com[body] - shift)
                        for i in range(self.nq):
                            j[:, i] = rotation.T @ (jr[body, i].T @ rel - r[body].T @ jt[body, i])
                            for l in range(self.nq):
                                hy[:, i, l] = rotation.T @ (
                                    hr[body, i, l].T @ rel
                                    - jr[body, i].T @ jt[body, l]
                                    - jr[body, l].T @ jt[body, i]
                                    - r[body].T @ ht[body, i, l]
                                )
                    for a, node in enumerate(ids):
                        if self.mapping[node] < 0:
                            continue
                        col = self.nq + 3 * self.mapping[node]
                        j[:, col : col + 3] = bary[a] * world.T
                        if body >= 0:
                            for i in range(self.nq):
                                hy[:, i, col : col + 3] = bary[a] * rotation.T @ jr[body, i].T
                                hy[:, col : col + 3, i] = hy[:, i, col : col + 3]
                    yield y, j, hy, world.T @ soft, self.areas[face] / 3, shape

    def freeze(self, dt, joint_force):
        """Freeze the physical step and its complete residual reference."""
        self.dt = dt
        self.joint_force = np.asarray(joint_force).copy()
        self.pt = self.t + dt * self.tv
        self.pr = self.r + dt * self.rv @ self.r
        self.start = list(self.frame(self.q, self.x))
        self.r0 = self.evaluate(np.r_[self.q, self.x[self.tet.free].ravel()])[1]

    def evaluate(self, z):
        """Return exact E/r/Dr, fitted stopping scale and local-domain measurements."""
        dt = self.dt
        q = z[: self.nq]
        x = self.x.copy()
        x[self.tet.free] = z[self.nq :].reshape(-1, 3)
        poses = self.k.evaluate(q)
        _t, r, jt, jr, *_ = poses
        if np.any(abs(q - self.q)[self.k.joint_type[self.k.joint_type != 3] == 1] >= np.pi / 2) or np.any(
            np.einsum("bij,bij->b", r, self.r) <= 1
        ):
            raise ValueError("Rotation left local domain")
        dets = [
            np.linalg.det((x[ids[1:]] - x[ids[0]]).T @ pose)
            for ids, pose in zip(self.tet.indices, self.tet.poses, strict=True)
        ]
        if min(dets) < 0.2:
            raise ValueError("detF below frozen guard")
        force = self.k.body_mass[:, None] * self.gravity
        energy, qr, qh = self.k.inertia(q, self.pt, self.pr, dt, force)
        energy += np.sum(force * self.t)
        je, jres, jb = joint_terms(
            q,
            self.q,
            self.v,
            dt,
            self.params,
            self.armature,
            self.damping,
            np.zeros(self.nq),
            np.zeros(self.nq),
            self.joint_force,
        )
        energy += je.sum()
        qr += jres
        qh += np.diag(jb)
        te, xr, xh = self.tet.evaluate(x, self.x, self.xv, self.gravity, self.particle_force, dt)
        energy += te.sum()
        res = np.r_[qr, xr]
        h = np.zeros((self.n, self.n))
        h[: self.nq, : self.nq] = qh
        h[self.nq :, self.nq :] = xh
        b = h.copy()
        # Positive link rotation fit, without the reduced-coordinate geometric term.
        b[: self.nq, : self.nq] = np.diag(jb)
        for body in range(self.k.b):
            c = r[body] @ self.k.second[body] @ self.pr[body].T
            kr = (np.trace(c) * np.eye(3) - (c + c.T) / 2) / dt**2
            w, v = np.linalg.eigh(kr)
            kr = (v * np.maximum(w, 0)) @ v.T
            a = jr[body] @ r[body].T
            jw = np.stack((a[:, 2, 1], a[:, 0, 2], a[:, 1, 0]), axis=1)
            b[: self.nq, : self.nq] += float(self.k.body_mass[body]) * jt[body] @ jt[body].T / dt**2 + jw @ kr @ jw.T
        # Kim fitted material B omits the stress-geometric term, as declared.
        xb = self.tet.mass / dt**2
        for ids, pose, params in zip(self.tet.indices, self.tet.poses, self.tet.materials, strict=True):
            f = (x[ids[1:]] - x[ids[0]]).T @ pose
            a, c, d = f.T
            cof = np.column_stack((np.cross(c, d), np.cross(d, a), np.cross(a, c))).ravel(order="F")
            fitted = params[0] * np.eye(9) + (params[0] + params[1]) * np.outer(cof, cof)
            grads = np.vstack((-pose.sum(axis=0), pose))
            g = np.column_stack([np.kron(ga[:, None], np.eye(3)) for ga in grads])
            dofs = (3 * ids[:, None] + np.arange(3)).ravel()
            xb[np.ix_(dofs, dofs)] += g.T @ fitted @ g / (6 * np.linalg.det(pose))
        b[self.nq :, self.nq :] = xb[np.ix_(self.tet.dofs, self.tet.dofs)]
        loads = []
        particle_force = np.zeros_like(x)
        body_force = np.zeros_like(self.t)
        friction_work = 0.0
        slips = []
        gaps = []
        for key, (current, old) in enumerate(zip(self.frame(q, x), self.start, strict=True)):
            y, j, hy, soft, area, shape = current
            yn = old[0]
            phi, g, _ = geometry(y, self.kinds[shape], self.scales[shape])
            gap = phi - self.offset[shape]
            en, nload, _ = penalty(gap, 1e7 * area, self.width)
            if (en > 0 or nload > 0) and (np.linalg.norm(g) == 0 or (gap < 0 and soft @ g > 0)):
                raise ValueError("Contact outside alignment/normal domain")
            ce, cr, dr, fit = local(
                y,
                yn,
                self.kinds[shape],
                self.scales[shape],
                offset=self.offset[shape],
                stiffness=1e7 * area,
                width=self.width,
                mu=self.mu,
                a=0.01 * dt,
            )
            rotation = self.transforms[shape][:3, :3]
            body = self.shapes[shape]
            world = rotation if body < 0 else r[body] @ rotation
            reaction = world @ cr
            face = key // (3 * len(self.shapes))
            slot = (key // len(self.shapes)) % 3
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            particle_force[self.triangles[face]] -= bary[:, None] * reaction
            if body >= 0:
                body_force[body] += reaction
            friction_work += float((cr + nload * g) @ (y - yn))
            energy += ce.sum()
            res += j.T @ cr
            h += j.T @ dr @ j + np.einsum("a,aij->ij", cr, hy)
            b += j.T @ fit @ j
            n = g / np.linalg.norm(g)
            loads.append(nload)
            gaps.append(gap)
            slips.append(np.linalg.norm((np.eye(3) - np.outer(n, n)) @ (y - yn)))
        dynamic = self.armature.copy()
        for body in range(self.k.b):
            inertia = r[body] @ (np.trace(self.k.second[body]) * np.eye(3) - self.k.second[body]) @ r[body].T
            a = jr[body] @ r[body].T
            jw = np.stack((a[:, 2, 1], a[:, 0, 2], a[:, 1, 0]), axis=1)
            dynamic += self.k.body_mass[body] * np.sum(jt[body] ** 2, axis=1)
            dynamic += np.einsum("ni,ij,nj->n", jw, inertia, jw)
        dynamic = np.r_[dynamic, np.diag(self.tet.mass)[self.tet.dofs]] / dt**2
        scale = 1 / np.sqrt(np.maximum(np.diag(b), 1e-6 * dynamic))
        return (
            energy,
            res,
            h,
            scale,
            {
                "min_det_f": min(dets),
                "normal_load": sum(loads),
                "friction_work": friction_work,
                "force_balance": np.linalg.norm(particle_force.sum(axis=0) + body_force.sum(axis=0))
                / max(
                    np.linalg.norm(particle_force, axis=1).sum() + np.linalg.norm(body_force, axis=1).sum(),
                    1.2695789400826758e-5,
                ),
                "penetration": max(0.0, -min(gaps)),
                "slip": max(slips),
            },
        )

    def step(self, dt, joint_force):
        """Solve at most 100 Newton iterations and 32 backtracks; never fill failed steps."""
        self.freeze(dt, joint_force)
        z = np.r_[self.q + dt * self.v, (self.x + dt * self.xv)[self.tet.free].ravel()]
        for iteration in range(101):
            energy, res, h, scale, metrics = self.evaluate(z)
            ratios = []
            for part, absolute, relative in (
                (slice(None), 1e-7, 1e-6),
                (slice(0, self.nq), 2.5e-8, 1e-5),
                (slice(self.nq, None), 1e-7, 1e-6),
            ):
                size = len(res[part])
                ref = np.linalg.norm(scale[part] * self.r0[part]) / np.sqrt(size)
                goal = self.goal_scale * (absolute + relative * ref)
                ratios.append(np.linalg.norm(scale[part] * res[part]) / np.sqrt(size) / goal)
            if max(ratios) <= 1:
                break
            if iteration == 100:
                raise RuntimeError(f"Oracle Newton budget exhausted: {ratios}")
            direction = np.linalg.solve(h, -res)
            norm = np.linalg.norm(scale * res)
            for search in range(32):
                trial = z + (0.5**search) * direction
                try:
                    tr = self.evaluate(trial)[1]
                except ValueError:
                    continue
                if np.linalg.norm(scale * tr) < norm:
                    z = trial
                    break
            else:
                raise RuntimeError(f"Oracle line search exhausted: {ratios}")
        q = z[: self.nq]
        x = self.x.copy()
        x[self.tet.free] = z[self.nq :].reshape(-1, 3)
        t, r, *_ = self.k.evaluate(q)
        self.tv = (t - self.t) / dt
        self.rv = (r - self.r) @ r.transpose(0, 2, 1) / dt
        self.v = (q - self.q) / dt
        self.xv = (x - self.x) / dt
        self.q, self.x, self.t, self.r = q.copy(), x, t, r
        return dict(**metrics, iterations=iteration, ratios=ratios, energy=energy)
