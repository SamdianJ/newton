# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy homogeneous-transform derivatives for PR2 verification.

This module deliberately imports neither Warp nor production solver math.
"""

import numpy as np
from numpy.polynomial import Polynomial


def skew(v):
    """Return the matrix representing cross product with v."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def transform(value):
    """Decode a stored translation/quaternion using FP64 normalization."""
    value = np.asarray(value, dtype=np.float64)
    v = value[3:6] / np.linalg.norm(value[3:])
    w = value[6] / np.linalg.norm(value[3:])
    cross = skew(v)
    result = np.eye(4)
    result[:3, :3] += 2 * w * cross + 2 * cross @ cross
    result[:3, 3] = value[:3]
    return result


class Kinematics:
    """Differentiate homogeneous matrix products to second order independently."""

    def __init__(self, model):
        for name in (
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_q_start",
            "joint_axis",
            "joint_X_p",
            "joint_X_c",
            "body_com",
            "body_mass",
            "body_inertia",
        ):
            value = getattr(model, name)
            setattr(self, name, np.asarray(value.numpy() if hasattr(value, "numpy") else value).copy())
        self.parent_transform = np.array([transform(value) for value in self.joint_X_p])
        self.child_inverse = np.array([np.linalg.inv(transform(value)) for value in self.joint_X_c])
        self.axes = self.joint_axis.astype(float)
        self.axes /= np.linalg.norm(self.axes, axis=1)[:, None]
        self.n = len(self.joint_axis)
        self.b = len(self.body_mass)
        self.second = np.trace(self.body_inertia.astype(float), axis1=1, axis2=2)[:, None, None] * np.eye(
            3
        ) / 2 - self.body_inertia.astype(float)

    def evaluate(self, q):
        """Return COM, rotation, and their first and second coordinate derivatives."""
        poses = np.zeros((self.b, 4, 4))
        first = np.zeros((self.b, self.n, 4, 4))
        second = np.zeros((self.b, self.n, self.n, 4, 4))
        for j, (parent, child) in enumerate(zip(self.joint_parent, self.joint_child, strict=True)):
            motion = np.eye(4)
            dm, hm = np.zeros((4, 4)), np.zeros((4, 4))
            i = self.joint_q_start[j]
            # Newton's stable joint enum values: PRISMATIC=0, REVOLUTE=1, FIXED=3.
            moving = self.joint_type[j] != 3
            if moving:
                axis = self.axes[i]
                if self.joint_type[j] == 1:
                    cross = skew(axis)
                    motion[:3, :3] += np.sin(q[i]) * cross + (1 - np.cos(q[i])) * cross @ cross
                    dm[:3, :3] = cross @ motion[:3, :3]
                    hm[:3, :3] = cross @ dm[:3, :3]
                else:
                    motion[:3, 3] = axis * q[i]
                    dm[:3, 3] = axis
            left = self.parent_transform[j]
            right = self.child_inverse[j]
            local, dl, hl = left @ motion @ right, left @ dm @ right, left @ hm @ right
            p = np.eye(4) if parent < 0 else poses[parent]
            dp = np.zeros_like(first[child]) if parent < 0 else first[parent]
            hp = np.zeros_like(second[child]) if parent < 0 else second[parent]
            poses[child] = p @ local
            first[child] = dp @ local
            second[child] = hp @ local
            if moving:
                first[child, i] += p @ dl
                second[child, i, :] += dp @ dl
                second[child, :, i] += dp @ dl
                second[child, i, i] += p @ hl
        local_com = np.column_stack((self.body_com.astype(float), np.ones(self.b)))
        com = np.einsum("bij,bj->bi", poses, local_com)[:, :3]
        jt = np.einsum("bnij,bj->bni", first, local_com)[:, :, :3]
        ht = np.einsum("bnmij,bj->bnmi", second, local_com)[:, :, :, :3]
        return com, poses[:, :3, :3], jt, first[:, :, :3, :3], ht, second[:, :, :, :3, :3]

    def mass_and_bias(self, q, v):
        """Project Cartesian kinetic inertia and convective acceleration to q."""
        _, _, jt, jr, ht, hr = self.evaluate(q)
        mass = np.einsum("bni,bmi,b->nm", jt, jt, self.body_mass)
        mass += np.einsum("bnij,bmij->nm", jr @ self.second[:, None], jr)
        acceleration_t = np.einsum("bnmi,n,m->bi", ht, v, v)
        acceleration_r = np.einsum("bnmij,n,m->bij", hr, v, v)
        bias = np.einsum("bni,bi,b->n", jt, acceleration_t, self.body_mass)
        bias += np.einsum("bnij,bij->n", jr @ self.second[:, None], acceleration_r)
        return mass, bias

    def inertia(self, q, predicted_com, predicted_rotation, h, force):
        """Return exact incremental link energy, gradient and reduced Hessian."""
        t, r, jt, jr, ht, hr = self.evaluate(q)
        e, gradient, hessian = 0.0, np.zeros(self.n), np.zeros((self.n, self.n))
        for b in range(self.b):
            dt, dr = t[b] - predicted_com[b], r[b] - predicted_rotation[b]
            mass, s = float(self.body_mass[b]), self.second[b]
            ft = mass * dt / h**2 - force[b]
            fr = dr @ s / h**2
            e += mass * dt @ dt / (2 * h**2) + np.sum(dr * (dr @ s)) / (2 * h**2) - force[b] @ t[b]
            gradient += jt[b] @ ft + np.einsum("nij,ij->n", jr[b], fr)
            hessian += mass * jt[b] @ jt[b].T / h**2 + np.einsum("nij,mij->nm", jr[b] @ s, jr[b]) / h**2
            hessian += np.einsum("nmi,i->nm", ht[b], ft) + np.einsum("nmij,ij->nm", hr[b], fr)
        return e, gradient, hessian


def joint_terms(q, qn, vn, h, parameters, armature, damping, qt, vt, force):
    """Integrate each scalar force law independently, including limit damping."""

    e, r, b = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
    for i in range(len(q)):
        delta, v = q[i] - qn[i], (q[i] - qn[i]) / h
        kp, kd, effort, lo, hi, ke, kl, width, friction, eps = parameters[i]
        e[i] = armature[i] * (delta - h * vn[i]) ** 2 / (2 * h * h) + damping[i] * delta**2 / (2 * h) - force[i] * delta
        r[i] = armature[i] * (delta - h * vn[i]) / (h * h) + damping[i] * v - force[i]
        b[i] = armature[i] / (h * h) + damping[i] / h
        a = kp + kd / h
        if a:
            u = kp * (q[i] - qt[i]) + kd * (v - vt[i])
            e[i] += (u * u / 2 if abs(u) < effort else effort * (abs(u) - effort / 2)) / a
            r[i] += np.clip(u, -effort, effort)
            b[i] += a if abs(u) < effort else 0
        for sign, bound in ((1, hi), (-1, lo)):
            d, dn = sign * (q[i] - bound), sign * (qn[i] - bound)
            p = max(d, 0)
            z = np.clip(d / width, 0, 1)
            activation = z * z * (3 - 2 * z)
            derivative = 6 * z * (1 - z) / width
            e[i] += ke * p * p / 2
            if d > max(dn, 0):
                lower = max(dn, 0)
                if lower < width:
                    polynomial = (Polynomial([0, 0, 3 / width**2, -2 / width**3]) * Polynomial([-dn, 1])).integ()
                    e[i] += kl / h * (polynomial(min(d, width)) - polynomial(lower))
                if d > width:
                    lower = max(lower, width)
                    e[i] += kl / h * ((d * d - lower * lower) / 2 - dn * (d - lower))
            r[i] += sign * (ke * p + kl / h * activation * max(d - dn, 0))
            b[i] += ke * (d > 0) + kl / h * (derivative * max(d - dn, 0) + activation * (d > dn))
        denom = np.hypot(v, eps)
        e[i] += h * friction * v * v / (denom + eps)
        r[i] += friction * v / denom
        b[i] += friction * eps * eps / (h * denom**3)
    return e, r, b


class DiscreteOracle:
    """Solve the same incremental equations with independent FP64 exact Hessians."""

    def __init__(self, model, parameters, q, v, h, *, goal_scale=1.0):
        if isinstance(goal_scale, bool) or goal_scale not in (1.0, 0.1):
            raise ValueError("Oracle goal scale must be 1 or 0.1")
        self.goal_scale = goal_scale
        self.kinematics = Kinematics(model)
        self.parameters = np.asarray(parameters, dtype=float).copy()
        self.armature = model.joint_armature.numpy().astype(float)
        self.damping = model.joint_damping.numpy().astype(float)
        self.q, self.v = np.asarray(q, dtype=float).copy(), np.asarray(v, dtype=float).copy()
        self.t, self.r, jt, jr, _, _ = self.kinematics.evaluate(self.q)
        self.linear_velocity = np.einsum("bni,n->bi", jt, self.v)
        rotation_rate = np.einsum("bnij,n->bij", jr, self.v)
        a = rotation_rate @ self.r.transpose(0, 2, 1)
        w = np.stack((a[:, 2, 1], a[:, 0, 2], a[:, 1, 0]), axis=1)
        if np.any(h * np.linalg.norm(w, axis=1) >= 1):
            raise ValueError("Oracle initialization outside local domain")
        sym = np.array([h * skew(value) @ skew(value) / (1 + np.sqrt(1 - h * h * np.dot(value, value))) for value in w])
        self.rotation_velocity = a + sym

    def step(self, h, qt, vt, force, joint_force):
        predicted_t = self.t + h * self.linear_velocity
        predicted_r = self.r + h * self.rotation_velocity @ self.r
        qn, vn = self.q.copy(), self.v.copy()
        q = qn + h * vn

        def evaluate(candidate):
            e, r, b = self.kinematics.inertia(candidate, predicted_t, predicted_r, h, force)
            je, jr, jb = joint_terms(
                candidate, qn, vn, h, self.parameters, self.armature, self.damping, qt, vt, joint_force
            )
            return e + np.sum(force * self.t) + je.sum(), r + jr, b + np.diag(jb)

        initial = None
        converged = False
        for iteration in range(101):
            _, r, b = evaluate(q)
            scale = 1 / np.sqrt(np.maximum(np.diag(b), 1e-6))
            if initial is None:
                initial = r.copy()
            goal = (2.5e-7 + 1e-4 * np.linalg.norm(scale * initial) / np.sqrt(len(q))) / 1000 * self.goal_scale
            norm = np.linalg.norm(scale * r) / np.sqrt(len(q))
            if norm <= goal:
                converged = True
                break
            if iteration == 100:
                break
            direction = np.linalg.solve(b, -r)
            accepted = False
            for search in range(32):
                candidate = q + (0.5**search) * direction
                _, trial_r, _ = evaluate(candidate)
                if np.linalg.norm(scale * trial_r) < np.linalg.norm(scale * r):
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break
        if not converged:
            raise RuntimeError(f"ORACLE_UNCERTIFIED residual={norm}, goal={goal}, iterations={iteration}")
        t, r, *_ = self.kinematics.evaluate(q)
        revolute = self.kinematics.joint_q_start[:-1][self.kinematics.joint_type == 1]
        if np.any(abs(q[revolute] - qn[revolute]) >= np.pi / 2) or np.any(np.einsum("bij,bij->b", r, self.r) <= 1.0):
            raise RuntimeError("ORACLE_UNCERTIFIED: root outside the local rotation domain")
        self.linear_velocity = (t - self.t) / h
        self.rotation_velocity = (r - self.r) @ r.transpose(0, 2, 1) / h
        self.q, self.v = q, (q - qn) / h
        self.t, self.r = t, r
        return {
            "q": self.q.copy(),
            "qd": self.v.copy(),
            "com": t.copy(),
            "rotation": r.copy(),
            "residual_rms": float(norm),
            "goal": float(goal),
            "iterations": iteration,
        }
