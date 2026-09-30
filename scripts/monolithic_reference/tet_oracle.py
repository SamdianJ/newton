# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy FP64 P3 objective, gradient and exact Hessian oracle."""

import numpy as np


def material(f, mu, lam, smith):
    """Return density, first Piola stress and column-major exact material Hessian."""
    a, b, c = f.T
    cof = np.column_stack((np.cross(b, c), np.cross(c, a), np.cross(a, b)))
    dc = np.empty((9, 9))
    for j in range(9):
        df = np.zeros((3, 3))
        df[j % 3, j // 3] = 1
        da, db, dd = df.T
        dc[:, j] = np.column_stack(
            (
                np.cross(db, c) + np.cross(b, dd),
                np.cross(dd, a) + np.cross(c, da),
                np.cross(da, b) + np.cross(a, db),
            )
        ).ravel(order="F")
    d, ic = np.linalg.det(f) - 1, np.sum(f * f)
    cv, fv = cof.ravel(order="F"), f.ravel(order="F")
    if smith:
        mh, lh = 4 * mu / 3, lam + 5 * mu / 6
        av, s = mh * (1 - 1 / (ic + 1)), lh * d - mu
        energy = mh / 2 * (ic - 3 - np.log((ic + 1) / 4)) + lh / 2 * d * d - mu * d
        hessian = av * np.eye(9) + 2 * mh / (ic + 1) ** 2 * np.outer(fv, fv) + lh * np.outer(cv, cv) + s * dc
    else:
        av, s = mu, (lam + mu) * d - mu
        energy = mu / 2 * (ic - 3) - mu * d + (lam + mu) / 2 * d * d
        hessian = mu * np.eye(9) + (lam + mu) * np.outer(cv, cv) + s * dc
    return energy, av * f + s * cof, hessian


class TetOracle:
    """Capture exactly the model data supplied to the production step."""

    def __init__(self, model, physics):
        self.indices = model.tet_indices.numpy()
        self.poses = physics.poses.numpy().astype(float)
        self.materials = physics.materials.numpy().astype(float)
        self.smith = physics.material_model == "smith_log_stabilized"
        self.free = np.flatnonzero(model.particle_inv_mass.numpy() > 0)
        self.dofs = (3 * self.free[:, None] + np.arange(3)).ravel()
        n = model.particle_count
        mass = np.zeros((n, n))
        if physics.mass_mode == "consistent":
            for ids, coefficient in zip(self.indices, physics.coefficients.numpy(), strict=True):
                mass[np.ix_(ids, ids)] += float(coefficient) * (np.ones((4, 4)) + np.eye(4))
        else:
            mass = np.diag(physics.mass.numpy().astype(float))
        self.mass = np.kron(mass, np.eye(3))
        self.row_mass = physics.row_mass.numpy().astype(float)

    def evaluate(self, x, xn, vn, gravity, forces, dt):
        """Evaluate the complete reduced objective without using production evaluators."""
        x, xn, vn, forces = (np.asarray(v, dtype=float) for v in (x, xn, vn, forces))
        delta = (x - xn).ravel()
        d = delta - dt * vn.ravel()
        grav = (self.row_mass[:, None] * np.asarray(gravity)).ravel()
        r = self.mass @ d / dt**2 - grav - forces.ravel()
        h = self.mass / dt**2
        parts = [0.5 * d @ self.mass @ d / dt**2, 0.0, -grav @ delta, -forces.ravel() @ delta]
        for ids, pose, params in zip(self.indices, self.poses, self.materials, strict=True):
            f = (x[ids[1:]] - x[ids[0]]).T @ pose
            volume = 1 / (6 * np.linalg.det(pose))
            e, stress, raw = material(f, params[0], params[1], self.smith)
            gradients = np.vstack((-pose.sum(axis=0), pose))
            g = np.column_stack([np.kron(ga[:, None], np.eye(3)) for ga in gradients])
            dofs = (3 * ids[:, None] + np.arange(3)).ravel()
            parts[1] += volume * e
            r[dofs] += volume * g.T @ stress.ravel(order="F")
            h[np.ix_(dofs, dofs)] += volume * g.T @ raw @ g
        return np.asarray(parts), r[self.dofs], h[np.ix_(self.dofs, self.dofs)]
