# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Native shell E/r with full projected membrane and curvature-material tangents.

The six-node curvature convention follows Mochi's fem_shell element. Missing
neighbors are eliminated by affine extrapolation before differentiation.
"""

from functools import lru_cache

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


@lru_cache(maxsize=2)
def _kernels(dtype):
    vec = wp.types.vector(3, dtype)
    points_type = wp.types.matrix((6, 3), dtype)
    normals_type = wp.types.matrix((4, 3), dtype)
    jac_type = wp.types.matrix((3, 18), dtype)
    mat6 = wp.types.matrix((6, 6), dtype)
    mat2 = wp.types.matrix((2, 2), dtype)
    mat3 = wp.types.matrix((3, 3), dtype)
    vec6 = wp.types.vector(6, dtype)
    map_type = wp.types.matrix((6, 9), dtype)

    @wp.func
    def curvature(p: points_type, dp: points_type):
        normals = normals_type()
        derivatives = normals_type()
        for face in range(4):
            a, b, c = int(0), int(1), int(2)
            if face > 0:
                opposite = face - 1
                a, b, c = 3 + opposite, (opposite + 2) % 3, (opposite + 1) % 3
            u, v = p[b] - p[a], p[c] - p[a]
            du, dv = dp[b] - dp[a], dp[c] - dp[a]
            cross = wp.cross(u, v)
            length = wp.length(cross)
            normal = cross / length
            dn = wp.cross(du, v) + wp.cross(u, dv)
            normals[face] = normal
            derivatives[face] = (dn - normal * wp.dot(normal, dn)) / length
        mean = mat3()
        dmean = mat3()
        for i in range(3):
            n = (normals[0] + normals[i + 1]) * dtype(0.5)
            dn = (derivatives[0] + derivatives[i + 1]) * dtype(0.5)
            norm = wp.dot(n, n)
            mean[i] = n / norm
            dmean[i] = dn / norm - n * (dtype(2.0) * wp.dot(n, dn) / (norm * norm))
        e, f = p[1] - p[0], p[0] - p[2]
        de, df = dp[1] - dp[0], dp[0] - dp[2]
        value = vec(wp.dot(mean[1] - mean[0], e), wp.dot(mean[0] - mean[2], f), wp.dot(mean[0], f)) * dtype(2.0)
        differential = vec(
            wp.dot(dmean[1] - dmean[0], e) + wp.dot(mean[1] - mean[0], de),
            wp.dot(dmean[0] - dmean[2], f) + wp.dot(mean[0] - mean[2], df),
            wp.dot(dmean[0], f) + wp.dot(mean[0], df),
        ) * dtype(2.0)
        return value, differential

    @wp.func
    def absolute_dot(a: vec, b: vec):
        return wp.abs(a[0] * b[0]) + wp.abs(a[1] * b[1]) + wp.abs(a[2] * b[2])

    @wp.func
    def curvature_magnitude(p: points_type):
        """Bound each curvature component's operands; it rounds to a few eps times this.

        Componentwise products keep the bound at zero where an axis-aligned flat
        stencil evaluates exactly.
        """
        normals = normals_type()
        for face in range(4):
            a, b, c = int(0), int(1), int(2)
            if face > 0:
                opposite = face - 1
                a, b, c = 3 + opposite, (opposite + 2) % 3, (opposite + 1) % 3
            cross = wp.cross(p[b] - p[a], p[c] - p[a])
            normals[face] = cross / wp.length(cross)
        mean = mat3()
        for i in range(3):
            n = (normals[0] + normals[i + 1]) * dtype(0.5)
            m = n / wp.dot(n, n)
            mean[i] = vec(wp.abs(m[0]), wp.abs(m[1]), wp.abs(m[2]))
        e, f = p[1] - p[0], p[0] - p[2]
        return vec(
            absolute_dot(mean[1] + mean[0], e), absolute_dot(mean[0] + mean[2], f), absolute_dot(mean[0], f)
        ) * dtype(2.0)

    @wp.func
    def strain(b: vec, transform: mat2):
        tensor = mat2(b[0], b[2], b[2], b[1])
        mapped = wp.transpose(transform) * tensor * transform
        return vec(mapped[0, 0], mapped[1, 1], wp.sqrt(dtype(2.0)) * mapped[0, 1])

    @wp.kernel
    def evaluate(
        origin: wp.array[vec],
        step: wp.array[vec],
        stencils: wp.array2d[int],
        transforms: wp.array[mat2],
        reference_curvature: wp.array[vec],
        areas: wp.array2d[dtype],
        materials: wp.array2d[dtype],
        build_tangent: bool,
        energy: wp.array2d[dtype],
        residual: wp.array2d[dtype],
        rounding: wp.array2d[dtype],
        tangent: wp.array3d[dtype],
        flags: wp.array[int],
    ):
        face = wp.tid()
        # The energy is translation invariant; place the stencil relative to its
        # first node so a small step is not added to large absolute coordinates.
        first = stencils[face, 0]
        p = points_type()
        for i in range(6):
            node = stencils[face, i]
            if node >= 0:
                p[i] = (origin[node] - origin[first]) + (step[node] - step[first])
        for i in range(3):
            if stencils[face, i + 3] < 0:
                p[i + 3] = p[(i + 1) % 3] + p[(i + 2) % 3] - p[i]
        normals = normals_type()
        for k in range(4):
            a, b, c = int(0), int(1), int(2)
            if k > 0:
                a, b, c = k + 2, (k + 1) % 3, k % 3
            u, v = p[b] - p[a], p[c] - p[a]
            normal = wp.cross(u, v)
            length = wp.length(normal)
            bound = wp.max(
                wp.length(u) * wp.length(v), wp.max(wp.length(u) * wp.length(v - u), wp.length(v) * wp.length(v - u))
            )
            if not wp.isfinite(length) or length <= dtype(1e-8) * wp.max(areas[face, k], bound):
                wp.atomic_max(flags, 0, 1)
                return
            normals[k] = normal / length
            if k > 0 and wp.length_sq(normals[0] + normals[k]) <= dtype(1e-8):
                wp.atomic_max(flags, 0, 1)
                return
        transform = transforms[face]
        # Membrane frame f, g at the step start plus its exact step change. The
        # metric C - I is formed from the step-start metric, fixed within the
        # step, plus terms of the size of the change, so near rest it is never
        # the difference of unit-size products.
        edge_n, other_n = origin[stencils[face, 1]] - origin[first], origin[stencils[face, 2]] - origin[first]
        edge_d, other_d = step[stencils[face, 1]] - step[first], step[stencils[face, 2]] - step[first]
        fn = edge_n * transform[0, 0] + other_n * transform[1, 0]
        gn = edge_n * transform[0, 1] + other_n * transform[1, 1]
        df_step = edge_d * transform[0, 0] + other_d * transform[1, 0]
        dg_step = edge_d * transform[0, 1] + other_d * transform[1, 1]
        f, g = fn + df_step, gn + dg_step
        e00_n, e11_n, c01_n = wp.dot(fn, fn) - dtype(1.0), wp.dot(gn, gn) - dtype(1.0), wp.dot(fn, gn)
        de00 = dtype(2.0) * wp.dot(fn, df_step) + wp.dot(df_step, df_step)
        de11 = dtype(2.0) * wp.dot(gn, dg_step) + wp.dot(dg_step, dg_step)
        dc01 = wp.dot(fn, dg_step) + wp.dot(df_step, g)
        e00, e11, c01 = e00_n + de00, e11_n + de11, c01_n + dc01
        c00, c11 = dtype(1.0) + e00, dtype(1.0) + e11
        # j^2 - 1 = det(C) - 1 and j - 1 = (j^2 - 1) / (j + 1). Near area preservation
        # det(C) - 1 is a small sum of strain-sized terms: form it from its step-start
        # value, fixed within the step, plus the step's terms, so iterates differ by
        # rounding of the step's size rather than of the strain's.
        j2m1_n = e00_n + e11_n + e00_n * e11_n - c01_n * c01_n
        j2m1 = j2m1_n + (de00 + de11 + de00 * e11_n + e00_n * de11 + de00 * de11 - (dtype(2.0) * c01_n + dc01) * dc01)
        # Operand sizes of the step-dependent parts, which bound their rounding.
        s00 = dtype(2.0) * absolute_dot(fn, df_step) + absolute_dot(df_step, df_step)
        s11 = dtype(2.0) * absolute_dot(gn, dg_step) + absolute_dot(dg_step, dg_step)
        s01 = absolute_dot(fn, dg_step) + absolute_dot(df_step, g)
        sj = s00 + s11 + s00 * wp.abs(e11) + wp.abs(e00) * s11 + dtype(2.0) * wp.abs(c01) * s01
        j = wp.sqrt(dtype(1.0) + j2m1)
        jm1 = j2m1 / (j + dtype(1.0))
        cross = wp.cross(f, g)
        n = cross / wp.length(cross)
        lam, mu = materials[face, 0], materials[face, 1]
        alpha, beta = materials[face, 2], materials[face, 3]
        area = areas[face, 0] * dtype(0.5)
        # W = mu (c00 + c11 - 2 j) / 2 + (lambda + mu) (j - 1)^2 / 2, where
        # c00 + c11 - 2 j = ((c00 - c11)^2 + 4 c01^2) / (c00 + c11 + 2 j).
        distortion = ((e00 - e11) * (e00 - e11) + dtype(4.0) * c01 * c01) / (c00 + c11 + dtype(2.0) * j)
        energy[face, 0] = area * dtype(0.5) * (mu * distortion + (lam + mu) * jm1 * jm1)
        # dW/dC from small quantities: 1 - c11 / j = (jm1 - e11) / j.
        w00 = dtype(0.5) * (mu * (jm1 - e11) + (lam + mu) * jm1 * c11) / j
        w11 = dtype(0.5) * (mu * (jm1 - e00) + (lam + mu) * jm1 * c00) / j
        w01 = (mu - (lam + mu) * jm1) * c01 / j
        pressure = (lam + mu) * jm1 - mu
        # Membrane stress rounding from those step operands; j - 1 rounds with sj / (j + 1).
        sm = sj / (j + dtype(1.0))
        n00 = dtype(0.5) * (wp.abs(mu) * (sm + s11) + wp.abs(lam + mu) * (sm * wp.abs(c11) + wp.abs(jm1) * s11)) / j
        n11 = dtype(0.5) * (wp.abs(mu) * (sm + s00) + wp.abs(lam + mu) * (sm * wp.abs(c00) + wp.abs(jm1) * s00)) / j
        n01 = (wp.abs(lam + mu) * sm * wp.abs(c01) + wp.abs(mu - (lam + mu) * jm1) * s01) / j
        # Exact six-dimensional material derivatives, independent of the nodal map.
        dn = points_type()
        dj = vec6()
        for a in range(6):
            df, dg = vec(), vec()
            if a < 3:
                df[a] = dtype(1.0)
            else:
                dg[a - 3] = dtype(1.0)
            dn[a] = wp.cross(df, g) + wp.cross(f, dg)
            dj[a] = wp.dot(n, dn[a])
        h = mat6()
        vectors = mat6()
        if build_tangent:
            for a in range(6):
                vectors[a, a] = dtype(1.0)
                for b in range(6):
                    df, dg, ef, eg = vec(), vec(), vec(), vec()
                    if a < 3:
                        df[a] = dtype(1.0)
                    else:
                        dg[a - 3] = dtype(1.0)
                    if b < 3:
                        ef[b] = dtype(1.0)
                    else:
                        eg[b - 3] = dtype(1.0)
                    d2j = (wp.dot(dn[a], dn[b]) - dj[a] * dj[b]) / j + wp.dot(n, wp.cross(df, eg) + wp.cross(ef, dg))
                    h[a, b] = (lam + mu) * dj[a] * dj[b] + pressure * d2j
                    if a == b:
                        h[a, b] += mu
            # Cyclic Jacobi diagonalizes the entire symmetric material Hessian.
            for _sweep in range(16):
                for a in range(5):
                    for b in range(a + 1, 6):
                        off = h[a, b]
                        if wp.abs(off) > dtype(1e-12) * wp.max(mu, wp.max(wp.abs(h[a, a]), wp.abs(h[b, b]))):
                            tau = (h[b, b] - h[a, a]) / (dtype(2.0) * off)
                            t = dtype(1.0) / (wp.abs(tau) + wp.sqrt(dtype(1.0) + tau * tau))
                            if tau < dtype(0.0):
                                t = -t
                            cosine = dtype(1.0) / wp.sqrt(dtype(1.0) + t * t)
                            sine = t * cosine
                            h[a, a] -= t * off
                            h[b, b] += t * off
                            h[a, b] = dtype(0.0)
                            h[b, a] = dtype(0.0)
                            for k in range(6):
                                if k != a and k != b:
                                    ha, hb = h[k, a], h[k, b]
                                    h[k, a] = cosine * ha - sine * hb
                                    h[a, k] = h[k, a]
                                    h[k, b] = sine * ha + cosine * hb
                                    h[b, k] = h[k, b]
                                va, vb = vectors[k, a], vectors[k, b]
                                vectors[k, a] = cosine * va - sine * vb
                                vectors[k, b] = sine * va + cosine * vb
            diagonal_scale = mu
            off_max = dtype(0.0)
            for a in range(6):
                diagonal_scale = wp.max(diagonal_scale, wp.abs(h[a, a]))
                for b in range(a + 1, 6):
                    off_max = wp.max(off_max, wp.abs(h[a, b]))
            if not wp.isfinite(diagonal_scale) or off_max > dtype(1e-6) * diagonal_scale:
                wp.atomic_max(flags, 0, 2)
                return
        curvature_value, _unused = curvature(p, points_type())
        s = strain(curvature_value - reference_curvature[face], transform)
        stress = beta * s + alpha * (s[0] + s[1]) * vec(dtype(1.0), dtype(1.0), dtype(0.0))
        energy[face, 1] = area * dtype(0.5) * wp.dot(s, stress)
        # Near rest the curvature difference cancels: its rounding follows the operand
        # sizes, not the small difference, so bound the bending stress from them.
        operands = curvature_magnitude(p)
        magnitude = mat2(operands[0], operands[2], operands[2], operands[1])
        absolute = mat2(
            wp.abs(transform[0, 0]), wp.abs(transform[0, 1]), wp.abs(transform[1, 0]), wp.abs(transform[1, 1])
        )
        mapped = wp.transpose(absolute) * magnitude * absolute
        s_bound = vec(mapped[0, 0], mapped[1, 1], wp.sqrt(dtype(2.0)) * mapped[0, 1])
        stress_bound = wp.abs(beta) * s_bound + wp.abs(alpha) * (s_bound[0] + s_bound[1]) * vec(
            dtype(1.0), dtype(1.0), dtype(0.0)
        )
        jac = jac_type()
        mapping = map_type()
        for column in range(18):
            dp = points_type()
            vertex, axis = column // 3, column % 3
            if stencils[face, vertex] >= 0:
                dp[vertex, axis] = dtype(1.0)
            for i in range(3):
                if stencils[face, i + 3] < 0:
                    dp[i + 3] = dp[(i + 1) % 3] + dp[(i + 2) % 3] - dp[i]
            _unused_b, db = curvature(p, dp)
            ds = strain(db, transform)
            for k in range(3):
                jac[k, column] = ds[k]
            value = wp.dot(ds, stress)
            bending_bound = wp.abs(ds[0]) * stress_bound[0] + wp.abs(ds[1]) * stress_bound[1]
            bending_bound += wp.abs(ds[2]) * stress_bound[2]
            if column < 9:
                de, dg_edge = dp[1] - dp[0], dp[2] - dp[0]
                df = de * transform[0, 0] + dg_edge * transform[1, 0]
                dg = de * transform[0, 1] + dg_edge * transform[1, 1]
                value += dtype(2.0) * (w00 * wp.dot(f, df) + w11 * wp.dot(g, dg)) + w01 * (
                    wp.dot(f, dg) + wp.dot(g, df)
                )
                bending_bound += dtype(2.0) * (n00 * absolute_dot(f, df) + n11 * absolute_dot(g, dg)) + n01 * (
                    absolute_dot(f, dg) + absolute_dot(g, df)
                )
                for k in range(3):
                    mapping[k, column] = df[k]
                    mapping[k + 3, column] = dg[k]
            residual[face, column] = area * value
            rounding[face, column] = area * (wp.abs(value) + bending_bound)
        if build_tangent:
            for a in range(18):
                for bcol in range(a, 18):
                    value = alpha * (jac[0, a] + jac[1, a]) * (jac[0, bcol] + jac[1, bcol])
                    for k in range(3):
                        value += beta * jac[k, a] * jac[k, bcol]
                    if a < 9 and bcol < 9:
                        for k in range(6):
                            va, vb = dtype(0.0), dtype(0.0)
                            for row in range(6):
                                va += vectors[row, k] * mapping[row, a]
                                vb += vectors[row, k] * mapping[row, bcol]
                            value += wp.max(h[k, k], dtype(0.0)) * va * vb
                    tangent[face, a, bcol] = area * value
                    tangent[face, bcol, a] = area * value

    return evaluate


class ClothObjective:
    """Own per-face native residuals and full shell tangents."""

    def __init__(self, positions, stencils, materials, *, dtype=wp.float64, device):
        self.dtype, self.device = dtype, wp.get_device(device)
        self.vector = wp.vec3d if dtype == wp.float64 else wp.vec3
        stencils = np.asarray(stencils, dtype=np.int32)
        transforms, curvature, areas = [], [], []
        for stencil in stencils:
            p = np.asarray(positions, dtype=float)[np.maximum(stencil, 0)].copy()
            for i in range(3):
                if stencil[3 + i] < 0:
                    p[3 + i] = p[(i + 1) % 3] + p[(i + 2) % 3] - p[i]
            triangles = p[[[0, 1, 2], [3, 2, 1], [4, 0, 2], [5, 1, 0]]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            sizes = np.linalg.norm(normals, axis=1)
            normals /= sizes[:, None]
            means = (normals[0] + normals[1:]) / 2
            means /= np.sum(means * means, axis=1)[:, None]
            e, f = p[1] - p[0], p[0] - p[2]
            curvature.append(2 * np.array([(means[1] - means[0]) @ e, (means[0] - means[2]) @ f, means[0] @ f]))
            edges = (p[1:3] - p[0]).T
            transforms.append(np.linalg.cholesky(np.linalg.inv(edges.T @ edges)))
            areas.append(sizes)
        count = len(stencils)
        self.stencils = wp.array(stencils, dtype=int, device=device)
        self.transforms = wp.array(np.asarray(transforms), dtype=wp.types.matrix((2, 2), dtype), device=device)
        self.curvature0 = wp.array(curvature, dtype=self.vector, device=device)
        self.areas = wp.array(areas, dtype=dtype, device=device)
        self.materials = wp.array(materials, dtype=dtype, device=device)
        self.energy = wp.zeros((count, 2), dtype=dtype, device=device)
        self.residual = wp.zeros((count, 18), dtype=dtype, device=device)
        # Per-entry operand magnitude whose eps multiple bounds the residual rounding.
        self.rounding = wp.zeros((count, 18), dtype=dtype, device=device)
        self.tangent = wp.zeros((count, 18, 18), dtype=dtype, device=device)
        self.flags = wp.zeros(1, dtype=int, device=device)
        self.kernel = _kernels(dtype)
        self._no_step = None

    def evaluate(self, positions, *, build_tangent=True, step=None):
        """Overwrite element data without changing any reference quantity.

        With ``step``, the configuration is ``positions + step``: the step-start
        positions and the exact step displacement are combined only in
        stencil-relative differences.
        """
        if step is None:
            if self._no_step is None or self._no_step.size != positions.size:
                self._no_step = wp.zeros_like(positions)
            step = self._no_step
        self.flags.zero_()
        wp.launch(
            self.kernel,
            self.stencils.shape[0],
            [
                positions,
                step,
                self.stencils,
                self.transforms,
                self.curvature0,
                self.areas,
                self.materials,
                build_tangent,
                self.energy,
                self.residual,
                self.rounding,
                self.tangent,
                self.flags,
            ],
            device=self.device,
        )

    def evaluate_host(self, positions, step=None):
        """Return local element E/r/B for mathematical validation only."""
        if step is not None:
            step = wp.array(step, dtype=self.vector, device=self.device)
        self.evaluate(wp.array(positions, dtype=self.vector, device=self.device), step=step)
        arrays = self.energy.numpy(), self.residual.numpy(), self.tangent.numpy()
        if self.flags.numpy()[0] or not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid shell geometry, projection or element values")
        return arrays
