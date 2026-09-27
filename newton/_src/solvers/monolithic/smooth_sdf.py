# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private FP64 smooth scalar-field candidates; no change to the default SDF path."""

import hashlib
import json

import numpy as np
import warp as wp


@wp.struct
class SmoothSDFData:
    kind: int
    lower: wp.vec3d
    upper: wp.vec3d
    inv_dx: wp.vec3d
    dims: wp.vec3i
    coefficients: wp.array[wp.float64]


@wp.func
def _hermite(i: int, t: wp.float64):
    result = wp.vec3d(wp.float64(0.0))
    if i == 0:
        result = wp.vec3d(
            wp.float64(2.0) * t * t * t - wp.float64(3.0) * t * t + wp.float64(1.0),
            wp.float64(6.0) * t * t - wp.float64(6.0) * t,
            wp.float64(12.0) * t - wp.float64(6.0),
        )
    elif i == 1:
        result = wp.vec3d(
            -wp.float64(2.0) * t * t * t + wp.float64(3.0) * t * t,
            -wp.float64(6.0) * t * t + wp.float64(6.0) * t,
            -wp.float64(12.0) * t + wp.float64(6.0),
        )
    elif i == 2:
        result = wp.vec3d(
            t * t * t - wp.float64(2.0) * t * t + t,
            wp.float64(3.0) * t * t - wp.float64(4.0) * t + wp.float64(1.0),
            wp.float64(6.0) * t - wp.float64(4.0),
        )
    else:
        result = wp.vec3d(
            t * t * t - t * t, wp.float64(3.0) * t * t - wp.float64(2.0) * t, wp.float64(6.0) * t - wp.float64(2.0)
        )
    return result


@wp.func
def _bspline(i: int, t: wp.float64):
    result = wp.vec3d(wp.float64(0.0))
    if i == 0:
        s = wp.float64(1.0) - t
        result = wp.vec3d(s * s * s / wp.float64(6.0), -s * s / wp.float64(2.0), s)
    elif i == 1:
        result = wp.vec3d(
            (wp.float64(3.0) * t * t * t - wp.float64(6.0) * t * t + wp.float64(4.0)) / wp.float64(6.0),
            wp.float64(1.5) * t * t - wp.float64(2.0) * t,
            wp.float64(3.0) * t - wp.float64(2.0),
        )
    elif i == 2:
        result = wp.vec3d(
            (-wp.float64(3.0) * t * t * t + wp.float64(3.0) * t * t + wp.float64(3.0) * t + wp.float64(1.0))
            / wp.float64(6.0),
            -wp.float64(1.5) * t * t + t + wp.float64(0.5),
            -wp.float64(3.0) * t + wp.float64(1.0),
        )
    else:
        result = wp.vec3d(t * t * t / wp.float64(6.0), t * t / wp.float64(2.0), t)
    return result


@wp.func
def smooth_query(sdf: SmoothSDFData, point: wp.vec3d):
    clamped, mask, t = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(1.0)), wp.vec3d(wp.float64(0.0))
    cell = wp.vec3i(0)
    for a in range(3):
        clamped[a] = wp.clamp(point[a], sdf.lower[a], sdf.upper[a])
        if point[a] < sdf.lower[a] or point[a] > sdf.upper[a]:
            mask[a] = wp.float64(0.0)
        f = (clamped[a] - sdf.lower[a]) * sdf.inv_dx[a]
        cell[a] = wp.clamp(int(wp.floor(f)), 0, sdf.dims[a] - 2)
        t[a] = f - wp.float64(cell[a])
    value, gradient, hessian = wp.float64(0.0), wp.vec3d(wp.float64(0.0)), wp.mat33d(wp.float64(0.0))
    for k in range(64):
        basis = wp.vec3i(k % 4, (k / 4) % 4, k / 16)
        index = cell
        w, d, dd = wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0)), wp.vec3d(wp.float64(0.0))
        jet = int(0)
        for a in range(3):
            basis_values = _bspline(basis[a], t[a])
            if sdf.kind == 1:
                basis_values = _hermite(basis[a], t[a])
                index[a] += basis[a] % 2
                if basis[a] >= 2:
                    jet += 1 << a
                    basis_values /= sdf.inv_dx[a]
            else:
                index[a] += basis[a]
            w[a] = basis_values[0]
            d[a] = basis_values[1] * sdf.inv_dx[a]
            dd[a] = basis_values[2] * sdf.inv_dx[a] * sdf.inv_dx[a]
        stride = sdf.dims + wp.vec3i(2)
        offset = (index[2] * stride[1] + index[1]) * stride[0] + index[0]
        if sdf.kind == 1:
            offset = ((index[2] * sdf.dims[1] + index[1]) * sdf.dims[0] + index[0]) * 8 + jet
        coefficient = sdf.coefficients[offset]
        value += coefficient * w[0] * w[1] * w[2]
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            gradient[a] += coefficient * d[a] * w[b] * w[c]
            hessian[a, a] += coefficient * dd[a] * w[b] * w[c]
            hessian[a, b] += coefficient * d[a] * d[b] * w[c]
            hessian[a, c] += coefficient * d[a] * d[c] * w[b]
    delta = point - clamped
    length = wp.length(delta)
    if length > wp.float64(0.0):
        unit = delta / length
        value += length
        gradient = wp.cw_mul(mask, gradient) + unit
        for a in range(3):
            for b in range(3):
                hessian[a, b] = mask[a] * mask[b] * hessian[a, b] - unit[a] * unit[b] / length
            hessian[a, a] += (wp.float64(1.0) - mask[a]) / length
    return value, gradient, hessian


@wp.kernel
def sample_smooth(sdf: SmoothSDFData, points: wp.array[wp.vec3d], output: wp.array2d[wp.float64]):
    i = wp.tid()
    value, gradient, hessian = smooth_query(sdf, points[i])
    output[i, 0] = value
    for a in range(3):
        output[i, 1 + a] = gradient[a]
        for b in range(3):
            output[i, 4 + 3 * a + b] = hessian[a, b]


class SmoothSDF:
    """Own one immutable candidate grid and its device coefficient storage."""

    def __init__(self, coefficients, lower, upper, dims, representation, *, device):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        dims, coefficients = np.asarray(dims, dtype=int), np.asarray(coefficients, dtype=np.float64)
        if representation not in ("c1-hermite", "c2-bspline"):
            raise ValueError("Unknown smooth SDF representation")
        if lower.shape != (3,) or upper.shape != (3,) or dims.shape != (3,) or np.any(dims < 3):
            raise ValueError("Invalid grid dimensions")
        if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
            raise ValueError("Invalid grid bounds")
        shape = (*tuple(dims[::-1]), 8) if representation == "c1-hermite" else (*tuple(dims[::-1] + 2), 1)
        if coefficients.shape != shape or not np.isfinite(coefficients).all():
            raise ValueError("Invalid grid coefficients")
        self.device = wp.get_device(device)
        self.representation = representation
        self.identity = {
            "representation": representation,
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "dims": dims.tolist(),
            "coefficients_sha256": hashlib.sha256(np.ascontiguousarray(coefficients).tobytes()).hexdigest(),
        }
        self.data = SmoothSDFData()
        self.data.kind = 1 if representation == "c1-hermite" else 2
        self.data.lower, self.data.upper = wp.vec3d(lower), wp.vec3d(upper)
        self.data.inv_dx, self.data.dims = wp.vec3d((dims - 1) / (upper - lower)), wp.vec3i(dims)
        self.data.coefficients = wp.array(coefficients.ravel(), dtype=wp.float64, device=self.device)


@wp.kernel
def _check_coefficients(actual: wp.array[wp.float64], frozen: wp.array[wp.float64], flag: wp.array[int]):
    i = wp.tid()
    if actual[i] != frozen[i]:
        wp.atomic_max(flag, 0, 1)


class SmoothSDFSet:
    """Bind diagnostic fields to original SDF indices and reject graph-invalidating edits."""

    def __init__(self, fields, *, source_identity):
        if not fields or any(field.device != fields[0].device for field in fields):
            raise ValueError("Require candidate fields on one device")
        self.fields = tuple(fields)
        self.identity = {"sources": source_identity, "fields": [field.identity for field in fields]}
        self.digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.table = wp.array([field.data for field in fields], dtype=SmoothSDFData, device=fields[0].device)
        self._table = self.table
        self._descriptors = self.table.numpy().copy()
        self._arrays = tuple(field.data.coefficients for field in fields)
        self._frozen = tuple(wp.clone(array) for array in self._arrays)
        self._flag = wp.zeros(1, dtype=int, device=fields[0].device)

    def validate_binding(self):
        """The graph must keep the original descriptors and coefficient addresses."""
        if self.table is not self._table or any(
            field.data.coefficients is not array for field, array in zip(self.fields, self._arrays, strict=True)
        ):
            raise ValueError("Smooth SDF buffer replaced; rebuild the solver")

    def validate_identity(self):
        """Compare device contents without copying the full coefficient volume to the host."""
        self.validate_binding()
        if not np.array_equal(self.table.numpy(), self._descriptors):
            raise ValueError("Smooth SDF descriptors changed; rebuild the solver")
        self._flag.zero_()
        for actual, frozen in zip(self._arrays, self._frozen, strict=True):
            wp.launch(_check_coefficients, actual.size, [actual, frozen, self._flag], device=actual.device)
        if self._flag.numpy()[0]:
            raise ValueError("Smooth SDF coefficients changed; rebuild the solver")
