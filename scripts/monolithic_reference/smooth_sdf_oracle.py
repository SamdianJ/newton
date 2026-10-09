# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent CPU polynomials and SciPy natural splines for candidate qualification."""

import itertools

import numpy as np


class NaturalSpline:
    """Prepare independent natural-spline node jets for repeated offline queries.

    Tensor natural splines commute across axes. Their eight mixed first-order
    node derivatives determine each cell's tricubic Hermite polynomial. No
    native B-spline coefficients or point evaluator is used here.
    """

    def __init__(self, values, lower, upper):
        from scipy.interpolate import CubicSpline

        self.values = np.array(values, dtype=float, copy=True)
        self.lower, self.upper = np.array(lower, dtype=float, copy=True), np.array(upper, dtype=float, copy=True)
        if (
            self.values.ndim != 3
            or min(self.values.shape) < 2
            or self.lower.shape != (3,)
            or self.upper.shape != (3,)
            or not all(np.isfinite(a).all() for a in (self.values, self.lower, self.upper))
            or np.any(self.upper <= self.lower)
        ):
            raise ValueError("Invalid natural-spline oracle field")
        self.dims = np.array(self.values.shape[::-1])
        self.axes = tuple(np.linspace(self.lower[a], self.upper[a], self.dims[a]) for a in range(3))
        jets = {0: self.values}
        for mask in range(8):
            if mask:
                axis = (mask & -mask).bit_length() - 1
                source = jets[mask ^ (1 << axis)]
                jets[mask] = CubicSpline(self.axes[axis], source, axis=2 - axis, bc_type="natural")(
                    self.axes[axis], nu=1
                )
        self.jets = tuple(jets[m] for m in range(8))
        for array in (*self.jets, *self.axes, self.lower, self.upper, self.dims):
            array.setflags(write=False)
        self.payload_bytes = sum(a.nbytes for a in (*self.jets, *self.axes, self.lower, self.upper, self.dims))

    def evaluate(self, points):
        """Return physical phi/g/H with the existing clamp and Euclidean extension."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Invalid natural-spline oracle points")
        basis = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]], dtype=float)
        orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        orders += [tuple(int(k == a) + int(k == b) for k in range(3)) for a in range(3) for b in range(3)]
        result = np.empty((len(points), 13))
        for i, point in enumerate(points):
            p = np.clip(point, self.lower, self.upper)
            cell = np.clip(
                [np.searchsorted(a, p[k], side="right") - 1 for k, a in enumerate(self.axes)], 0, self.dims - 2
            )
            # Use the actual linspace intervals, including their rounding, as SciPy does.
            width = np.array([a[c + 1] - a[c] for a, c in zip(self.axes, cell, strict=True)])
            t = (p - [a[c] for a, c in zip(self.axes, cell, strict=True)]) / width
            coefficients = np.empty((4, 4, 4))
            for mask, jet in enumerate(self.jets):
                start = [2 * int(bool(mask & (1 << a))) for a in range(3)]
                coefficients[start[0] : start[0] + 2, start[1] : start[1] + 2, start[2] : start[2] + 2] = jet[
                    cell[2] : cell[2] + 2, cell[1] : cell[1] + 2, cell[0] : cell[0] + 2
                ].transpose(2, 1, 0) * np.prod([width[a] for a in range(3) if mask & (1 << a)])
            powers = (
                np.stack((np.ones(3), t, t**2, t**3)),
                np.stack((np.zeros(3), np.ones(3), 2 * t, 3 * t**2)),
                np.stack((np.zeros(3), np.zeros(3), np.full(3, 2), 6 * t)),
            )
            weights = np.stack([(basis @ a).T / width[:, None] ** d for d, a in enumerate(powers)], axis=1)
            derivatives = np.einsum("ijk,ai,bj,ck->abc", coefficients, *weights)
            row = np.array([derivatives[order] for order in orders])
            delta = point - p
            length = np.linalg.norm(delta)
            if length:
                mask = ((point >= self.lower) & (point <= self.upper)).astype(float)
                unit = delta / length
                row[0] += length
                row[1:4] = row[1:4] * mask + unit
                row[4:] = (
                    row[4:].reshape(3, 3) * np.outer(mask, mask) + (np.diag(1 - mask) - np.outer(unit, unit)) / length
                ).ravel()
            result[i] = row
        return result


def evaluate(values, lower, upper, points, representation):
    """Return phi/g/H using an independent polynomial representation, in physical units."""
    from scipy.interpolate import CubicSpline

    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    values = np.asarray(values, dtype=float)
    dims = np.array(values.shape[::-1])
    dx = (upper - lower) / (dims - 1)
    axes = [np.linspace(lower[a], upper[a], dims[a]) for a in range(3)]
    jets = {0: values}
    if representation == "c1-hermite":
        for mask in range(1, 8):
            axis = (mask & -mask).bit_length() - 1
            source = jets[mask ^ (1 << axis)]
            # Explicit finite-difference matrix, independent of np.gradient in the builder.
            n = dims[axis]
            derivative = np.zeros((n, n))
            derivative[0, :3], derivative[-1, -3:] = [-1.5, 2, -0.5], [0.5, -2, 1.5]
            for i in range(1, n - 1):
                derivative[i, i - 1], derivative[i, i + 1] = -0.5, 0.5
            data = np.moveaxis(source, 2 - axis, 0)
            jets[mask] = np.moveaxis(np.tensordot(derivative / dx[axis], data, axes=1), 0, 2 - axis)
    elif representation != "c2-bspline":
        raise ValueError("Unknown representation")
    spline_x = CubicSpline(axes[0], values, axis=2, bc_type="natural") if representation == "c2-bspline" else None
    # Columns are powers 0..3; rows: left/right value, left/right derivative.
    basis = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]], dtype=float)
    orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    orders += [tuple(int(k == a) + int(k == b) for k in range(3)) for a in range(3) for b in range(3)]
    result = []
    for point in np.asarray(points, dtype=float):
        p = np.clip(point, lower, upper)
        f = (p - lower) / dx
        cell = np.clip(np.floor(f).astype(int), 0, dims - 2)
        polynomial = np.zeros((4, 4, 4))
        if representation == "c1-hermite":
            for indices in itertools.product(range(4), repeat=3):
                node = cell + np.array(indices) % 2
                mask = sum((1 << a) for a in range(3) if indices[a] >= 2)
                coefficient = jets[mask][node[2], node[1], node[0]]
                coefficient *= np.prod([dx[a] if indices[a] >= 2 else 1 for a in range(3)])
                polynomial += coefficient * np.einsum("i,j,k->ijk", *(basis[i] for i in indices))
        row = []
        spline_y, spline_z = {}, {}
        for order in orders:
            if representation == "c2-bspline":
                ox, oy, oz = order
                if ox not in spline_y:
                    spline_y[ox] = CubicSpline(axes[1], spline_x(p[0], nu=ox), axis=1, bc_type="natural")
                if (ox, oy) not in spline_z:
                    spline_z[ox, oy] = CubicSpline(axes[2], spline_y[ox](p[1], nu=oy), axis=0, bc_type="natural")
                row.append(float(spline_z[ox, oy](p[2], nu=oz)))
            else:
                item = polynomial
                for a in range(3):
                    for _ in range(order[a]):
                        item = np.polynomial.polynomial.polyder(item, axis=a)
                row.append(np.polynomial.polynomial.polyval3d(*(f - cell), item) / np.prod(dx ** np.array(order)))
        row = np.array(row)
        delta = point - p
        length = np.linalg.norm(delta)
        if length:
            mask = ((point >= lower) & (point <= upper)).astype(float)
            unit = delta / length
            row[0] += length
            row[1:4] = row[1:4] * mask + unit
            row[4:] = (
                row[4:].reshape(3, 3) * np.outer(mask, mask) + (np.diag(1 - mask) - np.outer(unit, unit)) / length
            ).ravel()
        result.append(row)
    return np.asarray(result)
