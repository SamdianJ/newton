# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Qualify reusable natural-spline reference fields independently of native queries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference import smooth_sdf_oracle
from scripts.monolithic_reference.contact_full_oracle import export_mesh_fields
from scripts.monolithic_reference.contact_oracle import geometry, local


class TestNaturalSplineOracle(unittest.TestCase):
    def test_exported_field_reuse(self):
        """Query exported fields repeatedly without rebuilding any full-volume spline."""
        dims = np.array([3, 4, 5])
        lo, hi = np.array([-1, 0.3, 0.2]), np.array([1, 1.5, 1.2])
        dx = (hi - lo) / (dims - 1)
        z, y, x = np.meshgrid(*(lo[a] + dx[a] * np.arange(-1, dims[a] + 1) for a in (2, 1, 0)), indexing="ij")
        coefficients = 0.6 + 0.2 * x - 0.3 * y + 0.4 * z

        def array(value):
            return SimpleNamespace(numpy=lambda: value)

        field = SimpleNamespace(
            representation="c2-bspline",
            identity={"dims": dims, "lower": lo, "upper": hi},
            data=SimpleNamespace(coefficients=array(coefficients)),
        )
        model = SimpleNamespace(
            _texture_sdf_data=array(np.array([(False,), (False,)], dtype=[("scale_baked", bool)])),
            _shape_sdf_index=array(np.array([0, 1, 0])),
            shape_type=array(np.full(3, 8)),
            shape_flags=array(np.full(3, 4)),
            shape_scale=array(np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]])),
        )
        fields = export_mesh_fields(model, SimpleNamespace(fields=[field, field]))
        point = np.array([0.12, 0.51, 0.6])
        with patch("scipy.interpolate.CubicSpline", side_effect=AssertionError("Rebuilt a spline during query")):
            for _ in range(3):
                for shape, unit in enumerate((1, 1, 2)):
                    phi, gradient, hessian = geometry(unit * point, fields[shape], np.ones(3) * unit)
                    self.assertAlmostEqual(phi, unit * (0.6 + point @ [0.2, -0.3, 0.4]), places=12)
                    np.testing.assert_allclose(gradient, [0.2, -0.3, 0.4], rtol=0, atol=2e-12)
                    np.testing.assert_allclose(hessian, 0, rtol=0, atol=2e-12)
        self.assertIs(fields[0]["smooth_grid"], fields[1]["smooth_grid"])
        self.assertIsNot(fields[0]["smooth_grid"], fields[2]["smooth_grid"])
        with self.assertRaises(ValueError):
            fields[0]["smooth_grid"][0, 0, 0] = 99
        indices, flags = model._shape_sdf_index.numpy(), model.shape_flags.numpy()
        indices[0], flags[0] = -1, 1
        self.assertNotIn(0, export_mesh_fields(model, SimpleNamespace(fields=[field, field])))
        flags[0] = 4
        with self.assertRaises(ValueError):
            export_mesh_fields(model, SimpleNamespace(fields=[field, field]))
        for index in (-2, 2):
            indices[0], flags[0] = index, 1
            with self.assertRaises(ValueError):
                export_mesh_fields(model, SimpleNamespace(fields=[field, field]))

    def test_direct_equivalence(self):
        """Match direct SciPy phi, gradient and Hessian at cells, knots and exterior limits."""
        rng = np.random.default_rng(1093)
        lo, hi = np.array([-0.7, 0.5, -1.3]), np.array([0.9, 0.51, 2.2])
        orders = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        orders += [tuple(int(k == a) + int(k == b) for k in range(3)) for a in range(3) for b in range(3)]
        for shape in ((2, 2, 2), (4, 5, 6), (6, 3, 4)):
            values = rng.normal(size=shape)
            axes = [np.linspace(lo[a], hi[a], shape[2 - a]) for a in range(3)]
            knots = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
            midpoints = np.array(np.meshgrid(*[(a[1:] + a[:-1]) / 2 for a in axes], indexing="ij"))
            points = np.vstack((knots, midpoints.reshape(3, -1).T, rng.uniform(lo, hi, (19, 3))))
            for mask in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
                points = np.vstack((points, lo - 0.2 * (hi - lo) * mask, hi + 0.2 * (hi - lo) * mask))
            wanted = smooth_sdf_oracle.evaluate(values, lo, hi, points, "c2-bspline")
            actual = smooth_sdf_oracle.NaturalSpline(values, lo, hi).evaluate(points)
            dx = (hi - lo) / (np.array(shape[::-1]) - 1)
            units = max(1, np.max(np.abs(values))) / np.prod(dx ** np.array(orders), axis=1)
            np.testing.assert_allclose(actual / units, wanted / units, rtol=5e-12, atol=5e-12)

    def test_snapshot_and_reuse(self):
        """Own immutable inputs and reuse field derivatives through the local contact law."""
        lo, hi = np.array([-0.2, -0.1, -0.3]), np.array([0.2, 0.4, 0.3])
        z, y, x = np.meshgrid(
            np.linspace(lo[2], hi[2], 5), np.linspace(lo[1], hi[1], 4), np.linspace(lo[0], hi[0], 3), indexing="ij"
        )
        values = 0.8 * z + 0.1 * x + 0.3 * y + 0.2 * x * y * z
        prepared = smooth_sdf_oracle.NaturalSpline(values, lo, hi)
        raw = {"smooth_grid": values.copy(), "lower": lo.copy(), "upper": hi.copy(), "representation": "c2-bspline"}
        kind = {**raw, "natural_spline": prepared}
        point, old = np.array([0.01, 0.05, -0.02]), np.array([0, 0.05, -0.021])
        options = {"offset": 0.03, "stiffness": 11, "width": 0.01, "a": 0.005}
        wanted = local(point, old, raw, np.ones(3), **options)
        values.fill(np.nan)
        lo[:], hi[:] = 99, -99
        with patch("scipy.interpolate.CubicSpline", side_effect=AssertionError("Rebuilt a spline during query")):
            for _ in range(3):
                actual = local(point, old, kind, np.ones(3), **options)
                for a, b in zip(actual, wanted, strict=True):
                    np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11)
        with self.assertRaises(ValueError):
            prepared.values[0, 0, 0] = 1

    def test_invalid_inputs(self):
        """Reject malformed or nonfinite fields and queries before returning reference samples."""
        values, lo, hi = np.ones((2, 3, 4)), np.zeros(3), np.ones(3)
        for v, a, b in (
            (np.ones((3, 3)), lo, hi),
            (np.ones((1, 3, 4)), lo, hi),
            (values * np.nan, lo, hi),
            (values, lo[:2], hi),
            (values, hi, hi),
            (values, lo, lo),
            (values, lo, hi * np.inf),
        ):
            with self.assertRaises(ValueError):
                smooth_sdf_oracle.NaturalSpline(v, a, b)
        field = smooth_sdf_oracle.NaturalSpline(values, lo, hi)
        for points in ([1, 2, 3], [[1, 2]], [[np.nan, 0, 0]], [[np.inf, 0, 0]]):
            with self.assertRaises(ValueError):
                field.evaluate(points)
        self.assertEqual(field.evaluate(np.empty((0, 3))).shape, (0, 13))


if __name__ == "__main__":
    unittest.main()
