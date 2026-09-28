# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the public builder's strict qualification boundary."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF
from newton._src.solvers.monolithic.smooth_sdf_builder import build_coefficients
from newton.solvers.experimental.monolithic import SmoothSDFSet, build_smooth_sdf


class TestSmoothSDFBuilder(unittest.TestCase):
    def test_public_import(self):
        """Import the public builder without loading reference drivers or old solvers."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from newton.solvers.experimental.monolithic import build_smooth_sdf, SmoothSDFSet; "
                "import sys; assert not any(n.startswith('scripts.') or n.endswith(('solver_monolithic_legacy', 'articulation_solver')) for n in sys.modules)",
            ],
            check=True,
        )

    def test_qualified_validation(self):
        """Reject each corrupt qualification input without invoking model sampling."""
        values = np.arange(125, dtype=np.float64).reshape(5, 5, 5) / 100
        lower, upper, dims = [0.0] * 3, [1.0] * 3, [5] * 3
        field = SmoothSDF(
            build_coefficients(values, [0.25] * 3, "c2-bspline"), lower, upper, dims, "c2-bspline", device="cpu"
        )
        table = np.zeros(
            1,
            dtype=[
                ("sdf_box_lower", "f8", 3),
                ("sdf_box_upper", "f8", 3),
                ("inv_sdf_dx", "f8", 3),
                ("scale_baked", "?"),
            ],
        )
        table["sdf_box_upper"] = upper
        table["inv_sdf_dx"] = 4
        model = SimpleNamespace(
            device="cpu",
            _texture_sdf_data=SimpleNamespace(numpy=lambda: table),
            shape_scale=SimpleNamespace(numpy=lambda: np.ones((1, 3))),
            _texture_sdf_coarse_textures=[],
            _texture_sdf_subgrid_textures=[],
            _texture_sdf_subgrid_start_slots=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            nodes = directory / "00-field" / "nodes.npz"
            nodes.parent.mkdir()
            np.savez(nodes, values=values)
            row = {
                "source": "field",
                "node_sha256": hashlib.sha256(nodes.read_bytes()).hexdigest(),
                "sdf_index": 0,
                "lower": lower,
                "upper": upper,
                "dims": dims,
                "original_inv_dx": [4.0] * 3,
                "candidates": {"c2-bspline": {"identity": field.identity}},
            }
            original = {
                "candidate-qualification.json": {"passed": {"c2-bspline": True}, "source_textures": {}},
                "summary.json": {"geometry_qualified": {"c2-bspline": True}},
                "geometry-results.json": [row],
            }

            def write(payload):
                for name, data in payload.items():
                    (directory / name).write_text(json.dumps(data))

            write(original)
            result = build_smooth_sdf(model, qualified_directory=directory)
            self.assertIsInstance(result, SmoothSDFSet)
            self.assertTrue(result.build_metadata["qualified"])
            self.assertEqual(result.fields[0].identity, field.identity)
            changes = [
                ("candidate-qualification.json", lambda x: x["passed"].update({"c2-bspline": False})),
                ("candidate-qualification.json", lambda x: x.update(source_textures={"changed": "hash"})),
                ("summary.json", lambda x: x["geometry_qualified"].update({"c2-bspline": False})),
                ("geometry-results.json", lambda x: x.clear()),
                ("geometry-results.json", lambda x: x[0].update(node_sha256="changed")),
                ("geometry-results.json", lambda x: x[0].update(lower=[-1.0] * 3)),
                ("geometry-results.json", lambda x: x[0]["candidates"]["c2-bspline"]["identity"].update(dims=[9] * 3)),
            ]
            with patch("newton._src.solvers.monolithic.smooth_sdf_builder._sample_fields") as sample:
                for name, change in changes:
                    with self.subTest(name=name, change=change):
                        payload = json.loads(json.dumps(original))
                        change(payload[name])
                        write(payload)
                        with self.assertRaises(ValueError):
                            build_smooth_sdf(model, qualified_directory=directory)
                sample.assert_not_called()
            with self.assertRaises(ValueError):
                build_smooth_sdf(model, representation="unknown")


if __name__ == "__main__":
    unittest.main()
