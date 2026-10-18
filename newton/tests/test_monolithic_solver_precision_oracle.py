# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check offline scene math against independent references and native transactions."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_recording import serial, state_arrays
from newton.tests.test_monolithic_solver_coupled import scene, solver
from scripts.monolithic_reference.audit_precision import audit
from scripts.monolithic_reference.build_smooth_sdf import build_coefficients
from scripts.monolithic_reference.observe_precision import Observer
from scripts.monolithic_reference.run_precision import sha
from scripts.monolithic_reference.scene_oracle import SceneOracle, spline
from scripts.monolithic_reference.smooth_sdf_oracle import evaluate as spline_reference


class TestSceneAudit(unittest.TestCase):
    def test_promoted_history(self):
        """Recover coupled residuals with real prior histories and detect history corruption."""
        model = scene("cpu")
        run = solver(model)
        state = run.create_state(dt=0.001)
        run.step(state, state, model.control(), dt=0.001)
        self.assertTrue(run.last_stats.converged)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            observer = Observer(run, path)
            run.step(state, state, model.control(), dt=0.001)
            self.assertTrue(run.last_stats.converged)

            def load(name, path=path):
                with np.load(path / (name + ".npz")) as source:
                    return dict(source)

            oracle = SceneOracle(
                load("model"), load("physics"), load("contact"), {}, json.loads((path / "physics.json").read_text())
            )
            row = observer.rows[-1]
            data = dict(np.load(path / row["file"]))
            data["dt"] = row["dt"]
            result = oracle.evaluate(data)
            np.testing.assert_allclose(result["residual"], data["native.residual"], atol=1e-9, rtol=1e-7)
            self.assertAlmostEqual(result["energy"], float(data["native.energy"]), delta=1e-10)
            data["original.history.link_v_com"] = data["original.history.link_v_com"] + 0.01
            changed = oracle.evaluate(data)
            self.assertGreater(np.linalg.norm(changed["residual"] - result["residual"]), 1)

    def test_audit_archive(self):
        """Audit a captured successful transaction end to end, including trial and B32 checks."""
        model = scene("cpu")
        run = solver(
            model,
        )
        state = run.create_state(dt=0.001)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            Observer(run, path)
            run.step(state, state, model.control(), dt=0.001)
            self.assertTrue(run.last_stats.converged)
            np.savez_compressed(path / "states.npz", **{k: v[None] for k, v in state_arrays(state).items()})
            receipt = {
                "case": {"globalization": "ArmijoOrResNorm"},
                "status": "COMPLETE_NOT_AUDITED",
                "chunks": [{"file": "states.npz", "frames": 1, "sha256": sha(path / "states.npz")}],
            }
            (path / "receipt.json").write_text(json.dumps(receipt))
            with gzip.open(path / "steps.jsonl.gz", "wt") as stream:
                stream.write(
                    json.dumps(serial({"step": 1, "publication": "CONVERGED", "stats": run.last_stats})) + "\n"
                )
            result = audit(path)
            self.assertFalse(result["publications"][0]["false_convergence"])
            self.assertTrue(result["linear"])
            self.assertFalse(result["accepted_but_independent_rejects"])

    def test_spline_reference(self):
        """Match an independent natural spline at interior, boundary and exterior points."""
        values = np.random.default_rng(72).normal(size=(5, 6, 7))
        lower, upper = np.array([-1.0, -2.0, -3.0]), np.array([1.0, 2.0, 3.0])
        dims = np.array(values.shape[::-1])
        dx = (upper - lower) / (dims - 1)
        coefficients = build_coefficients(values, dx, "c2-bspline")
        field = {"0.lower": lower, "0.upper": upper, "0.inv_dx": 1 / dx, "0.dims": dims, "0.coefficients": coefficients}
        points = np.array([[0.1, -0.3, 0.4], lower, upper, [1.2, -0.5, 3.1]])
        value, gradient, _ = spline(field, 0, points)
        reference = spline_reference(values, lower, upper, points, "c2-bspline")
        np.testing.assert_allclose(value, reference[:, 0], atol=1e-12)
        np.testing.assert_allclose(gradient, reference[:, 1:4], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
