# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard T0 baseline isolation and the narrow scene parameterization seam."""

import ast
import copy
import gzip
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.monolithic_reference import check_p3_recording as recording
from scripts.monolithic_reference import run_p3_baseline as baseline

ROOT = Path(__file__).resolve().parents[2]


def constructor(path, name, *, frozen=False):
    source = (
        subprocess.check_output(["git", "show", f"{baseline.BASELINE}:{path}"], cwd=ROOT, text=True)
        if frozen
        else (ROOT / path).read_text()
    )
    module = ast.parse(source)
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)
    return copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"))


class TestP4Baseline(unittest.TestCase):
    def test_comparison_envelope(self):
        """Bound floating differences without relaxing metadata or layout equality."""
        reference = {
            "frame_steps": np.array([0, 1]),
            "frame_publication": np.array(["INITIAL", "CONVERGED"], dtype="U32"),
            "particle_q": np.array([[0.0], [1.0]]),
        }

        def compare(actual, limits=None):
            with patch.object(recording, "frames", side_effect=[[reference], [actual]]):
                return recording.compare(Path("reference"), Path("actual"), limits)["status"]

        self.assertEqual(compare(reference), "PASS_EXACT")
        changed = dict(reference, particle_q=np.array([[0.0], [1.125]]))
        self.assertEqual(compare(changed), "NOT_READY_TOLERANCE")
        self.assertEqual(compare(changed, {"particle_q": 0.25}), "PASS_FROZEN_ENVELOPE")
        self.assertEqual(compare(changed, {"particle_q": 0.0625}), "NOT_READY_TOLERANCE")
        changed = dict(reference, frame_publication=np.array(["INITIAL", "STOPPED_ACCEPTED"], dtype="U32"))
        self.assertEqual(compare(changed, {"frame_publication": 100}), "NOT_READY_TOLERANCE")
        changed = dict(reference, frame_steps=np.array([0, 2]))
        self.assertEqual(compare(changed, {"frame_steps": 100}), "NOT_READY_TOLERANCE")
        changed = dict(reference, particle_q=reference["particle_q"].astype(np.float32))
        with self.assertRaisesRegex(ValueError, "layouts differ"):
            compare(changed)

    def test_recording_integrity_gates(self):
        """Reject tampered chunks, missing history and wrong force generations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                name: np.zeros((2, 1))
                for name in ("joint_q", "joint_qd", "particle_q", "particle_qd", "history.link_vsym")
            }
            values.update(
                {"history.force_generation": np.array([[0], [1]]), "history.force_valid": np.array([[0], [1]])}
            )
            np.savez(root / "initial.npz", **{name: a[0] for name, a in values.items()})
            np.savez(root / "final.npz", **{name: a[-1] for name, a in values.items()})
            values.update(
                frame_steps=np.array([0, 1]),
                frame_times=np.array([0, 0.001]),
                frame_publication=np.array(["INITIAL", "CONVERGED"]),
            )
            path = root / "frames.npz"

            def write(arrays):
                np.savez(path, **arrays)
                (root / "trajectory-index.json").write_text(
                    json.dumps(
                        {
                            "chunks": [{"path": path.name, "count": 2, "sha256": recording.sha(path)}],
                            "includes_initial": True,
                            "total_frames": 2,
                        }
                    )
                )

            write(values)
            with gzip.open(root / "steps.jsonl.gz", "wt") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": 1,
                            "time": 0.001,
                            "publication": "CONVERGED",
                            "force_valid": 1,
                            "force_generation": 1,
                            "stats": {"rolled_back": False, "min_det_f": 1},
                        }
                    )
                    + "\n"
                )
            (root / "summary.json").write_text(json.dumps({"complete": True, "published_steps": 1}))
            self.assertEqual(recording.check(root, 1)["status"], "PASS")
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                recording.check(root, 1)
            write({key: value for key, value in values.items() if key != "history.link_vsym"})
            with self.assertRaisesRegex(ValueError, "state schema"):
                recording.check(root, 1)
            invalid = dict(values)
            invalid["history.force_generation"] = np.array([[0], [2]])
            write(invalid)
            with self.assertRaisesRegex(ValueError, "generation mismatch"):
                recording.check(root, 1)

    def test_matrix(self):
        """Keep exactly nine Franka variants and one reuse-only Sharpa case."""
        cases = baseline.cases()
        self.assertEqual(len({c["id"] for c in cases}), 10)
        franka = [c for c in cases if not c.get("reuse_only")]
        self.assertEqual(
            {(c["law"], c["globalization"]) for c in franka},
            {(law, g) for law in baseline.LAWS for g in baseline.GLOBALIZATIONS},
        )
        self.assertEqual(sum(c["steps"] for c in cases), 55917)

    def test_franka_constructor_seam(self):
        """Permit only explicit law/globalization and extra observational metadata."""
        original = constructor("scripts/monolithic_reference/franka_grasp.py", "FrankaGrasp", frozen=True)
        actual = constructor("scripts/monolithic_reference/p3_baseline_franka.py", "BaselineFranka")
        expected = copy.deepcopy(original)
        expected.args.kwonlyargs = [ast.arg(arg="law"), ast.arg(arg="globalization")]
        expected.args.kw_defaults = [None, None]
        changed = []
        for node in ast.walk(expected):
            if isinstance(node, ast.keyword) and node.arg in ("contact_model", "globalization"):
                name = "law" if node.arg == "contact_model" else "globalization"
                node.value = ast.Name(id=name, ctx=ast.Load())
                changed.append(node.arg)
        self.assertEqual(sorted(changed), ["contact_model", "globalization"])
        metadata = actual.body.pop()
        self.assertIsInstance(metadata, ast.Assign)
        self.assertEqual(ast.unparse(metadata.targets[0]), "self.metadata['baseline_options']")
        self.assertEqual(ast.dump(actual), ast.dump(expected))

    def test_recording_constructor_seam(self):
        """Keep the original recorder setup except for the parameterized case constructor."""
        expected = constructor("newton/examples/softbody/example_monolithic_franka_grasp.py", "Example", frozen=True)
        actual = constructor("scripts/monolithic_reference/p3_baseline_franka.py", "BaselineFrankaExample")
        calls = [
            n
            for n in ast.walk(expected)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "FrankaGrasp"
        ]
        self.assertEqual(len(calls), 1)
        calls[0].func.id = "BaselineFranka"
        for name in ("law", "globalization"):
            calls[0].keywords.append(
                ast.keyword(
                    arg=name, value=ast.Attribute(value=ast.Name(id="args", ctx=ast.Load()), attr=name, ctx=ast.Load())
                )
            )
        self.assertEqual(ast.dump(actual), ast.dump(expected))

    def test_changed_source_and_input_rejected(self):
        """Refuse drift even when the checkout reports the expected Git HEAD."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, asset = root / "solver.py", root / "asset.npz"
            source.write_bytes(b"original source")
            asset.write_bytes(b"original input")
            manifest = {
                "baseline_commit": baseline.BASELINE,
                "source_hashes": {source.name: baseline.sha(source)},
                "input_hashes": {str(asset): baseline.sha(asset)},
            }
            with patch.object(baseline.subprocess, "check_output", return_value=baseline.BASELINE + "\n"):
                baseline.validate_source(root, manifest)
                asset.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    baseline.validate_source(root, manifest)
                asset.write_bytes(b"original input")
                source.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    baseline.validate_source(root, manifest)

    def test_wrong_import_origin_rejected(self):
        """Reject an imported solver from a different checkout."""
        module = types.ModuleType("newton")
        module.__file__ = "/unexpected-checkout/newton/__init__.py"
        with patch.dict(baseline.sys.modules, {"newton": module}, clear=True):
            with self.assertRaisesRegex(ValueError, "Unexpected module origin"):
                baseline.module_origins(ROOT, {})


if __name__ == "__main__":
    unittest.main()
