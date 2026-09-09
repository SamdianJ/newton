# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check offline execution-option binding without launching simulation jobs."""

import json
import sys
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.monolithic_reference import profile_p2_owned_mass as mass_profile
from scripts.monolithic_reference import run_p2_8bc as runner
from scripts.monolithic_reference.profile_p2_owned_mass import load_samples


class FakeSolver:
    class PCGMode(Enum):
        DIAGNOSTIC = "diagnostic"
        PRODUCTION = "production"

    def __init__(self, **kwargs):
        self.use_optimized_articulation_mass_matrix = kwargs["use_optimized_articulation_mass_matrix"]
        self.pcg_mode = self.PCGMode(kwargs["pcg_mode"])


class TestP28BCRunner(unittest.TestCase):
    def test_mass_replay_accepts_json_mapping_representation(self):
        """Compare tuple-valued live mappings with their equivalent saved JSON lists."""
        manifest = {"urdf_sha256": "same", "joint_names": ["hinge"], "mapping": {"hinge": (0, 0)}}
        samples = [{"time": t, "q": [t], "qd": [0]} for t in (1.0, 3.5)]
        compact = {"complete": True, "manifest": json.loads(json.dumps(manifest)), "common_time": {"samples": samples}}
        self.assertEqual(load_samples(compact, manifest), samples)
        manifest["mapping"]["hinge"] = (0, 1)
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_samples(compact, manifest)

    def test_mass_helper_forwards_joint_terms(self):
        """Pass the original joint policy when constructing the owned comparison workspace."""
        case = SimpleNamespace(
            model=SimpleNamespace(device="cuda:0"),
            manifest={},
            solver=SimpleNamespace(_articulation=object(), _joint_terms=object()),
        )

        def construct(model, **kwargs):
            self.assertIs(model, case.model)
            self.assertIs(kwargs.get("joint_terms"), case.solver._joint_terms)
            return object()

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "compact.json", Path(directory) / "output"
            source.write_text(json.dumps({"mesh": "r3"}))
            with (
                patch.object(sys, "argv", ["profile", "--input", str(source), "--output", str(output)]),
                patch.object(mass_profile, "snapshot_sources"),
                patch.object(mass_profile.subprocess, "check_output", return_value="mock device"),
                patch.object(mass_profile.sharpa, "GraspCase", return_value=case),
                patch.object(mass_profile, "load_samples", return_value=[]),
                patch.object(mass_profile.articulation, "MonolithicArticulationWorkspace", side_effect=construct),
                patch.object(mass_profile.wp, "ScopedDevice"),
            ):
                mass_profile.main()

    def test_mass_replay_rejects_asset_mismatch(self):
        """Reject a saved pose if the current model uses different assets."""
        manifest = {"urdf_sha256": "old", "joint_names": ["hinge"], "mapping": [0]}
        compact = {"complete": True, "manifest": manifest}
        with self.assertRaisesRegex(ValueError, "urdf_sha256"):
            load_samples(compact, {**manifest, "urdf_sha256": "new"})

    def test_mass_replay_requires_both_complete_poses(self):
        """Require q and qd at both exact recorded physical times."""
        manifest = {"urdf_sha256": "same", "joint_names": ["hinge"], "mapping": [0]}
        samples = [{"time": t, "q": [t], "qd": [0]} for t in (1.0, 3.5)]
        compact = {"complete": True, "manifest": manifest, "common_time": {"samples": samples}}
        self.assertEqual(load_samples(compact, manifest), samples)
        samples[1].pop("qd")
        with self.assertRaisesRegex(ValueError, "3.5"):
            load_samples(compact, manifest)

    def test_bind_warmup_and_measurement_and_restore(self):
        """Bind both constructions and index the actual options before returning."""
        original = FakeSolver.__init__
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"

            def run():
                output.mkdir()
                FakeSolver()
                FakeSolver()
                record = json.loads((output / "execution-options.json").read_text())
                self.assertEqual(len(record["constructions"]), 2)
                self.assertEqual(record["constructions"], [record["requested"]] * 2)

            with patch.object(runner, "SolverMonolithic", FakeSolver), patch.object(runner.run_p2_8a, "main", run):
                runner.main(
                    ["--output", str(output), "--kind", "sharpa", "--mass-matrix", "owned", "--pcg-mode", "production"]
                )
            self.assertIs(FakeSolver.__init__, original)
            index = json.loads((output / "evidence-index.json").read_text())
            self.assertIn("execution-options.json", index)

    def test_conflicting_fixture_option_fails_and_restores(self):
        """Reject an explicit conflicting fixture option and retain failure evidence."""
        original = FakeSolver.__init__
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"

            def run():
                output.mkdir()
                FakeSolver(pcg_mode="diagnostic")

            with patch.object(runner, "SolverMonolithic", FakeSolver), patch.object(runner.run_p2_8a, "main", run):
                with self.assertRaisesRegex(ValueError, "Conflicting constructor option: pcg_mode"):
                    runner.main(["--output", str(output), "--kind", "tet", "--pcg-mode", "production"])
            self.assertIs(FakeSolver.__init__, original)
            record = json.loads((output / "execution-options.json").read_text())
            self.assertEqual(record["constructions"], [])
            self.assertIn("ValueError", record["failure"])

    def test_existing_evidence_is_untouched(self):
        """Reject output reuse before changing any existing evidence."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "execution-options.json"
            sentinel.write_text("preserve")
            with patch.object(runner.run_p2_8a, "main") as run:
                with self.assertRaises(FileExistsError):
                    runner.main(["--output", str(output), "--kind", "tet"])
                run.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
