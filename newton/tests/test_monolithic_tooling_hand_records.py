# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep hand task acceptance and playback bound to the recorded inputs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.monolithic_reference import render_hand_bottle, run_graspit_hand, run_hand_bottle
from scripts.monolithic_reference.retained_assets import sha


class Array:
    def __init__(self, values):
        self.values = np.array(values)

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = np.array(values)


class TestHandRecords(unittest.TestCase):
    def setUp(self):
        self.reset_fixture()

    def reset_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.dependencies = (
            "newton/_src/solvers/monolithic/solver_monolithic.py",
            "scripts/monolithic_reference/material_recipes.py",
            "newton/examples/softbody/monolithic_paper_box.py",
        )
        for name in ("newton/__init__.py", *self.dependencies):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# original\n")
        self.asset = self.root / "asset"
        self.asset.mkdir()
        (self.asset / "mesh.obj").write_text("original mesh")
        self.manifest = {"source_sha256": {}, "derived_sha256": {"mesh.obj": sha(self.asset / "mesh.obj")}}
        (self.asset / "manifest.json").write_text(json.dumps(self.manifest))
        self.head, self.dirty, self.steps = "initial-head", "", 0

    def run_task(self, kind, change=lambda: None, *, wrong_import=False):
        """Use passing publications to isolate the real runner's identity gate."""
        dofs = 20 if kind == "hand" else 21
        state = SimpleNamespace(
            joint_q=Array(np.zeros(dofs)),
            joint_qd=Array(np.zeros(dofs)),
            body_q=Array([[0, 0, 0, 0, 0, 0, 1]]),
            particle_q=Array(np.zeros((3, 3))),
            particle_qd=Array(np.zeros((3, 3))),
        )
        control = SimpleNamespace(joint_target_q=Array(np.zeros(dofs)), joint_target_qd=Array(np.zeros(dofs)))
        model = SimpleNamespace(
            body_count=1,
            joint_dof_count=dofs,
            shape_count=0,
            shape_flags=Array([]),
            shape_type=Array([]),
            tri_indices=Array([[0, 1, 2]]),
            joint_limit_lower=Array(-np.ones(dofs)),
            joint_limit_upper=Array(np.ones(dofs)),
        )
        forces = SimpleNamespace(
            force_valid=True,
            joint_force=np.zeros(dofs),
            body_wrench=np.zeros((1, 6)),
            particle_force=np.zeros((3, 3)),
            force_generation=0,
        )

        def step(*args, **kwargs):
            self.steps += 1
            if self.steps == 1:
                change()

        solver = SimpleNamespace(
            create_state=lambda **kwargs: state,
            create_control=lambda: control,
            step=step,
            get_contact_forces=lambda state: forces,
            save_checkpoint=lambda *args: None,
            last_stats=SimpleNamespace(
                rolled_back=False,
                variational={"publication_status": "CONVERGED"},
                failure_reason="",
                nonlinear_iterations=1,
                convergence_ratio=0,
                converged=True,
            ),
        )
        controller = SimpleNamespace(
            snapshot=lambda: None,
            restore=lambda snapshot: None,
            command=lambda *args, **kwargs: (np.zeros(20), np.zeros(20)),
            loads=lambda forces: np.zeros(4),
            axes=[SimpleNamespace(position=0, velocity=0)] * 4,
        )
        metrics = {
            "sampled_penetration_m": 0,
            "hand_table_penetration_m": 0,
            "bottle_bottom_clearance_m": 0.06,
            "deformation_max_m": 0.004,
            "self_intersections": 0,
            "area_ratio_min": 1,
            "area_ratio_max": 1,
            "limit_violation_rad": 0,
            "max_joint_speed_rad_s": 0,
            "hand_lift_speed_m_s": 0,
            "hand_lift_m": 0.08,
            "bottle_center_speed_m_s": 0,
            "hand_vertical_reaction_N": 0.1,
            "projection_error_Nm": 0,
            "lift_projection_error_N": 0,
        }
        initial = {**metrics, "bottle_bottom_clearance_m": 0}
        output = self.root / "result"
        args = SimpleNamespace(
            output=output,
            asset_dir=self.asset,
            device="cpu",
            precision="fp64",
            force=2.0,
            max_steps=run_hand_bottle.STEPS,
        )

        def git(command, **kwargs):
            return self.dirty if command[1] == "status" else self.head

        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(run_graspit_hand, "SOURCE", self.source, create=True))
            stack.enter_context(patch.object(run_graspit_hand.subprocess, "check_output", side_effect=git))
            imported = self.root / "other/newton/__init__.py" if wrong_import else self.source / "newton/__init__.py"
            stack.enter_context(patch.object(run_graspit_hand.newton, "__file__", str(imported)))
            stack.enter_context(
                patch.object(run_graspit_hand.wp, "get_device", return_value=SimpleNamespace(is_cuda=False))
            )
            stack.enter_context(
                patch.object(
                    run_graspit_hand,
                    "build_hand",
                    return_value=(SimpleNamespace(finalize=lambda **kwargs: model), {"total_mass_kg": 1}),
                )
            )
            stack.enter_context(patch.object(run_graspit_hand, "SolverMonolithic", Mock(return_value=solver)))
            stack.enter_context(patch.object(run_graspit_hand, "closed_pose", return_value=np.zeros(20)))
            stack.enter_context(
                patch.object(
                    run_hand_bottle,
                    "build_scene",
                    return_value=(model, solver, state, control, controller, {}, SimpleNamespace(build_metadata={})),
                )
            )
            stack.enter_context(
                patch.object(
                    run_hand_bottle,
                    "BottleMetrics",
                    return_value=SimpleNamespace(
                        measure=Mock(side_effect=[initial] + [metrics] * run_hand_bottle.STEPS)
                    ),
                )
            )
            passed = (
                run_graspit_hand.run(self.asset, output, device="cpu", precision="fp64", scenario="motion")
                if kind == "hand"
                else run_hand_bottle.run(args)
            )
        return passed, json.loads((output / "summary.json").read_text()), output

    def test_stable_identity_passes_and_covers_dependencies(self):
        """Bind passing task receipts to initial HEAD, all dependencies and actual asset content."""
        for kind in ("hand", "bottle"):
            with self.subTest(kind=kind):
                self.reset_fixture()
                passed, summary, output = self.run_task(kind)
                self.assertTrue(passed)
                self.assertTrue(summary["identity_unchanged"])
                self.assertFalse(summary["source_dirty"])
                self.assertTrue(set(self.dependencies) <= summary["source_files"].keys())
                before = json.loads((output / "input.json").read_text())
                self.assertEqual(before["source_head"], "initial-head")
                self.assertEqual(before["asset_manifest_sha256"], summary["asset_manifest_sha256"])

    def test_changed_identity_cannot_pass(self):
        """Reject changed solver, material, geometry, HEAD, manifest and asset bytes after valid steps."""
        for kind in ("hand", "bottle"):
            for item in (*self.dependencies, "head", "manifest", "asset"):
                with self.subTest(kind=kind, change=item):
                    self.reset_fixture()

                    def change(item=item):
                        if item == "head":
                            self.head = "changed-head"
                        elif item == "manifest":
                            (self.asset / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
                        elif item == "asset":
                            (self.asset / "mesh.obj").write_text("changed geometry")
                        else:
                            (self.source / item).write_text("# changed\n")

                    passed, summary, output = self.run_task(kind, change)
                    self.assertFalse(passed)
                    self.assertTrue(all(summary["gates"].values()))
                    self.assertFalse(summary["identity_unchanged"])
                    self.assertTrue(summary["identity_error"])
                    self.assertEqual(summary["source_head"], "initial-head")
                    self.assertEqual(json.loads((output / "input.json").read_text())["source_head"], "initial-head")

    def test_invalid_initial_identity_does_not_run_physics(self):
        """Keep a failure receipt and skip physics for dirty, wrong-import or inconsistent inputs."""
        for kind in ("hand", "bottle"):
            for invalid in ("dirty", "import", "asset"):
                with self.subTest(kind=kind, invalid=invalid):
                    self.reset_fixture()
                    if invalid == "dirty":
                        self.dirty = " M newton/_src/solvers/monolithic/solver_monolithic.py\n"
                    elif invalid == "asset":
                        (self.asset / "mesh.obj").write_text("wrong geometry")
                    passed, summary, output = self.run_task(kind, wrong_import=invalid == "import")
                    self.assertFalse(passed)
                    self.assertEqual(self.steps, 0)
                    self.assertTrue(summary["identity_error"])
                    expected = {"device": "cpu", "precision": "fp64"}
                    expected.update(
                        {"scenario": "motion", "requested_steps": 600}
                        if kind == "hand"
                        else {"target_force_N": 2.0, "requested_steps": 1400}
                    )
                    for record in (summary, json.loads((output / "input.json").read_text())):
                        for key, value in expected.items():
                            self.assertEqual(record[key], value)

    def test_playback_rejects_other_asset_before_loading_trajectory(self):
        """Reject a different valid asset before opening the recording or plotting anything."""
        result = self.root / "recording"
        result.mkdir()
        (result / "summary.json").write_text(json.dumps({"asset_manifest_sha256": "another-asset"}))
        with patch.object(render_hand_bottle.np, "load", side_effect=AssertionError("trajectory opened")) as load:
            with self.assertRaisesRegex(ValueError, "asset manifest"):
                render_hand_bottle.render(self.asset, result, video=False)
            load.assert_not_called()

    def test_playback_accepts_matching_asset_identity(self):
        """Allow the recorded asset through the identity guard to trajectory loading."""
        result = self.root / "recording"
        result.mkdir()
        (result / "summary.json").write_text(json.dumps({"asset_manifest_sha256": sha(self.asset / "manifest.json")}))
        with patch.object(render_hand_bottle.np, "load", side_effect=RuntimeError("trajectory reached")):
            with self.assertRaisesRegex(RuntimeError, "trajectory reached"):
                render_hand_bottle.render(self.asset, result, video=False)


if __name__ == "__main__":
    unittest.main()
