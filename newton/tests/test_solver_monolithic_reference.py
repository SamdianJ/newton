# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the shared tiny fixture and offline reference contracts."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.adapter import validate_adapter_scope, world_nodes
from scripts.monolithic_reference.manifest import (
    ComparisonManifest,
    MappingEntry,
    canonical_manifest_bytes,
    compare_runs,
    compute_manifest_sha256,
    load_manifest,
    require_reference_worktree,
    run_adapter,
    validate_manifest,
    validate_step_record,
)


class TestMonolithicReference(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(
            Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/tiny_draft_v1.json"
        )

    def test_native_local_to_world_rotated_translated(self):
        """Reject the identity-only shortcut using an independently known 90-degree rotation."""
        local = [[1, 0, 0], [0, 1, 2], [-2, 3, 4]]
        result = world_nodes(local, [0, 0, 2**-0.5, 2**-0.5], [0.3, -0.4, 0.5])
        np.testing.assert_allclose(result, [[0.3, 0.6, 0.5], [-0.7, -0.4, 2.5], [-2.7, -2.4, 4.5]], atol=2e-15)

    def test_adapter_rejects_unmapped_physics(self):
        """A normal-loading/fixed-node input cannot silently become the no-contact fixture."""
        path = Path(__file__).parents[2] / "scripts/monolithic_reference/fixtures/reference_no_contact_draft_v1.json"
        manifest = load_manifest(path)
        validate_adapter_scope(manifest.data["physics"])
        for field, value in (
            ("shapes", None),
            ("contact", {}),
            ("drive", {}),
            ("fixed_nodes", [False, False, False, True]),
            ("dt_s", None),
            ("boundary_faces", [[1, 2, 3]]),
            ("tet_indices", [[0, 2, 1, 3]]),
        ):
            with self.subTest(field=field):
                physics = copy.deepcopy(manifest.data["physics"])
                physics[field] = value
                with self.assertRaises(ValueError):
                    validate_adapter_scope(physics)

    def test_draft_is_not_frozen(self):
        """Accept a draft without claiming measured or frozen evidence."""
        validate_manifest(self.manifest, require_frozen=False)
        with self.assertRaisesRegex(ValueError, "FROZEN"):
            validate_manifest(self.manifest, require_frozen=True)
        promoted = copy.deepcopy(self.manifest.data)
        promoted["status"] = "FROZEN"
        with self.assertRaises(ValueError):
            validate_manifest(ComparisonManifest(promoted), require_frozen=False)

    def test_hash_is_canonical_and_sensitive(self):
        """Ignore dictionary insertion order while hashing every physics value."""
        reordered = ComparisonManifest(dict(reversed(list(self.manifest.data.items()))))
        self.assertEqual(compute_manifest_sha256(self.manifest), compute_manifest_sha256(reordered))
        self.assertEqual(json.loads(canonical_manifest_bytes(self.manifest)), self.manifest.data)
        changed = copy.deepcopy(self.manifest.data)
        changed["physics"]["particle_radius_m"] *= 2
        self.assertNotEqual(
            compute_manifest_sha256(self.manifest), compute_manifest_sha256(ComparisonManifest(changed))
        )

    def test_invalid_units_mapping_and_values(self):
        """Reject ambiguous units, mapping statuses, nonfinite data and invalid topology."""
        cases = [
            ("units", "force", "residual"),
            ("physics", "particle_radius_m", -1),
            ("physics", "density_kg_m3", float("nan")),
            ("physics", "tet_indices", [[0, 1, 2, 9]]),
        ]
        for section, field, value in cases:
            with self.subTest(field=field):
                data = copy.deepcopy(self.manifest.data)
                data[section][field] = value
                with self.assertRaises(ValueError):
                    validate_manifest(ComparisonManifest(data), require_frozen=False)
        data = copy.deepcopy(self.manifest.data)
        data["mappings"][0]["status"] = "ASSUMED_MATCHED"
        with self.assertRaises(ValueError):
            validate_manifest(ComparisonManifest(data), require_frozen=False)

    def _frozen_manifest_data(self):
        """Construct synthetic schema inputs without claiming measured evidence."""
        data = copy.deepcopy(self.manifest.data)
        data["status"] = "FROZEN"
        physics = data["physics"]
        physics.update(asset_sha256="a" * 64, dt_s=0.001, substeps=1)
        physics["shapes"] = [
            {
                "type": "sphere",
                "link": 1,
                "dimensions_m": [0.01, 0.01, 0.01],
                "xform": [0, 0, 0, 0, 0, 0, 1],
                "sdf": {
                    "resolution": [1, 1, 1],
                    "scale_baked": False,
                    "asset_sha256": "a" * 64,
                    "runtime_scale": [1, 1, 1],
                    "provenance": "Synthetic schema test only",
                },
            }
        ]
        physics["contact"] = {
            "soft_contact_gap_m": 0.002,
            "shape_margin_m": 0.0,
            "penetration_acceptance_limit_m": 0.001,
            "quadrature": "P1Q3",
            "barycentric": [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
            "weight_rule": "reference_area/3",
            "normal_law": "quadratic_hinge",
            "normal_stiffness_n_m3": 1000.0,
            "friction_coefficient": 0.0,
        }
        physics["drive"] = {
            "trajectory": [{"time_s": 0.0, "target_q": [0.2, 0.01], "target_qd": [0, 0]}],
            "stiffness": [1, 1],
            "damping": [0, 0],
        }
        calibration = {
            "global_r_min": 1e-5,
            "q_r_min": 1e-5,
            "x_r_min": 1e-5,
            "merit_noise": 1e-5,
            "nonlinear_tolerance": 1e-4,
            "pcg_true_residual_tolerance": 1e-4,
            "scaled_step_tolerance": 1e-4,
            "evidence_sha256": "a" * 64,
        }
        data["benchmark"] = {
            "stages": {"free_space_end_step": 1, "contact_loading_end_step": 2, "settle_end_step": 3},
            "contact_onset": {"force_floor_n": 0.01, "consecutive_substeps": 2},
            "approach_axis_world": [1, 0, 0],
            "rigid_reference_point_local_m": [0, 0, 0],
            "soft_probe": [0],
            "rigid_motion_removal": "none",
            "delta_soft_min_m": 0.005,
            "force_noise_floor_n": 0.01,
            "actual_closure_interval_m": [0, 0.01],
            "reference_envelope": {
                key: [0, 1]
                for key in ["onset_m", "secant_stiffness_n_m", "peak_force_n", "settled_force_n", "curve_work_j"]
            },
            "support_envelope": {
                "min_rigid_feature_m": 0.01,
                "soft_mesh_resolution_m": 0.01,
                "sdf_resolution_m": 0.001,
                "evidence_sha256": "a" * 64,
            },
            "calibration": {"cpu": calibration, "cuda": calibration},
            "evidence_sha256": "a" * 64,
        }
        # These ephemeral values exercise the schema, never a numerical acceptance gate.
        for mapping in data["mappings"]:
            mapping["status"] = "MATCHED"
            mapping["rationale"] = "Synthetic schema validation only, not measured evidence"
        return data

    def test_frozen_requires_physics_and_envelopes(self):
        """Reject incomplete frozen physics independently of missing envelope evidence."""
        data = self._frozen_manifest_data()
        validate_manifest(ComparisonManifest(data), require_frozen=True)
        paths = [
            ("physics", "fixed_nodes"),
            ("physics", "tet_materials_pa_pa_pas"),
            ("physics", "dt_s"),
            ("physics", "shapes"),
            ("physics", "drive"),
            ("physics", "contact", "soft_contact_gap_m"),
            ("physics", "contact", "shape_margin_m"),
            ("physics", "contact", "penetration_acceptance_limit_m"),
            ("benchmark", "contact_onset"),
            ("benchmark", "reference_envelope"),
            ("benchmark", "support_envelope"),
            ("benchmark", "calibration", "cuda"),
            ("benchmark", "delta_soft_min_m"),
            ("benchmark", "evidence_sha256"),
        ]
        for path in paths:
            with self.subTest(missing=path):
                incomplete = copy.deepcopy(data)
                owner = incomplete
                for key in path[:-1]:
                    owner = owner[key]
                del owner[path[-1]]
                with self.assertRaises(ValueError):
                    validate_manifest(ComparisonManifest(incomplete), require_frozen=True)
        data["mappings"][0]["status"] = "UNMAPPED"
        with self.assertRaisesRegex(ValueError, "UNMAPPED"):
            validate_manifest(ComparisonManifest(data), require_frozen=True)

    def test_frozen_rejects_incoherent_physics(self):
        """Reject inconsistent dimensions, topology, quadrature, scales and benchmark ranges."""
        mutations = [
            (("physics", "initial", "qd"), [0.1]),
            (("physics", "joints", 1, "child"), 99),
            (("physics", "joints", 1, "parent"), 1),
            (("physics", "joints", 0, "axis"), [0, 0, 0]),
            (("physics", "shapes", 0, "link"), 99),
            (("physics", "shapes", 0, "xform"), [0, 0, 0, 0, 0, 0, 0]),
            (("physics", "contact", "barycentric"), [[0, 0, 0]] * 3),
            (("physics", "shapes", 0, "sdf", "runtime_scale"), [-1, -1, -1]),
            (("physics", "shapes", 0, "sdf", "resolution"), [0, 1, 1]),
            (("physics", "node_masses_kg", 0), 0),
            (("physics", "node_masses_kg", 0), 1e-100),
            (("physics", "gravity_m_s2", 0), 1e100),
            (("physics", "tet_materials_pa_pa_pas", 0, 0), 0),
            (("physics", "boundary_faces"), [[1, 2, 3]]),
            (("physics", "tet_indices"), [[0, 2, 1, 3]]),
            (("physics", "drive", "stiffness"), [1]),
            (("physics", "drive", "trajectory", 0, "target_q"), [0.2]),
            (("benchmark", "reference_envelope", "peak_force_n"), [10, 1]),
            (("benchmark", "actual_closure_interval_m"), [1, 0]),
            (("benchmark", "stages", "free_space_end_step"), 99),
            (("benchmark", "approach_axis_world"), [0, 0, 0]),
            (("benchmark", "soft_probe"), [99]),
        ]
        for path, value in mutations:
            data = self._frozen_manifest_data()
            owner = data
            for key in path[:-1]:
                owner = owner[key]
            owner[path[-1]] = value
            with self.subTest(field=path), self.assertRaises(ValueError):
                validate_manifest(ComparisonManifest(data), require_frozen=True)

    def test_frozen_volume_scale_provenance(self):
        """Accept baked nonuniform scales and reject unsupported unbaked volume scales."""
        data = self._frozen_manifest_data()
        shape = data["physics"]["shapes"][0]
        shape["type"] = "volume_sdf"
        for scale in ([0, 0, 0], [-1, -1, -1], [1, 2, 1]):
            shape["sdf"]["runtime_scale"] = scale
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                validate_manifest(ComparisonManifest(data), require_frozen=True)
        shape["sdf"]["runtime_scale"] = [2, 2, 2]
        validate_manifest(ComparisonManifest(data), require_frozen=True)
        shape["sdf"]["scale_baked"] = True
        shape["sdf"]["runtime_scale"] = [1, 2, 3]
        validate_manifest(ComparisonManifest(data), require_frozen=True)
        shape["type"] = "box"
        shape["sdf"]["scale_baked"] = False
        shape["dimensions_m"] = [0.01, 0.02, 0.03]
        validate_manifest(ComparisonManifest(data), require_frozen=True)

    def test_duplicate_json_keys_rejected(self):
        """Reject duplicate JSON keys before canonical hashing loses information."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"status":"DRAFT","status":"FROZEN"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(path)

    def test_worktree_guards(self):
        """Reject wrong commits and tracked or untracked worktree modifications."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

            git("init", "-q")
            (root / "source").write_text("baseline")
            git("add", "source")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Initialize")
            sha = git("rev-parse", "HEAD")
            identity = require_reference_worktree(root, expected_sha=sha)
            self.assertEqual(identity.sha, sha)
            with self.assertRaisesRegex(ValueError, "SHA"):
                require_reference_worktree(root, expected_sha="0" * 40)
            with self.assertRaisesRegex(ValueError, "full"):
                require_reference_worktree(root, expected_sha=sha[:8])
            (root / "untracked").write_text("uncommitted")
            with self.assertRaisesRegex(ValueError, "dirty"):
                require_reference_worktree(root, expected_sha=sha)
            (root / "untracked").unlink()
            (root / "source").write_text("changed")
            with self.assertRaisesRegex(ValueError, "dirty"):
                require_reference_worktree(root, expected_sha=sha)

    def test_adapter_does_not_fabricate_results(self):
        """Leave output absent when invalid adapter inputs are requested."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            with self.assertRaises((OSError, ValueError)):
                run_adapter(
                    "superdex",
                    Path("unused.json"),
                    output,
                    worktree=Path(tmp),
                    device="cpu",
                    build_type="release",
                    timeout_s=1,
                )
            self.assertFalse(output.exists())

    def _step_record(self):
        return {
            "step": 1,
            "time_s": 0.001,
            "implementation": "newton",
            "manifest_sha256": "a" * 64,
            "newton_sha": "b" * 40,
            "superdex_sha": "c" * 40,
            "device": "cpu",
            "hardware": "synthetic schema test",
            "build_type": "test",
            "warp_or_build_version": "test",
            "thread_count": 1,
            "actual_parameters_json": '{"dt_s":0.001}',
            "joint_q": [0.2, 0.01],
            "link_xform": [],
            "soft_com_m": [0, 0, 0],
            "node_positions_m": [],
            "min_det_f": 1.0,
            "penetration_m": 0.0,
            "normal_physical_force_n": [0, 0, 0],
            "tangential_physical_force_n": [0, 0, 0],
            "residual_contribution": {"q": [0, 0], "x": []},
            "physical_world_force_or_wrench": [],
            "generalized_physical_force": [0, 0],
            "q_sign_projection_error": 0.0,
            "x_sign_projection_error": 0.0,
            "nonlinear_iterations": 1,
            "linear_iterations": 1,
            "convergence_status": "NONLINEAR_MAX_ITERATIONS",
            "converged": False,
            "committed_unconverged": True,
            "rolled_back": False,
            "timing_s": {"assembly": 0, "collision": 0, "linear_solve": 0, "total": 0},
        }

    def test_step_record_contract(self):
        """Keep force and residual data separate and enforce terminal status consistency."""
        record = self._step_record()
        validate_step_record(record)
        for field in ("residual_contribution", "physical_world_force_or_wrench", "generalized_physical_force"):
            incomplete = copy.deepcopy(record)
            del incomplete[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_step_record(incomplete)
        record["rolled_back"] = True
        with self.assertRaisesRegex(ValueError, "rolled-back"):
            validate_step_record(record)

    def test_record_rejects_ambiguous_force(self):
        """Require separate physical force, generalized force and residual fields."""
        with self.assertRaisesRegex(ValueError, "force"):
            validate_step_record({"force": [1, 2, 3]})

    def test_unavailable_measurements_require_reasons(self):
        """Represent unavailable reference telemetry as null with exact field reasons."""
        record = self._step_record()
        record["implementation"] = "superdex"
        record["linear_iterations"] = None
        record["rolled_back"] = None
        record["committed_unconverged"] = None
        record["timing_s"]["linear_solve"] = None
        record["unavailable_fields"] = {
            "linear_iterations": "Pinned public SolverStats does not expose linear iterations",
            "rolled_back": "Pinned public API does not expose rollback",
            "committed_unconverged": "Pinned public API does not expose commit status",
            "timing_s.linear_solve": "Public timing reports aggregate solve duration only",
        }
        validate_step_record(record)
        for path in list(record["unavailable_fields"]):
            missing = copy.deepcopy(record)
            del missing["unavailable_fields"][path]
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "unavailable"):
                validate_step_record(missing)
        stale = copy.deepcopy(record)
        stale["unavailable_fields"]["time_s"] = "A fabricated unavailable marker for measured data"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_step_record(stale)

    def test_record_rejects_array_dimension_mismatch(self):
        """Reject incompatible force and residual dimensions instead of comparing wrong slices."""
        record = self._step_record()
        record["residual_contribution"]["q"] = [0.0]
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_step_record(record)

    def test_comparison_preserves_mapping_limits(self):
        """Report measured differences without turning intentionally different laws into an equality gate."""
        left = self._step_record()
        right = copy.deepcopy(left)
        right["implementation"] = "superdex"
        right["joint_q"][0] += 0.01
        mapping = MappingEntry(
            "contact.normal_law",
            "physics.contact",
            "hinge",
            "PolyReLU",
            "N/m^3",
            "INTENTIONALLY_DIFFERENT",
            "Different frozen laws",
            (),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            newton_output, superdex_output, output = (
                root / "newton.jsonl",
                root / "superdex.jsonl",
                root / "comparison.json",
            )
            newton_output.write_text(json.dumps(left) + "\n")
            superdex_output.write_text(json.dumps(right) + "\n")
            self.assertEqual(compare_runs(newton_output, superdex_output, [mapping], output), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "OBSERVED")
            self.assertAlmostEqual(report["differences"][0]["joint_q_max_absolute"], 0.01)
            blocked = MappingEntry(
                "control.command",
                "physics.drive",
                "force",
                "unknown",
                "N",
                "UNMAPPED",
                "No control adapter evidence",
                ("trajectory",),
            )
            self.assertEqual(compare_runs(newton_output, superdex_output, [mapping, blocked], output), 2)
            self.assertEqual(json.loads(output.read_text())["blocked_conclusions"], ["trajectory"])
            right["manifest_sha256"] = "d" * 64
            superdex_output.write_text(json.dumps(right) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest"):
                compare_runs(newton_output, superdex_output, [mapping], output)


def test_tiny_fixture(test, device):
    """Build the frozen fourteen-scalar fixture with explicit physics inputs."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, state, spec = fixture.model, fixture.state, fixture.spec
    test.assertEqual(model.joint_dof_count, 2)
    test.assertEqual(model.particle_count, 4)
    test.assertEqual(spec.scalar_dof_count, 14)
    np.testing.assert_array_equal(model.tet_indices.numpy(), [[0, 1, 2, 3]])
    test.assertEqual(
        {tuple(sorted(f)) for f in model.tri_indices.numpy()}, {tuple(sorted(f)) for f in spec.boundary_faces}
    )
    np.testing.assert_array_equal(state.particle_q.numpy(), np.asarray(spec.rest_positions, dtype=np.float32))
    np.testing.assert_array_equal(state.particle_qd.numpy(), np.zeros((4, 3)))
    np.testing.assert_array_equal(state.joint_q.numpy(), np.asarray(spec.q0, dtype=np.float32))
    np.testing.assert_array_equal(state.joint_qd.numpy(), np.asarray(spec.qd0, dtype=np.float32))
    np.testing.assert_allclose(model.particle_mass.numpy(), np.full(4, 0.001), rtol=2e-7)
    np.testing.assert_array_equal(model.particle_radius.numpy(), np.full(4, 0.001, dtype=np.float32))
    np.testing.assert_array_equal(model.tet_materials.numpy(), [[3846.15380859375, 5769.23095703125, 0.0]])
    np.testing.assert_array_equal(model.tri_materials.numpy(), np.zeros((4, 5)))
    np.testing.assert_array_equal(model.edge_bending_properties.numpy(), np.zeros((6, 2)))
    np.testing.assert_array_equal(model.body_mass.numpy(), [1.0, 0.5])
    np.testing.assert_allclose(model.body_com.numpy(), [[0.05, 0, 0], [0.02, 0, 0]])
    np.testing.assert_allclose(
        model.body_inertia.numpy(), [np.diag([0.002, 0.003, 0.004]), np.diag([0.001, 0.0015, 0.002])]
    )
    test.assertNotEqual(state.joint_q.ptr, fixture.state_next.joint_q.ptr)
    np.testing.assert_array_equal(state.joint_q.numpy(), fixture.state_next.joint_q.numpy())


for device in get_test_devices():
    add_function_test(TestMonolithicReference, "test_tiny_fixture", test_tiny_fixture, devices=[device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
