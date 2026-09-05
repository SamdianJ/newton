# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the P1Q3 calibration harness against measurable sampling contracts."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.calibrate_p1q3 import (
    _freeze_audit,
    _is_exact_supported_fixture,
    _source_provenance,
    _validate_new_output,
    calibrate,
    measure_motion,
    measure_motion_comparison,
    measure_static,
    refined_tetrahedron,
)
from scripts.monolithic_reference.p1q3_oracle import integrate_contact_over_mesh


def test_uniform_plane_refinement(test, device):
    """Preserve geometry and uniform-plane total force under boundary refinement."""
    rows = [measure_static(device, level=level, case="plane") for level in range(3)]
    test.assertEqual([row["boundary_face_count"] for row in rows], [4, 16, 64])
    test.assertEqual([row["tet_count"] for row in rows], [1, 8, 64])
    forces = np.array([row["normal_force_n"] for row in rows])
    test.assertLessEqual(float(np.max(np.abs(forces / forces[-1] - 1.0))), 0.05)
    np.testing.assert_allclose(forces, 0.056, rtol=2e-5)
    for row in rows:
        test.assertEqual(row["support_envelope"], "SUPPORTED")
        test.assertEqual(row["same_mesh_quadrature_gate"], "PASS")
        test.assertEqual(row["same_mesh_oracle"]["integration_domain"], "full_tet_boundary")
        test.assertIsNone(row["full_boundary_oracle_observation"])
        test.assertLessEqual(row["same_mesh_oracle"]["relative_force_error"], 0.05)
        test.assertEqual(row["evaluation_status"], 0)
        test.assertLess(row["contact_sign_error"], 1e-4)
        test.assertGreater(row["active_sample_count"], 0)
    for level in range(3):
        points, tets = refined_tetrahedron(level)
        determinant = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
        test.assertTrue(np.all(determinant > 0.0))
        test.assertAlmostEqual(float(determinant.sum() / 6.0), 0.04**3 / 6.0, delta=1e-11)


def test_sharp_feature_envelope(test, device):
    """Distinguish an inter-sample miss from a broad-feature positive control."""
    sharp = measure_static(device, level=0, case="sharp_box")
    broad = measure_static(device, level=0, case="broad_box")
    test.assertGreater(sharp["witness_penetration_m"], 0.0)
    test.assertEqual(sharp["active_sample_count"], 0)
    test.assertEqual(sharp["normal_force_n"], 0.0)
    test.assertEqual(sharp["sampling_classification"], "UNSUPPORTED_SAMPLING_MISS")
    test.assertEqual(sharp["support_envelope"], "UNSUPPORTED")
    test.assertEqual(sharp["support_reason"], "legacy_inter_sample_box_w2_l012")
    test.assertGreater(broad["active_sample_count"], 0)
    test.assertGreater(broad["normal_force_n"], 0.0)
    test.assertEqual(broad["evaluation_status"], 0)
    test.assertEqual(broad["support_envelope"], "SUPPORTED")
    test.assertEqual(broad["same_mesh_quadrature_gate"], "PASS")
    resolved = [measure_static(device, level=level, case="resolved_sharp_box") for level in (2, 3, 4)]
    test.assertEqual([row["surface_multiplier"] for row in resolved], [1, 4, 16])
    for row in resolved:
        test.assertEqual(row["support_envelope"], "SUPPORTED")
        test.assertEqual(row["manifest"]["support_fixture_id"], "sample_between_box_w10_l234_v1")
        test.assertTrue(_is_exact_supported_fixture(row["manifest"]))
        test.assertEqual(row["sampling_classification"], "DETECTED")
        test.assertGreater(row["same_mesh_oracle"]["active_area_m2"], 0.0)
        test.assertGreater(row["active_sample_count"], 0)
        test.assertEqual(row["same_mesh_quadrature_gate"], "PASS")
    changed_width = {**resolved[0]["manifest"], "feature_width_m": 0.0101}
    test.assertFalse(_is_exact_supported_fixture(changed_width))


def test_independent_plane_oracle(test, device):
    """Integrate the hinge independently of Warp records and P1Q3 slots."""
    del device
    points, _ = refined_tetrahedron(0)
    faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    oracle = integrate_contact_over_mesh(
        points,
        faces,
        shape="plane",
        center=np.array((0.0, 0.0, 0.00025)),
        scale=np.zeros(3),
        particle_radius=0.0001,
        stiffness=2.0e5,
    )
    test.assertGreater(oracle.force_magnitude, 0.056)
    test.assertLess(abs(0.056 / oracle.force_magnitude - 1.0), 0.05)
    test.assertLessEqual(oracle.force_magnitude_absolute_error / oracle.force_magnitude, 1.0e-3)


def test_independent_oracle_resolves_sample_between_box(test, device):
    """Do not infer zero contact when all three production samples miss."""
    del device
    points, _ = refined_tetrahedron(0)
    oracle = integrate_contact_over_mesh(
        points,
        np.array(((0, 2, 1),), dtype=np.int32),
        shape="box",
        center=np.array((0.04 / 3, 0.04 / 3, -0.0004)),
        scale=np.array((0.001, 0.001, 0.0006)),
        particle_radius=0.0001,
        stiffness=2.0e5,
    )
    test.assertGreater(oracle.force_magnitude, 0.0)
    test.assertGreater(oracle.active_area, 0.0)
    shallower = integrate_contact_over_mesh(
        points,
        np.array(((0, 2, 1),), dtype=np.int32),
        shape="box",
        center=np.array((0.04 / 3, 0.04 / 3, -0.0004)),
        scale=np.array((0.001, 0.001, 0.0006)),
        particle_radius=0.0001,
        stiffness=2.0e5,
        max_depth=6,
    )
    test.assertLess(abs(shallower.force_magnitude / oracle.force_magnitude - 1.0), 1.0e-3)


def test_c4_gate_is_scoped_and_machine_readable(test, device):
    """Never freeze C4 without the required dynamic trial."""
    result = calibrate(device, steps=0, dt=0.001)
    test.assertEqual(result["c4_gate"], "FAIL")
    test.assertEqual(result["same_mesh_quadrature_gate"], "PASS")
    test.assertEqual(result["sampling_detection_gate"], "PASS")
    test.assertEqual(result["supported_local_motion_gate"], "NOT_RUN")
    test.assertEqual(result["support_envelope"]["status"], "CANDIDATE")
    test.assertEqual(result["dirichlet_mass_audit_gate"], "PASS")
    unsupported = {row["case"] for row in result["static"] if row["support_envelope"] == "UNSUPPORTED"}
    test.assertTrue({"shared_edge", "shared_vertex", "sharp_box"}.issubset(unsupported))


def test_supported_shared_feature_motion(test, device):
    """Meet the 10% motion gate for resolved shared-edge and shared-vertex contact."""
    for case in ("supported_edge", "supported_vertex"):
        static = [measure_static(device, level=level, case=case) for level in (2, 3, 4)]
        test.assertEqual([row["surface_multiplier"] for row in static], [1, 4, 16])
        test.assertEqual([row["boundary_face_count"] for row in static], [64, 256, 1024])
        for row in static:
            test.assertEqual(row["support_envelope"], "SUPPORTED")
            test.assertEqual(
                row["manifest"]["support_fixture_id"], f"shared_{case.removeprefix('supported_')}_r30_l234_v1"
            )
            test.assertTrue(_is_exact_supported_fixture(row["manifest"]))
            test.assertEqual(row["sampling_classification"], "DETECTED")
            test.assertLessEqual(row["same_mesh_oracle"]["relative_force_error"], 0.1)
        rows, comparison = measure_motion_comparison(device, case=case, steps=20, dt=0.001)
        test.assertEqual(comparison["gate"], "PASS")
        test.assertTrue(all(row["all_steps_success"] for row in rows))
        for metric in comparison["metrics"].values():
            test.assertEqual(metric["normalization"], "RESOLVED")
            test.assertLessEqual(max(metric["relative_difference_from_finest"]), 0.1)


def _synthetic_supported_manifest(fixture_id, level, surface_multiplier):
    manifest = {
        "support_fixture_id": fixture_id,
        "level": level,
        "surface_multiplier": surface_multiplier,
        "length_m": 0.04,
        "density_kg_m3": 1000.0,
        "contact_stiffness_n_m3": 2.0e5,
        "particle_radius_m": 0.0001,
        "soft_contact_gap_m": 0.0002,
        "shape_margin_m": 0.0,
        "solver_parameters": {"newton_max_iterations": 20},
        "shape_scale": [[0.0, 0.0, 0.0]],
        "nominal_initial_penetration_m": None,
    }
    if fixture_id == "plane_uniform_face_z025mm_l012_v2":
        manifest.update(case="plane", shape_kind="plane", shape_center_m=(0.0, 0.0, 0.00025))
    elif fixture_id == "broad_box_full_face_l012_v1":
        manifest.update(
            case="broad_box",
            shape_kind="box",
            feature_width_m=0.08,
            shape_center_m=(0.04 / 3, 0.04 / 3, -0.0004),
        )
    elif fixture_id == "sample_between_box_w10_l234_v1":
        manifest.update(
            case="resolved_sharp_box",
            shape_kind="box",
            feature_width_m=0.01,
            shape_center_m=(0.04 / 3, 0.04 / 3, -0.0004),
        )
    else:
        location = "edge" if fixture_id == "shared_edge_r30_l234_v1" else "vertex"
        direction = np.asarray((0.0, -1.0, -1.0) if location == "edge" else (-1.0, -1.0, -1.0))
        anchor = np.asarray((0.02, 0.0, 0.0) if location == "edge" else (0.0, 0.0, 0.0))
        center = anchor + direction / np.linalg.norm(direction) * (0.03 + 0.0001 - 0.008)
        manifest.update(
            case=f"supported_{location}",
            shape_kind="sphere",
            feature_width_m=0.06,
            shape_center_m=center.tolist(),
            shape_scale=[[0.03, 0.03, 0.03]],
            nominal_initial_penetration_m=0.008,
        )
    return manifest


def _synthetic_freeze_evidence():
    fixture_levels = {
        "plane_uniform_face_z025mm_l012_v2": (0, 1, 2),
        "broad_box_full_face_l012_v1": (0, 1, 2),
        "sample_between_box_w10_l234_v1": (2, 3, 4),
        "shared_edge_r30_l234_v1": (2, 3, 4),
        "shared_vertex_r30_l234_v1": (2, 3, 4),
    }
    support = {
        "id": "p1q3-resolved-local-v2",
        "status": "FROZEN",
        "supported_fixture_ids": sorted(fixture_levels),
    }
    static = []
    for fixture_id, levels in fixture_levels.items():
        for level, multiplier in zip(levels, (1, 4, 16), strict=True):
            threshold = 0.05 if fixture_id.startswith("plane_") else 0.1
            static.append(
                {
                    "case": _synthetic_supported_manifest(fixture_id, level, multiplier)["case"],
                    "level": level,
                    "surface_multiplier": multiplier,
                    "support_envelope": "SUPPORTED",
                    "manifest": _synthetic_supported_manifest(fixture_id, level, multiplier),
                    "normal_force_n": 1.0,
                    "evaluation_status": 0,
                    "same_mesh_quadrature_gate": "PASS",
                    "same_mesh_quadrature_threshold": threshold,
                    "same_mesh_oracle": {
                        "integration_domain": "full_tet_boundary",
                        "normal_force_n": 1.0,
                        "relative_force_error": 0.0,
                    },
                    "sampling_classification": "DETECTED",
                }
            )
    motion = []
    comparisons = []
    factors = (1.05, 1.02, 1.0)
    for case, fixture_id in (
        ("supported_edge", "shared_edge_r30_l234_v1"),
        ("supported_vertex", "shared_vertex_r30_l234_v1"),
    ):
        metric_values = {
            "rigid_displacement_m": [factor * 1.0e-4 for factor in factors],
            "max_surface_penetration_m": [factor * 1.0e-3 for factor in factors],
            "peak_surface_penetration_m": [factor * 1.0e-3 for factor in factors],
            "soft_volume_mean_displacement_m": [factor * 2.0e-4 for factor in factors],
        }
        for level, multiplier, factor in zip((2, 3, 4), (1, 4, 16), factors, strict=True):
            trace_row = {
                "converged": True,
                "rolled_back": False,
                "sampling_miss": False,
                "rigid_displacement_m": factor * 1.0e-4,
                "max_surface_penetration_m": factor * 1.0e-3,
                "soft_volume_mean_displacement_m": [0.0, 0.0, factor * 2.0e-4],
            }
            motion.append(
                {
                    "case": case,
                    "level": level,
                    "manifest": _synthetic_supported_manifest(fixture_id, level, multiplier),
                    "applied_joint_force_n": 0.1,
                    "dt_s": 0.001,
                    "steps": 20,
                    "trace": [deepcopy(trace_row) for _ in range(20)],
                    "all_steps_success": True,
                }
            )
        metrics = {}
        for name, values in metric_values.items():
            relative = [abs(value - values[-1]) / abs(values[-1]) for value in values]
            floor = float(np.spacing(np.float32(0.04))) if "displacement" in name else 0.0
            metrics[name] = {
                "values": values,
                "relative_difference_from_finest": relative,
                "absolute_difference_from_finest": [abs(value - values[-1]) for value in values],
                "reference_floor_m": floor,
                "normalization": "RESOLVED",
            }
        comparisons.append(
            {
                "case": case,
                "role": "HARD_SUPPORTED_GATE",
                "gate": "PASS",
                "threshold": 0.1,
                "metrics": metrics,
                "sampling_gate": "PASS",
            }
        )
    return [
        {
            "device": name,
            "c4_gate": "PASS",
            "support_envelope": deepcopy(support),
            "same_mesh_quadrature_gate": "PASS",
            "sampling_detection_gate": "PASS",
            "dirichlet_mass_audit_gate": "PASS",
            "supported_local_motion_gate": "PASS",
            "static": deepcopy(static),
            "motion": deepcopy(motion),
            "local_motion_comparisons": deepcopy(comparisons),
        }
        for name in ("cpu", "cuda:0")
    ]


def test_freeze_audit_rejects_incomplete_provenance(test, device):
    """Require exactly one CPU/CUDA result and finite evidence before freezing."""
    del device
    root = Path(__file__).resolve().parents[2]
    provenance = _source_provenance(root)
    evidence = _synthetic_freeze_evidence()
    frozen, reasons = _freeze_audit(
        requested_devices=["cpu", "cuda:0"],
        evidence=evidence,
        dynamic_steps=20,
        dt=0.001,
        git_dirty=False,
        nonfinite_fields=[],
        **provenance,
    )
    test.assertTrue(frozen)
    test.assertEqual(reasons, [])
    frozen, reasons = _freeze_audit(
        requested_devices=["cpu", "cuda:0"],
        evidence=evidence,
        dynamic_steps=20,
        dt=0.001,
        git_dirty=False,
        nonfinite_fields=[],
    )
    test.assertFalse(frozen)
    test.assertIn("invalid_source_provenance", reasons)
    for requested, rows, nonfinite, reason in (
        (["cpu"], evidence[:1], [], "requires_exactly_one_cpu_and_one_cuda_request"),
        (["cpu", "cpu"], [evidence[0], evidence[0]], [], "requires_exactly_one_cpu_and_one_cuda_request"),
        (["cpu", "cuda:0"], evidence, ["/evidence/0/value"], "nonfinite_evidence"),
    ):
        frozen, reasons = _freeze_audit(
            requested_devices=requested,
            evidence=rows,
            dynamic_steps=20,
            dt=0.001,
            git_dirty=False,
            nonfinite_fields=nonfinite,
            **provenance,
        )
        test.assertFalse(frozen)
        test.assertIn(reason, reasons)

    for mutation, reason in (
        (lambda rows: rows[0].pop("same_mesh_quadrature_gate"), "incomplete_or_failed_device_subgates"),
        (
            lambda rows: rows[0]["support_envelope"]["supported_fixture_ids"].pop(),
            "invalid_supported_fixture_ids",
        ),
        (lambda rows: rows[0]["local_motion_comparisons"].pop(), "invalid_supported_motion_comparisons"),
        (
            lambda rows: rows[0]["static"].pop(
                next(
                    index
                    for index, row in enumerate(rows[0]["static"])
                    if row["manifest"]["support_fixture_id"] == "plane_uniform_face_z025mm_l012_v2"
                    and row["level"] == 1
                )
            ),
            "invalid_supported_fixture_evidence",
        ),
        (
            lambda rows: next(
                row
                for row in rows[0]["static"]
                if row["manifest"]["support_fixture_id"] == "plane_uniform_face_z025mm_l012_v2"
            )["same_mesh_oracle"].pop("integration_domain"),
            "invalid_supported_fixture_evidence",
        ),
        (
            lambda rows: next(
                row for row in rows[0]["static"] if row["manifest"]["support_fixture_id"] == "shared_edge_r30_l234_v1"
            )["same_mesh_oracle"].update(relative_force_error=0.01),
            "invalid_supported_fixture_evidence",
        ),
        (
            lambda rows: rows[0]["motion"].pop(
                next(
                    index
                    for index, row in enumerate(rows[0]["motion"])
                    if row["case"] == "supported_edge" and row["level"] == 3
                )
            ),
            "invalid_supported_motion_evidence",
        ),
        (
            lambda rows: rows[0]["local_motion_comparisons"][0]["metrics"].clear(),
            "invalid_supported_motion_metrics",
        ),
        (
            lambda rows: rows[0]["local_motion_comparisons"][0]["metrics"]["rigid_displacement_m"].update(
                relative_difference_from_finest=[0.0, 0.0, 0.0]
            ),
            "invalid_supported_motion_metrics",
        ),
    ):
        invalid = deepcopy(evidence)
        mutation(invalid)
        frozen, reasons = _freeze_audit(
            requested_devices=["cpu", "cuda:0"],
            evidence=invalid,
            dynamic_steps=20,
            dt=0.001,
            git_dirty=False,
            nonfinite_fields=[],
            **provenance,
        )
        test.assertFalse(frozen)
        test.assertIn(reason, reasons)

    invalid_sources = dict(provenance["source_sha256"])
    invalid_sources.pop("newton/_src/solvers/monolithic/contact.py")
    frozen, reasons = _freeze_audit(
        requested_devices=["cpu", "cuda:0"],
        evidence=evidence,
        dynamic_steps=20,
        dt=0.001,
        git_dirty=False,
        nonfinite_fields=[],
        source_sha256=invalid_sources,
        source_set_sha256=provenance["source_set_sha256"],
    )
    test.assertFalse(frozen)
    test.assertIn("invalid_source_provenance", reasons)


def test_artifact_output_is_immutable(test, device):
    """Fail before calibration when the requested artifact already exists."""
    del device
    with TemporaryDirectory() as directory:
        output = Path(directory) / "existing.json"
        output.touch()
        with test.assertRaises(FileExistsError):
            _validate_new_output(output)
        _validate_new_output(Path(directory) / "new.json")


def test_c4_source_provenance_is_closed(test, device):
    """Hash every monolithic module used by the dynamic C4 path."""
    del device
    root = Path(__file__).resolve().parents[2]
    provenance = _source_provenance(root)
    required = {
        "scripts/monolithic_reference/calibrate_p1q3.py",
        "scripts/monolithic_reference/p1q3_oracle.py",
        "newton/_src/solvers/monolithic/collision.py",
        "newton/_src/solvers/monolithic/contact.py",
        "newton/_src/solvers/monolithic/tet.py",
        "newton/_src/solvers/monolithic/linear.py",
        "newton/_src/solvers/monolithic/articulation.py",
        "newton/_src/solvers/monolithic/solver_monolithic.py",
    }
    test.assertEqual(set(provenance["source_sha256"]), required)
    test.assertEqual(len(provenance["source_set_sha256"]), 64)


def test_unsupported_probe_classes_do_not_overlap_supported_class(test, device):
    """Keep legacy failed probes distinct from the resolved R30 support fixtures."""
    result = calibrate(device, steps=0, dt=0.001)
    unsupported = set(result["support_envelope"]["unsupported_probe_classes"])
    test.assertEqual(
        unsupported,
        {"legacy_curved_local_patch_l012", "legacy_inter_sample_box_w2_l012"},
    )
    supported_classes = {
        row["manifest"]["declared_support_class"] for row in result["static"] if row["support_envelope"] == "SUPPORTED"
    }
    test.assertTrue(unsupported.isdisjoint(supported_classes))


def test_broad_curvature_control(test, device):
    """Require real contact and resolved motion in broad edge/vertex controls."""
    for case in ("wide_edge_160", "wide_vertex_160"):
        row = measure_static(device, level=0, case=case)
        test.assertGreater(row["active_sample_count"], 0)
        test.assertGreater(row["normal_force_n"], 0)
        motion = measure_motion(device, level=0, case=case, steps=2, dt=0.001)
        test.assertTrue(motion["all_steps_success"])
        test.assertFalse(any(row["sampling_miss"] for row in motion["trace"]))
        test.assertGreater(
            np.linalg.norm(motion["trace"][-1]["soft_volume_mean_displacement_m"]), float(np.spacing(np.float32(0.04)))
        )


class TestMonolithicSupport(unittest.TestCase):
    """Verify CPU and CUDA measurement behavior without claiming unsupported cases pass."""


for function in (
    test_uniform_plane_refinement,
    test_sharp_feature_envelope,
    test_independent_plane_oracle,
    test_independent_oracle_resolves_sample_between_box,
    test_c4_gate_is_scoped_and_machine_readable,
    test_supported_shared_feature_motion,
    test_freeze_audit_rejects_incomplete_provenance,
    test_artifact_output_is_immutable,
    test_c4_source_provenance_is_closed,
    test_unsupported_probe_classes_do_not_overlap_supported_class,
    test_broad_curvature_control,
):
    add_function_test(TestMonolithicSupport, function.__name__, function, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
