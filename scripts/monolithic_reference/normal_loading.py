# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared frozen normal-loading model and control inputs; no solver or runner."""

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .internal_envelope import load_envelope

FIXTURE_DIRECTORY = Path(__file__).with_name("fixtures")
LEGACY_FIXTURE = FIXTURE_DIRECTORY / "normal_loading_draft_v1.json"
DEFAULT_FIXTURE = FIXTURE_DIRECTORY / "normal_loading_v2.json"
_STATUS_AXES = ("calibration", "support", "reference")


def load_fixture(path=DEFAULT_FIXTURE):
    """Read a supported version of the normal-loading input."""
    fixture = json.loads(Path(path).read_text())
    _validate_fixture(fixture)
    return fixture


def _axis_status(fixture, axis):
    if fixture["schema_version"] == "normal_loading_draft/v1":
        if axis == "calibration":
            return fixture["calibration_status"]
        if axis == "reference" and fixture["reference_envelope"] is not None:
            return "FROZEN"
        return "UNFROZEN"
    return fixture[f"{axis}_status"]


def _axis_provenance(fixture, axis):
    return fixture.get(f"{axis}_provenance")


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_status_axes(fixture):
    for axis in _STATUS_AXES:
        status = _axis_status(fixture, axis)
        allowed = (
            ("UNFROZEN", "CANDIDATE", "BLOCKED", "FROZEN")
            if axis == "reference"
            else (
                "UNFROZEN",
                "CANDIDATE",
                "FROZEN",
            )
        )
        if status not in allowed:
            raise ValueError(f"Invalid {axis}_status")
        provenance = _axis_provenance(fixture, axis)
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError(f"Invalid {axis}_provenance")
        if status == "FROZEN":
            if (
                provenance is None
                or not isinstance(provenance.get("version"), str)
                or not provenance["version"]
                or not _valid_sha256(provenance.get("sha256"))
            ):
                raise ValueError(f"Frozen {axis} evidence requires a nonempty version and SHA-256")
        if status == "BLOCKED" and (
            provenance is None or not isinstance(provenance.get("reason"), str) or not provenance["reason"].strip()
        ):
            raise ValueError(f"Blocked {axis} evidence requires a nonempty reason")
    calibration = _axis_provenance(fixture, "calibration")
    if calibration is not None and calibration.get("candidate_private_config") is not None:
        candidate = calibration["candidate_private_config"]
        if not isinstance(candidate, dict) or any(not isinstance(value, dict) for value in candidate.values()):
            raise ValueError("candidate_private_config must map device classes to config dictionaries")
    if _axis_status(fixture, "reference") == "FROZEN" and fixture["reference_envelope"] is None:
        raise ValueError("Frozen reference evidence requires reference_envelope")


def _validate_fixture(fixture):
    """Check only the concrete single-tet, Z-plane input this runner implements."""
    rigid, soft, contact, drive, limits = (
        fixture[name] for name in ("rigid", "soft", "contact", "drive", "acceptance")
    )
    schema_version = fixture["schema_version"]
    if (fixture["status"], schema_version) not in (
        ("DRAFT", "normal_loading_draft/v1"),
        ("DRAFT", "normal_loading/v2"),
        ("FROZEN", "normal_loading/v3"),
    ):
        raise ValueError("This runner only supports DRAFT normal_loading_draft/v1 or normal_loading/v2 input")
    if schema_version == "normal_loading_draft/v1":
        if fixture["calibration_status"] != "UNFROZEN" or fixture["reference_envelope"] is not None:
            raise ValueError("Legacy normal_loading_draft/v1 must remain unfrozen")
    else:
        _validate_status_axes(fixture)
    if fixture["status"] == "FROZEN":
        load_envelope(fixture)
    if rigid["shape_type"] != "infinite_plane" or rigid["joint_type"] != "PRISMATIC" or contact["quadrature"] != "P1Q3":
        raise ValueError("Only PRISMATIC infinite_plane with P1Q3 is supported")
    if rigid["approach_axis_world"] != [0.0, 0.0, 1.0]:
        raise ValueError("The plane and compressive approach axis must be world +Z")
    if (
        drive["kind"] != "external frozen joint force: kp*(q_target-q)+kd*(qd_target-qd)"
        or drive["trajectory"] != "quintic position and analytic velocity, followed by constant settle target"
    ):
        raise ValueError("Only the declared frozen-force quintic drive is supported")
    for values, positive, nonnegative in (
        (fixture, ("dt_s",), ()),
        (rigid, ("mass_kg",), ()),
        (soft, ("density_kg_m3", "mu_pa", "lambda_pa"), ("particle_radius_m",)),
        (contact, ("stiffness_n_m3",), ("soft_contact_gap_m", "shape_margin_m")),
        (drive, ("kp_n_m", "free_end_s", "loading_end_s", "free_target_m", "loading_target_m"), ("kd_n_s_m",)),
        (
            limits,
            (
                "min_det_f",
                "penetration_acceptance_limit_m",
                "delta_soft_min_m",
                "force_floor_n",
                "force_balance_relative_tolerance",
                "linear_tolerance",
            ),
            (),
        ),
    ):
        for name in positive + nonnegative:
            value = values[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                or (name in positive and value == 0)
            ):
                raise ValueError(f"Invalid finite range for {name}")
    for values, names in (
        (fixture, ("substeps",)),
        (limits, ("minimum_substeps", "onset_consecutive_substeps", "settled_window_substeps")),
    ):
        if any(type(values[name]) is not int or values[name] <= 0 for name in names):
            raise ValueError("Step counts must be positive integers")
    if type(limits["maximum_consecutive_non_success"]) is not int or limits["maximum_consecutive_non_success"] < 0:
        raise ValueError("Maximum consecutive failures must be a nonnegative integer")
    for name in ("minimum_converged_ratio", "free_space_completion_minimum"):
        if not 0 < limits[name] <= 1:
            raise ValueError(f"Invalid acceptance ratio {name}")
    if not (
        drive["free_end_s"] < drive["loading_end_s"] < fixture["substeps"] * fixture["dt_s"]
        and drive["free_target_m"] < drive["loading_target_m"]
    ):
        raise ValueError("Drive must have ordered free/loading/settle phases and increasing targets")
    for values, name in ((rigid, "inertia_kg_m2"), (rigid, "reference_point_body_m"), (fixture, "gravity_m_s2")):
        vector = np.asarray(values[name], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all() or (name == "inertia_kg_m2" and np.any(vector <= 0)):
            raise ValueError(f"Invalid three-vector {name}")
    interval = np.asarray(limits["closure_secant_interval_m"], dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not 0 < interval[0] < interval[1]:
        raise ValueError("Secant closure interval must be finite, positive and ordered")
    if (
        soft["tet_indices"] != [[0, 1, 2, 3]]
        or soft["fixed_nodes"] != [False, False, False, True]
        or soft["probe_nodes"] != [3]
        or soft["surface_nodes"] != [0, 1, 2]
        or soft["damping"] != 0
    ):
        raise ValueError("Only one ordered tetrahedron with fixed apex, three base nodes and zero damping is supported")
    positions = np.asarray(soft["rest_positions_m"], dtype=float)
    if positions.shape != (4, 3) or not np.isfinite(positions).all():
        raise ValueError("Expected four finite three-dimensional rest positions")
    if (
        not np.all(positions[:3, 2] == positions[0, 2])
        or positions[3, 2] <= positions[0, 2]
        or np.linalg.det((positions[1:] - positions[0]).T) <= 0
    ):
        raise ValueError("Expected a positive-volume tetrahedron with horizontal base below the apex")
    if (
        not math.isfinite(rigid["initial_plane_z_m"])
        or rigid["initial_plane_z_m"] + soft["particle_radius_m"] + contact["shape_margin_m"] >= positions[0, 2]
    ):
        raise ValueError("The initial plane must lie below the base with positive physical clearance")


def command_at(time_s, fixture):
    """Evaluate the declared quintic command without using it as measured closure."""
    drive = fixture["drive"]
    if time_s < drive["free_end_s"]:
        start, duration, q0, q1, phase = 0.0, drive["free_end_s"], 0.0, drive["free_target_m"], "free_space"
    elif time_s < drive["loading_end_s"]:
        start, duration = drive["free_end_s"], drive["loading_end_s"] - drive["free_end_s"]
        q0, q1, phase = drive["free_target_m"], drive["loading_target_m"], "loading"
    else:
        return drive["loading_target_m"], 0.0, "settle"
    t = min(max((time_s - start) / duration, 0.0), 1.0)
    position = t**3 * (10.0 - 15.0 * t + 6.0 * t * t)
    velocity = 30.0 * t * t * (1.0 - t) ** 2 / duration
    return q0 + (q1 - q0) * position, (q1 - q0) * velocity, phase


def build_model(fixture, *, device):
    """Build the declared single-finger articulation and anchored tetrahedron."""
    _validate_fixture(fixture)
    rigid, soft, contact = fixture["rigid"], fixture["soft"], fixture["contact"]
    builder = newton.ModelBuilder(gravity=tuple(fixture["gravity_m_s2"]))
    body = builder.add_link(mass=rigid["mass_kg"], inertia=wp.diag(wp.vec3(*rigid["inertia_kg_m2"])))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=tuple(rigid["approach_axis_world"]),
        parent_xform=wp.transform((0.0, 0.0, rigid["initial_plane_z_m"]), wp.quat_identity()),
        armature=0.0,
        damping=0.0,
        friction=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=0.0,
        target_kd=0.0,
        actuator_mode=newton.JointTargetMode.NONE,
    )
    builder.add_articulation([joint])
    builder.add_shape_plane(
        body=body, width=0.0, length=0.0, cfg=builder.ShapeConfig(density=0.0, margin=contact["shape_margin_m"])
    )
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=soft["rest_positions_m"],
        indices=np.asarray(soft["tet_indices"]).ravel().tolist(),
        density=soft["density_kg_m3"],
        k_mu=soft["mu_pa"],
        k_lambda=soft["lambda_pa"],
        k_damp=soft["damping"],
        particle_radius=soft["particle_radius_m"],
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    for particle, fixed in enumerate(soft["fixed_nodes"]):
        if fixed:
            builder.particle_mass[particle] = 0.0
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    return model
