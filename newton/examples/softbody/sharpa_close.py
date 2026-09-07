# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Asset and target preparation for the internal monolithic G1H fixture."""

import csv
import hashlib
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import ShapeFlags

# This is an adapted URDF closure, not a reconstruction of the source CAD axes.
# Source name, URDF name, direction, zero offset [rad].
JOINT_MAPPING = (
    ("thumb_mcp_fe", "left_thumb_CMC_FE", 1, 0.0),
    ("thumb_mcp_aa", "left_thumb_CMC_AA", -1, 0.0),
    ("thumb_pip_fe", "left_thumb_MCP_FE", -1, 0.0),
    ("thumb_pip_aa", "left_thumb_MCP_AA", -1, 0.0),
    ("thumb_dip_fe", "left_thumb_IP", -1, 0.0),
    *(
        (f"{finger}_{source}", f"left_{finger}_{target}", -1, 0.0)
        for finger in ("index", "middle", "ring")
        for source, target in (("mcp_fe", "MCP_FE"), ("mcp_aa", "MCP_AA"), ("pip_fe", "PIP"), ("dip_fe", "DIP"))
    ),
    ("pinky_cmc_aa", "left_pinky_CMC", -1, 0.0),
    *(
        (f"pinky_{source}", f"left_pinky_{target}", -1, 0.0)
        for source, target in (("mcp_fe", "MCP_FE"), ("mcp_aa", "MCP_AA"), ("pip_fe", "PIP"), ("dip_fe", "DIP"))
    ),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ClosureTrajectory:
    """Interpolate time-stamped joint angles [rad] in model DOF order."""

    def __init__(self, path, joint_names, lower, upper, velocity, *, mapping=JOINT_MAPPING):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        if not rows or rows[0][0] != "Time(sec)":
            raise ValueError("Expected Time(sec) CSV header")
        channels = []
        for header in rows[0][1:]:
            match = re.fullmatch(
                r"Curve\d+:SubSystem_Left_Hand_Flex/FN_(.+)_STEP/Expression/Value\(Dimensionless\)", header
            )
            if match is None:
                raise ValueError(f"Invalid trajectory channel: {header}")
            channels.append(match[1])
        source = {row[0]: row for row in mapping}
        target = {row[1]: row for row in mapping}
        if (
            len(channels) != 22
            or len(set(channels)) != 22
            or set(channels) != set(source)
            or len(source) != 22
            or len(target) != 22
            or len(joint_names) != 22
            or set(joint_names) != set(target)
        ):
            raise ValueError("Trajectory must map exactly 22 unique channels and joints")
        try:
            data = np.asarray(rows[1:], dtype=np.float64)
        except ValueError as error:
            raise ValueError("Invalid trajectory numeric rows") from error
        if data.shape != (201, 23) or not np.isfinite(data).all():
            raise ValueError("Expected 201 finite trajectory frames")
        self.time = data[:, 0].copy()
        if self.time[0] != 0 or not np.all(np.diff(self.time) > 0) or abs(self.time[-1] - 4) > 1e-9:
            raise ValueError("Trajectory must increase strictly from 0 to 4 seconds")
        self.time[-1] = 4.0  # Normalize only the export's endpoint roundoff.
        self.names = tuple(joint_names)
        self.mapping = tuple(target[name] for name in joint_names)
        self.q = np.column_stack([data[:, 1 + channels.index(row[0])] * row[2] + row[3] for row in self.mapping])
        for row in self.mapping:
            if row[2] not in (-1, 1) or not np.isfinite(row[3]):
                raise ValueError("Mapping requires a signed axis and finite zero offset")
        self.slopes = np.diff(self.q, axis=0) / np.diff(self.time)[:, None]
        if np.any(self.q < np.asarray(lower) - 1e-7) or np.any(self.q > np.asarray(upper) + 1e-7):
            raise ValueError("Mapped trajectory exceeds URDF joint limits; do not clip")
        if np.any(np.abs(self.slopes) > np.asarray(velocity)):
            raise ValueError("Trajectory exceeds URDF target velocity limits")
        if np.max(np.abs(self.q[self.time >= 3] - self.q[-1])) > 1e-10:
            raise ValueError("Trajectory must include at least one second of final hold")

    def sample(self, time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Target time must be finite and nonnegative")
        if time_s == 0:
            return self.q[0].copy(), np.zeros(22)
        if time_s >= self.time[-1]:
            return self.q[-1].copy(), np.zeros(22)
        i = np.searchsorted(self.time, time_s, side="right") - 1
        return self.q[i] + (time_s - self.time[i]) * self.slopes[i], self.slopes[i].copy()


def load_hand(asset_dir, *, device, parameters):
    """Load the base URDF with explicit scalar properties and no collision."""
    asset_dir = Path(asset_dir).resolve()
    path = asset_dir / "left_sharpa_wave.urdf"
    root = ET.parse(path).getroot()
    links, joints = root.findall("link"), root.findall("joint")
    moving = {j.get("name"): j for j in joints if j.get("type") == "revolute"}
    if len(links) != 33 or len(joints) != 32 or set(moving) != {row[1] for row in JOINT_MAPPING}:
        raise ValueError("Expected the unmodified 33-link, 22-DoF base Sharpa hand")
    meshes = {}
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        prefix = "package://left_sharpa_wave/"
        if not name.startswith(prefix):
            raise ValueError(f"Unexpected Sharpa mesh URI: {name}")
        resolved = (asset_dir / name[len(prefix) :]).resolve()
        if not resolved.is_relative_to(asset_dir) or not resolved.is_file():
            raise ValueError(f"Missing or invalid Sharpa mesh: {name}")
        meshes[name] = sha256(resolved)
        mesh.set("filename", str(resolved))
    builder = newton.ModelBuilder(gravity=tuple(parameters["gravity"]))
    with wp.ScopedDevice(device):
        builder.add_urdf(ET.tostring(root, encoding="unicode"), floating=False, enable_self_collisions=False)
    builder.shape_flags = [
        int(f) & ~int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES) for f in builder.shape_flags
    ]
    names = []
    for j, label in enumerate(builder.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name not in moving:
            continue
        names.append(name)
        dof = builder.joint_qd_start[j]
        limit = moving[name].find("limit")
        for source, dest in (
            ("lower", "joint_limit_lower"),
            ("upper", "joint_limit_upper"),
            ("effort", "joint_effort_limit"),
            ("velocity", "joint_velocity_limit"),
        ):
            value = float(limit.get(source))
            if not np.isfinite(value) or (source in ("effort", "velocity") and value <= 0):
                raise ValueError(f"Invalid URDF {source}: {name}")
            getattr(builder, dest)[dof] = value
        cfg = parameters["joints"][name]
        for field in ("target_ke", "target_kd", "limit_ke", "limit_kd", "friction"):
            getattr(builder, f"joint_{field}")[dof] = cfg[field]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_armature[dof] = 0.0
        builder.joint_damping[dof] = 0.0
    before = np.asarray(builder.body_inertia, dtype=np.float64).reshape(-1, 3, 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = builder.finalize(device=device)
    after = model.body_inertia.numpy().astype(float)
    corrections = [
        {"link": model.body_label[i], "before": before[i].tolist(), "after": after[i].tolist()}
        for i in range(model.body_count)
        if not np.allclose(before[i], after[i], rtol=1e-5, atol=1e-15)
    ]
    if (model.body_count, model.joint_dof_count, model.shape_count) != (33, 22, 54):
        raise ValueError("Sharpa import lost bodies, joints, or meshes")
    manifest = {
        "urdf_sha256": sha256(path),
        "mesh_sha256": meshes,
        "joint_names": names,
        "body_count": model.body_count,
        "shape_count": model.shape_count,
        "mass_kg": float(model.body_mass.numpy().sum()),
        "inertia_corrections": corrections,
        "import_warnings": [str(w.message) for w in caught],
        "collision_enabled": False,
    }
    return model, manifest
