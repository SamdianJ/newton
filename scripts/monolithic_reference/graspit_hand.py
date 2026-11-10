# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare and import the external GraspIt HumanHand20DOF research asset.

Keep upstream geometry and joint frames unchanged. Use closed convex collision
proxies and uniform-density inertias for an explicitly uncalibrated first model.
"""

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.retained_assets import sha

UPSTREAM_COMMIT = "089c8802ca62ae2158d7a827acead94493e896b8"


def prepare(asset):
    """Write a portable derived URDF, convex meshes and their source manifest."""
    import trimesh

    asset = Path(asset).resolve()
    source = asset / "upstream"
    target = asset / "derived"
    (target / "collision").mkdir(parents=True, exist_ok=True)
    xml_path = source / "urdf/HumanHand20DOF.urdf"
    root = ET.parse(xml_path).getroot()
    source_hashes = {str(xml_path.relative_to(asset)): sha(xml_path)}
    rows = []
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_xml = visual.find("geometry/mesh")
        path = source / "data/HumanHand20DOF" / Path(mesh_xml.get("filename")).name
        source_hashes[str(path.relative_to(asset))] = sha(path)
        mesh = trimesh.load_mesh(path)
        mesh.apply_scale(np.fromstring(mesh_xml.get("scale"), sep=" "))
        hull = mesh.convex_hull
        hull.density = 1000.0  # kg/m^3; assumed bulk density, not a human-hand calibration.
        if not hull.is_volume or np.linalg.eigvalsh(hull.moment_inertia).min() <= 0:
            raise ValueError(f"Invalid convex mass geometry: {path}")
        name = link.get("name")
        collision_path = target / "collision" / (name + ".obj")
        hull.export(collision_path)
        mesh_xml.set("filename", "../upstream/data/HumanHand20DOF/" + path.name)
        collision = copy.deepcopy(visual)
        collision.tag = "collision"
        collision.find("geometry/mesh").set("filename", "collision/" + name + ".obj")
        collision.find("geometry/mesh").set("scale", "1 1 1")
        link.append(collision)
        old_mass = float(link.find("inertial/mass").get("value"))
        link.remove(link.find("inertial"))
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, hull.center_mass)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(hull.mass))
        inertia = hull.moment_inertia
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: str(inertia[i, j])
                for key, i, j in (
                    ("ixx", 0, 0),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyy", 1, 1),
                    ("iyz", 1, 2),
                    ("izz", 2, 2),
                )
            },
        )
        rows.append(
            {
                "link": name,
                "source_watertight": bool(mesh.is_watertight),
                "source_signed_volume_m3": float(mesh.volume),
                "original_mass_kg": old_mass,
                "mass_kg": float(hull.mass),
                "com_m": hull.center_mass.tolist(),
                "inertia_kg_m2": inertia.tolist(),
                "convex_vertices": len(hull.vertices),
                "convex_triangles": len(hull.faces),
                "extent_m": hull.extents.tolist(),
            }
        )
    if len(rows) != 16 or len(root.findall("joint[@type='revolute']")) != 20:
        raise ValueError("Expected 16 physical segments and 20 revolute joints")
    ET.indent(root)
    ET.ElementTree(root).write(target / "HumanHand20DOF.urdf", encoding="utf-8", xml_declaration=True)
    manifest = {
        "source_repository": "https://github.com/samarth-robo/GraspIt2URDF",
        "source_commit": UPSTREAM_COMMIT,
        "source_sha256": source_hashes,
        "derived_sha256": {str(p.relative_to(asset)): sha(p) for p in sorted(target.rglob("*")) if p.is_file()},
        "density_kg_m3": 1000.0,
        "total_mass_kg": sum(r["mass_kg"] for r in rows),
        "meshes": rows,
        "limitations": [
            "uncalibrated bulk density",
            "convex proxy fills concavities",
            "single-tree self contact unsupported",
            "SDF geometry not qualified",
        ],
    }
    (asset / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_hand(asset, *, device="cpu", contact=False, lift=False):
    """Import the hand with a fixed palm or one actuated vertical slide and optional mesh contact."""
    asset = Path(asset).resolve()
    manifest = json.loads((asset / "manifest.json").read_text())
    for relative, expected in (manifest["source_sha256"] | manifest["derived_sha256"]).items():
        if sha(asset / relative) != expected:
            raise ValueError(f"Asset hash mismatch: {relative}")
    if contact and not wp.get_device(device).is_cuda:
        raise ValueError("GraspIt mesh contact requires CUDA; CPU checks cover articulation only")
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.default_shape_cfg.density = 0
    builder.default_shape_cfg.margin = 0
    builder.default_shape_cfg.has_particle_collision = False
    builder.add_urdf(
        str(asset / "derived/HumanHand20DOF.urdf"),
        floating=None if lift else False,
        base_joint={
            "joint_type": newton.JointType.PRISMATIC,
            "linear_axes": [
                builder.JointDofConfig(
                    axis=(0, 1, 0),
                    target_ke=2e4,
                    target_kd=200,
                    effort_limit=100,
                    limit_lower=-0.01,
                    limit_upper=0.15,
                    limit_ke=2e4,
                    limit_kd=200,
                    armature=0,
                    damping=0,
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                )
            ],
        }
        if lift
        else None,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        force_position_velocity_actuation=True,
        mesh_maxhullvert=512,
    )
    # Retain the world connection; removing it leaves five roots in one articulation.
    # Other fixed frames only deepen the FP32 swept-FK roundoff bound.
    roots = [i for i, parent in enumerate(builder.joint_parent) if parent == -1]
    if len(roots) != 1:
        raise ValueError("Expected one fixed palm root")
    physical = []
    for shape, flags in enumerate(builder.shape_flags):
        if flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
            body = builder.shape_body[shape]
            if not np.allclose(builder.shape_transform[shape], wp.transform_identity()):
                raise ValueError("Expected identity source collision frames")
            physical.append((shape, builder.body_mass[body], builder.body_com[body], builder.body_inertia[body]))
    builder.collapse_fixed_joints(joints_to_keep=roots)
    # Each surviving physical body owns exactly one source solid. Restore its
    # COM inertia by rotation only: generic fixed collapse adds origin offsets.
    assigned = set()
    for shape, mass, com, inertia in physical:
        body = builder.shape_body[shape]
        if body < 0 or body in assigned:
            raise ValueError("Expected one source solid per retained physical body")
        assigned.add(body)
        pose = builder.shape_transform[shape]
        rotation = wp.quat_to_matrix(wp.transform_get_rotation(pose))
        rotated_inertia = rotation @ wp.mat33(inertia) @ wp.transpose(rotation)
        builder.body_mass[body], builder.body_inv_mass[body] = mass, 1.0 / mass
        builder.body_com[body] = wp.transform_point(pose, com)
        builder.body_inertia[body], builder.body_inv_inertia[body] = rotated_inertia, wp.inverse(rotated_inertia)
    if builder.joint_dof_count != 20 + int(lift):
        raise ValueError("Import changed the hand's 20 DOFs")
    for i in range(int(lift), builder.joint_dof_count):
        builder.joint_target_ke[i] = 0.2  # N m/rad
        builder.joint_target_kd[i] = 0.01  # N m s/rad
        builder.joint_effort_limit[i] = 0.05  # N m, finite actuator saturation.
        builder.joint_limit_ke[i] = 2.0
        builder.joint_limit_kd[i] = 0.02
        builder.joint_armature[i] = 0.0
        builder.joint_damping[i] = 0.0
    count = 0
    for i, flags in enumerate(builder.shape_flags):
        if flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
            count += 1
            active_flags = flags & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
            builder.shape_flags[i] = active_flags
            if contact:
                builder.shape_source[i].build_sdf(
                    device=device,
                    max_resolution=64,
                    margin=0.003,
                    narrow_band_range=(-0.005, 0.005),
                    texture_format="float32",
                    sign_method="winding",
                    cache_dir=asset / "sdf-cache",
                )
            else:
                builder.shape_flags[i] = active_flags & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    if count != 16:
        raise ValueError(f"Expected 16 collision proxies, got {count}")
    return builder, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.asset_dir), indent=2))
