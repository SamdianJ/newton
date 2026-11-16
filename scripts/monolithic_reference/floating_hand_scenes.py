# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allegro left-hand scenes: wrist or floating mount, FREE objects and a capsule chain.

The hand is the pinned ``asset/monolithic/data/inputs/allegro_left`` copy of
newton-assets ``wonik_allegro`` (provenance in its ``provenance.json``). Finger
and wrist gains follow the task profile ``manip/hands/allegro_left.json`` of
monolithic-tasks, whose sha256 the provenance records.
"""

import json

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.retained_assets import ASSET_ROOT, INPUTS, retained_hashes, verify_retained

ALLEGRO = INPUTS / "allegro_left"
URDF = ALLEGRO / "wonik_allegro/urdf/allegro_hand_description_left.urdf"
HAND_DOFS = 16
HAND_BODIES = 21
HAND_COLLISION_SHAPES = 21
FINGER_KE, FINGER_KD, FINGER_ARMATURE = 150.0, 5.0, 0.01
WRIST_LINEAR_KE, WRIST_LINEAR_KD = 20000.0, 200.0
WRIST_ANGULAR_KE, WRIST_ANGULAR_KD = 200.0, 5.0
WRIST_LINEAR_LIMIT, WRIST_ANGULAR_LIMIT = 0.3, 1.5
WRIST_LINK_MASS, WRIST_LINK_INERTIA = 0.01, 1e-5
WRIST_JOINTS = (
    ("px", "prismatic", (1.0, 0.0, 0.0)),
    ("py", "prismatic", (0.0, 1.0, 0.0)),
    ("pz", "prismatic", (0.0, 0.0, 1.0)),
    ("rz", "revolute", (0.0, 0.0, 1.0)),
    ("ry", "revolute", (0.0, 1.0, 0.0)),
    ("rx", "revolute", (1.0, 0.0, 0.0)),
)
MOUNTS = ("wrist", "floating")


def verify_allegro():
    """Check every pinned Allegro file against the tracked manifest and return the asset directory."""
    prefix = str(ALLEGRO.relative_to(ASSET_ROOT)) + "/"
    hashes = retained_hashes()
    listed = sorted(name for name in hashes if name.startswith(prefix))
    if not listed:
        raise ValueError(f"asset/monolithic/files.sha256 lists no {prefix} entries")
    if not ALLEGRO.is_dir():
        raise FileNotFoundError(f"Allegro asset missing: {ALLEGRO} (pinned by P9d PR0; link or publish the assets)")
    present = sorted(str(p.relative_to(ASSET_ROOT)) for p in ALLEGRO.rglob("*") if p.is_file())
    if present != listed:
        raise ValueError(f"Allegro asset files differ from the manifest: {sorted(set(present) ^ set(listed))}")
    for name in listed:
        verify_retained(ASSET_ROOT / name, hashes)
    return ALLEGRO


def provenance():
    return json.loads((verify_allegro() / "provenance.json").read_text())


def _short(label):
    return label.rsplit("/", 1)[-1]


def _add_wrist(builder, mount):
    """Fixed mount frame, then Px Py Pz Rz Ry Rx; return the last wrist link and the wrist joints."""
    inertia = wp.diag(wp.vec3(WRIST_LINK_INERTIA, WRIST_LINK_INERTIA, WRIST_LINK_INERTIA))
    base = builder.add_link(mass=WRIST_LINK_MASS, inertia=inertia, label="wrist_mount")
    joints = [builder.add_joint_fixed(-1, base, parent_xform=mount, label="wrist_fixed")]
    parent = base
    for name, kind, axis in WRIST_JOINTS:
        link = builder.add_link(mass=WRIST_LINK_MASS, inertia=inertia, label=f"wrist_{name}_link")
        if kind == "prismatic":
            joint = builder.add_joint_prismatic(
                parent,
                link,
                axis=axis,
                target_ke=WRIST_LINEAR_KE,
                target_kd=WRIST_LINEAR_KD,
                limit_lower=-WRIST_LINEAR_LIMIT,
                limit_upper=WRIST_LINEAR_LIMIT,
                label=f"wrist_{name}",
            )
        else:
            limit = np.pi if name == "rz" else WRIST_ANGULAR_LIMIT
            joint = builder.add_joint_revolute(
                parent,
                link,
                axis=axis,
                target_ke=WRIST_ANGULAR_KE,
                target_kd=WRIST_ANGULAR_KD,
                limit_lower=-limit,
                limit_upper=limit,
                label=f"wrist_{name}",
            )
        joints.append(joint)
        parent = link
    builder.add_articulation(joints, label="wrist")
    return parent, joints


def add_allegro(builder, *, mount, palm_pose, self_collisions=True):
    """Add the hand with a 6-joint wrist (``mount='wrist'``) or a FREE root (``mount='floating'``).

    ``palm_pose`` is the palm frame in world. For the wrist it is the fixed mount
    frame, so zero wrist coordinates place the palm there. Every moving DoF gets
    POSITION_VELOCITY targets; fingers use the task profile gains.
    Returns ``{"dofs": [...], "bodies": {name: index}, "root_joint": j, "first_dof": d}``.
    """
    if mount not in MOUNTS:
        raise ValueError(f"mount must be one of {MOUNTS}, got {mount!r}")
    verify_allegro()
    first_body, first_joint, first_dof = builder.body_count, builder.joint_count, builder.joint_dof_count
    options = {
        "floating": mount == "floating",
        "enable_self_collisions": self_collisions,
        "collapse_fixed_joints": False,
        "force_position_velocity_actuation": True,
    }
    if mount == "wrist":
        parent, _ = _add_wrist(builder, palm_pose)
        builder.add_urdf(str(URDF), parent_body=parent, **options)
    else:
        builder.add_urdf(str(URDF), xform=palm_pose, **options)
    bodies = {_short(builder.body_label[b]): b for b in range(first_body, builder.body_count)}
    hand = [b for name, b in bodies.items() if not name.startswith("wrist_")]
    if len(hand) != HAND_BODIES or "palm_link" not in bodies:
        raise ValueError(f"Expected {HAND_BODIES} Allegro bodies with palm_link, got {sorted(bodies)}")
    dofs = []
    for j in range(first_joint, builder.joint_count):
        if builder.joint_type[j] == newton.JointType.FIXED:
            continue
        start = builder.joint_qd_start[j]
        if builder.joint_type[j] == newton.JointType.FREE:
            # Root pose PD gains come from the recipe; the mode is set here.
            for dof in range(start, start + 6):
                builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
            continue
        name = _short(builder.joint_label[j])
        finger = not name.startswith("wrist_")
        dofs.append(name)
        builder.joint_target_mode[start] = int(newton.JointTargetMode.POSITION_VELOCITY)
        if finger:
            builder.joint_target_ke[start] = FINGER_KE
            builder.joint_target_kd[start] = FINGER_KD
            builder.joint_armature[start] = FINGER_ARMATURE
    if sum(not name.startswith("wrist_") for name in dofs) != HAND_DOFS:
        raise ValueError(f"Expected {HAND_DOFS} Allegro finger DoFs, got {dofs}")
    root = next(j for j in range(first_joint, builder.joint_count) if builder.joint_parent[j] == -1)
    return {"dofs": dofs, "bodies": bodies, "root_joint": root, "first_dof": first_dof}


FINGER_ROLES = {
    "abduct": ("joint_0.0", "joint_4.0", "joint_8.0"),
    "flex": tuple(f"joint_{i}.0" for i in (1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15)),
    "thumb_oppose": ("joint_12.0",),
    "thumb_abduct": ("joint_13.0",),
}
FIST_THUMB_ABDUCT = 0.6


def finger_pose(model, layout, kind):
    """Finger coordinates by joint name: ``open`` (zero, clipped to limits) or ``fist`` (flexion at the upper limits)."""
    if kind not in ("open", "fist"):
        raise ValueError(f"finger pose must be 'open' or 'fist', got {kind!r}")
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    start = model.joint_qd_start.numpy()
    labels = [_short(model.joint_label[j]) for j in range(model.joint_count)]
    pose = {}
    for name in (n for role in FINGER_ROLES.values() for n in role):
        dof = int(start[labels.index(name)])
        lo, hi = float(lower[dof]), float(upper[dof])
        if kind == "open":
            value = 0.0
        elif name in FINGER_ROLES["abduct"]:
            value = 0.0
        elif name in FINGER_ROLES["thumb_abduct"]:
            value = FIST_THUMB_ABDUCT
        else:
            value = hi
        pose[name] = min(max(value, lo), hi)
    return pose


def set_finger_coordinates(model, joint_q, pose):
    """Write named finger coordinates into a joint_q vector (numpy, Model coordinate layout)."""
    q_start = model.joint_q_start.numpy()
    labels = [_short(model.joint_label[j]) for j in range(model.joint_count)]
    joint_q = np.array(joint_q, dtype=np.float64, copy=True)
    for name, value in pose.items():
        joint_q[int(q_start[labels.index(name)])] = value
    return joint_q
