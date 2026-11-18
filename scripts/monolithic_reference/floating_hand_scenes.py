# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allegro left-hand scenes: wrist or floating mount, FREE objects and a capsule chain.

The hand is the pinned ``asset/monolithic/data/inputs/allegro_left`` copy of
newton-assets ``wonik_allegro`` (provenance in its ``provenance.json``). Finger
and wrist gains follow the task profile ``manip/hands/allegro_left.json`` of
monolithic-tasks, whose sha256 the provenance records.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.retained_assets import ASSET_ROOT, INPUTS, retained_hashes, verify_retained

ALLEGRO = INPUTS / "allegro_left"
URDF = ALLEGRO / "wonik_allegro/urdf/allegro_hand_description_left.urdf"
HAND_DOFS = 16
HAND_BODIES = 17
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
    """Px Py Pz Rz Ry Rx from the world mount frame; return the last wrist link and the wrist joints.

    The first joint hangs on the world with the mount as its parent frame (no
    welded mount body): each FK level doubles the FP32 swept-FK roundoff
    allowance (contact_sweep), and a mount body would put the fingertips past it.
    """
    inertia = wp.diag(wp.vec3(WRIST_LINK_INERTIA, WRIST_LINK_INERTIA, WRIST_LINK_INERTIA))
    joints, parent = [], -1
    for name, kind, axis in WRIST_JOINTS:
        link = builder.add_link(mass=WRIST_LINK_MASS, inertia=inertia, label=f"wrist_{name}_link")
        frame = {"parent_xform": mount} if parent == -1 else {}
        if kind == "prismatic":
            joint = builder.add_joint_prismatic(
                parent,
                link,
                **frame,
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
                **frame,
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


def add_allegro(builder, *, mount, palm_pose, attach=None, self_collisions=True):
    """Add the hand with a 6-joint wrist (``mount='wrist'``) or a FREE root (``mount='floating'``).

    For the floating mount ``palm_pose`` is the palm frame in world. For the wrist
    it is the fixed mount frame and ``attach`` (default identity) the last wrist
    link to palm transform, as in the task grasp tool. Every moving DoF gets
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
        builder.add_urdf(str(URDF), parent_body=parent, xform=attach or wp.transform_identity(), **options)
    else:
        builder.add_urdf(str(URDF), xform=palm_pose, **options)
    # Merge the five fingertip frames into their distal links (FP32 swept-FK depth, see _add_wrist).
    tips = {
        j
        for j in range(first_joint, builder.joint_count)
        if _short(builder.body_label[builder.joint_child[j]]).endswith("_tip")
    }
    if len(tips) != 4 or any(builder.joint_type[j] != newton.JointType.FIXED for j in tips):
        raise ValueError(f"Expected four fixed Allegro fingertip joints, got {sorted(tips)}")
    builder.collapse_fixed_joints(joints_to_keep=[j for j in range(builder.joint_count) if j not in tips])
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
            # URDF <dynamics> (damping 3-10 N m s/rad, friction 5-12 N m against a 15 N m effort
            # limit) is not physical for these joints; the task profile also zeroes damping.
            builder.joint_damping[start] = 0.0
            builder.joint_friction[start] = 0.0
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


def hand_keyframes(model, recipe):
    """Names, pregrasp, open and closed values (wrist then fingers) of a grasp recipe.

    Closed = grasp + squeeze_rad along sign(grasp - open) on flex and oppose joints, clipped to the
    joint limits: the task screen's rule (monolithic-tasks manip/screen.py hand_schedule).
    """
    g = recipe["grasp"]
    names = [f"wrist_{n}" for n, _, _ in WRIST_JOINTS] + [n for role in FINGER_ROLES.values() for n in role]
    pre = np.array([g["pregrasp"][n] for n in names])
    open_ = np.array([g["open_at_grasp"][n] for n in names])
    grasp = np.array([g["grasp"][n] for n in names])
    squeeze = np.zeros(len(names))
    for i, n in enumerate(names):
        if n in FINGER_ROLES["flex"] or n in FINGER_ROLES["thumb_oppose"]:
            squeeze[i] = recipe["squeeze_rad"] * np.sign(grasp[i] - open_[i])
    labels = [_short(model.joint_label[j]) for j in range(model.joint_count)]
    start = model.joint_qd_start.numpy()
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    lo, hi = [], []
    for n in names:
        if n in labels:
            dof = int(start[labels.index(n)])
            lo.append(lower[dof])
            hi.append(upper[dof])
        else:  # Floating mount: wrist keyframes only define the palm path; keep them within the wrist limits.
            limit = (
                WRIST_LINEAR_LIMIT
                if n in ("wrist_px", "wrist_py", "wrist_pz")
                else (np.pi if n == "wrist_rz" else WRIST_ANGULAR_LIMIT)
            )
            for key, values in (("pregrasp", pre), ("open_at_grasp", open_), ("grasp", grasp)):
                value = values[names.index(n)]
                if abs(value) > limit:
                    raise ValueError(
                        f"grasp.{key}.{n} = {value} is outside the wrist limit {limit}; both mounts must see the same palm path"
                    )
            lo.append(-limit)
            hi.append(limit)
    return names, pre, open_, np.clip(grasp + squeeze, lo, hi)


def set_finger_coordinates(model, joint_q, pose):
    """Write named finger coordinates into a joint_q vector (numpy, Model coordinate layout)."""
    q_start = model.joint_q_start.numpy()
    labels = [_short(model.joint_label[j]) for j in range(model.joint_count)]
    joint_q = np.array(joint_q, dtype=np.float64, copy=True)
    for name, value in pose.items():
        joint_q[int(q_start[labels.index(name)])] = value
    return joint_q


# ---------------------------------------------------------------------------
# Recipes and scenes
# ---------------------------------------------------------------------------

RECIPE_SCHEMA = "p9d-floating-hand/v1"
SCENES = ("B1", "E1a", "E1b", "E2", "E3")
GRASP_PHASES = ("approach", "close", "lift", "hold")
FIST_PHASES = ("close", "hold")
CHAIN_PHASES = ("flight", "settle")


def load_recipe(path):
    """Read and check a floating-hand recipe; errors name the bad field."""
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise ValueError(f"{path}: schema must be {RECIPE_SCHEMA!r}, got {recipe.get('schema')!r}")
    scene = recipe.get("scene")
    if scene not in SCENES:
        raise ValueError(f"{path}: scene must be one of {SCENES}, got {scene!r}")
    dt = recipe.get("dt_s")
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{path}: dt_s must be a positive number, got {dt!r}")
    expected = {"E2": CHAIN_PHASES, "E3": FIST_PHASES}.get(scene, GRASP_PHASES)
    phases = recipe.get("phases_s", {})
    if tuple(phases) != expected:
        raise ValueError(f"{path}: phases_s must list {expected} in order, got {tuple(phases)}")
    for name, duration in phases.items():
        steps = duration / dt
        if not np.isfinite(steps) or steps <= 0 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"{path}: phases_s.{name} = {duration} s is not a positive whole number of dt_s steps")
    table = recipe.get("table")
    if table is not None and (table.get("kind") != "plane" or table.get("z") != 0.0):
        raise ValueError(
            f"{path}: table must be {{'kind': 'plane', 'z': 0.0}} (the criteria assume the plane z = 0), got {table}"
        )
    for key in ("solver", "joint_terms"):
        if not isinstance(recipe.get(key), dict):
            raise ValueError(f"{path}: {key} must be an object")
    if scene in ("B1", "E1a", "E1b", "E3") and recipe.get("mount") not in MOUNTS:
        raise ValueError(f"{path}: mount must be one of {MOUNTS}, got {recipe.get('mount')!r}")
    return recipe


def _matrix(transform):
    t = np.asarray(transform, dtype=np.float64)
    if t.shape == (4, 4):
        return t
    from scripts.monolithic_reference.floating_hand_criteria import quaternion_matrix  # noqa: PLC0415

    m = np.eye(4)
    m[:3, :3], m[:3, 3] = quaternion_matrix(t[3:]), t[:3]
    return m


def _transform(matrix):
    """4x4 matrix to wp.transform with a positive-w quaternion."""
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    q = q if q[3] >= 0 else -q
    return wp.transform(wp.vec3(*map(float, matrix[:3, 3])), wp.quat(*map(float, q)))


def _axis_rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    i = "xyz".index(axis)
    m = np.eye(4)
    a, b = [(1, 2), (2, 0), (0, 1)][i]
    m[a, a], m[a, b], m[b, a], m[b, b] = c, -s, s, c
    return m


def wrist_palm_matrix(mount, attach, q):
    """Palm pose of the wrist chain: mount, Px Py Pz, Rz, Ry, Rx, attach (matches Newton FK)."""
    m = _matrix(mount).copy()
    translation = np.eye(4)
    translation[:3, 3] = q[:3]
    m = m @ translation
    for axis, angle in zip("zyx", q[3:6], strict=True):
        m = m @ _axis_rotation(axis, angle)
    return m @ _matrix(attach)


def _quintic(s):
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s), 30.0 * s * s * (1.0 - s) ** 2


def _object_volume(o):
    if o["kind"] == "box":
        return float(np.prod(2 * np.asarray(o["half_extents"])))
    r, h = o["radius"], o["half_height"]
    return float(np.pi * r * r * 2 * h + 4.0 / 3.0 * np.pi * r**3)


CONVEX_HULL = INPUTS / "p9d_convex_capsule/hull_r20_hh40.npz"
# Texture SDF build for the pinned hull (CUDA only); its C2 field is checked against the exact hull
# distance in test_monolithic_contact_convex_mesh.py (P9d PR1).
CONVEX_HULL_SDF = {
    "max_resolution": 128,
    "margin": 0.02,
    "narrow_band_range": (-0.05, 0.05),
    "texture_format": "float32",
    "sign_method": "winding",
}


def convex_hull_asset(spec):
    """Pinned E1b hull (vertices, indices), checked against the manifest and the recipe's capsule size."""
    if not CONVEX_HULL.is_file():
        raise FileNotFoundError(f"Convex object asset missing: {CONVEX_HULL} (pinned by P9d PR0)")
    verify_retained(CONVEX_HULL, retained_hashes())
    if (spec["radius"], spec["half_height"]) != (0.02, 0.04):
        raise ValueError(
            f"The pinned hull is r 0.02 m, half_height 0.04 m; the object asks for {spec['radius']}, {spec['half_height']}"
        )
    data = np.load(CONVEX_HULL)
    return data["vertices"].astype(np.float64), data["indices"].astype(np.int32)


def add_object(builder, spec, device=None):
    """Add the FREE object (box, capsule or convex hull of a capsule) at its recipe pose."""
    kind = spec["kind"]
    if kind not in ("box", "capsule", "convex"):
        raise ValueError(f"object.kind must be box, capsule or convex, got {kind!r}")
    pose = wp.transform(wp.vec3(*spec["position"]), wp.quat(*spec["quaternion_xyzw"]))
    density = spec["mass_kg"] / _object_volume({**spec, "kind": "box" if kind == "box" else "capsule"})
    cfg = builder.ShapeConfig(density=density, margin=0.0, has_particle_collision=False)
    body = builder.add_link(xform=pose, label="object")
    builder.add_articulation([builder.add_joint_free(body, parent_xform=pose, label="object_free")], label="object")
    if kind == "box":
        builder.add_shape_box(
            body,
            hx=spec["half_extents"][0],
            hy=spec["half_extents"][1],
            hz=spec["half_extents"][2],
            cfg=cfg,
            label="object_box",
        )
    elif kind == "capsule":
        builder.add_shape_capsule(
            body, radius=spec["radius"], half_height=spec["half_height"], cfg=cfg, label="object_capsule"
        )
    else:
        points, indices = convex_hull_asset(spec)
        mesh = newton.Mesh(points, indices)
        if device is not None and wp.get_device(device).is_cuda:
            mesh.build_sdf(device=device, **CONVEX_HULL_SDF)
        builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg, label="object_convex")
    return body


def add_chain(builder, spec):
    """E2 floating chain: a root box and two capsule links on revolute springs (kd = 0)."""
    pose = wp.transform(wp.vec3(*spec["position"]), wp.quat(*spec["quaternion_xyzw"]))
    root_half = spec["root_half_extents"]
    cfg = builder.ShapeConfig(density=spec["density_kg_m3"], margin=0.0, has_particle_collision=False)
    root = builder.add_link(xform=pose, label="chain_root")
    builder.add_shape_box(root, hx=root_half[0], hy=root_half[1], hz=root_half[2], cfg=cfg, label="chain_root_box")
    joints = [builder.add_joint_free(root, parent_xform=pose, label="chain_free")]
    parent, offset = root, root_half[0]
    r, h = spec["link_radius"], spec["link_half_height"]
    for i in range(2):
        link = builder.add_link(label=f"chain_link_{i}")
        builder.add_shape_capsule(
            link,
            xform=wp.transform((r + h, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)),
            radius=r,
            half_height=h,
            cfg=cfg,
            label=f"chain_capsule_{i}",
        )
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform((offset + spec["joint_gap_m"], 0.0, 0.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                target_ke=spec["spring_ke"],
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                label=f"chain_joint_{i}",
            )
        )
        parent, offset = link, 2 * (r + h)
    builder.add_articulation(joints, label="chain")
    for j in joints[1:]:
        builder.joint_target_mode[builder.joint_qd_start[j]] = int(newton.JointTargetMode.POSITION_VELOCITY)
    return root


def articulation_dof_count(model):
    """DoFs the solver's joint terms cover: every articulation tree, floating roots included; FREE
    single bodies (rigid actors) excluded."""
    from newton._src.solvers.monolithic.actors import ActorTable  # noqa: PLC0415

    return sum(actor.dof_count for actor in ActorTable.from_model(model).actors if actor.kind == "articulation")


class FloatingHandScene:
    """One recipe scene with its solver, state and per-step control targets.

    ``mount`` overrides the recipe mount (``wrist`` or ``floating``);
    ``self_collision='off'`` drops the recipe's ``self_collision`` block (the
    E3 negative control and every trunk baseline).
    """

    def __init__(self, recipe, *, device, precision="fp32", use_cuda_graph=False, mount=None, self_collision="recipe"):
        from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

        if self_collision not in ("recipe", "off"):
            raise ValueError(f"self_collision must be 'recipe' or 'off', got {self_collision!r}")
        self.recipe, self.scene = recipe, recipe["scene"]
        self.dt = float(recipe["dt_s"])
        self.phase_steps = {name: round(d / self.dt) for name, d in recipe["phases_s"].items()}
        self.total_steps = sum(self.phase_steps.values())
        self.mount = mount or recipe.get("mount")
        builder = newton.ModelBuilder(gravity=tuple(recipe.get("gravity", (0.0, 0.0, -9.81))))
        builder.default_shape_cfg.margin = 0.0
        self.layout, self.object_body = None, None
        if self.scene == "E2":
            self.root_body = add_chain(builder, recipe["chain"])
        else:
            self._add_hand(builder, recipe)
            if "object" in recipe:
                self.object_body = add_object(builder, recipe["object"], device)
        if recipe.get("table", {}).get("kind") == "plane":
            cfg = builder.ShapeConfig(density=0.0, margin=0.0, has_particle_collision=False)
            builder.add_shape_plane(body=-1, width=0.0, length=0.0, cfg=cfg, label="table")
        self.model = builder.finalize(device=device)
        self._schedule()
        options = dict(recipe["solver"])
        terms = recipe["joint_terms"]
        dofs = articulation_dof_count(self.model)
        joint_terms = SolverMonolithic.JointTerms(
            implicit_pd=bool(terms.get("implicit_pd", True)),
            limits=bool(terms.get("limits", False)),
            limit_width=np.full(dofs, terms["limit_width"]) if terms.get("limits") else None,
        )
        if self_collision == "recipe" and recipe.get("self_collision"):
            options["self_collision"] = recipe["self_collision"]
        # Only shapes with a texture SDF (the E1b hull) need smooth fields; visual meshes have none.
        if self.model.device.is_cuda and (self.model._shape_sdf_index.numpy() >= 0).any():
            from newton.solvers.experimental.monolithic import build_smooth_sdf  # noqa: PLC0415

            options["smooth_sdf"] = build_smooth_sdf(self.model)
        self.solver = SolverMonolithic(
            self.model,
            joint_terms=joint_terms,
            pcg_mode="production",
            use_cuda_graph=use_cuda_graph,
            precision=precision,
            **options,
        )
        source = self.model.state()
        source.joint_q.assign(self.initial_q.astype(np.float32))
        source.joint_qd.assign(self.initial_qd.astype(np.float32))
        newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
        self.state = self.solver.create_state(dt=self.dt, source=source)
        self.control = self.model.control()
        self.steps = 0

    def _add_hand(self, builder, recipe):
        grasp = recipe.get("grasp")
        if self.scene == "E3":
            palm = _matrix(recipe["palm_pose"])
            mount_frame, attach = palm, np.eye(4)
        else:
            mount_frame, attach = _matrix(grasp["mount_world"]), _matrix(grasp["attach_in_wrist"])
        self.mount_frame, self.attach = mount_frame, attach
        if self.mount == "wrist":
            frame = _transform(mount_frame)
            self.layout = add_allegro(builder, mount="wrist", palm_pose=frame, attach=_transform(attach))
            # Same effort limits as the floating root, so both mounts compare like for like.
            root_pd, first = recipe["root_pd"], self.layout["first_dof"]
            for k in range(6):
                builder.joint_effort_limit[first + k] = root_pd["force_limit_N" if k < 3 else "torque_limit_Nm"]
        else:
            start = self._wrist_start(recipe)
            frame = _transform(wrist_palm_matrix(mount_frame, attach, start))
            self.layout = add_allegro(builder, mount="floating", palm_pose=frame)
            root_pd = recipe["root_pd"]
            first = self.layout["first_dof"]
            for k in range(6):
                linear = k < 3
                builder.joint_target_ke[first + k] = root_pd["linear_ke" if linear else "angular_ke"]
                builder.joint_target_kd[first + k] = root_pd["linear_kd" if linear else "angular_kd"]
                builder.joint_effort_limit[first + k] = root_pd["force_limit_N" if linear else "torque_limit_Nm"]
        effort = recipe.get("hand_import", {}).get("finger_effort_limit_Nm")
        if effort is not None:
            first = self.layout["first_dof"] + 6
            for dof in range(first, first + HAND_DOFS):
                builder.joint_effort_limit[dof] = effort
        self.palm_body = self.layout["bodies"]["palm_link"]

    def _wrist_start(self, recipe):
        if self.scene == "E3":
            return np.zeros(6)
        return np.array([recipe["grasp"]["pregrasp"][f"wrist_{n}"] for n, _, _ in WRIST_JOINTS])

    def _hand_keyframes(self):
        """Pregrasp, open and squeezed-closed coordinates by name (wrist and fingers), as the task screen does."""
        return hand_keyframes(self.model, self.recipe)

    def _limits(self, names):
        lower, upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        index = self._dof_index(names)
        return lower[index], upper[index]

    def _target_index(self, names):
        labels = [_short(self.model.joint_label[j]) for j in range(self.model.joint_count)]
        start = self.model.joint_target_q_start.numpy()
        return np.array([int(start[labels.index(n)]) for n in names])

    def _dof_index(self, names):
        labels = [_short(self.model.joint_label[j]) for j in range(self.model.joint_count)]
        start = self.model.joint_qd_start.numpy()
        return np.array([int(start[labels.index(n)]) for n in names])

    def _schedule(self):
        """Per-step named coordinate and rate targets, evaluated at the end of each step."""
        n, dt = self.total_steps, self.dt
        bounds, t0 = {}, 0
        for name, count in self.phase_steps.items():
            bounds[name] = (t0, t0 + count)
            t0 += count
        self.bounds = bounds
        if self.scene == "E2":
            self.names, self.targets_q, self.targets_qd = [], np.zeros((n, 0)), np.zeros((n, 0))
            chain = self.recipe["chain"]
            q = self.model.joint_q.numpy().astype(np.float64)
            qd = np.zeros(self.model.joint_dof_count)
            qd[:3], qd[3:6] = chain["linear_velocity"], chain["angular_velocity"]
            self.initial_q, self.initial_qd = q, qd
            return
        if self.scene == "E3":
            open_pose = finger_pose(self.model, self.layout, "open")
            fist = finger_pose(self.model, self.layout, "fist")
            self.names = [f"wrist_{n}" for n, _, _ in WRIST_JOINTS] + list(open_pose)
            pre = np.r_[np.zeros(6), [open_pose[k] for k in open_pose]]
            closed = np.r_[np.zeros(6), [fist[k] for k in open_pose]]
            keys = {"close": (pre, closed), "hold": (closed, closed)}
            lift = np.zeros(6)
        else:
            self.names, pre, open_, closed = self._hand_keyframes()
            keys = {
                "approach": (pre, open_),
                "close": (open_, closed),
                "lift": (closed, closed),
                "hold": (closed, closed),
            }
            lift = np.r_[_matrix(self.mount_frame)[:3, :3].T @ np.array([0.0, 0.0, self.recipe["lift_m"]]), np.zeros(3)]
        q, qd = np.zeros((n, len(self.names))), np.zeros((n, len(self.names)))
        for name, (a, b) in bounds.items():
            start_q, end_q = keys[name]
            for k in range(a, b):
                s = (k + 1 - a) / (b - a)
                blend, rate = _quintic(s)
                q[k] = start_q + blend * (end_q - start_q)
                qd[k] = rate * (end_q - start_q) / ((b - a) * dt)
                if name in ("lift", "hold"):
                    wrist_blend, wrist_rate = _quintic(s) if name == "lift" else (1.0, 0.0)
                    q[k, :6] += wrist_blend * lift
                    qd[k, :6] += wrist_rate * lift / ((b - a) * dt) if name == "lift" else 0.0
        self.targets_q, self.targets_qd = q, qd
        first = pre
        if self.mount == "wrist":
            init = self.model.joint_q.numpy().astype(np.float64)
            init[self._dof_index(self.names)] = first
            self.initial_q = init
        else:
            init = self.model.joint_q.numpy().astype(np.float64)
            fingers = self.names[6:]
            q_start = self.model.joint_q_start.numpy()
            labels = [_short(self.model.joint_label[j]) for j in range(self.model.joint_count)]
            for name, value in zip(fingers, first[6:], strict=True):
                init[int(q_start[labels.index(name)])] = value
            self.initial_q = init
        self.initial_qd = np.zeros(self.model.joint_dof_count)

    def palm_target(self, k):
        """Palm target pose (4x4) at the end of step ``k``; the floating root follows it."""
        if self.scene == "E2":
            return None
        return wrist_palm_matrix(self.mount_frame, self.attach, self.targets_q[k, :6])

    def root_target(self, k):
        """Floating root target at the end of step ``k`` in Newton's FREE convention (coordinate layout).

        Returns the joint-frame position and xyzw quaternion relative to the parent anchor, and the root
        COM velocity and angular velocity in the anchor frame, from the palm path's step-to-step change.
        """
        root = self.layout["root_joint"]
        parent, child = (_matrix(getattr(self.model, n).numpy()[root]) for n in ("joint_X_p", "joint_X_c"))
        body = int(self.model.joint_child.numpy()[root])
        com = self.model.body_com.numpy()[body].astype(np.float64)
        previous = (
            self.palm_target(k - 1)
            if k > 0
            else wrist_palm_matrix(self.mount_frame, self.attach, self._wrist_start(self.recipe))
        )
        current = self.palm_target(k)
        coordinates = np.linalg.inv(parent) @ current @ child
        velocity = (current[:3, :3] @ com + current[:3, 3] - previous[:3, :3] @ com - previous[:3, 3]) / self.dt
        relative = current[:3, :3] @ previous[:3, :3].T
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
        axial = np.array(
            [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
        )
        omega = axial * (0.5 if angle < 1e-8 else angle / (2 * np.sin(angle))) / self.dt
        anchor = parent[:3, :3]
        from scipy.spatial.transform import Rotation

        quaternion = Rotation.from_matrix(coordinates[:3, :3]).as_quat()
        pose = np.r_[coordinates[:3, 3], quaternion if quaternion[3] >= 0 else -quaternion]
        return pose, np.r_[anchor.T @ velocity, anchor.T @ omega]

    @property
    def phase(self):
        for name, (a, b) in self.bounds.items():
            if a <= self.steps < b:
                return name
        return "done"

    def _write_control(self, k):
        if self.scene == "E2":
            return
        target_q = self.control.joint_target_q.numpy()
        target_qd = self.control.joint_target_qd.numpy()
        # Position targets use the coordinate layout (joint_target_q_start), rates the DoF layout; they
        # differ after a floating root's 7 coordinates.
        names = self.names if self.mount == "wrist" else self.names[6:]
        columns = slice(None) if self.mount == "wrist" else slice(6, None)
        target_q[self._target_index(names)] = self.targets_q[k, columns]
        target_qd[self._dof_index(names)] = self.targets_qd[k, columns]
        if self.mount != "wrist":
            pose, rates = self.root_target(k)
            root = self.layout["root_joint"]
            target_q[int(self.model.joint_target_q_start.numpy()[root]) + np.arange(7)] = pose
            target_qd[int(self.model.joint_qd_start.numpy()[root]) + np.arange(6)] = rates
        self.control.joint_target_q.assign(target_q)
        self.control.joint_target_qd.assign(target_qd)

    def step(self):
        """Advance one step; the clock moves only when the solver publishes (rollbacks leave it)."""
        if self.steps >= self.total_steps:
            raise RuntimeError(f"{self.scene}: all {self.total_steps} planned steps are done")
        self._write_control(self.steps)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        if not self.solver.last_stats.rolled_back:
            self.steps += 1
