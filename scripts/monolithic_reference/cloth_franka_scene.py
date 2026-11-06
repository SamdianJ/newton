# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared SI Franka folding scene with one coupled Monolithic step per request."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton
import newton.ik as ik

from .cloth_contact_run import create_solver, publication
from .material_recipes import shell_coefficients

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = Path(__file__).with_name("fixtures") / "franka_cloth_fold_v1.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(source):
    """Read and validate physical inputs before accessing an asset or device."""
    r = copy.deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    if r.get("schema_version") != 1 or r.get("units") != "SI" or not r.get("keyframes"):
        raise ValueError("Expected a version 1 SI folding recipe with keyframes")
    material, robot = r["material"], r["robot"]
    positive = [
        r["dt"],
        r["time_scale"],
        material["young_modulus_Pa"],
        material["thickness_m"],
        material["surface_density_kg_m2"],
    ]
    positive += [frame["duration_s"] for frame in r["keyframes"]]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Time intervals and material properties must be finite and positive")
    if not 0 <= material["poisson_ratio"] < 0.5:
        raise ValueError("Poisson ratio must be in [0, 0.5)")
    nonnegative = [r["initial_hold_s"], r["final_hold_s"], r["asset"]["table_clearance_m"], robot["finger_closed_m"]]
    if not np.isfinite(nonnegative).all() or min(nonnegative) < 0:
        raise ValueError("Holds, clearance and finger targets must be finite and nonnegative")
    if not np.isfinite(robot["finger_open_m"]) or robot["finger_open_m"] <= robot["finger_closed_m"]:
        raise ValueError("Open finger target must exceed the closed target")
    for frame in r["keyframes"]:
        p, q = np.asarray(frame["position_m"]), np.asarray(frame["quaternion_xyzw"])
        if (
            p.shape != (3,)
            or q.shape != (4,)
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or np.linalg.norm(q) == 0
        ):
            raise ValueError("Each keyframe needs a finite position and nonzero quaternion")
        if frame["gripper"] not in ("open", "closed") or not frame["phase"]:
            raise ValueError("Each keyframe needs a phase and open/closed gripper target")
    duration = r["initial_hold_s"] + sum(f["duration_s"] for f in r["keyframes"]) * r["time_scale"] + r["final_hold_s"]
    if not np.isclose(duration / r["dt"], round(duration / r["dt"]), rtol=0, atol=1e-7):
        raise ValueError("Recipe duration must be an integer number of physical steps")
    return r


def tcp_targets(recipe):
    """Map the old link-frame TCP to the selected physical hand frame, in meters."""
    robot = recipe["robot"]
    offset = np.asarray(robot["tcp_offset_m"]) - robot["source_tcp_offset_m"]
    local = np.asarray(robot["tcp_rotation_xyzw"], dtype=float)
    local /= np.linalg.norm(local)
    positions, rotations = [], []
    for frame in recipe["keyframes"]:
        q = np.asarray(frame["quaternion_xyzw"], dtype=float)
        q /= np.linalg.norm(q)
        positions.append(
            np.asarray(frame["position_m"]) + offset + 2 * np.cross(q[:3], np.cross(q[:3], offset) + q[3] * offset)
        )
        rotations.append(
            np.r_[q[3] * local[:3] + local[3] * q[:3] + np.cross(q[:3], local[:3]), q[3] * local[3] - q[:3] @ local[:3]]
        )
    return np.asarray(positions), np.asarray(rotations)


def fold_targets(times, poses, time_s):
    """Interpolate joint targets with zero endpoint velocity on the physical clock."""
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Trajectory time must be finite and nonnegative")
    index = min(int(np.searchsorted(times[1:], time_s, side="right")), len(times) - 2)
    duration = times[index + 1] - times[index]
    u = np.clip((time_s - times[index]) / duration, 0, 1)
    blend = u**3 * (10 + u * (-15 + 6 * u))
    rate = 30 * u**2 * (1 - u) ** 2 / duration
    delta = poses[index + 1] - poses[index]
    return poses[index] + blend * delta, rate * delta, index


def build_model(recipe, *, device, cache, prepare_sdf=True):
    """Build dynamic robot and free cloth; no cloth vertex is attached to the hand."""
    from pxr import Usd, UsdGeom

    r = load_recipe(recipe)
    robot, material, table = r["robot"], r["material"], r["table"]
    urdf, asset = REPO / robot["urdf"], REPO / r["asset"]["path"]
    if not urdf.is_file() or not asset.is_file():
        raise FileNotFoundError(f"Missing folding input: {urdf if not urdf.is_file() else asset}")
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    builder.add_urdf(
        str(urdf),
        xform=wp.transform(robot["base_position_m"], wp.quat_identity()),
        floating=False,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )
    if builder.joint_dof_count != 9 or builder.body_count != 9:
        raise ValueError("Expected seven dynamic arm joints and two dynamic fingers")
    # Match the existing Monolithic Franka: both fingers stay dynamic with equal targets.
    mimic = {}
    for name in ("joint0", "joint1", "coef0", "coef1", "enabled", "label", "world"):
        values = getattr(builder, "constraint_mimic_" + name)
        mimic[name] = list(values)
        values.clear()
    for j in range(9):
        builder.joint_target_ke[j] = robot["arm_kp" if j < 7 else "finger_kp"]
        builder.joint_target_kd[j] = robot["arm_kd" if j < 7 else "finger_kd"]
        builder.joint_target_mode[j] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_limit_ke[j], builder.joint_limit_kd[j] = 1e4, 10.0
    for i, kind in enumerate(builder.shape_type):
        builder.shape_margin[i] = 0.0
        if (
            prepare_sdf
            and kind == newton.GeoType.MESH
            and builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
        ):
            sdf = r["sdf"]
            builder.shape_source[i].build_sdf(
                device=device,
                max_resolution=sdf["max_resolution"],
                margin=sdf["margin_m"],
                narrow_band_range=tuple(sdf["narrow_band_m"]),
                texture_format="float32",
                sign_method="winding",
                cache_dir=Path(cache) / "textures",
            )
    builder.add_shape_box(
        -1,
        xform=wp.transform(table["center_m"], wp.quat_identity()),
        hx=table["half_extents_m"][0],
        hy=table["half_extents_m"][1],
        hz=table["half_extents_m"][2],
    )
    stage = Usd.Stage.Open(str(asset))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(r["asset"]["prim"]))
    if not mesh or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError("Derived shirt must contain the selected mesh with metersPerUnit=1")
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not len(counts) or np.any(counts != 3):
        raise ValueError("Derived shirt must be a nonempty triangle mesh")
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    x = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    angle = r["asset"]["rotation_z_rad"]
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    x = x @ rotation.T
    translation = np.r_[
        r["asset"]["translation_xy_m"],
        table["center_m"][2] + table["half_extents_m"][2] + r["asset"]["table_clearance_m"] - x[:, 2].min(),
    ]
    x = (x + translation).astype(np.float32).astype(float)
    faces = x[triangles]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
    if not np.isfinite(x).all() or np.any(area <= 0):
        raise ValueError("Shirt has nonfinite positions or degenerate triangles")
    mass = np.zeros(len(x))
    np.add.at(mass, triangles.ravel(), np.repeat(area * material["surface_density_kg_m2"] / 3, 3))
    lam, mu, alpha, beta = shell_coefficients(material)
    for point, value in zip(x, mass, strict=True):
        builder.add_particle(wp.vec3(point), wp.vec3(0), value, radius=material["thickness_m"] / 2)
    for face in triangles:
        builder.add_triangle(*face, tri_ke=mu, tri_ka=lam, tri_kd=0, tri_drag=0, tri_lift=0)
    model = builder.finalize(device=device)
    flags = model.shape_flags.numpy()
    flags &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    cloth = [
        SolverMonolithic.ClothInput(
            triangle_ids=tuple(range(len(triangles))),
            bending=(alpha, beta),
            thickness=material["thickness_m"],
            surface_density=material["surface_density_kg_m2"],
        )
    ]
    metadata = {
        "asset": str(asset),
        "asset_sha256": _sha(asset),
        "urdf": str(urdf),
        "urdf_sha256": _sha(urdf),
        "robot_asset_hashes": {
            str(p.relative_to(urdf.parent.parent)): _sha(p) for p in urdf.parent.parent.rglob("*") if p.is_file()
        },
        "cloth_translation_m": translation.tolist(),
        "cloth_rotation_z_rad": angle,
        "rest_area_m2": float(area.sum()),
        "cloth_mass_kg": float(mass.sum()),
        "material": material,
        "removed_mimic": mimic,
        "collision_scope": "cloth-shape and method cloth-cloth; robot-table rigid collision is not enabled",
        "cloth_attachment": "none",
        "robot_gravity": "enabled",
        "sdf_prepared": prepare_sdf,
    }
    return model, cloth, metadata


def plan_joint_targets(model, recipe):
    """Solve TCP targets once, recording reach errors before dynamic execution."""
    robot = recipe["robot"]
    positions, rotations = tcp_targets(recipe)
    positions = np.vstack((robot["initial_tcp_position_m"], positions))
    rotations = np.vstack((rotations[0], rotations))
    body = model.body_label.index(robot["tcp_link"])
    position = ik.IKObjectivePosition(
        link_index=body,
        link_offset=wp.vec3(robot["tcp_offset_m"]),
        target_positions=wp.array([wp.vec3(positions[0])], dtype=wp.vec3, device=model.device),
    )
    orientation = ik.IKObjectiveRotation(
        link_index=body,
        link_offset_rotation=wp.quat(robot["tcp_rotation_xyzw"]),
        target_rotations=wp.array([wp.vec4(rotations[0])], dtype=wp.vec4, device=model.device),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[position, orientation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.array(np.asarray([robot["initial_q"]], dtype=np.float32), dtype=float, device=model.device)
    poses, errors = [], []
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    state = model.state()
    for i, (point, rotation) in enumerate(zip(positions, rotations, strict=True)):
        position.set_target_position(0, wp.vec3(point))
        orientation.set_target_rotation(0, wp.quat(rotation))
        for restart in range(2):
            if restart:
                q.assign(np.asarray([robot["initial_q"]], dtype=np.float32))
            solver.step(q, q, iterations=robot["ik_iterations"])
            result = q.numpy()[0].astype(float)
            gripper = "open" if i == 0 else recipe["keyframes"][i - 1]["gripper"]
            result[7:] = robot[f"finger_{gripper}_m"]
            state.joint_q.assign(result)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            transform = wp.transform(*state.body_q.numpy()[body])
            actual_position = np.asarray(wp.transform_point(transform, wp.vec3(robot["tcp_offset_m"])))
            actual_rotation = np.asarray(wp.transform_get_rotation(transform) * wp.quat(robot["tcp_rotation_xyzw"]))
            position_error = float(np.linalg.norm(actual_position - point))
            rotation_error = float(
                2 * np.arccos(np.clip(abs(actual_rotation @ rotation) / np.linalg.norm(actual_rotation), 0, 1))
            )
            valid = (
                np.isfinite(result).all()
                and np.all(result >= lower)
                and np.all(result <= upper)
                and position_error <= robot["ik_position_tolerance_m"]
                and rotation_error <= robot["ik_rotation_tolerance_rad"]
            )
            if valid:
                break
        errors.append(
            {"keyframe": i - 1, "position_m": position_error, "rotation_rad": rotation_error, "seed_restarts": restart}
        )
        if not valid:
            raise ValueError(f"Unreachable folding keyframe {i - 1}: {errors[-1]}")
        poses.append(result)
    times, targets, phases = [0.0], [poses[0]], []
    if recipe["initial_hold_s"]:
        times.append(recipe["initial_hold_s"])
        targets.append(poses[0])
        phases.append("settle")
    for i, frame in enumerate(recipe["keyframes"]):
        times.append(times[-1] + frame["duration_s"] * recipe["time_scale"])
        targets.append(poses[i + 1])
        phases.append(frame["phase"])
    if recipe["final_hold_s"]:
        times.append(times[-1] + recipe["final_hold_s"])
        targets.append(poses[-1])
        phases.append("released_hold")
    return (
        np.asarray(times),
        np.asarray(targets),
        phases,
        {
            "tcp_positions_m": positions.tolist(),
            "tcp_rotations_xyzw": rotations.tolist(),
            "ik_errors": errors,
            "joint_targets": np.asarray(targets).tolist(),
            "interpolation": "quintic joint targets; IK is not written to dynamic state",
            "max_joint_segment_delta": np.max(np.abs(np.diff(targets, axis=0)), axis=0).tolist(),
            "max_joint_target_speed": np.max(
                1.875 * np.abs(np.diff(targets, axis=0)) / np.diff(times)[:, None], axis=0
            ).tolist(),
            "path_clearance": "not certified; endpoint IK does not check the interpolated robot-table path",
        },
    )


class FrankaClothScene:
    """Shared folding task; contact methods only change the solver adapter."""

    def __init__(
        self,
        recipe=DEFAULT_RECIPE,
        *,
        device="cpu",
        precision="fp64",
        use_cuda_graph=False,
        method="off",
        method_config=None,
        cache=None,
    ):
        self.recipe = r = load_recipe(recipe)
        if method not in ("off", "point-cloud", "vfee"):
            raise ValueError("Unknown cloth contact method")
        cache = Path(cache) if cache is not None else REPO.parent.parent / "dev/monolithic/test/cache/franka-cloth"
        cache.mkdir(parents=True, exist_ok=True)
        self.model, cloth_inputs, self.metadata = build_model(r, device=device, cache=cache)
        self.times, self.joint_targets, self.phases, plan = plan_joint_targets(self.model, r)
        smooth = build_smooth_sdf(self.model)
        widths = (0.01,) * 7 + (0.001,) * 2
        options = r["solver"] | {
            "precision": precision,
            "use_cuda_graph": use_cuda_graph,
            "smooth_sdf": smooth,
            "joint_terms": SolverMonolithic.JointTerms(
                implicit_pd=True, limits=True, friction=True, limit_width=widths, friction_velocity_scale=widths
            ),
        }
        self.solver = create_solver(self.model, cloth_inputs, options, method, method_config)
        self.dt = r["dt"]
        self.total_steps = round(self.times[-1] / self.dt)
        source = self.model.state()
        source.joint_q.assign(self.joint_targets[0])
        source.joint_qd.zero_()
        newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
        self.state = self.solver.create_state(dt=self.dt, source=source)
        self.control = self.solver.create_control()
        self.control.joint_target_q.assign(self.joint_targets[0])
        self.control.joint_target_qd.zero_()
        self.triangles = self.model.tri_indices.numpy()
        self.owners = np.zeros(len(self.triangles), dtype=np.int32)
        self.thicknesses = np.full(len(self.triangles), r["material"]["thickness_m"])
        self.steps, self.phase, self.failed = 0, "initial", False
        self.last_forces, self._published_generation = None, 0
        self.metadata.update(
            recipe=r,
            method=method,
            method_config=method_config or {},
            plan=plan,
            dt=self.dt,
            total_steps=self.total_steps,
            precision=precision,
            use_cuda_graph=use_cuda_graph,
            sdf_identity=smooth.identity,
            sdf_checks=smooth.build_metadata["checks"],
            sdf_geometric_certification="not certified for folding task",
            recipe_sha256=hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
            trajectory_source_sha256=_sha(REPO / r["source_trajectory"]["path"]),
        )

    def step(self):
        """Advance exactly one physical step; retain clock and commands on failure."""
        if self.failed or self.steps >= self.total_steps:
            return None
        q, v, index = fold_targets(self.times, self.joint_targets, (self.steps + 1) * self.dt)
        old_q, old_v = self.control.joint_target_q.numpy(), self.control.joint_target_qd.numpy()
        try:
            self.control.joint_target_q.assign(q)
            self.control.joint_target_qd.assign(v)
            self.solver.step(self.state, self.state, self.control, dt=self.dt)
            stats = self.solver.last_stats
            forces = self.solver.get_contact_forces(self.state)
            if publication(stats, forces, previous_generation=self._published_generation) == "HARD_FAILURE":
                raise RuntimeError(f"Invalid folding publication at request {self.steps + 1}")
        except Exception:
            self.control.joint_target_q.assign(old_q)
            self.control.joint_target_qd.assign(old_v)
            self.failed = True
            raise
        self.steps += 1
        self.phase = self.phases[index]
        self.last_forces, self._published_generation = forces, stats.step_generation
        return stats
