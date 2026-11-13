# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared SI Franka folding scene with one coupled Monolithic step per request."""

import copy
import hashlib
import json
from itertools import product
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
    bending_scale = material.get("bending_scale", 1.0)
    if (
        isinstance(bending_scale, bool)
        or not isinstance(bending_scale, (int, float))
        or not np.isfinite(bending_scale)
        or bending_scale <= 0
    ):
        raise ValueError("Bending scale must be a finite positive number")
    nonnegative = [r["initial_hold_s"], r["final_hold_s"], r["asset"]["table_clearance_m"], robot["finger_closed_m"]]
    if not np.isfinite(nonnegative).all() or min(nonnegative) < 0:
        raise ValueError("Holds, clearance and finger targets must be finite and nonnegative")
    drag = r.get("initial_velocity_damping_per_s", 0.0)
    if isinstance(drag, bool) or not isinstance(drag, (int, float)) or not np.isfinite(drag) or drag < 0:
        raise ValueError("Initial velocity damping must be a finite nonnegative rate [1/s]")
    if drag > 0 and r["initial_hold_s"] == 0:
        raise ValueError("Initial velocity damping requires a positive initial hold")
    compensate = robot.get("gravity_compensation", False)
    if not isinstance(compensate, bool):
        raise ValueError("Gravity compensation must be a boolean")
    if compensate:
        with np.errstate(over="ignore", under="ignore"):
            gains = np.asarray([robot["arm_kp"], robot["finger_kp"]], dtype=np.float32)
        if not np.isfinite(gains).all() or min(gains) <= 0:
            raise ValueError("Gravity compensation requires finite positive FP32 PD stiffnesses")
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


def fold_targets(times, poses, time_s, *, phase_times=None):
    """Interpolate joint targets with zero endpoint velocity on the physical clock."""
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Trajectory time must be finite and nonnegative")
    boundaries = times if phase_times is None else phase_times
    phase = min(int(np.searchsorted(boundaries[1:], time_s, side="right")), len(boundaries) - 2)
    duration = boundaries[phase + 1] - boundaries[phase]
    u = np.clip((time_s - boundaries[phase]) / duration, 0, 1)
    blend = u**3 * (10 + u * (-15 + 6 * u))
    rate = 30 * u**2 * (1 - u) ** 2 / duration
    if phase_times is not None:
        # One time law per original phase; IK nodes do not introduce extra stops.
        path_time = boundaries[phase] + duration * blend
        lo, hi = np.searchsorted(times, boundaries[phase : phase + 2])
        index = min(max(int(np.searchsorted(times, path_time, side="right")) - 1, lo), hi - 1)
        width = times[index + 1] - times[index]
        v = np.clip((path_time - times[index]) / width, 0, 1)

        def tangent(node):
            if node in (lo, hi):
                a, b = (lo, lo + 1) if node == lo else (hi - 1, hi)
                return (poses[b] - poses[a]) / (times[b] - times[a])
            left = (poses[node] - poses[node - 1]) / (times[node] - times[node - 1])
            right = (poses[node + 1] - poses[node]) / (times[node + 1] - times[node])
            # Equal path intervals: harmonic slopes preserve each joint's range.
            return np.divide(2 * left * right, left + right, out=np.zeros_like(left), where=left * right > 0)

        a, b = poses[index], poses[index + 1]
        da, db = tangent(index) * width, tangent(index + 1) * width
        value = (
            (2 * v**3 - 3 * v**2 + 1) * a + (v**3 - 2 * v**2 + v) * da + (-2 * v**3 + 3 * v**2) * b + (v**3 - v**2) * db
        )
        derivative = (
            (6 * v**2 - 6 * v) * a + (3 * v**2 - 4 * v + 1) * da + (-6 * v**2 + 6 * v) * b + (3 * v**2 - 2 * v) * db
        )
        return value, derivative * duration * rate / width, index
    index = phase
    delta = poses[index + 1] - poses[index]
    return poses[index] + blend * delta, rate * delta, index


def _finger_shapes(model):
    bodies, flags, kinds = model.shape_body.numpy(), model.shape_flags.numpy(), model.shape_type.numpy()
    ids = [
        i
        for i, body in enumerate(bodies)
        if body >= 0 and "finger" in model.body_label[body] and flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)
    ]
    if len(ids) != 8 or any(kinds[i] != newton.GeoType.BOX for i in ids):
        raise ValueError("Folding clearance requires the eight actual Franka finger boxes")
    return ids


def _finger_corners(model, state):
    poses, bodies = state.body_q.numpy(), model.shape_body.numpy()
    local, sizes = model.shape_transform.numpy(), model.shape_scale.numpy()
    return np.asarray(
        [
            wp.transform_point(
                wp.transform(*poses[bodies[i]]) * wp.transform(*local[i]), wp.vec3(np.asarray(sign) * sizes[i])
            )
            for i in _finger_shapes(model)
            for sign in product((-1, 1), repeat=3)
        ],
        dtype=float,
    )


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
    bending_scale = material.get("bending_scale", 1.0)
    alpha, beta = alpha * bending_scale, beta * bending_scale
    for point, value in zip(x, mass, strict=True):
        builder.add_particle(wp.vec3(point), wp.vec3(0), value, radius=material["thickness_m"] / 2)
    for face in triangles:
        builder.add_triangle(*face, tri_ke=mu, tri_ka=lam, tri_kd=0, tri_drag=0, tri_lift=0)
    model = builder.finalize(device=device)
    flags = model.shape_flags.numpy()
    flags &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    table_id = model.shape_count - 1
    flags[[*_finger_shapes(model), table_id]] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
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
        "shell_coefficients": {
            "lambda_N_per_m": lam,
            "mu_N_per_m": mu,
            "alpha_N_m": alpha,
            "beta_N_m": beta,
        },
        "removed_mimic": mimic,
        "collision_scope": "cloth-shape, method cloth-cloth and eight finger-box/table A-static pairs; no arm-table collision",
        "finger_table_pairs": [[i, table_id] for i in _finger_shapes(model)],
        "finger_table_actor_samples": 288,
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
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    state = model.state()
    table_top = sum((recipe["table"]["center_m"][2], recipe["table"]["half_extents_m"][2]))
    clearance = recipe["material"]["thickness_m"] / 2
    tcp_local = wp.transform(robot["tcp_offset_m"], robot["tcp_rotation_xyzw"])
    times, phase_times, targets, phases, errors, lifts = [0.0], [0.0], [], [], [], []

    def solve(point, rotation, opening, frame, fraction):
        # Geometry is expressed in the requested TCP frame before solving the arm.
        seed = q.numpy()[0]
        seed[7:] = opening
        state.joint_q.assign(seed)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        actual_tcp = wp.transform(*state.body_q.numpy()[body]) * tcp_local
        mapping = wp.transform(point, rotation) * wp.transform_inverse(actual_tcp)
        corners = np.asarray([wp.transform_point(mapping, wp.vec3(p)) for p in _finger_corners(model, state)])
        lift = max(0.0, table_top + clearance - float(corners[:, 2].min()))
        corrected = np.asarray(point, dtype=float).copy()
        corrected[2] += lift
        position.set_target_position(0, wp.vec3(corrected))
        orientation.set_target_rotation(0, wp.quat(rotation))
        q.assign(seed[None])
        solver.step(q, q, iterations=robot["ik_iterations"])
        result = q.numpy()[0].astype(float)
        result[7:] = opening
        q.assign(result[None].astype(np.float32))
        state.joint_q.assign(result)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        transform = wp.transform(*state.body_q.numpy()[body])
        actual_position = np.asarray(wp.transform_point(transform, wp.vec3(robot["tcp_offset_m"])))
        actual_rotation = np.asarray(wp.transform_get_rotation(transform) * wp.quat(robot["tcp_rotation_xyzw"]))
        position_error = float(np.linalg.norm(actual_position - corrected))
        rotation_error = float(
            2 * np.arccos(np.clip(abs(actual_rotation @ rotation) / np.linalg.norm(actual_rotation), 0, 1))
        )
        error = {"keyframe": frame, "fraction": fraction, "position_m": position_error, "rotation_rad": rotation_error}
        if not (
            np.isfinite(result).all()
            and np.all(result >= lower)
            and np.all(result <= upper)
            and position_error <= robot["ik_position_tolerance_m"]
            and rotation_error <= robot["ik_rotation_tolerance_rad"]
        ):
            raise ValueError(f"Unreachable continuous-seed folding node: {error}; target={corrected.tolist()}")
        node_gap = float(_finger_corners(model, state)[:, 2].min() - table_top)
        if node_gap < 0:
            raise ValueError(f"Finger/table overlap at IK node {frame}, fraction {fraction}: {node_gap} m")
        error["finger_table_gap_m"] = node_gap
        errors.append(error)
        lifts.append(
            {
                "keyframe": frame,
                "fraction": fraction,
                "vertical_correction_m": lift,
                "tcp_position_m": corrected.tolist(),
            }
        )
        return result

    targets.append(solve(positions[0], rotations[0], robot["finger_open_m"], -1, 0.0))
    if recipe["initial_hold_s"]:
        times.append(recipe["initial_hold_s"])
        phase_times.append(times[-1])
        targets.append(targets[-1].copy())
        phases.append("settle")
    opening = robot["finger_open_m"]
    for i, frame in enumerate(recipe["keyframes"]):
        end_opening = robot[f"finger_{frame['gripper']}_m"]
        moving = not (
            np.allclose(positions[i], positions[i + 1]) and np.isclose(abs(rotations[i] @ rotations[i + 1]), 1)
        )
        nodes = 8 if moving else 1
        start = times[-1]
        duration = frame["duration_s"] * recipe["time_scale"]
        for node in range(1, nodes + 1):
            fraction = node / nodes
            point = (1 - fraction) * positions[i] + fraction * positions[i + 1]
            rotation = np.asarray(
                wp.quat_slerp(wp.quat(rotations[i]), wp.quat(rotations[i + 1]), fraction), dtype=float
            )
            targets.append(solve(point, rotation, opening + fraction * (end_opening - opening), i, fraction))
            times.append(start + fraction * duration)
            phases.append(frame["phase"])
        opening = end_opening
        phase_times.append(times[-1])
    if recipe["final_hold_s"]:
        times.append(times[-1] + recipe["final_hold_s"])
        phase_times.append(times[-1])
        targets.append(targets[-1].copy())
        phases.append("released_hold")
    times, targets, phase_times = np.asarray(times), np.asarray(targets), np.asarray(phase_times)
    minimum = min(row["finger_table_gap_m"] for row in errors)
    max_speed = np.zeros(9)
    for i in range(len(times) - 1):
        for fraction in (0.0, 0.5, 1.0):
            value, velocity, _ = fold_targets(
                times, targets, times[i] + fraction * (times[i + 1] - times[i]), phase_times=phase_times
            )
            max_speed = np.maximum(max_speed, np.abs(velocity))
            if np.any(value < lower) or np.any(value > upper):
                raise ValueError(f"Interpolated folding target violates joint limits in {phases[i]} at node {i}")
            state.joint_q.assign(value)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            gap = float(_finger_corners(model, state)[:, 2].min() - table_top)
            minimum = min(minimum, gap)
            if gap < 0:
                raise ValueError(f"Finger/table overlap in planned {phases[i]} node {i}, fraction {fraction}: {gap} m")
    return (
        times,
        targets,
        phases,
        {
            "tcp_positions_m": positions.tolist(),
            "tcp_rotations_xyzw": rotations.tolist(),
            "ik_errors": errors,
            "joint_targets": targets.tolist(),
            "times_s": times.tolist(),
            "phase_times_s": phase_times.tolist(),
            "phases": phases,
            "interpolation": "eight Cartesian nodes per moving phase, continuous-seed IK; shape-preserving Hermite q with one quintic time law per original phase; IK never writes dynamic state",
            "tcp_height_adjustments": lifts,
            "finger_table_clearance_m": clearance,
            "clearance_reason": "half material thickness leaves fingertip at the thin cloth layer, using actual eight BOX corners",
            "max_joint_segment_delta": np.max(np.abs(np.diff(targets, axis=0)), axis=0).tolist(),
            "max_sampled_joint_target_speed": max_speed.tolist(),
            "path_clearance": {
                "scope": "eight actual finger boxes above tabletop; IK nodes and uniform physical-time samples, not continuous certification",
                "minimum_sampled_gap_m": minimum,
                "samples": len(errors) + 3 * (len(times) - 1),
            },
        },
    )


class FrankaClothScene:
    """Shared folding task; contact methods only change the solver adapter.

    Optional ``material.bending_scale`` multiplies both shell bending
    coefficients independently of membrane stiffness, mass and thickness.
    It defaults to one; effective coefficients are recorded in scene metadata.

    The optional recipe rate ``initial_velocity_damping_per_s`` applies the
    lagged external force ``-rate * mass * velocity`` during the initial hold.
    Its interval uses step-start time; the first subsequent step clears it.
    This is a controlled preparation force, not implicit material damping.
    Applied forces are recorded for offline displacement-work evaluation.

    Optional ``robot.gravity_compensation`` uses a separate FP32 query state
    to evaluate lagged ``g(q_n)`` and offset the position drive by ``g/Kp``.
    Nominal velocity and the existing total drive effort limit are retained.
    Nominal targets and requested gravity forces [N or N m] are recorded
    separately from the rounded FP32 drive target and actual limited torque.
    """

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
        if method not in ("off", "point-cloud"):
            raise ValueError("Unknown cloth contact method")
        cache = Path(cache) if cache is not None else REPO.parent.parent / "dev/monolithic/test/cache/franka-cloth"
        cache.mkdir(parents=True, exist_ok=True)
        self.model, cloth_inputs, self.metadata = build_model(r, device=device, cache=cache)
        self.times, self.joint_targets, self.phases, plan = plan_joint_targets(self.model, r)
        self.phase_times = np.asarray(plan["phase_times_s"])
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
        self._initial_drag_rate = r.get("initial_velocity_damping_per_s", 0.0)
        self._particle_mass = self.model.particle_mass.numpy()[:, None] if self._initial_drag_rate else None
        self.control = self.solver.create_control()
        self.control.joint_target_q.assign(self.joint_targets[0])
        self.control.joint_target_qd.zero_()
        compensate = r["robot"].get("gravity_compensation", False)
        self._gravity_state = self.model.state() if compensate else None
        self._gravity_force = (
            wp.zeros(self.model.joint_dof_count, dtype=wp.float32, device=self.model.device) if compensate else None
        )
        self._drive_stiffness = self.model.joint_target_ke.numpy().astype(float) if compensate else None
        self.joint_reference_q = self.joint_targets[0].astype(np.float32)
        self.joint_gravity_force = np.zeros(self.model.joint_dof_count, dtype=np.float32)
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
            gravity_compensation={
                "enabled": compensate,
                "implementation": "lagged g(q_n)/Kp position target bias; nominal velocity; original total effort clipping",
                "query_precision": "FP32 joint coordinates, FK and inverse dynamics",
                "reference_array": "joint_reference_q",
                "requested_gravity_array": "joint_gravity_force",
            },
            recipe_sha256=hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest(),
            trajectory_source_sha256=_sha(REPO / r["source_trajectory"]["path"]),
        )

    def step(self):
        """Advance exactly one physical step; retain clock and commands on failure."""
        if self.failed or self.steps >= self.total_steps:
            return None
        q, v, index = fold_targets(
            self.times, self.joint_targets, (self.steps + 1) * self.dt, phase_times=self.phase_times
        )
        reference = q.astype(np.float32)
        gravity = np.zeros_like(reference)
        old_q, old_v = self.control.joint_target_q.numpy().copy(), self.control.joint_target_qd.numpy().copy()
        # Clear the lagged force once after the hold, then leave subsequent controls alone.
        apply_drag = self._initial_drag_rate > 0 and (self.steps - 1) * self.dt < self.recipe["initial_hold_s"]
        old_f = self.state.particle_f.numpy().copy() if apply_drag else None
        try:
            if self._gravity_state is not None:
                self._gravity_state.joint_q.assign(self.state.joint_q.numpy())
                newton.eval_fk(
                    self.model, self._gravity_state.joint_q, self._gravity_state.joint_qd, self._gravity_state
                )
                newton.eval_inverse_dynamics_passive(self.model, self._gravity_state, gravity_force=self._gravity_force)
                gravity = self._gravity_force.numpy().copy()
                q = q + gravity / self._drive_stiffness
            if apply_drag:
                force = np.zeros_like(old_f)
                if self.steps * self.dt < self.recipe["initial_hold_s"]:
                    force = -self._initial_drag_rate * self._particle_mass * self.state.particle_qd.numpy()
                self.state.particle_f.assign(force)
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
            if old_f is not None:
                self.state.particle_f.assign(old_f)
            self.failed = True
            raise
        self.steps += 1
        self.phase = self.phases[index]
        self.joint_reference_q, self.joint_gravity_force = reference, gravity
        self.last_forces, self._published_generation = forces, stats.step_generation
        return stats
