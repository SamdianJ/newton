# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental fully dynamic Franka / P3 cube grasp support for the checkout example."""

import hashlib
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton
import newton.ik as ik
import newton.utils
from newton.examples.softbody.monolithic_soft_cube import load_cube

REPO = Path(__file__).resolve().parents[2]
DT = 0.001
TIMES = np.array([0, 0.25, 1.0, 2.0, 3.0, 3.5, 4.25, 4.75, 5.25])
PHASES = ("settle", "approach", "close", "lift", "hold", "lower", "release", "retract")
CUBE_POSITION = (0.45, 0.0, 0.02)
TCP_OFFSET = (0.0, 0.0, 0.2104)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def targets(high, low, time_s):
    """Sample smooth joint-space segments with endpoint velocity targets, in SI units."""
    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("Time must be finite and nonnegative")
    i = min(int(np.searchsorted(TIMES[1:], time_s, side="right")), len(PHASES) - 1)
    poses = np.array([high, high, low, low, high, high, low, low, high], dtype=float)
    poses[:, 7:] = np.array([0.04, 0.04, 0.04, 0.014, 0.014, 0.014, 0.014, 0.04, 0.04])[:, None]
    duration = TIMES[i + 1] - TIMES[i]
    u = np.clip((time_s - TIMES[i]) / duration, 0, 1)
    blend = u * u * (3 - 2 * u)
    rate = 6 * u * (1 - u) / duration
    delta = poses[i + 1] - poses[i]
    return poses[i] + blend * delta, rate * delta, PHASES[i]


def build_model(cache, device="cuda:0"):
    """Keep URDF masses/colliders and add the unchanged 40 mm all-dynamic P3 cube."""
    asset = newton.utils.download_asset("franka_emika_panda")
    urdf = asset / "urdf/fr3_franka_hand.urdf"
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    builder.add_urdf(str(urdf), floating=False, collapse_fixed_joints=True, enable_self_collisions=False)
    if builder.joint_dof_count != 9 or builder.body_count != 9:
        raise ValueError("Expected seven dynamic arm joints and two dynamic finger joints")
    mimic_names = ("joint0", "joint1", "coef0", "coef1", "enabled", "label", "world")
    mimic = {name: list(getattr(builder, "constraint_mimic_" + name)) for name in mimic_names}
    if len(mimic["joint0"]) != 1 or mimic["coef0"] != [0.0] or mimic["coef1"] != [1.0]:
        raise ValueError("Unexpected Franka mimic layout")
    if not builder.joint_label[mimic["joint0"][0]].endswith("finger_joint2") or not builder.joint_label[
        mimic["joint1"][0]
    ].endswith("finger_joint1"):
        raise ValueError("Expected the single symmetric finger mimic")
    # Approved example approximation: both fingers remain dynamic, with equal PD targets.
    for name in mimic_names:
        getattr(builder, "constraint_mimic_" + name).clear()
    collision = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for i, flags in enumerate(builder.shape_flags):
        if flags & collision and builder.shape_type[i] == newton.GeoType.MESH:
            builder.shape_source[i].build_sdf(
                device=device,
                max_resolution=128,
                margin=0.003,
                narrow_band_range=(-0.003, 0.003),
                texture_format="float32",
                sign_method="winding",
                cache_dir=cache / "textures",
            )
    for j in range(9):
        builder.joint_target_ke[j] = 20000.0 if j < 7 else 2000.0
        builder.joint_target_kd[j] = 200.0 if j < 7 else 10.0
        builder.joint_target_mode[j] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_limit_ke[j] = 10000.0
        builder.joint_limit_kd[j] = 10.0
    cube = REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz"
    builder.add_soft_mesh(
        pos=CUBE_POSITION,
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        mesh=load_cube(cube),
        particle_radius=0.0002,
        add_surface_mesh_edges=False,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    flags = model.shape_flags.numpy()
    flags &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    metadata = {
        "urdf": str(urdf),
        "urdf_sha256": sha(urdf),
        "cube_sha256": sha(cube),
        "asset_hashes": {str(p.relative_to(asset)): sha(p) for p in asset.rglob("*") if p.is_file()},
        "cube_position_m": CUBE_POSITION,
        "finger_coupling": "independent dynamic PD with symmetric targets",
        "finger_open_target_m": 0.04,
        "finger_closed_target_m": 0.014,
        "removed_mimic": mimic,
        "collision_scope": "rigid-soft and soft-ground; no rigid-rigid",
        "joint_ke": model.joint_target_ke.numpy().tolist(),
        "joint_kd": model.joint_target_kd.numpy().tolist(),
        "collision_shapes": int(np.count_nonzero(flags & collision)),
        "arm_dofs": 7,
        "finger_dofs": 2,
    }
    return model, metadata


def smooth_fields(model):
    """Return the shared builder result with the existing recorder check layout."""
    result = build_smooth_sdf(model)
    return result, result.build_metadata["checks"]


def plan_poses(model):
    """Solve two downward TCP poses once; dynamics subsequently follows joint PD targets."""
    body = model.body_label.index("fr3/fr3_link7")
    position = ik.IKObjectivePosition(
        link_index=body,
        link_offset=wp.vec3(TCP_OFFSET),
        target_positions=wp.array([wp.vec3(0.45, 0, 0.10)], dtype=wp.vec3, device=model.device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=body,
        link_offset_rotation=wp.quat_rpy(0.0, 0.0, -np.pi / 4),
        target_rotations=wp.array([wp.vec4(1, 0, 0, 0)], dtype=wp.vec4, device=model.device),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[position, rotation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.array(
        np.array([[0, -0.5, 0, -2.2, 0, 1.7, 0.7854, 0.04, 0.04]], np.float32), dtype=float, device=model.device
    )
    poses = []
    errors = []
    rotation_errors = []
    for z in (0.10, 0.025):
        position.set_target_position(0, wp.vec3(0.45, 0, z))
        solver.step(q, q, iterations=200)
        result = q.numpy()[0].astype(float)
        result[7:] = model.joint_limit_upper.numpy()[7:]
        state = model.state()
        state.joint_q.assign(result.astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        tcp = wp.transform_point(wp.transform(*state.body_q.numpy()[body]), wp.vec3(TCP_OFFSET))
        error = float(np.linalg.norm(np.array(tcp) - [0.45, 0, z]))
        orient = np.array(wp.quat(*state.body_q.numpy()[body, 3:]) * wp.quat_rpy(0.0, 0.0, -np.pi / 4), dtype=float)
        orient /= np.linalg.norm(orient)
        rotation_error = float(2 * np.arccos(np.clip(abs(orient[0]), 0, 1)))
        if (
            error > 1e-5
            or rotation_error > 1e-4
            or np.any(result < model.joint_limit_lower.numpy())
            or np.any(result > model.joint_limit_upper.numpy())
        ):
            raise ValueError(f"Franka target unreachable or outside limits: {error}")
        poses.append(result)
        errors.append(error)
        rotation_errors.append(rotation_error)
    return *poses, {
        "tcp_positions_m": [[0.45, 0, 0.10], [0.45, 0, 0.025]],
        "tcp_position_errors_m": errors,
        "tcp_rotation_errors_rad": rotation_errors,
        "high": poses[0].tolist(),
        "low": poses[1].tolist(),
        "interpolation": "cubic smoothstep in joint space",
    }


class FrankaGrasp:
    """One fixed experimental scene with an explicit P3 solver and endpoint controller."""

    def __init__(self, cache, device="cuda:0"):
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        self.model, self.metadata = build_model(cache, device=device)
        self.high, self.low, plan = plan_poses(self.model)
        smooth, checks = smooth_fields(self.model)
        self.metadata.update(
            plan=plan,
            sdf_checks=checks,
            sdf_identity=smooth.identity,
            sdf_geometric_certification="not yet certified for new Franka assets",
            publish_policy="current_iterate",
            dt=DT,
            duration=float(TIMES[-1]),
        )
        widths = (0.01,) * 7 + (0.001,) * 2
        self.solver = SolverMonolithic(
            self.model,
            contact_model="C-IE",
            globalization="ArmijoOrResNorm",
            alignment_policy="mask",
            publish_policy="current_iterate",
            allow_nonconservative_energy=True,
            friction_velocity_scale=0.01,
            contact_stiffness=1e7,
            friction_coefficient=0.5,
            normal_smoothing=1e-4,
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            tet_rest_density=wp.full(self.model.tet_count, 1000.0, dtype=float, device=self.model.device),
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True, limits=True, friction=True, limit_width=widths, friction_velocity_scale=widths
            ),
            pcg_mode="production",
            use_cuda_graph=True,
            smooth_sdf=smooth,
        )
        source = self.model.state()
        source.joint_q.assign(self.high.astype(np.float32))
        source.joint_qd.zero_()
        newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
        self.state = self.solver.create_state(dt=DT, source=source)
        self.control = self.model.control()
        self.steps = 0
        self.failed = False
        self.phase = "settle"
        self.metadata["actual_config"] = self.solver._config
        self.metadata["config_digest"] = self.solver._kinematics.digest.hex()

    def step(self):
        if self.failed or self.steps >= round(TIMES[-1] / DT):
            return None
        q, v, self.phase = targets(self.high, self.low, (self.steps + 1) * DT)
        self.control.joint_target_q.assign(q.astype(np.float32))
        self.control.joint_target_qd.assign(v.astype(np.float32))
        self.solver.step(self.state, self.state, self.control, dt=DT)
        self.failed = self.solver.last_stats.rolled_back
        if not self.failed:
            self.steps += 1
        return self.solver.last_stats
